"""30Hz motion blend controller.

This module provides the BlendController for orchestrating motion sources
at a consistent 30Hz rate. It supports PRIMARY (mutually exclusive) and
OVERLAY (additive) motion sources.

Features:
    - 30Hz control loop with drift correction
    - PRIMARY source management (idle, emotion, tracking)
    - OVERLAY source management (wobble, nod)
    - Pose blending and clamping to hardware limits
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import structlog

from reachy_agent.robot.client import HeadPose, ReachyClient


class MotionSourceType(Enum):
    """Motion source types."""

    PRIMARY = "primary"  # Mutually exclusive (idle, emotion, tracking)
    OVERLAY = "overlay"  # Additive (wobble, nod)


@dataclass
class PoseOffset:
    """Offset to add to base pose."""

    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    z: float = 0.0


@dataclass
class AntennaState:
    """Antenna positions."""

    left: float = 0.0
    right: float = 0.0


@dataclass
class MotionOutput:
    """Output from a motion source."""

    head: HeadPose | PoseOffset | None = None
    antennas: AntennaState | None = None
    body_angle: float | None = None


class MotionSource(Protocol):
    """Interface for motion sources.

    Motion sources provide position data (PRIMARY) or offset data (OVERLAY)
    at each tick of the 30Hz control loop.

    Methods:
        tick: Called at 30Hz, returns motion output
        start: Initialize the motion source
        stop: Clean up the motion source

    Properties:
        name: Unique identifier for the source
        source_type: PRIMARY or OVERLAY
        is_active: Whether the source is currently running
    """

    @property
    def name(self) -> str:
        """Unique name for this motion source."""
        ...

    @property
    def source_type(self) -> MotionSourceType:
        """Type of motion source (PRIMARY or OVERLAY)."""
        ...

    @property
    def is_active(self) -> bool:
        """Whether the source is currently active."""
        ...

    async def start(self) -> None:
        """Start the motion source."""
        ...

    async def stop(self) -> None:
        """Stop the motion source."""
        ...

    def tick(self) -> MotionOutput | None:
        """
        Called at 30Hz. Returns motion output or None if inactive.

        For PRIMARY sources: Returns absolute positions
        For OVERLAY sources: Returns position deltas/offsets

        Returns:
            MotionOutput with head, antenna, and/or body positions/offsets
        """
        ...

    def get_positions(self) -> dict[str, float]:
        """
        Get current joint positions (for PRIMARY sources).

        Returns:
            Dictionary mapping joint names to absolute positions
        """
        ...

    def get_deltas(self) -> dict[str, float]:
        """
        Get current position deltas (for OVERLAY sources).

        Returns:
            Dictionary mapping joint names to offset values
        """
        ...


class BlendController:
    """
    Orchestrates motion sources at 30Hz.

    Manages PRIMARY (mutually exclusive) and OVERLAY (additive) motion sources,
    blending their outputs and sending to robot hardware.

    The control loop runs at 30Hz (33.33ms per tick) with drift correction
    to maintain consistent timing regardless of processing time.

    Attributes:
        TICK_HZ: Control loop frequency (30Hz)
        TICK_INTERVAL: Time between ticks (33.33ms)

    Examples:
        >>> controller = BlendController(robot_client)
        >>> await controller.start()
        >>> await controller.set_primary(idle_source)
        >>> await controller.add_overlay(wobble_source)
        >>> # Controller runs at 30Hz in background
        >>> await controller.stop()
    """

    TICK_HZ = 30
    TICK_INTERVAL = 1.0 / TICK_HZ

    # Joint limits for clamping
    HEAD_LIMITS = {
        "pitch": (-45.0, 35.0),
        "yaw": (-60.0, 60.0),
        "roll": (-35.0, 35.0),
        "z": (0.0, 50.0),
    }

    ANTENNA_LIMITS = {
        "left": (-90.0, 90.0),
        "right": (-90.0, 90.0),
    }

    BODY_LIMITS = (-180.0, 180.0)

    def __init__(
        self,
        client: ReachyClient,
        mock_mode: bool = False,
    ) -> None:
        """
        Initialize blend controller.

        Args:
            client: Robot client for sending commands
            mock_mode: If True, don't send commands to robot
        """
        self._client = client
        self._mock_mode = mock_mode
        self._primary: MotionSource | None = None
        self._overlays: dict[str, MotionSource] = {}
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._tick_count = 0
        self._last_pose: HeadPose | None = None
        self._log = structlog.get_logger("motion.controller")

    @property
    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    @property
    def tick_count(self) -> int:
        """Get number of ticks since start."""
        return self._tick_count

    @property
    def primary_source(self) -> MotionSource | None:
        """Get current PRIMARY source."""
        return self._primary

    @property
    def overlay_sources(self) -> dict[str, MotionSource]:
        """Get current OVERLAY sources."""
        return self._overlays.copy()

    async def start(self) -> None:
        """
        Start the motion control loop.

        Initializes the 30Hz background task. The loop continues until
        stop() is called.

        Raises:
            RuntimeError: If controller is already running
        """
        if self._running:
            raise RuntimeError("BlendController already running")

        self._running = True
        self._tick_count = 0
        self._task = asyncio.create_task(self._loop())
        self._log.info("motion_controller_started", tick_hz=self.TICK_HZ)

    async def stop(self) -> None:
        """
        Stop the motion control loop.

        Gracefully stops the background task, allowing the current tick
        to complete before shutting down. Also stops all motion sources.
        """
        if not self._running:
            return

        self._running = False

        # Wait for current tick to complete
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=self.TICK_INTERVAL * 2)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

        # Stop all sources
        if self._primary:
            await self._primary.stop()
            self._primary = None

        for overlay in list(self._overlays.values()):
            await overlay.stop()
        self._overlays.clear()

        self._log.info(
            "motion_controller_stopped",
            total_ticks=self._tick_count,
        )

    async def set_primary(self, source: MotionSource) -> None:
        """
        Set the PRIMARY motion source.

        PRIMARY sources are mutually exclusive - setting a new primary
        stops the previous one.

        Args:
            source: New primary motion source

        Raises:
            ValueError: If source is not a PRIMARY type
        """
        if source.source_type != MotionSourceType.PRIMARY:
            raise ValueError(f"Expected PRIMARY source, got {source.source_type}")

        # Stop existing primary
        if self._primary:
            await self._primary.stop()

        self._primary = source
        await self._primary.start()
        self._log.info("primary_source_set", name=source.name)

    async def add_overlay(self, source: MotionSource) -> None:
        """
        Add an OVERLAY motion source.

        OVERLAY sources are additive - their deltas are summed
        on top of the PRIMARY source output.

        Args:
            source: Overlay motion source to add

        Raises:
            ValueError: If source is not an OVERLAY type
        """
        if source.source_type != MotionSourceType.OVERLAY:
            raise ValueError(f"Expected OVERLAY source, got {source.source_type}")

        self._overlays[source.name] = source
        await source.start()
        self._log.info("overlay_added", name=source.name)

    async def remove_overlay(self, name: str) -> None:
        """
        Remove an OVERLAY motion source.

        Args:
            name: Name of overlay to remove
        """
        if name in self._overlays:
            await self._overlays[name].stop()
            del self._overlays[name]
            self._log.info("overlay_removed", name=name)

    async def _loop(self) -> None:
        """
        Main 30Hz control loop with drift correction.

        Maintains consistent 30Hz timing by tracking the expected tick time
        and adjusting sleep duration to compensate for processing time.
        """
        next_tick = time.time()

        while self._running:
            tick_start = time.time()

            # Get and apply blended pose
            try:
                blended = self._blend_sources()
                if blended and not self._mock_mode:
                    await self._apply_output(blended)
                self._last_pose = blended.head if blended else self._last_pose
            except Exception as e:
                self._log.error("blend_error", error=str(e))

            self._tick_count += 1

            # Maintain tick rate with drift correction
            next_tick += self.TICK_INTERVAL
            sleep_time = next_tick - time.time()

            # Handle timing drift
            if sleep_time < 0:
                # We're behind - skip ticks to catch up
                missed_ticks = int(-sleep_time / self.TICK_INTERVAL)
                if missed_ticks > 0:
                    self._log.warning("ticks_missed", count=missed_ticks)
                    next_tick = time.time() + self.TICK_INTERVAL
                    sleep_time = self.TICK_INTERVAL
            elif sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def _blend_sources(self) -> MotionOutput | None:
        """
        Blend PRIMARY and OVERLAY sources.

        Gets the base pose from PRIMARY source, then adds all OVERLAY
        deltas on top, clamping the final result to hardware limits.

        Returns:
            Blended MotionOutput or None if no sources active
        """
        # Get primary output
        primary_output: MotionOutput | None = None
        if self._primary and self._primary.is_active:
            primary_output = self._primary.tick()

        # Start with primary or default pose
        if primary_output and isinstance(primary_output.head, HeadPose):
            base_head = primary_output.head
        elif self._last_pose:
            base_head = self._last_pose
        else:
            base_head = HeadPose(pitch=0, yaw=0, roll=0, z=0)

        # Accumulate overlay deltas
        pitch_delta = 0.0
        yaw_delta = 0.0
        roll_delta = 0.0
        z_delta = 0.0

        for overlay in self._overlays.values():
            if overlay.is_active:
                overlay_output = overlay.tick()
                if overlay_output and isinstance(overlay_output.head, PoseOffset):
                    pitch_delta += overlay_output.head.pitch
                    yaw_delta += overlay_output.head.yaw
                    roll_delta += overlay_output.head.roll
                    z_delta += overlay_output.head.z

        # Apply deltas to base pose
        final_pose = HeadPose(
            pitch=base_head.pitch + pitch_delta,
            yaw=base_head.yaw + yaw_delta,
            roll=base_head.roll + roll_delta,
            z=base_head.z + z_delta,
        )

        # Clamp to limits
        final_pose = self._clamp_pose(final_pose)

        # Handle antennas and body
        antennas = None
        body_angle = None

        if primary_output:
            antennas = primary_output.antennas
            body_angle = primary_output.body_angle

        return MotionOutput(
            head=final_pose,
            antennas=antennas,
            body_angle=body_angle,
        )

    def _clamp_pose(self, pose: HeadPose) -> HeadPose:
        """
        Clamp pose to hardware limits.

        Args:
            pose: Input pose

        Returns:
            Clamped pose
        """
        pitch_min, pitch_max = self.HEAD_LIMITS["pitch"]
        yaw_min, yaw_max = self.HEAD_LIMITS["yaw"]
        roll_min, roll_max = self.HEAD_LIMITS["roll"]
        z_min, z_max = self.HEAD_LIMITS["z"]

        return HeadPose(
            pitch=max(pitch_min, min(pitch_max, pose.pitch)),
            yaw=max(yaw_min, min(yaw_max, pose.yaw)),
            roll=max(roll_min, min(roll_max, pose.roll)),
            z=max(z_min, min(z_max, pose.z)),
        )

    async def _apply_output(self, output: MotionOutput) -> None:
        """
        Apply blended output to robot.

        Args:
            output: Blended motion output to apply
        """
        if output.head and isinstance(output.head, HeadPose):
            try:
                await self._client.move_head(
                    pitch=output.head.pitch,
                    yaw=output.head.yaw,
                    roll=output.head.roll,
                    duration=self.TICK_INTERVAL,
                )
            except Exception as e:
                self._log.warning("move_head_error", error=str(e))

        if output.antennas:
            try:
                await self._client.move_antennas(
                    left=output.antennas.left,
                    right=output.antennas.right,
                    duration=self.TICK_INTERVAL,
                )
            except (AttributeError, Exception):
                # move_antennas may not be implemented
                pass

        if output.body_angle is not None:
            try:
                await self._client.rotate_body(
                    angle=output.body_angle,
                    duration=self.TICK_INTERVAL,
                )
            except Exception as e:
                self._log.warning("rotate_body_error", error=str(e))
