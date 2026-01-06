"""30Hz motion blend controller."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol

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

    head: HeadPose | PoseOffset
    antennas: Optional[AntennaState] = None
    body_angle: Optional[float] = None


class MotionSource(Protocol):
    """Interface for motion sources."""

    @property
    def name(self) -> str:
        ...

    @property
    def source_type(self) -> MotionSourceType:
        ...

    @property
    def is_active(self) -> bool:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def tick(self) -> Optional[MotionOutput]:
        """Called at 30Hz. Returns motion output or None if inactive."""
        ...


class BlendController:
    """
    Orchestrates motion sources at 30Hz.

    Manages PRIMARY (mutually exclusive) and OVERLAY (additive) motion sources,
    blending their outputs and sending to robot hardware.

    TODO: Complete implementation as per spec 04-motion.md
    - 30Hz control loop
    - PRIMARY source management
    - OVERLAY source management
    - Pose blending and clamping
    - Integration with voice events
    """

    TICK_HZ = 30
    TICK_INTERVAL = 1.0 / TICK_HZ

    def __init__(self, client: ReachyClient):
        """
        Initialize blend controller.

        Args:
            client: Robot client for sending commands
        """
        self._client = client
        self._primary: Optional[MotionSource] = None
        self._overlays: Dict[str, MotionSource] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._log = structlog.get_logger()

    async def start(self) -> None:
        """
        Start the motion control loop.

        TODO: Implement startup
        - Start background task
        - Initialize default idle behavior
        """
        self._running = True
        self._task = asyncio.create_task(self._loop())
        self._log.info("motion_controller_started")

        # TODO: Set default idle behavior
        # await self.set_primary(IdleBehavior())

    async def stop(self) -> None:
        """Stop the motion control loop."""
        self._running = False
        if self._task:
            await self._task
        self._log.info("motion_controller_stopped")

    async def set_primary(self, source: MotionSource) -> None:
        """
        Set the PRIMARY motion source.

        Args:
            source: New primary motion source
        """
        if self._primary:
            await self._primary.stop()
        self._primary = source
        await self._primary.start()
        self._log.info("primary_source_set", name=source.name)

    async def add_overlay(self, source: MotionSource) -> None:
        """
        Add an OVERLAY motion source.

        Args:
            source: Overlay motion source to add
        """
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
        Main 30Hz control loop.

        TODO: Implement control loop
        - Get PRIMARY output
        - Get and sum OVERLAY outputs
        - Blend poses
        - Clamp to hardware limits
        - Send to robot
        - Maintain 30Hz tick rate
        """
        next_tick = time.time()

        while self._running:
            # TODO: Implement blending logic
            # - Get primary output
            # - Sum overlay offsets
            # - Apply offsets to primary
            # - Clamp to limits
            # - Send to robot

            # Maintain tick rate
            next_tick += self.TICK_INTERVAL
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def _clamp_pose(self, pose: HeadPose) -> HeadPose:
        """
        Clamp pose to hardware limits.

        Args:
            pose: Input pose

        Returns:
            Clamped pose
        """
        return HeadPose(
            pitch=max(-45, min(35, pose.pitch)),
            yaw=max(-60, min(60, pose.yaw)),
            roll=max(-35, min(35, pose.roll)),
            z=max(0, min(50, pose.z)),
        )
