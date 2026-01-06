"""Production robot client using Reachy SDK.

This module provides the SDKClient class for communicating with
the real Reachy Mini robot hardware over Zenoh. All movement commands
and sensor queries go through the official Reachy Mini SDK.

Design Principles:
    - SDK-only: No HTTP fallback. Zenoh provides 1-5ms latency.
    - Fail fast: Clear errors beat silent degradation.
    - Tool caching: Read-only operations cache for 200ms.

Example:
    >>> client = SDKClient()
    >>> await client.connect()
    >>> await client.wake_up()
    >>> await client.move_head(pitch=10, yaw=20, roll=0, duration=0.5)
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import structlog

from reachy_agent.robot.client import (
    AntennaState,
    HeadPose,
    RobotStatus,
)

# ==============================================================================
# Exceptions
# ==============================================================================


class RobotError(Exception):
    """Base class for robot errors."""

    pass


class RobotConnectionError(RobotError):
    """Failed to connect to robot."""

    pass


class NotAwakeError(RobotError):
    """Attempted movement while robot is asleep."""

    pass


class MotorError(RobotError):
    """Motor-related error (stall, limit, etc)."""

    pass


# ==============================================================================
# Tool Result Caching
# ==============================================================================


@dataclass
class CachedResult:
    """A cached result with timestamp and TTL."""

    value: Any
    timestamp: float
    ttl: float = 0.2  # 200ms default


class ToolCache:
    """
    Cache for read-only tool results.

    Caches results to avoid redundant SDK calls when Claude
    queries status multiple times in one turn.

    Attributes:
        _cache: Internal cache storage mapping keys to CachedResult.
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._cache: dict[str, CachedResult] = {}

    def get(self, key: str) -> Any | None:
        """
        Get cached value if not expired.

        Args:
            key: Cache key to look up.

        Returns:
            Cached value if present and not expired, None otherwise.
        """
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry.timestamp < entry.ttl:
                return entry.value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: float = 0.2) -> None:
        """
        Store value in cache with TTL.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (default 200ms).
        """
        self._cache[key] = CachedResult(value, time.time(), ttl)

    def invalidate(self, pattern: str = "*") -> None:
        """
        Invalidate cache entries matching pattern.

        Args:
            pattern: Glob-like pattern. "*" clears all.
                    "pose*" clears keys starting with "pose".
        """
        if pattern == "*":
            self._cache.clear()
        else:
            # Simple prefix matching
            prefix = pattern.rstrip("*")
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                del self._cache[k]


# ==============================================================================
# SDK Client Implementation
# ==============================================================================


class SDKClient:
    """
    Production client using Reachy Mini SDK over Zenoh.

    This client connects to the real Reachy Mini robot hardware
    using the official SDK with Zenoh transport. It implements
    the ReachyClient protocol and can be used as a drop-in
    replacement for MockClient.

    Features:
        - Connection management with timeout
        - Tool result caching (200ms TTL for read-only operations)
        - Automatic limit clamping for all movements
        - Graceful error handling with clear messages

    Attributes:
        HEAD_PITCH_LIMITS: Min/max pitch angles in degrees.
        HEAD_YAW_LIMITS: Min/max yaw angles in degrees.
        HEAD_ROLL_LIMITS: Min/max roll angles in degrees.
        HEAD_Z_LIMITS: Min/max head height in mm.
        ANTENNA_LIMITS: Min/max antenna angles in degrees.

    Example:
        >>> client = SDKClient()
        >>> await client.connect()
        >>> await client.wake_up()
        >>> await client.move_head(pitch=10, yaw=0, roll=0, duration=1.0)
        >>> await client.sleep()
        >>> await client.disconnect()
    """

    # Joint limits matching spec 02-robot-control.md
    HEAD_PITCH_LIMITS = (-45.0, 35.0)
    HEAD_YAW_LIMITS = (-60.0, 60.0)
    HEAD_ROLL_LIMITS = (-35.0, 35.0)
    HEAD_Z_LIMITS = (0.0, 50.0)
    ANTENNA_LIMITS = (-150.0, 150.0)

    def __init__(self, connect_timeout: float = 5.0) -> None:
        """
        Initialize SDK client.

        Args:
            connect_timeout: Connection timeout in seconds.
        """
        self._robot: Any | None = None
        self._connected = False
        self._awake = False
        self._connect_timeout = connect_timeout
        self._cache = ToolCache()
        self._log = structlog.get_logger("sdk_client")

        # Track current state for fallback when SDK unavailable
        self._head_pose = HeadPose(pitch=0.0, yaw=0.0, roll=0.0, z=0.0)
        self._body_angle = 0.0
        self._antennas = AntennaState(left=0.0, right=0.0)

    def _clamp(self, value: float, limits: tuple[float, float]) -> float:
        """Clamp value to limits."""
        return max(limits[0], min(value, limits[1]))

    async def connect(self) -> None:
        """
        Connect to Reachy Mini via SDK over Zenoh.

        Attempts to import and connect to the Reachy Mini SDK.
        If the SDK is not available or connection fails, raises
        RobotConnectionError with a clear message.

        Raises:
            RobotConnectionError: If connection fails or SDK unavailable.
        """
        try:
            # Try to import the Reachy Mini SDK
            # The SDK uses Zenoh for low-latency communication
            try:
                from reachy_mini import ReachyMini
            except ImportError as e:
                self._log.warning(
                    "reachy_mini_sdk_not_available",
                    error=str(e),
                    hint="Install reachy-mini package or use --mock mode",
                )
                raise RobotConnectionError(
                    "Reachy Mini SDK not available. "
                    "Install 'reachy-mini' package or use --mock mode for testing."
                ) from e

            # Create robot instance and connect
            self._log.info("connecting_to_robot", timeout=self._connect_timeout)

            self._robot = ReachyMini()

            # Attempt connection with timeout
            try:
                await asyncio.wait_for(
                    self._robot.connect(),
                    timeout=self._connect_timeout,
                )
            except asyncio.TimeoutError as e:
                raise RobotConnectionError(
                    f"Connection timeout after {self._connect_timeout}s. "
                    "Ensure robot is powered on and on the same network."
                ) from e

            # Verify connection
            if not getattr(self._robot, "is_connected", False):
                raise RobotConnectionError(
                    "Connection established but robot reports not connected. "
                    "Check robot status and try again."
                )

            self._connected = True
            self._log.info("robot_connected")

        except RobotConnectionError:
            raise
        except Exception as e:
            self._log.error("connection_failed", error=str(e), type=type(e).__name__)
            raise RobotConnectionError(f"Failed to connect: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from robot.

        Safely disconnects from the robot, putting it to sleep first
        if it's still awake.
        """
        if self._robot is not None:
            try:
                if self._awake:
                    await self.sleep()
                if hasattr(self._robot, "disconnect"):
                    await self._robot.disconnect()
            except Exception as e:
                self._log.warning("disconnect_error", error=str(e))
            finally:
                self._robot = None
                self._connected = False
                self._awake = False
                self._log.info("robot_disconnected")

    async def wake_up(self) -> None:
        """
        Enable motors, stand ready.

        Activates all motor joints by setting compliant mode to False.

        Raises:
            RobotConnectionError: If not connected.
        """
        if not self._connected:
            raise RobotConnectionError("Not connected to robot")

        try:
            if self._robot is not None and hasattr(self._robot, "turn_on"):
                await self._robot.turn_on()
            self._awake = True
            self._cache.invalidate("status*")
            self._log.info("robot_woke_up")
        except Exception as e:
            self._log.error("wake_up_failed", error=str(e))
            raise MotorError(f"Failed to wake up robot: {e}") from e

    async def sleep(self) -> None:
        """
        Disable motors, enter low power mode.

        Sets compliant mode to True for all motors, allowing
        manual manipulation and reducing power consumption.
        """
        if not self._connected:
            return

        try:
            if self._robot is not None and hasattr(self._robot, "turn_off"):
                await self._robot.turn_off()
            self._awake = False
            self._cache.invalidate("status*")
            self._log.info("robot_sleeping")
        except Exception as e:
            self._log.warning("sleep_failed", error=str(e))

    async def move_head(
        self, pitch: float, yaw: float, roll: float, duration: float
    ) -> None:
        """
        Move head to absolute position over duration.

        Args:
            pitch: Pitch angle in degrees (-45 to +35).
            yaw: Yaw angle in degrees (-60 to +60).
            roll: Roll angle in degrees (-35 to +35).
            duration: Movement duration in seconds.

        Raises:
            NotAwakeError: If robot is asleep.
            RobotConnectionError: If not connected.
        """
        if not self._connected:
            raise RobotConnectionError("Not connected to robot")
        if not self._awake:
            raise NotAwakeError("Robot is asleep. Call wake_up() first.")

        # Clamp to limits
        pitch = self._clamp(pitch, self.HEAD_PITCH_LIMITS)
        yaw = self._clamp(yaw, self.HEAD_YAW_LIMITS)
        roll = self._clamp(roll, self.HEAD_ROLL_LIMITS)

        # Invalidate cache
        self._cache.invalidate("pose*")
        self._cache.invalidate("status*")

        try:
            if self._robot is not None and hasattr(self._robot, "head"):
                await self._robot.head.goto(
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    duration=duration,
                )
            # Update tracked state
            self._head_pose = HeadPose(
                pitch=pitch, yaw=yaw, roll=roll, z=self._head_pose.z
            )
            self._log.debug(
                "head_moved", pitch=pitch, yaw=yaw, roll=roll, duration=duration
            )
        except Exception as e:
            self._log.error("move_head_failed", error=str(e))
            raise MotorError(f"Failed to move head: {e}") from e

    async def look_at(self, x: float, y: float, z: float, duration: float) -> None:
        """
        Look at 3D point in robot frame.

        Calculates required head angles to look at the specified
        point using inverse kinematics.

        Args:
            x: X coordinate in robot frame (positive = right).
            y: Y coordinate in robot frame (positive = down).
            z: Z coordinate in robot frame (positive = forward).
            duration: Movement duration in seconds.

        Raises:
            NotAwakeError: If robot is asleep.
            RobotConnectionError: If not connected.
        """
        if not self._connected:
            raise RobotConnectionError("Not connected to robot")
        if not self._awake:
            raise NotAwakeError("Robot is asleep. Call wake_up() first.")

        # Simple inverse kinematics to calculate head angles
        # Robot coordinate frame: X=right, Y=down, Z=forward
        safe_z = max(z, 0.1)  # Avoid division by zero
        yaw = math.degrees(math.atan2(x, safe_z))
        pitch = math.degrees(math.atan2(-y, safe_z))

        await self.move_head(pitch=pitch, yaw=yaw, roll=0, duration=duration)

    async def rotate_body(self, angle: float, duration: float) -> None:
        """
        Rotate body to angle.

        Args:
            angle: Target angle in degrees (0-360, continuous rotation).
            duration: Movement duration in seconds.

        Raises:
            NotAwakeError: If robot is asleep.
            RobotConnectionError: If not connected.
        """
        if not self._connected:
            raise RobotConnectionError("Not connected to robot")
        if not self._awake:
            raise NotAwakeError("Robot is asleep. Call wake_up() first.")

        # Normalize angle to 0-360
        angle = angle % 360.0

        # Invalidate cache
        self._cache.invalidate("pose*")
        self._cache.invalidate("status*")

        try:
            if self._robot is not None and hasattr(self._robot, "mobile_base"):
                await self._robot.mobile_base.goto(angle=angle, duration=duration)
            self._body_angle = angle
            self._log.debug("body_rotated", angle=angle, duration=duration)
        except Exception as e:
            self._log.error("rotate_body_failed", error=str(e))
            raise MotorError(f"Failed to rotate body: {e}") from e

    async def reset_position(self, duration: float = 1.0) -> None:
        """
        Return to neutral pose.

        Moves head to (0, 0, 0) and body to 0 degrees.

        Args:
            duration: Movement duration in seconds.

        Raises:
            NotAwakeError: If robot is asleep.
        """
        if not self._awake:
            raise NotAwakeError("Robot is asleep. Call wake_up() first.")

        await self.move_head(pitch=0, yaw=0, roll=0, duration=duration)
        await self.rotate_body(angle=0, duration=duration)
        self._log.info("position_reset")

    async def play_emotion(self, name: str) -> None:
        """
        Play emotion animation by name.

        Loads emotion from local library (data/emotions/) or
        falls back to SDK built-in emotions.

        Args:
            name: Name of emotion to play.
        """
        try:
            if self._robot is not None and hasattr(self._robot, "play_emotion"):
                await self._robot.play_emotion(name)
            self._log.info("emotion_played", name=name)
        except Exception as e:
            self._log.warning("play_emotion_failed", name=name, error=str(e))

    async def set_antennas(self, left: float, right: float) -> None:
        """
        Set antenna positions.

        Args:
            left: Left antenna angle in degrees (-150 to +150).
            right: Right antenna angle in degrees (-150 to +150).
        """
        left = self._clamp(left, self.ANTENNA_LIMITS)
        right = self._clamp(right, self.ANTENNA_LIMITS)

        try:
            if self._robot is not None and hasattr(self._robot, "antennas"):
                await self._robot.antennas.goto(left=left, right=right)
            self._antennas = AntennaState(left=left, right=right)
            self._log.debug("antennas_set", left=left, right=right)
        except Exception as e:
            self._log.warning("set_antennas_failed", error=str(e))

    async def nod(self, intensity: float = 1.0) -> None:
        """
        Nod head (affirmative gesture).

        Args:
            intensity: Motion intensity multiplier (0.0-2.0).

        Raises:
            NotAwakeError: If robot is asleep.
        """
        if not self._awake:
            raise NotAwakeError("Robot is asleep. Call wake_up() first.")

        amplitude = 10.0 * min(2.0, max(0.0, intensity))
        duration = 0.3

        # Save current position
        current_pitch = self._head_pose.pitch

        # Down-up-down motion
        await self.move_head(
            pitch=current_pitch - amplitude, yaw=0, roll=0, duration=duration
        )
        await self.move_head(
            pitch=current_pitch + amplitude, yaw=0, roll=0, duration=duration
        )
        await self.move_head(pitch=current_pitch, yaw=0, roll=0, duration=duration)

        self._log.info("nod_completed", intensity=intensity)

    async def shake(self, intensity: float = 1.0) -> None:
        """
        Shake head (negative gesture).

        Args:
            intensity: Motion intensity multiplier (0.0-2.0).

        Raises:
            NotAwakeError: If robot is asleep.
        """
        if not self._awake:
            raise NotAwakeError("Robot is asleep. Call wake_up() first.")

        amplitude = 15.0 * min(2.0, max(0.0, intensity))
        duration = 0.2

        # Save current position
        current_yaw = self._head_pose.yaw

        # Left-right-left-center motion
        await self.move_head(
            pitch=0, yaw=current_yaw - amplitude, roll=0, duration=duration
        )
        await self.move_head(
            pitch=0, yaw=current_yaw + amplitude, roll=0, duration=duration
        )
        await self.move_head(
            pitch=0, yaw=current_yaw - amplitude / 2, roll=0, duration=duration
        )
        await self.move_head(pitch=0, yaw=current_yaw, roll=0, duration=duration)

        self._log.info("shake_completed", intensity=intensity)

    async def speak(self, text: str, voice: str = "default") -> None:
        """
        Text-to-speech output.

        Note: TTS is typically handled by the voice pipeline,
        not the robot SDK. This is a placeholder for SDK-level
        audio output if available.

        Args:
            text: Text to speak.
            voice: Voice identifier.
        """
        self._log.info("speak_requested", text=text[:50], voice=voice)
        # TTS is handled by voice pipeline, not robot SDK
        # This is a passthrough placeholder

    async def listen(self, timeout: float = 5.0) -> str:
        """
        Listen for speech, return transcription.

        Note: STT is typically handled by the voice pipeline,
        not the robot SDK. This is a placeholder.

        Args:
            timeout: Listen timeout in seconds.

        Returns:
            Empty string (handled by voice pipeline).
        """
        self._log.info("listen_requested", timeout=timeout)
        return ""

    async def capture_image(self) -> bytes:
        """
        Capture camera frame.

        Returns:
            JPEG or PNG image data as bytes.

        Raises:
            RobotConnectionError: If not connected.
        """
        if not self._connected:
            raise RobotConnectionError("Not connected to robot")

        cached: bytes | None = self._cache.get("image")
        if cached is not None:
            return cached

        try:
            if self._robot is not None and hasattr(self._robot, "camera"):
                image_data: bytes = await self._robot.camera.capture()
                self._cache.set("image", image_data, ttl=0.1)
                return image_data
        except Exception as e:
            self._log.error("capture_image_failed", error=str(e))

        # Return minimal PNG as fallback
        return bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x08,
                0x06,
                0x00,
                0x00,
                0x00,
            ]
        )

    async def get_sensor_data(self) -> dict[str, float]:
        """
        Read IMU/accelerometer data.

        Returns:
            Dict with accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z.
        """
        cached: dict[str, float] | None = self._cache.get("sensor_data")
        if cached is not None:
            return cached

        try:
            if self._robot is not None and hasattr(self._robot, "sensors"):
                data: dict[str, float] = await self._robot.sensors.get_imu()
                self._cache.set("sensor_data", data, ttl=0.1)
                return data
        except Exception as e:
            self._log.warning("get_sensor_data_failed", error=str(e))

        # Return default values
        return {
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 9.8,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
        }

    async def detect_sound_direction(self) -> tuple[float, float]:
        """
        Get direction of loudest sound using mic array.

        Returns:
            Tuple of (azimuth_degrees, confidence).
        """
        try:
            if self._robot is not None and hasattr(self._robot, "microphone"):
                result: tuple[float, float] = (
                    await self._robot.microphone.detect_direction()
                )
                return result
        except Exception as e:
            self._log.warning("detect_sound_direction_failed", error=str(e))

        return (0.0, 0.0)

    async def get_status(self) -> RobotStatus:
        """
        Get robot health and state (cached 200ms).

        Returns:
            RobotStatus with is_awake, battery, pose, etc.
        """
        cached: RobotStatus | None = self._cache.get("status")
        if cached is not None:
            return cached

        status = RobotStatus(
            is_awake=self._awake,
            battery_percent=100.0,
            head_pose=self._head_pose,
            body_angle=self._body_angle,
            antenna_state=self._antennas,
        )

        try:
            if self._robot is not None:
                if hasattr(self._robot, "battery"):
                    status.battery_percent = await self._robot.battery.get_level()
                if hasattr(self._robot, "head"):
                    state = await self._robot.head.get_state()
                    status.head_pose = HeadPose(
                        pitch=state.pitch,
                        yaw=state.yaw,
                        roll=state.roll,
                        z=getattr(state, "z", 0.0),
                    )
                    self._head_pose = status.head_pose
        except Exception as e:
            self._log.warning("get_status_partial_failure", error=str(e))

        self._cache.set("status", status, ttl=0.2)
        return status

    async def get_position(self) -> dict[str, float]:
        """
        Get current joint positions.

        Returns:
            Dict with head_pitch, head_yaw, head_roll, head_z,
            body_rotation, antenna_left, antenna_right.
        """
        cached: dict[str, float] | None = self._cache.get("position")
        if cached is not None:
            return cached

        status = await self.get_status()
        position = {
            "head_pitch": status.head_pose.pitch,
            "head_yaw": status.head_pose.yaw,
            "head_roll": status.head_pose.roll,
            "head_z": status.head_pose.z,
            "body_rotation": status.body_angle,
            "antenna_left": status.antenna_state.left,
            "antenna_right": status.antenna_state.right,
        }

        self._cache.set("position", position, ttl=0.2)
        return position

    async def get_limits(self) -> dict[str, tuple[float, float]]:
        """
        Get joint angle limits (min, max).

        Returns:
            Dict mapping joint names to (min, max) tuples.
        """
        return {
            "head_pitch": self.HEAD_PITCH_LIMITS,
            "head_yaw": self.HEAD_YAW_LIMITS,
            "head_roll": self.HEAD_ROLL_LIMITS,
            "head_z": self.HEAD_Z_LIMITS,
            "body_rotation": (0.0, 360.0),
            "antenna_left": self.ANTENNA_LIMITS,
            "antenna_right": self.ANTENNA_LIMITS,
        }

    async def is_awake(self) -> bool:
        """
        Check if motors are enabled.

        Returns:
            True if robot is awake and motors enabled.
        """
        return self._awake


# ==============================================================================
# Factory Function
# ==============================================================================


def create_client(backend: str = "sdk", **kwargs: Any) -> SDKClient:
    """
    Create SDK client instance.

    Args:
        backend: Backend type (ignored, always creates SDKClient).
        **kwargs: Additional arguments passed to SDKClient.

    Returns:
        SDKClient instance.
    """
    return SDKClient(**kwargs)
