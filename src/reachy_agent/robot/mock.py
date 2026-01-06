"""Mock robot client for testing without hardware."""

import asyncio
import random

import structlog

from reachy_agent.robot.client import (
    AntennaState,
    HeadPose,
    RobotStatus,
)


class MockClient:
    """
    Mock robot client for testing without hardware.

    Simulates all robot operations in-memory with realistic timing.
    Returns plausible mock data for sensors and status.

    Examples:
        >>> client = MockClient()
        >>> await client.connect()
        >>> await client.wake_up()
        >>> await client.move_head(pitch=10, yaw=20, roll=0, duration=0.5)
    """

    # Joint limits for mock validation
    HEAD_PITCH_LIMITS = (-45.0, 35.0)
    HEAD_YAW_LIMITS = (-60.0, 60.0)
    HEAD_ROLL_LIMITS = (-35.0, 35.0)
    HEAD_Z_LIMITS = (0.0, 50.0)
    ANTENNA_LIMITS = (-150.0, 150.0)

    def __init__(self) -> None:
        """Initialize mock client."""
        self._awake = False
        self._connected = False
        self._head_pose = HeadPose(pitch=0.0, yaw=0.0, roll=0.0, z=0.0)
        self._body_angle = 0.0
        self._antennas = AntennaState(left=0.0, right=0.0)
        self._log = structlog.get_logger("mock_client")

    async def connect(self) -> None:
        """Mock connection (always succeeds)."""
        self._connected = True
        self._log.info("mock_connected")

    async def disconnect(self) -> None:
        """Mock disconnection."""
        self._connected = False
        self._awake = False
        self._log.info("mock_disconnected")

    async def wake_up(self) -> None:
        """Enable mock motors."""
        self._awake = True
        self._log.info("mock_woke_up")

    async def sleep(self) -> None:
        """Disable mock motors."""
        self._awake = False
        self._log.info("mock_sleeping")

    async def move_head(
        self, pitch: float, yaw: float, roll: float, duration: float
    ) -> None:
        """
        Simulate head movement with validation.

        Args:
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees
            roll: Roll angle in degrees
            duration: Movement duration in seconds
        """
        if not self._awake:
            raise RuntimeError("Robot is asleep")

        # Validate limits
        pitch = max(self.HEAD_PITCH_LIMITS[0], min(pitch, self.HEAD_PITCH_LIMITS[1]))
        yaw = max(self.HEAD_YAW_LIMITS[0], min(yaw, self.HEAD_YAW_LIMITS[1]))
        roll = max(self.HEAD_ROLL_LIMITS[0], min(roll, self.HEAD_ROLL_LIMITS[1]))

        # Simulate movement time
        await asyncio.sleep(duration)
        self._head_pose = HeadPose(pitch=pitch, yaw=yaw, roll=roll, z=self._head_pose.z)
        self._log.debug("mock_head_moved", pitch=pitch, yaw=yaw, roll=roll)

    async def look_at(self, x: float, y: float, z: float, duration: float) -> None:
        """
        Simulate looking at 3D point.

        Simplified IK: calculate approximate pitch and yaw.
        """
        if not self._awake:
            raise RuntimeError("Robot is asleep")

        import math

        # Simple IK approximation
        yaw = math.degrees(math.atan2(x, max(z, 0.1)))
        pitch = math.degrees(math.atan2(-y, max(z, 0.1)))

        await self.move_head(pitch=pitch, yaw=yaw, roll=0, duration=duration)

    async def rotate_body(self, angle: float, duration: float) -> None:
        """Simulate body rotation."""
        if not self._awake:
            raise RuntimeError("Robot is asleep")

        await asyncio.sleep(duration)
        self._body_angle = angle % 360.0
        self._log.debug("mock_body_rotated", angle=self._body_angle)

    async def reset_position(self, duration: float = 1.0) -> None:
        """Return to neutral pose."""
        if not self._awake:
            raise RuntimeError("Robot is asleep")

        await asyncio.sleep(duration)
        self._head_pose = HeadPose(pitch=0.0, yaw=0.0, roll=0.0, z=0.0)
        self._body_angle = 0.0
        self._log.info("mock_reset_position")

    async def play_emotion(self, name: str) -> None:
        """Simulate emotion playback."""
        self._log.info("mock_emotion", name=name)
        await asyncio.sleep(1.0)  # Simulate animation time

    async def set_antennas(self, left: float, right: float) -> None:
        """Set antenna positions with validation."""
        left = max(self.ANTENNA_LIMITS[0], min(left, self.ANTENNA_LIMITS[1]))
        right = max(self.ANTENNA_LIMITS[0], min(right, self.ANTENNA_LIMITS[1]))
        self._antennas = AntennaState(left=left, right=right)
        self._log.debug("mock_antennas", left=left, right=right)

    async def nod(self, intensity: float = 1.0) -> None:
        """
        Simulate nod gesture.

        Args:
            intensity: Motion intensity multiplier (0.0-2.0)
        """
        if not self._awake:
            raise RuntimeError("Robot is asleep")

        # Scale motion by intensity
        amplitude = 10.0 * min(2.0, max(0.0, intensity))
        duration = 0.3

        # Down-up-down motion
        await self.move_head(pitch=-amplitude, yaw=0, roll=0, duration=duration)
        await self.move_head(pitch=amplitude, yaw=0, roll=0, duration=duration)
        await self.move_head(pitch=0, yaw=0, roll=0, duration=duration)

        self._log.info("mock_nod", intensity=intensity)

    async def shake(self, intensity: float = 1.0) -> None:
        """
        Simulate shake gesture.

        Args:
            intensity: Motion intensity multiplier (0.0-2.0)
        """
        if not self._awake:
            raise RuntimeError("Robot is asleep")

        amplitude = 15.0 * min(2.0, max(0.0, intensity))
        duration = 0.2

        # Left-right-left-center motion
        await self.move_head(pitch=0, yaw=-amplitude, roll=0, duration=duration)
        await self.move_head(pitch=0, yaw=amplitude, roll=0, duration=duration)
        await self.move_head(pitch=0, yaw=-amplitude / 2, roll=0, duration=duration)
        await self.move_head(pitch=0, yaw=0, roll=0, duration=duration)

        self._log.info("mock_shake", intensity=intensity)

    async def speak(self, text: str, voice: str = "default") -> None:
        """Simulate speech."""
        self._log.info("mock_speak", text=text[:50], voice=voice)
        # Roughly 100ms per word
        words = len(text.split())
        await asyncio.sleep(words * 0.1)

    async def listen(self, timeout: float = 5.0) -> str:
        """Simulate listening and return mock transcription."""
        self._log.info("mock_listen", timeout=timeout)
        await asyncio.sleep(min(timeout, 2.0))  # Simulate some listening time
        return "This is a mock transcription."

    async def capture_image(self) -> bytes:
        """Return mock image data (PNG placeholder)."""
        # Minimal PNG header (1x1 pixel transparent)
        return bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,  # Width: 1
                0x00,
                0x00,
                0x00,
                0x01,  # Height: 1
                0x08,
                0x06,  # 8-bit RGBA
                0x00,
                0x00,
                0x00,
            ]
        )

    async def get_sensor_data(self) -> dict[str, float]:
        """Return realistic mock sensor data with slight noise."""
        return {
            "accel_x": random.gauss(0.0, 0.1),
            "accel_y": random.gauss(0.0, 0.1),
            "accel_z": random.gauss(9.8, 0.1),  # Gravity
            "gyro_x": random.gauss(0.0, 0.01),
            "gyro_y": random.gauss(0.0, 0.01),
            "gyro_z": random.gauss(0.0, 0.01),
        }

    async def detect_sound_direction(self) -> tuple[float, float]:
        """
        Return mock sound direction.

        Returns:
            Tuple of (azimuth_degrees, confidence)
        """
        return (random.uniform(-30, 30), random.uniform(0.5, 0.9))

    async def get_status(self) -> RobotStatus:
        """Get mock robot status."""
        return RobotStatus(
            is_awake=self._awake,
            battery_percent=random.uniform(85.0, 100.0),  # Realistic variation
            head_pose=self._head_pose,
            body_angle=self._body_angle,
            antenna_state=self._antennas,
        )

    async def get_position(self) -> dict[str, float]:
        """Get current joint positions."""
        return {
            "head_pitch": self._head_pose.pitch,
            "head_yaw": self._head_pose.yaw,
            "head_roll": self._head_pose.roll,
            "head_z": self._head_pose.z,
            "body_rotation": self._body_angle,
            "antenna_left": self._antennas.left,
            "antenna_right": self._antennas.right,
        }

    async def get_limits(self) -> dict[str, tuple[float, float]]:
        """Get joint angle limits (min, max)."""
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
        """Check if motors are enabled."""
        return self._awake
