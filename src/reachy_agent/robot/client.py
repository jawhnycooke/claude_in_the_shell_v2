"""Unified robot client interface."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Backend(Enum):
    """Robot communication backend."""

    SDK = "sdk"  # Production: Reachy SDK via Zenoh
    MOCK = "mock"  # Testing: In-memory mock


@dataclass
class HeadPose:
    """Head position in degrees and mm."""

    pitch: float  # degrees, -45 to +35
    yaw: float  # degrees, -60 to +60
    roll: float  # degrees, -35 to +35
    z: float  # mm, 0 to 50 (head height)


@dataclass
class AntennaState:
    """Antenna positions."""

    left: float  # degrees, -150 to +150
    right: float  # degrees, -150 to +150


@dataclass
class RobotStatus:
    """Complete robot status."""

    is_awake: bool
    battery_percent: float
    head_pose: HeadPose
    body_angle: float  # degrees, 0-360
    antenna_state: AntennaState


class ReachyClient(Protocol):
    """
    Unified interface for robot control.

    All robot operations go through this interface, allowing easy
    switching between SDK and mock implementations.

    See spec 02-robot-control.md for details.
    """

    # Lifecycle
    async def connect(self) -> None:
        """Connect to robot."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from robot."""
        ...

    async def wake_up(self) -> None:
        """Enable motors, stand ready."""
        ...

    async def sleep(self) -> None:
        """Disable motors, low power."""
        ...

    # Movement
    async def move_head(
        self, pitch: float, yaw: float, roll: float, duration: float
    ) -> None:
        """Move head to absolute position over duration."""
        ...

    async def look_at(self, x: float, y: float, z: float, duration: float) -> None:
        """Look at 3D point in robot frame."""
        ...

    async def rotate_body(self, angle: float, duration: float) -> None:
        """Rotate body to angle."""
        ...

    async def reset_position(self, duration: float = 1.0) -> None:
        """Return to neutral pose."""
        ...

    # Expression
    async def play_emotion(self, name: str) -> None:
        """Play emotion animation by name."""
        ...

    async def set_antennas(self, left: float, right: float) -> None:
        """Set antenna positions."""
        ...

    async def nod(self, intensity: float = 1.0) -> None:
        """Nod head (affirmative gesture)."""
        ...

    async def shake(self, intensity: float = 1.0) -> None:
        """Shake head (negative gesture)."""
        ...

    # Audio
    async def speak(self, text: str, voice: str = "default") -> None:
        """Text-to-speech output."""
        ...

    async def listen(self, timeout: float = 5.0) -> str:
        """Listen for speech, return transcription."""
        ...

    # Perception
    async def capture_image(self) -> bytes:
        """Capture camera frame."""
        ...

    async def get_sensor_data(self) -> dict[str, float]:
        """Read IMU/accelerometer."""
        ...

    async def detect_sound_direction(self) -> tuple[float, float]:
        """Get direction of loudest sound (degrees, confidence)."""
        ...

    # Status (cached)
    async def get_status(self) -> RobotStatus:
        """Get robot health and state."""
        ...

    async def get_position(self) -> dict[str, float]:
        """Get current joint positions."""
        ...

    async def get_limits(self) -> dict[str, tuple[float, float]]:
        """Get joint angle limits (min, max)."""
        ...

    async def is_awake(self) -> bool:
        """Check if motors are enabled."""
        ...
