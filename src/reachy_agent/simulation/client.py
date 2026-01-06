"""MuJoCo-based implementation of ReachyClient protocol.

This module provides a physics-accurate simulation of the Reachy Mini robot
using the MuJoCo physics engine.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from reachy_agent.robot.client import (
    AntennaState,
    HeadPose,
    RobotStatus,
)

if TYPE_CHECKING:
    from reachy_agent.simulation.environment import SimulationEnvironment

logger = structlog.get_logger()


class MuJoCoReachyClient:
    """
    ReachyClient implementation using MuJoCo physics simulation.

    This client provides physics-accurate simulation of the Reachy Mini robot,
    enabling development and testing without physical hardware.

    Args:
        model_path: Path to MJCF model file (default: built-in model)
        realtime: Run simulation in real-time (default: True)
        viewer: Enable visualization window (default: False)

    Example:
        >>> client = MuJoCoReachyClient(viewer=True)
        >>> await client.connect()
        >>> await client.wake_up()
        >>> await client.move_head(pitch=10, yaw=20, roll=0, duration=1.0)
        >>> await client.disconnect()
    """

    def __init__(
        self,
        model_path: str | None = None,
        realtime: bool = True,
        viewer: bool = False,
    ) -> None:
        """Initialize MuJoCo client."""
        self._model_path = model_path
        self._realtime = realtime
        self._viewer_enabled = viewer
        self._env: SimulationEnvironment | None = None
        self._connected = False
        self._awake = False
        self._log = logger.bind(component="mujoco_client")

    # Lifecycle methods

    async def connect(self) -> None:
        """Connect to simulation (load model and initialize)."""
        from reachy_agent.simulation.environment import SimulationEnvironment

        self._log.info("connecting_to_simulation", model_path=self._model_path)

        self._env = SimulationEnvironment(
            model_path=self._model_path,
            realtime=self._realtime,
        )
        await self._env.start()

        if self._viewer_enabled:
            await self._env.start_viewer()

        self._connected = True
        self._log.info("simulation_connected")

    async def disconnect(self) -> None:
        """Disconnect from simulation."""
        self._log.info("disconnecting_from_simulation")

        if self._env:
            await self._env.stop()
            self._env = None

        self._connected = False
        self._awake = False
        self._log.info("simulation_disconnected")

    async def wake_up(self) -> None:
        """Enable motors in simulation."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.info("waking_up_simulation")
        await self._env.enable_actuators()
        self._awake = True

    async def sleep(self) -> None:
        """Disable motors in simulation."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.info("sleeping_simulation")
        await self._env.disable_actuators()
        self._awake = False

    # Movement methods

    async def move_head(
        self, pitch: float, yaw: float, roll: float, duration: float
    ) -> None:
        """Move head to absolute position over duration."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.debug("move_head", pitch=pitch, yaw=yaw, roll=roll, duration=duration)
        await self._env.move_joints(
            targets={"head_pitch": pitch, "head_yaw": yaw, "head_roll": roll},
            duration=duration,
        )

    async def look_at(self, x: float, y: float, z: float, duration: float) -> None:
        """Look at 3D point in robot frame."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        # Calculate head angles from target point using simple geometric IK
        import math

        distance = math.sqrt(x * x + y * y + z * z)
        if distance < 0.001:
            return

        yaw = math.degrees(math.atan2(y, x))
        pitch = math.degrees(math.atan2(-z, math.sqrt(x * x + y * y)))

        await self.move_head(pitch=pitch, yaw=yaw, roll=0, duration=duration)

    async def rotate_body(self, angle: float, duration: float) -> None:
        """Rotate body to angle."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.debug("rotate_body", angle=angle, duration=duration)
        await self._env.move_joints(targets={"body_rotation": angle}, duration=duration)

    async def reset_position(self, duration: float = 1.0) -> None:
        """Return to neutral pose."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.info("reset_position", duration=duration)
        await self._env.move_joints(
            targets={
                "head_pitch": 0,
                "head_yaw": 0,
                "head_roll": 0,
                "head_z": 0,
                "body_rotation": 0,
                "antenna_left": 0,
                "antenna_right": 0,
            },
            duration=duration,
        )

    # Expression methods

    async def play_emotion(self, name: str) -> None:
        """Play emotion animation by name."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.info("play_emotion", name=name)
        await self._env.play_animation(name)

    async def set_antennas(self, left: float, right: float) -> None:
        """Set antenna positions."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.debug("set_antennas", left=left, right=right)
        await self._env.move_joints(
            targets={"antenna_left": left, "antenna_right": right},
            duration=0.2,
        )

    async def nod(self, intensity: float = 1.0) -> None:
        """Nod head (affirmative gesture)."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.debug("nod", intensity=intensity)
        amplitude = 15 * intensity  # degrees

        # Get current position
        positions = await self.get_position()
        base_pitch = positions.get("head_pitch", 0)

        # Nod sequence: down-up-down-up
        for _ in range(2):
            await self.move_head(
                pitch=base_pitch + amplitude,
                yaw=positions.get("head_yaw", 0),
                roll=positions.get("head_roll", 0),
                duration=0.15,
            )
            await self.move_head(
                pitch=base_pitch - amplitude * 0.5,
                yaw=positions.get("head_yaw", 0),
                roll=positions.get("head_roll", 0),
                duration=0.15,
            )

        # Return to original
        await self.move_head(
            pitch=base_pitch,
            yaw=positions.get("head_yaw", 0),
            roll=positions.get("head_roll", 0),
            duration=0.1,
        )

    async def shake(self, intensity: float = 1.0) -> None:
        """Shake head (negative gesture)."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.debug("shake", intensity=intensity)
        amplitude = 20 * intensity  # degrees

        # Get current position
        positions = await self.get_position()
        base_yaw = positions.get("head_yaw", 0)

        # Shake sequence: left-right-left-right
        for _ in range(2):
            await self.move_head(
                pitch=positions.get("head_pitch", 0),
                yaw=base_yaw - amplitude,
                roll=positions.get("head_roll", 0),
                duration=0.12,
            )
            await self.move_head(
                pitch=positions.get("head_pitch", 0),
                yaw=base_yaw + amplitude,
                roll=positions.get("head_roll", 0),
                duration=0.12,
            )

        # Return to original
        await self.move_head(
            pitch=positions.get("head_pitch", 0),
            yaw=base_yaw,
            roll=positions.get("head_roll", 0),
            duration=0.1,
        )

    # Audio methods (simulated)

    async def speak(self, text: str, voice: str = "default") -> None:
        """Text-to-speech output (simulated in MuJoCo)."""
        self._log.info("speak_simulated", text=text[:50], voice=voice)
        # In simulation, we just log the speech
        # Real TTS would be handled by the voice pipeline
        await asyncio.sleep(len(text) * 0.05)  # Simulate speaking time

    async def listen(self, timeout: float = 5.0) -> str:
        """Listen for speech (simulated in MuJoCo)."""
        self._log.info("listen_simulated", timeout=timeout)
        # In simulation, return empty string
        await asyncio.sleep(0.1)
        return ""

    # Perception methods

    async def capture_image(self) -> bytes:
        """Capture camera frame from simulation."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        self._log.debug("capture_image")
        return await self._env.render_camera()

    async def get_sensor_data(self) -> dict[str, float]:
        """Read IMU/accelerometer from simulation."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        return await self._env.get_sensor_data()

    async def detect_sound_direction(self) -> tuple[float, float]:
        """Get direction of loudest sound (simulated)."""
        # In simulation, return a default direction
        return (0.0, 0.0)

    # Status methods

    async def get_status(self) -> RobotStatus:
        """Get robot health and state."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        positions = await self.get_position()

        return RobotStatus(
            is_awake=self._awake,
            battery_percent=100.0,  # Simulated
            head_pose=HeadPose(
                pitch=positions.get("head_pitch", 0),
                yaw=positions.get("head_yaw", 0),
                roll=positions.get("head_roll", 0),
                z=positions.get("head_z", 0),
            ),
            body_angle=positions.get("body_rotation", 0),
            antenna_state=AntennaState(
                left=positions.get("antenna_left", 0),
                right=positions.get("antenna_right", 0),
            ),
        )

    async def get_position(self) -> dict[str, float]:
        """Get current joint positions."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        return await self._env.get_joint_positions()

    async def get_limits(self) -> dict[str, tuple[float, float]]:
        """Get joint angle limits (min, max)."""
        if not self._env:
            raise RuntimeError("Not connected to simulation")

        return await self._env.get_joint_limits()

    async def is_awake(self) -> bool:
        """Check if motors are enabled."""
        return self._awake
