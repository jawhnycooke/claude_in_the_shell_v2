"""MuJoCo simulation environment wrapper.

This module provides a high-level wrapper around MuJoCo for simulating
the Reachy Mini robot.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

try:
    import mujoco
    from mujoco import viewer as mj_viewer

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore
    mj_viewer = None  # type: ignore
    MUJOCO_AVAILABLE = False

from reachy_agent.simulation.physics import PDController, TrajectoryInterpolator

logger = structlog.get_logger()

# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent.parent / "data" / "models" / "reachy_mini" / "reachy_mini.xml"

# Joint configuration matching Reachy Mini
JOINT_CONFIG = {
    "head_pitch": {"limit": (-45, 35), "default": 0},
    "head_yaw": {"limit": (-60, 60), "default": 0},
    "head_roll": {"limit": (-35, 35), "default": 0},
    "head_z": {"limit": (0, 50), "default": 0},
    "body_rotation": {"limit": (-180, 180), "default": 0},
    "antenna_left": {"limit": (-150, 150), "default": 0},
    "antenna_right": {"limit": (-150, 150), "default": 0},
}


class SimulationEnvironment:
    """
    High-level wrapper for MuJoCo simulation of Reachy Mini.

    Args:
        model_path: Path to MJCF model file
        timestep: Physics timestep in seconds (default: 0.002 for 500Hz)
        realtime: Run simulation in real-time (default: True)

    Example:
        >>> env = SimulationEnvironment()
        >>> await env.start()
        >>> await env.move_joints({"head_pitch": 10}, duration=1.0)
        >>> await env.stop()
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        timestep: float = 0.002,
        realtime: bool = True,
    ) -> None:
        """Initialize simulation environment."""
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._timestep = timestep
        self._realtime = realtime

        self._model: Any = None
        self._data: Any = None
        self._viewer: Any = None
        self._running = False
        self._physics_task: asyncio.Task[None] | None = None
        self._actuators_enabled = False

        # Controllers for each joint
        self._controllers: dict[str, PDController] = {
            name: PDController(kp=100.0, kd=10.0) for name in JOINT_CONFIG
        }
        self._interpolator = TrajectoryInterpolator(method="minimum_jerk")

        # Target positions for motion
        self._targets: dict[str, float] = {}
        self._motion_start_time: float = 0
        self._motion_duration: float = 0
        self._motion_start_positions: dict[str, float] = {}

        self._log = logger.bind(component="sim_environment")

    async def start(self) -> None:
        """Start the simulation."""
        if not MUJOCO_AVAILABLE:
            self._log.warning("mujoco_not_available", using_mock=True)
            self._running = True
            self._targets = {name: cfg["default"] for name, cfg in JOINT_CONFIG.items()}
            return

        self._log.info("starting_simulation", model_path=str(self._model_path))

        try:
            self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
            self._model.opt.timestep = self._timestep
            self._data = mujoco.MjData(self._model)

            # Initialize targets to current positions
            self._targets = await self.get_joint_positions()

            self._running = True
            self._physics_task = asyncio.create_task(self._physics_loop())
            self._log.info("simulation_started")

        except Exception as e:
            self._log.error("simulation_start_failed", error=str(e))
            # Fall back to mock mode
            self._running = True
            self._targets = {name: cfg["default"] for name, cfg in JOINT_CONFIG.items()}

    async def stop(self) -> None:
        """Stop the simulation."""
        self._log.info("stopping_simulation")
        self._running = False

        if self._physics_task:
            self._physics_task.cancel()
            try:
                await self._physics_task
            except asyncio.CancelledError:
                pass
            self._physics_task = None

        if self._viewer:
            await self.stop_viewer()

        self._model = None
        self._data = None
        self._log.info("simulation_stopped")

    async def start_viewer(self) -> None:
        """Start the visualization window."""
        if not MUJOCO_AVAILABLE or self._model is None:
            self._log.warning("cannot_start_viewer", reason="mujoco_not_available")
            return

        self._log.info("starting_viewer")
        self._viewer = mj_viewer.launch_passive(self._model, self._data)

    async def stop_viewer(self) -> None:
        """Stop the visualization window."""
        if self._viewer:
            self._log.info("stopping_viewer")
            self._viewer.close()
            self._viewer = None

    async def step(self) -> None:
        """Advance simulation by one timestep."""
        if not MUJOCO_AVAILABLE or self._model is None:
            return

        # Apply control based on targets
        self._apply_control()

        # Step physics
        mujoco.mj_step(self._model, self._data)

        # Sync viewer if active
        if self._viewer:
            self._viewer.sync()

    async def _physics_loop(self) -> None:
        """Main physics loop running at specified timestep."""
        last_time = time.perf_counter()

        while self._running:
            current_time = time.perf_counter()

            if self._realtime:
                # Wait to maintain real-time
                elapsed = current_time - last_time
                if elapsed < self._timestep:
                    await asyncio.sleep(self._timestep - elapsed)
                last_time = time.perf_counter()
            else:
                # Fast-forward mode
                await asyncio.sleep(0)  # Yield to event loop

            await self.step()

    def _apply_control(self) -> None:
        """Apply PD control to reach target positions."""
        if not MUJOCO_AVAILABLE or self._data is None:
            return

        current_time = time.time()
        elapsed = current_time - self._motion_start_time

        for joint_name, controller in self._controllers.items():
            if joint_name not in self._targets:
                continue

            # Get joint ID
            joint_id = self._get_joint_id(joint_name)
            if joint_id is None:
                continue

            # Interpolate target
            if joint_name in self._motion_start_positions:
                start = self._motion_start_positions[joint_name]
                end = self._targets[joint_name]
                target = self._interpolator.interpolate(
                    start, end, elapsed, self._motion_duration
                )
            else:
                target = self._targets[joint_name]

            # Get current state
            current = self._data.qpos[joint_id]
            velocity = self._data.qvel[joint_id]

            # Compute and apply control
            if self._actuators_enabled:
                torque = controller.compute(
                    np.radians(target), current, velocity
                )
                actuator_id = self._get_actuator_id(joint_name)
                if actuator_id is not None:
                    self._data.ctrl[actuator_id] = torque

    def _get_joint_id(self, joint_name: str) -> int | None:
        """Get MuJoCo joint ID by name."""
        if not MUJOCO_AVAILABLE or self._model is None:
            return None
        try:
            return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        except Exception:
            return None

    def _get_actuator_id(self, joint_name: str) -> int | None:
        """Get MuJoCo actuator ID by joint name."""
        if not MUJOCO_AVAILABLE or self._model is None:
            return None
        try:
            actuator_name = f"{joint_name}_actuator"
            return mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
        except Exception:
            return None

    async def enable_actuators(self) -> None:
        """Enable all actuators."""
        self._actuators_enabled = True
        self._log.info("actuators_enabled")

    async def disable_actuators(self) -> None:
        """Disable all actuators (gravity compensation only)."""
        self._actuators_enabled = False
        if self._data is not None:
            self._data.ctrl[:] = 0
        self._log.info("actuators_disabled")

    async def move_joints(
        self,
        targets: dict[str, float],
        duration: float,
    ) -> None:
        """Move joints to target positions over duration.

        Args:
            targets: Dict mapping joint names to target positions (degrees)
            duration: Time to complete motion (seconds)
        """
        # Store motion parameters
        self._motion_start_time = time.time()
        self._motion_duration = duration
        self._motion_start_positions = await self.get_joint_positions()

        # Update targets
        for name, value in targets.items():
            if name in JOINT_CONFIG:
                # Clamp to limits
                limits = JOINT_CONFIG[name]["limit"]
                self._targets[name] = max(limits[0], min(limits[1], value))

        # Wait for motion to complete
        await asyncio.sleep(duration)

    async def play_animation(self, name: str) -> None:
        """Play a named animation/emotion."""
        self._log.info("playing_animation", name=name)
        # TODO: Load animation from file and play
        await asyncio.sleep(0.5)  # Placeholder

    async def get_joint_positions(self) -> dict[str, float]:
        """Get current joint positions in degrees."""
        if not MUJOCO_AVAILABLE or self._data is None:
            return dict(self._targets)

        positions = {}
        for joint_name in JOINT_CONFIG:
            joint_id = self._get_joint_id(joint_name)
            if joint_id is not None:
                positions[joint_name] = float(np.degrees(self._data.qpos[joint_id]))
            else:
                positions[joint_name] = self._targets.get(joint_name, 0)

        return positions

    async def get_joint_velocities(self) -> dict[str, float]:
        """Get current joint velocities in degrees/second."""
        if not MUJOCO_AVAILABLE or self._data is None:
            return {name: 0.0 for name in JOINT_CONFIG}

        velocities = {}
        for joint_name in JOINT_CONFIG:
            joint_id = self._get_joint_id(joint_name)
            if joint_id is not None:
                velocities[joint_name] = float(np.degrees(self._data.qvel[joint_id]))
            else:
                velocities[joint_name] = 0.0

        return velocities

    async def get_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Get joint angle limits (min, max) in degrees."""
        return {name: cfg["limit"] for name, cfg in JOINT_CONFIG.items()}

    async def get_sensor_data(self) -> dict[str, float]:
        """Get IMU sensor data."""
        if not MUJOCO_AVAILABLE or self._data is None:
            return {
                "accel_x": 0.0,
                "accel_y": 0.0,
                "accel_z": 9.81,
                "gyro_x": 0.0,
                "gyro_y": 0.0,
                "gyro_z": 0.0,
            }

        # Read from MuJoCo sensors
        # TODO: Map to actual sensor IDs in MJCF
        return {
            "accel_x": float(self._data.sensordata[0]) if len(self._data.sensordata) > 0 else 0.0,
            "accel_y": float(self._data.sensordata[1]) if len(self._data.sensordata) > 1 else 0.0,
            "accel_z": float(self._data.sensordata[2]) if len(self._data.sensordata) > 2 else 9.81,
            "gyro_x": float(self._data.sensordata[3]) if len(self._data.sensordata) > 3 else 0.0,
            "gyro_y": float(self._data.sensordata[4]) if len(self._data.sensordata) > 4 else 0.0,
            "gyro_z": float(self._data.sensordata[5]) if len(self._data.sensordata) > 5 else 0.0,
        }

    async def render_camera(self, camera_name: str = "head_camera") -> bytes:
        """Render image from camera.

        Args:
            camera_name: Name of camera in MJCF model

        Returns:
            JPEG-encoded image bytes
        """
        if not MUJOCO_AVAILABLE or self._model is None or self._data is None:
            # Return a placeholder 1x1 black image
            import io

            try:
                from PIL import Image

                img = Image.new("RGB", (640, 480), color="black")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                return buffer.getvalue()
            except ImportError:
                return b""

        # Render using MuJoCo
        width, height = 640, 480
        renderer = mujoco.Renderer(self._model, height, width)

        try:
            camera_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
        except Exception:
            camera_id = -1  # Free camera

        renderer.update_scene(self._data, camera=camera_id)
        pixels = renderer.render()

        # Convert to JPEG
        import io

        try:
            from PIL import Image

            img = Image.fromarray(pixels)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return buffer.getvalue()
        except ImportError:
            return pixels.tobytes()

    async def reset(self) -> None:
        """Reset simulation to initial state."""
        if MUJOCO_AVAILABLE and self._model is not None:
            mujoco.mj_resetData(self._model, self._data)

        self._targets = {name: cfg["default"] for name, cfg in JOINT_CONFIG.items()}
        self._motion_start_positions = {}
        self._log.info("simulation_reset")

    @property
    def time(self) -> float:
        """Get current simulation time."""
        if self._data is not None:
            return float(self._data.time)
        return 0.0

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running
