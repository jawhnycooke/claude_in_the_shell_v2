"""Tests for MuJoCo simulation support.

This module contains unit and integration tests for the MuJoCo simulation
subsystem, including the client, environment, physics, and viewer components.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

# Check if MuJoCo is available
try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from reachy_agent.robot.client import Backend
from reachy_agent.simulation.config import (
    PhysicsConfig,
    RenderConfig,
    RenderQuality,
    SimulationConfig,
)
from reachy_agent.simulation.physics import (
    PDController,
    TrajectoryInterpolator,
    add_sensor_noise,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sim_config() -> SimulationConfig:
    """Create a test simulation config."""
    return SimulationConfig(
        model_path="data/models/reachy_mini/reachy_mini.xml",
        timestep=0.002,
        realtime=False,  # Fast-forward for tests
        viewer=False,
        headless=True,
    )


@pytest.fixture
def pd_controller() -> PDController:
    """Create a PD controller for testing."""
    return PDController(kp=100.0, kd=10.0, max_torque=50.0)


@pytest.fixture
def interpolator() -> TrajectoryInterpolator:
    """Create a trajectory interpolator for testing."""
    return TrajectoryInterpolator(method="minimum_jerk")


@pytest.fixture
async def sim_client():
    """Create a simulation client for testing."""
    from reachy_agent.simulation.client import MuJoCoReachyClient

    client = MuJoCoReachyClient(realtime=False, viewer=False)
    await client.connect()
    yield client
    await client.disconnect()


@pytest.fixture
async def sim_environment():
    """Create a simulation environment for testing."""
    from reachy_agent.simulation.environment import SimulationEnvironment

    env = SimulationEnvironment(realtime=False)
    await env.start()
    yield env
    await env.stop()


# ============================================================================
# Configuration Tests (F222, F223)
# ============================================================================


class TestSimulationConfig:
    """Tests for simulation configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.timestep == 0.002
        assert config.n_substeps == 4
        assert config.realtime is True
        assert config.viewer is False

    def test_physics_hz(self) -> None:
        """Test physics frequency calculation."""
        config = SimulationConfig(timestep=0.002, n_substeps=4)
        assert config.physics_hz == 2000.0  # 0.002 / 4 = 0.0005s = 2000Hz

    def test_render_config(self) -> None:
        """Test render configuration."""
        render = RenderConfig(width=1280, height=720, quality=RenderQuality.HIGH)
        assert render.width == 1280
        assert render.height == 720
        assert render.shadows is True

    def test_physics_config(self) -> None:
        """Test physics configuration."""
        physics = PhysicsConfig(gravity=(0, 0, -10.0), friction=0.5)
        assert physics.gravity == (0, 0, -10.0)
        assert physics.friction == 0.5

    def test_from_yaml(self) -> None:
        """Test creating config from YAML dict."""
        yaml_data = {
            "simulation": {
                "timestep": 0.001,
                "realtime": False,
                "viewer": True,
            }
        }
        config = SimulationConfig.from_yaml(yaml_data)
        assert config.timestep == 0.001
        assert config.realtime is False
        assert config.viewer is True


# ============================================================================
# Physics Tests (F261-F278)
# ============================================================================


class TestPDController:
    """Tests for PD controller."""

    def test_compute_zero_error(self, pd_controller: PDController) -> None:
        """Test controller output with zero error."""
        torque = pd_controller.compute(target=0, current=0, velocity=0)
        assert torque == 0.0

    def test_compute_positive_error(self, pd_controller: PDController) -> None:
        """Test controller output with positive error."""
        torque = pd_controller.compute(target=10, current=0, velocity=0)
        assert torque > 0  # Should push toward target

    def test_compute_negative_error(self, pd_controller: PDController) -> None:
        """Test controller output with negative error."""
        torque = pd_controller.compute(target=-10, current=0, velocity=0)
        assert torque < 0  # Should push toward target

    def test_compute_velocity_damping(self, pd_controller: PDController) -> None:
        """Test velocity damping effect."""
        # Use smaller error to avoid clamping
        controller = PDController(kp=10.0, kd=5.0, max_torque=500.0)
        torque_stationary = controller.compute(target=1, current=0, velocity=0)
        torque_moving = controller.compute(target=1, current=0, velocity=1)
        assert torque_moving < torque_stationary  # Damping reduces torque

    def test_torque_clamping(self, pd_controller: PDController) -> None:
        """Test torque clamping to max_torque."""
        torque = pd_controller.compute(target=1000, current=0, velocity=0)
        assert abs(torque) <= pd_controller.max_torque


class TestTrajectoryInterpolator:
    """Tests for trajectory interpolation."""

    def test_linear_interpolation(self) -> None:
        """Test linear interpolation."""
        interp = TrajectoryInterpolator(method="linear")
        assert interp.interpolate(0, 10, t=0, duration=1) == 0
        assert interp.interpolate(0, 10, t=0.5, duration=1) == 5
        assert interp.interpolate(0, 10, t=1, duration=1) == 10

    def test_minimum_jerk_endpoints(self, interpolator: TrajectoryInterpolator) -> None:
        """Test minimum jerk trajectory endpoints."""
        assert interpolator.interpolate(0, 10, t=0, duration=1) == 0
        assert abs(interpolator.interpolate(0, 10, t=1, duration=1) - 10) < 0.001

    def test_minimum_jerk_smooth(self, interpolator: TrajectoryInterpolator) -> None:
        """Test minimum jerk trajectory smoothness."""
        # Sample trajectory
        positions = [
            interpolator.interpolate(0, 10, t=i * 0.1, duration=1) for i in range(11)
        ]
        # Check monotonically increasing
        for i in range(len(positions) - 1):
            assert positions[i] <= positions[i + 1]

    def test_velocity_at_endpoints(self, interpolator: TrajectoryInterpolator) -> None:
        """Test velocity is zero at endpoints for minimum jerk."""
        v_start = interpolator.interpolate_velocity(0, 10, t=0, duration=1)
        v_end = interpolator.interpolate_velocity(0, 10, t=1, duration=1)
        assert abs(v_start) < 0.001
        assert abs(v_end) < 0.001

    def test_zero_duration(self, interpolator: TrajectoryInterpolator) -> None:
        """Test behavior with zero duration."""
        result = interpolator.interpolate(0, 10, t=0, duration=0)
        assert result == 10  # Should jump to end


class TestSensorNoise:
    """Tests for sensor noise simulation."""

    def test_no_noise(self) -> None:
        """Test with zero noise."""
        value = add_sensor_noise(5.0, noise_std=0.0, bias=0.0)
        assert value == 5.0

    def test_bias_only(self) -> None:
        """Test with bias only."""
        value = add_sensor_noise(5.0, noise_std=0.0, bias=1.0)
        assert value == 6.0

    def test_noise_distribution(self) -> None:
        """Test noise has correct distribution."""
        import numpy as np

        values = [add_sensor_noise(0.0, noise_std=1.0, bias=0.0) for _ in range(1000)]
        # Check mean is approximately 0
        assert abs(np.mean(values)) < 0.1
        # Check std is approximately 1
        assert abs(np.std(values) - 1.0) < 0.1


# ============================================================================
# Model Tests (F214-F221)
# ============================================================================


class TestModelLoader:
    """Tests for MJCF model loading."""

    def test_model_path_exists(self) -> None:
        """Test that default model path exists."""
        model_path = Path("data/models/reachy_mini/reachy_mini.xml")
        assert model_path.exists(), f"Model file not found: {model_path}"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_model_loads(self) -> None:
        """Test that model loads without errors."""
        from reachy_agent.simulation.model_loader import MJCFModelLoader

        loader = MJCFModelLoader()
        model, data = loader.load_model("data/models/reachy_mini/reachy_mini.xml")
        assert model is not None
        assert data is not None

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_model_has_joints(self) -> None:
        """Test that model has expected joints."""
        from reachy_agent.simulation.model_loader import MJCFModelLoader

        loader = MJCFModelLoader()
        model, _ = loader.load_model("data/models/reachy_mini/reachy_mini.xml")
        info = loader.get_model_info(model)

        expected_joints = [
            "body_rotation",
            "head_z",
            "head_yaw",
            "head_pitch",
            "head_roll",
            "antenna_left",
            "antenna_right",
        ]
        for joint in expected_joints:
            assert joint in info.joint_names, f"Missing joint: {joint}"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_model_has_actuators(self) -> None:
        """Test that model has expected actuators."""
        from reachy_agent.simulation.model_loader import MJCFModelLoader

        loader = MJCFModelLoader()
        model, _ = loader.load_model("data/models/reachy_mini/reachy_mini.xml")
        info = loader.get_model_info(model)

        assert info.n_actuators >= 7, f"Expected at least 7 actuators, got {info.n_actuators}"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_model_validation(self) -> None:
        """Test model validation."""
        from reachy_agent.simulation.model_loader import MJCFModelLoader

        loader = MJCFModelLoader()
        model, _ = loader.load_model("data/models/reachy_mini/reachy_mini.xml")
        result = loader.validate_model(model)

        assert result.valid, f"Model validation failed: {result.errors}"

    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    def test_load_missing_file(self) -> None:
        """Test error handling for missing file."""
        from reachy_agent.simulation.model_loader import MJCFModelLoader

        loader = MJCFModelLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_model("nonexistent_model.xml")


# ============================================================================
# Client Tests (F229-F246)
# ============================================================================


class TestMuJoCoClient:
    """Tests for MuJoCo client."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        """Test client connection lifecycle."""
        from reachy_agent.simulation.client import MuJoCoReachyClient

        client = MuJoCoReachyClient(realtime=False)
        await client.connect()
        assert client._connected
        await client.disconnect()
        assert not client._connected

    @pytest.mark.asyncio
    async def test_wake_sleep(self, sim_client) -> None:
        """Test wake up and sleep."""
        await sim_client.wake_up()
        assert await sim_client.is_awake()

        await sim_client.sleep()
        assert not await sim_client.is_awake()

    @pytest.mark.asyncio
    async def test_get_status(self, sim_client) -> None:
        """Test getting robot status."""
        await sim_client.wake_up()
        status = await sim_client.get_status()

        assert status.is_awake
        assert status.battery_percent == 100.0  # Simulated
        assert hasattr(status, "head_pose")
        assert hasattr(status, "antenna_state")

    @pytest.mark.asyncio
    async def test_get_position(self, sim_client) -> None:
        """Test getting joint positions."""
        positions = await sim_client.get_position()

        assert "head_pitch" in positions
        assert "head_yaw" in positions
        assert "head_roll" in positions
        assert "antenna_left" in positions
        assert "antenna_right" in positions

    @pytest.mark.asyncio
    async def test_get_limits(self, sim_client) -> None:
        """Test getting joint limits."""
        limits = await sim_client.get_limits()

        assert "head_pitch" in limits
        min_val, max_val = limits["head_pitch"]
        assert min_val == -45
        assert max_val == 35

    @pytest.mark.asyncio
    async def test_move_head(self, sim_client) -> None:
        """Test head movement."""
        await sim_client.wake_up()
        await sim_client.move_head(pitch=10, yaw=20, roll=5, duration=0.1)

        positions = await sim_client.get_position()
        # Allow some tolerance due to physics
        assert abs(positions["head_pitch"] - 10) < 5
        assert abs(positions["head_yaw"] - 20) < 5

    @pytest.mark.asyncio
    async def test_set_antennas(self, sim_client) -> None:
        """Test antenna movement."""
        await sim_client.wake_up()
        await sim_client.set_antennas(left=45, right=-45)

        positions = await sim_client.get_position()
        assert abs(positions["antenna_left"] - 45) < 10
        assert abs(positions["antenna_right"] - (-45)) < 10

    @pytest.mark.asyncio
    async def test_reset_position(self, sim_client) -> None:
        """Test reset to neutral position."""
        await sim_client.wake_up()
        await sim_client.move_head(pitch=20, yaw=30, roll=10, duration=0.1)
        await sim_client.reset_position(duration=0.1)

        positions = await sim_client.get_position()
        assert abs(positions["head_pitch"]) < 5
        assert abs(positions["head_yaw"]) < 5
        assert abs(positions["head_roll"]) < 5


# ============================================================================
# Environment Tests (F224-F225)
# ============================================================================


class TestSimulationEnvironment:
    """Tests for simulation environment."""

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test environment lifecycle."""
        from reachy_agent.simulation.environment import SimulationEnvironment

        env = SimulationEnvironment(realtime=False)
        await env.start()
        assert env.is_running
        await env.stop()
        assert not env.is_running

    @pytest.mark.asyncio
    async def test_get_joint_positions(self, sim_environment) -> None:
        """Test getting joint positions."""
        positions = await sim_environment.get_joint_positions()
        assert isinstance(positions, dict)
        assert "head_pitch" in positions

    @pytest.mark.asyncio
    async def test_get_joint_limits(self, sim_environment) -> None:
        """Test getting joint limits."""
        limits = await sim_environment.get_joint_limits()
        assert "head_pitch" in limits
        assert limits["head_pitch"] == (-45, 35)

    @pytest.mark.asyncio
    async def test_move_joints(self, sim_environment) -> None:
        """Test moving joints."""
        await sim_environment.enable_actuators()
        await sim_environment.move_joints({"head_pitch": 15}, duration=0.1)

        positions = await sim_environment.get_joint_positions()
        # Allow tolerance due to physics
        assert abs(positions["head_pitch"] - 15) < 10

    @pytest.mark.asyncio
    async def test_reset(self, sim_environment) -> None:
        """Test environment reset."""
        await sim_environment.enable_actuators()
        await sim_environment.move_joints({"head_yaw": 30}, duration=0.1)
        await sim_environment.reset()

        positions = await sim_environment.get_joint_positions()
        assert abs(positions["head_yaw"]) < 5


# ============================================================================
# Backend Integration Tests (F228, F296)
# ============================================================================


class TestBackendIntegration:
    """Tests for backend selection and integration."""

    def test_backend_enum_has_sim(self) -> None:
        """Test Backend enum includes SIM."""
        assert Backend.SIM.value == "sim"

    def test_backend_values(self) -> None:
        """Test all backend values."""
        assert Backend.SDK.value == "sdk"
        assert Backend.MOCK.value == "mock"
        assert Backend.SIM.value == "sim"


# ============================================================================
# Performance Tests (F286)
# ============================================================================


class TestSimulationPerformance:
    """Performance tests for simulation."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
    async def test_step_time(self, sim_environment) -> None:
        """Test physics step time is reasonable."""
        import time

        # Step 100 times and measure
        start = time.perf_counter()
        for _ in range(100):
            await sim_environment.step()
        elapsed = time.perf_counter() - start

        # Should be much faster than real-time in non-realtime mode
        avg_step_time = elapsed / 100
        assert avg_step_time < 0.01, f"Step time too slow: {avg_step_time:.4f}s"
