"""Tests for MuJoCo simulation support.

This module contains unit and integration tests for the MuJoCo simulation
subsystem, including the client, environment, physics, and viewer components.
"""

from __future__ import annotations

from pathlib import Path

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

        assert (
            info.n_actuators >= 7
        ), f"Expected at least 7 actuators, got {info.n_actuators}"

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


# ============================================================================
# Domain Randomization Tests (F307-F309)
# ============================================================================


class TestDomainRandomization:
    """Tests for domain randomization."""

    def test_domain_randomizer_config(self) -> None:
        """Test domain randomization config defaults."""
        from reachy_agent.simulation.randomization import DomainRandomizationConfig

        config = DomainRandomizationConfig()
        assert config.enabled is False
        assert config.mass_range == (0.8, 1.2)
        assert config.friction_range == (0.5, 1.2)
        assert config.sensor_noise_std == 0.01

    def test_domain_randomizer_disabled(self) -> None:
        """Test domain randomizer when disabled."""
        from reachy_agent.simulation.randomization import (
            DomainRandomizationConfig,
            DomainRandomizer,
        )

        config = DomainRandomizationConfig(enabled=False)
        randomizer = DomainRandomizer(config)

        # Should return data unchanged when disabled
        import numpy as np

        data = np.array([1.0, 2.0, 3.0])
        result = randomizer.add_sensor_noise(data)
        assert np.array_equal(result, data)

    def test_add_sensor_noise(self) -> None:
        """Test sensor noise injection."""
        import numpy as np

        from reachy_agent.simulation.randomization import (
            DomainRandomizationConfig,
            DomainRandomizer,
        )

        config = DomainRandomizationConfig(enabled=True, sensor_noise_std=0.1, seed=42)
        randomizer = DomainRandomizer(config)

        data = np.zeros(100)
        noisy_data = randomizer.add_sensor_noise(data)

        # Should have added noise
        assert not np.array_equal(noisy_data, data)
        # Check noise is reasonable
        assert np.abs(np.mean(noisy_data)) < 0.5

    def test_add_joint_noise(self) -> None:
        """Test joint position noise injection."""
        import numpy as np

        from reachy_agent.simulation.randomization import (
            DomainRandomizationConfig,
            DomainRandomizer,
        )

        config = DomainRandomizationConfig(enabled=True, joint_noise_std=0.05, seed=42)
        randomizer = DomainRandomizer(config)

        positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        noisy_positions = randomizer.add_joint_noise(positions)

        # Should have added noise
        assert not np.array_equal(noisy_positions, positions)

    def test_visual_randomization_config(self) -> None:
        """Test visual randomization config defaults."""
        from reachy_agent.simulation.randomization import VisualRandomizationConfig

        config = VisualRandomizationConfig()
        assert config.enabled is False
        assert config.texture_variation == 0.2
        assert config.lighting_range == (0.7, 1.3)

    def test_visual_randomizer_camera(self) -> None:
        """Test camera pose randomization."""
        import numpy as np

        from reachy_agent.simulation.randomization import (
            VisualRandomizationConfig,
            VisualRandomizer,
        )

        config = VisualRandomizationConfig(enabled=True)
        randomizer = VisualRandomizer(config)

        position = np.array([0.0, 0.0, 1.0])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        new_pos, new_orient = randomizer.randomize_camera(position, orientation)

        # Position should be slightly changed
        assert not np.array_equal(new_pos, position)


# ============================================================================
# Parallel Simulation Tests (F310-F311)
# ============================================================================


class TestParallelSimulation:
    """Tests for parallel simulation."""

    def test_parallel_simulation_init(self) -> None:
        """Test parallel simulation initialization."""
        from reachy_agent.simulation.parallel import ParallelSimulation

        parallel = ParallelSimulation(n_envs=4)
        assert parallel.n_envs == 4
        assert parallel.n_obs == 20
        assert parallel.n_actions == 7

    def test_parallel_simulation_start_stop(self) -> None:
        """Test parallel simulation start/stop."""
        from reachy_agent.simulation.parallel import ParallelSimulation

        parallel = ParallelSimulation(n_envs=2)
        parallel.start()
        assert parallel._running
        parallel.stop()
        assert not parallel._running

    def test_parallel_simulation_reset(self) -> None:
        """Test parallel simulation reset."""
        import numpy as np

        from reachy_agent.simulation.parallel import ParallelSimulation

        parallel = ParallelSimulation(n_envs=4)
        parallel.start()

        obs = parallel.reset()
        assert obs.shape == (4, 20)
        assert obs.dtype == np.float32

        parallel.stop()

    def test_parallel_simulation_step(self) -> None:
        """Test parallel simulation step."""
        import numpy as np

        from reachy_agent.simulation.parallel import ParallelSimulation

        parallel = ParallelSimulation(n_envs=4)
        parallel.start()
        parallel.reset()

        actions = np.zeros((4, 7), dtype=np.float32)
        obs, rewards, dones = parallel.step(actions)

        assert obs.shape == (4, 20)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

        parallel.stop()

    def test_vectorized_env_init(self) -> None:
        """Test vectorized environment initialization."""
        from reachy_agent.simulation.parallel import VectorizedEnv

        vec_env = VectorizedEnv(n_envs=2)
        assert vec_env.num_envs == 2
        vec_env.close()

    def test_vectorized_env_reset(self) -> None:
        """Test vectorized environment reset."""

        from reachy_agent.simulation.parallel import VectorizedEnv

        vec_env = VectorizedEnv(n_envs=2)
        obs, infos = vec_env.reset(seed=42)

        assert obs.shape == (2, 20)
        assert "step_counts" in infos

        vec_env.close()

    def test_vectorized_env_step(self) -> None:
        """Test vectorized environment step."""
        import numpy as np

        from reachy_agent.simulation.parallel import VectorizedEnv

        vec_env = VectorizedEnv(n_envs=2)
        vec_env.reset()

        actions = np.zeros((2, 7), dtype=np.float32)
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)

        assert obs.shape == (2, 20)
        assert len(rewards) == 2
        assert len(terminated) == 2
        assert len(truncated) == 2

        vec_env.close()


# ============================================================================
# Gymnasium Environment Tests (F319)
# ============================================================================

# Check if gymnasium is available
try:
    import gymnasium

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False


class TestGymnasiumEnvironment:
    """Tests for Gymnasium environment wrapper."""

    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
    def test_gym_env_init(self) -> None:
        """Test Gymnasium environment initialization."""
        from reachy_agent.simulation.gym_env import ReachyGymEnv

        env = ReachyGymEnv(render_mode=None)
        assert env.n_joints == 7
        assert env.n_obs == 20
        env.close()

    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
    def test_gym_env_spaces(self) -> None:
        """Test Gymnasium environment spaces."""
        from reachy_agent.simulation.gym_env import ReachyGymEnv

        env = ReachyGymEnv()
        assert env.observation_space.shape == (20,)
        assert env.action_space.shape == (7,)
        env.close()

    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
    def test_gym_env_reset(self) -> None:
        """Test Gymnasium environment reset."""
        from reachy_agent.simulation.gym_env import ReachyGymEnv

        env = ReachyGymEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (20,)
        assert "step_count" in info
        env.close()

    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
    def test_pose_tracking_env(self) -> None:
        """Test pose tracking environment."""
        import numpy as np

        from reachy_agent.simulation.gym_env import ReachyPoseTrackingEnv

        target = np.zeros(7)
        env = ReachyPoseTrackingEnv(target_pose=target)
        assert env.n_obs == 27  # 20 base + 7 target
        env.close()


# ============================================================================
# Scenario Loading Tests (F312-F313)
# ============================================================================


class TestScenarioLoading:
    """Tests for scenario loading."""

    def test_scenario_config_defaults(self) -> None:
        """Test scenario config default values."""
        from reachy_agent.simulation.config import ScenarioConfig

        config = ScenarioConfig()
        assert config.name == "default"
        assert config.initial_pose == {}
        assert config.objects == []

    def test_scenario_config_custom(self) -> None:
        """Test scenario config with custom values."""
        from reachy_agent.simulation.config import ScenarioConfig

        config = ScenarioConfig(
            name="tabletop",
            initial_pose={"head_pitch": 10.0, "head_yaw": 5.0},
            objects=[{"type": "box", "size": [0.1, 0.1, 0.1]}],
        )
        assert config.name == "tabletop"
        assert config.initial_pose["head_pitch"] == 10.0


# ============================================================================
# Reward Function Tests (F318)
# ============================================================================


class TestRewardFunctions:
    """Tests for reward function framework."""

    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
    def test_default_reward(self) -> None:
        """Test default reward computation."""
        import numpy as np

        from reachy_agent.simulation.gym_env import ReachyGymEnv

        env = ReachyGymEnv()
        obs = np.zeros(20)
        action = np.zeros(7)

        reward = env._compute_reward(obs, action)
        assert isinstance(reward, float)
        # Default reward is alive_bonus + action_penalty
        assert reward == 0.1  # No action penalty when action is zero

        env.close()

    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
    def test_pose_tracking_reward(self) -> None:
        """Test pose tracking reward."""
        import numpy as np

        from reachy_agent.simulation.gym_env import ReachyPoseTrackingEnv

        target = np.zeros(7)
        env = ReachyPoseTrackingEnv(target_pose=target)

        # Perfect tracking (obs starts with current pose, ends with target)
        obs = np.zeros(27)  # All zeros = perfect tracking
        action = np.zeros(7)

        reward = env._compute_reward(obs, action)
        # Should get bonus for being at target
        assert reward > 9.0  # 10.0 bonus - small penalty

        env.close()


# ============================================================================
# Speed Control Tests (F321-F322)
# ============================================================================


class TestSpeedControl:
    """Tests for simulation speed control."""

    def test_simulation_config_realtime(self) -> None:
        """Test realtime mode configuration."""
        config = SimulationConfig(realtime=True)
        assert config.realtime is True

    def test_simulation_config_fastforward(self) -> None:
        """Test fast-forward mode configuration."""
        config = SimulationConfig(realtime=False)
        assert config.realtime is False

    @pytest.mark.asyncio
    async def test_step_async_sync(self) -> None:
        """Test async step operations."""
        import numpy as np

        from reachy_agent.simulation.parallel import ParallelSimulation

        parallel = ParallelSimulation(n_envs=2)
        parallel.start()
        parallel.reset()

        actions = np.zeros((2, 7), dtype=np.float32)
        parallel.step_async(actions)
        obs, rewards, dones = parallel.step_wait()

        assert obs.shape == (2, 20)
        parallel.stop()
