"""Gymnasium environment wrapper for Reachy Mini simulation.

This module provides a Gymnasium-compatible environment for reinforcement
learning with the Reachy Mini robot simulation.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import structlog

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    gym = None  # type: ignore
    spaces = None  # type: ignore
    GYMNASIUM_AVAILABLE = False

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore
    MUJOCO_AVAILABLE = False

logger = structlog.get_logger()


class ReachyGymEnv:
    """
    Gymnasium environment for Reachy Mini robot.

    Implements the standard Gymnasium interface (step, reset, render, close)
    for reinforcement learning with the Reachy Mini simulation.

    Observation space:
        - Joint positions (7 joints) in radians
        - Joint velocities (7 joints) in rad/s
        - IMU accelerometer (3 axes)
        - IMU gyroscope (3 axes)
        Total: 20 dimensions

    Action space:
        - Target joint positions (7 joints) normalized to [-1, 1]

    Args:
        render_mode: "human" for viewer, "rgb_array" for image
        max_episode_steps: Maximum steps per episode
        control_frequency: Control frequency in Hz

    Example:
        >>> env = ReachyGymEnv(render_mode="human")
        >>> obs, info = env.reset()
        >>> for _ in range(1000):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        >>> env.close()
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        control_frequency: float = 50.0,
    ) -> None:
        """Initialize the environment."""
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium not installed. Install with: pip install gymnasium"
            )

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.control_dt = 1.0 / control_frequency

        # Joint limits (degrees converted to radians)
        self.joint_limits = {
            "body_rotation": np.radians([-180, 180]),
            "head_z": [0, 0.05],  # meters
            "head_yaw": np.radians([-60, 60]),
            "head_pitch": np.radians([-45, 35]),
            "head_roll": np.radians([-35, 35]),
            "antenna_left": np.radians([-150, 150]),
            "antenna_right": np.radians([-150, 150]),
        }

        self.n_joints = 7
        self.n_obs = 20  # 7 pos + 7 vel + 3 accel + 3 gyro

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_obs,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_joints,),
            dtype=np.float32,
        )

        # MuJoCo model/data
        self._model: Any = None
        self._data: Any = None
        self._viewer: Any = None

        self._step_count = 0
        self._log = logger.bind(component="reachy_gym_env")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Additional options (e.g., initial pose)

        Returns:
            Tuple of (observation, info_dict)
        """
        if seed is not None:
            np.random.seed(seed)

        self._step_count = 0

        # Load model if not loaded
        if self._model is None:
            self._load_model()

        # Reset simulation
        if MUJOCO_AVAILABLE and self._model is not None:
            mujoco.mj_resetData(self._model, self._data)

            # Apply initial pose if provided
            if options and "initial_pose" in options:
                for i, pos in enumerate(options["initial_pose"]):
                    self._data.qpos[i] = pos

            mujoco.mj_forward(self._model, self._data)

        obs = self._get_observation()
        info = {"step_count": 0}

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Normalized joint position targets [-1, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # Convert normalized action to joint targets
        targets = self._action_to_targets(action)

        # Apply control
        if MUJOCO_AVAILABLE and self._data is not None:
            self._data.ctrl[:] = targets

            # Step simulation
            n_steps = int(self.control_dt / self._model.opt.timestep)
            for _ in range(n_steps):
                mujoco.mj_step(self._model, self._data)

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(obs, action)

        # Check termination
        terminated = self._check_termination(obs)
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "step_count": self._step_count,
            "joint_positions": obs[: self.n_joints].tolist(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if not MUJOCO_AVAILABLE or self._model is None:
            return None

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer

                self._viewer = mj_viewer.launch_passive(self._model, self._data)
            self._viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self._model, 480, 640)
            renderer.update_scene(self._data)
            return renderer.render()

        return None

    def close(self) -> None:
        """Clean up environment resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._model = None
        self._data = None

    def _load_model(self) -> None:
        """Load the MuJoCo model."""
        if not MUJOCO_AVAILABLE:
            self._log.warning("mujoco_not_available")
            return

        from pathlib import Path

        model_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "models"
            / "reachy_mini"
            / "reachy_mini.xml"
        )

        if model_path.exists():
            self._model = mujoco.MjModel.from_xml_path(str(model_path))
            self._data = mujoco.MjData(self._model)
            self._log.info("model_loaded", path=str(model_path))
        else:
            self._log.error("model_not_found", path=str(model_path))

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if not MUJOCO_AVAILABLE or self._data is None:
            return np.zeros(self.n_obs, dtype=np.float32)

        # Joint positions and velocities
        qpos = self._data.qpos[: self.n_joints].astype(np.float32)
        qvel = self._data.qvel[: self.n_joints].astype(np.float32)

        # IMU data (if available)
        if len(self._data.sensordata) >= 6:
            accel = self._data.sensordata[:3].astype(np.float32)
            gyro = self._data.sensordata[3:6].astype(np.float32)
        else:
            accel = np.zeros(3, dtype=np.float32)
            gyro = np.zeros(3, dtype=np.float32)

        return np.concatenate([qpos, qvel, accel, gyro])

    def _action_to_targets(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action to joint targets.

        Args:
            action: Normalized action [-1, 1]

        Returns:
            Joint targets in radians
        """
        targets = np.zeros(self.n_joints)
        limits = list(self.joint_limits.values())

        for i, (act, lim) in enumerate(zip(action, limits)):
            # Map [-1, 1] to [min, max]
            targets[i] = lim[0] + (act + 1) * 0.5 * (lim[1] - lim[0])

        return targets

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute step reward.

        Override this method to implement custom reward functions.

        Args:
            obs: Current observation
            action: Action taken

        Returns:
            Scalar reward
        """
        # Default: penalize large actions (energy efficiency)
        action_penalty = -0.01 * np.sum(np.square(action))

        # Bonus for being alive
        alive_bonus = 0.1

        return float(action_penalty + alive_bonus)

    def _check_termination(self, obs: np.ndarray) -> bool:
        """Check if episode should terminate.

        Override this method for custom termination conditions.

        Args:
            obs: Current observation

        Returns:
            True if episode should terminate
        """
        # Default: never terminate early
        return False


class ReachyPoseTrackingEnv(ReachyGymEnv):
    """
    Gymnasium environment for pose tracking task.

    The agent must track a target pose with the robot head.

    Args:
        target_pose: Target joint positions (7 values)
        pose_threshold: Position error threshold for success
    """

    def __init__(
        self,
        target_pose: np.ndarray | None = None,
        pose_threshold: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize pose tracking environment."""
        super().__init__(**kwargs)

        self.target_pose = (
            target_pose if target_pose is not None else np.zeros(self.n_joints)
        )
        self.pose_threshold = pose_threshold

        # Add target to observation space
        self.n_obs += self.n_joints
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_obs,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset with optional new target pose."""
        if options and "target_pose" in options:
            self.target_pose = np.array(options["target_pose"])

        obs, info = super().reset(seed=seed, options=options)

        # Append target to observation
        full_obs = np.concatenate([obs, self.target_pose.astype(np.float32)])
        info["target_pose"] = self.target_pose.tolist()

        return full_obs, info

    def _get_observation(self) -> np.ndarray:
        """Get observation with target pose."""
        base_obs = super()._get_observation()
        return np.concatenate([base_obs, self.target_pose.astype(np.float32)])

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Compute reward based on pose tracking error."""
        current_pose = obs[: self.n_joints]
        target_pose = obs[-self.n_joints :]

        # Pose error
        pose_error = np.sum(np.square(current_pose - target_pose))
        pose_reward = -pose_error

        # Bonus for reaching target
        if pose_error < self.pose_threshold:
            pose_reward += 10.0

        # Action penalty
        action_penalty = -0.01 * np.sum(np.square(action))

        return float(pose_reward + action_penalty)

    def _check_termination(self, obs: np.ndarray) -> bool:
        """Terminate when target is reached."""
        current_pose = obs[: self.n_joints]
        target_pose = obs[-self.n_joints :]
        error = np.sum(np.square(current_pose - target_pose))
        return error < self.pose_threshold
