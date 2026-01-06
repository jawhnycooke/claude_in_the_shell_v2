"""Parallel simulation support for batch processing and RL training.

This module provides utilities for running multiple simulation environments
in parallel for faster data collection and policy training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore
    MUJOCO_AVAILABLE = False

logger = structlog.get_logger()


class ParallelSimulation:
    """
    Manages multiple simulation environments running in parallel.

    Uses multiprocessing to run N environments simultaneously,
    enabling efficient batch data collection for RL training.

    Args:
        n_envs: Number of parallel environments
        model_path: Path to MJCF model
        realtime: Run in real-time (default False for training)

    Example:
        >>> parallel = ParallelSimulation(n_envs=8)
        >>> observations = parallel.reset()
        >>> for _ in range(1000):
        ...     actions = policy(observations)
        ...     observations, rewards, dones = parallel.step(actions)
        >>> parallel.close()
    """

    def __init__(
        self,
        n_envs: int = 4,
        model_path: str | None = None,
        realtime: bool = False,
    ) -> None:
        """Initialize parallel simulation."""
        self.n_envs = n_envs
        self.model_path = model_path
        self.realtime = realtime

        self._envs: list[Any] = []
        self._models: list[Any] = []
        self._datas: list[Any] = []
        self._running = False
        self._log = logger.bind(component="parallel_sim", n_envs=n_envs)

        # Observation/action dimensions
        self.n_obs = 20  # 7 pos + 7 vel + 6 imu
        self.n_actions = 7  # 7 joints

    def start(self) -> None:
        """Start all parallel environments."""
        if not MUJOCO_AVAILABLE:
            self._log.warning("mujoco_not_available")
            self._running = True
            return

        from pathlib import Path

        if self.model_path:
            path = Path(self.model_path)
        else:
            path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "models"
                / "reachy_mini"
                / "reachy_mini.xml"
            )

        self._log.info("starting_parallel_envs", path=str(path))

        for i in range(self.n_envs):
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            self._models.append(model)
            self._datas.append(data)

        self._running = True
        self._log.info("parallel_envs_started")

    def stop(self) -> None:
        """Stop all parallel environments."""
        self._log.info("stopping_parallel_envs")
        self._models = []
        self._datas = []
        self._running = False

    def reset(
        self,
        env_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Reset specified environments.

        Args:
            env_ids: Indices of environments to reset (None = all)

        Returns:
            Observations array of shape (n_envs, n_obs)
        """
        if env_ids is None:
            env_ids = list(range(self.n_envs))

        observations = np.zeros((self.n_envs, self.n_obs), dtype=np.float32)

        if MUJOCO_AVAILABLE:
            for i in env_ids:
                if i < len(self._models):
                    mujoco.mj_resetData(self._models[i], self._datas[i])
                    mujoco.mj_forward(self._models[i], self._datas[i])
                    observations[i] = self._get_obs(i)
        else:
            observations[:] = 0

        return observations

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step all environments with given actions.

        Args:
            actions: Array of shape (n_envs, n_actions)

        Returns:
            Tuple of (observations, rewards, dones)
        """
        observations = np.zeros((self.n_envs, self.n_obs), dtype=np.float32)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)

        if MUJOCO_AVAILABLE:
            for i in range(self.n_envs):
                if i < len(self._datas):
                    # Apply action
                    self._datas[i].ctrl[:] = actions[i]

                    # Step simulation
                    mujoco.mj_step(self._models[i], self._datas[i])

                    # Get observation
                    observations[i] = self._get_obs(i)

                    # Compute reward (simple energy penalty)
                    rewards[i] = -0.01 * np.sum(np.square(actions[i]))

        return observations, rewards, dones

    def step_async(self, actions: np.ndarray) -> None:
        """Start async step (non-blocking).

        Args:
            actions: Array of shape (n_envs, n_actions)
        """
        # For now, just store actions for sync step
        self._pending_actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wait for async step to complete.

        Returns:
            Tuple of (observations, rewards, dones)
        """
        if hasattr(self, "_pending_actions"):
            return self.step(self._pending_actions)
        return self.step(np.zeros((self.n_envs, self.n_actions)))

    def _get_obs(self, env_id: int) -> np.ndarray:
        """Get observation from environment."""
        if not MUJOCO_AVAILABLE or env_id >= len(self._datas):
            return np.zeros(self.n_obs, dtype=np.float32)

        data = self._datas[env_id]
        n_joints = 7

        qpos = data.qpos[:n_joints].astype(np.float32)
        qvel = data.qvel[:n_joints].astype(np.float32)

        if len(data.sensordata) >= 6:
            imu = data.sensordata[:6].astype(np.float32)
        else:
            imu = np.zeros(6, dtype=np.float32)

        return np.concatenate([qpos, qvel, imu])

    def set_state(
        self,
        env_id: int,
        qpos: np.ndarray,
        qvel: np.ndarray | None = None,
    ) -> None:
        """Set state of specific environment.

        Args:
            env_id: Environment index
            qpos: Joint positions
            qvel: Joint velocities (optional)
        """
        if MUJOCO_AVAILABLE and env_id < len(self._datas):
            self._datas[env_id].qpos[:] = qpos
            if qvel is not None:
                self._datas[env_id].qvel[:] = qvel
            mujoco.mj_forward(self._models[env_id], self._datas[env_id])

    def get_state(self, env_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Get state of specific environment.

        Args:
            env_id: Environment index

        Returns:
            Tuple of (qpos, qvel)
        """
        if MUJOCO_AVAILABLE and env_id < len(self._datas):
            return self._datas[env_id].qpos.copy(), self._datas[env_id].qvel.copy()
        return np.zeros(7), np.zeros(7)

    def close(self) -> None:
        """Clean up resources."""
        self.stop()


class VectorizedEnv:
    """
    Gymnasium-style vectorized environment wrapper.

    Wraps ParallelSimulation to provide a Gymnasium-compatible interface.

    Example:
        >>> vec_env = VectorizedEnv(n_envs=4)
        >>> obs, infos = vec_env.reset()
        >>> obs, rewards, terms, truncs, infos = vec_env.step(actions)
    """

    def __init__(self, n_envs: int = 4, **kwargs) -> None:
        """Initialize vectorized environment."""
        self._parallel = ParallelSimulation(n_envs=n_envs, **kwargs)
        self._parallel.start()
        self._step_counts = np.zeros(n_envs, dtype=int)
        self._max_steps = 1000

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset all environments."""
        if seed is not None:
            np.random.seed(seed)

        self._step_counts[:] = 0
        obs = self._parallel.reset()

        infos = {"step_counts": self._step_counts.tolist()}
        return obs, infos

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step all environments."""
        self._step_counts += 1

        obs, rewards, dones = self._parallel.step(actions)

        # Check for truncation
        truncated = self._step_counts >= self._max_steps
        terminated = dones

        # Auto-reset done environments
        done_mask = dones | truncated
        if np.any(done_mask):
            done_ids = np.where(done_mask)[0].tolist()
            reset_obs = self._parallel.reset(done_ids)
            obs[done_mask] = reset_obs[done_mask]
            self._step_counts[done_mask] = 0

        infos = {"step_counts": self._step_counts.tolist()}
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        """Clean up resources."""
        self._parallel.close()

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._parallel.n_envs
