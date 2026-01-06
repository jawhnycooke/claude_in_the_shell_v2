"""Domain randomization utilities for sim-to-real transfer.

This module provides tools for randomizing physics, visual, and sensor
parameters to improve policy robustness for sim-to-real transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization.

    Attributes:
        enabled: Enable domain randomization
        seed: Random seed for reproducibility
        mass_range: Range for mass scaling (min, max)
        friction_range: Range for friction coefficient
        damping_range: Range for damping coefficient
        joint_noise_std: Standard deviation for joint position noise
        sensor_noise_std: Standard deviation for sensor noise
        sensor_bias_range: Range for sensor bias
        actuator_strength_range: Range for actuator strength scaling
    """

    enabled: bool = False
    seed: int | None = None

    # Physics randomization
    mass_range: tuple[float, float] = (0.8, 1.2)
    friction_range: tuple[float, float] = (0.5, 1.2)
    damping_range: tuple[float, float] = (0.5, 2.0)
    joint_noise_std: float = 0.01  # radians

    # Sensor randomization
    sensor_noise_std: float = 0.01
    sensor_bias_range: tuple[float, float] = (-0.02, 0.02)
    sensor_delay_range: tuple[int, int] = (0, 2)  # timesteps

    # Actuator randomization
    actuator_strength_range: tuple[float, float] = (0.9, 1.1)


@dataclass
class VisualRandomizationConfig:
    """Configuration for visual domain randomization.

    Attributes:
        enabled: Enable visual randomization
        texture_variation: Variation factor for textures
        lighting_range: Range for light intensity scaling
        camera_position_noise: Noise for camera position (meters)
        camera_orientation_noise: Noise for camera orientation (radians)
    """

    enabled: bool = False
    texture_variation: float = 0.2
    lighting_range: tuple[float, float] = (0.7, 1.3)
    camera_position_noise: float = 0.01
    camera_orientation_noise: float = 0.05


class DomainRandomizer:
    """
    Applies domain randomization to MuJoCo model for sim-to-real transfer.

    Example:
        >>> config = DomainRandomizationConfig(enabled=True, seed=42)
        >>> randomizer = DomainRandomizer(config)
        >>> randomizer.apply(model)  # Randomize physics
    """

    def __init__(self, config: DomainRandomizationConfig | None = None) -> None:
        """Initialize domain randomizer."""
        self.config = config or DomainRandomizationConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._original_values: dict[str, Any] = {}
        self._log = logger.bind(component="domain_randomizer")

    def apply(self, model: Any) -> None:
        """Apply domain randomization to model.

        Args:
            model: MuJoCo model to randomize
        """
        if not self.config.enabled or not MUJOCO_AVAILABLE:
            return

        self._log.debug("applying_domain_randomization")

        # Store original values if not already stored
        if not self._original_values:
            self._store_original_values(model)

        # Apply randomizations
        self._randomize_masses(model)
        self._randomize_friction(model)
        self._randomize_damping(model)
        self._randomize_actuators(model)

    def reset(self, model: Any) -> None:
        """Reset model to original values.

        Args:
            model: MuJoCo model to reset
        """
        if not self._original_values or not MUJOCO_AVAILABLE:
            return

        self._log.debug("resetting_domain_randomization")

        # Restore original values
        if "body_mass" in self._original_values:
            model.body_mass[:] = self._original_values["body_mass"]
        if "geom_friction" in self._original_values:
            model.geom_friction[:] = self._original_values["geom_friction"]
        if "dof_damping" in self._original_values:
            model.dof_damping[:] = self._original_values["dof_damping"]
        if "actuator_gainprm" in self._original_values:
            model.actuator_gainprm[:] = self._original_values["actuator_gainprm"]

    def _store_original_values(self, model: Any) -> None:
        """Store original model values for later reset."""
        self._original_values = {
            "body_mass": model.body_mass.copy(),
            "geom_friction": model.geom_friction.copy(),
            "dof_damping": model.dof_damping.copy(),
            "actuator_gainprm": model.actuator_gainprm.copy(),
        }

    def _randomize_masses(self, model: Any) -> None:
        """Randomize body masses."""
        scale = self._rng.uniform(*self.config.mass_range, size=model.nbody)
        model.body_mass[:] = self._original_values["body_mass"] * scale

    def _randomize_friction(self, model: Any) -> None:
        """Randomize geom friction coefficients."""
        scale = self._rng.uniform(*self.config.friction_range, size=model.ngeom)
        # Only scale the first column (sliding friction)
        model.geom_friction[:, 0] = self._original_values["geom_friction"][:, 0] * scale

    def _randomize_damping(self, model: Any) -> None:
        """Randomize joint damping."""
        scale = self._rng.uniform(*self.config.damping_range, size=model.nv)
        model.dof_damping[:] = self._original_values["dof_damping"] * scale

    def _randomize_actuators(self, model: Any) -> None:
        """Randomize actuator strengths."""
        scale = self._rng.uniform(*self.config.actuator_strength_range, size=model.nu)
        # Scale the actuator gain parameter
        model.actuator_gainprm[:, 0] = (
            self._original_values["actuator_gainprm"][:, 0] * scale
        )

    def add_sensor_noise(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Add noise and bias to sensor data.

        Args:
            data: Clean sensor data

        Returns:
            Noisy sensor data
        """
        if not self.config.enabled:
            return data

        noise = self._rng.normal(0, self.config.sensor_noise_std, data.shape)
        bias = self._rng.uniform(*self.config.sensor_bias_range, data.shape)
        return data + noise + bias

    def add_joint_noise(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Add noise to joint positions.

        Args:
            positions: Joint positions (radians)

        Returns:
            Noisy joint positions
        """
        if not self.config.enabled:
            return positions

        noise = self._rng.normal(0, self.config.joint_noise_std, positions.shape)
        return positions + noise


class VisualRandomizer:
    """
    Applies visual domain randomization for robust perception.

    Example:
        >>> config = VisualRandomizationConfig(enabled=True)
        >>> randomizer = VisualRandomizer(config)
        >>> randomizer.apply(model)
    """

    def __init__(self, config: VisualRandomizationConfig | None = None) -> None:
        """Initialize visual randomizer."""
        self.config = config or VisualRandomizationConfig()
        self._rng = np.random.default_rng()
        self._original_values: dict[str, Any] = {}
        self._log = logger.bind(component="visual_randomizer")

    def apply(self, model: Any) -> None:
        """Apply visual randomization to model.

        Args:
            model: MuJoCo model to randomize
        """
        if not self.config.enabled or not MUJOCO_AVAILABLE:
            return

        self._log.debug("applying_visual_randomization")

        # Randomize lighting
        self._randomize_lighting(model)

    def reset(self, model: Any) -> None:
        """Reset model to original visual values.

        Args:
            model: MuJoCo model to reset
        """
        if not self._original_values or not MUJOCO_AVAILABLE:
            return

        self._log.debug("resetting_visual_randomization")

        if "light_diffuse" in self._original_values:
            model.light_diffuse[:] = self._original_values["light_diffuse"]

    def _randomize_lighting(self, model: Any) -> None:
        """Randomize light intensities."""
        if not hasattr(model, "light_diffuse"):
            return

        if "light_diffuse" not in self._original_values:
            self._original_values["light_diffuse"] = model.light_diffuse.copy()

        scale = self._rng.uniform(*self.config.lighting_range, size=model.nlight)
        for i in range(model.nlight):
            model.light_diffuse[i] = (
                self._original_values["light_diffuse"][i] * scale[i]
            )

    def randomize_camera(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomize camera pose.

        Args:
            position: Camera position
            orientation: Camera orientation (quaternion)

        Returns:
            Tuple of (randomized_position, randomized_orientation)
        """
        if not self.config.enabled:
            return position, orientation

        pos_noise = self._rng.normal(0, self.config.camera_position_noise, 3)
        new_position = position + pos_noise

        # Add small rotation noise via axis-angle
        angle = self._rng.normal(0, self.config.camera_orientation_noise)
        axis = self._rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)

        # Convert axis-angle to quaternion and multiply
        half_angle = angle / 2
        noise_quat = np.array(
            [
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
            ]
        )

        # Quaternion multiplication (simplified)
        new_orientation = orientation  # TODO: proper quaternion multiply

        return new_position, new_orientation
