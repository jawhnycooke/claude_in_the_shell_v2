"""MJCF model loading and validation utilities.

This module provides utilities for loading and validating MuJoCo XML models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore
    MUJOCO_AVAILABLE = False

logger = structlog.get_logger()

# Expected configuration for Reachy Mini
EXPECTED_JOINTS = [
    "body_rotation",
    "head_z",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "antenna_left",
    "antenna_right",
]

EXPECTED_ACTUATORS = [
    "body_rotation_actuator",
    "head_z_actuator",
    "head_yaw_actuator",
    "head_pitch_actuator",
    "head_roll_actuator",
    "antenna_left_actuator",
    "antenna_right_actuator",
]

EXPECTED_SENSORS = [
    "body_rotation_pos",
    "head_z_pos",
    "head_yaw_pos",
    "head_pitch_pos",
    "head_roll_pos",
    "antenna_left_pos",
    "antenna_right_pos",
]


@dataclass
class ModelInfo:
    """Information about a loaded MuJoCo model."""

    path: Path
    n_joints: int
    n_actuators: int
    n_sensors: int
    n_bodies: int
    joint_names: list[str]
    actuator_names: list[str]
    sensor_names: list[str]
    timestep: float


@dataclass
class ValidationResult:
    """Result of model validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]


class MJCFModelLoader:
    """
    Utility for loading and validating MJCF models.

    Example:
        >>> loader = MJCFModelLoader()
        >>> model, data = loader.load_model("path/to/model.xml")
        >>> result = loader.validate_model(model)
        >>> if not result.valid:
        ...     print(result.errors)
    """

    def __init__(self) -> None:
        """Initialize model loader."""
        self._log = logger.bind(component="model_loader")

    def load_model(self, path: str | Path) -> tuple[Any, Any]:
        """Load MJCF model from file.

        Args:
            path: Path to MJCF XML file

        Returns:
            Tuple of (MjModel, MjData)

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If MuJoCo is not available
            ValueError: If model fails to load
        """
        if not MUJOCO_AVAILABLE:
            raise RuntimeError(
                "MuJoCo is not installed. Install with: pip install mujoco"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self._log.info("loading_model", path=str(path))

        try:
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            self._log.info(
                "model_loaded",
                n_joints=model.njnt,
                n_actuators=model.nu,
                n_sensors=model.nsensor,
            )
            return model, data
        except Exception as e:
            self._log.error("model_load_failed", error=str(e))
            raise ValueError(f"Failed to load model: {e}") from e

    def get_model_info(self, model: Any) -> ModelInfo:
        """Get information about a loaded model.

        Args:
            model: MuJoCo model

        Returns:
            ModelInfo with model details
        """
        if not MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo is not installed")

        joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
        ]
        actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        ]
        sensor_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            for i in range(model.nsensor)
        ]

        return ModelInfo(
            path=Path("unknown"),
            n_joints=model.njnt,
            n_actuators=model.nu,
            n_sensors=model.nsensor,
            n_bodies=model.nbody,
            joint_names=[n for n in joint_names if n],
            actuator_names=[n for n in actuator_names if n],
            sensor_names=[n for n in sensor_names if n],
            timestep=model.opt.timestep,
        )

    def validate_model(
        self,
        model: Any,
        strict: bool = False,
    ) -> ValidationResult:
        """Validate model against expected configuration.

        Args:
            model: MuJoCo model to validate
            strict: If True, missing optional elements are errors

        Returns:
            ValidationResult with validation status
        """
        if not MUJOCO_AVAILABLE:
            return ValidationResult(
                valid=False,
                errors=["MuJoCo is not installed"],
                warnings=[],
            )

        errors: list[str] = []
        warnings: list[str] = []

        info = self.get_model_info(model)

        # Check joints
        for joint in EXPECTED_JOINTS:
            if joint not in info.joint_names:
                errors.append(f"Missing required joint: {joint}")

        # Check actuators
        for actuator in EXPECTED_ACTUATORS:
            if actuator not in info.actuator_names:
                errors.append(f"Missing required actuator: {actuator}")

        # Check sensors
        for sensor in EXPECTED_SENSORS:
            if sensor not in info.sensor_names:
                if strict:
                    errors.append(f"Missing required sensor: {sensor}")
                else:
                    warnings.append(f"Missing sensor: {sensor}")

        # Check for extra joints (not necessarily an error)
        extra_joints = set(info.joint_names) - set(EXPECTED_JOINTS)
        if extra_joints:
            warnings.append(f"Extra joints found: {extra_joints}")

        # Validate timestep
        if model.opt.timestep > 0.01:
            warnings.append(
                f"Large timestep ({model.opt.timestep}s) may cause instability"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def get_joint_limits(self, model: Any) -> dict[str, tuple[float, float]]:
        """Get joint limits from model.

        Args:
            model: MuJoCo model

        Returns:
            Dict mapping joint names to (min, max) limits in degrees
        """
        if not MUJOCO_AVAILABLE:
            return {}

        import numpy as np

        limits = {}
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and model.jnt_limited[i]:
                # Convert from radians to degrees for hinge joints
                jnt_type = model.jnt_type[i]
                if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                    limits[name] = (
                        float(np.degrees(model.jnt_range[i, 0])),
                        float(np.degrees(model.jnt_range[i, 1])),
                    )
                elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    limits[name] = (
                        float(model.jnt_range[i, 0]),
                        float(model.jnt_range[i, 1]),
                    )

        return limits

    def get_actuator_limits(self, model: Any) -> dict[str, tuple[float, float]]:
        """Get actuator control limits from model.

        Args:
            model: MuJoCo model

        Returns:
            Dict mapping actuator names to (min, max) control limits
        """
        if not MUJOCO_AVAILABLE:
            return {}

        limits = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and model.actuator_ctrllimited[i]:
                limits[name] = (
                    float(model.actuator_ctrlrange[i, 0]),
                    float(model.actuator_ctrlrange[i, 1]),
                )

        return limits


def load_reachy_mini_model() -> tuple[Any, Any]:
    """Load the default Reachy Mini model.

    Returns:
        Tuple of (MjModel, MjData)
    """
    from reachy_agent.simulation.environment import DEFAULT_MODEL_PATH

    loader = MJCFModelLoader()
    return loader.load_model(DEFAULT_MODEL_PATH)


def validate_reachy_mini_model(model: Any) -> bool:
    """Validate that a model is a valid Reachy Mini model.

    Args:
        model: MuJoCo model to validate

    Returns:
        True if valid, False otherwise
    """
    loader = MJCFModelLoader()
    result = loader.validate_model(model)
    return result.valid
