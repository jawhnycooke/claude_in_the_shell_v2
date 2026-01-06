"""Physics utilities for MuJoCo simulation.

This module provides physics-related utilities including:
- PD controllers for joint position control
- Trajectory interpolation
- Contact detection
- Force/torque sensing
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None  # type: ignore


@dataclass
class PDController:
    """Proportional-Derivative controller for joint position control.

    Args:
        kp: Proportional gain
        kd: Derivative gain
        max_torque: Maximum output torque
    """

    kp: float = 100.0
    kd: float = 10.0
    max_torque: float = 50.0

    def compute(
        self,
        target: float,
        current: float,
        velocity: float,
    ) -> float:
        """Compute control torque.

        Args:
            target: Target position
            current: Current position
            velocity: Current velocity

        Returns:
            Control torque (clamped to max_torque)
        """
        error = target - current
        torque = self.kp * error - self.kd * velocity
        return float(np.clip(torque, -self.max_torque, self.max_torque))


@dataclass
class TrajectoryInterpolator:
    """Interpolates between waypoints for smooth motion.

    Supports linear, cubic spline, and minimum-jerk interpolation.
    """

    method: str = "minimum_jerk"  # "linear", "cubic", "minimum_jerk"

    def interpolate(
        self,
        start: float,
        end: float,
        t: float,
        duration: float,
    ) -> float:
        """Interpolate position at time t.

        Args:
            start: Start position
            end: End position
            t: Current time (0 to duration)
            duration: Total duration

        Returns:
            Interpolated position
        """
        if duration <= 0:
            return end

        s = min(1.0, max(0.0, t / duration))  # Normalized time [0, 1]

        if self.method == "linear":
            return start + (end - start) * s

        elif self.method == "cubic":
            # Cubic ease in-out
            if s < 0.5:
                return start + (end - start) * (4 * s * s * s)
            else:
                return start + (end - start) * (1 - pow(-2 * s + 2, 3) / 2)

        else:  # minimum_jerk
            # Minimum jerk trajectory (5th order polynomial)
            # x(s) = 10s³ - 15s⁴ + 6s⁵
            s_mj = 10 * s**3 - 15 * s**4 + 6 * s**5
            return start + (end - start) * s_mj

    def interpolate_velocity(
        self,
        start: float,
        end: float,
        t: float,
        duration: float,
    ) -> float:
        """Get velocity at time t.

        Args:
            start: Start position
            end: End position
            t: Current time
            duration: Total duration

        Returns:
            Velocity at time t
        """
        if duration <= 0:
            return 0.0

        s = min(1.0, max(0.0, t / duration))
        delta = end - start

        if self.method == "linear":
            return delta / duration if 0 < s < 1 else 0.0

        elif self.method == "cubic":
            if s < 0.5:
                ds_dt = 12 * s * s / duration
            else:
                ds_dt = 6 * pow(-2 * s + 2, 2) / duration
            return delta * ds_dt

        else:  # minimum_jerk
            ds_dt = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
            return delta * ds_dt


@dataclass
class ContactInfo:
    """Information about a contact point."""

    body1: str
    body2: str
    position: np.ndarray
    force: np.ndarray
    normal: np.ndarray

    @property
    def force_magnitude(self) -> float:
        """Get magnitude of contact force."""
        return float(np.linalg.norm(self.force))


@dataclass
class PhysicsState:
    """Complete physics state at a timestep."""

    time: float
    joint_positions: dict[str, float] = field(default_factory=dict)
    joint_velocities: dict[str, float] = field(default_factory=dict)
    joint_torques: dict[str, float] = field(default_factory=dict)
    contacts: list[ContactInfo] = field(default_factory=list)
    accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyroscope: np.ndarray = field(default_factory=lambda: np.zeros(3))


def add_sensor_noise(
    value: float | np.ndarray,
    noise_std: float = 0.01,
    bias: float = 0.0,
) -> float | np.ndarray:
    """Add Gaussian noise and bias to sensor reading.

    Args:
        value: Clean sensor value
        noise_std: Standard deviation of noise
        bias: Constant bias to add

    Returns:
        Noisy sensor value
    """
    if isinstance(value, np.ndarray):
        noise = np.random.normal(0, noise_std, value.shape)
        return value + noise + bias
    else:
        noise = np.random.normal(0, noise_std)
        return value + noise + bias


def compute_gravity_compensation(
    model: object,
    data: object,
    joint_ids: list[int],
) -> np.ndarray:
    """Compute gravity compensation torques for joints.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_ids: List of joint IDs to compensate

    Returns:
        Array of compensation torques
    """
    if mujoco is None:
        return np.zeros(len(joint_ids))

    # Use MuJoCo's passive force computation
    qfrc_passive = data.qfrc_passive.copy()  # type: ignore
    return qfrc_passive[joint_ids]


def compute_external_force(
    body_pos: np.ndarray,
    force_pos: np.ndarray,
    force_dir: np.ndarray,
    force_mag: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute force and torque to apply at body origin.

    Args:
        body_pos: Position of body origin
        force_pos: Position where force is applied
        force_dir: Direction of force (unit vector)
        force_mag: Magnitude of force

    Returns:
        Tuple of (force vector, torque vector)
    """
    force = force_dir * force_mag
    moment_arm = force_pos - body_pos
    torque = np.cross(moment_arm, force)
    return force, torque
