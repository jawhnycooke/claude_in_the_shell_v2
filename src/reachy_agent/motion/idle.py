"""Idle look-around behavior.

This module provides the IdleBehavior class for generating organic,
subtle head movements that make the robot feel alive when not actively
engaged in interaction.

Features:
    - Perlin-like noise for organic movement
    - Slow head drift (pitch, yaw variations)
    - Independent antenna wiggle
    - Micro-adjustments for "breathing" feel
"""

import math
import random
from dataclasses import dataclass

from reachy_agent.motion.controller import (
    AntennaState,
    MotionOutput,
    MotionSourceType,
)
from reachy_agent.robot.client import HeadPose


@dataclass
class IdleConfig:
    """Configuration for idle behavior."""

    speed: float = 0.1  # Movement speed (0-1)
    amplitude: float = 0.3  # Movement range as fraction of limits
    antenna_drift: float = 0.2  # Antenna variation (0-1)
    micro_adjust_chance: float = 0.01  # Chance of micro-adjustment per tick


def _simple_noise(t: float, seed: float = 0.0) -> float:
    """
    Generate simple Perlin-like noise using layered sine waves.

    Args:
        t: Time value
        seed: Seed for variation

    Returns:
        Noise value between -1 and 1
    """
    # Layer multiple sine waves at different frequencies
    result = 0.0
    result += math.sin(t * 0.5 + seed) * 0.5
    result += math.sin(t * 1.3 + seed * 2.1) * 0.3
    result += math.sin(t * 2.7 + seed * 0.7) * 0.15
    result += math.sin(t * 5.3 + seed * 1.3) * 0.05

    return max(-1.0, min(1.0, result))


class IdleBehavior:
    """
    Idle look-around behavior using layered sine waves (Perlin-like).

    Creates organic, subtle head movements that make the robot feel alive.
    Implements the MotionSource protocol as a PRIMARY source.

    Attributes:
        name: Source identifier ("idle")
        source_type: PRIMARY (mutually exclusive)
        is_active: Whether behavior is running

    Examples:
        >>> idle = IdleBehavior(speed=0.1, amplitude=0.3)
        >>> await idle.start()
        >>> output = idle.tick()
        >>> # output.head contains subtle position variations
        >>> await idle.stop()
    """

    def __init__(
        self,
        speed: float = 0.1,
        amplitude: float = 0.3,
        antenna_drift: float = 0.2,
        micro_adjust_chance: float = 0.01,
    ):
        """
        Initialize idle behavior.

        Args:
            speed: Movement speed (0-1), controls how fast movements occur
            amplitude: Movement range as fraction of joint limits (0-1)
            antenna_drift: Antenna variation amount (0-1)
            micro_adjust_chance: Probability of micro-adjustment per tick
        """
        self._name = "idle"
        self._source_type = MotionSourceType.PRIMARY
        self._config = IdleConfig(
            speed=speed,
            amplitude=amplitude,
            antenna_drift=antenna_drift,
            micro_adjust_chance=micro_adjust_chance,
        )
        self._active = False
        self._time = 0.0
        self._tick_count = 0

        # Seeds for independent movement of each axis
        self._pitch_seed = random.random() * 100
        self._yaw_seed = random.random() * 100
        self._roll_seed = random.random() * 100
        self._left_antenna_seed = random.random() * 100
        self._right_antenna_seed = random.random() * 100

        # Micro-adjustment state
        self._micro_offset = HeadPose(pitch=0, yaw=0, roll=0, z=0)
        self._micro_decay = 0.95  # How fast micro-adjustments decay

    @property
    def name(self) -> str:
        """Get source name."""
        return self._name

    @property
    def source_type(self) -> MotionSourceType:
        """Get source type (PRIMARY)."""
        return self._source_type

    @property
    def is_active(self) -> bool:
        """Check if behavior is active."""
        return self._active

    @property
    def tick_count(self) -> int:
        """Get number of ticks since start."""
        return self._tick_count

    async def start(self) -> None:
        """Start idle behavior."""
        self._active = True
        self._time = 0.0
        self._tick_count = 0
        # Re-randomize seeds on start for variety
        self._pitch_seed = random.random() * 100
        self._yaw_seed = random.random() * 100
        self._roll_seed = random.random() * 100

    async def stop(self) -> None:
        """Stop idle behavior."""
        self._active = False

    def tick(self) -> MotionOutput | None:
        """
        Generate motion for this tick.

        Returns:
            Motion output with head pose and antenna state, or None if inactive
        """
        if not self._active:
            return None

        self._tick_count += 1
        self._time += (1.0 / 30.0) * self._config.speed  # Time scaled by speed

        # Generate head pose from noise
        head = self._generate_head_pose()

        # Generate antenna state from noise
        antennas = self._generate_antenna_state()

        return MotionOutput(head=head, antennas=antennas)

    def _generate_head_pose(self) -> HeadPose:
        """Generate head pose using layered noise."""
        # Joint limits (from controller)
        pitch_range = 35.0 + 45.0  # -45 to 35
        yaw_range = 60.0 + 60.0  # -60 to 60
        roll_range = 35.0 + 35.0  # -35 to 35

        # Get noise values
        pitch_noise = _simple_noise(self._time, self._pitch_seed)
        yaw_noise = _simple_noise(self._time * 0.7, self._yaw_seed)  # Slower yaw
        roll_noise = _simple_noise(
            self._time * 0.5, self._roll_seed
        )  # Even slower roll

        # Scale by amplitude and range, centered around neutral
        base_pitch = pitch_noise * self._config.amplitude * (pitch_range / 2)
        base_yaw = yaw_noise * self._config.amplitude * (yaw_range / 2)
        base_roll = roll_noise * self._config.amplitude * (roll_range / 4)  # Less roll

        # Apply micro-adjustments
        self._update_micro_adjustments()
        pitch = base_pitch + self._micro_offset.pitch
        yaw = base_yaw + self._micro_offset.yaw
        roll = base_roll + self._micro_offset.roll

        return HeadPose(pitch=pitch, yaw=yaw, roll=roll, z=0)

    def _generate_antenna_state(self) -> AntennaState:
        """Generate independent antenna movements."""
        # Antenna range (typical -90 to 90)
        antenna_range = 90.0

        # Independent noise for each antenna (different speeds)
        left_noise = _simple_noise(self._time * 1.3, self._left_antenna_seed)
        right_noise = _simple_noise(self._time * 1.1, self._right_antenna_seed)

        # Scale by drift setting
        left = left_noise * self._config.antenna_drift * antenna_range
        right = right_noise * self._config.antenna_drift * antenna_range

        return AntennaState(left=left, right=right)

    def _update_micro_adjustments(self) -> None:
        """Apply occasional micro-adjustments for "alive" feel."""
        # Decay existing micro-offset
        self._micro_offset = HeadPose(
            pitch=self._micro_offset.pitch * self._micro_decay,
            yaw=self._micro_offset.yaw * self._micro_decay,
            roll=self._micro_offset.roll * self._micro_decay,
            z=0,
        )

        # Random chance to add new micro-adjustment
        if random.random() < self._config.micro_adjust_chance:
            self._micro_offset = HeadPose(
                pitch=self._micro_offset.pitch + random.uniform(-3, 3),
                yaw=self._micro_offset.yaw + random.uniform(-5, 5),
                roll=self._micro_offset.roll + random.uniform(-2, 2),
                z=0,
            )

    def get_positions(self) -> dict[str, float]:
        """
        Get current joint positions.

        Returns:
            Dictionary mapping joint names to positions
        """
        if not self._active:
            return {"pitch": 0, "yaw": 0, "roll": 0, "z": 0}

        head = self._generate_head_pose()
        return {
            "pitch": head.pitch,
            "yaw": head.yaw,
            "roll": head.roll,
            "z": head.z,
        }

    def get_deltas(self) -> dict[str, float]:
        """
        Get current position deltas (not used for PRIMARY sources).

        Returns:
            Empty dictionary (PRIMARY sources use absolute positions)
        """
        return {}
