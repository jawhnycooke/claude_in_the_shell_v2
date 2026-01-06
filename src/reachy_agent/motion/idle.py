"""Idle look-around behavior."""

import random
import time
from typing import Optional

from reachy_agent.motion.controller import (
    AntennaState,
    HeadPose,
    MotionOutput,
    MotionSourceType,
)


class IdleBehavior:
    """
    Idle look-around behavior using Perlin noise.

    Creates organic, subtle head movements that make the robot feel alive.

    TODO: Implement using Perlin noise
    - Install noise library
    - Implement tick() with organic movement
    - Antenna drift
    """

    def __init__(
        self,
        speed: float = 0.1,
        amplitude: float = 0.3,
        antenna_drift: float = 0.2,
    ):
        """
        Initialize idle behavior.

        Args:
            speed: Movement speed (0-1)
            amplitude: Movement range as fraction of limits
            antenna_drift: Antenna variation (0-1)
        """
        self.name = "idle"
        self.source_type = MotionSourceType.PRIMARY
        self._speed = speed
        self._amplitude = amplitude
        self._antenna_drift = antenna_drift
        self._active = False
        self._time = 0.0
        self._seed = random.random() * 1000

    @property
    def is_active(self) -> bool:
        """Check if behavior is active."""
        return self._active

    async def start(self) -> None:
        """Start idle behavior."""
        self._active = True
        self._time = 0.0

    async def stop(self) -> None:
        """Stop idle behavior."""
        self._active = False

    def tick(self) -> Optional[MotionOutput]:
        """
        Generate motion for this tick.

        Returns:
            Motion output with head pose and antenna state

        TODO: Implement Perlin noise motion
        - Use noise.pnoise1() for each axis
        - Different seeds for organic movement
        - Scale by amplitude
        """
        if not self._active:
            return None

        self._time += 1.0 / 30.0  # 30Hz

        # TODO: Replace with actual Perlin noise
        # For now, return neutral pose
        return MotionOutput(
            head=HeadPose(0, 0, 0, 0), antennas=AntennaState(0, 0)
        )
