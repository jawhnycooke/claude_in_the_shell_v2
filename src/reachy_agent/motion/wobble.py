"""Speech wobble overlay."""

import math
from typing import Optional

from reachy_agent.motion.controller import (
    MotionOutput,
    MotionSourceType,
    PoseOffset,
)


class SpeechWobble:
    """
    Audio-reactive head wobble during speech.

    Adds subtle pitch/roll oscillation modulated by audio level.

    TODO: Complete implementation
    - Audio level tracking
    - Oscillation based on speech
    - Configurable intensity and frequency
    """

    def __init__(self, intensity: float = 1.0, frequency: float = 4.0):
        """
        Initialize speech wobble.

        Args:
            intensity: Wobble intensity (0-1)
            frequency: Oscillation frequency in Hz
        """
        self.name = "wobble"
        self.source_type = MotionSourceType.OVERLAY
        self._intensity = intensity
        self._frequency = frequency
        self._active = False
        self._audio_level = 0.0
        self._phase = 0.0

    @property
    def is_active(self) -> bool:
        """Check if wobble is active."""
        return self._active

    async def start(self) -> None:
        """Start wobble."""
        self._active = True
        self._phase = 0.0

    async def stop(self) -> None:
        """Stop wobble."""
        self._active = False

    def set_audio_level(self, level: float) -> None:
        """
        Set current audio level for reactivity.

        Args:
            level: Audio level 0-1
        """
        self._audio_level = max(0, min(1, level))

    def tick(self) -> Optional[MotionOutput]:
        """
        Generate wobble offset for this tick.

        Returns:
            Pose offset to add to primary motion
        """
        if not self._active:
            return None

        self._phase += self._frequency / 30.0  # 30Hz
        modulation = self._audio_level * self._intensity

        # Subtle pitch/roll oscillation
        pitch = math.sin(self._phase * 2 * math.pi) * 3 * modulation
        roll = math.cos(self._phase * 2 * math.pi * 1.3) * 2 * modulation

        return MotionOutput(head=PoseOffset(pitch=pitch, roll=roll))
