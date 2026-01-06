"""Speech wobble overlay for audio-reactive head motion.

This module provides the SpeechWobble class that adds subtle head
oscillation modulated by audio level during speech. It's used as
an OVERLAY source with the BlendController.

Features:
    - Audio-reactive intensity (louder = more wobble)
    - Configurable oscillation frequency
    - Smooth sine/cosine based motion
    - Implements MotionSource protocol as OVERLAY
"""

import math
from dataclasses import dataclass

from reachy_agent.motion.controller import (
    MotionOutput,
    MotionSourceType,
    PoseOffset,
)


@dataclass
class WobbleConfig:
    """Configuration for speech wobble behavior."""

    intensity: float = 1.0  # Overall wobble strength (0-1)
    frequency: float = 4.0  # Oscillation frequency in Hz
    pitch_amplitude: float = 3.0  # Max pitch offset in degrees
    roll_amplitude: float = 2.0  # Max roll offset in degrees
    yaw_amplitude: float = 1.0  # Max yaw offset in degrees


class SpeechWobble:
    """
    Audio-reactive head wobble during speech.

    Adds subtle pitch/roll/yaw oscillation modulated by audio level.
    Implements the MotionSource protocol as an OVERLAY source.

    The wobble creates natural-looking head motion that suggests
    the robot is "speaking" or reacting to audio output.

    Attributes:
        name: Source identifier ("wobble")
        source_type: OVERLAY (adds to primary motion)
        is_active: Whether wobble is currently running

    Examples:
        >>> wobble = SpeechWobble(intensity=0.8, frequency=5.0)
        >>> await wobble.start()
        >>> wobble.set_audio_level(0.7)  # From audio analysis
        >>> offset = wobble.tick()
        >>> # offset contains head motion deltas to add
        >>> await wobble.stop()
    """

    def __init__(
        self,
        intensity: float = 1.0,
        frequency: float = 4.0,
        pitch_amplitude: float = 3.0,
        roll_amplitude: float = 2.0,
        yaw_amplitude: float = 1.0,
    ):
        """
        Initialize speech wobble.

        Args:
            intensity: Wobble intensity multiplier (0-1)
            frequency: Oscillation frequency in Hz (default 4Hz)
            pitch_amplitude: Maximum pitch offset in degrees
            roll_amplitude: Maximum roll offset in degrees
            yaw_amplitude: Maximum yaw offset in degrees
        """
        self._name = "wobble"
        self._source_type = MotionSourceType.OVERLAY
        self._config = WobbleConfig(
            intensity=intensity,
            frequency=frequency,
            pitch_amplitude=pitch_amplitude,
            roll_amplitude=roll_amplitude,
            yaw_amplitude=yaw_amplitude,
        )
        self._active = False
        self._audio_level = 0.0
        self._phase = 0.0
        self._tick_count = 0

        # Current deltas for protocol compliance
        self._current_pitch = 0.0
        self._current_yaw = 0.0
        self._current_roll = 0.0

    @property
    def name(self) -> str:
        """Get source name."""
        return self._name

    @property
    def source_type(self) -> MotionSourceType:
        """Get source type (OVERLAY)."""
        return self._source_type

    @property
    def is_active(self) -> bool:
        """Check if wobble is active."""
        return self._active

    @property
    def tick_count(self) -> int:
        """Get number of ticks since start."""
        return self._tick_count

    @property
    def audio_level(self) -> float:
        """Get current audio level."""
        return self._audio_level

    @property
    def intensity(self) -> float:
        """Get wobble intensity."""
        return self._config.intensity

    @intensity.setter
    def intensity(self, value: float) -> None:
        """Set wobble intensity (clamped 0-1)."""
        self._config.intensity = max(0.0, min(1.0, value))

    @property
    def frequency(self) -> float:
        """Get oscillation frequency."""
        return self._config.frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        """Set oscillation frequency (Hz)."""
        self._config.frequency = max(0.1, min(20.0, value))

    async def start(self) -> None:
        """Start wobble effect."""
        self._active = True
        self._phase = 0.0
        self._tick_count = 0
        self._audio_level = 0.0

    async def stop(self) -> None:
        """Stop wobble effect."""
        self._active = False
        self._current_pitch = 0.0
        self._current_yaw = 0.0
        self._current_roll = 0.0

    def set_audio_level(self, level: float) -> None:
        """
        Set current audio level for reactivity.

        Higher audio levels create more pronounced wobble.

        Args:
            level: Audio level 0-1 (typically from audio RMS analysis)
        """
        self._audio_level = max(0.0, min(1.0, level))

    def tick(self) -> MotionOutput | None:
        """
        Generate wobble offset for this tick.

        The wobble is a combination of sine waves at different
        phases to create natural-looking oscillation. The amplitude
        is modulated by the current audio level.

        Returns:
            MotionOutput with pose offsets, or None if inactive
        """
        if not self._active:
            return None

        self._tick_count += 1
        self._phase += self._config.frequency / 30.0  # 30Hz tick rate

        # Modulation factor based on audio level and intensity
        modulation = self._audio_level * self._config.intensity

        # Generate oscillating offsets using sine/cosine
        # Different frequencies and phases for natural motion
        phase_rad = self._phase * 2 * math.pi

        self._current_pitch = (
            math.sin(phase_rad) * self._config.pitch_amplitude * modulation
        )
        self._current_roll = (
            math.cos(phase_rad * 1.3) * self._config.roll_amplitude * modulation
        )
        self._current_yaw = (
            math.sin(phase_rad * 0.7 + 0.5) * self._config.yaw_amplitude * modulation
        )

        return MotionOutput(
            head=PoseOffset(
                pitch=self._current_pitch,
                yaw=self._current_yaw,
                roll=self._current_roll,
                z=0.0,
            )
        )

    def get_positions(self) -> dict[str, float]:
        """
        Get current joint positions.

        OVERLAY sources return empty dict for positions
        (they provide deltas, not absolute positions).

        Returns:
            Empty dictionary
        """
        return {}

    def get_deltas(self) -> dict[str, float]:
        """
        Get current position deltas to add to primary motion.

        Returns:
            Dictionary mapping joint names to offset values
        """
        if not self._active:
            return {"pitch": 0, "yaw": 0, "roll": 0, "z": 0}

        return {
            "pitch": self._current_pitch,
            "yaw": self._current_yaw,
            "roll": self._current_roll,
            "z": 0.0,
        }
