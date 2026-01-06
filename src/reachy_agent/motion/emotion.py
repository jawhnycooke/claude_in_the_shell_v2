"""Emotion playback from library.

This module provides EmotionPlayback for playing back pre-recorded
emotion animations. It implements the MotionSource protocol as a
PRIMARY source that outputs absolute positions from animation frames.

Features:
    - Frame-by-frame playback from dictionary data
    - Configurable FPS setting
    - Automatic completion detection
    - Support for head pose and antenna animations
"""

from typing import Any

from reachy_agent.motion.controller import (
    AntennaState,
    MotionOutput,
    MotionSourceType,
)
from reachy_agent.robot.client import HeadPose


class EmotionPlayback:
    """
    Play recorded emotion from library.

    Implements the MotionSource protocol as a PRIMARY source.
    Plays back animation frames sequentially, outputting head
    and antenna positions for each tick.

    Attributes:
        name: Source identifier ("emotion")
        source_type: PRIMARY (mutually exclusive with other PRIMARY sources)
        is_active: Whether playback is in progress

    Examples:
        >>> emotion_data = {
        ...     "fps": 30,
        ...     "frames": [
        ...         {"pitch": 10, "yaw": 5, "antenna_left": 20},
        ...         {"pitch": 15, "yaw": 10, "antenna_left": 30},
        ...     ]
        ... }
        >>> emotion = EmotionPlayback(emotion_data)
        >>> await emotion.start()
        >>> output = emotion.tick()  # First frame
        >>> print(output.head.pitch)  # 10
    """

    def __init__(self, emotion_data: dict[str, Any]):
        """
        Initialize emotion playback.

        Args:
            emotion_data: Emotion animation data with frames.
                Expected keys:
                - "frames": List of frame dicts with pose values
                - "fps": Frames per second (default 30)
        """
        self.name = "emotion"
        self.source_type = MotionSourceType.PRIMARY
        self._frames: list[dict[str, Any]] = emotion_data.get("frames", [])
        self._fps = emotion_data.get("fps", 30)
        self._active = False
        self._frame_index = 0
        self._current_frame: dict[str, Any] = {}

    @property
    def is_active(self) -> bool:
        """Check if still playing."""
        return self._active and self._frame_index < len(self._frames)

    async def start(self) -> None:
        """Start playback from the beginning."""
        self._active = True
        self._frame_index = 0
        self._current_frame = {}

    async def stop(self) -> None:
        """Stop playback."""
        self._active = False
        self._current_frame = {}

    def tick(self) -> MotionOutput | None:
        """
        Get next frame.

        Advances the frame index and returns the current frame's
        motion output. When all frames have been played, returns
        None and sets is_active to False.

        Returns:
            Current frame as motion output, or None if complete
        """
        if not self.is_active:
            return None

        frame = self._frames[self._frame_index]
        self._current_frame = frame
        self._frame_index += 1

        return MotionOutput(
            head=HeadPose(
                pitch=frame.get("pitch", 0),
                yaw=frame.get("yaw", 0),
                roll=frame.get("roll", 0),
                z=frame.get("z", 0),
            ),
            antennas=AntennaState(
                left=frame.get("antenna_left", 0),
                right=frame.get("antenna_right", 0),
            ),
        )

    def get_positions(self) -> dict[str, float]:
        """
        Get current joint positions.

        PRIMARY sources return absolute positions from the
        current animation frame.

        Returns:
            Dictionary mapping joint names to positions
        """
        if not self._current_frame:
            return {"pitch": 0, "yaw": 0, "roll": 0, "z": 0}

        return {
            "pitch": self._current_frame.get("pitch", 0),
            "yaw": self._current_frame.get("yaw", 0),
            "roll": self._current_frame.get("roll", 0),
            "z": self._current_frame.get("z", 0),
        }

    def get_deltas(self) -> dict[str, float]:
        """
        Get current position deltas (not used for PRIMARY sources).

        PRIMARY sources use absolute positions, so this returns
        an empty dictionary.

        Returns:
            Empty dictionary
        """
        return {}
