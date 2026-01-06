"""Emotion playback from library."""

from typing import Any, Dict, List, Optional

from reachy_agent.motion.controller import (
    AntennaState,
    HeadPose,
    MotionOutput,
    MotionSourceType,
)


class EmotionPlayback:
    """
    Play recorded emotion from library.

    TODO: Complete implementation
    - Load emotion data from data/emotions/
    - Frame-by-frame playback
    - Completion detection
    """

    def __init__(self, emotion_data: Dict[str, Any]):
        """
        Initialize emotion playback.

        Args:
            emotion_data: Emotion animation data with frames
        """
        self.name = "emotion"
        self.source_type = MotionSourceType.PRIMARY
        self._frames: List[Dict] = emotion_data.get("frames", [])
        self._fps = emotion_data.get("fps", 30)
        self._active = False
        self._frame_index = 0

    @property
    def is_active(self) -> bool:
        """Check if still playing."""
        return self._active and self._frame_index < len(self._frames)

    async def start(self) -> None:
        """Start playback."""
        self._active = True
        self._frame_index = 0

    async def stop(self) -> None:
        """Stop playback."""
        self._active = False

    def tick(self) -> Optional[MotionOutput]:
        """
        Get next frame.

        Returns:
            Current frame as motion output
        """
        if not self.is_active:
            return None

        frame = self._frames[self._frame_index]
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
