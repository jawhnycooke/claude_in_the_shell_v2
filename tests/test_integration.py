"""Integration tests for component interactions."""

import asyncio

import pytest

from reachy_agent.motion.controller import BlendController, MotionOutput
from reachy_agent.motion.idle import IdleBehavior
from reachy_agent.motion.wobble import SpeechWobble
from reachy_agent.robot.client import HeadPose
from reachy_agent.voice.pipeline import VoicePipeline


# ==============================================================================
# F126: Voice-Motion Integration tests
# ==============================================================================


class MockVoiceMotionIntegration:
    """Helper to integrate voice pipeline with motion controller."""

    def __init__(
        self,
        voice: VoicePipeline,
        motion: BlendController,
        wobble: SpeechWobble,
    ):
        self.voice = voice
        self.motion = motion
        self.wobble = wobble
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up event handlers for voice-motion coordination."""

        @self.voice.on("speaking_start")
        async def on_speaking_start(event: dict) -> None:
            """Start wobble when TTS begins."""
            await self.motion.add_overlay(self.wobble)
            self.wobble.set_audio_level(0.7)

        @self.voice.on("speaking_end")
        async def on_speaking_end(event: dict) -> None:
            """Stop wobble when TTS ends."""
            await self.motion.remove_overlay("wobble")


@pytest.mark.asyncio
async def test_voice_motion_integration(mock_robot) -> None:
    """
    Test voice-motion integration (F126).

    This test verifies:
    - speaking_start event activates wobble
    - speaking_end event deactivates wobble
    """
    # Create components
    voice = VoicePipeline(mock_mode=True)
    motion = BlendController(mock_robot, mock_mode=True)
    wobble = SpeechWobble(intensity=0.8)
    idle = IdleBehavior()

    # Set up idle as primary
    await motion.set_primary(idle)
    await motion.start()

    # Create integration
    integration = MockVoiceMotionIntegration(voice, motion, wobble)

    # Initially no wobble overlay
    assert "wobble" not in motion.overlay_sources

    # Emit speaking_start event
    await voice.emit("speaking_start", voice="shimmer", text="Hello")
    await asyncio.sleep(0.05)

    # Wobble should now be active
    assert "wobble" in motion.overlay_sources
    assert wobble.is_active

    # Emit speaking_end event
    await voice.emit("speaking_end")
    await asyncio.sleep(0.05)

    # Wobble should be removed
    assert "wobble" not in motion.overlay_sources

    await motion.stop()


@pytest.mark.asyncio
async def test_wobble_audio_level_updates(mock_robot) -> None:
    """Test that wobble intensity can be updated during speech."""
    voice = VoicePipeline(mock_mode=True)
    motion = BlendController(mock_robot, mock_mode=True)
    wobble = SpeechWobble(intensity=1.0)

    await motion.start()
    await motion.add_overlay(wobble)

    # Initial audio level
    wobble.set_audio_level(0.5)
    assert wobble.audio_level == 0.5

    # Update during "speech"
    wobble.set_audio_level(0.9)
    assert wobble.audio_level == 0.9

    wobble.set_audio_level(0.2)
    assert wobble.audio_level == 0.2

    await motion.stop()


@pytest.mark.asyncio
async def test_motion_blending_with_wobble(mock_robot) -> None:
    """Test that idle + wobble blends correctly."""
    motion = BlendController(mock_robot, mock_mode=True)
    idle = IdleBehavior(speed=0.1, amplitude=0.2)
    wobble = SpeechWobble(intensity=1.0, pitch_amplitude=3.0)

    await motion.set_primary(idle)
    await motion.add_overlay(wobble)
    await motion.start()

    # Set audio level
    wobble.set_audio_level(1.0)

    # Let it run
    await asyncio.sleep(0.1)

    # Get blended output
    blended = motion._blend_sources()

    # Should have valid output
    assert blended is not None
    assert isinstance(blended.head, HeadPose)

    await motion.stop()


@pytest.mark.asyncio
async def test_emotion_interrupts_idle(mock_robot) -> None:
    """Test that emotion playback replaces idle."""
    from reachy_agent.motion.emotion import EmotionPlayback

    motion = BlendController(mock_robot, mock_mode=True)
    idle = IdleBehavior()

    emotion_data = {
        "frames": [
            {"pitch": 20, "yaw": 10},
            {"pitch": 25, "yaw": 15},
        ]
    }
    emotion = EmotionPlayback(emotion_data)

    # Start with idle
    await motion.set_primary(idle)
    await motion.start()
    assert motion.primary_source is idle

    # Switch to emotion
    await motion.set_primary(emotion)
    assert motion.primary_source is emotion
    assert not idle.is_active
    assert emotion.is_active

    # Emotion should play its frames
    output = emotion.tick()
    assert output is not None
    assert output.head.pitch == 20

    await motion.stop()


# ==============================================================================
# F127-F128: Agent Options tests (placeholders - implemented in test_agent.py)
# ==============================================================================


@pytest.mark.asyncio
async def test_voice_pipeline_events_trigger_motion() -> None:
    """Test that voice pipeline events can trigger motion changes."""
    voice = VoicePipeline(mock_mode=True)

    events_received = []

    @voice.on("speaking_start")
    async def track_start(event):
        events_received.append("start")

    @voice.on("speaking_end")
    async def track_end(event):
        events_received.append("end")

    # Simulate TTS events
    await voice.emit("speaking_start", text="Hello")
    await voice.emit("speaking_end")

    assert events_received == ["start", "end"]


@pytest.mark.asyncio
async def test_full_voice_motion_cycle(mock_robot) -> None:
    """Test a complete voice-motion interaction cycle."""
    voice = VoicePipeline(mock_mode=True)
    motion = BlendController(mock_robot, mock_mode=True)
    idle = IdleBehavior()
    wobble = SpeechWobble()

    # Set up
    await motion.set_primary(idle)
    await motion.start()

    # Phase 1: Idle only
    assert motion.primary_source is idle
    assert len(motion.overlay_sources) == 0

    # Phase 2: User speaks (listening)
    await voice.emit("listening_start")
    # Motion continues with idle

    # Phase 3: Transcribed
    await voice.emit("transcribed", text="Hello robot")
    # Motion continues with idle

    # Phase 4: Processing
    await voice.emit("processing")
    # Could add thinking animation here

    # Phase 5: Response speaking
    await motion.add_overlay(wobble)
    wobble.set_audio_level(0.7)
    await voice.emit("speaking_start", text="Hi there!")
    assert "wobble" in motion.overlay_sources

    # Let wobble run
    await asyncio.sleep(0.1)

    # Phase 6: Speaking ends
    await voice.emit("speaking_end")
    await motion.remove_overlay("wobble")
    assert "wobble" not in motion.overlay_sources

    # Phase 7: Back to idle
    assert motion.primary_source is idle

    await motion.stop()
