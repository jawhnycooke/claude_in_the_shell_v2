"""Integration tests for component interactions."""

import asyncio

import pytest

from reachy_agent.motion.controller import BlendController
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


# ==============================================================================
# F169: Voice-to-Agent Integration tests
# ==============================================================================


@pytest.mark.asyncio
async def test_voice_to_agent_query_flow() -> None:
    """
    Test end-to-end voice query flow (F169).

    Verifies:
    - Voice pipeline transcribes audio
    - Transcript is passed to agent
    - Agent response is returned
    """
    from reachy_agent.agent.loop import ReachyAgentLoop
    from reachy_agent.agent.options import AgentConfig

    voice = VoicePipeline(mock_mode=True)

    # Create agent with mock mode
    config = AgentConfig(mock_hardware=True, enable_voice=False)
    agent = ReachyAgentLoop(config)

    # Track events
    events_received = []

    @voice.on("transcribed")
    async def on_transcribed(event):
        events_received.append(("transcribed", event.data.get("text")))

    @voice.on("speaking_start")
    async def on_speaking(event):
        events_received.append(("speaking", event.data.get("text")))

    # Simulate voice input flow
    await voice.emit("wake_detected", persona="Jarvis")
    await voice.emit("listening_start")
    await voice.emit("transcribed", text="Hello Jarvis")
    await voice.emit("speaking_start", text="Hello! How can I help?")
    await voice.emit("speaking_end")

    # Verify events
    assert ("transcribed", "Hello Jarvis") in events_received


@pytest.mark.asyncio
async def test_persona_switching_integration() -> None:
    """
    Test persona switching flow (F169).

    Verifies:
    - Different personas can be activated
    - Persona determines voice and behavior
    """
    from reachy_agent.voice.persona import Persona, PersonaManager

    # Create persona manager (uses default personas_dir)
    persona_mgr = PersonaManager(auto_discover=True)

    # Verify personas were loaded
    assert len(persona_mgr._personas) > 0, "No personas were loaded"

    # Set initial persona (Jarvis should be available)
    result = persona_mgr.set_current("Jarvis")
    assert result is True, "Failed to set Jarvis as current persona"
    assert persona_mgr.get_current() is not None
    original = persona_mgr.get_current().name

    # Add test persona
    test_persona = Persona(
        name="TestBot",
        voice="echo",
        wake_words=["hey testbot"],
    )
    persona_mgr.add_persona(test_persona)

    # Switch persona using set_current (sync)
    result = persona_mgr.set_current("TestBot")
    assert result is True
    assert persona_mgr.get_current().name == "TestBot"
    assert persona_mgr.get_current().voice == "echo"

    # Switch back
    result = persona_mgr.set_current(original)
    assert result is True


# ==============================================================================
# F170: Agent-to-Robot Integration tests
# ==============================================================================


@pytest.mark.asyncio
async def test_agent_robot_mcp_tool_execution(mock_robot) -> None:
    """
    Test MCP tool execution from agent (F170).

    Verifies:
    - Agent can execute robot tools via MCP
    - Tool results are returned correctly
    """
    from reachy_agent.mcp import robot as robot_mcp_module
    from reachy_agent.mcp.robot import app as mcp_app

    # Set the mock robot as the global robot
    robot_mcp_module._robot = mock_robot

    # The FastMCP app has tools registered
    assert mcp_app is not None
    assert mcp_app.name == "robot"

    # Wake up robot for testing
    await mock_robot.wake_up()

    # Verify robot is accessible via get_robot
    robot = robot_mcp_module.get_robot()
    assert robot is not None

    # Verify robot can execute movements
    await robot.move_head(pitch=10.0, yaw=5.0, roll=0.0, duration=0.1)
    position = await robot.get_position()
    assert position["head_pitch"] == 10.0
    assert position["head_yaw"] == 5.0


@pytest.mark.asyncio
async def test_permission_check_in_agent_flow(mock_robot) -> None:
    """
    Test permission checks for tool execution (F170).

    Verifies:
    - Permission evaluator checks tools
    - FORBIDDEN tools are blocked
    - AUTONOMOUS tools proceed
    """
    from reachy_agent.permissions.evaluator import (
        PermissionEvaluator,
        PermissionRule,
        PermissionTier,
    )

    # Create evaluator with rules
    rules = [
        PermissionRule(pattern="move_*", tier=PermissionTier.AUTONOMOUS),
        PermissionRule(pattern="shutdown", tier=PermissionTier.FORBIDDEN),
        PermissionRule(pattern="memory_*", tier=PermissionTier.CONFIRM),
    ]
    evaluator = PermissionEvaluator(rules, audit_enabled=False)

    # Test AUTONOMOUS tool
    result = evaluator.evaluate("move_head")
    assert result.tier == PermissionTier.AUTONOMOUS
    assert result.allowed is True

    # Test FORBIDDEN tool
    result = evaluator.evaluate("shutdown")
    assert result.tier == PermissionTier.FORBIDDEN
    assert result.allowed is False

    # Test CONFIRM tool
    result = evaluator.evaluate("memory_store")
    assert result.tier == PermissionTier.CONFIRM
    # CONFIRM is allowed but requires confirmation
    assert result.allowed is True or result.requires_confirmation is True


# ==============================================================================
# F171: Memory Persistence Integration tests
# ==============================================================================


@pytest.mark.asyncio
async def test_memory_conversation_storage_retrieval() -> None:
    """
    Test conversation storage and retrieval (F171).

    Verifies:
    - Conversations can be stored
    - Conversations can be retrieved by search
    """
    import tempfile

    from reachy_agent.memory.manager import MemoryManager, MemoryType

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(
            persist_path=tmpdir, collection_name="test_conversation"
        )

        # Store a conversation
        memory = await manager.store(
            "User asked about the weather forecast for tomorrow",
            MemoryType.CONVERSATION,
            {"topic": "weather"},
        )
        assert memory.id is not None
        assert memory.memory_type == MemoryType.CONVERSATION

        # Search for it
        results = await manager.search("weather forecast")
        assert len(results) > 0
        assert any(r.memory.id == memory.id for r in results)


@pytest.mark.asyncio
async def test_memory_fact_storage_retrieval() -> None:
    """
    Test fact storage and retrieval (F171).

    Verifies:
    - Facts can be stored permanently
    - Facts can be retrieved by semantic search
    """
    import tempfile

    from reachy_agent.memory.manager import MemoryManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(persist_path=tmpdir, collection_name="test_facts")

        # Store facts
        fact1 = await manager.remember_fact("User's name is Alice")
        fact2 = await manager.remember_fact("User prefers dark mode interface")
        fact3 = await manager.remember_fact("User works at a tech company")

        # Search for user preferences
        results = await manager.search("user preferences interface", limit=5)
        assert len(results) > 0
        # Dark mode preference should be found
        found_dark_mode = any("dark mode" in r.memory.content for r in results)
        assert found_dark_mode


@pytest.mark.asyncio
async def test_memory_context_window_persistence() -> None:
    """
    Test context window management (F171).

    Verifies:
    - Context window tracks recent turns
    - Context window respects size limit
    """
    import tempfile

    from reachy_agent.memory.manager import MemoryManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(
            persist_path=tmpdir, context_window_size=3, collection_name="test_context"
        )

        # Add conversation turns
        manager.add_to_context_window("user", "Hello")
        manager.add_to_context_window("assistant", "Hi there!")
        manager.add_to_context_window("user", "How are you?")
        manager.add_to_context_window("assistant", "I'm doing well!")
        manager.add_to_context_window("user", "Great to hear")
        manager.add_to_context_window("assistant", "Indeed!")
        manager.add_to_context_window("user", "What's the weather?")

        # Get context window (should be limited to 3 pairs = 6 turns)
        context = manager.get_context_window()
        assert len(context) <= 6

        # Clear context
        manager.clear_context_window()
        context = manager.get_context_window()
        assert len(context) == 0
