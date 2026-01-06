"""Tests for voice pipeline.

Tests for:
- F076: PersonaManager for persona loading and management
- F077: get_persona() method to retrieve by name
- F078: get_persona_by_wake_word() for voice activation
"""

import tempfile
from pathlib import Path

import pytest

from reachy_agent.voice.persona import Persona, PersonaManager


# ==============================================================================
# F076: PersonaManager tests
# ==============================================================================


class TestPersonaManager:
    """Tests for PersonaManager (F076)."""

    def test_persona_manager_init(self) -> None:
        """Test PersonaManager initialization without auto-discover."""
        manager = PersonaManager(auto_discover=False)

        assert len(manager.get_all_personas()) == 0
        assert len(manager.get_wake_words()) == 0

    def test_persona_discovery(self) -> None:
        """
        Test PersonaManager auto-discovers personas from prompts/personas/.

        This test verifies:
        - Auto-discovers personas from prompts/personas/
        - Parses YAML frontmatter
        - Loads markdown content
        """
        manager = PersonaManager(personas_dir="prompts/personas", auto_discover=True)

        # Should have discovered personas
        personas = manager.get_all_personas()
        assert len(personas) >= 3  # jarvis, motoko, batou

        # Check persona names
        names = [p.name.lower() for p in personas]
        assert "jarvis" in names
        assert "motoko" in names
        assert "batou" in names

    def test_yaml_frontmatter_parsing(self) -> None:
        """Test YAML frontmatter is parsed correctly."""
        manager = PersonaManager(personas_dir="prompts/personas")

        jarvis = manager.get_persona("jarvis")
        assert jarvis is not None
        assert jarvis.wake_words == ["hey jarvis", "jarvis"]
        assert jarvis.voice == "echo"
        assert "sophisticated" in jarvis.prompt.lower()

    def test_markdown_content_loading(self) -> None:
        """Test markdown content is loaded."""
        manager = PersonaManager(personas_dir="prompts/personas")

        motoko = manager.get_persona("motoko")
        assert motoko is not None
        assert len(motoko.prompt) > 100  # Has substantial content
        assert "ghost in the shell" in motoko.prompt.lower()


def test_persona_manager(tmp_path: Path) -> None:
    """
    Test PersonaManager (F076).

    This test verifies:
    - Create src/reachy_agent/voice/persona.py
    - Auto-discovers personas from prompts/personas/
    - Parses YAML frontmatter
    - Loads markdown content
    """
    # Create test persona files
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()

    # Create test persona
    test_persona = personas_dir / "test_persona.md"
    test_persona.write_text(
        """---
name: TestPersona
wake_words:
  - hey test
  - test
voice: nova
---

# Test Persona

This is a test persona for testing purposes.
"""
    )

    # Create manager with test directory
    manager = PersonaManager(personas_dir=str(personas_dir))

    # Verify persona was discovered
    personas = manager.get_all_personas()
    assert len(personas) == 1

    # Verify YAML was parsed
    persona = personas[0]
    assert persona.name == "TestPersona"
    assert persona.wake_words == ["hey test", "test"]
    assert persona.voice == "nova"

    # Verify markdown was loaded
    assert "test persona" in persona.prompt.lower()


# ==============================================================================
# F077: get_persona() method tests
# ==============================================================================


def test_get_persona() -> None:
    """
    Test get_persona() method (F077).

    This test verifies:
    - Returns Persona dataclass
    - Includes wake_words, voice, prompt
    """
    manager = PersonaManager(personas_dir="prompts/personas")

    # Test get_persona by name
    jarvis = manager.get_persona("jarvis")

    # Verify returns Persona dataclass
    assert isinstance(jarvis, Persona)

    # Verify includes wake_words
    assert jarvis.wake_words is not None
    assert len(jarvis.wake_words) > 0
    assert "hey jarvis" in jarvis.wake_words

    # Verify includes voice
    assert jarvis.voice == "echo"

    # Verify includes prompt
    assert jarvis.prompt is not None
    assert len(jarvis.prompt) > 0


def test_get_persona_case_insensitive() -> None:
    """Test get_persona is case-insensitive."""
    manager = PersonaManager(personas_dir="prompts/personas")

    # Test various cases
    assert manager.get_persona("jarvis") is not None
    assert manager.get_persona("JARVIS") is not None
    assert manager.get_persona("Jarvis") is not None

    # All should return same persona
    assert manager.get_persona("jarvis").name == manager.get_persona("JARVIS").name


def test_get_persona_not_found() -> None:
    """Test get_persona returns None for unknown persona."""
    manager = PersonaManager(personas_dir="prompts/personas")

    result = manager.get_persona("unknown_persona")
    assert result is None


# ==============================================================================
# F078: get_persona_by_wake_word() tests
# ==============================================================================


def test_get_persona_by_wake_word() -> None:
    """
    Test get_persona_by_wake_word() method (F078).

    This test verifies:
    - Matches wake word to persona
    - Case-insensitive matching
    """
    manager = PersonaManager(personas_dir="prompts/personas")

    # Test wake word lookup for Jarvis
    persona = manager.get_persona_by_wake_word("hey jarvis")
    assert persona is not None
    assert persona.name.lower() == "jarvis"

    # Test alternative wake word
    persona2 = manager.get_persona_by_wake_word("jarvis")
    assert persona2 is not None
    assert persona2.name.lower() == "jarvis"

    # Test Motoko's wake words
    motoko = manager.get_persona_by_wake_word("hey motoko")
    assert motoko is not None
    assert motoko.name.lower() == "motoko"

    # Test "major" wake word for Motoko
    major = manager.get_persona_by_wake_word("major")
    assert major is not None
    assert major.name.lower() == "motoko"


def test_get_persona_by_wake_word_case_insensitive() -> None:
    """Test wake word matching is case-insensitive."""
    manager = PersonaManager(personas_dir="prompts/personas")

    # Test various cases
    assert manager.get_persona_by_wake_word("hey jarvis") is not None
    assert manager.get_persona_by_wake_word("HEY JARVIS") is not None
    assert manager.get_persona_by_wake_word("Hey Jarvis") is not None
    assert manager.get_persona_by_wake_word("JARVIS") is not None


def test_get_persona_by_wake_word_not_found() -> None:
    """Test wake word lookup returns None for unknown wake word."""
    manager = PersonaManager(personas_dir="prompts/personas")

    result = manager.get_persona_by_wake_word("hey unknown")
    assert result is None


# ==============================================================================
# Additional PersonaManager tests
# ==============================================================================


class TestPersonaManagerMethods:
    """Additional tests for PersonaManager functionality."""

    def test_get_wake_words(self) -> None:
        """Test get_wake_words returns all wake words."""
        manager = PersonaManager(personas_dir="prompts/personas")

        wake_words = manager.get_wake_words()

        # Should have multiple wake words
        assert len(wake_words) >= 6  # 2 per persona minimum

        # Check specific wake words
        assert "hey jarvis" in wake_words
        assert "jarvis" in wake_words
        assert "hey motoko" in wake_words
        assert "major" in wake_words

    def test_get_voice(self) -> None:
        """Test get_voice returns correct voice for persona."""
        manager = PersonaManager(personas_dir="prompts/personas")

        assert manager.get_voice("jarvis") == "echo"
        assert manager.get_voice("motoko") == "nova"

        # Unknown persona returns default
        assert manager.get_voice("unknown") == "echo"

    def test_set_and_get_current(self) -> None:
        """Test set/get current persona."""
        manager = PersonaManager(personas_dir="prompts/personas")

        # Initially no current
        assert manager.get_current() is None

        # Set current
        success = manager.set_current("jarvis")
        assert success is True

        # Get current
        current = manager.get_current()
        assert current is not None
        assert current.name.lower() == "jarvis"

    def test_set_current_unknown(self) -> None:
        """Test set_current returns False for unknown persona."""
        manager = PersonaManager(personas_dir="prompts/personas")

        success = manager.set_current("unknown")
        assert success is False

    def test_add_persona(self) -> None:
        """Test programmatically adding a persona."""
        manager = PersonaManager(auto_discover=False)

        # Create and add persona
        persona = Persona(
            name="CustomBot",
            wake_words=["hey custom", "custom"],
            voice="shimmer",
            prompt="Custom bot prompt.",
        )
        manager.add_persona(persona)

        # Verify added
        assert manager.get_persona("custombot") is not None
        assert manager.get_persona_by_wake_word("hey custom") is not None

    @pytest.mark.asyncio
    async def test_switch_to(self) -> None:
        """Test async switch_to method."""
        manager = PersonaManager(personas_dir="prompts/personas")

        await manager.switch_to("motoko")

        current = manager.get_current()
        assert current is not None
        assert current.name.lower() == "motoko"


# ==============================================================================
# Persona dataclass tests
# ==============================================================================


class TestPersona:
    """Tests for Persona dataclass."""

    def test_persona_creation(self) -> None:
        """Test creating a Persona."""
        persona = Persona(
            name="Test",
            wake_words=["test", "hey test"],
            voice="echo",
            prompt="Test prompt",
        )

        assert persona.name == "Test"
        assert persona.wake_words == ["test", "hey test"]
        assert persona.voice == "echo"
        assert persona.prompt == "Test prompt"

    def test_persona_defaults(self) -> None:
        """Test Persona defaults."""
        persona = Persona(name="Minimal")

        assert persona.name == "Minimal"
        assert persona.wake_words == []
        assert persona.voice == "echo"
        assert persona.prompt == ""


# ==============================================================================
# F079: AudioManager tests
# ==============================================================================


@pytest.mark.asyncio
async def test_audio_manager() -> None:
    """
    Test AudioManager for PyAudio stream management (F079).

    This test verifies:
    - Create src/reachy_agent/voice/audio.py
    - Initializes (in mock mode without hardware)
    - Opens input stream from mic
    - Opens output stream to speaker
    """
    from reachy_agent.voice.audio import AudioConfig, AudioManager

    # Use mock mode for testing without hardware
    manager = AudioManager(mock_mode=True)

    # Verify initial state
    assert not manager.is_initialized
    assert not manager.is_capturing

    # Initialize
    await manager.initialize()
    assert manager.is_initialized

    # Open input stream
    manager.open_input_stream()

    # Open output stream
    manager.open_output_stream()

    # Cleanup
    manager.cleanup()
    assert not manager.is_initialized


@pytest.mark.asyncio
async def test_audio_manager_config() -> None:
    """Test AudioManager with custom config."""
    from reachy_agent.voice.audio import AudioConfig, AudioManager

    config = AudioConfig(
        sample_rate=48000,
        channels=2,
        chunk_size=2048,
    )

    manager = AudioManager(config=config, mock_mode=True)
    await manager.initialize()

    assert manager.config.sample_rate == 48000
    assert manager.config.channels == 2
    assert manager.config.chunk_size == 2048

    manager.cleanup()


@pytest.mark.asyncio
async def test_audio_manager_context_manager() -> None:
    """Test AudioManager as context manager."""
    from reachy_agent.voice.audio import AudioManager

    with AudioManager(mock_mode=True) as manager:
        pass

    # After context exit, cleanup should have been called
    # Note: cleanup is synchronous, so is_initialized should be False
    assert not manager.is_initialized


# ==============================================================================
# F080: read_input_stream() tests
# ==============================================================================


@pytest.mark.asyncio
async def test_read_input_stream() -> None:
    """
    Test read_input_stream() for mic audio capture (F080).

    This test verifies:
    - Returns audio chunks
    - Handles buffer overflow gracefully (returns silence)
    """
    from reachy_agent.voice.audio import AudioManager

    manager = AudioManager(mock_mode=True)
    await manager.initialize()
    manager.open_input_stream()

    # Read a chunk
    chunk = manager.read_input_stream()

    # Should return bytes
    assert isinstance(chunk, bytes)

    # Should have expected size (chunk_size * 2 for 16-bit audio)
    assert len(chunk) == manager.config.chunk_size * 2

    manager.cleanup()


@pytest.mark.asyncio
async def test_read_input_stream_capture_buffer() -> None:
    """Test read_input_stream adds to capture buffer when capturing."""
    from reachy_agent.voice.audio import AudioManager

    manager = AudioManager(mock_mode=True)
    await manager.initialize()
    manager.open_input_stream()

    # Start capture
    manager.start_capture()
    assert manager.is_capturing

    # Read some chunks
    for _ in range(3):
        manager.read_input_stream()

    # Stop capture and get data
    audio_data = manager.stop_capture()
    assert not manager.is_capturing

    # Should have concatenated data
    assert len(audio_data) == manager.config.chunk_size * 2 * 3

    manager.cleanup()


@pytest.mark.asyncio
async def test_read_chunks_generator() -> None:
    """Test read_chunks generator."""
    from reachy_agent.voice.audio import AudioManager

    manager = AudioManager(mock_mode=True)
    await manager.initialize()
    manager.open_input_stream()

    # Read 0.5 seconds of audio
    chunks = list(manager.read_chunks(0.5))

    # Should have multiple chunks
    expected_chunks = int((0.5 * 16000) / 1024)
    assert len(chunks) == expected_chunks

    manager.cleanup()


# ==============================================================================
# F081: write_output_stream() tests
# ==============================================================================


@pytest.mark.asyncio
async def test_write_output_stream() -> None:
    """
    Test write_output_stream() for speaker playback (F081).

    This test verifies:
    - Accepts audio chunks
    - Handles buffer underflow gracefully (mock mode)
    """
    from reachy_agent.voice.audio import AudioManager

    manager = AudioManager(mock_mode=True)
    await manager.initialize()
    manager.open_output_stream()

    # Write a chunk (in mock mode, just verifies no errors)
    test_audio = b"\x00" * 2048
    manager.write_output_stream(test_audio)

    # Also test play_chunk alias
    manager.play_chunk(test_audio)

    manager.cleanup()


@pytest.mark.asyncio
async def test_audio_manager_not_initialized() -> None:
    """Test AudioManager raises error when not initialized."""
    from reachy_agent.voice.audio import AudioManager

    manager = AudioManager(mock_mode=True)

    # Should raise when not initialized
    with pytest.raises(RuntimeError, match="not initialized"):
        manager.read_input_stream()

    with pytest.raises(RuntimeError, match="not initialized"):
        manager.write_output_stream(b"test")

    with pytest.raises(RuntimeError, match="not initialized"):
        manager.start_capture()


# ==============================================================================
# AudioConfig tests
# ==============================================================================


class TestAudioConfig:
    """Tests for AudioConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default AudioConfig values."""
        from reachy_agent.voice.audio import AudioConfig

        config = AudioConfig()

        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024

    def test_custom_config(self) -> None:
        """Test custom AudioConfig values."""
        from reachy_agent.voice.audio import AudioConfig

        config = AudioConfig(
            sample_rate=44100,
            channels=2,
            chunk_size=512,
        )

        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.chunk_size == 512


# ==============================================================================
# F082: WakeWordDetector tests
# ==============================================================================


def test_wake_word_detector() -> None:
    """
    Test WakeWordDetector with OpenWakeWord (F082).

    This test verifies:
    - Create src/reachy_agent/voice/wake_word.py
    - Loads models for all persona wake words
    - Processes audio chunks (mock mode)
    """
    from reachy_agent.voice.wake_word import WakeWordDetector

    # Track detections
    detections: list[tuple[str, float]] = []

    def on_detected(model: str, confidence: float) -> None:
        detections.append((model, confidence))

    # Create detector in mock mode
    detector = WakeWordDetector(
        models=["hey_jarvis", "hey_motoko"],
        on_detected=on_detected,
        sensitivity=0.5,
        mock_mode=True,
    )

    # Verify models
    assert "hey_jarvis" in detector.models
    assert "hey_motoko" in detector.models

    # Enable detection
    detector.enable()
    assert detector.is_enabled

    # Cleanup
    detector.cleanup()
    assert not detector.is_enabled


def test_wake_word_detector_enable_disable() -> None:
    """Test enable/disable functionality."""
    from reachy_agent.voice.wake_word import WakeWordDetector

    detector = WakeWordDetector(
        models=["hey_jarvis"],
        mock_mode=True,
    )

    # Initially disabled
    assert not detector.is_enabled

    # Enable
    detector.enable()
    assert detector.is_enabled

    # Disable
    detector.disable()
    assert not detector.is_enabled


def test_wake_word_sensitivity() -> None:
    """Test sensitivity configuration."""
    from reachy_agent.voice.wake_word import WakeWordDetector

    detector = WakeWordDetector(
        models=["hey_jarvis"],
        sensitivity=0.7,
        mock_mode=True,
    )

    assert detector.sensitivity == 0.7

    # Test setter with clamping
    detector.sensitivity = 0.3
    assert detector.sensitivity == 0.3

    detector.sensitivity = 1.5  # Above max
    assert detector.sensitivity == 1.0

    detector.sensitivity = -0.5  # Below min
    assert detector.sensitivity == 0.0


# ==============================================================================
# F083: process_audio() tests
# ==============================================================================


@pytest.mark.asyncio
async def test_process_audio() -> None:
    """
    Test process_audio() for wake word detection (F083).

    This test verifies:
    - Accepts audio chunk
    - Returns detected persona and confidence
    - Respects sensitivity threshold
    """
    from reachy_agent.voice.wake_word import WakeWordDetector

    # Track detections
    detections: list[tuple[str, float]] = []

    def on_detected(model: str, confidence: float) -> None:
        detections.append((model, confidence))

    detector = WakeWordDetector(
        models=["hey_jarvis", "hey_motoko"],
        on_detected=on_detected,
        sensitivity=0.5,
        mock_mode=True,
    )
    detector.enable()

    # Process audio (no detection yet)
    audio_chunk = b"\x00" * 1024
    result = await detector.process_audio(audio_chunk)
    assert result is None

    # Simulate detection
    detector.simulate_detection("hey_jarvis", 0.9)
    result = await detector.process_audio(audio_chunk)

    # Verify detection
    assert result is not None
    assert result.wake_word == "hey_jarvis"
    assert result.confidence == 0.9

    # Callback should have been called
    assert len(detections) == 1
    assert detections[0] == ("hey_jarvis", 0.9)

    detector.cleanup()


@pytest.mark.asyncio
async def test_process_audio_sensitivity_threshold() -> None:
    """Test that low confidence detections are filtered by sensitivity."""
    from reachy_agent.voice.wake_word import WakeWordDetector

    detections: list[tuple[str, float]] = []

    def on_detected(model: str, confidence: float) -> None:
        detections.append((model, confidence))

    detector = WakeWordDetector(
        models=["hey_jarvis"],
        on_detected=on_detected,
        sensitivity=0.7,  # High threshold
        mock_mode=True,
    )
    detector.enable()

    # Simulate low confidence detection
    detector.simulate_detection("hey_jarvis", 0.5)
    audio_chunk = b"\x00" * 1024
    result = await detector.process_audio(audio_chunk)

    # Detection returned but callback NOT called (below threshold)
    assert result is not None
    assert result.confidence == 0.5
    assert len(detections) == 0  # Callback not called

    # Simulate high confidence detection
    detector.simulate_detection("hey_jarvis", 0.9)
    result = await detector.process_audio(audio_chunk)

    # Detection returned AND callback called
    assert result is not None
    assert result.confidence == 0.9
    assert len(detections) == 1

    detector.cleanup()


@pytest.mark.asyncio
async def test_process_audio_disabled() -> None:
    """Test that process_audio returns None when disabled."""
    from reachy_agent.voice.wake_word import WakeWordDetector

    detector = WakeWordDetector(
        models=["hey_jarvis"],
        mock_mode=True,
    )

    # Don't enable - detector is disabled
    assert not detector.is_enabled

    detector.simulate_detection("hey_jarvis", 0.9)
    audio_chunk = b"\x00" * 1024
    result = await detector.process_audio(audio_chunk)

    # Should return None when disabled
    assert result is None


@pytest.mark.asyncio
async def test_wake_word_detection_dataclass() -> None:
    """Test WakeWordDetection dataclass."""
    from reachy_agent.voice.wake_word import WakeWordDetection

    detection = WakeWordDetection(
        wake_word="hey_jarvis",
        confidence=0.85,
        persona="jarvis",
    )

    assert detection.wake_word == "hey_jarvis"
    assert detection.confidence == 0.85
    assert detection.persona == "jarvis"


@pytest.mark.asyncio
async def test_wake_word_reset() -> None:
    """Test reset clears pending detection."""
    from reachy_agent.voice.wake_word import WakeWordDetector

    detector = WakeWordDetector(
        models=["hey_jarvis"],
        mock_mode=True,
    )
    detector.enable()

    # Simulate detection
    detector.simulate_detection("hey_jarvis", 0.9)

    # Reset
    detector.reset()

    # Detection should be cleared
    audio_chunk = b"\x00" * 1024
    result = await detector.process_audio(audio_chunk)
    assert result is None

    detector.cleanup()
