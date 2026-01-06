"""Shared pytest fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def config_path() -> Path:
    """Path to test config file."""
    return Path("config/default.yaml")


@pytest.fixture
def mock_robot():
    """Mock robot client for testing."""
    from reachy_agent.robot.mock import MockClient

    client = MockClient()
    yield client
    # Cleanup if needed


@pytest.fixture
async def agent_config():
    """Default agent configuration for testing."""
    from reachy_agent.agent.options import AgentConfig

    return AgentConfig(mock_hardware=True, enable_voice=False)


@pytest.fixture
def mock_memory_manager() -> Generator:
    """
    Mock memory manager with temporary storage.

    Creates a MemoryManager with a temp directory that is
    cleaned up after the test.
    """
    from reachy_agent.memory.manager import MemoryManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = MemoryManager(
            persist_path=tmpdir,
            context_window_size=5,
            collection_name="test_memories",
        )
        yield manager


@pytest.fixture
def mock_permission_evaluator():
    """
    Mock permission evaluator with default rules.

    Returns an evaluator with standard rules for testing.
    """
    from reachy_agent.permissions.evaluator import (
        PermissionEvaluator,
        PermissionRule,
        PermissionTier,
    )

    rules = [
        PermissionRule(pattern="move_*", tier=PermissionTier.AUTONOMOUS),
        PermissionRule(pattern="look_*", tier=PermissionTier.AUTONOMOUS),
        PermissionRule(pattern="speak", tier=PermissionTier.AUTONOMOUS),
        PermissionRule(pattern="memory_*", tier=PermissionTier.CONFIRM),
        PermissionRule(pattern="shutdown", tier=PermissionTier.FORBIDDEN),
    ]
    return PermissionEvaluator(rules, audit_enabled=False)


@pytest.fixture
def test_config() -> dict:
    """
    Test configuration dictionary.

    Returns a minimal config for testing without file I/O.
    """
    return {
        "agent": {
            "model": "claude-sonnet-4-20250514",
            "name": "TestBot",
            "mock_mode": True,
        },
        "voice": {
            "enabled": False,
            "wake_sensitivity": 0.5,
        },
        "motion": {
            "tick_hz": 30,
        },
        "memory": {
            "path": "/tmp/test_memory",
            "context_window_size": 5,
        },
    }


@pytest.fixture
def sample_emotion_data() -> dict:
    """
    Sample emotion animation data for testing.

    Returns a minimal emotion with a few frames.
    """
    return {
        "name": "test_emotion",
        "fps": 30,
        "frames": [
            {"pitch": 0, "yaw": 0, "roll": 0, "antenna_left": 0, "antenna_right": 0},
            {"pitch": 5, "yaw": 2, "roll": 1, "antenna_left": 10, "antenna_right": 10},
            {"pitch": 10, "yaw": 5, "roll": 2, "antenna_left": 20, "antenna_right": 20},
            {"pitch": 5, "yaw": 2, "roll": 1, "antenna_left": 10, "antenna_right": 10},
            {"pitch": 0, "yaw": 0, "roll": 0, "antenna_left": 0, "antenna_right": 0},
        ],
    }


@pytest.fixture
def temp_persona_dir() -> Generator:
    """
    Temporary directory with test persona files.

    Creates persona files for testing persona loading.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        persona_path = Path(tmpdir)

        # Create a test persona
        default_persona = persona_path / "default.md"
        default_persona.write_text(
            "# Default Persona\n\n"
            "You are a helpful robot assistant.\n"
            "Be friendly and concise.\n"
        )

        friendly_persona = persona_path / "friendly.md"
        friendly_persona.write_text(
            "# Friendly Persona\n\n"
            "You are a cheerful, enthusiastic robot.\n"
            "Always be positive and encouraging.\n"
        )

        yield persona_path
