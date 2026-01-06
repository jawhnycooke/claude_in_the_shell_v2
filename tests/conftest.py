"""Shared pytest fixtures."""

import pytest
from pathlib import Path


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


# TODO: Add more fixtures as needed
# - Mock memory manager
# - Mock permissions evaluator
# - Test persona files
# - Sample emotion data
