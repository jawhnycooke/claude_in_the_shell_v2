"""Tests for agent loop."""

import pytest


class TestAgentLoop:
    """Tests for ReachyAgentLoop."""

    def test_initialization(self, agent_config):
        """Test agent can be initialized."""
        from reachy_agent.agent.loop import ReachyAgentLoop

        agent = ReachyAgentLoop(agent_config)
        assert agent is not None

    # TODO: Add more tests
    # - test_start_stop
    # - test_process_input
    # - test_tool_execution
    # - test_error_recovery
