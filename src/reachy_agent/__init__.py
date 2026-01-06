"""Claude in the Shell v2 - Embodied AI agent for Reachy Mini robot."""

__version__ = "0.1.0"
__author__ = "Claude in the Shell Team"

from reachy_agent.agent.loop import ReachyAgentLoop
from reachy_agent.agent.options import AgentConfig

__all__ = ["ReachyAgentLoop", "AgentConfig", "__version__"]
