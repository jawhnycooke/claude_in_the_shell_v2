"""MuJoCo simulation support for Reachy Agent.

This module provides physics-based simulation of the Reachy Mini robot
using the MuJoCo physics engine. It enables development and testing
without physical hardware.

Key Components:
    - MuJoCoReachyClient: ReachyClient protocol implementation using MuJoCo
    - SimulationEnvironment: Wrapper for MuJoCo model and data
    - SimulationViewer: Real-time 3D visualization
    - PhysicsLoop: Configurable physics stepping

Usage:
    >>> from reachy_agent.simulation import MuJoCoReachyClient
    >>> client = MuJoCoReachyClient()
    >>> await client.connect()
    >>> await client.move_head(pitch=10, yaw=0, roll=0, duration=1.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy imports to avoid loading MuJoCo unless needed
if TYPE_CHECKING:
    from reachy_agent.simulation.client import MuJoCoReachyClient
    from reachy_agent.simulation.environment import SimulationEnvironment
    from reachy_agent.simulation.viewer import SimulationViewer

__all__ = [
    "MuJoCoReachyClient",
    "SimulationEnvironment",
    "SimulationViewer",
]


def __getattr__(name: str) -> object:
    """Lazy import simulation components."""
    if name == "MuJoCoReachyClient":
        from reachy_agent.simulation.client import MuJoCoReachyClient

        return MuJoCoReachyClient
    elif name == "SimulationEnvironment":
        from reachy_agent.simulation.environment import SimulationEnvironment

        return SimulationEnvironment
    elif name == "SimulationViewer":
        from reachy_agent.simulation.viewer import SimulationViewer

        return SimulationViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
