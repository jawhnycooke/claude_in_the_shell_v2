"""Factory for creating robot clients based on backend selection.

This module provides a factory function to create the appropriate
ReachyClient implementation based on the selected backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from reachy_agent.robot.client import Backend

if TYPE_CHECKING:
    from reachy_agent.robot.client import ReachyClient

logger = structlog.get_logger()


def create_client(
    backend: Backend | str,
    *,
    simulation_viewer: bool = False,
    simulation_realtime: bool = True,
    model_path: str | None = None,
) -> ReachyClient:
    """Create a ReachyClient for the specified backend.

    Args:
        backend: Backend to use (SDK, MOCK, or SIM)
        simulation_viewer: Enable viewer window (SIM only)
        simulation_realtime: Run in real-time (SIM only)
        model_path: Path to MJCF model (SIM only)

    Returns:
        ReachyClient implementation for the backend

    Raises:
        ValueError: If backend is not recognized
        ImportError: If required dependencies are not installed

    Example:
        >>> client = create_client(Backend.SIM, simulation_viewer=True)
        >>> await client.connect()
        >>> await client.move_head(pitch=10, yaw=0, roll=0, duration=1.0)
    """
    # Convert string to Backend enum
    if isinstance(backend, str):
        try:
            backend = Backend(backend.lower())
        except ValueError:
            raise ValueError(
                f"Unknown backend: {backend}. Must be one of: sdk, mock, sim"
            )

    log = logger.bind(backend=backend.value)
    log.info("creating_robot_client")

    if backend == Backend.SDK:
        log.debug("loading_sdk_client")
        try:
            from reachy_agent.robot.sdk import SDKClient

            return SDKClient()
        except ImportError as e:
            log.error("sdk_import_failed", error=str(e))
            raise ImportError(
                "Reachy SDK not installed. Install with: pip install reachy-sdk"
            ) from e

    elif backend == Backend.MOCK:
        log.debug("loading_mock_client")
        from reachy_agent.robot.mock import MockClient

        return MockClient()

    elif backend == Backend.SIM:
        log.debug("loading_sim_client")
        try:
            from reachy_agent.simulation.client import MuJoCoReachyClient

            return MuJoCoReachyClient(
                model_path=model_path,
                realtime=simulation_realtime,
                viewer=simulation_viewer,
            )
        except ImportError as e:
            log.error("mujoco_import_failed", error=str(e))
            raise ImportError(
                "MuJoCo not installed. Install with: pip install mujoco"
            ) from e

    else:
        raise ValueError(f"Unknown backend: {backend}")


async def create_and_connect(
    backend: Backend | str,
    **kwargs,
) -> ReachyClient:
    """Create and connect a ReachyClient.

    Convenience function that creates a client and connects it.

    Args:
        backend: Backend to use
        **kwargs: Additional arguments passed to create_client

    Returns:
        Connected ReachyClient

    Example:
        >>> client = await create_and_connect(Backend.SIM)
        >>> # Client is already connected and ready to use
        >>> await client.wake_up()
    """
    client = create_client(backend, **kwargs)
    await client.connect()
    return client
