#!/usr/bin/env python3
"""Basic simulation example.

This example demonstrates how to use the MuJoCo simulation client
for basic robot control without physical hardware.

Usage:
    uv run python examples/sim_basic.py
"""

import asyncio

from reachy_agent.simulation.client import MuJoCoReachyClient


async def main() -> None:
    """Run basic simulation example."""
    print("Creating MuJoCo simulation client...")
    client = MuJoCoReachyClient(realtime=True, viewer=False)

    print("Connecting...")
    await client.connect()

    print("Waking up robot...")
    await client.wake_up()

    # Get initial status
    status = await client.get_status()
    print(f"Robot status: awake={status.is_awake}, battery={status.battery_percent}%")

    # Get joint positions
    positions = await client.get_position()
    print(f"Initial positions: {positions}")

    # Move head
    print("Moving head to look left...")
    await client.move_head(pitch=0, yaw=30, roll=0, duration=1.0)

    print("Moving head to look right...")
    await client.move_head(pitch=0, yaw=-30, roll=0, duration=1.0)

    print("Moving head to look up...")
    await client.move_head(pitch=-20, yaw=0, roll=0, duration=1.0)

    print("Moving head to look down...")
    await client.move_head(pitch=20, yaw=0, roll=0, duration=1.0)

    # Reset to neutral
    print("Resetting to neutral position...")
    await client.reset_position(duration=1.0)

    # Move antennas
    print("Wiggling antennas...")
    await client.set_antennas(left=60, right=-60)
    await asyncio.sleep(0.5)
    await client.set_antennas(left=-60, right=60)
    await asyncio.sleep(0.5)
    await client.set_antennas(left=0, right=0)

    # Express some emotions
    print("Nodding...")
    await client.nod(intensity=0.8)

    print("Shaking head...")
    await client.shake(intensity=0.5)

    # Clean up
    print("Putting robot to sleep...")
    await client.sleep()

    print("Disconnecting...")
    await client.disconnect()

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
