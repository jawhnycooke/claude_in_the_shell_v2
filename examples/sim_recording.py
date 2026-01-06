#!/usr/bin/env python3
"""Simulation recording example.

This example demonstrates how to record video from the simulation
for debugging, documentation, or training data collection.

Usage:
    uv run python examples/sim_recording.py

Requires:
    - MuJoCo installed
    - imageio and imageio-ffmpeg for video encoding
"""

import asyncio
from pathlib import Path

from reachy_agent.simulation.client import MuJoCoReachyClient
from reachy_agent.simulation.viewer import HeadlessRenderer


async def main() -> None:
    """Run recording example."""
    print("Creating MuJoCo simulation client with headless rendering...")

    # Create client and renderer
    client = MuJoCoReachyClient(realtime=False, viewer=False)
    await client.connect()
    await client.wake_up()

    # Create headless renderer for capturing frames
    try:
        renderer = HeadlessRenderer(width=640, height=480)
    except RuntimeError as e:
        print(f"Warning: Could not create renderer - {e}")
        print("Recording will be skipped, but robot control will continue.")
        renderer = None

    frames = []
    print("Recording robot movements...")

    # Perform some movements and capture frames
    movements = [
        ("Looking left", {"pitch": 0, "yaw": 30, "roll": 0}),
        ("Looking right", {"pitch": 0, "yaw": -30, "roll": 0}),
        ("Looking up", {"pitch": -20, "yaw": 0, "roll": 0}),
        ("Looking down", {"pitch": 20, "yaw": 0, "roll": 0}),
        ("Neutral", {"pitch": 0, "yaw": 0, "roll": 0}),
    ]

    for description, pose in movements:
        print(f"  {description}...")
        await client.move_head(**pose, duration=0.5)

        # Capture frame (if renderer available)
        if renderer is not None and client._env is not None:
            try:
                frame = renderer.render(client._env._model, client._env._data)
                frames.append(frame)
            except Exception as e:
                print(f"  Warning: Could not capture frame - {e}")

    # Save video if we have frames
    if frames:
        output_path = Path("simulation_recording.mp4")
        print(f"Saving {len(frames)} frames to {output_path}...")

        try:
            import imageio

            imageio.mimsave(str(output_path), frames, fps=30)
            print(f"Video saved to {output_path}")
        except ImportError:
            print("imageio not installed - cannot save video")
            print("Install with: pip install imageio imageio-ffmpeg")
    else:
        print("No frames captured - skipping video save")

    # Clean up
    print("Cleaning up...")
    if renderer is not None:
        renderer.close()
    await client.sleep()
    await client.disconnect()

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
