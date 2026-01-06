#!/usr/bin/env python3
"""Teleoperation example.

This example demonstrates how to control the robot using keyboard input
in the simulation. Note: This requires a graphical display and viewer.

Usage:
    uv run python examples/sim_teleop.py

Controls:
    W/S - Head pitch (up/down)
    A/D - Head yaw (left/right)
    Q/E - Head roll (tilt)
    Z/X - Body rotation (left/right)
    R   - Reset to neutral position
    ESC - Exit
"""

import asyncio

from reachy_agent.simulation.client import MuJoCoReachyClient

# Keyboard mapping for teleoperation
KEYBOARD_MAPPINGS = {
    "w": ("head_pitch", -5),  # Look up
    "s": ("head_pitch", 5),  # Look down
    "a": ("head_yaw", 5),  # Look left
    "d": ("head_yaw", -5),  # Look right
    "q": ("head_roll", -5),  # Tilt left
    "e": ("head_roll", 5),  # Tilt right
    "z": ("body_rotation", 5),  # Rotate left
    "x": ("body_rotation", -5),  # Rotate right
}


class TeleoperationController:
    """Simple teleoperation controller for simulation."""

    def __init__(self, client: MuJoCoReachyClient) -> None:
        """Initialize teleoperation controller.

        Args:
            client: MuJoCo simulation client
        """
        self.client = client
        self.current_pose = {
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "body_rotation": 0.0,
        }

    async def process_key(self, key: str) -> bool:
        """Process a keyboard input.

        Args:
            key: Key pressed

        Returns:
            True to continue, False to exit
        """
        key = key.lower()

        if key == "\x1b":  # ESC
            return False

        if key == "r":
            # Reset to neutral
            self.current_pose = {
                "head_pitch": 0.0,
                "head_yaw": 0.0,
                "head_roll": 0.0,
                "body_rotation": 0.0,
            }
            await self._apply_pose()
            print("Reset to neutral position")
            return True

        if key in KEYBOARD_MAPPINGS:
            joint, delta = KEYBOARD_MAPPINGS[key]
            self.current_pose[joint] = self._clamp(
                self.current_pose[joint] + delta, joint
            )
            await self._apply_pose()
            print(f"  {joint}: {self.current_pose[joint]:.1f}")

        return True

    def _clamp(self, value: float, joint: str) -> float:
        """Clamp joint value to limits."""
        limits = {
            "head_pitch": (-45, 35),
            "head_yaw": (-60, 60),
            "head_roll": (-35, 35),
            "body_rotation": (-180, 180),
        }
        min_val, max_val = limits.get(joint, (-180, 180))
        return max(min_val, min(max_val, value))

    async def _apply_pose(self) -> None:
        """Apply current pose to robot."""
        await self.client.move_head(
            pitch=self.current_pose["head_pitch"],
            yaw=self.current_pose["head_yaw"],
            roll=self.current_pose["head_roll"],
            duration=0.1,
        )


async def main() -> None:
    """Run teleoperation example."""
    print("Reachy Mini Teleoperation")
    print("=" * 50)
    print("\nControls:")
    print("  W/S - Head pitch (up/down)")
    print("  A/D - Head yaw (left/right)")
    print("  Q/E - Head roll (tilt)")
    print("  Z/X - Body rotation")
    print("  R   - Reset to neutral")
    print("  ESC - Exit")
    print("\n" + "=" * 50)

    print("\nCreating simulation client...")
    client = MuJoCoReachyClient(realtime=True, viewer=False)

    await client.connect()
    await client.wake_up()

    controller = TeleoperationController(client)

    print("\nTeleoperation active. Press keys to control robot.")
    print("(Note: In this example, we simulate key input)")
    print()

    # Simulate some key presses for demonstration
    demo_keys = ["w", "w", "a", "d", "d", "s", "r"]
    for key in demo_keys:
        print(f"Pressing '{key}'...")
        await controller.process_key(key)
        await asyncio.sleep(0.3)

    # Clean up
    print("\nExiting teleoperation...")
    await client.sleep()
    await client.disconnect()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
