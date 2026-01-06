#!/usr/bin/env python3
"""Domain randomization example.

This example demonstrates how to use domain randomization for
sim-to-real transfer and robust policy training.

Usage:
    uv run python examples/sim_domain_rand.py
"""

import asyncio

import numpy as np

from reachy_agent.simulation.randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
    VisualRandomizationConfig,
    VisualRandomizer,
)


async def main() -> None:
    """Run domain randomization example."""
    print("Domain Randomization Example")
    print("=" * 50)

    # Configure physics domain randomization
    physics_config = DomainRandomizationConfig(
        enabled=True,
        seed=42,  # For reproducibility
        mass_range=(0.8, 1.2),  # 80% to 120% of original mass
        friction_range=(0.5, 1.5),  # Friction coefficient range
        damping_range=(0.5, 2.0),  # Joint damping range
        sensor_noise_std=0.02,  # Sensor noise standard deviation
        sensor_bias_range=(-0.05, 0.05),  # Sensor bias range
        joint_noise_std=0.01,  # Joint position noise (radians)
    )

    physics_randomizer = DomainRandomizer(physics_config)

    # Configure visual domain randomization
    visual_config = VisualRandomizationConfig(
        enabled=True,
        texture_variation=0.3,  # Texture color variation
        lighting_range=(0.5, 1.5),  # Light intensity range
        camera_position_noise=0.02,  # Camera position noise (meters)
        camera_orientation_noise=0.05,  # Camera orientation noise (radians)
    )

    visual_randomizer = VisualRandomizer(visual_config)

    # Demonstrate sensor noise
    print("\n1. Sensor Noise Demonstration")
    print("-" * 30)
    clean_sensor_data = np.array([0.0, 0.0, 9.81])  # Accelerometer reading
    print(f"Clean sensor data: {clean_sensor_data}")

    for i in range(5):
        noisy_data = physics_randomizer.add_sensor_noise(clean_sensor_data.copy())
        print(f"  Sample {i + 1}: {noisy_data}")

    # Demonstrate joint noise
    print("\n2. Joint Position Noise Demonstration")
    print("-" * 30)
    clean_positions = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print(f"Clean joint positions: {clean_positions}")

    for i in range(5):
        noisy_positions = physics_randomizer.add_joint_noise(clean_positions.copy())
        print(f"  Sample {i + 1}: {noisy_positions}")

    # Demonstrate camera pose randomization
    print("\n3. Camera Pose Randomization")
    print("-" * 30)
    camera_pos = np.array([0.0, 0.0, 1.0])
    camera_orient = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    print(f"Original camera position: {camera_pos}")

    for i in range(5):
        new_pos, new_orient = visual_randomizer.randomize_camera(
            camera_pos.copy(), camera_orient.copy()
        )
        print(f"  Sample {i + 1} position: {new_pos}")

    # Explain usage with MuJoCo
    print("\n4. Usage with MuJoCo Model")
    print("-" * 30)
    print(
        """
To apply domain randomization to a MuJoCo model:

    # Load model
    model = mujoco.MjModel.from_xml_path("path/to/model.xml")
    data = mujoco.MjData(model)

    # Create randomizer
    randomizer = DomainRandomizer(config)

    # Apply randomization (modifies model in place)
    randomizer.apply(model)

    # Run simulation...

    # Reset to original values when done
    randomizer.reset(model)

This is useful for:
- Training robust policies that transfer to real hardware
- Data augmentation for imitation learning
- Testing controller robustness
"""
    )

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
