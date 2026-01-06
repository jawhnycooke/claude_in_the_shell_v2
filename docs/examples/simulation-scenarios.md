# Simulation Scenarios

Working examples for MuJoCo simulation and testing.

## Basic Simulation

### Start Simulation with Viewer

```python
import asyncio
from reachy_agent.simulation import MuJoCoReachyClient

async def basic_simulation():
    """Start simulation with 3D viewer."""
    client = MuJoCoReachyClient(
        model_path="data/models/reachy_mini/reachy_mini.xml",
        realtime=True,
        viewer=True
    )

    await client.connect()
    print("Simulation started. 3D viewer should open.")

    await client.wake_up()

    # Move head and watch in viewer
    await client.move_head(pitch=20, yaw=30, duration=1.5)
    await client.move_head(pitch=-10, yaw=-20, roll=10, duration=1.5)

    # Play emotion
    await client.play_emotion("happy")

    input("Press Enter to exit...")
    await client.disconnect()

asyncio.run(basic_simulation())
```

### Headless Simulation

```python
async def headless_simulation():
    """Run simulation without viewer (faster)."""
    client = MuJoCoReachyClient(
        realtime=False,  # Run as fast as possible
        viewer=False     # No 3D window
    )

    await client.connect()
    await client.wake_up()

    # Run many iterations quickly
    for i in range(100):
        yaw = (i % 120) - 60  # Oscillate -60 to 60
        await client.move_head(yaw=yaw, duration=0.1)

    await client.disconnect()
    print("Completed 100 head movements")

asyncio.run(headless_simulation())
```

---

## Physics Testing

### Test Joint Limits

```python
async def test_joint_limits():
    """Verify joint limits are enforced."""
    client = MuJoCoReachyClient(realtime=True, viewer=True)
    await client.connect()
    await client.wake_up()

    print("Testing joint limits...")

    # Test pitch limits (-45 to +35)
    print("\nPitch limits:")
    await client.move_head(pitch=-45, duration=0.5)
    pos = await client.get_position()
    print(f"  Min pitch: {pos['head_pitch']:.1f}° (limit: -45°)")

    await client.move_head(pitch=35, duration=0.5)
    pos = await client.get_position()
    print(f"  Max pitch: {pos['head_pitch']:.1f}° (limit: +35°)")

    # Test yaw limits (-60 to +60)
    print("\nYaw limits:")
    await client.move_head(yaw=-60, duration=0.5)
    pos = await client.get_position()
    print(f"  Min yaw: {pos['head_yaw']:.1f}° (limit: -60°)")

    await client.move_head(yaw=60, duration=0.5)
    pos = await client.get_position()
    print(f"  Max yaw: {pos['head_yaw']:.1f}° (limit: +60°)")

    # Test beyond limits (should clamp)
    print("\nTesting clamping:")
    await client.move_head(pitch=100, yaw=100, duration=0.5)
    pos = await client.get_position()
    print(f"  Requested (100, 100), got ({pos['head_pitch']:.1f}, {pos['head_yaw']:.1f})")

    await client.disconnect()

asyncio.run(test_joint_limits())
```

### Test Movement Timing

```python
import time

async def test_movement_timing():
    """Verify movement durations are accurate."""
    client = MuJoCoReachyClient(realtime=True, viewer=True)
    await client.connect()
    await client.wake_up()

    durations = [0.5, 1.0, 2.0]

    for target_duration in durations:
        # Reset
        await client.move_head(pitch=0, yaw=0, duration=0.3)
        await asyncio.sleep(0.2)

        # Timed movement
        start = time.time()
        await client.move_head(yaw=45, duration=target_duration)
        actual_duration = time.time() - start

        print(f"Target: {target_duration:.1f}s, Actual: {actual_duration:.2f}s")

    await client.disconnect()

asyncio.run(test_movement_timing())
```

---

## Recording and Playback

### Record Simulation Video

```python
from reachy_agent.simulation.viewer import SimulationViewer

async def record_video():
    """Record simulation to video file."""
    client = MuJoCoReachyClient(realtime=True, viewer=True)
    await client.connect()
    await client.wake_up()

    # Access viewer and start recording
    viewer = client._env._viewer
    viewer.start_recording()

    print("Recording...")

    # Perform movements
    await client.play_emotion("happy")
    await asyncio.sleep(0.5)
    await client.move_head(yaw=-30, pitch=10, duration=1.0)
    await client.move_head(yaw=30, pitch=-10, duration=1.0)
    await client.play_emotion("curious")
    await asyncio.sleep(0.5)

    # Stop and save
    frames = viewer.stop_recording()
    viewer.save_video("robot_demo.mp4", fps=30, frames=frames)

    print(f"Saved video with {len(frames)} frames")

    await client.disconnect()

asyncio.run(record_video())
```

### Camera Presets

```python
async def camera_presets():
    """Demo different camera angles."""
    client = MuJoCoReachyClient(realtime=True, viewer=True)
    await client.connect()
    await client.wake_up()

    viewer = client._env._viewer

    presets = ["front", "side", "top"]

    for preset in presets:
        print(f"Camera: {preset}")
        viewer.set_camera_preset(preset)

        # Do some movement
        await client.move_head(yaw=30, duration=1.0)
        await client.move_head(yaw=-30, duration=1.0)

        await asyncio.sleep(1.0)

    await client.disconnect()

asyncio.run(camera_presets())
```

---

## Sensor Simulation

### Read Simulated Sensors

```python
async def sensor_demo():
    """Read simulated sensor data."""
    client = MuJoCoReachyClient(realtime=True, viewer=True)
    await client.connect()
    await client.wake_up()

    # Read sensors during movement
    for i in range(20):
        # Get sensor data
        sensors = await client.get_sensor_data()

        print(f"\nFrame {i}:")
        print(f"  Accelerometer: {sensors['accelerometer']}")
        print(f"  Gyroscope: {sensors['gyroscope']}")

        # Move head
        yaw = 30 * (1 if i % 2 == 0 else -1)
        await client.move_head(yaw=yaw, duration=0.3)

    await client.disconnect()

asyncio.run(sensor_demo())
```

### Capture Simulated Images

```python
async def camera_capture():
    """Capture images from simulated camera."""
    client = MuJoCoReachyClient(realtime=True, viewer=True)
    await client.connect()
    await client.wake_up()

    # Capture image
    image_data = await client.capture_image()

    # Save to file
    with open("sim_capture.png", "wb") as f:
        f.write(image_data)

    print(f"Saved image ({len(image_data)} bytes)")

    # Capture from different angles
    positions = [
        {"yaw": -45, "pitch": 0},
        {"yaw": 0, "pitch": 20},
        {"yaw": 45, "pitch": -10},
    ]

    for i, pos in enumerate(positions):
        await client.move_head(**pos, duration=0.5)
        image_data = await client.capture_image()
        with open(f"sim_capture_{i}.png", "wb") as f:
            f.write(image_data)
        print(f"Saved sim_capture_{i}.png")

    await client.disconnect()

asyncio.run(camera_capture())
```

---

## Advanced Scenarios

### Domain Randomization

```python
from reachy_agent.simulation.config import DomainRandomizationConfig

async def domain_randomization():
    """Test with domain randomization for robust training."""
    dr_config = DomainRandomizationConfig(
        enabled=True,
        mass_range=(0.8, 1.2),      # ±20% mass variation
        friction_range=(0.5, 1.5),  # Friction variation
        noise_std=0.02,             # Sensor noise
    )

    client = MuJoCoReachyClient(
        realtime=False,
        viewer=False,
        domain_randomization=dr_config
    )

    results = []

    # Run multiple episodes with randomization
    for episode in range(10):
        await client.connect()  # Randomizes environment
        await client.wake_up()

        # Execute movement
        start_time = asyncio.get_event_loop().time()
        await client.move_head(yaw=45, duration=1.0)
        end_time = asyncio.get_event_loop().time()

        pos = await client.get_position()
        results.append({
            "episode": episode,
            "final_yaw": pos["head_yaw"],
            "duration": end_time - start_time
        })

        await client.disconnect()

    # Analyze results
    yaws = [r["final_yaw"] for r in results]
    print(f"Yaw across episodes: mean={sum(yaws)/len(yaws):.2f}, "
          f"min={min(yaws):.2f}, max={max(yaws):.2f}")

asyncio.run(domain_randomization())
```

### Parallel Environments

```python
from reachy_agent.simulation import ParallelSimulation

async def parallel_simulation():
    """Run multiple simulations in parallel."""
    parallel = ParallelSimulation(n_envs=4)

    # Reset all environments
    observations = parallel.reset()
    print(f"Running {len(observations)} parallel environments")

    # Run simulation steps
    for step in range(100):
        # Random actions for each environment
        actions = [
            {"pitch": random.uniform(-20, 20), "yaw": random.uniform(-40, 40)}
            for _ in range(4)
        ]

        observations, rewards, dones = parallel.step(actions)

        if step % 20 == 0:
            print(f"Step {step}: observations={len(observations)}")

    parallel.close()

asyncio.run(parallel_simulation())
```

### Gymnasium Environment

```python
from reachy_agent.simulation import ReachyGymEnv

async def gym_environment():
    """Use OpenAI Gym interface for RL."""
    env = ReachyGymEnv()

    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    total_reward = 0

    for step in range(100):
        # Random action
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            obs, info = env.reset()

    print(f"Total reward: {total_reward:.2f}")
    env.close()

asyncio.run(gym_environment())
```

---

## Agent in Simulation

### Full Agent with Simulation

```python
from reachy_agent.agent.loop import ReachyAgentLoop
from reachy_agent.agent.options import AgentConfig
from reachy_agent.robot.factory import Backend

async def agent_in_simulation():
    """Run full agent with MuJoCo backend."""
    config = AgentConfig(
        model="claude-haiku-4-5-20251001",
        enable_voice=False,
        enable_motion=True,
        backend=Backend.SIM,
        sim_viewer=True,
        sim_realtime=True
    )

    agent = ReachyAgentLoop(config)

    try:
        await agent.start()
        print("Agent running in simulation. 3D viewer should open.")

        # Send commands
        responses = [
            await agent.process("Look to your left"),
            await agent.process("Play a happy emotion"),
            await agent.process("Rotate your body 180 degrees"),
            await agent.process("Look up and wiggle your antennas"),
        ]

        for response in responses:
            print(f"Agent: {response[:100]}...")
            await asyncio.sleep(1.0)

        input("Press Enter to exit...")

    finally:
        await agent.stop()

asyncio.run(agent_in_simulation())
```

---

## Testing Framework

### Simulation Test Suite

```python
import pytest

class TestSimulation:
    """Test suite for simulation."""

    @pytest.fixture
    async def client(self):
        """Create and connect simulation client."""
        client = MuJoCoReachyClient(realtime=False, viewer=False)
        await client.connect()
        await client.wake_up()
        yield client
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_head_movement(self, client):
        """Test head can move to target position."""
        await client.move_head(pitch=20, yaw=30, duration=0.5)
        pos = await client.get_position()

        assert abs(pos["head_pitch"] - 20) < 2.0
        assert abs(pos["head_yaw"] - 30) < 2.0

    @pytest.mark.asyncio
    async def test_body_rotation(self, client):
        """Test body rotation."""
        await client.rotate_body(angle=90, duration=0.5)
        pos = await client.get_position()

        assert abs(pos["body_rotation"] - 90) < 5.0

    @pytest.mark.asyncio
    async def test_joint_limits_enforced(self, client):
        """Test that joint limits prevent over-rotation."""
        # Request beyond limits
        await client.move_head(pitch=100, duration=0.5)
        pos = await client.get_position()

        # Should be clamped to 35
        assert pos["head_pitch"] <= 35
```

Run tests:

```bash
uv run pytest tests/test_simulation.py -v
```

---

## Performance Benchmarks

```python
import time

async def benchmark_simulation():
    """Benchmark simulation performance."""
    client = MuJoCoReachyClient(realtime=False, viewer=False)
    await client.connect()
    await client.wake_up()

    # Benchmark movement commands
    n_commands = 1000
    start = time.time()

    for i in range(n_commands):
        yaw = (i % 120) - 60
        await client.move_head(yaw=yaw, duration=0.01)

    elapsed = time.time() - start
    commands_per_sec = n_commands / elapsed

    print(f"Movement commands: {commands_per_sec:.1f}/sec")

    # Benchmark position reads
    n_reads = 1000
    start = time.time()

    for _ in range(n_reads):
        await client.get_position()

    elapsed = time.time() - start
    reads_per_sec = n_reads / elapsed

    print(f"Position reads: {reads_per_sec:.1f}/sec")

    await client.disconnect()

asyncio.run(benchmark_simulation())
```

---

## Next Steps

- [MuJoCo Integration Guide](../developer-guide/mujoco-integration.md) - Deep dive
- [MuJoCo Setup Tutorial](../tutorials/mujoco-setup.md) - Installation guide
- [Basic Movements](basic-movements.md) - Movement examples
