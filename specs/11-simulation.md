# Spec 11: MuJoCo Simulation

## Overview

The MuJoCo simulation subsystem provides physics-accurate simulation of the Reachy Mini robot, enabling development and testing without physical hardware. It uses the MuJoCo physics engine for realistic dynamics and supports both real-time interactive use and fast-forward batch simulation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Simulation Subsystem                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │ MuJoCoReachyClient│────▶│ SimulationEnv    │                      │
│  │ (ReachyClient)    │     │ (MjModel/MjData) │                      │
│  └──────────────────┘     └────────┬─────────┘                      │
│           │                        │                                 │
│           │                        ▼                                 │
│           │               ┌──────────────────┐                      │
│           │               │   PhysicsLoop    │                      │
│           │               │   (500Hz)        │                      │
│           │               └────────┬─────────┘                      │
│           │                        │                                 │
│           ▼                        ▼                                 │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │ SimulationViewer │     │   PD Controllers │                      │
│  │ (3D Window)      │     │   (per joint)    │                      │
│  └──────────────────┘     └──────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## MJCF Model Structure

### Kinematic Chain

```
world
└── base (body_rotation: hinge, -180° to +180°)
    └── torso
        └── head_lift (head_z: slide, 0mm to 50mm)
            └── head_yaw_link (head_yaw: hinge, -60° to +60°)
                └── head_pitch_link (head_pitch: hinge, -45° to +35°)
                    └── head_roll_link (head_roll: hinge, -35° to +35°)
                        └── head
                            ├── antenna_left_base (antenna_left: hinge, -150° to +150°)
                            └── antenna_right_base (antenna_right: hinge, -150° to +150°)
```

### Joint Configuration

| Joint | Type | Limits | Default | Axis |
|-------|------|--------|---------|------|
| body_rotation | hinge | -180° to +180° | 0° | Z |
| head_z | slide | 0mm to 50mm | 0mm | Z |
| head_yaw | hinge | -60° to +60° | 0° | Z |
| head_pitch | hinge | -45° to +35° | 0° | Y |
| head_roll | hinge | -35° to +35° | 0° | X |
| antenna_left | hinge | -150° to +150° | 0° | Y |
| antenna_right | hinge | -150° to +150° | 0° | Y |

### Actuators

Each joint has a position-controlled actuator with PD gains:

| Actuator | Joint | Kp | Kv | Control Range |
|----------|-------|----|----|---------------|
| body_rotation_actuator | body_rotation | 200 | 20 | -180° to +180° |
| head_z_actuator | head_z | 500 | 50 | 0mm to 50mm |
| head_yaw_actuator | head_yaw | 100 | 10 | -60° to +60° |
| head_pitch_actuator | head_pitch | 100 | 10 | -45° to +35° |
| head_roll_actuator | head_roll | 100 | 10 | -35° to +35° |
| antenna_left_actuator | antenna_left | 50 | 5 | -150° to +150° |
| antenna_right_actuator | antenna_right | 50 | 5 | -150° to +150° |

### Sensors

| Sensor | Type | Description |
|--------|------|-------------|
| *_pos | jointpos | Position for each joint |
| *_vel | jointvel | Velocity for each joint |
| head_accel | accelerometer | 3-axis accelerometer on head |
| head_gyro | gyro | 3-axis gyroscope on head |
| head_camera | camera | RGB camera (90° FOV) |

## Components

### MuJoCoReachyClient

Implements the `ReachyClient` protocol using MuJoCo simulation.

```python
from reachy_agent.simulation import MuJoCoReachyClient

client = MuJoCoReachyClient(
    model_path="data/models/reachy_mini/reachy_mini.xml",
    realtime=True,
    viewer=True,
)

await client.connect()
await client.wake_up()
await client.move_head(pitch=10, yaw=20, roll=0, duration=1.0)
await client.disconnect()
```

### SimulationEnvironment

Low-level wrapper around MuJoCo model and data.

```python
from reachy_agent.simulation.environment import SimulationEnvironment

env = SimulationEnvironment(timestep=0.002, realtime=False)
await env.start()
await env.move_joints({"head_pitch": 15}, duration=0.5)
positions = await env.get_joint_positions()
await env.stop()
```

### SimulationViewer

Real-time 3D visualization with camera controls.

```python
from reachy_agent.simulation.viewer import SimulationViewer

viewer = SimulationViewer(model, data)
viewer.start()
viewer.set_camera_preset("front")
viewer.toggle_joint_overlay()
viewer.start_recording()
# ... simulation runs ...
frames = viewer.stop_recording()
viewer.save_video("output.mp4", fps=30, frames=frames)
viewer.stop()
```

### Physics Controllers

PD controllers for joint position control.

```python
from reachy_agent.simulation.physics import PDController, TrajectoryInterpolator

controller = PDController(kp=100, kd=10, max_torque=50)
torque = controller.compute(target=30, current=0, velocity=0)

interpolator = TrajectoryInterpolator(method="minimum_jerk")
position = interpolator.interpolate(start=0, end=30, t=0.5, duration=1.0)
```

## Configuration

### config/default.yaml

```yaml
simulation:
  model_path: data/models/reachy_mini/reachy_mini.xml
  timestep: 0.002  # 500Hz physics
  n_substeps: 4
  realtime: true
  viewer: false
  headless: false
  render:
    width: 640
    height: 480
    quality: medium
    shadows: true
    antialiasing: true
  physics:
    gravity: [0, 0, -9.81]
    friction: 0.8
    damping: 1.0
  recording:
    fps: 30
    format: mp4
```

## CLI Usage

```bash
# Run with simulation backend
reachy-agent run --sim

# Run with simulation and viewer
reachy-agent run --sim --sim-viewer

# Run fast-forward (non-realtime) simulation
reachy-agent run --sim --no-sim-realtime

# Check simulation availability
reachy-agent check
```

## Advanced Features

### Domain Randomization

For robust sim-to-real transfer:

```python
from reachy_agent.simulation.config import DomainRandomizationConfig

dr_config = DomainRandomizationConfig(
    enabled=True,
    mass_range=(0.8, 1.2),
    friction_range=(0.5, 1.2),
    noise_std=0.01,
)
```

### Parallel Simulation

For batch processing and RL training:

```python
from reachy_agent.simulation import ParallelSimulation

parallel = ParallelSimulation(n_envs=8)
observations = parallel.reset()
for _ in range(1000):
    actions = policy(observations)
    observations, rewards, dones = parallel.step(actions)
```

### Gymnasium Environment

For reinforcement learning:

```python
from reachy_agent.simulation import ReachyGymEnv

env = ReachyGymEnv()
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## Performance Considerations

- Physics runs at 500Hz (2ms timestep with 4 substeps = 0.5ms effective)
- Real-time mode matches wall clock time
- Fast-forward mode runs as fast as possible
- Viewer sync adds ~1ms per frame
- Headless rendering uses EGL/OSMesa (no display required)

## Testing

```bash
# Run simulation tests
uv run pytest tests/test_simulation.py -v

# Run with MuJoCo if available
uv run pytest tests/test_simulation.py -v --ignore-mujoco-skip

# Run performance benchmarks
uv run pytest tests/test_simulation.py::TestSimulationPerformance -v
```

## Dependencies

Install simulation dependencies:

```bash
uv pip install -e '.[sim]'
```

Required packages:
- mujoco >= 3.0.0
- dm_control >= 1.0.0
- imageio >= 2.31.0
- imageio-ffmpeg >= 0.4.8
- glfw >= 2.6.0
- gymnasium >= 0.29.0
