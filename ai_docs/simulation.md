# MuJoCo Simulation Quick Reference

## Overview

The MuJoCo simulation provides physics-accurate simulation of the Reachy Mini robot.

## Installation

```bash
# Install simulation dependencies
uv pip install -e '.[sim]'

# Or manually
pip install mujoco dm_control gymnasium imageio imageio-ffmpeg glfw
```

## CLI Usage

```bash
# Run with simulation (headless)
reachy-agent run --sim

# Run with 3D viewer
reachy-agent run --sim --sim-viewer

# Run fast-forward (for testing)
reachy-agent run --sim --no-sim-realtime

# Check simulation availability
reachy-agent check
```

## Python API

### Basic Usage

```python
from reachy_agent.simulation import MuJoCoReachyClient

# Create and connect
client = MuJoCoReachyClient(viewer=True, realtime=True)
await client.connect()
await client.wake_up()

# Move robot
await client.move_head(pitch=10, yaw=20, roll=0, duration=1.0)
await client.set_antennas(left=45, right=-45)
await client.nod(intensity=0.8)

# Get state
status = await client.get_status()
positions = await client.get_position()

# Cleanup
await client.sleep()
await client.disconnect()
```

### Client Factory

```python
from reachy_agent.robot.factory import create_client
from reachy_agent.robot.client import Backend

# Create client for any backend
client = create_client(Backend.SIM, simulation_viewer=True)
# or
client = create_client("sim", simulation_viewer=True)
```

### Low-Level Environment

```python
from reachy_agent.simulation.environment import SimulationEnvironment

env = SimulationEnvironment(timestep=0.002, realtime=False)
await env.start()

# Direct joint control
await env.move_joints({"head_pitch": 15, "head_yaw": 10}, duration=0.5)

# Get physics state
positions = await env.get_joint_positions()
sensor_data = await env.get_sensor_data()

await env.stop()
```

### Physics Controllers

```python
from reachy_agent.simulation.physics import PDController, TrajectoryInterpolator

# PD controller for position control
pd = PDController(kp=100, kd=10, max_torque=50)
torque = pd.compute(target=30, current=0, velocity=0)

# Smooth trajectory interpolation
interp = TrajectoryInterpolator(method="minimum_jerk")
pos = interp.interpolate(start=0, end=30, t=0.5, duration=1.0)
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
  physics:
    gravity: [0, 0, -9.81]
    friction: 0.8
```

### SimulationConfig Model

```python
from reachy_agent.simulation.config import SimulationConfig

config = SimulationConfig(
    timestep=0.002,
    realtime=True,
    viewer=True,
)
print(config.physics_hz)  # 2000.0
```

## MJCF Model

The Reachy Mini model is located at `data/models/reachy_mini/reachy_mini.xml`.

### Joints

| Joint | Type | Limits | Default |
|-------|------|--------|---------|
| body_rotation | hinge | -180° to +180° | 0° |
| head_z | slide | 0 to 50mm | 0mm |
| head_yaw | hinge | -60° to +60° | 0° |
| head_pitch | hinge | -45° to +35° | 0° |
| head_roll | hinge | -35° to +35° | 0° |
| antenna_left | hinge | -150° to +150° | 0° |
| antenna_right | hinge | -150° to +150° | 0° |

### Sensors

- Joint position sensors (`*_pos`)
- Joint velocity sensors (`*_vel`)
- IMU (accelerometer + gyroscope)
- Camera (90° FOV, 640x480)

## Viewer Controls

When using `--sim-viewer`:

| Control | Action |
|---------|--------|
| Left click + drag | Rotate camera |
| Right click + drag | Pan camera |
| Scroll | Zoom |
| Double-click | Center view on body |
| Space | Pause/resume |
| Tab | Toggle UI panels |

## Testing

```bash
# Run simulation tests
uv run pytest tests/test_simulation.py -v

# Run only tests that don't need MuJoCo
uv run pytest tests/test_simulation.py -v -k "not mujoco"
```

## Troubleshooting

### MuJoCo not found

```bash
pip install mujoco
```

### No display (headless server)

Set environment variable before running:

```bash
export MUJOCO_GL=egl  # or osmesa
reachy-agent run --sim
```

### Slow simulation

- Disable viewer: Remove `--sim-viewer`
- Use fast-forward: Add `--no-sim-realtime`
- Reduce render quality in config

### Model loading errors

Check model file exists:

```bash
ls data/models/reachy_mini/reachy_mini.xml
```
