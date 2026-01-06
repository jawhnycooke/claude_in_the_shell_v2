# Tutorial: MuJoCo Simulation Setup

Set up physics-accurate simulation for development without hardware.

**Time**: 20 minutes
**Prerequisites**: Basic familiarity with the agent

## What You'll Learn

- Install MuJoCo simulation dependencies
- Run the agent in simulation mode
- Use the 3D viewer
- Record simulation videos
- Configure physics parameters

## Step 1: Install Simulation Dependencies

```bash
uv pip install -e ".[sim]"
```

This installs:
- `gymnasium[mujoco]` - MuJoCo physics engine
- `imageio` / `imageio-ffmpeg` - Video recording
- `glfw` - Window management

Verify installation:
```bash
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
```

Expected output:
```
MuJoCo version: 3.0.0
```

## Step 2: Run Basic Simulation

Start the agent in simulation mode:

```bash
python -m reachy_agent run --sim
```

You should see:
```
🤖 Reachy Agent (simulation mode)
📦 MuJoCo model loaded: data/models/reachy_mini/reachy_mini.xml
⚙️ Physics running at 500Hz

>
```

Try some commands:
```
> Wake up

Waking up... *motors enabled in simulation*

> Move your head up

*simulated head moves with physics*
Head moved to pitch: 20°
```

## Step 3: Enable the 3D Viewer

Add `--sim-viewer` to see the robot:

```bash
python -m reachy_agent run --sim --sim-viewer
```

A 3D window will open showing the robot model.

### Viewer Controls

**Keyboard:**
| Key | Action |
|-----|--------|
| `Tab` | Cycle camera presets (front, side, top, iso) |
| `Space` | Pause/resume simulation |
| `R` | Reset to initial pose |
| `J` | Toggle joint angle overlay |
| `Escape` | Close viewer |

**Mouse:**
| Action | Effect |
|--------|--------|
| Left drag | Orbit camera |
| Right drag | Pan camera |
| Scroll | Zoom in/out |

## Step 4: Test Physics

The simulation has realistic physics. Try:

```
> Move your head quickly to look left

*head swings left with momentum, slight overshoot, then settles*
```

Compare with slow movement:
```
> Slowly look to the right over 3 seconds

*head smoothly rotates right without overshoot*
```

## Step 5: Camera Presets

Change the view while running:

Press `Tab` to cycle through:

1. **Front** - Face the robot head-on
2. **Side** - Profile view
3. **Top** - Bird's eye view
4. **Isometric** - 3/4 angle view

Or use voice/text:
```
> Set the camera to the front view

*viewer switches to front camera preset*
```

## Step 6: Joint Overlay

Press `J` to toggle the joint angle display:

```
Joint Positions:
├── body_rotation: 0.0°
├── head_z: 0.0mm
├── head_yaw: 0.0°
├── head_pitch: 0.0°
├── head_roll: 0.0°
├── antenna_left: 0.0°
└── antenna_right: 0.0°
```

This updates in real-time as the robot moves.

## Step 7: Fast-Forward Mode

For batch processing or testing, disable real-time:

```bash
python -m reachy_agent run --sim --no-sim-realtime
```

The simulation runs as fast as possible (useful for automated tests).

## Step 8: Record Video

### From the Viewer

1. Start with viewer: `python -m reachy_agent run --sim --sim-viewer`
2. Press `V` to start recording
3. Perform actions
4. Press `V` again to stop and save

### Programmatically

```python
from reachy_agent.simulation.viewer import SimulationViewer

# In your script
viewer.start_recording()

# ... perform actions ...

frames = viewer.stop_recording()
viewer.save_video("my_demo.mp4", fps=30, frames=frames)
```

## Step 9: Configure Physics

Edit `config/default.yaml`:

```yaml
simulation:
  # Physics settings
  timestep: 0.002      # 500Hz physics (2ms)
  n_substeps: 4        # Physics substeps per step
  realtime: true       # Match wall clock time

  # Rendering
  render:
    width: 1280
    height: 720
    quality: high      # low, medium, high
    shadows: true
    antialiasing: true

  # Physics tuning
  physics:
    gravity: [0, 0, -9.81]
    friction: 0.8
    damping: 1.0
```

### Physics Quality Presets

**Low (fast):**
```yaml
timestep: 0.004
n_substeps: 2
render:
  quality: low
  shadows: false
```

**High (accurate):**
```yaml
timestep: 0.001
n_substeps: 8
render:
  quality: high
  shadows: true
```

## Step 10: Headless Mode

For servers without a display:

```bash
# Use EGL rendering
export MUJOCO_GL=egl
python -m reachy_agent run --sim

# Or OSMesa (software rendering)
export MUJOCO_GL=osmesa
python -m reachy_agent run --sim
```

## Advanced: Python API

### Direct Environment Access

```python
from reachy_agent.simulation.environment import SimulationEnvironment

# Create environment
env = SimulationEnvironment(timestep=0.002, realtime=False)
await env.start()

# Direct joint control
await env.move_joints({"head_pitch": 15, "head_yaw": 30}, duration=0.5)

# Read state
positions = await env.get_joint_positions()
print(f"Head pitch: {positions['head_pitch']:.1f}°")

# Get sensor data
sensors = await env.get_sensor_data()
print(f"Accelerometer: {sensors['accelerometer']}")

await env.stop()
```

### Gymnasium Environment

For reinforcement learning:

```python
from reachy_agent.simulation import ReachyGymEnv

env = ReachyGymEnv()
obs, info = env.reset()

for step in range(1000):
    # Random action
    action = env.action_space.sample()

    # Step simulation
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Troubleshooting

### "MuJoCo not found"

```bash
pip install gymnasium[mujoco]
# Or specifically
pip install mujoco
```

### Viewer Won't Open

**Linux:**
```bash
# Check display
echo $DISPLAY

# Install GLFW
sudo apt-get install libglfw3 libglfw3-dev
```

**macOS:**
```bash
# Allow the app in Security settings
# System Preferences > Security & Privacy
```

**SSH Sessions:**
```bash
# Use X forwarding
ssh -X user@host

# Or headless mode
export MUJOCO_GL=egl
```

### Slow Performance

1. Use `--no-sim-realtime` for batch processing
2. Lower render quality in config
3. Disable shadows
4. Use smaller window size

### Physics Instability

1. Increase substeps: `n_substeps: 8`
2. Decrease timestep: `timestep: 0.001`
3. Reduce PD gains in model

## What's Next?

You've set up MuJoCo simulation! Continue to:

- [Custom Behaviors](custom-behaviors.md) - Create custom motion behaviors
- [Simulation Guide](../user-guide/simulation.md) - Complete reference
- [Developer Guide](../developer-guide/mujoco-integration.md) - Architecture details

## Summary

In this tutorial, you learned to:

- [x] Install simulation dependencies
- [x] Run the agent in simulation mode
- [x] Use the 3D viewer with controls
- [x] Test physics behavior
- [x] Use camera presets and overlays
- [x] Run in fast-forward mode
- [x] Record simulation videos
- [x] Configure physics parameters
- [x] Use headless mode
- [x] Access the Python API

---

**Congratulations!** You've set up MuJoCo simulation.
