# Reachy Mini MuJoCo Validation Suite

This document describes the comprehensive validation suite for testing Reachy Mini robot commands in MuJoCo simulation.

## Overview

The validation suite provides automated testing and visual demonstration of all robot capabilities without requiring physical hardware. It uses the MuJoCo physics engine to simulate robot behavior and validate that all commands work correctly.

## Quick Start

### Prerequisites

Install simulation dependencies:

```bash
uv pip install -e ".[sim]"
```

This installs:
- `gymnasium[mujoco]` - Gymnasium with MuJoCo physics
- `imageio` - Video recording support
- `glfw` - OpenGL windowing

### Running Validation

**CLI Command (Recommended):**

```bash
# Full validation with 3D viewer
reachy-agent validate

# Headless validation (no GUI)
reachy-agent validate --no-viewer

# Quick smoke test
reachy-agent validate --quick

# Fast-forward mode (not real-time)
reachy-agent validate --no-realtime
```

**Visual Demo Script:**

```bash
# Full visual demonstration
uv run python examples/sim_validation_demo.py

# Headless mode
uv run python examples/sim_validation_demo.py --no-viewer

# Fast-forward mode
uv run python examples/sim_validation_demo.py --no-realtime
```

**pytest Integration:**

```bash
# Run validation test suite
uv run pytest tests/test_validation_suite.py -v

# Run specific test category
uv run pytest tests/test_validation_suite.py::TestBasicMovements -v
```

## Test Categories

### 1. Basic Movements

Tests fundamental head movement capabilities:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Head pitch positive | Look down (20°) | pitch ≈ 20° |
| Head pitch negative | Look up (-20°) | pitch ≈ -20° |
| Head yaw left | Turn left (30°) | yaw ≈ 30° |
| Head yaw right | Turn right (-30°) | yaw ≈ -30° |
| Head roll left | Tilt left (15°) | roll ≈ 15° |
| Head roll right | Tilt right (-15°) | roll ≈ -15° |
| Combined movement | All axes | All angles within tolerance |
| Reset position | Return to neutral | All angles ≈ 0° |

### 2. Antenna Movements

Tests expressive antenna control:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Symmetric up | Both antennas 60° | left ≈ 60°, right ≈ 60° |
| Symmetric down | Both antennas -60° | left ≈ -60°, right ≈ -60° |
| Asymmetric | Left 45°, right -45° | Opposing positions |
| Extreme positions | Test limits | Within ±150° |

### 3. Body Rotation

Tests base rotation capabilities:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Clockwise 45° | Rotate CW | No errors |
| Counter-clockwise 45° | Rotate CCW | No errors |
| Full rotation 360° | Complete spin | No errors |

### 4. Gestures

Tests pre-defined gestures:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Nod (low) | Gentle agreement | Smooth pitch oscillation |
| Nod (medium) | Normal nod | Moderate amplitude |
| Nod (high) | Emphatic nod | Large amplitude |
| Shake (low) | Gentle disagreement | Smooth yaw oscillation |
| Shake (medium) | Normal shake | Moderate amplitude |
| Shake (high) | Emphatic shake | Large amplitude |

### 5. Motor Control

Tests wake/sleep functionality:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Wake up | Enable motors | is_awake = True |
| Sleep | Disable motors | is_awake = False |
| Status check | Read status | Valid status object |

### 6. Sensors

Tests sensor reading capabilities:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Get position | Read joint angles | Dict with all joints |
| Get limits | Read joint limits | Valid min/max pairs |
| Position consistency | Multiple reads | Values stable |

### 7. Look-At Behavior

Tests gaze targeting:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Look center | Target (1, 0, 0) | yaw ≈ 0°, pitch ≈ 0° |
| Look left | Target (1, 1, 0) | yaw > 0° |
| Look right | Target (1, -1, 0) | yaw < 0° |
| Look up | Target (1, 0, 1) | pitch < 0° |
| Look down | Target (1, 0, -1) | pitch > 0° |
| Track target | Moving target | Smooth tracking |

### 8. Choreography

Tests complex movement sequences:

| Test | Description | Duration |
|------|-------------|----------|
| Head sweep | Left-right scan | ~3s |
| Antenna wave | Alternating pattern | ~2s |
| Combined routine | Full choreography | ~30s |

### 9. Edge Cases

Tests boundary conditions:

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Joint limits | Move to limits | Clamped to safe range |
| Rapid commands | 10 commands fast | No errors |
| Zero duration | Very short move | Completes instantly |

## Validation Report Format

After running validation, you'll see a report like this:

```
╭─────────────────── Reachy Mini Validation Report ───────────────────╮
│ Test Suite: MuJoCo Simulation Validation                            │
│ Date: 2025-01-05 18:30:00                                           │
├──────────────────────────────────────────────────────────────────────┤
│ ✓ Basic Movements        [ 8/8 passed]                              │
│ ✓ Antennas              [ 4/4 passed]                              │
│ ✓ Gestures              [ 6/6 passed]                              │
│ ✓ Motor Control         [ 3/3 passed]                              │
│ ✓ Sensors               [ 3/3 passed]                              │
│ ✓ Look-At               [ 6/6 passed]                              │
│ ✓ Choreography          [ 1/1 passed]                              │
├──────────────────────────────────────────────────────────────────────┤
│ Total: 31/31 tests passed (100%)                                    │
│ Duration: 45.2s                                                     │
╰──────────────────────────────────────────────────────────────────────╯
```

## Interpreting Results

### Pass Criteria

A test passes if:
1. The command executes without raising an exception
2. The resulting position is within tolerance (default: 5°) of the target
3. The operation completes within the expected time

### Common Failures

| Failure Type | Possible Cause | Resolution |
|--------------|----------------|------------|
| Position mismatch | Physics instability | Increase duration |
| Timeout | Slow execution | Check CPU load |
| Import error | Missing dependencies | Run `uv pip install -e ".[sim]"` |
| Model not found | Missing MJCF | Check `data/models/` |

### Tolerance Configuration

Edit the validation config in `tests/test_validation_suite.py`:

```python
@dataclass
class ValidationConfig:
    position_tolerance: float = 5.0   # degrees
    timing_tolerance: float = 0.5     # seconds
    movement_duration: float = 0.5    # seconds
    pause_between_tests: float = 0.1  # seconds
```

## Visual Demo Features

The `sim_validation_demo.py` script provides:

### Startup Sequence
- Robot wakes up
- Resets to neutral position
- Status indicators displayed

### Basic Movements Demo
- Head pitch sweep: -30° to +30°
- Head yaw sweep: -60° to +60°
- Head roll sweep: -30° to +30°
- Body rotation: 360° spin
- Antenna wiggle pattern

### Gesture Showcase
- Nod at 3 intensities (low, medium, high)
- Shake at 3 intensities (low, medium, high)

### Emotion Library Demo
- Happy: Perky antennas, slight head tilt
- Sad: Droopy antennas, head down
- Surprised: Antennas up, head back
- Curious: Head tilt, asymmetric antennas
- Confused: Alternating movements
- Excited: Rapid oscillations
- Sleepy: Slow, droopy movements
- Attentive: Alert posture

### Choreographed Dance
- 30-second coordinated routine
- Combines all movement types
- Smooth transitions
- Dynamic antenna expressions

### Look-At Behavior
- Track points in 3D space
- Smooth pursuit of moving targets

### Sensor Validation
- Joint position readout
- Limit verification
- Status monitoring

### Shutdown Sequence
- Return to neutral
- Enter sleep mode

## Video Recording

To record a validation session:

```bash
# Using the demo script
uv run python examples/sim_validation_demo.py --record

# Recording is saved to:
# data/recordings/validation_YYYY-MM-DD_HHMMSS.mp4
```

Recording requires `imageio` and `imageio-ffmpeg` to be installed (included in `[sim]` extras).

## Continuous Integration

Add validation to your CI pipeline:

```yaml
# .github/workflows/validation.yml
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[sim,dev]"
      - name: Run validation
        run: |
          # Headless mode for CI
          reachy-agent validate --no-viewer --no-realtime
```

## Extending the Suite

### Adding New Tests

1. Add test function to appropriate class in `test_validation_suite.py`:

```python
@pytest.mark.asyncio
async def test_my_new_command(self, sim_client) -> None:
    """Test description."""
    await sim_client.my_command(param=value)
    result = await sim_client.get_result()
    assert result == expected
```

2. Add to demo script for visual verification:

```python
async def test_my_feature(self) -> None:
    async def my_test():
        await self.client.my_command(param=value)
    await self.run_test("My Feature", "category", my_test)
```

### Custom Validation Criteria

Override the validation config:

```python
CONFIG = ValidationConfig(
    position_tolerance=3.0,  # Stricter tolerance
    movement_duration=1.0,   # Slower movements
)
```

## Troubleshooting

### "MuJoCo not installed"

```bash
uv pip install -e ".[sim]"
```

### "Model not found"

Ensure the MJCF model exists:
```bash
ls data/models/reachy_mini/reachy_mini.xml
```

### "GL context error"

On headless servers, use:
```bash
reachy-agent validate --no-viewer
```

Or set:
```bash
export MUJOCO_GL=osmesa  # or egl
```

### "Slow performance"

Use fast-forward mode:
```bash
reachy-agent validate --no-realtime
```

## Related Documentation

- [Simulation Overview](../specs/11-simulation.md)
- [Robot Client API](../ai_docs/mcp-tools.md)
- [MuJoCo Model](../data/models/reachy_mini/README.md)
