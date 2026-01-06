# Reachy SDK Documentation Quick Reference

**Source:** https://docs.pollen-robotics.com/ and https://pollen-robotics.github.io/reachy-sdk
**Note:** This is a quick reference guide. For complete documentation, visit the official docs.
**SDK Version:** 1.0.0+
**Developer:** Pollen Robotics

---

## Installation

```bash
pip install reachy-sdk
```

For development:
```bash
pip install reachy-sdk[dev]
```

---

## Quick Start

### Connecting to Reachy

```python
from reachy_sdk import ReachySDK

# Connect to Reachy (default: localhost)
reachy = ReachySDK(host='localhost')

# Check connection
print(f"Connected to Reachy: {reachy.is_connected()}")

# Always disconnect when done
reachy.disconnect()
```

### Context Manager Usage (Recommended)

```python
from reachy_sdk import ReachySDK

with ReachySDK(host='192.168.1.42') as reachy:
    # Your robot control code here
    print(f"Reachy is connected: {reachy.is_connected()}")
    # Automatic disconnect when exiting context
```

---

## Core Components

### Robot Structure

The Reachy robot has the following main components accessible via the SDK:

```python
from reachy_sdk import ReachySDK

reachy = ReachySDK(host='localhost')

# Access robot parts
head = reachy.head
l_arm = reachy.l_arm  # Left arm
r_arm = reachy.r_arm  # Right arm
```

### Available Modules

Based on the SDK structure, the following modules are available:

- **Arms** - Left and right arm control
- **Head** - Head movement (pan, tilt)
- **Camera** - Camera access and image capture
- **Joints** - Individual joint control
- **Force Sensor** - Force sensing capabilities
- **Fan** - Cooling system control
- **Device Holder** - Mobile base or holder attachment

---

## Joint Control

### Reading Joint Positions

```python
with ReachySDK(host='localhost') as reachy:
    # Access specific joints
    shoulder_pitch = reachy.r_arm.shoulder_pitch

    # Read current position
    current_pos = shoulder_pitch.present_position
    print(f"Shoulder pitch position: {current_pos}°")

    # Read all arm joint positions
    for joint in reachy.r_arm.joints.values():
        print(f"{joint.name}: {joint.present_position}°")
```

### Setting Joint Positions

```python
with ReachySDK(host='localhost') as reachy:
    # Enable compliance mode (stiff/compliant)
    reachy.r_arm.shoulder_pitch.compliant = False  # Stiff (can be controlled)

    # Set target position
    reachy.r_arm.shoulder_pitch.goal_position = 45.0  # degrees

    # Set multiple joints
    reachy.r_arm.shoulder_pitch.goal_position = 30.0
    reachy.r_arm.elbow_pitch.goal_position = -90.0
```

### Compliance Control

```python
# Make arm compliant (loose, can be moved by hand)
reachy.r_arm.shoulder_pitch.compliant = True

# Make arm stiff (holds position, can be controlled)
reachy.r_arm.shoulder_pitch.compliant = False

# Set compliance for entire arm
for joint in reachy.r_arm.joints.values():
    joint.compliant = False  # Stiffen all joints
```

---

## Motion Control

### Goto Function (Smooth Motion)

```python
import time

with ReachySDK(host='localhost') as reachy:
    # Prepare for motion
    for joint in reachy.r_arm.joints.values():
        joint.compliant = False

    # Define target positions for all joints
    target_positions = {
        'shoulder_pitch': 30.0,
        'shoulder_roll': 0.0,
        'arm_yaw': 0.0,
        'elbow_pitch': -90.0,
        'forearm_yaw': 0.0,
        'wrist_pitch': 0.0,
        'wrist_roll': 0.0,
    }

    # Send goto command
    reachy.r_arm.goto(
        goal_positions=target_positions,
        duration=2.0,  # 2 seconds to reach target
        interpolation_mode='minimum_jerk'  # Smooth trajectory
    )

    # Wait for motion to complete
    time.sleep(2.0)
```

### Interpolation Modes

- `'minimum_jerk'` - Smooth, natural motion (default)
- `'linear'` - Constant velocity
- Other modes may be available depending on SDK version

---

## Head Control

### Moving the Head

```python
with ReachySDK(host='localhost') as reachy:
    # Access head joints
    head_pan = reachy.head.neck_roll  # Left/right
    head_tilt = reachy.head.neck_pitch  # Up/down

    # Set head to look at specific angle
    head_pan.compliant = False
    head_tilt.compliant = False

    head_pan.goal_position = 30.0  # Look right
    head_tilt.goal_position = -15.0  # Look down
```

### Head Look-At Function

```python
# Look at a specific point in space
reachy.head.look_at(
    x=0.5,  # meters forward
    y=0.0,  # meters left/right
    z=0.0,  # meters up/down
    duration=1.0
)
```

---

## Camera Access

### Capturing Images

```python
import cv2

with ReachySDK(host='localhost') as reachy:
    # Get camera image
    image = reachy.head.left_camera.get_image()

    # Display or save image
    cv2.imshow('Reachy Vision', image)
    cv2.waitKey(1)

    # Or save to file
    cv2.imwrite('reachy_view.jpg', image)
```

### Camera Properties

```python
# Access camera properties
left_cam = reachy.head.left_camera
right_cam = reachy.head.right_camera

# Get resolution, FPS, etc.
print(f"Camera resolution: {left_cam.width}x{left_cam.height}")
```

---

## Force Sensors

### Reading Force Data

```python
with ReachySDK(host='localhost') as reachy:
    # Read force sensor values
    force = reachy.r_arm.force_sensor.read()

    print(f"Force X: {force.fx}")
    print(f"Force Y: {force.fy}")
    print(f"Force Z: {force.fz}")
    print(f"Torque X: {force.tx}")
    print(f"Torque Y: {force.ty}")
    print(f"Torque Z: {force.tz}")
```

---

## Safety & Best Practices

### Emergency Stop

```python
# Make all joints compliant immediately (emergency stop)
for joint in reachy.r_arm.joints.values():
    joint.compliant = True

for joint in reachy.l_arm.joints.values():
    joint.compliant = True
```

### Safe Initialization Pattern

```python
from reachy_sdk import ReachySDK
import time

def safe_move(reachy, arm, positions, duration=2.0):
    """Safely move arm to target positions."""
    try:
        # Stiffen joints
        for joint in arm.joints.values():
            joint.compliant = False

        time.sleep(0.1)  # Brief delay for stiffening

        # Execute motion
        arm.goto(
            goal_positions=positions,
            duration=duration,
            interpolation_mode='minimum_jerk'
        )

        time.sleep(duration)

    except Exception as e:
        print(f"Error during motion: {e}")
        # Make compliant on error
        for joint in arm.joints.values():
            joint.compliant = True
        raise

# Usage
with ReachySDK(host='localhost') as reachy:
    safe_move(
        reachy,
        reachy.r_arm,
        {'shoulder_pitch': 30.0, 'elbow_pitch': -90.0},
        duration=2.0
    )
```

### Connection Check

```python
# Always verify connection before operations
with ReachySDK(host='192.168.1.42') as reachy:
    if not reachy.is_connected():
        raise ConnectionError("Failed to connect to Reachy")

    # Proceed with operations
    print("Reachy is ready!")
```

---

## Mock Mode for Development

The Reachy SDK likely supports a mock mode for development without hardware:

```python
# Potential mock mode usage (verify with official docs)
from reachy_sdk import ReachySDK

# Connect to mock/simulated robot
reachy = ReachySDK(host='localhost', mock=True)
```

---

## Common Patterns for Embodied AI

### Gesture Execution

```python
def execute_gesture(reachy, gesture_name: str):
    """Execute predefined gestures."""
    gestures = {
        'wave': {
            'shoulder_pitch': 0.0,
            'shoulder_roll': -30.0,
            'elbow_pitch': -90.0,
            'wrist_pitch': 0.0,
        },
        'point': {
            'shoulder_pitch': 30.0,
            'shoulder_roll': 0.0,
            'elbow_pitch': -45.0,
            'wrist_pitch': -30.0,
        }
    }

    if gesture_name not in gestures:
        raise ValueError(f"Unknown gesture: {gesture_name}")

    reachy.r_arm.goto(
        goal_positions=gestures[gesture_name],
        duration=1.5,
        interpolation_mode='minimum_jerk'
    )
```

### Look at Speaker

```python
def look_at_speaker(reachy, x: float, y: float, z: float):
    """Orient head toward speaker location."""
    reachy.head.look_at(x=x, y=y, z=z, duration=0.8)
```

### Reactive Behaviors

```python
import time

def reactive_loop(reachy):
    """Monitor sensors and react to environment."""
    while True:
        # Read force sensor
        force = reachy.r_arm.force_sensor.read()

        # React to touch
        if abs(force.fz) > 5.0:  # Threshold in Newtons
            print("Detected touch!")
            # Respond with gesture or speech

        time.sleep(0.1)  # 10Hz update rate
```

---

## Performance Considerations

### Update Rate

```python
# Typical control loop for 30Hz updates
import time

CONTROL_FREQ = 30  # Hz
dt = 1.0 / CONTROL_FREQ

while running:
    start_time = time.time()

    # Read sensors
    # Update control
    # Send commands

    # Maintain loop rate
    elapsed = time.time() - start_time
    if elapsed < dt:
        time.sleep(dt - elapsed)
```

### Batch Joint Updates

```python
# More efficient to update multiple joints at once
target_positions = {
    'shoulder_pitch': 30.0,
    'shoulder_roll': 0.0,
    'elbow_pitch': -90.0,
    # ... more joints
}

# Single goto call vs multiple individual updates
reachy.r_arm.goto(goal_positions=target_positions, duration=2.0)
```

---

## Troubleshooting

### Connection Issues

```python
# Verify network connection
import socket

def check_reachy_connection(host: str, port: int = 50055) -> bool:
    """Check if Reachy is reachable on network."""
    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        return True
    except (socket.timeout, socket.error):
        return False

# Usage
if check_reachy_connection('192.168.1.42'):
    reachy = ReachySDK(host='192.168.1.42')
else:
    print("Cannot reach Reachy at specified address")
```

### Joint Limits

```python
# Always respect joint limits
JOINT_LIMITS = {
    'shoulder_pitch': (-180, 90),
    'shoulder_roll': (-180, 10),
    'elbow_pitch': (-135, 0),
    # Add other joint limits
}

def safe_position(joint_name: str, target: float) -> float:
    """Clamp target position to safe limits."""
    min_val, max_val = JOINT_LIMITS.get(joint_name, (-180, 180))
    return max(min_val, min(max_val, target))
```

---

## Additional Resources

### Official Documentation
- Main Docs: https://docs.pollen-robotics.com/
- SDK Reference: https://pollen-robotics.github.io/reachy-sdk
- GitHub: https://github.com/pollen-robotics/reachy-sdk

### Community
- Pollen Robotics Forum: https://forum.pollen-robotics.com/
- Discord/Slack: Check Pollen Robotics website for links

### Example Projects
- Check the `examples/` directory in the SDK repository
- Community projects on GitHub

---

## Summary

The Reachy SDK provides:
- ✅ Simple connection management with context managers
- ✅ Joint-level control (position, compliance)
- ✅ Smooth motion with goto and interpolation
- ✅ Head control and camera access
- ✅ Force sensor integration
- ✅ Python 3.7+ compatibility
- ✅ Mock mode for development

**Key Points for Embodied AI:**
- Use 30Hz control loop for responsive behavior
- Always check compliance before moving joints
- Use context managers for safe connection handling
- Implement emergency stop functionality
- Test with mock mode before hardware deployment

**Note:** This is a quick reference based on available information. For complete, accurate, and up-to-date documentation, always refer to the official Pollen Robotics documentation at https://docs.pollen-robotics.com/
