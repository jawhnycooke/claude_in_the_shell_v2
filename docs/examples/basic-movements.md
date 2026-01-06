# Basic Movement Examples

Working examples for controlling robot movements.

## Head Control

### Move to Absolute Position

```python
import asyncio
from reachy_agent.robot.factory import create_client, Backend

async def head_absolute():
    """Move head to absolute positions."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Look up
    await client.move_head(pitch=30, yaw=0, roll=0, duration=1.0)

    # Look left
    await client.move_head(pitch=0, yaw=-45, roll=0, duration=1.0)

    # Look down-right with tilt
    await client.move_head(pitch=-20, yaw=30, roll=15, duration=1.0)

    # Return to center
    await client.move_head(pitch=0, yaw=0, roll=0, duration=1.0)

    await client.disconnect()

asyncio.run(head_absolute())
```

### Look at 3D Point

```python
async def look_at_point():
    """Look at specific 3D coordinates."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Look at a point 1 meter ahead, slightly down
    await client.look_at(x=1.0, y=0.0, z=-0.2, duration=1.0)

    # Look at a point to the left
    await client.look_at(x=0.8, y=-0.5, z=0.0, duration=1.0)

    # Look at a point to the right and up
    await client.look_at(x=0.5, y=0.3, z=0.3, duration=1.0)

    await client.disconnect()

asyncio.run(look_at_point())
```

### Head Scanning Pattern

```python
import math

async def scan_pattern():
    """Create a scanning pattern."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Horizontal scan
    for yaw in range(-45, 46, 15):
        await client.move_head(pitch=0, yaw=yaw, duration=0.3)

    # Return to center
    await client.move_head(pitch=0, yaw=0, duration=0.5)

    # Vertical scan
    for pitch in range(-30, 31, 10):
        await client.move_head(pitch=pitch, yaw=0, duration=0.3)

    # Return to center
    await client.reset_position()

    await client.disconnect()

asyncio.run(scan_pattern())
```

---

## Body Rotation

### Basic Rotation

```python
async def body_rotation():
    """Rotate the body."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Rotate 90 degrees clockwise
    await client.rotate_body(angle=90, duration=1.5)

    # Rotate to 180 degrees (face back)
    await client.rotate_body(angle=180, duration=1.5)

    # Return to front
    await client.rotate_body(angle=0, duration=1.5)

    await client.disconnect()

asyncio.run(body_rotation())
```

### Coordinated Head and Body

```python
async def coordinated_movement():
    """Move head and body together."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Look left while rotating right
    # (creates interesting tracking effect)
    await asyncio.gather(
        client.move_head(yaw=-30, duration=2.0),
        client.rotate_body(angle=45, duration=2.0)
    )

    # Reset both
    await asyncio.gather(
        client.reset_position(duration=1.0),
        client.rotate_body(angle=0, duration=1.0)
    )

    await client.disconnect()

asyncio.run(coordinated_movement())
```

---

## Antenna Control

### Basic Antenna Positions

```python
async def antenna_positions():
    """Control antenna positions."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Both up (alert)
    await client.set_antennas(left=90, right=90)
    await asyncio.sleep(0.5)

    # Both down (sad)
    await client.set_antennas(left=-90, right=-90)
    await asyncio.sleep(0.5)

    # Asymmetric (curious)
    await client.set_antennas(left=60, right=-30)
    await asyncio.sleep(0.5)

    # Neutral
    await client.set_antennas(left=0, right=0)

    await client.disconnect()

asyncio.run(antenna_positions())
```

### Animated Antennas

```python
async def animated_antennas():
    """Create antenna animations."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Wiggle animation
    for _ in range(3):
        await client.set_antennas(left=45, right=-45)
        await asyncio.sleep(0.2)
        await client.set_antennas(left=-45, right=45)
        await asyncio.sleep(0.2)

    # Wave animation
    for i in range(20):
        angle = math.sin(i * 0.5) * 60
        await client.set_antennas(left=angle, right=-angle)
        await asyncio.sleep(0.1)

    # Return to neutral
    await client.set_antennas(left=0, right=0)

    await client.disconnect()

asyncio.run(animated_antennas())
```

---

## Gesture Sequences

### Nod Yes

```python
async def nod_gesture():
    """Perform nodding gesture."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Use built-in nod
    await client.nod(intensity=1.0)

    # Or manual nod sequence
    for _ in range(2):
        await client.move_head(pitch=15, duration=0.2)
        await client.move_head(pitch=-5, duration=0.2)

    await client.reset_position()
    await client.disconnect()

asyncio.run(nod_gesture())
```

### Shake No

```python
async def shake_gesture():
    """Perform head shake gesture."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Use built-in shake
    await client.shake(intensity=1.0)

    # Or manual shake sequence
    for _ in range(2):
        await client.move_head(yaw=20, duration=0.15)
        await client.move_head(yaw=-20, duration=0.15)

    await client.reset_position()
    await client.disconnect()

asyncio.run(shake_gesture())
```

### Curious Tilt

```python
async def curious_tilt():
    """Perform curious head tilt."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Tilt with antenna expression
    await asyncio.gather(
        client.move_head(pitch=5, yaw=-10, roll=-15, duration=0.5),
        client.set_antennas(left=30, right=-20)
    )

    await asyncio.sleep(1.0)

    # Reset
    await asyncio.gather(
        client.reset_position(),
        client.set_antennas(left=0, right=0)
    )

    await client.disconnect()

asyncio.run(curious_tilt())
```

---

## Emotion Expressions

### Play Built-in Emotions

```python
async def play_emotions():
    """Play built-in emotion animations."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    emotions = ["happy", "sad", "curious", "surprised", "thinking"]

    for emotion in emotions:
        print(f"Playing: {emotion}")
        await client.play_emotion(emotion)
        await asyncio.sleep(1.0)

    await client.reset_position()
    await client.disconnect()

asyncio.run(play_emotions())
```

### Emotion Sequence

```python
async def emotion_sequence():
    """Play a sequence of emotions."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Thinking → Curious → Happy sequence
    await client.play_sequence(
        emotions=["thinking", "curious", "happy"],
        delays=[0.5, 0.5, 0.5]
    )

    await client.disconnect()

asyncio.run(emotion_sequence())
```

---

## Complete Movement Demo

```python
import asyncio
from reachy_agent.robot.factory import create_client, Backend

async def full_demo():
    """Complete movement demonstration."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    print("Starting movement demo...")

    # 1. Wake up sequence
    print("1. Wake up")
    await client.move_head(pitch=20, duration=0.5)
    await client.move_head(pitch=0, duration=0.5)
    await client.set_antennas(left=45, right=45)
    await asyncio.sleep(0.3)
    await client.set_antennas(left=0, right=0)

    # 2. Look around
    print("2. Looking around")
    await client.move_head(yaw=-45, duration=0.8)
    await asyncio.sleep(0.3)
    await client.move_head(yaw=45, duration=1.0)
    await asyncio.sleep(0.3)
    await client.move_head(yaw=0, duration=0.6)

    # 3. Express curiosity
    print("3. Curious expression")
    await client.play_emotion("curious")
    await asyncio.sleep(0.5)

    # 4. Nod agreement
    print("4. Nodding")
    await client.nod(intensity=1.0)

    # 5. Body rotation
    print("5. Body rotation")
    await client.rotate_body(angle=180, duration=2.0)
    await asyncio.sleep(0.5)
    await client.rotate_body(angle=0, duration=2.0)

    # 6. Happy expression
    print("6. Happy expression")
    await client.play_emotion("happy")

    # 7. Return to neutral
    print("7. Return to neutral")
    await client.reset_position()

    print("Demo complete!")
    await client.disconnect()

asyncio.run(full_demo())
```

---

## Running the Examples

Save any example to a file and run:

```bash
# Activate environment
source .venv/bin/activate

# Run example
python my_example.py
```

For simulation with visualization:

```python
# Change Backend.MOCK to use simulation
from reachy_agent.simulation import MuJoCoReachyClient

async def with_simulation():
    client = MuJoCoReachyClient(realtime=True, viewer=True)
    await client.connect()
    # ... rest of code
```

---

## Next Steps

- [Voice Commands](voice-commands.md) - Add voice interaction
- [Simulation Scenarios](simulation-scenarios.md) - Test in simulation
- [Custom Behaviors](../tutorials/custom-behaviors.md) - Create custom behaviors
