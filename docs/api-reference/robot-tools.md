# Robot MCP Tools Reference

Complete documentation for all 20 robot control MCP tools.

## Overview

The Robot MCP Server provides tools for controlling the Reachy Mini robot. All tools communicate via the Reachy SDK (Zenoh) with 1-5ms latency.

### Tool Categories

| Category | Count | Description |
|----------|-------|-------------|
| Movement | 4 | Head and body positioning |
| Expression | 5 | Emotions, antennas, gestures |
| Audio | 2 | Speech input/output |
| Perception | 3 | Camera, sensors, sound |
| Lifecycle | 3 | Power management |
| Status | 3 | Robot state queries |

---

## Movement Tools

### move_head

Move the head to an absolute position.

```python
async def move_head(
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    duration: float = 1.0,
) -> dict
```

**Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `pitch` | float | -45 to +35 | Head tilt up/down (degrees) |
| `yaw` | float | -60 to +60 | Head turn left/right (degrees) |
| `roll` | float | -35 to +35 | Head tilt sideways (degrees) |
| `duration` | float | 0.1 to 10.0 | Movement time (seconds) |

**Returns:**
```python
{
    "success": True,
    "position": {
        "pitch": 10.0,
        "yaw": -30.0,
        "roll": 0.0,
        "z": 0.0,
        "body_rotation": 0.0,
        "antenna_left": 0.0,
        "antenna_right": 0.0
    }
}
```

**Example:**
```python
# Look up and to the left
result = await move_head(pitch=20, yaw=-40, roll=0, duration=1.5)
```

**Permission:** AUTONOMOUS

---

### look_at

Look at a 3D point in the robot's coordinate frame.

```python
async def look_at(
    x: float,
    y: float,
    z: float,
    duration: float = 1.0,
) -> dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | float | Left/right distance in meters (+ = right) |
| `y` | float | Up/down distance in meters (+ = down) |
| `z` | float | Forward distance in meters (+ = forward) |
| `duration` | float | Movement time (seconds) |

**Coordinate System:**
```
        +Y (down)
         │
         │
         └───── +X (right)
        /
       /
      +Z (forward)
```

**Returns:**
```python
{"success": True}
```

**Example:**
```python
# Look at a point 1 meter ahead and slightly up
result = await look_at(x=0.0, y=-0.2, z=1.0, duration=1.0)
```

**Permission:** AUTONOMOUS

---

### rotate_body

Rotate the body base to a specific angle.

```python
async def rotate_body(
    angle: float,
    duration: float = 1.0,
) -> dict
```

**Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `angle` | float | 0 to 360 | Target angle (degrees) |
| `duration` | float | 0.1 to 10.0 | Movement time (seconds) |

**Returns:**
```python
{
    "success": True,
    "body_rotation": 180.0
}
```

**Example:**
```python
# Turn around 180 degrees
result = await rotate_body(angle=180, duration=2.0)
```

**Note:** Body rotation is continuous (0° = 360°).

**Permission:** AUTONOMOUS

---

### reset_position

Return to neutral pose (head centered, body at 0°).

```python
async def reset_position(duration: float = 1.0) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration` | float | 1.0 | Movement time (seconds) |

**Returns:**
```python
{"success": True}
```

**Example:**
```python
result = await reset_position(duration=1.5)
```

**Permission:** AUTONOMOUS

---

## Expression Tools

### play_emotion

Play a predefined emotion animation.

```python
async def play_emotion(emotion_name: str) -> dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `emotion_name` | str | Name of the emotion to play |

**Available Emotions:**

| Category | Emotions |
|----------|----------|
| Basic | `happy`, `sad`, `curious`, `surprised`, `confused`, `tired` |
| Social | `greeting`, `farewell`, `acknowledgment`, `thinking` |
| Reactions | `yes`, `no`, `maybe`, `excitement`, `disappointment` |
| Complex | `interested`, `skeptical`, `amused`, `concerned` |

**Returns:**
```python
{
    "success": True,
    "emotion": "happy"
}
```

**Example:**
```python
result = await play_emotion(emotion_name="curious")
```

**Permission:** AUTONOMOUS

---

### play_sequence

Play a sequence of emotions with delays between them.

```python
async def play_sequence(
    emotions: list[str],
    delays: list[float] | None = None,
) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `emotions` | list[str] | (required) | List of emotion names |
| `delays` | list[float] | [0.5, ...] | Delays between emotions (seconds) |

**Returns:**
```python
{
    "success": True,
    "emotions_played": ["curious", "thinking", "happy"]
}
```

**Example:**
```python
result = await play_sequence(
    emotions=["curious", "thinking", "happy"],
    delays=[0.5, 0.3]  # Wait 0.5s after curious, 0.3s after thinking
)
```

**Permission:** AUTONOMOUS

---

### set_antennas

Set antenna positions for expression.

```python
async def set_antennas(left: float, right: float) -> dict
```

**Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `left` | float | -150 to +150 | Left antenna angle (degrees) |
| `right` | float | -150 to +150 | Right antenna angle (degrees) |

**Common Positions:**

| Expression | Left | Right |
|------------|------|-------|
| Neutral | 0° | 0° |
| Excited | +60° | +60° |
| Sad | -90° | -90° |
| Curious | +45° | +30° |
| Confused | -30° | +45° |

**Returns:**
```python
{
    "success": True,
    "left": 60.0,
    "right": 60.0
}
```

**Example:**
```python
# Excited antennas
result = await set_antennas(left=60, right=60)
```

**Permission:** AUTONOMOUS

---

### nod

Perform a nodding gesture (affirmative).

```python
async def nod(intensity: float = 1.0) -> dict
```

**Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `intensity` | float | 0.0 to 2.0 | Motion amplitude multiplier |

**Returns:**
```python
{"success": True}
```

**Example:**
```python
# Enthusiastic nod
result = await nod(intensity=1.5)
```

**Permission:** AUTONOMOUS

---

### shake

Perform a head shake gesture (negative).

```python
async def shake(intensity: float = 1.0) -> dict
```

**Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `intensity` | float | 0.0 to 2.0 | Motion amplitude multiplier |

**Returns:**
```python
{"success": True}
```

**Example:**
```python
# Emphatic head shake
result = await shake(intensity=1.5)
```

**Permission:** AUTONOMOUS

---

## Audio Tools

### speak

Convert text to speech and play through robot speaker.

```python
async def speak(text: str, voice: str = "default") -> dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | str | Text to speak |
| `voice` | str | Voice preset |

**Voice Options:**

| Voice | Description |
|-------|-------------|
| `default` | System default |
| `alloy` | Neutral, balanced |
| `echo` | Male, warm |
| `fable` | Expressive |
| `nova` | Female, professional |
| `onyx` | Male, deep |
| `shimmer` | Female, gentle |

**Returns:**
```python
{
    "success": True,
    "spoken": "Hello, how are you?"  # Truncated if > 50 chars
}
```

**Example:**
```python
result = await speak(text="Hello, how can I help you?", voice="nova")
```

**Permission:** AUTONOMOUS

---

### listen

Listen for speech and return transcription.

```python
async def listen(timeout: float = 5.0) -> dict
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | float | 5.0 | Maximum listen time (seconds) |

**Returns:**
```python
{
    "success": True,
    "transcription": "What's the weather like?"
}
```

**Example:**
```python
result = await listen(timeout=10.0)
print(f"User said: {result['transcription']}")
```

**Permission:** AUTONOMOUS

---

## Perception Tools

### capture_image

Capture a frame from the robot's camera.

```python
async def capture_image() -> dict
```

**Returns:**
```python
{
    "success": True,
    "image_bytes": 245760,  # Size in bytes
    "format": "png"
}
```

**Example:**
```python
result = await capture_image()
# In production, image data would be base64 encoded
```

**Permission:** AUTONOMOUS

---

### get_sensor_data

Read IMU/accelerometer sensor data.

```python
async def get_sensor_data() -> dict
```

**Returns:**
```python
{
    "success": True,
    "sensors": {
        "accelerometer": {"x": 0.1, "y": 0.0, "z": 9.8},
        "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
        "temperature": 25.5
    }
}
```

**Example:**
```python
result = await get_sensor_data()
accel = result["sensors"]["accelerometer"]
print(f"Acceleration: ({accel['x']}, {accel['y']}, {accel['z']})")
```

**Permission:** AUTONOMOUS

---

### detect_sound_direction

Get direction of loudest sound using mic array.

```python
async def detect_sound_direction() -> dict
```

**Returns:**
```python
{
    "success": True,
    "azimuth_degrees": 45.0,   # -180 to +180
    "confidence": 0.85          # 0 to 1
}
```

**Example:**
```python
result = await detect_sound_direction()
if result["confidence"] > 0.7:
    await look_at_direction(result["azimuth_degrees"])
```

**Permission:** AUTONOMOUS

---

## Lifecycle Tools

### wake_up

Enable motors and prepare for movement.

```python
async def wake_up() -> dict
```

**Returns:**
```python
{
    "success": True,
    "awake": True
}
```

**Example:**
```python
result = await wake_up()
# Now safe to issue movement commands
```

**Permission:** AUTONOMOUS

---

### sleep_robot

Disable motors and enter low power mode.

```python
async def sleep_robot() -> dict
```

**Returns:**
```python
{
    "success": True,
    "awake": False
}
```

**Example:**
```python
result = await sleep_robot()
# Robot is now in low power mode
```

**Note:** Movement commands will fail while asleep.

**Permission:** AUTONOMOUS

---

### is_awake

Check if motors are enabled.

```python
async def is_awake() -> dict
```

**Returns:**
```python
{"awake": True}
```

**Example:**
```python
result = await is_awake()
if not result["awake"]:
    await wake_up()
```

**Caching:** Results cached for 200ms.

**Permission:** AUTONOMOUS

---

## Status Tools

### get_status

Get comprehensive robot status.

```python
async def get_status() -> dict
```

**Returns:**
```python
{
    "success": True,
    "is_awake": True,
    "battery_percent": 85.5,
    "head_pose": {
        "pitch": 10.0,
        "yaw": -30.0,
        "roll": 0.0,
        "z": 5.0
    },
    "body_angle": 0.0,
    "antennas": {
        "left": 15.0,
        "right": 12.0
    }
}
```

**Caching:** Results cached for 200ms.

**Permission:** AUTONOMOUS

---

### get_position

Get current joint positions.

```python
async def get_position() -> dict
```

**Returns:**
```python
{
    "success": True,
    "position": {
        "pitch": 10.0,
        "yaw": -30.0,
        "roll": 0.0,
        "z": 5.0,
        "body_rotation": 0.0,
        "antenna_left": 15.0,
        "antenna_right": 12.0
    }
}
```

**Caching:** Results cached for 200ms.

**Permission:** AUTONOMOUS

---

### get_limits

Get joint angle limits.

```python
async def get_limits() -> dict
```

**Returns:**
```python
{
    "success": True,
    "limits": {
        "pitch": {"min": -45, "max": 35},
        "yaw": {"min": -60, "max": 60},
        "roll": {"min": -35, "max": 35},
        "z": {"min": 0, "max": 50},
        "body_rotation": {"min": 0, "max": 360},
        "antenna": {"min": -150, "max": 150}
    }
}
```

**Caching:** Results cached for 60 seconds (limits are static).

**Permission:** AUTONOMOUS

---

## Tool Caching

Read-only tools cache results to avoid redundant SDK calls:

| Tool | Cache TTL | Invalidated By |
|------|-----------|----------------|
| `is_awake` | 200ms | Wake/sleep operations |
| `get_status` | 200ms | Any movement tool |
| `get_position` | 200ms | Any movement tool |
| `get_limits` | 60s | Never |

## Error Handling

All tools may return errors:

```python
{
    "success": False,
    "error": "Robot is not awake. Call wake_up() first."
}
```

**Common Errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| `NotAwakeError` | Motors disabled | Call `wake_up()` |
| `ConnectionError` | SDK disconnected | Check hardware |
| `TimeoutError` | Operation too slow | Reduce duration |
| `LimitError` | Out of range | Check joint limits |
