# API Reference

Complete documentation for all MCP tools and Python APIs.

## MCP Tools

The agent exposes 23 MCP tools organized into two servers:

### Robot MCP Server (20 tools)

| Category | Tools | Description |
|----------|-------|-------------|
| **[Movement](robot-tools.md#movement-tools)** | `move_head`, `look_at`, `rotate_body`, `reset_position` | Control robot pose |
| **[Expression](robot-tools.md#expression-tools)** | `play_emotion`, `play_sequence`, `set_antennas`, `nod`, `shake` | Emotional expression |
| **[Audio](robot-tools.md#audio-tools)** | `speak`, `listen` | Voice I/O |
| **[Perception](robot-tools.md#perception-tools)** | `capture_image`, `get_sensor_data`, `detect_sound_direction` | Sensors |
| **[Lifecycle](robot-tools.md#lifecycle-tools)** | `wake_up`, `sleep_robot`, `is_awake` | Power state |
| **[Status](robot-tools.md#status-tools)** | `get_status`, `get_position`, `get_limits` | Robot state |

### Memory MCP Server (3 tools)

| Tool | Description |
|------|-------------|
| **[store_memory](memory-tools.md#store_memory)** | Store new semantic memory |
| **[search_memory](memory-tools.md#search_memory)** | Search memories with similarity |
| **[forget_memory](memory-tools.md#forget_memory)** | Delete specific memory |

## Python APIs

### Core Modules

| Module | Description |
|--------|-------------|
| **[Voice Pipeline](voice-pipeline.md)** | Event-driven voice interaction |
| **[Motion Control](motion-control.md)** | 30Hz blend controller |
| **[Memory Manager](memory-tools.md#python-api)** | ChromaDB memory operations |

### Client Interfaces

| Interface | Description |
|-----------|-------------|
| **[ReachyClient](python-api.md#reachyclient-protocol)** | Hardware abstraction protocol |
| **[MockClient](python-api.md#mockclient)** | Testing without hardware |
| **[MuJoCoReachyClient](python-api.md#mujocoreachyclient)** | Physics simulation |

## Quick Reference

### Tool Permission Tiers

| Tier | Behavior | Tools |
|------|----------|-------|
| **AUTONOMOUS** | Execute immediately | All `get_*`, `move_*`, `play_*`, `speak`, `listen` |
| **CONFIRM** | Ask user first | `store_memory`, `forget_memory` |
| **FORBIDDEN** | Never allowed | `exec_*`, `shell_*` |

### Hardware Limits

| Joint | Minimum | Maximum | Unit |
|-------|---------|---------|------|
| Head Pitch | -45 | +35 | degrees |
| Head Yaw | -60 | +60 | degrees |
| Head Roll | -35 | +35 | degrees |
| Head Z | 0 | 50 | mm |
| Body Rotation | 0 | 360 | degrees |
| Antennas | -150 | +150 | degrees |

### Memory Types

| Type | Expiry | Permission | Use Case |
|------|--------|------------|----------|
| `fact` | Never | CONFIRM | Permanent info |
| `conversation` | 30 days | CONFIRM | Dialog history |
| `context` | End of day | CONFIRM | Session data |

## Code Examples

### Basic Movement

```python
from reachy_agent.mcp.robot import move_head, look_at, rotate_body

# Move head to specific angles
result = await move_head(pitch=10, yaw=-30, roll=0, duration=1.0)

# Look at a point in 3D space
result = await look_at(x=0.5, y=-0.2, z=1.0, duration=1.0)

# Rotate the body
result = await rotate_body(angle=180, duration=2.0)
```

### Expression

```python
from reachy_agent.mcp.robot import play_emotion, play_sequence, nod

# Play single emotion
result = await play_emotion(emotion_name="happy")

# Play sequence of emotions
result = await play_sequence(
    emotions=["curious", "thinking", "happy"],
    delays=[0.5, 0.5]
)

# Gesture
result = await nod(intensity=1.0)
```

### Memory

```python
from reachy_agent.mcp.memory import store_memory, search_memory, forget_memory

# Store a fact
result = await store_memory(
    content="User's favorite color is blue",
    memory_type="fact",
    metadata={"category": "preferences"}
)

# Search memories
results = await search_memory(
    query="color preferences",
    memory_type="fact",
    limit=5
)

# Delete memory
result = await forget_memory(memory_id="abc123...")
```

### Voice Pipeline

```python
from reachy_agent.voice.pipeline import VoicePipeline

# Create and start pipeline
pipeline = VoicePipeline(agent, persona_manager, debug=True)
await pipeline.start()

# Handle events
@pipeline.on("wake_detected")
async def on_wake(event):
    print(f"Wake word detected: {event.data['persona']}")

@pipeline.on("transcribed")
async def on_transcribed(event):
    print(f"User said: {event.data['text']}")

# Stop pipeline
await pipeline.stop()
```

### Motion Control

```python
from reachy_agent.motion.controller import BlendController
from reachy_agent.motion.idle import IdleBehavior

# Create blend controller
controller = BlendController(client)
await controller.start()

# Set primary motion source
await controller.set_primary(IdleBehavior())

# Add overlay
await controller.add_overlay(SpeechWobble())

# Stop
await controller.stop()
```

## Error Handling

All tools return structured responses:

### Success Response
```python
{
    "success": True,
    "position": {"pitch": 10, "yaw": -30, ...}
}
```

### Error Response
```python
{
    "success": False,
    "error": "Robot is not awake"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `NotAwakeError` | Motors disabled | Call `wake_up()` first |
| `ConnectionError` | Robot disconnected | Check hardware connection |
| `PermissionDenied` | FORBIDDEN tool | Cannot use this tool |
| `ValidationError` | Invalid parameters | Check parameter ranges |

## API Documentation

- **[Robot Tools](robot-tools.md)** - Complete robot MCP tool reference
- **[Memory Tools](memory-tools.md)** - Memory MCP tool reference
- **[Voice Pipeline](voice-pipeline.md)** - Voice system API
- **[Motion Control](motion-control.md)** - Motion system API
- **[Python API](python-api.md)** - Core Python interfaces
