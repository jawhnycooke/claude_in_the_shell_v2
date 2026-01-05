# Robot Control Specification

## Overview

The Robot MCP Server exposes 20 tools for controlling the Reachy Mini robot. All communication happens through the Reachy SDK (Zenoh) with no HTTP fallback - if the SDK is unavailable, we fail fast with clear errors.

## Design Principles

1. **SDK-only** - No HTTP fallback. Zenoh provides 1-5ms latency; HTTP at 10-50ms is too slow for real-time motion anyway.
2. **Unified client** - Single `ReachyClient` interface abstracts the backend.
3. **Tool caching** - Read-only tools cache results for 200ms to avoid redundant hardware queries.
4. **Fail fast** - Clear errors beat silent degradation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Robot MCP Server                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    20 MCP Tools                           │   │
│  │  Movement │ Expression │ Audio │ Perception │ Lifecycle   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Tool Result Cache                       │   │
│  │              (200ms TTL for read-only tools)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ReachyClient                           │   │
│  │              (Unified interface to SDK)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Reachy Mini SDK (Zenoh)                       │
│                       1-5ms latency                              │
└─────────────────────────────────────────────────────────────────┘
```

## MCP Tools (20 total)

### Movement Tools (4)

| Tool | Description | Parameters | Permission |
|------|-------------|------------|------------|
| `move_head` | Move head to absolute position | `pitch`, `yaw`, `roll`, `duration` | AUTONOMOUS |
| `look_at` | Look at 3D point in robot frame | `x`, `y`, `z`, `duration` | AUTONOMOUS |
| `rotate_body` | Rotate body to angle | `angle`, `duration` | AUTONOMOUS |
| `reset_position` | Return to neutral pose | `duration` | AUTONOMOUS |

**Removed from v1**: `look_at_world` (redundant with `look_at`), `look_at_pixel` (rarely used, camera-dependent)

### Expression Tools (5)

| Tool | Description | Parameters | Permission |
|------|-------------|------------|------------|
| `play_emotion` | Play emotion animation | `emotion_name` | AUTONOMOUS |
| `play_sequence` | Play emotion sequence | `emotions[]`, `delays[]` | AUTONOMOUS |
| `set_antennas` | Set antenna positions | `left_angle`, `right_angle` | AUTONOMOUS |
| `nod` | Nod head yes | `intensity` | AUTONOMOUS |
| `shake` | Shake head no | `intensity` | AUTONOMOUS |

**New in v2**: `play_sequence` for compound expressions (e.g., `["curious", "happy", "nod"]`)

**Removed from v1**: `dance` (merged into `play_emotion` - dances are just longer emotions)

### Audio Tools (2)

| Tool | Description | Parameters | Permission |
|------|-------------|------------|------------|
| `speak` | Text-to-speech output | `text`, `voice` | AUTONOMOUS |
| `listen` | Listen for speech, return transcription | `timeout_seconds` | AUTONOMOUS |

### Perception Tools (3)

| Tool | Description | Parameters | Permission |
|------|-------------|------------|------------|
| `capture_image` | Capture camera frame | `format` | AUTONOMOUS |
| `get_sensor_data` | Read IMU/accelerometer | - | AUTONOMOUS |
| `detect_sound_direction` | Get direction of loudest sound | - | AUTONOMOUS |

**Removed from v1**: `look_at_sound` (was convenience wrapper - use `detect_sound_direction` + `look_at`)

### Lifecycle Tools (3)

| Tool | Description | Parameters | Permission |
|------|-------------|------------|------------|
| `wake_up` | Enable motors, stand ready | - | AUTONOMOUS |
| `sleep` | Disable motors, low power | - | AUTONOMOUS |
| `is_awake` | Check if robot is active | - | AUTONOMOUS |

**Removed from v1**: `rest` (confusing distinction from `sleep`)

### Status Tools (3)

| Tool | Description | Parameters | Permission |
|------|-------------|------------|------------|
| `get_status` | Robot health and state | - | AUTONOMOUS |
| `get_pose` | Current head/body position | - | AUTONOMOUS |
| `get_battery` | Battery level percentage | - | AUTONOMOUS |

**These are cached** - results valid for 200ms to avoid hammering hardware.

## Tool Result Caching

Read-only tools cache their results to avoid redundant SDK calls when Claude queries status multiple times in one turn.

```python
@dataclass
class CachedResult:
    value: Any
    timestamp: float
    ttl: float = 0.2  # 200ms default

class ToolCache:
    def __init__(self):
        self._cache: dict[str, CachedResult] = {}

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry.timestamp < entry.ttl:
                return entry.value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: float = 0.2):
        self._cache[key] = CachedResult(value, time.time(), ttl)

    def invalidate(self, pattern: str = "*"):
        """Invalidate cache entries matching pattern."""
        if pattern == "*":
            self._cache.clear()
        else:
            # Simple prefix matching
            keys = [k for k in self._cache if k.startswith(pattern.rstrip("*"))]
            for k in keys:
                del self._cache[k]
```

### Cached Tools

| Tool | Cache TTL | Invalidated By |
|------|-----------|----------------|
| `get_status` | 200ms | Any movement tool |
| `get_pose` | 200ms | Any movement tool |
| `get_battery` | 5000ms | Never (battery changes slowly) |
| `get_sensor_data` | 100ms | Never |

## ReachyClient Interface

Single unified interface for all robot communication.

```python
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

class Backend(Enum):
    SDK = "sdk"    # Production: Zenoh
    MOCK = "mock"  # Testing: In-memory

@dataclass
class HeadPose:
    pitch: float  # degrees, -45 to +35
    yaw: float    # degrees, -60 to +60
    roll: float   # degrees, -35 to +35
    z: float      # mm, 0 to 50 (head height)

@dataclass
class AntennaState:
    left: float   # degrees, -150 to +150
    right: float  # degrees, -150 to +150

@dataclass
class RobotStatus:
    is_awake: bool
    battery_percent: float
    head_pose: HeadPose
    body_angle: float  # degrees, 0-360
    antenna_state: AntennaState

class ReachyClient(Protocol):
    """Unified interface for robot control."""

    # Lifecycle
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def wake_up(self) -> None: ...
    async def sleep(self) -> None: ...

    # Movement
    async def move_head(self, pose: HeadPose, duration: float) -> None: ...
    async def look_at(self, x: float, y: float, z: float, duration: float) -> None: ...
    async def rotate_body(self, angle: float, duration: float) -> None: ...

    # Expression
    async def play_emotion(self, name: str) -> None: ...
    async def set_antennas(self, left: float, right: float) -> None: ...

    # Audio
    async def speak(self, text: str, voice: str = "default") -> None: ...
    async def listen(self, timeout: float = 5.0) -> str: ...

    # Perception
    async def capture_image(self) -> bytes: ...
    async def get_sensor_data(self) -> dict: ...
    async def detect_sound_direction(self) -> float: ...

    # Status (these hit cache first)
    async def get_status(self) -> RobotStatus: ...
    async def get_pose(self) -> HeadPose: ...
    async def get_battery(self) -> float: ...
```

## SDK Client Implementation

Production implementation using Reachy Mini SDK over Zenoh.

```python
from reachy_mini import ReachyMini

class SDKClient:
    """Production client using Reachy Mini SDK."""

    def __init__(self):
        self._robot: ReachyMini | None = None
        self._cache = ToolCache()

    async def connect(self) -> None:
        self._robot = ReachyMini()
        await self._robot.connect()
        if not self._robot.is_connected:
            raise ConnectionError("Failed to connect to Reachy Mini")

    async def move_head(self, pose: HeadPose, duration: float) -> None:
        """Move head to position over duration."""
        self._cache.invalidate("pose*")
        self._cache.invalidate("status*")

        await self._robot.head.goto(
            pitch=pose.pitch,
            yaw=pose.yaw,
            roll=pose.roll,
            duration=duration
        )

    async def get_pose(self) -> HeadPose:
        """Get current head pose (cached)."""
        cached = self._cache.get("pose")
        if cached:
            return cached

        state = await self._robot.head.get_state()
        pose = HeadPose(
            pitch=state.pitch,
            yaw=state.yaw,
            roll=state.roll,
            z=state.z
        )
        self._cache.set("pose", pose)
        return pose

    async def play_emotion(self, name: str) -> None:
        """Play emotion from local library or SDK."""
        # Try local bundled emotions first
        emotion_data = load_local_emotion(name)
        if emotion_data:
            await self._robot.play_recorded_move(emotion_data)
        else:
            # Fallback to SDK emotions
            await self._robot.play_emotion(name)
```

## Mock Client Implementation

Testing implementation with no hardware.

```python
class MockClient:
    """Mock client for testing without hardware."""

    def __init__(self):
        self._awake = False
        self._head_pose = HeadPose(0, 0, 0, 0)
        self._body_angle = 0.0
        self._antennas = AntennaState(0, 0)

    async def connect(self) -> None:
        pass  # Always succeeds

    async def wake_up(self) -> None:
        self._awake = True

    async def move_head(self, pose: HeadPose, duration: float) -> None:
        if not self._awake:
            raise RuntimeError("Robot is asleep")
        await asyncio.sleep(duration)  # Simulate movement time
        self._head_pose = pose

    async def get_status(self) -> RobotStatus:
        return RobotStatus(
            is_awake=self._awake,
            battery_percent=100.0,
            head_pose=self._head_pose,
            body_angle=self._body_angle,
            antenna_state=self._antennas
        )
```

## Client Factory

```python
def create_client(backend: Backend = Backend.SDK) -> ReachyClient:
    """Create appropriate client based on backend."""
    match backend:
        case Backend.SDK:
            return SDKClient()
        case Backend.MOCK:
            return MockClient()
```

## Error Handling

No silent fallbacks. Clear errors with actionable information.

```python
class RobotError(Exception):
    """Base class for robot errors."""
    pass

class ConnectionError(RobotError):
    """Failed to connect to robot."""
    pass

class MotorError(RobotError):
    """Motor-related error (stall, limit, etc)."""
    pass

class NotAwakeError(RobotError):
    """Attempted movement while robot is asleep."""
    pass
```

## Hardware Limits

| Joint | Min | Max | Unit |
|-------|-----|-----|------|
| Head Pitch | -45 | +35 | degrees |
| Head Yaw | -60 | +60 | degrees |
| Head Roll | -35 | +35 | degrees |
| Head Z | 0 | 50 | mm |
| Body Rotation | 0 | 360 | degrees (continuous) |
| Antenna L/R | -150 | +150 | degrees |

All movement commands clamp to these limits automatically.

## Emotion Library

81 emotions bundled locally in `data/emotions/`:

**Categories**:
- **Basic**: happy, sad, curious, surprised, confused, tired
- **Social**: greeting, farewell, acknowledgment, thinking
- **Reactions**: yes, no, maybe, excitement, disappointment
- **Complex**: interested, skeptical, amused, concerned

See `data/emotions/manifest.json` for complete list with metadata.

## Configuration

```yaml
# config/default.yaml
robot:
  backend: sdk  # or "mock" for testing
  connect_timeout: 5.0

cache:
  status_ttl: 0.2   # 200ms
  pose_ttl: 0.2     # 200ms
  battery_ttl: 5.0  # 5 seconds
  sensor_ttl: 0.1   # 100ms
```

## What Changed from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Tools | 27 | 20 (removed redundant) |
| Communication | SDK + HTTP fallback | SDK only |
| Caching | None | 200ms for read-only |
| Client interface | Separate daemon/SDK | Unified ReachyClient |
| Error handling | Silent fallback | Fail fast |
| `look_at` variants | 3 (`look_at`, `look_at_world`, `look_at_pixel`) | 1 |
| Dance/emotion | Separate | Merged |

## Related Specs

- [01-overview.md](./01-overview.md) - System architecture
- [04-motion.md](./04-motion.md) - Motion control system
- [06-permissions.md](./06-permissions.md) - Permission tiers
