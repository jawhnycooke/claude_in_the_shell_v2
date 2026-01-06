# Python API Reference

Core Python interfaces and client implementations.

## Client Interfaces

### ReachyClient Protocol

Abstract interface for robot control. All client implementations follow this protocol.

```python
from typing import Protocol
from reachy_agent.robot.client import HeadPose, AntennaState, RobotStatus

class ReachyClient(Protocol):
    """Unified interface for robot control."""

    # === Lifecycle ===

    async def connect(self) -> None:
        """Connect to the robot."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the robot."""
        ...

    async def wake_up(self) -> None:
        """Enable motors, stand ready."""
        ...

    async def sleep(self) -> None:
        """Disable motors, low power mode."""
        ...

    # === Movement ===

    async def move_head(
        self,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        duration: float = 1.0
    ) -> None:
        """Move head to absolute position."""
        ...

    async def look_at(
        self,
        x: float,
        y: float,
        z: float,
        duration: float = 1.0
    ) -> None:
        """Look at 3D point in robot frame."""
        ...

    async def rotate_body(
        self,
        angle: float,
        duration: float = 1.0
    ) -> None:
        """Rotate body to angle (0-360)."""
        ...

    async def reset_position(self, duration: float = 1.0) -> None:
        """Return to neutral pose."""
        ...

    # === Expression ===

    async def play_emotion(self, name: str) -> None:
        """Play emotion animation."""
        ...

    async def set_antennas(self, left: float, right: float) -> None:
        """Set antenna positions."""
        ...

    async def nod(self, intensity: float = 1.0) -> None:
        """Perform nod gesture."""
        ...

    async def shake(self, intensity: float = 1.0) -> None:
        """Perform head shake gesture."""
        ...

    # === Audio ===

    async def speak(self, text: str, voice: str = "default") -> None:
        """Text-to-speech output."""
        ...

    async def listen(self, timeout: float = 5.0) -> str:
        """Listen for speech, return transcription."""
        ...

    # === Perception ===

    async def capture_image(self) -> bytes:
        """Capture camera frame."""
        ...

    async def get_sensor_data(self) -> dict:
        """Read IMU/accelerometer."""
        ...

    async def detect_sound_direction(self) -> tuple[float, float]:
        """Get direction (azimuth, confidence) of loudest sound."""
        ...

    # === Status ===

    async def is_awake(self) -> bool:
        """Check if robot is active."""
        ...

    async def get_status(self) -> RobotStatus:
        """Get comprehensive robot state."""
        ...

    async def get_position(self) -> dict:
        """Get current joint positions."""
        ...

    async def get_limits(self) -> dict:
        """Get joint angle limits."""
        ...
```

---

### MockClient

Testing implementation without hardware.

```python
from reachy_agent.robot.mock import MockClient

class MockClient:
    """Mock client for testing without hardware."""

    def __init__(self):
        self._connected = False
        self._awake = False
        self._head_pose = HeadPose(0, 0, 0, 0)
        self._body_angle = 0.0
        self._antennas = AntennaState(0, 0)

    async def connect(self) -> None:
        """Always succeeds."""
        self._connected = True

    async def wake_up(self) -> None:
        """Enable mock motors."""
        self._awake = True

    async def move_head(
        self,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        duration: float = 1.0
    ) -> None:
        """Simulate head movement."""
        if not self._awake:
            raise RuntimeError("Robot is asleep")
        await asyncio.sleep(duration)  # Simulate movement time
        self._head_pose = HeadPose(pitch, yaw, roll, 0)

    # ... other methods follow same pattern
```

**Usage:**
```python
client = MockClient()
await client.connect()
await client.wake_up()
await client.move_head(pitch=10, yaw=-30, duration=1.0)
position = await client.get_position()
```

---

### SDKClient

Production implementation using Reachy Mini SDK.

```python
from reachy_agent.robot.sdk import SDKClient

class SDKClient:
    """Production client using Reachy Mini SDK over Zenoh."""

    def __init__(self):
        self._robot: ReachyMini | None = None
        self._cache = ToolCache()

    async def connect(self) -> None:
        """Connect to physical robot."""
        self._robot = ReachyMini()
        await self._robot.connect()
        if not self._robot.is_connected:
            raise ConnectionError("Failed to connect to Reachy Mini")

    async def move_head(
        self,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        duration: float = 1.0
    ) -> None:
        """Move head on physical robot."""
        self._cache.invalidate("pose*")
        self._cache.invalidate("status*")

        await self._robot.head.goto(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            duration=duration
        )
```

---

### MuJoCoReachyClient

Simulation implementation using MuJoCo physics.

```python
from reachy_agent.simulation.client import MuJoCoReachyClient

class MuJoCoReachyClient:
    """Simulation client using MuJoCo physics."""

    def __init__(
        self,
        model_path: str = "data/models/reachy_mini/reachy_mini.xml",
        realtime: bool = True,
        viewer: bool = False
    ):
        """
        Create simulation client.

        Args:
            model_path: Path to MJCF model file
            realtime: Match wall clock time
            viewer: Enable 3D visualization
        """
        self._model_path = model_path
        self._realtime = realtime
        self._viewer = viewer
        self._env: SimulationEnvironment | None = None

    async def connect(self) -> None:
        """Initialize simulation environment."""
        self._env = SimulationEnvironment(
            model_path=self._model_path,
            realtime=self._realtime,
            viewer=self._viewer
        )
        await self._env.start()

    async def move_head(
        self,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        duration: float = 1.0
    ) -> None:
        """Move head in simulation."""
        await self._env.move_joints(
            {
                "head_pitch": pitch,
                "head_yaw": yaw,
                "head_roll": roll
            },
            duration=duration
        )
```

**Usage:**
```python
client = MuJoCoReachyClient(realtime=True, viewer=True)
await client.connect()
await client.wake_up()
await client.move_head(pitch=15, yaw=20, duration=1.0)
# 3D viewer shows movement with physics
await client.disconnect()
```

---

## Client Factory

Create appropriate client based on backend.

```python
from reachy_agent.robot.factory import create_client, Backend

class Backend(Enum):
    SDK = "sdk"    # Production: Physical robot
    MOCK = "mock"  # Testing: No hardware
    SIM = "sim"    # Simulation: MuJoCo

def create_client(backend: Backend = Backend.SDK, **kwargs) -> ReachyClient:
    """
    Create appropriate client based on backend.

    Args:
        backend: SDK, MOCK, or SIM
        **kwargs: Backend-specific options

    Returns:
        ReachyClient implementation
    """
    match backend:
        case Backend.SDK:
            return SDKClient()
        case Backend.MOCK:
            return MockClient()
        case Backend.SIM:
            return MuJoCoReachyClient(**kwargs)
```

**Usage:**
```python
from reachy_agent.robot.factory import create_client, Backend

# For development
client = create_client(Backend.MOCK)

# For simulation with viewer
client = create_client(Backend.SIM, realtime=True, viewer=True)

# For production
client = create_client(Backend.SDK)
```

---

## Data Classes

### HeadPose

```python
from dataclasses import dataclass

@dataclass
class HeadPose:
    """Absolute head position."""
    pitch: float  # degrees, -45 to +35
    yaw: float    # degrees, -60 to +60
    roll: float   # degrees, -35 to +35
    z: float      # mm, 0 to 50
```

### AntennaState

```python
@dataclass
class AntennaState:
    """Antenna positions."""
    left: float   # degrees, -150 to +150
    right: float  # degrees, -150 to +150
```

### RobotStatus

```python
@dataclass
class RobotStatus:
    """Comprehensive robot state."""
    is_awake: bool
    battery_percent: float
    head_pose: HeadPose
    body_angle: float  # degrees, 0-360
    antenna_state: AntennaState
```

---

## Agent Loop

Main agent coordinator.

```python
from reachy_agent.agent.loop import ReachyAgentLoop
from reachy_agent.agent.options import AgentConfig

@dataclass
class AgentConfig:
    """Agent configuration."""
    model: str = "claude-haiku-4-5-20251001"
    name: str = "Jarvis"
    max_tokens: int = 4096
    temperature: float = 0.7
    enable_voice: bool = False
    enable_motion: bool = True
    mock_hardware: bool = False
    system_prompt_path: str = "prompts/system.md"
    persona_path: str | None = None

class ReachyAgentLoop:
    """Main agent coordinator."""

    def __init__(self, config: AgentConfig):
        """Create agent with configuration."""
        ...

    async def start(self) -> None:
        """Initialize all components and start the agent."""
        ...

    async def stop(self) -> None:
        """Gracefully shutdown all components."""
        ...

    async def process(self, user_input: str) -> str:
        """
        Process a single user input and return response.

        Args:
            user_input: User's message or command

        Returns:
            Agent's response text
        """
        ...
```

**Usage:**
```python
from reachy_agent.agent.loop import ReachyAgentLoop
from reachy_agent.agent.options import AgentConfig

config = AgentConfig(
    model="claude-haiku-4-5-20251001",
    enable_voice=False,
    mock_hardware=True
)

agent = ReachyAgentLoop(config)

try:
    await agent.start()

    # Process input
    response = await agent.process("Hello, who are you?")
    print(response)

    # Interactive loop
    while True:
        user_input = input("> ")
        if user_input.lower() in ("quit", "exit"):
            break
        response = await agent.process(user_input)
        print(response)

finally:
    await agent.stop()
```

---

## Error Classes

```python
from reachy_agent.robot.client import RobotError

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

**Handling:**
```python
try:
    await client.move_head(pitch=10)
except NotAwakeError:
    await client.wake_up()
    await client.move_head(pitch=10)
except ConnectionError as e:
    print(f"Lost connection: {e}")
    await client.connect()
except MotorError as e:
    print(f"Motor error: {e}")
```

---

## Utilities

### Configuration

```python
from reachy_agent.utils.config import load_config, Config

def load_config(path: str = "config/default.yaml") -> Config:
    """Load and validate configuration."""
    ...

@dataclass
class Config:
    agent: AgentConfig
    robot: RobotConfig
    voice: VoiceConfig
    motion: MotionConfig
    memory: MemoryConfig
    simulation: SimulationConfig
```

### Logging

```python
from reachy_agent.utils.logging import setup_logging, get_logger

def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    ...

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger with the given name."""
    ...
```

**Usage:**
```python
from reachy_agent.utils.logging import get_logger

log = get_logger("my_module")
log.info("starting", component="robot")
log.error("failed", error="connection timeout")
```

### Events

```python
from reachy_agent.utils.events import EventEmitter

class EventEmitter:
    """Generic async event emitter."""

    def on(self, event_name: str) -> Callable:
        """Decorator to register event handler."""
        ...

    async def emit(self, event_name: str, **data) -> None:
        """Emit event to all handlers."""
        ...
```
