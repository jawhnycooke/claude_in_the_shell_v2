# Code Examples

Working code examples for common tasks with the Reachy Agent.

## Example Categories

| Category | Description |
|----------|-------------|
| [Basic Movements](basic-movements.md) | Head control, body rotation, antenna positioning |
| [Voice Commands](voice-commands.md) | Voice interaction patterns and custom wake words |
| [Simulation Scenarios](simulation-scenarios.md) | MuJoCo simulation scripts and testing |

---

## Quick Examples

### Start the Agent

```python
import asyncio
from reachy_agent.agent.loop import ReachyAgentLoop
from reachy_agent.agent.options import AgentConfig

async def main():
    config = AgentConfig(
        model="claude-haiku-4-5-20251001",
        enable_voice=False,
        mock_hardware=True  # No physical robot needed
    )

    agent = ReachyAgentLoop(config)

    try:
        await agent.start()

        # Interactive loop
        while True:
            user_input = input("> ")
            if user_input.lower() in ("quit", "exit"):
                break
            response = await agent.process(user_input)
            print(response)

    finally:
        await agent.stop()

asyncio.run(main())
```

### Direct Robot Control

```python
import asyncio
from reachy_agent.robot.factory import create_client, Backend

async def main():
    # Create client (mock for development)
    client = create_client(Backend.MOCK)

    await client.connect()
    await client.wake_up()

    # Move head
    await client.move_head(pitch=10, yaw=-30, duration=1.0)

    # Play emotion
    await client.play_emotion("happy")

    # Rotate body
    await client.rotate_body(angle=180, duration=2.0)

    await client.sleep()
    await client.disconnect()

asyncio.run(main())
```

### MuJoCo Simulation

```python
import asyncio
from reachy_agent.simulation import MuJoCoReachyClient

async def main():
    client = MuJoCoReachyClient(
        realtime=True,
        viewer=True  # Open 3D visualization
    )

    await client.connect()
    await client.wake_up()

    # Watch the robot move in the viewer
    await client.move_head(pitch=20, yaw=30, duration=1.5)
    await client.play_emotion("curious")

    input("Press Enter to exit...")
    await client.disconnect()

asyncio.run(main())
```

### Memory Operations

```python
import asyncio
from reachy_agent.memory.manager import MemoryManager, MemoryType

async def main():
    manager = MemoryManager()
    await manager.initialize()

    # Store a fact
    await manager.store(
        content="User's favorite color is blue",
        memory_type=MemoryType.FACT
    )

    # Search memories
    results = await manager.search("color preferences", limit=5)
    for result in results:
        print(f"[{result.score:.2f}] {result.memory.content}")

    await manager.close()

asyncio.run(main())
```

---

## Running Examples

All examples can be run from the project root:

```bash
# Activate environment
source .venv/bin/activate

# Run an example
python examples/basic_movement.py
python examples/voice_demo.py
python examples/simulation_demo.py
```

## Prerequisites

Before running examples, ensure you have:

1. **Environment setup**: `uv venv && source .venv/bin/activate`
2. **Dependencies installed**: `uv pip install -e ".[dev]"`
3. **API key configured**: `ANTHROPIC_API_KEY` in `.env`
4. **For voice**: `OPENAI_API_KEY` in `.env`
5. **For simulation**: `uv pip install -e ".[sim]"`

---

## Example Files

```
examples/
├── basic_movement.py      # Head and body control
├── emotion_sequences.py   # Playing emotions
├── voice_demo.py          # Voice interaction
├── memory_demo.py         # Memory operations
├── simulation_demo.py     # MuJoCo simulation
├── custom_behavior.py     # Creating behaviors
└── full_agent.py          # Complete agent setup
```

---

## Next Steps

- [Basic Movements](basic-movements.md) - Detailed movement examples
- [Voice Commands](voice-commands.md) - Voice interaction patterns
- [Simulation Scenarios](simulation-scenarios.md) - Testing in simulation
