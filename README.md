# Claude in the Shell v2

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude](https://img.shields.io/badge/Powered%20by-Claude-blueviolet)](https://anthropic.com)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](./docs/index.md)

An embodied AI agent for the Reachy Mini desktop robot. Claude runs on the robot's Raspberry Pi 4, using MCP tools to control motors, speak, listen, and perceive the world. The robot has personality, memory, and can hold natural voice conversations with barge-in support.

## Features

- **Voice Interaction**: Multi-persona wake words with barge-in support ("Hey Motoko", "Hey Batou")
- **Robot Control**: 20 MCP tools for motor control, sensors, and expressions
- **Memory System**: ChromaDB-backed semantic memory with 3 types (CONVERSATION, FACT, CONTEXT)
- **Motion Control**: 30Hz unified control loop with idle behavior and speech wobble
- **Permissions**: 3-tier tool authorization (AUTONOMOUS, CONFIRM, FORBIDDEN)
- **Observability**: Structured logging, health checks, and debugging tools
- **Simulation**: MuJoCo physics simulation for development without hardware

## Quick Start

### Installation

```bash
# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev,voice]"

# Install with simulation support (MuJoCo)
uv pip install -e ".[sim]"

# Copy environment template
cp .env.example .env

# Add your API keys to .env
```

### Running

```bash
# Interactive text mode
python -m reachy_agent run

# Voice mode with barge-in
python -m reachy_agent run --voice

# Mock mode (no hardware)
python -m reachy_agent run --mock

# Simulation mode (MuJoCo physics)
python -m reachy_agent run --sim

# Simulation with viewer
python -m reachy_agent run --sim --sim-viewer

# Voice mode with debug logging
python -m reachy_agent run --voice --debug-voice

# Health check
python -m reachy_agent check
```

## Documentation

**[View Full Documentation](./docs/index.md)** - Comprehensive guides, tutorials, and API reference.

### Quick Links

| Section | Description |
|---------|-------------|
| [Getting Started](./docs/getting-started.md) | Installation and first steps |
| [Quick Start](./docs/quick-start.md) | 5-minute tutorial |
| [User Guide](./docs/user-guide/index.md) | Complete user documentation |
| [Tutorials](./docs/tutorials/index.md) | Step-by-step tutorials |
| [API Reference](./docs/api-reference/index.md) | MCP tools and Python APIs |
| [Developer Guide](./docs/developer-guide/index.md) | Architecture and contributing |
| [Examples](./docs/examples/index.md) | Working code examples |

## Project Structure

```
.
├── config/              # Configuration files
├── prompts/             # System and persona prompts
├── src/
│   └── reachy_agent/    # Main source code
│       ├── agent/       # Agent core (loop, options)
│       ├── mcp/         # MCP servers (robot, memory)
│       ├── robot/       # Robot hardware abstraction
│       ├── voice/       # Voice pipeline
│       ├── motion/      # Motion control
│       ├── memory/      # Memory system
│       ├── permissions/ # Permission system
│       ├── simulation/  # MuJoCo simulation support
│       └── utils/       # Utilities
├── tests/               # Test suite
├── data/                # Static data (emotions, wake words, scenarios)
├── examples/            # Example scripts (simulation demos)
├── docs/                # Documentation (GitHub Pages)
├── ai_docs/             # AI agent reference docs
├── scripts/             # Utility scripts
└── specs/               # Technical specifications
```

## Development

```bash
# Format code
uvx black .

# Sort imports
uvx isort . --profile black

# Type check
uvx mypy .

# Lint
uvx ruff check .

# Test
pytest -v

# Test with coverage
pytest --cov=src --cov-report=html
```

## Configuration

Configuration is convention-based with sensible defaults. Most things auto-discover from well-known paths.

See `config/default.yaml` for minimal configuration options.

## Architecture

The system uses an event-driven architecture with:

- **Agent Loop**: Main coordinator using Claude SDK
- **Robot MCP**: 20 tools for robot control via Zenoh/SDK
- **Voice Pipeline**: Event-driven voice interaction with OpenAI Realtime API
- **Motion Control**: 30Hz blend controller (idle + wobble)
- **Memory**: ChromaDB-only storage with conversation window
- **Permissions**: 3-tier authorization with audit logging

See the [Architecture Documentation](./docs/developer-guide/architecture.md) for detailed design.

## Hardware

**Reachy Mini Wireless** (Raspberry Pi 4):
- Head: 6 DOF (pitch, yaw, roll, z, 2 antennas)
- Body: 360° continuous rotation
- Camera: Wide-angle USB
- Audio: 4-mic array + 5W speaker
- Connectivity: WiFi

## MCP Tools

### Robot Control (20 tools)

| Category | Tools |
|----------|-------|
| Movement | `move_head`, `look_at`, `rotate_body`, `reset_position` |
| Expression | `play_emotion`, `play_sequence`, `set_antennas`, `nod`, `shake` |
| Audio | `speak`, `listen` |
| Perception | `capture_image`, `get_sensor_data`, `detect_sound_direction` |
| Lifecycle | `wake_up`, `sleep`, `is_awake` |
| Status | `get_status`, `get_position`, `get_limits` |

### Memory Tools (3 tools)

| Tool | Description |
|------|-------------|
| `store_memory` | Store facts, conversations, or context |
| `search_memories` | Semantic search across memories |
| `forget_memory` | Delete specific memories |

See the [API Reference](./docs/api-reference/index.md) for complete documentation.

## Contributing

See [Contributing Guide](./docs/developer-guide/contributing.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

## License

MIT

## Links

- [Documentation](./docs/index.md)
- [Technical Specifications](./specs/)
- [AI Agent Reference](./ai_docs/)
