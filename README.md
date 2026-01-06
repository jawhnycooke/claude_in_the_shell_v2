# Claude in the Shell v2

An embodied AI agent for the Reachy Mini desktop robot. Claude runs on the robot's Raspberry Pi 4, using MCP tools to control motors, speak, listen, and perceive the world. The robot has personality, memory, and can hold natural voice conversations with barge-in support.

## Features

- **Voice Interaction**: Multi-persona wake words with barge-in support ("Hey Motoko", "Hey Batou")
- **Robot Control**: 20 MCP tools for motor control, sensors, and expressions
- **Memory System**: ChromaDB-backed semantic memory with 3 types (CONVERSATION, FACT, CONTEXT)
- **Motion Control**: 30Hz unified control loop with idle behavior and speech wobble
- **Permissions**: 3-tier tool authorization (AUTONOMOUS, CONFIRM, FORBIDDEN)
- **Observability**: Structured logging, health checks, and debugging tools

## Quick Start

### Installation

```bash
# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev,voice]"

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

# Voice mode with debug logging
python -m reachy_agent run --voice --debug-voice

# Health check
python -m reachy_agent check
```

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
│       └── utils/       # Utilities
├── tests/               # Test suite
├── data/                # Static data (emotions, wake words)
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

## Documentation

- `specs/` - Complete technical specifications
- `ai_docs/` - Quick reference for AI agents working on the codebase
- `CLAUDE.md` - Project instructions for Claude Code

## Architecture

The system uses an event-driven architecture with:

- **Agent Loop**: Main coordinator using Claude SDK
- **Robot MCP**: 20 tools for robot control via Zenoh/SDK
- **Voice Pipeline**: Event-driven voice interaction with OpenAI Realtime API
- **Motion Control**: 30Hz blend controller (idle + wobble)
- **Memory**: ChromaDB-only storage with conversation window
- **Permissions**: 3-tier authorization with audit logging

See `specs/01-overview.md` for detailed architecture.

## Hardware

**Reachy Mini Wireless** (Raspberry Pi 4):
- Head: 6 DOF (pitch, yaw, roll, z, 2 antennas)
- Body: 360° continuous rotation
- Camera: Wide-angle USB
- Audio: 4-mic array + 5W speaker
- Connectivity: WiFi

## License

MIT
