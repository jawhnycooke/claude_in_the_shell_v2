# External References

Links to external documentation, SDKs, and resources needed to implement the Reachy Agent project.

## Core Technologies

### Anthropic / Claude

| Resource | Link |
|----------|------|
| Claude API Documentation | https://docs.anthropic.com/en/api/ |
| Claude Models Overview | https://docs.anthropic.com/en/docs/about-claude/models |
| Tool Use Guide | https://docs.anthropic.com/en/docs/agents-and-tools/tool-use |
| Streaming Guide | https://docs.anthropic.com/en/api/messages-streaming |
| Python SDK | https://github.com/anthropics/anthropic-sdk-python |

### Model Context Protocol (MCP)

| Resource | Link |
|----------|------|
| MCP Specification | https://spec.modelcontextprotocol.io/ |
| MCP Python SDK | https://github.com/modelcontextprotocol/python-sdk |
| FastMCP (Simplified) | https://github.com/modelcontextprotocol/python-sdk/tree/main/src/mcp/server/fastmcp |
| MCP Inspector | https://github.com/modelcontextprotocol/inspector |

## Reachy Mini Hardware

### Pollen Robotics

| Resource | Link |
|----------|------|
| Reachy Mini Product | https://www.pollen-robotics.com/reachy-mini/ |
| Reachy Mini SDK | https://github.com/pollen-robotics/reachy_mini |
| SDK Documentation | https://pollen-robotics.github.io/reachy_mini/ |
| Emotions Library | https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library |

### Hardware Specifications

| Specification | Value |
|---------------|-------|
| Processor | Raspberry Pi 4 (4GB) |
| Head DOF | 6 (pitch, yaw, roll, z, 2 antennas) |
| Body | 360° continuous rotation |
| Camera | Wide-angle USB camera |
| Audio | 4-mic array + 5W speaker |
| Connectivity | WiFi |

## Voice Pipeline Components

### OpenAI Realtime API

| Resource | Link |
|----------|------|
| Realtime API Guide | https://platform.openai.com/docs/guides/realtime |
| WebSocket Reference | https://platform.openai.com/docs/api-reference/realtime |
| Audio Formats | https://platform.openai.com/docs/guides/realtime/audio-formats |
| Voice Options | https://platform.openai.com/docs/guides/text-to-speech/voice-options |

### Wake Word Detection

| Resource | Link |
|----------|------|
| OpenWakeWord | https://github.com/dscripka/openWakeWord |
| Pre-trained Models | https://github.com/dscripka/openWakeWord/tree/main/models |
| Training Custom Models | https://github.com/dscripka/openWakeWord/blob/main/docs/custom_models.md |

### Audio Processing

| Resource | Link |
|----------|------|
| PyAudio | https://pypi.org/project/PyAudio/ |
| PortAudio (underlying) | http://www.portaudio.com/ |

## Memory System

### ChromaDB

| Resource | Link |
|----------|------|
| Documentation | https://docs.trychroma.com/ |
| Python Quickstart | https://docs.trychroma.com/getting-started/quickstart |
| Embedding Functions | https://docs.trychroma.com/embeddings |
| Persistence | https://docs.trychroma.com/deployment/persistence |

### Sentence Transformers

| Resource | Link |
|----------|------|
| Documentation | https://www.sbert.net/ |
| Model Hub | https://huggingface.co/sentence-transformers |
| all-MiniLM-L6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |

## Python Dependencies

### Core Framework

| Package | Link | Purpose |
|---------|------|---------|
| Pydantic | https://docs.pydantic.dev/ | Data validation, config |
| httpx | https://www.python-httpx.org/ | Async HTTP client |
| structlog | https://www.structlog.org/ | Structured logging |
| Typer | https://typer.tiangolo.com/ | CLI framework |
| Rich | https://rich.readthedocs.io/ | Terminal formatting |
| PyYAML | https://pyyaml.org/ | YAML parsing |

### Async Programming

| Package | Link | Purpose |
|---------|------|---------|
| asyncio (stdlib) | https://docs.python.org/3/library/asyncio.html | Async runtime |
| websockets | https://websockets.readthedocs.io/ | WebSocket client |

### Development Tools

| Package | Link | Purpose |
|---------|------|---------|
| pytest | https://docs.pytest.org/ | Testing framework |
| pytest-asyncio | https://pytest-asyncio.readthedocs.io/ | Async test support |
| Black | https://black.readthedocs.io/ | Code formatting |
| Ruff | https://docs.astral.sh/ruff/ | Fast linting |
| mypy | https://mypy.readthedocs.io/ | Type checking |

## Communication Protocols

### Zenoh (SDK Communication)

| Resource | Link |
|----------|------|
| Zenoh Documentation | https://zenoh.io/docs/ |
| Python API | https://github.com/eclipse-zenoh/zenoh-python |
| Getting Started | https://zenoh.io/docs/getting-started/first-app/ |

## Raspberry Pi

### Setup & Configuration

| Resource | Link |
|----------|------|
| Raspberry Pi OS | https://www.raspberrypi.com/software/ |
| Audio Configuration | https://www.raspberrypi.com/documentation/computers/audio.html |
| Python GPIO | https://gpiozero.readthedocs.io/ |

### Performance

| Resource | Link |
|----------|------|
| Thermal Management | https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#cooling |
| CPU Governor | https://www.raspberrypi.com/documentation/computers/os.html#cpu-frequency |

## Motion & Animation

| Resource | Link |
|----------|------|
| Perlin Noise (Wikipedia) | https://en.wikipedia.org/wiki/Perlin_noise |
| noise library | https://github.com/caseman/noise |
| Easing Functions | https://easings.net/ |

## Additional Resources

### Ghost in the Shell (Persona Inspiration)

| Resource | Description |
|----------|-------------|
| Ghost in the Shell Wiki | https://ghostintheshell.fandom.com/wiki/Ghost_in_the_Shell_Wiki |
| Character References | https://ghostintheshell.fandom.com/wiki/Category:Characters |

## API Keys Required

| Service | Environment Variable | Required For |
|---------|---------------------|--------------|
| Anthropic | `ANTHROPIC_API_KEY` | Claude API (required) |
| OpenAI | `OPENAI_API_KEY` | Voice STT/TTS (optional) |

## Version Compatibility

| Component | Minimum Version | Recommended |
|-----------|-----------------|-------------|
| Python | 3.10 | 3.11+ |
| Anthropic SDK | 0.30.0 | Latest |
| ChromaDB | 0.4.0 | 0.5.x |
| Reachy Mini SDK | 0.1.0 | Latest |
| OpenWakeWord | 0.6.0 | Latest |

## Development Cheat Sheet

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,voice]"

# Quality checks
uvx black .
uvx ruff check .
uvx mypy .

# Testing
pytest -v
pytest --cov=src

# Run agent
python -m reachy_agent run --mock
python -m reachy_agent run --voice

# MCP Inspector
npx @modelcontextprotocol/inspector python -m reachy_agent.mcp.robot
```

## Related Specs

- [01-overview.md](../01-overview.md) - System architecture
- [07-project-structure.md](../07-project-structure.md) - Project layout
