# Project Structure Specification

## Overview

A clean, flat folder structure that supports modular development. Convention over configuration - most things auto-discover from well-known paths.

## Root Structure

```
reachy-agent/
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore patterns
├── CLAUDE.md                 # Claude Code project instructions
├── README.md                 # Project documentation
├── pyproject.toml            # Python project configuration (single source)
│
├── ai_docs/                  # AI agent reference documentation
├── config/                   # Configuration files
├── data/                     # Static data assets
├── prompts/                  # System and persona prompts
├── scripts/                  # Utility scripts
├── specs/                    # Technical specifications (this folder)
├── src/                      # Source code
└── tests/                    # Test suite
```

## Key Directories

### `src/reachy_agent/` - Source Code

```
src/
└── reachy_agent/
    ├── __init__.py
    ├── __main__.py           # CLI entry: python -m reachy_agent
    │
    ├── agent/                # Agent core
    │   ├── __init__.py
    │   ├── loop.py           # ReachyAgentLoop - main coordinator
    │   └── options.py        # Agent configuration builder
    │
    ├── mcp/                  # MCP servers
    │   ├── __init__.py
    │   ├── robot.py          # Robot control (20 tools)
    │   └── memory.py         # Memory tools (3 tools)
    │
    ├── robot/                # Robot hardware abstraction
    │   ├── __init__.py
    │   ├── client.py         # ReachyClient interface
    │   ├── sdk.py            # SDK implementation
    │   └── mock.py           # Mock for testing
    │
    ├── voice/                # Voice pipeline
    │   ├── __init__.py
    │   ├── pipeline.py       # Event-driven pipeline
    │   ├── persona.py        # PersonaManager
    │   ├── realtime.py       # OpenAI Realtime client
    │   ├── wake_word.py      # OpenWakeWord detector
    │   └── audio.py          # AudioManager
    │
    ├── motion/               # Motion control
    │   ├── __init__.py
    │   ├── controller.py     # BlendController
    │   ├── idle.py           # IdleBehavior
    │   ├── wobble.py         # SpeechWobble
    │   └── emotion.py        # EmotionPlayback
    │
    ├── memory/               # Memory system
    │   ├── __init__.py
    │   └── manager.py        # MemoryManager (ChromaDB)
    │
    ├── permissions/          # Permission system
    │   ├── __init__.py
    │   ├── evaluator.py      # PermissionEvaluator
    │   └── hooks.py          # Agent SDK hooks
    │
    └── utils/                # Utilities
        ├── __init__.py
        ├── config.py         # Configuration loading
        ├── events.py         # EventEmitter base class
        └── logging.py        # Structured logging setup
```

**Key difference from v1**: Flatter structure. No deep nesting like `mcp_servers/reachy/`. Robot client lives in `robot/`, MCP tools in `mcp/`.

### `config/` - Configuration Files

```
config/
├── default.yaml              # Default configuration
└── permissions.yaml          # Permission rules
```

**Minimal config**. Most things auto-discover or use sensible defaults.

```yaml
# config/default.yaml
agent:
  model: claude-haiku-4-5-20251001
  name: Jarvis

voice:
  personas: [motoko, batou, jarvis]  # Auto-discovers from prompts/personas/
  wake_sensitivity: 0.5

motion:
  tick_hz: 30

memory:
  path: ~/.reachy/memory
```

### `prompts/` - System Prompts

```
prompts/
├── system.md                 # Default system prompt
└── personas/                 # Persona definitions (auto-discovered)
    ├── jarvis.md             # Default assistant
    ├── motoko.md             # Major Kusanagi
    └── batou.md              # Batou
```

**Auto-discovery**: Voice pipeline reads all `*.md` from `prompts/personas/` and extracts wake words, voices from YAML frontmatter.

### `data/` - Static Data

```
data/
├── emotions/                 # Emotion library (81 animations)
│   ├── manifest.json         # Index with metadata
│   ├── happy.npz
│   ├── curious.npz
│   └── ...
└── wake_words/               # Custom wake word models (optional)
    └── ...
```

### `ai_docs/` - AI Agent Reference

```
ai_docs/
├── REFERENCE.md              # External links and resources
├── code-standards.md         # Style, linting, typing rules
├── mcp-tools.md              # Tool quick reference
└── dev-commands.md           # Development cheat sheet
```

**Purpose**: Quick reference for AI agents working on the codebase.

### `specs/` - Technical Specifications

```
specs/
├── 01-overview.md            # System architecture
├── 02-robot-control.md       # MCP tools, unified client
├── 03-voice.md               # Event-driven voice pipeline
├── 04-motion.md              # Motion control
├── 05-memory.md              # Memory system
├── 06-permissions.md         # Permission tiers
└── 07-project-structure.md   # This file
```

### `tests/` - Test Suite

```
tests/
├── __init__.py
├── conftest.py               # Shared fixtures
├── test_agent.py
├── test_robot.py
├── test_voice.py
├── test_motion.py
├── test_memory.py
└── test_permissions.py
```

**Flat test structure** mirroring source. No deep `test_mcp_servers/test_reachy/` nesting.

### `scripts/` - Utility Scripts

```
scripts/
├── download_emotions.py      # Download emotion library
├── setup_pi.sh               # Raspberry Pi setup
└── test_hardware.py          # Hardware connectivity test
```

## Configuration Files

### `pyproject.toml` (Single Source of Truth)

```toml
[project]
name = "reachy-agent"
version = "0.1.0"
description = "Embodied AI agent for Reachy Mini robot"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.30.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "structlog>=24.0.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
voice = [
    "pyaudio>=0.2.13",
    "openwakeword>=0.6.0",
    "websockets>=12.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1",
    "black>=24.0",
    "ruff>=0.3.0",
    "mypy>=1.8",
]

[project.scripts]
reachy-agent = "reachy_agent.__main__:main"

[tool.black]
line-length = 88

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

**No requirements.txt** - use `pyproject.toml` for everything.

### `.env.example`

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional - Voice
OPENAI_API_KEY=sk-...

# Optional - Debug
REACHY_DEBUG=0
```

### `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
dist/
build/

# Environment
.env
.env.local

# Data
~/.reachy/
*.db
chroma_data/

# IDE
.vscode/
.idea/

# Testing
.coverage
htmlcov/
.pytest_cache/
```

## Module Dependencies

```
                    ┌─────────────┐
                    │  __main__   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │  agent   │ │  voice   │ │  motion  │
       └────┬─────┘ └────┬─────┘ └────┬─────┘
            │            │            │
            └────────────┼────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────┐      ┌────────┐      ┌────────┐
    │  mcp   │      │ memory │      │  robot │
    └────────┘      └────────┘      └────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
                    ┌────────┐
                    │ utils  │
                    └────────┘
```

**Rules**:
- `utils` has no dependencies on other modules
- `robot`, `memory` depend only on `utils`
- `mcp` depends on `robot`, `memory`, `utils`
- `voice`, `motion` depend on `robot`, `utils`
- `agent` depends on everything

## File Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Module | snake_case | `blend_controller.py` |
| Class | PascalCase | `BlendController` |
| Function | snake_case | `process_audio()` |
| Constant | UPPER_SNAKE | `MAX_TURNS` |
| Config | snake_case.yaml | `permissions.yaml` |
| Spec | NN-name.md | `01-overview.md` |
| Test | test_*.py | `test_voice.py` |

## Import Order

```python
# Standard library
import asyncio
from datetime import datetime
from pathlib import Path

# Third-party
import structlog
from pydantic import BaseModel

# Local
from reachy_agent.memory import MemoryManager
from reachy_agent.utils import config
```

## CLI Commands

```bash
# Run agent (interactive)
python -m reachy_agent run

# Run with voice
python -m reachy_agent run --voice

# Run with mock hardware
python -m reachy_agent run --mock

# Debug voice events
python -m reachy_agent run --voice --debug-voice

# Health check
python -m reachy_agent check

# Show version
python -m reachy_agent version
```

## Development Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,voice]"

# Quality
uvx black .
uvx ruff check .
uvx mypy .

# Test
pytest -v
pytest --cov=src

# Run MCP Inspector
npx @modelcontextprotocol/inspector \
  python -m reachy_agent.mcp.robot
```

## Storage Locations

| Data | Location | Purpose |
|------|----------|---------|
| Memory DB | `~/.reachy/memory/` | ChromaDB persistence |
| Audit logs | `~/.reachy/audit.jsonl` | Permission decisions |
| Cache | `~/.reachy/cache/` | Temporary data |
| Config | `./config/` | Project config |

## What Changed from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Source depth | Deep (`mcp_servers/reachy/`) | Flat (`mcp/`, `robot/`) |
| Config files | requirements.txt + pyproject.toml | pyproject.toml only |
| Test structure | Mirrors source depth | Flat |
| Spec count | 8 | 7 |
| behaviors/ folder | Yes | Renamed to `motion/` |
| emotions/ folder | In src | Moved to `data/` |

## Related Specs

- [01-overview.md](./01-overview.md) - System architecture
- [ai_docs/REFERENCE.md](../ai_docs/REFERENCE.md) - External links
