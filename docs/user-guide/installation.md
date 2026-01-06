# Installation Guide

This guide covers installation for all platforms and use cases.

## System Requirements

### Minimum Requirements
- Python 3.10 or higher
- 4GB RAM
- 2GB disk space

### Recommended
- Python 3.11+
- 8GB RAM
- SSD storage
- NVIDIA GPU (for simulation)

### Platform Support
- **Linux**: Full support (recommended for Raspberry Pi)
- **macOS**: Full support (Intel and Apple Silicon)
- **Windows**: Supported with WSL2

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone repository
git clone https://github.com/your-repo/claude-in-the-shell-v2.git
cd claude-in-the-shell-v2

# Create environment with uv (fast)
uv venv && source .venv/bin/activate

# Install with all optional dependencies
uv pip install -e ".[voice,sim,dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### Method 2: Standard pip Install

```bash
# Clone repository
git clone https://github.com/your-repo/claude-in-the-shell-v2.git
cd claude-in-the-shell-v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install
pip install -e .

# Optional: Install extras
pip install -e ".[voice]"  # Voice features
pip install -e ".[sim]"    # MuJoCo simulation
pip install -e ".[dev]"    # Development tools
```

### Method 3: Docker (Coming Soon)

```bash
docker pull ghcr.io/your-repo/reachy-agent:latest
docker run -it --env-file .env reachy-agent
```

## Dependency Groups

### Core Dependencies
Installed by default with `pip install -e .`:

| Package | Purpose |
|---------|---------|
| anthropic | Claude API client |
| chromadb | Vector memory storage |
| sentence-transformers | Text embeddings |
| pydantic | Configuration validation |
| structlog | Structured logging |
| typer | CLI interface |
| rich | Terminal formatting |
| httpx | HTTP client |
| fastmcp | MCP server framework |

### Voice Dependencies
Installed with `pip install -e ".[voice]"`:

| Package | Purpose |
|---------|---------|
| pyaudio | Audio capture/playback |
| openwakeword | Wake word detection |
| websockets | OpenAI Realtime connection |

**Platform-specific notes:**

**Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
PyAudio may require pre-built wheels:
```bash
pip install pipwin
pipwin install pyaudio
```

### Simulation Dependencies
Installed with `pip install -e ".[sim]"`:

| Package | Purpose |
|---------|---------|
| gymnasium[mujoco] | MuJoCo simulation |
| imageio | Video recording |
| imageio-ffmpeg | Video encoding |
| glfw | Window management |

**GPU Support:**
For NVIDIA GPU acceleration:
```bash
pip install mujoco-py
```

### Development Dependencies
Installed with `pip install -e ".[dev]"`:

| Package | Purpose |
|---------|---------|
| pytest | Testing framework |
| pytest-asyncio | Async test support |
| pytest-cov | Coverage reporting |
| black | Code formatting |
| ruff | Linting |
| mypy | Type checking |

## Configuration

### Environment Variables

Create `.env` from template:
```bash
cp .env.example .env
```

Required variables:
```bash
# Required for all modes
ANTHROPIC_API_KEY=sk-ant-api03-...

# Required for voice features
OPENAI_API_KEY=sk-...

# Optional
REACHY_CONFIG_PATH=config/default.yaml
REACHY_LOG_LEVEL=INFO
```

### Configuration File

Edit `config/default.yaml` for advanced settings:

```yaml
agent:
  model: claude-haiku-4-5-20251001
  name: Jarvis
  max_tokens: 4096
  temperature: 0.7

robot:
  backend: mock  # mock, sdk, or sim
  connect_timeout: 5.0

voice:
  personas: [motoko, batou, jarvis]
  wake_sensitivity: 0.5
  silence_threshold: 0.3
  max_listen_time: 30.0

motion:
  tick_hz: 30
  idle:
    speed: 0.1
    amplitude: 0.3
    antenna_drift: 0.2
  wobble:
    intensity: 1.0
    frequency: 4.0

memory:
  path: ~/.reachy/memory
  context_window_size: 5
  embedding_model: all-MiniLM-L6-v2

simulation:
  model_path: data/models/reachy_mini/reachy_mini.xml
  timestep: 0.002
  realtime: true
  viewer: false
```

## Verification

### 1. Health Check

```bash
python -m reachy_agent check
```

Expected output:
```
✅ Python version: 3.11.4
✅ Anthropic API: Connected
✅ ChromaDB: Initialized
✅ Configuration: Loaded
✅ Robot: Mock mode
```

### 2. Run Tests

```bash
# All tests
pytest -v

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/test_memory.py -v
```

### 3. Quick Test

```bash
python -m reachy_agent run --mock
```

Type "hello" and verify you get a response.

## Platform-Specific Notes

### Raspberry Pi (Reachy Mini)

The robot runs on Raspberry Pi 4. For on-device installation:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv portaudio19-dev

# Clone and install
git clone https://github.com/your-repo/claude-in-the-shell-v2.git
cd claude-in-the-shell-v2
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[voice]"
```

For SDK communication:
```bash
# Install Reachy Mini SDK dependencies
pip install reachy-mini-sdk zenoh
```

### macOS Apple Silicon

For M1/M2/M3 Macs:
```bash
# Install Rosetta (if needed for some packages)
softwareupdate --install-rosetta

# Use native ARM builds when available
pip install --prefer-binary -e ".[voice,sim]"
```

### Windows with WSL2

```bash
# In WSL2 Ubuntu
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv portaudio19-dev

# Follow standard Linux installation
```

For audio in WSL2, you may need PulseAudio:
```bash
sudo apt-get install pulseaudio
```

## Troubleshooting Installation

### "pip: command not found"
```bash
python -m pip install -e .
```

### "uv: command not found"
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### ChromaDB SQLite version error
```bash
pip install pysqlite3-binary
```

Add to your script:
```python
import sys
sys.modules["sqlite3"] = __import__("pysqlite3")
```

### PyAudio build failure
**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

### MuJoCo rendering issues
```bash
# For headless rendering
export MUJOCO_GL=egl  # or osmesa
```

### OpenWakeWord model download issues
Models download automatically on first use. If they fail:
```bash
# Manual download
pip install openwakeword
python -c "import openwakeword; openwakeword.download_models()"
```

## Upgrading

### From Source

```bash
cd claude-in-the-shell-v2
git pull origin main
uv pip install -e ".[voice,sim,dev]"
```

### Check for Updates

```bash
pip list --outdated
pip install --upgrade anthropic chromadb
```

---

**Installation complete?** Continue to [Getting Started](../getting-started.md) or jump to the [Quick Start](../quick-start.md).
