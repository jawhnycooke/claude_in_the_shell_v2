# Development Commands Cheat Sheet

## Setup

```bash
# Create environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev,voice]"

# Copy environment template
cp .env.example .env
```

## Running

```bash
# Text mode
python -m reachy_agent run

# Voice mode
python -m reachy_agent run --voice

# Mock hardware
python -m reachy_agent run --mock

# Voice debug
python -m reachy_agent run --voice --debug-voice

# Health check
python -m reachy_agent check
```

## Quality

```bash
# Format
uvx black .
uvx isort . --profile black

# Lint
uvx ruff check .

# Type check
uvx mypy .

# Security
uvx bandit -r src/

# All checks
uvx black . && uvx isort . && uvx ruff check . && uvx mypy .
```

## Testing

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_robot.py -v

# Specific test
pytest tests/test_robot.py::TestMockClient::test_connect -v
```

## MCP Debugging

```bash
# Inspect robot MCP
npx @modelcontextprotocol/inspector python -m reachy_agent.mcp.robot

# Inspect memory MCP
npx @modelcontextprotocol/inspector python -m reachy_agent.mcp.memory
```

## Raspberry Pi

```bash
# Setup (run once)
./scripts/setup_pi.sh

# Test hardware
python scripts/test_hardware.py

# View logs
sudo journalctl -u reachy-agent -f

# Restart service
sudo systemctl restart reachy-agent
```

## Utilities

```bash
# Download emotions
python scripts/download_emotions.py

# Clean build artifacts
rm -rf dist/ build/ *.egg-info/

# Clean cache
rm -rf __pycache__/ .pytest_cache/ .mypy_cache/

# Reset memory
rm -rf ~/.reachy/memory/
```
