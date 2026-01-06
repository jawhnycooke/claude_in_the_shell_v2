# Contributing Guide

How to contribute to Claude in the Shell v2.

## Getting Started

### Prerequisites

- Python 3.10+
- uv (recommended) or pip
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/claude-in-the-shell-v2.git
cd claude-in-the-shell-v2

# Create virtual environment
uv venv && source .venv/bin/activate

# Install with all development dependencies
uv pip install -e ".[dev,voice,sim]"

# Copy environment template
cp .env.example .env
# Add your API keys
```

### Verify Setup

```bash
# Run health check
python -m reachy_agent check

# Run tests
pytest -v

# Check code quality
uvx black . --check
uvx ruff check .
uvx mypy .
```

---

## Development Workflow

### 1. Create a Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes
git checkout -b fix/issue-123-description
```

**Branch naming:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 2. Make Changes

Follow the code style guidelines below.

### 3. Test Your Changes

```bash
# Run all tests
pytest -v

# Run specific tests
pytest tests/test_module.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### 4. Check Code Quality

```bash
# Format code
uvx black .
uvx isort . --profile black

# Lint
uvx ruff check .

# Type check
uvx mypy .
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new emotion animation support

- Added EmotionSequence class for chaining emotions
- Updated play_emotion tool with sequence support
- Added tests for new functionality"
```

**Commit message format:**
```
<type>: <short description>

<longer description if needed>

<references to issues if applicable>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting
- `refactor` - Code restructuring
- `test` - Tests
- `chore` - Maintenance

### 6. Create Pull Request

```bash
# Push branch
git push -u origin feature/your-feature-name
```

Then create a PR on GitHub with:
- Clear description of changes
- Reference to any related issues
- Screenshots/videos if UI changes
- Test results

---

## Code Style

### Python Style

We follow PEP 8 with Black formatting:

```python
# Good
from dataclasses import dataclass
from typing import Any

@dataclass
class HeadPose:
    """Absolute head position in degrees."""

    pitch: float = 0.0  # -45 to +35
    yaw: float = 0.0    # -60 to +60
    roll: float = 0.0   # -35 to +35
    z: float = 0.0      # 0 to 50 mm


async def move_head(
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    duration: float = 1.0,
) -> dict[str, Any]:
    """
    Move head to absolute position.

    Args:
        pitch: Head tilt up/down in degrees.
        yaw: Head turn left/right in degrees.
        roll: Head tilt sideways in degrees.
        duration: Movement time in seconds.

    Returns:
        Success status and final position.
    """
    ...
```

### Type Hints

Always use type hints:

```python
# Good
def process(self, user_input: str) -> str:
    ...

# Good with complex types
def search(
    self,
    query: str,
    memory_type: MemoryType | None = None,
    limit: int = 5,
) -> list[SearchResult]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def store_memory(
    content: str,
    memory_type: str = "fact",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Store a new memory with semantic embedding.

    Args:
        content: Text content to remember.
        memory_type: One of "fact", "conversation", or "context".
        metadata: Optional key-value pairs for additional context.

    Returns:
        The stored memory with its generated ID.

    Raises:
        ValueError: If content is empty.

    Examples:
        >>> result = await store_memory("User likes blue", "fact")
        >>> print(result["memory_id"])
        "abc123..."
    """
```

### Error Handling

```python
# Good - specific exceptions with context
class NotAwakeError(RobotError):
    """Attempted movement while robot is asleep."""
    pass

async def move_head(self, ...):
    if not self._awake:
        raise NotAwakeError("Robot must be awake to move. Call wake_up() first.")

# Good - handle and log appropriately
try:
    await self._robot.move_head(...)
except ConnectionError as e:
    self._log.error("move_failed", error=str(e))
    raise
```

### Async/Await

Use async consistently:

```python
# Good
async def process(self, input: str) -> str:
    result = await self._call_api(input)
    await self._log_result(result)
    return result

# Avoid blocking in async code
# Bad
def get_data(self):
    return requests.get(url)  # Blocks!

# Good
async def get_data(self):
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

---

## Testing Guidelines

### Test Structure

```python
# tests/test_memory.py

import pytest
from reachy_agent.memory.manager import MemoryManager, MemoryType


class TestMemoryStore:
    """Tests for memory storage operations."""

    @pytest.fixture
    def manager(self):
        """Create fresh memory manager for each test."""
        return MemoryManager(":memory:")

    async def test_store_fact_memory(self, manager):
        """Facts should store with no expiry."""
        memory = await manager.store(
            content="User prefers dark mode",
            memory_type=MemoryType.FACT,
        )

        assert memory.id is not None
        assert memory.content == "User prefers dark mode"
        assert memory.memory_type == MemoryType.FACT
        assert memory.expires_at is None

    async def test_store_with_metadata(self, manager):
        """Metadata should be preserved."""
        memory = await manager.store(
            content="Test content",
            memory_type=MemoryType.FACT,
            metadata={"category": "test"},
        )

        assert memory.metadata["category"] == "test"


class TestMemorySearch:
    """Tests for memory search operations."""

    @pytest.fixture
    async def manager_with_data(self):
        """Manager with pre-populated test data."""
        manager = MemoryManager(":memory:")
        await manager.store("User likes blue", MemoryType.FACT)
        await manager.store("User likes green", MemoryType.FACT)
        await manager.store("Weather discussion", MemoryType.CONVERSATION)
        return manager

    async def test_search_finds_relevant(self, manager_with_data):
        """Search should return relevant results."""
        results = await manager_with_data.search("color preferences")

        assert len(results) > 0
        assert any("blue" in r.memory.content for r in results)
```

### Test Categories

```bash
# Unit tests (fast, isolated)
pytest tests/test_memory.py -v

# Integration tests (slower, component interaction)
pytest tests/test_integration.py -v

# Simulation tests (requires MuJoCo)
pytest tests/test_simulation.py -v

# All tests with coverage
pytest --cov=src --cov-report=html
```

### Fixtures

Use pytest fixtures for common setup:

```python
# conftest.py
import pytest
from reachy_agent.robot.mock import MockClient

@pytest.fixture
def mock_client():
    """Provide connected mock client."""
    client = MockClient()
    asyncio.run(client.connect())
    yield client

@pytest.fixture
def agent_config():
    """Provide test configuration."""
    return AgentConfig(
        model="claude-haiku-4-5-20251001",
        mock_hardware=True,
        enable_voice=False,
    )
```

---

## Documentation

### Updating Docs

Documentation is in `docs/` using Markdown:

```bash
# View documentation
open docs/index.md

# After changes, verify links work
# (Future: mkdocs serve)
```

### Documentation Style

- Use clear, concise language
- Include code examples that work
- Add diagrams for complex concepts (use Mermaid)
- Cross-link related pages
- Update when code changes

---

## Pull Request Checklist

Before submitting a PR:

- [ ] Code follows style guidelines
- [ ] All tests pass (`pytest -v`)
- [ ] Code is formatted (`uvx black .`)
- [ ] No lint errors (`uvx ruff check .`)
- [ ] Type hints pass (`uvx mypy .`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] PR description explains changes

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@example.com

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow project guidelines

Thank you for contributing!
