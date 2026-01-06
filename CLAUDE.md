# CLAUDE.md - Project Instructions

This is the Claude in the Shell v2 project - an embodied AI agent for the Reachy Mini robot.

## Project Overview

See `specs/01-overview.md` for complete system architecture.

## Development Workflow

1. Follow Python-first development guidelines from `~/.claude/CLAUDE.md`
2. Use `uv` for package management
3. Run `uvx black .` and `uvx ruff check .` before committing
4. All functions must have type hints and docstrings
5. Minimum 80% test coverage required

## Key Directories

- `src/reachy_agent/` - Main source code
- `specs/` - Technical specifications (read these first!)
- `ai_docs/` - Quick reference for AI agents
- `config/` - Configuration files
- `prompts/` - System and persona prompts

## Getting Started with Development

1. Read `specs/01-overview.md` for architecture
2. Read relevant specs for the module you're working on
3. Follow the implementation order from spec 01
4. Use `--mock` mode for development without hardware

## Important Notes

- SDK-only communication (no HTTP fallback)
- Event-driven voice pipeline (not state machine)
- ChromaDB-only for memory (no SQLite)
- 3 permission tiers (removed NOTIFY from v1)
- 30Hz unified motion control loop
