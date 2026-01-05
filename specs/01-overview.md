# Claude in the Shell v2 - System Overview

## What This Is

An embodied AI agent that inhabits a Reachy Mini desktop robot. Claude runs on the robot's Raspberry Pi 4, using MCP tools to control motors, speak, listen, and perceive the world. The robot has personality, memory, and can hold natural voice conversations with barge-in support.

## Design Principles

1. **Simplicity over flexibility** - Fewer options, sensible defaults
2. **Event-driven over state machines** - Easier to debug and extend
3. **Single backends** - One storage system, one communication layer
4. **Convention over configuration** - Auto-discover, don't configure
5. **Fail fast** - Clear errors beat silent fallbacks

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Claude Agent                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Agent Loop                             │   │
│  │         Query → Process → Respond → (repeat)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                  │
│              ▼               ▼               ▼                  │
│      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│      │ Robot MCP   │ │ Memory MCP  │ │ Permissions │           │
│      │ (20 tools)  │ │ (3 tools)   │ │ (3 tiers)   │           │
│      └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    Voice      │     │    Motion     │     │    Memory     │
│   Pipeline    │     │   Control     │     │    Store      │
│  (events)     │     │  (30Hz SDK)   │     │  (ChromaDB)   │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Reachy Mini (SDK/Zenoh)                       │
│   6-DOF Head  │  360° Body  │  2 Antennas  │  Mic + Speaker     │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

| Component | Purpose | Spec |
|-----------|---------|------|
| **Agent Loop** | Claude SDK integration, query processing | [08-agent-loop.md](./08-agent-loop.md) |
| **Robot MCP** | 20 tools for motor/sensor control | [02-robot-control.md](./02-robot-control.md) |
| **Voice Pipeline** | Event-driven voice interaction with barge-in | [03-voice.md](./03-voice.md) |
| **Motion Control** | 30Hz blend controller (idle + wobble) | [04-motion.md](./04-motion.md) |
| **Memory** | ChromaDB-only, 3 types, conversation window | [05-memory.md](./05-memory.md) |
| **Permissions** | 3-tier tool authorization | [06-permissions.md](./06-permissions.md) |
| **Personality** | Behavioral guidelines, emotional expression | [09-personality.md](./09-personality.md) |
| **Observability** | Health endpoints, logging, debugging | [10-observability.md](./10-observability.md) |

## Technology Stack

| Layer | Technology | Note |
|-------|-----------|------|
| AI Model | Claude Haiku 4.5 | Fast responses |
| Agent SDK | claude-agent-sdk | Official SDK |
| MCP | FastMCP | Tool exposure |
| Voice STT/TTS | OpenAI Realtime | WebSocket streaming |
| Wake Word | OpenWakeWord | Multi-model |
| Memory | ChromaDB | Single backend |
| Embeddings | all-MiniLM-L6-v2 | 384 dimensions |
| Hardware | Reachy Mini SDK (Zenoh) | 1-5ms latency |
| Config | Pydantic | Type-safe |

## Key Features

### Voice Interaction
- Multi-persona wake words ("Hey Motoko", "Hey Batou")
- **Barge-in support** - interrupt during TTS
- Event-driven pipeline for easy debugging
- Automatic persona switching

### Robot Control
- 20 MCP tools (streamlined from 27)
- Unified `ReachyClient` interface
- SDK-only communication (no HTTP fallback)
- Tool result caching (200ms)

### Memory
- ChromaDB only (no SQLite)
- 3 memory types: CONVERSATION, FACT, CONTEXT
- Built-in conversation window (last 5 turns)
- Simple search API

### Motion
- 30Hz unified control loop
- 2 motion sources: PRIMARY (idle/emotions) and OVERLAY (wobble)
- Emotion sequences support
- No breathing animation (idle handles "alive" feeling)

### Permissions
- 3 tiers: AUTONOMOUS, CONFIRM, FORBIDDEN
- Glob pattern matching
- Audit logging

## CLI Commands

```bash
# Interactive mode
python -m reachy_agent run

# Voice mode with barge-in
python -m reachy_agent run --voice

# Voice mode with debug logging
python -m reachy_agent run --voice --debug-voice

# Mock mode (no hardware)
python -m reachy_agent run --mock

# Health check
python -m reachy_agent check
```

## Hardware Platform

**Reachy Mini Wireless** (Raspberry Pi 4):
- Head: 6 DOF (pitch, yaw, roll, z, 2 antennas)
- Body: 360° continuous rotation
- Camera: Wide-angle USB
- Audio: 4-mic array + 5W speaker
- Connectivity: WiFi

## Configuration Philosophy

**Convention over configuration.** Most things auto-discover:

```yaml
# config/default.yaml - Keep it minimal
agent:
  model: claude-haiku-4-5-20251001
  name: Jarvis

voice:
  personas: [motoko, batou, jarvis]  # Auto-discovers from prompts/personas/
  wake_sensitivity: 0.5

motion:
  quality: medium  # Preset instead of 4 separate timing params

memory:
  path: ~/.reachy/memory
```

Personas auto-discovered from `prompts/personas/*.md`. Wake word models inferred from persona names. Voices inferred from persona files.

## What's Different from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Voice | 7-state machine | Event-driven |
| Barge-in | No | Yes |
| Memory backend | ChromaDB + SQLite | ChromaDB only |
| Memory types | 6 | 3 |
| Permission tiers | 4 | 3 |
| Motion sources | 4 | 2 |
| Hardware comm | SDK + HTTP fallback | SDK only |
| MCP tools | 27 | 20 |
| Config style | Deep nested YAML | Flat, auto-discover |

## Quick Start (For Implementers)

If you're building this system from these specs, here's the order:

1. **Project setup** ([07-project-structure.md](./07-project-structure.md)) - Create folder structure
2. **Robot client** ([02-robot-control.md](./02-robot-control.md)) - Get hardware talking
3. **Memory system** ([05-memory.md](./05-memory.md)) - Simple, do it early
4. **Permissions** ([06-permissions.md](./06-permissions.md)) - Wire in before tools
5. **Agent loop** ([08-agent-loop.md](./08-agent-loop.md)) - Core coordinator
6. **Motion control** ([04-motion.md](./04-motion.md)) - Make it feel alive
7. **Voice pipeline** ([03-voice.md](./03-voice.md)) - Most complex, do last
8. **Observability** ([10-observability.md](./10-observability.md)) - Add throughout
9. **Personality** ([09-personality.md](./09-personality.md)) - Polish and soul

Start with `--mock` mode. Get text interaction working. Then add voice. Then tune personality.

## All Specifications

| # | Spec | Description |
|---|------|-------------|
| 01 | [overview.md](./01-overview.md) | This file - system architecture |
| 02 | [robot-control.md](./02-robot-control.md) | MCP tools, unified client, caching |
| 03 | [voice.md](./03-voice.md) | Event-driven voice pipeline with barge-in |
| 04 | [motion.md](./04-motion.md) | 30Hz blend controller, idle, wobble |
| 05 | [memory.md](./05-memory.md) | ChromaDB-only, 3 types, context window |
| 06 | [permissions.md](./06-permissions.md) | 3-tier authorization |
| 07 | [project-structure.md](./07-project-structure.md) | Folder layout, conventions |
| 08 | [agent-loop.md](./08-agent-loop.md) | Main coordinator, lifecycle |
| 09 | [personality.md](./09-personality.md) | Behavioral guidelines, emotions |
| 10 | [observability.md](./10-observability.md) | Health, logging, debugging |

## External References

See [ai_docs/REFERENCE.md](./ai_docs/REFERENCE.md) for links to:
- Claude API documentation
- MCP specification
- Reachy Mini SDK
- OpenAI Realtime API
- ChromaDB, OpenWakeWord, and other dependencies
