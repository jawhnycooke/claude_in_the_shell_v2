# Quick Start

Get up and running with Claude in the Shell in 5 minutes.

## TL;DR

```bash
# Install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Run
python -m reachy_agent run --mock
```

## Step 1: Install (1 minute)

```bash
# Clone and enter directory
git clone https://github.com/your-repo/claude-in-the-shell-v2.git
cd claude-in-the-shell-v2

# Create environment and install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Step 2: Configure (30 seconds)

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

## Step 3: Run (30 seconds)

```bash
python -m reachy_agent run --mock
```

## Step 4: Chat!

```
🤖 Reachy Agent (type 'quit' to exit)

> Hello! What can you do?

Hello! I'm Jarvis, an embodied AI agent. Here's what I can do:

**Movement**
- Move my head in any direction (pitch, yaw, roll)
- Rotate my body 360°
- Control my antennas for expression

**Emotions**
- Display emotions: happy, sad, curious, surprised, thinking
- Chain emotions in sequences
- Nod yes or shake no

**Memory**
- Remember facts about you (permanent)
- Track our conversation (30 days)
- Store session context (today only)

**Perception**
- Capture camera images
- Detect sound direction
- Read sensor data

What would you like to try?

> Move your head to look up and to the left

*moving head*
I've moved my head to look up (pitch: 20°) and to the left (yaw: -30°).

> Remember that my name is Alex

I'll remember that! I've stored that your name is Alex as a permanent fact.

> Play happy emotion

*plays happy animation*
😊 That felt good! I expressed happiness for you, Alex.
```

## What's Next?

### Try Different Modes

```bash
# Simulation with physics (requires MuJoCo)
python -m reachy_agent run --sim --sim-viewer

# Voice interaction (requires OpenAI key)
python -m reachy_agent run --mock --voice

# Debug voice events
python -m reachy_agent run --mock --voice --debug-voice
```

### Try More Commands

```
> Look at the point 0.5 meters ahead and slightly down

> Play a sequence of emotions: curious, thinking, happy

> What do you remember about me?

> Rotate your body 180 degrees

> Reset to neutral position
```

### Explore the Documentation

| Guide | Description |
|-------|-------------|
| [Robot Movements](tutorials/first-robot-movement.md) | Control head, body, antennas |
| [Voice Interaction](tutorials/voice-interaction.md) | Set up wake words and talking |
| [Emotion Expressions](tutorials/emotion-expressions.md) | Play emotions and sequences |
| [Memory System](user-guide/memory-system.md) | Store and search memories |
| [MuJoCo Simulation](tutorials/mujoco-setup.md) | Physics simulation setup |
| [Code Examples](examples/index.md) | Working code examples |

### Check the API

```
> What tools do you have?

I have 20 robot tools and 3 memory tools:

**Movement**: move_head, look_at, rotate_body, reset_position
**Expression**: play_emotion, play_sequence, set_antennas, nod, shake
**Audio**: speak, listen
**Perception**: capture_image, get_sensor_data, detect_sound_direction
**Lifecycle**: wake_up, sleep, is_awake
**Status**: get_status, get_position, get_limits

**Memory**: store_memory, search_memory, forget_memory
```

## Quick Reference

### CLI Commands

```bash
# Run modes
python -m reachy_agent run              # Interactive (hardware required)
python -m reachy_agent run --mock       # Mock mode (no hardware)
python -m reachy_agent run --sim        # MuJoCo simulation
python -m reachy_agent run --voice      # Voice interaction

# Utilities
python -m reachy_agent check            # Health check
python -m reachy_agent version          # Show version
```

### Common Tool Parameters

| Tool | Key Parameters |
|------|----------------|
| `move_head` | pitch (-45 to +35), yaw (-60 to +60), roll (-35 to +35), duration |
| `look_at` | x, y, z (meters), duration |
| `rotate_body` | angle (0-360), duration |
| `play_emotion` | emotion_name (happy, sad, curious, surprised, thinking) |
| `store_memory` | content, memory_type (fact, conversation, context) |
| `search_memory` | query, memory_type (optional), limit |

### Hardware Limits

| Joint | Range | Unit |
|-------|-------|------|
| Head Pitch | -45 to +35 | degrees |
| Head Yaw | -60 to +60 | degrees |
| Head Roll | -35 to +35 | degrees |
| Head Z | 0 to 50 | mm |
| Body Rotation | 0 to 360 | degrees |
| Antennas | -150 to +150 | degrees |

---

**Need more detail?** Check the [Getting Started Guide](getting-started.md) or dive into the [API Reference](api-reference/index.md).
