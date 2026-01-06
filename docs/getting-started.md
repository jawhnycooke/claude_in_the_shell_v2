# Getting Started

This guide will help you install Claude in the Shell v2 and run your first conversation with the Reachy agent.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **uv** package manager (recommended) or pip
- **API keys** for Anthropic (Claude) and OpenAI (voice features)
- **Reachy Mini robot** (optional - use `--mock` or `--sim` mode for development)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/claude-in-the-shell-v2.git
cd claude-in-the-shell-v2
```

### 2. Create Virtual Environment

Using uv (recommended):
```bash
uv venv && source .venv/bin/activate
```

Or with standard Python:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

**Basic installation:**
```bash
uv pip install -e .
```

**With voice support:**
```bash
uv pip install -e ".[voice]"
```

**With simulation support:**
```bash
uv pip install -e ".[sim]"
```

**Full installation (all features + development):**
```bash
uv pip install -e ".[voice,sim,dev]"
```

### 4. Configure Environment

Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# For voice features
OPENAI_API_KEY=sk-...

# Optional: Custom configuration
REACHY_CONFIG_PATH=config/default.yaml
```

## Verify Installation

Run the health check to verify everything is working:

```bash
python -m reachy_agent check
```

You should see output like:

```
✅ Python version: 3.10.12
✅ Anthropic API: Connected
✅ ChromaDB: Initialized
✅ Configuration: Loaded
✅ Robot: Mock mode (no hardware)
```

## Your First Conversation

### Text Mode (No Hardware Required)

Start the agent in mock mode for testing without hardware:

```bash
python -m reachy_agent run --mock
```

You'll see the interactive prompt:

```
🤖 Reachy Agent (type 'quit' to exit)

>
```

Try some commands:

```
> Hello, who are you?

I'm Jarvis, an embodied AI agent inhabiting a Reachy Mini robot. I can
move my head, express emotions, remember things about our conversations,
and help you with various tasks.

> Can you nod for me?

*nods enthusiastically*
I've nodded to acknowledge your request!

> What can you do?

I have several capabilities:

**Movement**: I can move my head (pitch, yaw, roll), rotate my body 360°,
and use my antennas for expression.

**Emotions**: I can play various emotion animations like happy, curious,
surprised, thinking, and sad.

**Memory**: I can remember things you tell me - facts you want me to
remember permanently, or conversation context that fades over time.

**Perception**: I can capture images from my camera, detect sound
directions, and read sensor data.

Would you like to try any of these?
```

### Simulation Mode (Physics Simulation)

Run with MuJoCo physics simulation:

```bash
python -m reachy_agent run --sim
```

With 3D viewer:

```bash
python -m reachy_agent run --sim --sim-viewer
```

### Voice Mode

If you have voice dependencies installed and an OpenAI API key:

```bash
python -m reachy_agent run --mock --voice
```

Say "Hey Jarvis" (or "Hey Motoko", "Hey Batou") to wake the robot and start talking!

## Configuration

The agent uses sensible defaults, but you can customize behavior in `config/default.yaml`:

```yaml
agent:
  model: claude-haiku-4-5-20251001
  name: Jarvis
  max_tokens: 4096
  temperature: 0.7

voice:
  personas: [motoko, batou, jarvis]
  wake_sensitivity: 0.5

motion:
  tick_hz: 30
  idle:
    speed: 0.1
    amplitude: 0.3

memory:
  path: ~/.reachy/memory
  context_window_size: 5
```

## Project Structure

```
claude-in-the-shell-v2/
├── config/              # Configuration files
├── prompts/             # System and persona prompts
│   └── personas/        # Persona definitions (motoko.md, batou.md, etc.)
├── src/reachy_agent/    # Main source code
│   ├── agent/           # Agent core (loop, options)
│   ├── mcp/             # MCP servers (robot, memory)
│   ├── robot/           # Robot hardware abstraction
│   ├── voice/           # Voice pipeline
│   ├── motion/          # Motion control
│   ├── memory/          # Memory system
│   ├── permissions/     # Permission system
│   └── simulation/      # MuJoCo simulation
├── data/                # Static data (emotions, scenarios)
├── tests/               # Test suite (397 tests)
└── docs/                # Documentation (you are here!)
```

## Next Steps

Now that you have the agent running:

1. **[Quick Start Tutorial](quick-start.md)** - 5-minute hands-on tutorial
2. **[Robot Movement Tutorial](tutorials/first-robot-movement.md)** - Learn to control the robot
3. **[Voice Control Guide](user-guide/voice-control.md)** - Set up voice interaction
4. **[Memory System Guide](user-guide/memory-system.md)** - Understand how memory works
5. **[API Reference](api-reference/index.md)** - Full documentation of all tools

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'reachy_agent'"**
- Ensure you installed with `-e` flag: `uv pip install -e .`
- Verify your virtual environment is activated

**"ANTHROPIC_API_KEY not found"**
- Check your `.env` file exists and has the correct key
- Try: `export ANTHROPIC_API_KEY=your-key` directly

**"ChromaDB initialization failed"**
- Delete the memory directory: `rm -rf ~/.reachy/memory`
- This resets all stored memories

**"Robot connection failed" (with real hardware)**
- Ensure the robot is powered on and connected to WiFi
- Check the robot's IP address in configuration
- Try `--mock` mode to test without hardware

### Debug Mode

For detailed logging, use the `--debug-voice` flag:

```bash
python -m reachy_agent run --mock --voice --debug-voice
```

This shows every voice pipeline event with timestamps.

### Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: You're reading it!

---

**Ready to dive deeper?** Continue to the [Quick Start Tutorial](quick-start.md) for a hands-on walkthrough.
