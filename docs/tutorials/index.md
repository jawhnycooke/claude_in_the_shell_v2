# Tutorials

Step-by-step guides to help you master Claude in the Shell v2.

## Getting Started

| Tutorial | Time | Description |
|----------|------|-------------|
| [First Robot Movement](first-robot-movement.md) | 10 min | Control head, body, and antennas |
| [Voice Interaction](voice-interaction.md) | 15 min | Set up wake words and conversations |
| [Emotion Expressions](emotion-expressions.md) | 10 min | Play emotions and create sequences |

## Advanced Topics

| Tutorial | Time | Description |
|----------|------|-------------|
| [MuJoCo Setup](mujoco-setup.md) | 20 min | Physics simulation for development |
| [Custom Behaviors](custom-behaviors.md) | 30 min | Create custom motion behaviors |

## Prerequisites

Before starting these tutorials, ensure you have:

1. **Installed the package**: See [Installation Guide](../user-guide/installation.md)
2. **Configured API keys**: Set `ANTHROPIC_API_KEY` in `.env`
3. **Verified installation**: Run `python -m reachy_agent check`

## Tutorial Conventions

### Code Blocks

Python code you can run:
```python
# This is executable Python code
await client.move_head(pitch=10, yaw=0)
```

Shell commands:
```bash
# Run in your terminal
python -m reachy_agent run --mock
```

Agent conversations:
```
> This is what you type
This is what the agent responds
```

### Icons

- :material-rocket-launch: - Quick start
- :material-lightbulb: - Tip or best practice
- :material-alert: - Important warning
- :material-check: - Completed step

## Learning Path

### Beginner

1. [First Robot Movement](first-robot-movement.md) - Basic movement control
2. [Emotion Expressions](emotion-expressions.md) - Making the robot expressive
3. [Voice Interaction](voice-interaction.md) - Voice-based conversations

### Intermediate

4. [MuJoCo Setup](mujoco-setup.md) - Development with simulation
5. [Custom Behaviors](custom-behaviors.md) - Creating new behaviors

### Advanced

- [API Reference](../api-reference/index.md) - Complete API documentation
- [Developer Guide](../developer-guide/index.md) - Architecture and internals

---

**Ready to start?** Begin with [First Robot Movement](first-robot-movement.md).
