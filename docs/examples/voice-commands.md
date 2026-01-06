# Voice Command Examples

Working examples for voice interaction and custom wake words.

## Basic Voice Setup

### Enable Voice Mode

```python
import asyncio
from reachy_agent.agent.loop import ReachyAgentLoop
from reachy_agent.agent.options import AgentConfig

async def voice_agent():
    """Start agent with voice interaction."""
    config = AgentConfig(
        model="claude-haiku-4-5-20251001",
        enable_voice=True,
        mock_hardware=True
    )

    agent = ReachyAgentLoop(config)

    try:
        await agent.start()
        print("Say 'Hey Jarvis' to start...")

        # Keep running until interrupted
        await asyncio.Event().wait()

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        await agent.stop()

asyncio.run(voice_agent())
```

### Debug Voice Events

```python
async def voice_debug():
    """Voice with debug logging."""
    config = AgentConfig(
        model="claude-haiku-4-5-20251001",
        enable_voice=True,
        mock_hardware=True,
        debug_voice=True  # Enable event logging
    )

    agent = ReachyAgentLoop(config)
    await agent.start()

    # Events will be logged:
    # [12:34:56.789] voice_event event=wake_detected persona=jarvis
    # [12:34:56.792] voice_event event=listening_start persona=jarvis
    # ...

    await asyncio.Event().wait()

asyncio.run(voice_debug())
```

---

## Custom Wake Words

### Single Custom Wake Word

```python
from reachy_agent.voice.wake import WakeWordDetector

async def custom_wake_word():
    """Use a custom wake word."""
    detector = WakeWordDetector(
        models=["my_custom_wake_word"],
        on_detected=handle_wake,
        sensitivity=0.6
    )

    detector.enable()
    print("Listening for custom wake word...")

    # Keep running
    await asyncio.Event().wait()

async def handle_wake(model: str, confidence: float):
    """Handle wake word detection."""
    print(f"Wake word detected! Model: {model}, Confidence: {confidence:.2f}")

asyncio.run(custom_wake_word())
```

### Multiple Personas

```python
from reachy_agent.voice.persona import PersonaManager
from pathlib import Path

async def multi_persona():
    """Use multiple personas with different wake words."""
    # Personas auto-discovered from directory
    personas = PersonaManager(Path("prompts/personas"))

    print("Available personas:")
    for name, persona in personas._personas.items():
        print(f"  - {name}: wake word '{persona.wake_word}'")

    # Get wake word models for detector
    wake_models = personas.get_wake_models()
    print(f"\nListening for: {wake_models}")

asyncio.run(multi_persona())
```

---

## Persona Configuration

### Create a Custom Persona

Create `prompts/personas/assistant.md`:

```markdown
---
name: Assistant
wake_word: hey_assistant
voice: nova
---

# Custom Assistant

You are a helpful assistant named Assistant. You are running on a Reachy Mini robot.

Your personality traits:
- Friendly and approachable
- Clear and concise in communication
- Helpful but not overly eager

When responding:
- Use natural, conversational language
- Keep responses brief unless asked for detail
- Use the robot's movement capabilities to express yourself
```

### Load Custom Persona

```python
from reachy_agent.voice.persona import PersonaManager, Persona
from pathlib import Path

async def load_custom_persona():
    """Load and use a custom persona."""
    personas = PersonaManager(Path("prompts/personas"))

    # Switch to custom persona
    await personas.switch_to("assistant")

    # Get voice for TTS
    voice = personas.get_voice("assistant")
    print(f"Using voice: {voice}")

    # Get system prompt
    persona = personas._personas["assistant"]
    print(f"System prompt preview: {persona.system_prompt[:100]}...")

asyncio.run(load_custom_persona())
```

---

## Voice Pipeline Events

### Custom Event Handlers

```python
from reachy_agent.voice.pipeline import VoicePipeline

async def custom_event_handlers():
    """Add custom handlers to voice events."""
    pipeline = VoicePipeline(agent=None, persona_manager=None)

    @pipeline.on("wake_detected")
    async def on_wake(event):
        persona = event.data["persona"]
        confidence = event.data["confidence"]
        print(f"Wake! Persona: {persona}, Confidence: {confidence:.2f}")

    @pipeline.on("transcribed")
    async def on_transcribed(event):
        text = event.data["text"]
        print(f"User said: {text}")

    @pipeline.on("response")
    async def on_response(event):
        text = event.data["text"]
        print(f"Agent response: {text}")

    @pipeline.on("error")
    async def on_error(event):
        error_type = event.data["type"]
        message = event.data["message"]
        print(f"Error ({error_type}): {message}")

    await pipeline.start()

asyncio.run(custom_event_handlers())
```

### Barge-in Handler

```python
async def barge_in_demo():
    """Handle barge-in (interruption) events."""
    pipeline = VoicePipeline(agent=None, persona_manager=None)

    @pipeline.on("interrupted")
    async def on_interrupt(event):
        interrupted_by = event.data.get("by", "unknown")
        print(f"Interrupted by: {interrupted_by}")

        if interrupted_by == "wake_word":
            print("User wants to say something new!")

    @pipeline.on("speaking_start")
    async def on_speaking(event):
        text = event.data["text"]
        print(f"Starting to say: {text[:50]}...")

    @pipeline.on("speaking_end")
    async def on_speaking_end(event):
        print("Finished speaking")

    await pipeline.start()

asyncio.run(barge_in_demo())
```

---

## Text-to-Speech

### Direct TTS

```python
from reachy_agent.robot.factory import create_client, Backend

async def tts_demo():
    """Direct text-to-speech."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Default voice
    await client.speak("Hello! I am Reachy.")

    # With specific voice
    await client.speak("This is a different voice.", voice="nova")

    await client.disconnect()

asyncio.run(tts_demo())
```

### TTS with Expression

```python
async def expressive_tts():
    """Combine TTS with expressions."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Happy greeting
    await asyncio.gather(
        client.speak("Hello! Nice to meet you!"),
        client.play_emotion("happy")
    )

    # Curious question
    await asyncio.gather(
        client.speak("What would you like to know?"),
        client.play_emotion("curious")
    )

    # Thoughtful response
    await asyncio.gather(
        client.speak("Let me think about that..."),
        client.play_emotion("thinking")
    )

    await client.disconnect()

asyncio.run(expressive_tts())
```

---

## Speech-to-Text

### Listen for Input

```python
async def stt_demo():
    """Listen for speech input."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    # Listen with default timeout
    text = await client.listen()
    print(f"You said: {text}")

    # Listen with custom timeout
    text = await client.listen(timeout=10.0)
    print(f"You said: {text}")

    await client.disconnect()

asyncio.run(stt_demo())
```

### Conversation Loop

```python
async def conversation_loop():
    """Simple voice conversation loop."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    await client.speak("Hello! I'm ready to chat. Say 'goodbye' to end.")

    while True:
        # Listen for input
        user_input = await client.listen(timeout=30.0)

        if not user_input:
            await client.speak("I didn't catch that. Could you repeat?")
            continue

        if "goodbye" in user_input.lower():
            await client.speak("Goodbye! It was nice talking to you.")
            break

        # Echo back (in real use, send to Claude)
        await client.speak(f"You said: {user_input}")

    await client.disconnect()

asyncio.run(conversation_loop())
```

---

## Sound Direction Detection

### React to Sound

```python
async def sound_direction():
    """Detect and react to sound direction."""
    client = create_client(Backend.MOCK)
    await client.connect()
    await client.wake_up()

    print("Making sounds around the robot...")

    for _ in range(10):
        # Get sound direction
        azimuth, confidence = await client.detect_sound_direction()

        if confidence > 0.5:
            print(f"Sound at {azimuth:.1f}°, confidence: {confidence:.2f}")

            # Turn toward sound
            await client.move_head(yaw=azimuth, duration=0.5)

        await asyncio.sleep(0.5)

    await client.disconnect()

asyncio.run(sound_direction())
```

---

## Complete Voice Demo

```python
import asyncio
from reachy_agent.agent.loop import ReachyAgentLoop
from reachy_agent.agent.options import AgentConfig
from reachy_agent.voice.pipeline import VoicePipeline

async def full_voice_demo():
    """Complete voice interaction demo."""
    config = AgentConfig(
        model="claude-haiku-4-5-20251001",
        enable_voice=True,
        enable_motion=True,
        mock_hardware=True,
        debug_voice=True
    )

    agent = ReachyAgentLoop(config)

    # Add custom event handlers
    @agent.voice_pipeline.on("wake_detected")
    async def on_wake(event):
        persona = event.data["persona"]
        print(f"\n>>> Activated: {persona}")

    @agent.voice_pipeline.on("transcribed")
    async def on_transcribed(event):
        print(f">>> Heard: {event.data['text']}")

    @agent.voice_pipeline.on("response")
    async def on_response(event):
        print(f">>> Response: {event.data['text'][:100]}...")

    try:
        await agent.start()
        print("\n" + "=" * 50)
        print("Voice Demo Running")
        print("Say 'Hey Jarvis' to start")
        print("Press Ctrl+C to exit")
        print("=" * 50 + "\n")

        await asyncio.Event().wait()

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        await agent.stop()

asyncio.run(full_voice_demo())
```

---

## Environment Setup

Required environment variables for voice:

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...  # For Realtime API
```

Required dependencies:

```bash
uv pip install -e ".[voice]"
```

---

## Next Steps

- [Simulation Scenarios](simulation-scenarios.md) - Test voice in simulation
- [Voice Pipeline Architecture](../developer-guide/voice-pipeline.md) - Deep dive
- [Voice Control User Guide](../user-guide/voice-control.md) - User documentation
