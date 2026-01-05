# Voice Pipeline Specification

## Overview

The voice pipeline enables natural voice conversations with barge-in support. Instead of a rigid state machine, it uses an **event-driven architecture** where each event triggers handlers that do one thing well. This makes debugging easier and extending behavior straightforward.

## Design Principles

1. **Event-driven** - No explicit state machine. State is implicit in which handlers are active.
2. **Barge-in support** - Wake word detection runs during TTS. Users can interrupt.
3. **Debug-friendly** - Every event logged with timestamp. `--debug-voice` shows full trace.
4. **Persona auto-discovery** - Personas discovered from `prompts/personas/*.md`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Voice Pipeline                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Event Emitter                           │   │
│  │                                                           │   │
│  │  Events:  wake_detected → listening → transcribed →       │   │
│  │           processing → response → speaking → [interrupt]  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│      ┌───────────────────────┼───────────────────────┐          │
│      │                       │                       │          │
│      ▼                       ▼                       ▼          │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐        │
│  │   Wake     │      │  OpenAI    │      │   Audio    │        │
│  │   Word     │      │  Realtime  │      │  Manager   │        │
│  │  Detector  │      │   Client   │      │            │        │
│  └────────────┘      └────────────┘      └────────────┘        │
│  OpenWakeWord        WebSocket STT/TTS    PyAudio streams      │
└─────────────────────────────────────────────────────────────────┘
```

## Events

| Event | Payload | Description |
|-------|---------|-------------|
| `wake_detected` | `{persona: str, confidence: float}` | Wake word triggered |
| `listening_start` | `{persona: str}` | Started capturing audio |
| `listening_end` | `{audio_duration: float}` | Stopped capturing (silence detected) |
| `transcribed` | `{text: str, confidence: float}` | Speech-to-text complete |
| `processing` | `{text: str}` | Sending to Claude |
| `response` | `{text: str}` | Claude response received |
| `speaking_start` | `{text: str, voice: str}` | TTS playback starting |
| `speaking_end` | `{}` | TTS playback complete |
| `interrupted` | `{by: str}` | TTS interrupted (wake word or error) |
| `error` | `{type: str, message: str}` | Error occurred |
| `timeout` | `{phase: str}` | Timeout during phase |

## Event Flow

### Normal Conversation

```
wake_detected(persona="motoko")
    │
    ▼
listening_start(persona="motoko")
    │  [user speaks]
    ▼
listening_end(audio_duration=2.3)
    │
    ▼
transcribed(text="What's the weather like?")
    │
    ▼
processing(text="What's the weather like?")
    │  [Claude thinks]
    ▼
response(text="I don't have access to weather data...")
    │
    ▼
speaking_start(text="I don't have...", voice="nova")
    │  [TTS plays]
    ▼
speaking_end()
    │
    └──► [returns to wake word detection]
```

### Barge-in (Interrupt)

```
speaking_start(text="Let me tell you about...")
    │  [TTS playing]
    │
    │  ◄── wake_detected(persona="motoko")  [user says "Hey Motoko"]
    │
    ▼
interrupted(by="wake_word")
    │
    ▼
listening_start(persona="motoko")
    │
    └──► [continues normal flow]
```

## Core Implementation

### Event Emitter Base

```python
from dataclasses import dataclass, field
from typing import Callable, Any
from collections import defaultdict
import asyncio
import structlog

@dataclass
class Event:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

class EventEmitter:
    def __init__(self, debug: bool = False):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._debug = debug
        self._log = structlog.get_logger()

    def on(self, event_name: str):
        """Decorator to register event handler."""
        def decorator(fn: Callable):
            self._handlers[event_name].append(fn)
            return fn
        return decorator

    async def emit(self, event_name: str, **data):
        """Emit event to all registered handlers."""
        event = Event(name=event_name, data=data)

        if self._debug:
            self._log.debug("voice_event", event=event_name, **data)

        for handler in self._handlers[event_name]:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                await self.emit("error", type=type(e).__name__, message=str(e))
```

### Voice Pipeline

```python
class VoicePipeline(EventEmitter):
    """Event-driven voice interaction pipeline."""

    def __init__(
        self,
        agent: "ReachyAgentLoop",
        persona_manager: "PersonaManager",
        debug: bool = False
    ):
        super().__init__(debug=debug)
        self._agent = agent
        self._personas = persona_manager
        self._wake_detector: WakeWordDetector | None = None
        self._realtime: OpenAIRealtimeClient | None = None
        self._audio: AudioManager | None = None
        self._current_persona: str | None = None
        self._is_speaking = False

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register all event handlers."""

        @self.on("wake_detected")
        async def handle_wake(event: Event):
            persona = event.data["persona"]
            self._current_persona = persona

            # Switch persona context
            await self._personas.switch_to(persona)

            # Start listening
            await self.emit("listening_start", persona=persona)

        @self.on("listening_start")
        async def handle_listening_start(event: Event):
            # Start audio capture
            self._audio.start_capture()

            # Start STT session
            await self._realtime.start_session(
                voice=self._personas.get_voice(self._current_persona)
            )

        @self.on("listening_end")
        async def handle_listening_end(event: Event):
            # Stop capture and get transcription
            audio_data = self._audio.stop_capture()

            # Send to OpenAI Realtime for transcription
            text = await self._realtime.transcribe(audio_data)

            if text.strip():
                await self.emit("transcribed", text=text, confidence=0.9)
            else:
                await self.emit("timeout", phase="transcription")

        @self.on("transcribed")
        async def handle_transcribed(event: Event):
            text = event.data["text"]
            await self.emit("processing", text=text)

            # Send to Claude
            response = await self._agent.process(text)
            await self.emit("response", text=response)

        @self.on("response")
        async def handle_response(event: Event):
            text = event.data["text"]
            voice = self._personas.get_voice(self._current_persona)

            await self.emit("speaking_start", text=text, voice=voice)

        @self.on("speaking_start")
        async def handle_speaking_start(event: Event):
            self._is_speaking = True

            # Enable wake word detection during speech (for barge-in)
            self._wake_detector.enable()

            # Start TTS playback
            await self._realtime.speak(
                event.data["text"],
                voice=event.data["voice"],
                on_complete=lambda: asyncio.create_task(
                    self.emit("speaking_end")
                )
            )

        @self.on("speaking_end")
        async def handle_speaking_end(event: Event):
            self._is_speaking = False
            # Return to passive wake word detection
            # (already enabled, just waiting)

        @self.on("interrupted")
        async def handle_interrupted(event: Event):
            self._is_speaking = False

            # Stop TTS immediately
            await self._realtime.stop_speaking()

            # If interrupted by wake word, start listening
            if event.data.get("by") == "wake_word":
                await self.emit("listening_start", persona=self._current_persona)

    async def start(self):
        """Start the voice pipeline."""
        # Initialize components
        self._audio = AudioManager()
        self._realtime = OpenAIRealtimeClient()
        self._wake_detector = WakeWordDetector(
            models=self._personas.get_wake_models(),
            on_detected=self._on_wake_word
        )

        await self._audio.initialize()
        await self._realtime.connect()

        # Start wake word detection
        self._wake_detector.enable()

    async def _on_wake_word(self, model: str, confidence: float):
        """Called when wake word detected."""
        persona = self._personas.persona_for_model(model)

        if self._is_speaking:
            # Barge-in: interrupt current speech
            await self.emit("interrupted", by="wake_word")

        await self.emit("wake_detected", persona=persona, confidence=confidence)

    async def stop(self):
        """Stop the voice pipeline."""
        self._wake_detector.disable()
        await self._realtime.disconnect()
        self._audio.cleanup()
```

## Barge-in Implementation

The key to natural conversation is allowing interruption during TTS:

```python
class WakeWordDetector:
    """Continuous wake word detection with barge-in support."""

    def __init__(
        self,
        models: list[str],
        on_detected: Callable[[str, float], None],
        sensitivity: float = 0.5
    ):
        self._models = models
        self._on_detected = on_detected
        self._sensitivity = sensitivity
        self._enabled = False
        self._oww: openwakeword.Model | None = None

    def enable(self):
        """Enable wake word detection."""
        if not self._oww:
            self._oww = openwakeword.Model(
                wakeword_models=self._models,
                inference_framework="onnx"
            )
        self._enabled = True

    def disable(self):
        """Disable wake word detection."""
        self._enabled = False

    async def process_audio(self, audio_chunk: bytes):
        """Process audio chunk for wake words."""
        if not self._enabled:
            return

        # Run inference
        predictions = self._oww.predict(audio_chunk)

        for model, confidence in predictions.items():
            if confidence > self._sensitivity:
                await self._on_detected(model, confidence)
                # Don't disable - allow re-triggering for barge-in
```

**Key insight**: Wake word detection stays enabled during TTS. When detected during speech, it triggers an `interrupted` event that stops TTS and transitions to listening.

## Persona System

Personas are auto-discovered from `prompts/personas/`.

```
prompts/personas/
├── motoko.md     # Wake word: "hey_motoko", Voice: "nova"
├── batou.md      # Wake word: "hey_batou", Voice: "onyx"
└── jarvis.md     # Wake word: "hey_jarvis", Voice: "echo"
```

### Persona File Format

```markdown
---
name: Motoko
wake_word: hey_motoko
voice: nova
---

# Major Motoko Kusanagi

You are Major Motoko Kusanagi from Ghost in the Shell...
[system prompt content]
```

### Persona Manager

```python
@dataclass
class Persona:
    name: str
    wake_word: str
    voice: str
    system_prompt: str

class PersonaManager:
    def __init__(self, personas_dir: Path = Path("prompts/personas")):
        self._personas: dict[str, Persona] = {}
        self._current: str | None = None
        self._load_personas(personas_dir)

    def _load_personas(self, personas_dir: Path):
        """Auto-discover personas from directory."""
        for path in personas_dir.glob("*.md"):
            persona = self._parse_persona(path)
            self._personas[persona.name.lower()] = persona

    def _parse_persona(self, path: Path) -> Persona:
        """Parse persona from markdown with YAML frontmatter."""
        content = path.read_text()

        # Extract YAML frontmatter
        if content.startswith("---"):
            _, frontmatter, body = content.split("---", 2)
            meta = yaml.safe_load(frontmatter)
        else:
            raise ValueError(f"Persona {path} missing YAML frontmatter")

        return Persona(
            name=meta["name"],
            wake_word=meta["wake_word"],
            voice=meta.get("voice", "alloy"),
            system_prompt=body.strip()
        )

    def get_wake_models(self) -> list[str]:
        """Get list of wake word models for all personas."""
        return [p.wake_word for p in self._personas.values()]

    def persona_for_model(self, model: str) -> str:
        """Get persona name for wake word model."""
        for name, persona in self._personas.items():
            if persona.wake_word == model:
                return name
        return "default"

    def get_voice(self, persona: str) -> str:
        """Get TTS voice for persona."""
        return self._personas.get(persona, self._personas["jarvis"]).voice

    async def switch_to(self, persona: str):
        """Switch active persona."""
        self._current = persona
        # Agent will use get_system_prompt() on next query
```

## OpenAI Realtime Client

WebSocket client for OpenAI Realtime API (STT + TTS).

```python
class OpenAIRealtimeClient:
    """WebSocket client for OpenAI Realtime API."""

    def __init__(self):
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._session_id: str | None = None

    async def connect(self):
        """Connect to OpenAI Realtime API."""
        self._ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime",
            extra_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        )

    async def start_session(self, voice: str = "alloy"):
        """Start a new realtime session."""
        await self._ws.send(json.dumps({
            "type": "session.create",
            "session": {
                "voice": voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16"
            }
        }))

        response = await self._ws.recv()
        data = json.loads(response)
        self._session_id = data["session"]["id"]

    async def transcribe(self, audio: bytes) -> str:
        """Send audio and get transcription."""
        # Send audio
        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode()
        }))

        # Commit and request transcription
        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.commit"
        }))

        # Wait for transcription
        async for msg in self._ws:
            data = json.loads(msg)
            if data["type"] == "conversation.item.input_audio_transcription.completed":
                return data["transcript"]

    async def speak(self, text: str, voice: str, on_complete: Callable):
        """Stream TTS audio."""
        await self._ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "instructions": text
            }
        }))

        # Stream audio chunks
        async for msg in self._ws:
            data = json.loads(msg)
            if data["type"] == "response.audio.delta":
                audio = base64.b64decode(data["delta"])
                self._audio_manager.play_chunk(audio)
            elif data["type"] == "response.audio.done":
                on_complete()
                break

    async def stop_speaking(self):
        """Cancel current TTS playback."""
        await self._ws.send(json.dumps({
            "type": "response.cancel"
        }))
```

## Audio Manager

Handles audio capture and playback.

```python
class AudioManager:
    """Audio capture and playback using PyAudio."""

    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16

    def __init__(self):
        self._pa: pyaudio.PyAudio | None = None
        self._input_stream: pyaudio.Stream | None = None
        self._output_stream: pyaudio.Stream | None = None
        self._capture_buffer: list[bytes] = []

    async def initialize(self):
        """Initialize PyAudio."""
        self._pa = pyaudio.PyAudio()

    def start_capture(self):
        """Start audio capture."""
        self._capture_buffer = []
        self._input_stream = self._pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE,
            stream_callback=self._capture_callback
        )
        self._input_stream.start_stream()

    def _capture_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for capture."""
        self._capture_buffer.append(in_data)
        return (None, pyaudio.paContinue)

    def stop_capture(self) -> bytes:
        """Stop capture and return audio data."""
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
            self._input_stream = None
        return b"".join(self._capture_buffer)

    def play_chunk(self, audio: bytes):
        """Play audio chunk."""
        if not self._output_stream:
            self._output_stream = self._pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=24000,  # OpenAI outputs 24kHz
                output=True
            )
        self._output_stream.write(audio)

    def cleanup(self):
        """Clean up audio resources."""
        if self._input_stream:
            self._input_stream.close()
        if self._output_stream:
            self._output_stream.close()
        if self._pa:
            self._pa.terminate()
```

## Debug Mode

```bash
# Run with voice debug logging
python -m reachy_agent run --voice --debug-voice
```

Output:
```
[12:34:56.789] voice_event event=wake_detected persona=motoko confidence=0.87
[12:34:56.792] voice_event event=listening_start persona=motoko
[12:34:59.123] voice_event event=listening_end audio_duration=2.331
[12:34:59.456] voice_event event=transcribed text="What time is it?" confidence=0.94
[12:34:59.458] voice_event event=processing text="What time is it?"
[12:35:00.234] voice_event event=response text="I don't have access to the current time."
[12:35:00.236] voice_event event=speaking_start text="I don't have..." voice=nova
[12:35:02.567] voice_event event=speaking_end
```

## Configuration

```yaml
# config/default.yaml
voice:
  personas: [motoko, batou, jarvis]  # Auto-discovered from prompts/personas/
  wake_sensitivity: 0.5
  silence_threshold: 0.3  # seconds of silence to end listening
  max_listen_time: 30.0   # max seconds to listen

openai:
  realtime_model: gpt-4o-realtime-preview
```

## What Changed from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Architecture | 7-state machine | Event-driven |
| Barge-in | Not supported | Full support |
| State transitions | Explicit code | Implicit via events |
| Debug logging | Limited | Full event trace |
| Persona config | Deeply nested YAML | Auto-discovery from files |
| Wake word during TTS | Disabled | Enabled (for interrupt) |

## Related Specs

- [01-overview.md](./01-overview.md) - System architecture
- [02-robot-control.md](./02-robot-control.md) - Audio tools (speak, listen)
