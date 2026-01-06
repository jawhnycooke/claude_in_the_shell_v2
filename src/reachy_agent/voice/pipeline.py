"""Event-driven voice interaction pipeline.

This module provides the main voice pipeline for the Reachy agent,
orchestrating wake word detection, STT, agent processing, and TTS.

Features:
    - Event-driven architecture with EventEmitter
    - Wake word detection for persona activation
    - Speech-to-text with OpenAI Realtime API
    - Text-to-speech with streaming playback
    - Barge-in support for interruption
    - Mock mode for testing

Events:
    - wake_detected: Wake word triggered
    - listening_start: Started capturing audio
    - listening_end: Stopped capturing audio
    - transcribed: Speech-to-text complete
    - processing: Sending to agent
    - response: Agent response received
    - speaking_start: TTS playback started
    - speaking_end: TTS playback complete
    - interrupted: Playback interrupted (barge-in)
    - error: Error occurred
"""

from typing import Any

import structlog

from reachy_agent.utils.events import Event, EventEmitter
from reachy_agent.voice.audio import AudioManager
from reachy_agent.voice.persona import PersonaManager
from reachy_agent.voice.realtime import OpenAIRealtimeClient
from reachy_agent.voice.wake_word import WakeWordDetector


class VoicePipeline(EventEmitter):
    """
    Event-driven voice interaction pipeline with barge-in support.

    Orchestrates the flow: wake_detected → listening → transcribed →
    processing → response → speaking → [interrupt]

    Attributes:
        is_running: Whether pipeline is active
        is_speaking: Whether TTS is playing
        current_persona: Currently active persona name

    Examples:
        >>> pipeline = VoicePipeline(agent, persona_manager)
        >>> await pipeline.start()
        >>> # Pipeline runs autonomously
        >>> await pipeline.stop()
    """

    def __init__(
        self,
        agent: Any | None = None,  # ReachyAgentLoop
        persona_manager: PersonaManager | None = None,
        debug: bool = False,
        mock_mode: bool = False,
    ):
        """
        Initialize voice pipeline.

        Args:
            agent: Agent loop instance for processing
            persona_manager: Persona manager for wake words/voices
            debug: Enable debug event logging
            mock_mode: If True, use mock components (no hardware/API)
        """
        super().__init__(debug=debug)
        self._agent = agent
        self._personas = persona_manager or PersonaManager(auto_discover=False)
        self._mock_mode = mock_mode

        self._wake_detector: WakeWordDetector | None = None
        self._realtime: OpenAIRealtimeClient | None = None
        self._audio: AudioManager | None = None

        self._current_persona: str | None = None
        self._is_speaking = False
        self._is_listening = False
        self._is_running = False

        self._log = structlog.get_logger("voice.pipeline")

        self._register_handlers()

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is playing."""
        return self._is_speaking

    @property
    def is_listening(self) -> bool:
        """Check if capturing audio."""
        return self._is_listening

    @property
    def current_persona(self) -> str | None:
        """Get currently active persona."""
        return self._current_persona

    def _register_handlers(self) -> None:
        """Register all event handlers."""

        @self.on("wake_detected")
        async def handle_wake(event: Event) -> None:
            """Handle wake word detection."""
            persona = event.data.get("persona", "default")
            confidence = event.data.get("confidence", 0.0)

            self._current_persona = persona
            self._log.info(
                "wake_detected",
                persona=persona,
                confidence=confidence,
            )

            # Start listening
            await self.emit("listening_start", persona=persona)

        @self.on("listening_start")
        async def handle_listening_start(event: Event) -> None:
            """Start audio capture."""
            self._is_listening = True
            persona = event.data.get("persona")

            if self._audio:
                self._audio.start_capture()

            self._log.info("listening_start", persona=persona)

            # In real implementation, we'd start STT session here
            if self._realtime:
                voice = self._personas.get_voice(persona or "default")
                await self._realtime.start_session(voice=voice)

        @self.on("listening_end")
        async def handle_listening_end(event: Event) -> None:
            """Stop audio capture and get transcript."""
            self._is_listening = False
            audio_duration = event.data.get("audio_duration", 0.0)

            audio_data = b""
            if self._audio:
                audio_data = self._audio.stop_capture()

            self._log.info("listening_end", audio_duration=audio_duration)

            # Send to STT
            if self._realtime and audio_data:
                text = await self._realtime.transcribe(audio_data)
                if text.strip():
                    await self.emit(
                        "transcribed",
                        text=text,
                        confidence=0.9,
                    )

        @self.on("transcribed")
        async def handle_transcribed(event: Event) -> None:
            """Send transcription to agent."""
            text = event.data.get("text", "")
            confidence = event.data.get("confidence", 0.0)

            self._log.info("transcribed", text=text, confidence=confidence)

            # Process with agent
            await self.emit("processing", text=text)

            if self._agent:
                # Agent would process and return response
                # This would be: response = await self._agent.process(text)
                pass

        @self.on("processing")
        async def handle_processing(event: Event) -> None:
            """Agent is processing."""
            text = event.data.get("text", "")
            self._log.debug("processing", text=text[:50])

        @self.on("response")
        async def handle_response(event: Event) -> None:
            """Start TTS playback."""
            text = event.data.get("text", "")
            voice = event.data.get("voice")

            if not voice and self._current_persona:
                voice = self._personas.get_voice(self._current_persona)

            self._log.info("response", text=text[:50], voice=voice)

            # Start speaking
            await self.emit("speaking_start", text=text, voice=voice)

        @self.on("speaking_start")
        async def handle_speaking_start(event: Event) -> None:
            """TTS playback started."""
            self._is_speaking = True
            text = event.data.get("text", "")
            voice = event.data.get("voice", "alloy")

            self._log.info("speaking_start", voice=voice)

            # Enable wake detector for barge-in
            if self._wake_detector:
                self._wake_detector.enable()

            # Start TTS
            if self._realtime:

                async def on_complete() -> None:
                    await self.emit("speaking_end")

                def on_audio(data: bytes) -> None:
                    if self._audio:
                        self._audio.play_chunk(data)

                await self._realtime.speak(
                    text,
                    voice=voice,
                    on_complete=on_complete,  # type: ignore[arg-type]
                    on_audio=on_audio,
                )

        @self.on("speaking_end")
        async def handle_speaking_end(event: Event) -> None:
            """TTS playback complete."""
            self._is_speaking = False
            self._log.info("speaking_end")

        @self.on("interrupted")
        async def handle_interrupted(event: Event) -> None:
            """Handle barge-in interrupt."""
            by = event.data.get("by", "unknown")
            self._is_speaking = False

            self._log.info("interrupted", by=by)

            # Stop TTS
            if self._realtime:
                await self._realtime.stop_speaking()

            # If interrupted by wake word, start listening
            if by == "wake_word":
                await self.emit("listening_start", persona=self._current_persona)

        @self.on("error")
        async def handle_error(event: Event) -> None:
            """Handle pipeline error."""
            error = event.data.get("error", "Unknown error")
            self._log.error("pipeline_error", error=error)

    async def start(self) -> None:
        """
        Start the voice pipeline.

        Initializes all components and begins wake word detection.
        """
        self._log.info("voice_pipeline_starting")

        # Initialize audio manager
        self._audio = AudioManager(mock_mode=self._mock_mode)
        await self._audio.initialize()

        # Initialize OpenAI Realtime client
        self._realtime = OpenAIRealtimeClient(mock_mode=self._mock_mode)
        await self._realtime.connect()

        # Initialize wake word detector
        wake_words = self._personas.get_wake_words()
        self._wake_detector = WakeWordDetector(
            models=wake_words,
            on_detected=lambda m, c: self._on_wake_word(m, c),  # type: ignore[misc, return-value]
            sensitivity=0.5,
            mock_mode=self._mock_mode,
        )

        # Enable wake word detection
        self._wake_detector.enable()

        self._is_running = True
        self._log.info("voice_pipeline_started", wake_words=wake_words)

    async def stop(self) -> None:
        """
        Stop the voice pipeline.

        Gracefully shuts down all components.
        """
        self._log.info("voice_pipeline_stopping")

        self._is_running = False

        # Stop wake word detection
        if self._wake_detector:
            self._wake_detector.disable()
            self._wake_detector.cleanup()
            self._wake_detector = None

        # Disconnect from Realtime API
        if self._realtime:
            await self._realtime.disconnect()
            self._realtime.cleanup()
            self._realtime = None

        # Clean up audio
        if self._audio:
            self._audio.cleanup()
            self._audio = None

        self._log.info("voice_pipeline_stopped")

    async def _on_wake_word(self, model: str, confidence: float) -> None:
        """
        Called when wake word detected.

        Args:
            model: Wake word model name
            confidence: Detection confidence
        """
        # Find persona for wake word
        persona = self._personas.get_persona_by_wake_word(model)
        persona_name = persona.name if persona else "default"

        if self._is_speaking:
            # Barge-in: interrupt current speech
            await self.emit("interrupted", by="wake_word")

        await self.emit(
            "wake_detected",
            persona=persona_name,
            confidence=confidence,
        )

    async def process_text(self, text: str) -> None:
        """
        Process text input (alternative to voice).

        Args:
            text: Text to process
        """
        await self.emit("transcribed", text=text, confidence=1.0)

    async def respond(self, text: str, voice: str | None = None) -> None:
        """
        Speak a response.

        Args:
            text: Text to speak
            voice: Optional voice override
        """
        await self.emit("response", text=text, voice=voice)

    def simulate_wake_word(self, wake_word: str, confidence: float = 0.9) -> None:
        """
        Simulate wake word detection (for testing).

        Args:
            wake_word: Wake word phrase
            confidence: Detection confidence
        """
        if self._wake_detector:
            self._wake_detector.simulate_detection(wake_word, confidence)
