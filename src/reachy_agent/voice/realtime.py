"""OpenAI Realtime API client for STT and TTS.

This module provides a WebSocket client for OpenAI's Realtime API,
enabling speech-to-text and text-to-speech capabilities.

Features:
    - WebSocket connection with retry logic
    - Audio transcription (STT)
    - Text-to-speech streaming (TTS)
    - Session management
    - Mock mode for testing

Note:
    Requires OPENAI_API_KEY environment variable.
    Install websockets with: uv pip install reachy-agent[voice]
"""

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import structlog

# websockets is optional
try:
    import websockets
    from websockets.asyncio.client import ClientConnection

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore[assignment]
    ClientConnection = Any  # type: ignore[assignment, misc]
    WEBSOCKETS_AVAILABLE = False


@dataclass
class TranscriptResult:
    """Result from speech-to-text."""

    text: str
    confidence: float
    is_final: bool = True


@dataclass
class AudioChunk:
    """Audio chunk from TTS."""

    data: bytes
    is_final: bool = False


class OpenAIRealtimeClient:
    """
    WebSocket client for OpenAI Realtime API (STT + TTS).

    Provides real-time speech recognition and text-to-speech
    capabilities via WebSocket connection.

    Attributes:
        is_connected: Whether currently connected
        session_id: Current session ID

    Examples:
        >>> client = OpenAIRealtimeClient()
        >>> await client.connect()
        >>> await client.start_session(voice="nova")
        >>> text = await client.transcribe(audio_data)
        >>> await client.speak("Hello!", voice="nova", on_complete=done_cb)
        >>> await client.disconnect()
    """

    REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

    def __init__(
        self,
        api_key: str | None = None,
        mock_mode: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize Realtime API client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env)
            mock_mode: If True, use mock connection (no API calls)
            max_retries: Maximum connection retry attempts
            retry_delay: Delay between retries in seconds
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._mock_mode = mock_mode
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._ws: Any = None
        self._session_id: str | None = None
        self._is_connected = False
        self._current_voice: str = "alloy"
        self._log = structlog.get_logger("voice.realtime")

        # Mock mode data
        self._mock_transcript: TranscriptResult | None = None
        self._mock_audio_chunks: list[AudioChunk] = []

    @property
    def is_connected(self) -> bool:
        """Check if connected to API."""
        return self._is_connected

    @property
    def session_id(self) -> str | None:
        """Get current session ID."""
        return self._session_id

    async def connect(self) -> None:
        """
        Connect to OpenAI Realtime API with retry logic.

        Handles connection errors with exponential backoff.

        Raises:
            ConnectionError: If all retry attempts fail
        """
        if self._mock_mode:
            self._is_connected = True
            self._session_id = "mock-session-id"
            self._log.info("realtime_connected_mock")
            return

        if not WEBSOCKETS_AVAILABLE:
            self._log.warning("websockets_not_available", falling_back="mock_mode")
            self._mock_mode = True
            self._is_connected = True
            self._session_id = "mock-session-id"
            return

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                self._ws = await websockets.connect(  # type: ignore[union-attr]
                    self.REALTIME_URL,
                    additional_headers=headers,
                )
                self._is_connected = True
                self._log.info("realtime_connected", attempt=attempt + 1)
                return

            except Exception as e:
                last_error = e
                self._log.warning(
                    "realtime_connect_failed",
                    attempt=attempt + 1,
                    max_retries=self._max_retries,
                    error=str(e),
                )

                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))

        raise ConnectionError(
            f"Failed to connect after {self._max_retries} attempts: {last_error}"
        )

    async def disconnect(self) -> None:
        """Disconnect from API gracefully."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:
                self._log.warning("realtime_disconnect_error", error=str(e))
            finally:
                self._ws = None

        self._is_connected = False
        self._session_id = None
        self._log.info("realtime_disconnected")

    async def start_session(self, voice: str = "alloy") -> None:
        """
        Start a new realtime session.

        Args:
            voice: Voice to use for TTS (alloy, echo, fable, onyx, nova, shimmer)
        """
        self._current_voice = voice

        if self._mock_mode:
            self._log.info("realtime_session_started_mock", voice=voice)
            return

        if not self._is_connected or self._ws is None:
            raise ConnectionError("Not connected to API")

        # Send session.create message
        session_msg = {
            "type": "session.create",
            "session": {
                "voice": voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
            },
        }

        await self._ws.send(json.dumps(session_msg))

        # Wait for session.created response
        response = await self._ws.recv()
        data = json.loads(response)

        if data.get("type") == "session.created":
            self._session_id = data.get("session", {}).get("id")
            self._log.info(
                "realtime_session_started",
                session_id=self._session_id,
                voice=voice,
            )
        else:
            self._log.warning("unexpected_session_response", data=data)

    async def send_audio(self, audio: bytes) -> None:
        """
        Send audio chunk for transcription.

        Args:
            audio: PCM16 audio data to send
        """
        if self._mock_mode:
            self._log.debug("realtime_send_audio_mock", bytes=len(audio))
            return

        if not self._is_connected or self._ws is None:
            raise ConnectionError("Not connected to API")

        msg = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode(),
        }

        await self._ws.send(json.dumps(msg))

    async def commit_audio(self) -> None:
        """Commit audio buffer and request transcription."""
        if self._mock_mode:
            return

        if not self._is_connected or self._ws is None:
            raise ConnectionError("Not connected to API")

        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def receive_transcript(self) -> TranscriptResult | None:
        """
        Receive transcription result.

        Returns:
            TranscriptResult with text and confidence, or None
        """
        if self._mock_mode:
            if self._mock_transcript:
                result = self._mock_transcript
                self._mock_transcript = None
                return result
            return None

        if not self._is_connected or self._ws is None:
            return None

        try:
            async for msg in self._ws:
                data = json.loads(msg)
                msg_type = data.get("type", "")

                # Handle transcription complete
                if msg_type == "conversation.item.input_audio_transcription.completed":
                    text = data.get("transcript", "")
                    return TranscriptResult(
                        text=text,
                        confidence=0.9,  # API doesn't return confidence
                        is_final=True,
                    )

                # Handle partial transcript
                elif msg_type == "conversation.item.input_audio_transcription.delta":
                    text = data.get("delta", "")
                    return TranscriptResult(
                        text=text,
                        confidence=0.5,
                        is_final=False,
                    )

        except Exception as e:
            self._log.warning("receive_transcript_error", error=str(e))

        return None

    async def transcribe(self, audio: bytes) -> str:
        """
        Send audio and get transcription (convenience method).

        Args:
            audio: PCM16 audio data

        Returns:
            Transcribed text
        """
        if self._mock_mode:
            if self._mock_transcript:
                result = self._mock_transcript
                self._mock_transcript = None
                return result.text
            return ""

        await self.send_audio(audio)
        await self.commit_audio()

        result = await self.receive_transcript()
        return result.text if result else ""

    async def send_tts_request(self, text: str, voice: str | None = None) -> None:
        """
        Request text-to-speech synthesis.

        Args:
            text: Text to speak
            voice: Voice to use (defaults to session voice)
        """
        voice = voice or self._current_voice

        if self._mock_mode:
            self._log.debug("realtime_tts_request_mock", text=text[:50], voice=voice)
            return

        if not self._is_connected or self._ws is None:
            raise ConnectionError("Not connected to API")

        msg = {
            "type": "response.create",
            "response": {
                "modalities": ["audio"],
                "instructions": text,
                "voice": voice,
            },
        }

        await self._ws.send(json.dumps(msg))
        self._log.debug("realtime_tts_requested", voice=voice)

    async def receive_audio(self) -> AsyncIterator[AudioChunk]:
        """
        Receive TTS audio chunks.

        Yields:
            AudioChunk objects with audio data
        """
        if self._mock_mode:
            for chunk in self._mock_audio_chunks:
                yield chunk
            self._mock_audio_chunks = []
            return

        if not self._is_connected or self._ws is None:
            return

        try:
            async for msg in self._ws:
                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "response.audio.delta":
                    audio_b64 = data.get("delta", "")
                    audio_bytes = base64.b64decode(audio_b64)
                    yield AudioChunk(data=audio_bytes, is_final=False)

                elif msg_type == "response.audio.done":
                    yield AudioChunk(data=b"", is_final=True)
                    break

        except Exception as e:
            self._log.warning("receive_audio_error", error=str(e))

    async def speak(
        self,
        text: str,
        voice: str,
        on_complete: Callable[[], None] | None = None,
        on_audio: Callable[[bytes], None] | None = None,
    ) -> None:
        """
        Stream TTS audio with callbacks.

        Args:
            text: Text to speak
            voice: Voice to use
            on_complete: Callback when complete
            on_audio: Callback for each audio chunk
        """
        if self._mock_mode:
            self._log.info("realtime_speak_mock", text=text[:50], voice=voice)
            # Simulate audio chunks
            for chunk in self._mock_audio_chunks:
                if on_audio:
                    on_audio(chunk.data)
            self._mock_audio_chunks = []
            if on_complete:
                on_complete()
            return

        await self.send_tts_request(text, voice)

        async for chunk in self.receive_audio():
            if chunk.data and on_audio:
                on_audio(chunk.data)
            if chunk.is_final:
                break

        if on_complete:
            on_complete()

    async def stop_speaking(self) -> None:
        """Cancel current TTS playback."""
        if self._mock_mode:
            self._mock_audio_chunks = []
            self._log.debug("realtime_stop_speaking_mock")
            return

        if not self._is_connected or self._ws is None:
            return

        try:
            await self._ws.send(json.dumps({"type": "response.cancel"}))
            self._log.debug("realtime_stop_speaking")
        except Exception as e:
            self._log.warning("stop_speaking_error", error=str(e))

    # Mock helper methods for testing

    def simulate_transcript(self, text: str, confidence: float = 0.9) -> None:
        """Simulate a transcript result (for testing)."""
        self._mock_transcript = TranscriptResult(
            text=text,
            confidence=confidence,
            is_final=True,
        )

    def simulate_audio_chunks(self, chunks: list[bytes]) -> None:
        """Simulate audio chunks (for testing)."""
        self._mock_audio_chunks = [
            AudioChunk(data=chunk, is_final=i == len(chunks) - 1)
            for i, chunk in enumerate(chunks)
        ]

    def cleanup(self) -> None:
        """Clean up resources."""
        self._mock_transcript = None
        self._mock_audio_chunks = []
        self._log.debug("realtime_cleanup")
