"""Audio capture and playback using PyAudio.

This module provides audio stream management for:
- Microphone input capture (16kHz mono for STT)
- Speaker output playback (24kHz mono from TTS)
- Buffer management with overflow/underflow handling

The module supports mocking for testing without hardware.
"""

from dataclasses import dataclass
from typing import Any, Generator

import structlog


@dataclass
class AudioConfig:
    """Audio configuration settings."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = 8  # pyaudio.paInt16 = 8


class AudioManager:
    """
    Audio capture and playback using PyAudio.

    Handles mic input and speaker output with proper buffering.
    Supports mock mode for testing without hardware.

    Attributes:
        config: Audio configuration settings
        mock_mode: Whether running in mock mode (no hardware)

    Examples:
        >>> manager = AudioManager()
        >>> await manager.initialize()
        >>> manager.start_capture()
        >>> audio = manager.stop_capture()
        >>> manager.cleanup()
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        mock_mode: bool = False,
    ) -> None:
        """
        Initialize audio manager.

        Args:
            config: Audio configuration settings
            mock_mode: If True, use mock streams (no hardware)
        """
        self.config = config or AudioConfig()
        self.mock_mode = mock_mode

        self._pa: Any | None = None
        self._input_stream: Any | None = None
        self._output_stream: Any | None = None
        self._capture_buffer: list[bytes] = []
        self._is_capturing: bool = False
        self._initialized: bool = False
        self._log = structlog.get_logger("voice.audio")

    async def initialize(self) -> None:
        """
        Initialize PyAudio instance.

        Creates the PyAudio instance and prepares for stream creation.
        In mock mode, just sets initialized flag.

        Raises:
            RuntimeError: If PyAudio fails to initialize
        """
        if self._initialized:
            return

        if self.mock_mode:
            self._log.info("audio_init_mock")
            self._initialized = True
            return

        try:
            import pyaudio  # type: ignore[import-untyped]

            self._pa = pyaudio.PyAudio()
            self._log.info(
                "audio_init",
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                chunk_size=self.config.chunk_size,
            )
            self._initialized = True

        except ImportError:
            self._log.warning("pyaudio_not_installed", falling_back="mock_mode")
            self.mock_mode = True
            self._initialized = True

        except Exception as e:
            self._log.error("audio_init_failed", error=str(e))
            raise RuntimeError(f"Failed to initialize PyAudio: {e}") from e

    def open_input_stream(self) -> None:
        """
        Open input stream for microphone capture.

        Opens a PyAudio input stream configured for speech recognition
        (16kHz mono, 16-bit).

        Raises:
            RuntimeError: If not initialized or stream fails to open
        """
        if not self._initialized:
            raise RuntimeError("AudioManager not initialized")

        if self.mock_mode:
            self._log.debug("input_stream_mock_open")
            return

        try:
            import pyaudio  # type: ignore[import-untyped]

            self._input_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
            )
            self._log.info("input_stream_opened")

        except Exception as e:
            self._log.error("input_stream_open_failed", error=str(e))
            raise RuntimeError(f"Failed to open input stream: {e}") from e

    def open_output_stream(self, sample_rate: int = 24000) -> None:
        """
        Open output stream for speaker playback.

        Opens a PyAudio output stream configured for TTS playback
        (default 24kHz mono, 16-bit).

        Args:
            sample_rate: Output sample rate (default 24kHz for TTS)

        Raises:
            RuntimeError: If not initialized or stream fails to open
        """
        if not self._initialized:
            raise RuntimeError("AudioManager not initialized")

        if self.mock_mode:
            self._log.debug("output_stream_mock_open")
            return

        try:
            import pyaudio  # type: ignore[import-untyped]

            self._output_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=self.config.chunk_size,
            )
            self._log.info("output_stream_opened", sample_rate=sample_rate)

        except Exception as e:
            self._log.error("output_stream_open_failed", error=str(e))
            raise RuntimeError(f"Failed to open output stream: {e}") from e

    def start_capture(self) -> None:
        """
        Start audio capture from microphone.

        Begins capturing audio into internal buffer. Use read_input_stream()
        to get audio chunks, or stop_capture() to get all buffered audio.
        """
        if not self._initialized:
            raise RuntimeError("AudioManager not initialized")

        self._capture_buffer = []
        self._is_capturing = True
        self._log.debug("capture_started")

    def stop_capture(self) -> bytes:
        """
        Stop capture and return all buffered audio data.

        Returns:
            Captured audio as bytes (concatenated chunks)
        """
        self._is_capturing = False
        audio_data = b"".join(self._capture_buffer)
        self._capture_buffer = []
        self._log.debug("capture_stopped", bytes=len(audio_data))
        return audio_data

    def read_input_stream(self) -> bytes:
        """
        Read a single audio chunk from the input stream.

        Returns:
            Audio chunk as bytes

        Raises:
            RuntimeError: If not initialized or stream not open
        """
        if not self._initialized:
            raise RuntimeError("AudioManager not initialized")

        if self.mock_mode:
            # Return silence in mock mode
            data = b"\x00" * self.config.chunk_size * 2
            if self._is_capturing:
                self._capture_buffer.append(data)
            return data

        if self._input_stream is None:
            raise RuntimeError("Input stream not opened")

        try:
            data = self._input_stream.read(
                self.config.chunk_size,
                exception_on_overflow=False,
            )
            if self._is_capturing:
                self._capture_buffer.append(data)
            return data

        except Exception as e:
            self._log.warning("read_overflow", error=str(e))
            # Return silence on overflow
            return b"\x00" * self.config.chunk_size * 2

    def write_output_stream(self, audio: bytes) -> None:
        """
        Write audio chunk to the output stream for playback.

        Args:
            audio: Audio data to play

        Raises:
            RuntimeError: If not initialized or stream not open
        """
        if not self._initialized:
            raise RuntimeError("AudioManager not initialized")

        if self.mock_mode:
            self._log.debug("output_mock_write", bytes=len(audio))
            return

        if self._output_stream is None:
            raise RuntimeError("Output stream not opened")

        try:
            self._output_stream.write(audio)

        except Exception as e:
            self._log.warning("write_underflow", error=str(e))
            # Silently handle underflow

    def play_chunk(self, audio: bytes) -> None:
        """
        Play audio chunk (alias for write_output_stream).

        Args:
            audio: Audio data to play
        """
        self.write_output_stream(audio)

    def read_chunks(
        self,
        duration_seconds: float,
    ) -> Generator[bytes, None, None]:
        """
        Generator that yields audio chunks for specified duration.

        Args:
            duration_seconds: How long to capture audio

        Yields:
            Audio chunks as bytes
        """
        chunks_needed = int(
            (duration_seconds * self.config.sample_rate) / self.config.chunk_size
        )

        for _ in range(chunks_needed):
            yield self.read_input_stream()

    def close_input_stream(self) -> None:
        """Close the input stream."""
        if self._input_stream is not None:
            try:
                self._input_stream.stop_stream()
                self._input_stream.close()
            except Exception as e:
                self._log.warning("input_close_error", error=str(e))
            finally:
                self._input_stream = None
                self._log.debug("input_stream_closed")

    def close_output_stream(self) -> None:
        """Close the output stream."""
        if self._output_stream is not None:
            try:
                self._output_stream.stop_stream()
                self._output_stream.close()
            except Exception as e:
                self._log.warning("output_close_error", error=str(e))
            finally:
                self._output_stream = None
                self._log.debug("output_stream_closed")

    def cleanup(self) -> None:
        """
        Clean up all audio resources.

        Closes all streams and terminates PyAudio instance.
        Safe to call multiple times.
        """
        self.close_input_stream()
        self.close_output_stream()

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as e:
                self._log.warning("terminate_error", error=str(e))
            finally:
                self._pa = None

        self._initialized = False
        self._log.info("audio_cleanup_complete")

    @property
    def is_initialized(self) -> bool:
        """Check if audio manager is initialized."""
        return self._initialized

    @property
    def is_capturing(self) -> bool:
        """Check if currently capturing audio."""
        return self._is_capturing

    def __enter__(self) -> "AudioManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup resources."""
        self.cleanup()
