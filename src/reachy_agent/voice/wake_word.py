"""Wake word detection using OpenWakeWord.

This module provides wake word detection for the voice pipeline.
It detects when users say specific wake words like "hey jarvis" or "hey motoko".

Features:
    - Multi-model detection (multiple wake words)
    - Sensitivity threshold configuration
    - Enable/disable control for barge-in support
    - Mock mode for testing without hardware/models

Note:
    OpenWakeWord is an optional dependency. Install with:
    uv pip install reachy-agent[voice]
"""

from dataclasses import dataclass
from typing import Any, Callable

import structlog

# OpenWakeWord is optional - graceful import
try:
    import openwakeword
    from openwakeword.model import Model as OWWModel

    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    openwakeword = None  # type: ignore[assignment]
    OWWModel = None  # type: ignore[assignment, misc]
    OPENWAKEWORD_AVAILABLE = False


@dataclass
class WakeWordDetection:
    """Result of wake word detection."""

    wake_word: str
    confidence: float
    persona: str | None = None


class WakeWordDetector:
    """
    Continuous wake word detection with barge-in support.

    Uses OpenWakeWord for multi-model detection that can run during TTS.
    Falls back to mock mode when OpenWakeWord is unavailable.

    Attributes:
        models: List of wake word model names
        sensitivity: Detection threshold (0.0 to 1.0)

    Examples:
        >>> def on_detected(model: str, confidence: float):
        ...     print(f"Detected {model} with {confidence:.2f}")
        >>> detector = WakeWordDetector(
        ...     models=["hey_jarvis", "hey_motoko"],
        ...     on_detected=on_detected,
        ...     sensitivity=0.5,
        ... )
        >>> detector.enable()
        >>> await detector.process_audio(audio_chunk)
    """

    def __init__(
        self,
        models: list[str],
        on_detected: Callable[[str, float], Any] | None = None,
        sensitivity: float = 0.5,
        mock_mode: bool = False,
    ):
        """
        Initialize wake word detector.

        Args:
            models: List of wake word model names
            on_detected: Callback(model, confidence) when detected
            sensitivity: Detection sensitivity threshold (0.0-1.0)
            mock_mode: If True, use mock detection (no hardware)
        """
        self._models = models
        self._on_detected = on_detected
        self._sensitivity = sensitivity
        self._enabled = False
        self._mock_mode = mock_mode
        self._oww: Any = None
        self._log = structlog.get_logger("voice.wake_word")

        # For mock mode, we can simulate detections
        self._mock_detection: WakeWordDetection | None = None

    @property
    def is_enabled(self) -> bool:
        """Check if detection is enabled."""
        return self._enabled

    @property
    def is_available(self) -> bool:
        """Check if OpenWakeWord is available."""
        return OPENWAKEWORD_AVAILABLE

    @property
    def sensitivity(self) -> float:
        """Get current sensitivity threshold."""
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        """Set sensitivity threshold."""
        self._sensitivity = max(0.0, min(1.0, value))

    @property
    def models(self) -> list[str]:
        """Get registered wake word models."""
        return self._models.copy()

    def enable(self) -> None:
        """
        Enable wake word detection.

        Initializes OpenWakeWord if not already loaded.
        """
        if self._mock_mode:
            self._enabled = True
            self._log.info("wake_detector_enabled_mock")
            return

        if not OPENWAKEWORD_AVAILABLE:
            self._log.warning("openwakeword_not_available", falling_back="mock_mode")
            self._mock_mode = True
            self._enabled = True
            return

        try:
            if self._oww is None:
                # Initialize OpenWakeWord with requested models
                self._oww = OWWModel(
                    wakeword_models=self._models,
                    inference_framework="onnx",
                )
                self._log.info(
                    "wake_detector_models_loaded",
                    models=self._models,
                )

            self._enabled = True
            self._log.info("wake_detector_enabled")

        except Exception as e:
            self._log.error("wake_detector_enable_failed", error=str(e))
            self._mock_mode = True
            self._enabled = True

    def disable(self) -> None:
        """Disable wake word detection."""
        self._enabled = False
        self._log.info("wake_detector_disabled")

    async def process_audio(
        self, audio_chunk: bytes
    ) -> WakeWordDetection | None:
        """
        Process audio chunk for wake words.

        Args:
            audio_chunk: Audio data to process (PCM 16-bit, 16kHz)

        Returns:
            WakeWordDetection if detected, None otherwise
        """
        if not self._enabled:
            return None

        # Mock mode - check for simulated detection
        if self._mock_mode:
            return self._process_mock()

        # Real OpenWakeWord detection
        return await self._process_real(audio_chunk)

    def _process_mock(self) -> WakeWordDetection | None:
        """Process in mock mode - return simulated detection."""
        if self._mock_detection is not None:
            detection = self._mock_detection
            self._mock_detection = None  # Clear after returning

            # Call callback if provided
            if self._on_detected and detection.confidence > self._sensitivity:
                self._on_detected(detection.wake_word, detection.confidence)

            self._log.debug(
                "wake_word_detected_mock",
                model=detection.wake_word,
                confidence=detection.confidence,
            )

            return detection

        return None

    async def _process_real(
        self, audio_chunk: bytes
    ) -> WakeWordDetection | None:
        """Process audio with real OpenWakeWord detection."""
        if self._oww is None:
            return None

        try:
            import numpy as np

            # Convert bytes to numpy array (16-bit PCM)
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

            # Run prediction
            predictions = self._oww.predict(audio_array)

            # Check each model for detection
            for model_name, confidence in predictions.items():
                if confidence > self._sensitivity:
                    self._log.debug(
                        "wake_word_detected",
                        model=model_name,
                        confidence=confidence,
                    )

                    # Call callback
                    if self._on_detected:
                        self._on_detected(model_name, confidence)

                    return WakeWordDetection(
                        wake_word=model_name,
                        confidence=confidence,
                    )

            return None

        except Exception as e:
            self._log.warning("wake_word_process_error", error=str(e))
            return None

    def simulate_detection(
        self, wake_word: str, confidence: float = 0.9
    ) -> None:
        """
        Simulate a wake word detection (for testing).

        Args:
            wake_word: Wake word to simulate
            confidence: Detection confidence
        """
        self._mock_detection = WakeWordDetection(
            wake_word=wake_word,
            confidence=confidence,
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._mock_detection = None
        if self._oww is not None:
            try:
                self._oww.reset()
            except AttributeError:
                pass  # Not all versions support reset

    def cleanup(self) -> None:
        """Clean up resources."""
        self.disable()
        self._oww = None
        self._mock_detection = None
        self._log.debug("wake_detector_cleanup")


def get_available_models() -> list[str]:
    """
    Get list of available wake word models.

    Returns:
        List of model names available for use
    """
    if not OPENWAKEWORD_AVAILABLE:
        return []

    try:
        # OpenWakeWord provides pre-trained models
        return [
            "alexa",
            "hey_mycroft",
            "hey_jarvis",
            "hey_mycroft",
            "computer",
        ]
    except Exception:
        return []
