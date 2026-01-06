"""Tests for utility modules."""

import asyncio
import json
import logging
import os
from pathlib import Path
from io import StringIO

import pytest
import structlog

from reachy_agent.utils.events import Event, EventEmitter
from reachy_agent.utils.logging import setup_logging, get_logger, _get_log_level_from_env
from reachy_agent.utils.config import (
    load_config,
    load_config_dict,
    load_yaml,
    ReachyConfig,
    AgentConfig,
    _apply_env_overrides,
)


@pytest.mark.asyncio
async def test_event_emitter() -> None:
    """
    Comprehensive test for EventEmitter supporting on(), emit(), and off() methods.

    This test verifies:
    - Event registration via on() as decorator and direct call
    - Event emission via emit() with data passing
    - Event unregistration via off() for specific and all handlers
    """
    emitter = EventEmitter(debug=True)
    received_events: list[Event] = []

    # Test on() as decorator
    @emitter.on("test_event")
    async def handler1(event: Event) -> None:
        received_events.append(event)

    # Test on() as direct call
    async def handler2(event: Event) -> None:
        received_events.append(event)

    emitter.on("test_event", handler2)

    # Test emit()
    await emitter.emit("test_event", value=42, message="hello")

    assert len(received_events) == 2
    assert received_events[0].name == "test_event"
    assert received_events[0].data["value"] == 42
    assert received_events[0].data["message"] == "hello"

    # Test off() with specific handler
    received_events.clear()
    result = emitter.off("test_event", handler1)
    assert result is True

    await emitter.emit("test_event", value=100)
    assert len(received_events) == 1  # Only handler2 received

    # Test off() to remove all handlers
    result = emitter.off("test_event")
    assert result is True

    received_events.clear()
    await emitter.emit("test_event", value=200)
    assert len(received_events) == 0  # No handlers left


def test_logging() -> None:
    """
    Comprehensive test for structured logging setup.

    This test verifies:
    - Logger configuration with JSON output
    - Log level control from environment variables
    - Basic logging functionality
    """
    # Save original environment
    original_debug = os.environ.get("REACHY_DEBUG")
    original_level = os.environ.get("REACHY_LOG_LEVEL")

    try:
        # Clean environment for test
        if "REACHY_DEBUG" in os.environ:
            del os.environ["REACHY_DEBUG"]
        if "REACHY_LOG_LEVEL" in os.environ:
            del os.environ["REACHY_LOG_LEVEL"]

        # Test 1: Basic logger setup
        log = setup_logging()
        assert log is not None
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "error")

        # Test 2: JSON output configuration
        json_log = setup_logging(json_output=True)
        assert json_log is not None

        # Test 3: Environment variable control - REACHY_LOG_LEVEL
        os.environ["REACHY_LOG_LEVEL"] = "WARNING"
        level = _get_log_level_from_env()
        assert level == logging.WARNING
        del os.environ["REACHY_LOG_LEVEL"]

        # Test 4: Environment variable control - REACHY_DEBUG
        os.environ["REACHY_DEBUG"] = "1"
        level = _get_log_level_from_env()
        assert level == logging.DEBUG
        del os.environ["REACHY_DEBUG"]

        # Test 5: Default log level
        level = _get_log_level_from_env()
        assert level == logging.INFO

    finally:
        # Restore original environment
        if original_debug is not None:
            os.environ["REACHY_DEBUG"] = original_debug
        elif "REACHY_DEBUG" in os.environ:
            del os.environ["REACHY_DEBUG"]

        if original_level is not None:
            os.environ["REACHY_LOG_LEVEL"] = original_level
        elif "REACHY_LOG_LEVEL" in os.environ:
            del os.environ["REACHY_LOG_LEVEL"]


def test_config_loading() -> None:
    """
    Comprehensive test for configuration loading from YAML files.

    This test verifies:
    - Loads config/default.yaml
    - Pydantic models for type safety
    - Environment variable overrides
    """
    # Save original environment variables
    original_model = os.environ.get("REACHY_AGENT_MODEL")
    original_backend = os.environ.get("REACHY_ROBOT_BACKEND")

    try:
        # Clean up environment for testing
        if "REACHY_AGENT_MODEL" in os.environ:
            del os.environ["REACHY_AGENT_MODEL"]
        if "REACHY_ROBOT_BACKEND" in os.environ:
            del os.environ["REACHY_ROBOT_BACKEND"]

        # Test 1: Load config/default.yaml - returns Pydantic model
        config = load_config()
        assert isinstance(config, ReachyConfig)
        assert isinstance(config.agent, AgentConfig)

        # Test 2: Pydantic models provide type safety
        assert config.agent.model == "claude-haiku-4-5-20251001"
        assert config.motion.tick_hz == 30
        assert isinstance(config.motion.tick_hz, int)
        assert isinstance(config.agent.temperature, float)

        # Test 3: Environment variable overrides
        os.environ["REACHY_AGENT_MODEL"] = "claude-sonnet-4-20250514"
        os.environ["REACHY_ROBOT_BACKEND"] = "mock"

        config_dict = {"agent": {"model": "original"}, "robot": {"backend": "sdk"}}
        overridden = _apply_env_overrides(config_dict)
        assert overridden["agent"]["model"] == "claude-sonnet-4-20250514"
        assert overridden["robot"]["backend"] == "mock"

        # Test 4: load_config_dict returns dictionary
        raw_config = load_config_dict()
        assert isinstance(raw_config, dict)
        assert "agent" in raw_config

    finally:
        # Restore original environment
        if original_model is not None:
            os.environ["REACHY_AGENT_MODEL"] = original_model
        elif "REACHY_AGENT_MODEL" in os.environ:
            del os.environ["REACHY_AGENT_MODEL"]

        if original_backend is not None:
            os.environ["REACHY_ROBOT_BACKEND"] = original_backend
        elif "REACHY_ROBOT_BACKEND" in os.environ:
            del os.environ["REACHY_ROBOT_BACKEND"]


class TestEvent:
    """Tests for Event dataclass."""

    def test_event_creation(self) -> None:
        """Test creating an event with defaults."""
        event = Event(name="test")
        assert event.name == "test"
        assert event.data == {}
        assert event.timestamp > 0

    def test_event_with_data(self) -> None:
        """Test creating an event with data."""
        event = Event(name="test", data={"key": "value"})
        assert event.name == "test"
        assert event.data == {"key": "value"}


class TestEventEmitter:
    """Tests for EventEmitter class."""

    @pytest.fixture
    def emitter(self) -> EventEmitter:
        """Create a fresh emitter for each test."""
        return EventEmitter(debug=True)

    @pytest.mark.asyncio
    async def test_event_registration_decorator(self, emitter: EventEmitter) -> None:
        """Test registering handlers with decorator."""
        received: list[Event] = []

        @emitter.on("test_event")
        async def handler(event: Event) -> None:
            received.append(event)

        await emitter.emit("test_event", value=42)

        assert len(received) == 1
        assert received[0].name == "test_event"
        assert received[0].data["value"] == 42

    @pytest.mark.asyncio
    async def test_event_registration_direct(self, emitter: EventEmitter) -> None:
        """Test registering handlers with direct call."""
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        emitter.on("test_event", handler)
        await emitter.emit("test_event", value=42)

        assert len(received) == 1
        assert received[0].data["value"] == 42

    @pytest.mark.asyncio
    async def test_event_emission(self, emitter: EventEmitter) -> None:
        """Test emitting events to multiple handlers."""
        count = 0

        @emitter.on("test_event")
        async def handler1(event: Event) -> None:
            nonlocal count
            count += 1

        @emitter.on("test_event")
        async def handler2(event: Event) -> None:
            nonlocal count
            count += 1

        await emitter.emit("test_event")

        assert count == 2

    @pytest.mark.asyncio
    async def test_event_unregistration_specific(self, emitter: EventEmitter) -> None:
        """Test unregistering a specific handler."""
        received: list[str] = []

        async def handler1(event: Event) -> None:
            received.append("handler1")

        async def handler2(event: Event) -> None:
            received.append("handler2")

        emitter.on("test_event", handler1)
        emitter.on("test_event", handler2)

        # Remove handler1
        result = emitter.off("test_event", handler1)
        assert result is True

        await emitter.emit("test_event")

        assert received == ["handler2"]

    @pytest.mark.asyncio
    async def test_event_unregistration_all(self, emitter: EventEmitter) -> None:
        """Test unregistering all handlers for an event."""
        received: list[str] = []

        async def handler1(event: Event) -> None:
            received.append("handler1")

        async def handler2(event: Event) -> None:
            received.append("handler2")

        emitter.on("test_event", handler1)
        emitter.on("test_event", handler2)

        # Remove all handlers
        result = emitter.off("test_event")
        assert result is True

        await emitter.emit("test_event")

        assert received == []

    def test_off_nonexistent_event(self, emitter: EventEmitter) -> None:
        """Test off() returns False for nonexistent event."""
        result = emitter.off("nonexistent")
        assert result is False

    def test_off_nonexistent_handler(self, emitter: EventEmitter) -> None:
        """Test off() returns False for nonexistent handler."""

        async def handler1(event: Event) -> None:
            pass

        async def handler2(event: Event) -> None:
            pass

        emitter.on("test_event", handler1)
        result = emitter.off("test_event", handler2)
        assert result is False

    @pytest.mark.asyncio
    async def test_sync_handler(self, emitter: EventEmitter) -> None:
        """Test that sync handlers work."""
        received: list[Event] = []

        def sync_handler(event: Event) -> None:
            received.append(event)

        emitter.on("test_event", sync_handler)
        await emitter.emit("test_event", sync=True)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_handler_error_handling(self, emitter: EventEmitter) -> None:
        """Test that handler errors are caught and logged."""
        error_events: list[Event] = []

        @emitter.on("test_event")
        async def failing_handler(event: Event) -> None:
            raise ValueError("Test error")

        @emitter.on("error")
        async def error_handler(event: Event) -> None:
            error_events.append(event)

        # Should not raise
        await emitter.emit("test_event")

        # Should have emitted error event
        assert len(error_events) == 1
        assert error_events[0].data["error_type"] == "ValueError"
        assert "Test error" in error_events[0].data["message"]

    @pytest.mark.asyncio
    async def test_emit_no_handlers(self, emitter: EventEmitter) -> None:
        """Test emitting to event with no handlers."""
        # Should not raise
        await emitter.emit("no_handlers_event", data="test")


class TestLogging:
    """Tests for logging setup."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self) -> None:
        """Clean up environment variables before/after each test."""
        # Save original values
        original_debug = os.environ.get("REACHY_DEBUG")
        original_level = os.environ.get("REACHY_LOG_LEVEL")

        # Clean environment
        if "REACHY_DEBUG" in os.environ:
            del os.environ["REACHY_DEBUG"]
        if "REACHY_LOG_LEVEL" in os.environ:
            del os.environ["REACHY_LOG_LEVEL"]

        yield

        # Restore original values
        if original_debug is not None:
            os.environ["REACHY_DEBUG"] = original_debug
        elif "REACHY_DEBUG" in os.environ:
            del os.environ["REACHY_DEBUG"]

        if original_level is not None:
            os.environ["REACHY_LOG_LEVEL"] = original_level
        elif "REACHY_LOG_LEVEL" in os.environ:
            del os.environ["REACHY_LOG_LEVEL"]

    def test_setup_logging_returns_logger(self) -> None:
        """Test that setup_logging returns a bound logger."""
        log = setup_logging()
        assert log is not None
        # Should have common logging methods
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "error")

    def test_debug_flag_sets_level(self) -> None:
        """Test debug flag sets log level to DEBUG."""
        setup_logging(debug=True)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_level_param_overrides_debug(self) -> None:
        """Test explicit level parameter overrides debug flag."""
        setup_logging(debug=True, level=logging.WARNING)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_env_variable_log_level(self) -> None:
        """Test REACHY_LOG_LEVEL environment variable."""
        os.environ["REACHY_LOG_LEVEL"] = "WARNING"
        level = _get_log_level_from_env()
        assert level == logging.WARNING

    def test_env_variable_debug(self) -> None:
        """Test REACHY_DEBUG environment variable."""
        os.environ["REACHY_DEBUG"] = "1"
        level = _get_log_level_from_env()
        assert level == logging.DEBUG

    def test_env_debug_true_variants(self) -> None:
        """Test various truthy values for REACHY_DEBUG."""
        for value in ("1", "true", "yes", "on"):
            os.environ["REACHY_DEBUG"] = value
            level = _get_log_level_from_env()
            assert level == logging.DEBUG, f"Failed for value: {value}"

    def test_default_log_level(self) -> None:
        """Test default log level is INFO."""
        level = _get_log_level_from_env()
        assert level == logging.INFO

    def test_get_logger(self) -> None:
        """Test get_logger returns a logger."""
        setup_logging()  # Initialize logging first
        log = get_logger("test_module")
        assert log is not None

    def test_json_output_mode(self) -> None:
        """Test JSON output configuration is set up."""
        # Just verify it doesn't raise
        log = setup_logging(json_output=True)
        assert log is not None


class TestConfigLoading:
    """Tests for configuration loading from YAML."""

    def test_load_config_returns_pydantic_model(self) -> None:
        """Test load_config returns validated Pydantic model."""
        config = load_config()
        assert isinstance(config, ReachyConfig)
        assert isinstance(config.agent, AgentConfig)

    def test_config_default_values(self) -> None:
        """Test configuration has expected default values."""
        config = load_config()
        assert config.agent.model == "claude-haiku-4-5-20251001"
        assert config.agent.name == "Jarvis"
        assert config.motion.tick_hz == 30
        assert config.memory.context_window_size == 5

    def test_load_config_dict(self) -> None:
        """Test load_config_dict returns dictionary."""
        config = load_config_dict()
        assert isinstance(config, dict)
        assert "agent" in config
        assert config["agent"]["model"] == "claude-haiku-4-5-20251001"

    def test_load_yaml_file_not_found(self) -> None:
        """Test load_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_yaml("nonexistent/config.yaml")

    def test_env_override_agent_model(self) -> None:
        """Test environment variable override for agent.model."""
        original = os.environ.get("REACHY_AGENT_MODEL")
        try:
            os.environ["REACHY_AGENT_MODEL"] = "claude-sonnet-4-20250514"
            config = {"agent": {"model": "original"}}
            result = _apply_env_overrides(config)
            assert result["agent"]["model"] == "claude-sonnet-4-20250514"
        finally:
            if original is not None:
                os.environ["REACHY_AGENT_MODEL"] = original
            elif "REACHY_AGENT_MODEL" in os.environ:
                del os.environ["REACHY_AGENT_MODEL"]

    def test_env_override_creates_section(self) -> None:
        """Test env override creates section if missing."""
        original = os.environ.get("REACHY_ROBOT_BACKEND")
        try:
            os.environ["REACHY_ROBOT_BACKEND"] = "mock"
            config: dict = {}
            result = _apply_env_overrides(config)
            assert result["robot"]["backend"] == "mock"
        finally:
            if original is not None:
                os.environ["REACHY_ROBOT_BACKEND"] = original
            elif "REACHY_ROBOT_BACKEND" in os.environ:
                del os.environ["REACHY_ROBOT_BACKEND"]

    def test_env_override_tick_hz_converted_to_int(self) -> None:
        """Test tick_hz is converted to integer from env var."""
        original = os.environ.get("REACHY_MOTION_TICK_HZ")
        try:
            os.environ["REACHY_MOTION_TICK_HZ"] = "60"
            config = {"motion": {"tick_hz": 30}}
            result = _apply_env_overrides(config)
            assert result["motion"]["tick_hz"] == 60
            assert isinstance(result["motion"]["tick_hz"], int)
        finally:
            if original is not None:
                os.environ["REACHY_MOTION_TICK_HZ"] = original
            elif "REACHY_MOTION_TICK_HZ" in os.environ:
                del os.environ["REACHY_MOTION_TICK_HZ"]

    def test_pydantic_validation(self) -> None:
        """Test Pydantic validates configuration types."""
        # Default config should be valid
        config = ReachyConfig()
        assert config.motion.tick_hz == 30

        # Can also create with custom values
        custom = ReachyConfig(agent=AgentConfig(name="Custom"))
        assert custom.agent.name == "Custom"
