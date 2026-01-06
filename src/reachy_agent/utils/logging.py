"""Structured logging setup using structlog."""

import logging
import os
import sys
from typing import Any

import structlog


def _get_log_level_from_env() -> int:
    """
    Get log level from environment variable.

    Checks REACHY_LOG_LEVEL (DEBUG, INFO, WARNING, ERROR) or REACHY_DEBUG (0/1).

    Returns:
        Logging level constant
    """
    level_str = os.environ.get("REACHY_LOG_LEVEL", "").upper()
    if level_str:
        return getattr(logging, level_str, logging.INFO)

    debug_env = os.environ.get("REACHY_DEBUG", "0")
    if debug_env in ("1", "true", "yes", "on"):
        return logging.DEBUG

    return logging.INFO


def setup_logging(
    debug: bool = False,
    json_output: bool = False,
    level: int | None = None,
) -> structlog.stdlib.BoundLogger:
    """
    Configure structured logging for the application.

    Log level precedence: level param > debug param > environment variables.

    Environment variables:
        REACHY_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR
        REACHY_DEBUG: Set to "1" for debug mode

    Args:
        debug: Enable debug-level logging (overridden by level param)
        json_output: Output logs as JSON instead of colored console output
        level: Explicit log level (highest priority)

    Returns:
        Configured bound logger

    Examples:
        >>> log = setup_logging(debug=True)
        >>> log.info("agent_started", version="0.1.0")

        >>> # JSON output for production
        >>> log = setup_logging(json_output=True)
    """
    # Determine log level: explicit > debug flag > environment
    if level is not None:
        log_level = level
    elif debug:
        log_level = logging.DEBUG
    else:
        log_level = _get_log_level_from_env()

    # Configure Python logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,  # Allow re-configuration
    )

    # Build processor chain
    shared_processors: list[Any] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add final renderer based on output mode
    if json_output:
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    return logger


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a bound logger instance.

    Args:
        name: Optional logger name (defaults to module name)

    Returns:
        Bound logger instance

    Examples:
        >>> log = get_logger(__name__)
        >>> log.info("processing_started", items=100)
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
