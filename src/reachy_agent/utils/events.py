"""Event emitter base class for event-driven components."""

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Union

import structlog

# Type alias for event handlers - can be sync or async
EventHandler = Union[
    "Callable[[Event], None]",
    "Callable[[Event], Awaitable[None]]",
]

# Forward reference for Callable to avoid circular import issues
from collections.abc import Callable


@dataclass
class Event:
    """An event with name, data, and timestamp."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())


class EventEmitter:
    """
    Base class for event-driven components.

    Provides event registration and emission with async support.

    Examples:
        >>> emitter = EventEmitter(debug=True)
        >>>
        >>> @emitter.on("test_event")
        >>> async def handle(event):
        >>>     print(f"Got: {event.data}")
        >>>
        >>> await emitter.emit("test_event", value=42)
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize event emitter.

        Args:
            debug: Enable debug logging for all events
        """
        self._handlers: dict[str, list[Callable[[Event], Any]]] = defaultdict(list)
        self._debug = debug
        self._log = structlog.get_logger()

    def on(
        self, event_name: str, handler: Callable[[Event], Any] | None = None
    ) -> Callable[[Event], Any]:
        """
        Register an event handler.

        Can be used as a decorator or called directly with a handler function.

        Args:
            event_name: Name of event to listen for
            handler: Optional handler function (if not using as decorator)

        Returns:
            Decorator function or the handler if provided directly

        Examples:
            >>> # As decorator
            >>> @emitter.on("data_received")
            >>> async def handle_data(event):
            >>>     process(event.data)
            >>>
            >>> # Direct call
            >>> emitter.on("data_received", my_handler)
        """
        if handler is not None:
            self._handlers[event_name].append(handler)
            return handler

        def decorator(fn: Callable[[Event], Any]) -> Callable[[Event], Any]:
            self._handlers[event_name].append(fn)
            return fn

        return decorator  # type: ignore[return-value]

    def off(
        self, event_name: str, handler: Callable[[Event], Any] | None = None
    ) -> bool:
        """
        Unregister an event handler.

        If no handler is specified, removes all handlers for the event.

        Args:
            event_name: Name of event to stop listening to
            handler: Optional specific handler to remove

        Returns:
            True if handler was removed, False if not found

        Examples:
            >>> # Remove specific handler
            >>> emitter.off("data_received", my_handler)
            >>>
            >>> # Remove all handlers for event
            >>> emitter.off("data_received")
        """
        if event_name not in self._handlers:
            return False

        if handler is None:
            # Remove all handlers for this event
            del self._handlers[event_name]
            return True

        try:
            self._handlers[event_name].remove(handler)
            return True
        except ValueError:
            return False

    async def emit(self, event_name: str, **data: Any) -> None:
        """
        Emit an event to all registered handlers.

        Args:
            event_name: Name of event to emit
            **data: Event data as keyword arguments

        Raises:
            Exception: If handler raises (logged but not propagated)

        Examples:
            >>> await emitter.emit("user_connected", user_id=123)
        """
        event = Event(name=event_name, data=data)

        if self._debug:
            self._log.debug("event_emitted", event_name=event_name, **data)

        for handler in self._handlers[event_name]:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._log.error(
                    "event_handler_error",
                    event_name=event_name,
                    handler=handler.__name__,
                    error=str(e),
                )
                # Re-emit as error event
                await self.emit("error", error_type=type(e).__name__, message=str(e))
