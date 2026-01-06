"""Agent SDK hooks for permission enforcement.

This module provides pre-tool-use hooks that enforce the permission
system before any tool is executed by the agent.

Features:
    - Block FORBIDDEN tools before execution
    - Prompt for user confirmation on CONFIRM tools
    - Allow AUTONOMOUS tools to proceed
    - Text and voice confirmation modes
"""

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import structlog

from reachy_agent.permissions.evaluator import (
    PermissionDecision,
    PermissionEvaluator,
    PermissionTier,
)


# Type alias for confirmation callback
ConfirmCallback = Callable[[str, dict[str, Any], str | None], Awaitable[bool]]


@dataclass
class PreToolUseResult:
    """
    Result of pre-tool-use permission check.

    Attributes:
        allow: Whether to allow tool execution
        message: Optional message to return to agent if blocked
        decision: The permission decision that was made
    """

    allow: bool
    message: str | None = None
    decision: PermissionDecision | None = None


# =============================================================================
# F069: PermissionHook - blocks FORBIDDEN tools
# =============================================================================


class PermissionHook:
    """
    Hook that enforces permissions before tool execution.

    Integrates with agent framework to intercept tool calls
    and enforce permission rules.

    Features:
        - Blocks FORBIDDEN tools with error message
        - Prompts for CONFIRM tools
        - Allows AUTONOMOUS tools immediately
        - Logs all decisions
    """

    def __init__(
        self,
        evaluator: PermissionEvaluator,
        confirm_callback: ConfirmCallback | None = None,
    ) -> None:
        """
        Initialize permission hook.

        Args:
            evaluator: Permission evaluator instance
            confirm_callback: Optional async function to ask user for confirmation
                              If None, uses default text-based confirmation
        """
        self._evaluator = evaluator
        self._confirm = confirm_callback or confirm_tool_use
        self._log = structlog.get_logger("permissions.hook")

    async def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> PreToolUseResult:
        """
        Check permission before tool execution.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments to the tool

        Returns:
            PreToolUseResult indicating whether to allow or block
        """
        # Evaluate permission
        decision = self._evaluator.evaluate(tool_name, tool_args)

        # Handle based on tier
        if decision.tier == PermissionTier.AUTONOMOUS:
            # Allow immediately
            self._log.debug("tool_allowed_autonomous", tool=tool_name)
            return PreToolUseResult(allow=True, decision=decision)

        elif decision.tier == PermissionTier.FORBIDDEN:
            # Block immediately
            message = f"Tool '{tool_name}' is forbidden: {decision.reason}"
            self._log.warning("tool_blocked_forbidden", tool=tool_name)
            return PreToolUseResult(allow=False, message=message, decision=decision)

        elif decision.tier == PermissionTier.CONFIRM:
            # Ask for confirmation
            self._log.info("tool_requires_confirmation", tool=tool_name)
            approved = await self._confirm(tool_name, tool_args, decision.reason)

            if approved:
                self._log.info("tool_approved_by_user", tool=tool_name)
                return PreToolUseResult(allow=True, decision=decision)
            else:
                message = f"Tool '{tool_name}' was denied by user"
                self._log.info("tool_denied_by_user", tool=tool_name)
                return PreToolUseResult(allow=False, message=message, decision=decision)

        # Fallback (shouldn't happen)
        return PreToolUseResult(
            allow=False, message="Unknown permission tier", decision=decision
        )

    async def check(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
    ) -> PreToolUseResult:
        """
        Alias for __call__ for explicit usage.

        Args:
            tool_name: Name of tool
            tool_args: Tool arguments

        Returns:
            PreToolUseResult
        """
        return await self.__call__(tool_name, tool_args or {})


# =============================================================================
# F070: Confirmation callbacks
# =============================================================================


async def confirm_tool_use(
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str | None,
) -> bool:
    """
    Ask user to approve tool execution (text mode).

    Displays tool info and waits for y/n response.

    Args:
        tool_name: Name of tool
        tool_args: Tool arguments
        reason: Optional reason for confirmation

    Returns:
        True if approved, False otherwise
    """
    print(f"\n🔐 Permission Required: {tool_name}")
    if tool_args:
        args_str = json.dumps(tool_args, indent=2)
        # Truncate if too long
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        print(f"   Args: {args_str}")
    if reason:
        print(f"   Reason: {reason}")

    try:
        response = input("   Allow? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


async def confirm_tool_use_voice(
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str | None,
    robot_speak: Callable[[str], Awaitable[None]] | None = None,
    robot_listen: Callable[[], Awaitable[str]] | None = None,
) -> bool:
    """
    Ask user verbally to approve tool execution (voice mode).

    Uses robot TTS to ask and STT to get response.

    Args:
        tool_name: Name of tool
        tool_args: Tool arguments
        reason: Optional reason for confirmation
        robot_speak: Async function to speak text
        robot_listen: Async function to listen for speech

    Returns:
        True if approved, False otherwise
    """
    if robot_speak is None or robot_listen is None:
        # Fall back to text confirmation
        return await confirm_tool_use(tool_name, tool_args, reason)

    # Build question
    question = f"Should I {tool_name.replace('_', ' ')}?"
    if reason:
        question = f"{reason}. {question}"

    # Ask
    await robot_speak(question)

    # Listen for response
    response = await robot_listen()
    response_lower = response.lower()

    # Parse affirmative/negative
    affirmative = ["yes", "yeah", "yep", "sure", "okay", "ok", "go ahead", "do it"]
    negative = ["no", "nope", "don't", "stop", "cancel"]

    for word in affirmative:
        if word in response_lower:
            return True

    for word in negative:
        if word in response_lower:
            return False

    # Unclear response - default to deny for safety
    await robot_speak("I didn't understand. I'll skip this for now.")
    return False


async def confirm_always_allow(
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str | None,
) -> bool:
    """
    Always allow - for testing or trusted environments.

    Args:
        tool_name: Ignored
        tool_args: Ignored
        reason: Ignored

    Returns:
        Always True
    """
    return True


async def confirm_always_deny(
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str | None,
) -> bool:
    """
    Always deny - for strict lockdown mode.

    Args:
        tool_name: Ignored
        tool_args: Ignored
        reason: Ignored

    Returns:
        Always False
    """
    return False


# =============================================================================
# F071: PostToolAuditHook - logs all tool executions
# =============================================================================


@dataclass
class PostToolResult:
    """
    Result of post-tool audit logging.

    Attributes:
        tool_name: Name of the tool executed
        logged: Whether the execution was logged
        log_path: Path to the audit log file
    """

    tool_name: str
    logged: bool
    log_path: str | None = None


class PostToolAuditHook:
    """
    Hook that logs all tool executions after completion.

    This hook runs after every tool execution (including AUTONOMOUS tools)
    and logs the execution details to the audit log.

    Features:
        - Logs all executions (not just CONFIRM/FORBIDDEN)
        - Records tool name, result, and timestamp
        - Appends to JSONL audit log
    """

    def __init__(
        self,
        audit_path: str = "~/.reachy/audit.jsonl",
        enabled: bool = True,
    ) -> None:
        """
        Initialize post-tool audit hook.

        Args:
            audit_path: Path to audit log file
            enabled: Whether logging is enabled
        """
        from reachy_agent.permissions.evaluator import AuditLogger

        self._audit = AuditLogger(path=audit_path, enabled=enabled)
        self._log = structlog.get_logger("permissions.post_hook")
        self._enabled = enabled
        self._path = audit_path

    async def __call__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        error: str | None = None,
    ) -> PostToolResult:
        """
        Log tool execution after completion.

        Args:
            tool_name: Name of the tool executed
            tool_args: Arguments passed to the tool
            result: Result returned by the tool
            error: Error message if execution failed

        Returns:
            PostToolResult indicating whether logging succeeded
        """
        if not self._enabled:
            return PostToolResult(tool_name=tool_name, logged=False)

        from datetime import datetime, timezone
        from pathlib import Path

        # Create log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "tool_executed",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "success": error is None,
            "error": error,
            "result_type": type(result).__name__ if result is not None else None,
        }

        try:
            path = Path(self._path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            self._log.debug(
                "tool_execution_logged",
                tool=tool_name,
                success=error is None,
            )

            return PostToolResult(
                tool_name=tool_name,
                logged=True,
                log_path=str(path),
            )

        except OSError as e:
            self._log.error("audit_log_failed", error=str(e))
            return PostToolResult(tool_name=tool_name, logged=False)

    async def log(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> PostToolResult:
        """
        Alias for __call__ for explicit usage.

        Args:
            tool_name: Name of tool
            tool_args: Tool arguments
            result: Tool result
            error: Error message if any

        Returns:
            PostToolResult
        """
        return await self.__call__(tool_name, tool_args or {}, result, error)


# =============================================================================
# Factory functions
# =============================================================================


def create_permission_hooks(
    evaluator: PermissionEvaluator,
    confirm_callback: ConfirmCallback | None = None,
    audit_path: str = "~/.reachy/audit.jsonl",
    audit_enabled: bool = True,
) -> tuple[PermissionHook, PostToolAuditHook]:
    """
    Create both pre-tool and post-tool hooks.

    Args:
        evaluator: Permission evaluator instance
        confirm_callback: Optional confirmation callback
        audit_path: Path to audit log
        audit_enabled: Whether audit logging is enabled

    Returns:
        Tuple of (pre_tool_hook, post_tool_hook)
    """
    pre_hook = PermissionHook(evaluator, confirm_callback)
    post_hook = PostToolAuditHook(audit_path=audit_path, enabled=audit_enabled)
    return pre_hook, post_hook
