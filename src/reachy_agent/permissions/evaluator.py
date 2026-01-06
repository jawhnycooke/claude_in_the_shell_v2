"""Permission rule evaluation with audit logging.

This module provides permission evaluation using glob patterns
and JSONL audit logging for all tool calls.

Features:
    - 3-tier permission system (AUTONOMOUS, CONFIRM, FORBIDDEN)
    - Glob pattern matching for tool names
    - First-match-wins rule resolution
    - JSONL audit logging to ~/.reachy/audit.jsonl
"""

import fnmatch
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog


class PermissionTier(Enum):
    """Permission tiers for tool execution."""

    AUTONOMOUS = "autonomous"  # Execute freely
    CONFIRM = "confirm"  # Require user approval
    FORBIDDEN = "forbidden"  # Never execute


@dataclass
class PermissionRule:
    """A rule mapping tool patterns to permission tiers."""

    pattern: str  # Glob pattern, e.g., "move_*", "get_*"
    tier: PermissionTier
    reason: str | None = None  # Optional explanation


@dataclass
class PermissionDecision:
    """Result of evaluating a tool call."""

    tool_name: str
    tier: PermissionTier
    matched_rule: str | None  # Which pattern matched
    allowed: bool
    reason: str | None


# =============================================================================
# F068: Audit logging
# =============================================================================


@dataclass
class AuditEntry:
    """Single audit log entry for tool usage."""

    timestamp: str
    tool_name: str
    tier: str
    matched_rule: str | None
    allowed: bool
    reason: str | None
    tool_args: dict[str, Any] | None = None

    def to_json(self) -> str:
        """Serialize to JSON line."""
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "tool_name": self.tool_name,
                "tier": self.tier,
                "matched_rule": self.matched_rule,
                "allowed": self.allowed,
                "reason": self.reason,
                "tool_args": self.tool_args,
            }
        )


class AuditLogger:
    """
    JSONL audit logger for tool permission decisions.

    Logs every tool call with its permission decision to
    ~/.reachy/audit.jsonl with one JSON object per line.
    """

    def __init__(
        self,
        path: str = "~/.reachy/audit.jsonl",
        enabled: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            path: Path to audit log file
            enabled: Whether logging is enabled
        """
        self._path = Path(path).expanduser()
        self._enabled = enabled
        self._log = structlog.get_logger("permissions.audit")

        if self._enabled:
            # Ensure directory exists
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        tool_name: str,
        tier: PermissionTier,
        matched_rule: str | None,
        allowed: bool,
        reason: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a permission decision.

        Args:
            tool_name: Name of tool
            tier: Permission tier
            matched_rule: Pattern that matched (or None)
            allowed: Whether execution was allowed
            reason: Optional reason for decision
            tool_args: Optional tool arguments
        """
        if not self._enabled:
            return

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            tier=tier.value,
            matched_rule=matched_rule,
            allowed=allowed,
            reason=reason,
            tool_args=tool_args,
        )

        try:
            with open(self._path, "a") as f:
                f.write(entry.to_json() + "\n")
        except OSError as e:
            self._log.error("audit_log_failed", error=str(e))


class PermissionEvaluator:
    """
    Evaluates and enforces tool permissions.

    Uses glob pattern matching to determine if tools can be executed
    autonomously, require confirmation, or are forbidden.

    Features:
        - Glob pattern matching with fnmatch
        - First-match-wins rule resolution
        - Audit logging of all decisions
        - Default tier for unmatched tools
    """

    def __init__(
        self,
        rules: list[PermissionRule],
        default_tier: PermissionTier = PermissionTier.CONFIRM,
        audit_path: str = "~/.reachy/audit.jsonl",
        audit_enabled: bool = True,
    ) -> None:
        """
        Initialize permission evaluator.

        Args:
            rules: List of permission rules (checked in order)
            default_tier: Default tier for unmatched tools
            audit_path: Path to audit log file
            audit_enabled: Whether to log decisions
        """
        self._rules = rules
        self._default = default_tier
        self._audit = AuditLogger(path=audit_path, enabled=audit_enabled)
        self._log = structlog.get_logger("permissions.evaluator")

    def evaluate(
        self, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> PermissionDecision:
        """
        Evaluate permission for a tool call.

        Uses first-match-wins: the first rule whose pattern matches
        the tool name determines the tier.

        Args:
            tool_name: Name of tool to evaluate
            tool_args: Optional tool arguments (for audit logging)

        Returns:
            PermissionDecision with tier, match info, and reasoning
        """
        # Find first matching rule (first-match-wins)
        matched_rule = None
        tier = self._default
        reason = None

        for rule in self._rules:
            if fnmatch.fnmatch(tool_name, rule.pattern):
                matched_rule = rule.pattern
                tier = rule.tier
                reason = rule.reason
                break

        # Build decision
        allowed = tier != PermissionTier.FORBIDDEN

        # Get reason if not from matched rule
        if reason is None:
            reason = self._get_reason(tier, matched_rule)

        decision = PermissionDecision(
            tool_name=tool_name,
            tier=tier,
            matched_rule=matched_rule,
            allowed=allowed,
            reason=reason,
        )

        # Log decision
        self._log.info(
            "permission_evaluated",
            tool=tool_name,
            tier=tier.value,
            allowed=allowed,
            matched=matched_rule,
        )

        # Audit log
        self._audit.log(
            tool_name=tool_name,
            tier=tier,
            matched_rule=matched_rule,
            allowed=allowed,
            reason=reason,
            tool_args=tool_args,
        )

        return decision

    def get_tier(self, tool_name: str) -> PermissionTier:
        """
        Quick lookup of tier without full decision.

        Args:
            tool_name: Name of tool

        Returns:
            Permission tier for the tool
        """
        for rule in self._rules:
            if fnmatch.fnmatch(tool_name, rule.pattern):
                return rule.tier
        return self._default

    def _get_reason(self, tier: PermissionTier, matched_rule: str | None) -> str | None:
        """Get human-readable reason for decision."""
        if matched_rule:
            for rule in self._rules:
                if rule.pattern == matched_rule and rule.reason:
                    return rule.reason

        # Default reasons
        if tier == PermissionTier.FORBIDDEN:
            return "This action is not permitted"
        elif tier == PermissionTier.CONFIRM:
            return "This action requires user approval"

        return None


def load_permissions(
    config_path: str = "config/permissions.yaml",
) -> PermissionEvaluator:
    """
    Load permission rules from YAML config file.

    Args:
        config_path: Path to permissions YAML file

    Returns:
        Configured PermissionEvaluator

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    import yaml  # type: ignore[import-untyped]

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Permissions config not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Parse rules
    rules = [
        PermissionRule(
            pattern=r["pattern"],
            tier=PermissionTier(r["tier"]),
            reason=r.get("reason"),
        )
        for r in config.get("rules", [])
    ]

    # Default tier
    default = PermissionTier(config.get("default_tier", "confirm"))

    # Audit config
    audit_config = config.get("audit", {})
    audit_enabled = audit_config.get("enabled", True)
    audit_path = audit_config.get("path", "~/.reachy/audit.jsonl")

    return PermissionEvaluator(
        rules=rules,
        default_tier=default,
        audit_path=audit_path,
        audit_enabled=audit_enabled,
    )
