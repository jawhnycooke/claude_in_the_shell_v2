"""Tests for permission system."""

import json
import tempfile
from pathlib import Path

import pytest

from reachy_agent.permissions.evaluator import (
    AuditEntry,
    AuditLogger,
    PermissionDecision,
    PermissionEvaluator,
    PermissionRule,
    PermissionTier,
    load_permissions,
)

# ==============================================================================
# F064: PermissionTier enum tests
# ==============================================================================


class TestPermissionTier:
    """Tests for PermissionTier enum (F064)."""

    def test_tier_values(self) -> None:
        """Test PermissionTier enum values."""
        assert PermissionTier.AUTONOMOUS.value == "autonomous"
        assert PermissionTier.CONFIRM.value == "confirm"
        assert PermissionTier.FORBIDDEN.value == "forbidden"

    def test_tier_from_string(self) -> None:
        """Test creating PermissionTier from string."""
        assert PermissionTier("autonomous") == PermissionTier.AUTONOMOUS
        assert PermissionTier("confirm") == PermissionTier.CONFIRM
        assert PermissionTier("forbidden") == PermissionTier.FORBIDDEN


# ==============================================================================
# F065: PermissionRule dataclass tests
# ==============================================================================


def test_permission_rule() -> None:
    """
    Comprehensive test for PermissionRule dataclass (F065).

    This test verifies:
    - pattern (glob string) field
    - tier field
    - reason field for FORBIDDEN tools
    """
    # Test basic rule
    rule = PermissionRule(pattern="get_*", tier=PermissionTier.AUTONOMOUS)
    assert rule.pattern == "get_*"
    assert rule.tier == PermissionTier.AUTONOMOUS
    assert rule.reason is None

    # Test rule with reason (for FORBIDDEN)
    forbidden_rule = PermissionRule(
        pattern="exec_*",
        tier=PermissionTier.FORBIDDEN,
        reason="Shell commands are dangerous",
    )
    assert forbidden_rule.pattern == "exec_*"
    assert forbidden_rule.tier == PermissionTier.FORBIDDEN
    assert forbidden_rule.reason == "Shell commands are dangerous"

    # Test rule with CONFIRM tier
    confirm_rule = PermissionRule(
        pattern="move_*",
        tier=PermissionTier.CONFIRM,
        reason="Movement requires approval",
    )
    assert confirm_rule.tier == PermissionTier.CONFIRM


# ==============================================================================
# F066: PermissionEvaluator initialization tests
# ==============================================================================


def test_evaluator_init() -> None:
    """
    Test PermissionEvaluator initialization (F066).

    This test verifies:
    - Loads rules correctly
    - Glob pattern matching works
    - First-match-wins rule resolution
    """
    # Create rules with specific patterns
    rules = [
        PermissionRule("get_*", PermissionTier.AUTONOMOUS),
        PermissionRule("move_*", PermissionTier.CONFIRM),
        PermissionRule("exec_*", PermissionTier.FORBIDDEN),
    ]

    evaluator = PermissionEvaluator(rules, default_tier=PermissionTier.CONFIRM)

    # Verify rules are loaded
    assert len(evaluator._rules) == 3
    assert evaluator._default == PermissionTier.CONFIRM


# ==============================================================================
# F067: evaluate() method tests
# ==============================================================================


def test_evaluate() -> None:
    """
    Test evaluate() method (F067).

    This test verifies:
    - Returns PermissionDecision
    - Matches correct tier based on pattern
    - Includes matched rule info
    """
    rules = [
        PermissionRule("get_*", PermissionTier.AUTONOMOUS),
        PermissionRule("move_*", PermissionTier.CONFIRM),
        PermissionRule("exec_*", PermissionTier.FORBIDDEN),
    ]

    evaluator = PermissionEvaluator(rules)

    # Test AUTONOMOUS match
    decision = evaluator.evaluate("get_status")
    assert isinstance(decision, PermissionDecision)
    assert decision.tool_name == "get_status"
    assert decision.tier == PermissionTier.AUTONOMOUS
    assert decision.matched_rule == "get_*"
    assert decision.allowed is True

    # Test CONFIRM match
    decision = evaluator.evaluate("move_head")
    assert decision.tier == PermissionTier.CONFIRM
    assert decision.matched_rule == "move_*"
    assert decision.allowed is True

    # Test FORBIDDEN match
    decision = evaluator.evaluate("exec_command")
    assert decision.tier == PermissionTier.FORBIDDEN
    assert decision.allowed is False


# ==============================================================================
# F068: Default tier tests
# ==============================================================================


def test_default_tier() -> None:
    """
    Test default tier fallback (F068).

    This test verifies:
    - Unmatched tools use default tier
    - Default tier can be configured
    """
    rules = [
        PermissionRule("get_*", PermissionTier.AUTONOMOUS),
    ]

    # Test with default=CONFIRM
    evaluator = PermissionEvaluator(rules, default_tier=PermissionTier.CONFIRM)
    decision = evaluator.evaluate("unknown_tool")
    assert decision.tier == PermissionTier.CONFIRM
    assert decision.matched_rule is None

    # Test with default=AUTONOMOUS
    evaluator2 = PermissionEvaluator(rules, default_tier=PermissionTier.AUTONOMOUS)
    decision2 = evaluator2.evaluate("another_unknown")
    assert decision2.tier == PermissionTier.AUTONOMOUS


# ==============================================================================
# F069: First-match-wins tests
# ==============================================================================


def test_first_match_wins() -> None:
    """
    Test first-match-wins rule resolution (F069).

    When multiple rules could match, the first one wins.
    """
    rules = [
        PermissionRule("move_head", PermissionTier.AUTONOMOUS),  # More specific
        PermissionRule("move_*", PermissionTier.CONFIRM),  # More general
    ]

    evaluator = PermissionEvaluator(rules)

    # move_head should match first rule (AUTONOMOUS), not second (CONFIRM)
    decision = evaluator.evaluate("move_head")
    assert decision.tier == PermissionTier.AUTONOMOUS
    assert decision.matched_rule == "move_head"

    # move_body should match second rule (CONFIRM)
    decision2 = evaluator.evaluate("move_body")
    assert decision2.tier == PermissionTier.CONFIRM
    assert decision2.matched_rule == "move_*"


# ==============================================================================
# F070: YAML loading tests
# ==============================================================================


def test_load_permissions() -> None:
    """
    Test loading permissions from YAML (F070).

    This test verifies:
    - Loads from config file
    - Parses rules correctly
    - Creates working evaluator
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
default_tier: confirm
rules:
  - pattern: "get_*"
    tier: autonomous
  - pattern: "move_*"
    tier: confirm
    reason: "Movement requires approval"
  - pattern: "exec_*"
    tier: forbidden
    reason: "Shell commands are dangerous"
"""
        )
        f.flush()
        config_path = f.name

    try:
        evaluator = load_permissions(config_path)

        # Test loaded rules work correctly
        decision = evaluator.evaluate("get_status")
        assert decision.tier == PermissionTier.AUTONOMOUS

        decision = evaluator.evaluate("move_head")
        assert decision.tier == PermissionTier.CONFIRM

        decision = evaluator.evaluate("exec_command")
        assert decision.tier == PermissionTier.FORBIDDEN

    finally:
        Path(config_path).unlink()


def test_load_permissions_file_not_found() -> None:
    """Test load_permissions raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_permissions("/nonexistent/path/permissions.yaml")


# ==============================================================================
# F071: Decision reason tests
# ==============================================================================


def test_decision_reason() -> None:
    """
    Test decision reason handling (F071).

    This test verifies:
    - Reasons are included from rules
    - Default reasons for tiers without explicit reason
    """
    rules = [
        PermissionRule(
            "exec_*",
            PermissionTier.FORBIDDEN,
            reason="Shell commands are dangerous",
        ),
        PermissionRule("move_*", PermissionTier.CONFIRM),
        PermissionRule("get_*", PermissionTier.AUTONOMOUS),
    ]

    evaluator = PermissionEvaluator(rules)

    # Test explicit reason
    decision = evaluator.evaluate("exec_command")
    assert decision.reason == "Shell commands are dangerous"

    # Test default reason for CONFIRM (no explicit reason in rule)
    decision = evaluator.evaluate("move_head")
    assert decision.reason == "This action requires user approval"

    # Test no reason for AUTONOMOUS
    decision = evaluator.evaluate("get_status")
    assert decision.reason is None


# ==============================================================================
# Legacy tests (keep for compatibility)
# ==============================================================================


class TestPermissionEvaluator:
    """Tests for PermissionEvaluator."""

    def test_glob_matching(self) -> None:
        """Test glob pattern matching."""
        rules = [
            PermissionRule("get_*", PermissionTier.AUTONOMOUS),
            PermissionRule("exec_*", PermissionTier.FORBIDDEN),
        ]

        evaluator = PermissionEvaluator(rules)
        decision = evaluator.evaluate("get_status")
        assert decision.tier == PermissionTier.AUTONOMOUS

        decision = evaluator.evaluate("exec_command")
        assert decision.tier == PermissionTier.FORBIDDEN
        assert not decision.allowed


# ==============================================================================
# F068: Audit logging tests
# ==============================================================================


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_audit_entry_to_json(self) -> None:
        """Test AuditEntry.to_json() serialization."""
        entry = AuditEntry(
            timestamp="2025-01-20T10:30:00",
            tool_name="move_head",
            tier="autonomous",
            matched_rule="move_*",
            allowed=True,
            reason=None,
            tool_args={"pitch": 10},
        )

        json_str = entry.to_json()
        data = json.loads(json_str)

        assert data["timestamp"] == "2025-01-20T10:30:00"
        assert data["tool_name"] == "move_head"
        assert data["tier"] == "autonomous"
        assert data["allowed"] is True
        assert data["tool_args"] == {"pitch": 10}


def test_audit_logging(tmp_path: Path) -> None:
    """
    Test audit logging (F068).

    This test verifies:
    - Logs tool name, tier, timestamp
    - Logs to ~/.reachy/audit.jsonl (or specified path)
    - One JSON object per line
    """
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=str(log_path), enabled=True)

    # Log a decision
    logger.log(
        tool_name="move_head",
        tier=PermissionTier.AUTONOMOUS,
        matched_rule="move_*",
        allowed=True,
        tool_args={"pitch": 10},
    )

    # Verify log file exists
    assert log_path.exists()

    # Verify content
    content = log_path.read_text().strip()
    lines = content.split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["tool_name"] == "move_head"
    assert data["tier"] == "autonomous"
    assert data["allowed"] is True
    assert "timestamp" in data


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_multiple_entries(self, tmp_path: Path) -> None:
        """Test multiple audit log entries."""
        log_path = tmp_path / "multi_audit.jsonl"
        logger = AuditLogger(path=str(log_path))

        # Log multiple decisions
        logger.log("get_status", PermissionTier.AUTONOMOUS, "get_*", True)
        logger.log(
            "store_memory", PermissionTier.CONFIRM, "store_memory", True, "Approved"
        )
        logger.log("exec_cmd", PermissionTier.FORBIDDEN, "exec_*", False, "Not allowed")

        # Verify three lines
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "tool_name" in data
            assert "tier" in data
            assert "timestamp" in data

    def test_disabled_logging(self, tmp_path: Path) -> None:
        """Test audit logging can be disabled."""
        log_path = tmp_path / "disabled_audit.jsonl"
        logger = AuditLogger(path=str(log_path), enabled=False)

        logger.log("get_status", PermissionTier.AUTONOMOUS, "get_*", True)

        # File should not be created when disabled
        assert not log_path.exists()

    def test_creates_directory(self, tmp_path: Path) -> None:
        """Test audit logger creates parent directory if needed."""
        log_path = tmp_path / "nested" / "dir" / "audit.jsonl"
        logger = AuditLogger(path=str(log_path))

        logger.log("test", PermissionTier.AUTONOMOUS, None, True)

        assert log_path.exists()


# ==============================================================================
# Integration tests
# ==============================================================================


def test_load_actual_config() -> None:
    """Test loading actual permissions.yaml config."""
    evaluator = load_permissions("config/permissions.yaml")

    # Test some expected rules from config
    assert evaluator.evaluate("get_status").tier == PermissionTier.AUTONOMOUS
    assert evaluator.evaluate("move_head").tier == PermissionTier.AUTONOMOUS
    assert evaluator.evaluate("store_memory").tier == PermissionTier.CONFIRM
    assert evaluator.evaluate("exec_command").tier == PermissionTier.FORBIDDEN


# ==============================================================================
# F069-F070: Permission hook tests
# ==============================================================================


class TestPermissionHook:
    """Tests for PermissionHook (F069-F070)."""

    @pytest.fixture
    def evaluator(self) -> PermissionEvaluator:
        """Create test evaluator."""
        rules = [
            PermissionRule("get_*", PermissionTier.AUTONOMOUS),
            PermissionRule("move_*", PermissionTier.CONFIRM),
            PermissionRule("exec_*", PermissionTier.FORBIDDEN, "Dangerous"),
        ]
        return PermissionEvaluator(rules, audit_enabled=False)

    @pytest.mark.asyncio
    async def test_forbidden_hook(self, evaluator: PermissionEvaluator) -> None:
        """
        Test FORBIDDEN tool blocking (F069).

        This test verifies:
        - Blocks FORBIDDEN tools before execution
        - Returns error message to agent
        """
        from reachy_agent.permissions.hooks import PermissionHook

        hook = PermissionHook(evaluator)

        result = await hook("exec_command", {"cmd": "rm -rf /"})

        assert result.allow is False
        assert result.message is not None
        assert "forbidden" in result.message.lower()
        assert result.decision is not None
        assert result.decision.tier == PermissionTier.FORBIDDEN

    @pytest.mark.asyncio
    async def test_autonomous_hook(self, evaluator: PermissionEvaluator) -> None:
        """Test AUTONOMOUS tool is allowed immediately."""
        from reachy_agent.permissions.hooks import PermissionHook

        hook = PermissionHook(evaluator)

        result = await hook("get_status", {})

        assert result.allow is True
        assert result.decision is not None
        assert result.decision.tier == PermissionTier.AUTONOMOUS

    @pytest.mark.asyncio
    async def test_confirm_hook_approved(self, evaluator: PermissionEvaluator) -> None:
        """
        Test CONFIRM tool with approval (F070).

        This test verifies:
        - Prompts for confirmation
        - Allows if confirmed
        """
        from reachy_agent.permissions.hooks import (
            PermissionHook,
            confirm_always_allow,
        )

        hook = PermissionHook(evaluator, confirm_callback=confirm_always_allow)

        result = await hook("move_head", {"pitch": 10})

        assert result.allow is True
        assert result.decision is not None
        assert result.decision.tier == PermissionTier.CONFIRM

    @pytest.mark.asyncio
    async def test_confirm_hook_denied(self, evaluator: PermissionEvaluator) -> None:
        """Test CONFIRM tool with denial."""
        from reachy_agent.permissions.hooks import PermissionHook, confirm_always_deny

        hook = PermissionHook(evaluator, confirm_callback=confirm_always_deny)

        result = await hook("move_head", {"pitch": 10})

        assert result.allow is False
        assert result.message is not None
        assert "denied" in result.message.lower()


# ==============================================================================
# F071: Post-tool audit hook tests
# ==============================================================================


@pytest.mark.asyncio
async def test_post_tool_audit(tmp_path: Path) -> None:
    """
    Test post-tool audit logging (F071).

    This test verifies:
    - Logs all tool executions (even AUTONOMOUS)
    - Includes tool name, result, timestamp
    - Appends to JSONL audit log
    """
    from reachy_agent.permissions.hooks import PostToolAuditHook

    log_path = tmp_path / "post_audit.jsonl"
    hook = PostToolAuditHook(audit_path=str(log_path), enabled=True)

    # Log a successful tool execution
    result = await hook("get_status", {"detail": True}, {"status": "ok"})

    assert result.logged is True
    assert result.tool_name == "get_status"
    assert log_path.exists()

    # Verify log content
    content = log_path.read_text().strip()
    lines = content.split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["tool_name"] == "get_status"
    assert data["event"] == "tool_executed"
    assert data["success"] is True
    assert "timestamp" in data
    assert data["tool_args"] == {"detail": True}
    assert data["result_type"] == "dict"


@pytest.mark.asyncio
async def test_post_tool_audit_with_error(tmp_path: Path) -> None:
    """Test post-tool audit logs errors correctly."""
    from reachy_agent.permissions.hooks import PostToolAuditHook

    log_path = tmp_path / "error_audit.jsonl"
    hook = PostToolAuditHook(audit_path=str(log_path), enabled=True)

    # Log a failed tool execution
    result = await hook("move_head", {"pitch": 100}, None, error="Pitch out of range")

    assert result.logged is True

    # Verify error is recorded
    data = json.loads(log_path.read_text().strip())
    assert data["success"] is False
    assert data["error"] == "Pitch out of range"
    assert data["tool_name"] == "move_head"


@pytest.mark.asyncio
async def test_post_tool_audit_disabled(tmp_path: Path) -> None:
    """Test post-tool audit can be disabled."""
    from reachy_agent.permissions.hooks import PostToolAuditHook

    log_path = tmp_path / "disabled_audit.jsonl"
    hook = PostToolAuditHook(audit_path=str(log_path), enabled=False)

    result = await hook("get_status", {}, {"status": "ok"})

    assert result.logged is False
    assert not log_path.exists()


@pytest.mark.asyncio
async def test_post_tool_audit_multiple_entries(tmp_path: Path) -> None:
    """Test multiple post-tool audit entries."""
    from reachy_agent.permissions.hooks import PostToolAuditHook

    log_path = tmp_path / "multi_post_audit.jsonl"
    hook = PostToolAuditHook(audit_path=str(log_path), enabled=True)

    # Log multiple executions
    await hook("get_status", {}, {"status": "ok"})
    await hook("move_head", {"pitch": 10}, {"success": True})
    await hook("store_memory", {"content": "test"}, {"id": "mem123"})

    # Verify three lines
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 3

    # Each line should be valid JSON with tool_name
    for line in lines:
        data = json.loads(line)
        assert "tool_name" in data
        assert "timestamp" in data
        assert "event" in data
