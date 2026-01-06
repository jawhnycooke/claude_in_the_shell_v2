#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pydantic>=2.0", "rich>=13.0"]
# ///
"""
SessionStart hook - loads agent state and outputs context (autonomous-coding pattern).

This hook runs at the start of each Claude Code session and displays:
1. Feature list summary from .claude/state/feature_list.json
2. Recent progress notes from claude-progress.txt
3. Git status showing uncommitted changes

The output helps the agent understand current project state and priorities.
"""
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Configure logging to stderr (stdout is for hook output)
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# PYDANTIC MODELS FOR FEATURE LIST
# ═══════════════════════════════════════════════════════════


class Feature(BaseModel):
    """A single feature in the horizon feature list."""

    id: str = Field(
        description="Feature identifier (e.g., 'F001') used for agent naming like 'horizon-F001'"
    )
    category: Literal["infrastructure", "functional", "style", "documentation"] = Field(
        description="Feature category: infrastructure, functional, style, documentation"
    )
    description: str = Field(description="Clear description of what this feature does")
    steps: list[str] = Field(
        default_factory=list, description="Verification steps for this feature"
    )
    passes: bool = Field(
        default=False, description="Whether the feature passes verification"
    )

    @field_validator("id")
    @classmethod
    def validate_feature_id(cls, v: str) -> str:
        """Validate feature ID matches pattern F001, F002, etc."""
        if not re.match(r"^F\d{3,}$", v):
            raise ValueError(
                f"Feature ID '{v}' must match pattern 'F001', 'F002', etc. (F followed by 3+ digits)"
            )
        return v

    @field_validator("steps")
    @classmethod
    def validate_steps_non_empty(cls, v: list[str]) -> list[str]:
        """Ensure steps list is non-empty for verifiable features."""
        if not v:
            raise ValueError("Feature must have at least one verification step")
        # Also ensure each step is non-empty
        for i, step in enumerate(v):
            if not step or not step.strip():
                raise ValueError(f"Verification step {i+1} cannot be empty")
        return v


class FeatureList(BaseModel):
    """Container for parsing feature_list.json as an array."""

    features: list[Feature]

    @classmethod
    def from_json_array(cls, data: list) -> "FeatureList":
        """Parse a JSON array of features into a FeatureList.

        Provides clear error messages with array index context when validation fails.
        """
        features = []
        for i, f in enumerate(data):
            try:
                features.append(Feature(**f))
            except ValidationError as e:
                # Re-raise with array index context for actionable error messages
                raise ValidationError.from_exception_data(
                    title=f"Feature at index {i}",
                    line_errors=e.errors(),
                ) from e
        return cls(features=features)


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════


def get_git_status() -> str | None:
    """Get short git status, returns None on error."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[:5]
            return "\n".join(lines) if lines and lines[0] else None
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Git status failed: {e}")
        return None


def load_feature_list(path: Path) -> FeatureList | None:
    """Load and validate feature_list.json."""
    if not path.exists():
        return None

    try:
        raw_data = json.loads(path.read_text())
        if not isinstance(raw_data, list):
            logger.error(
                f"feature_list.json must be an array, got {type(raw_data).__name__}"
            )
            raise ValueError("feature_list.json must be a JSON array")
        return FeatureList.from_json_array(raw_data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse feature_list.json: {e}")
        raise
    except ValidationError as e:
        logger.error(f"Invalid feature_list.json structure: {e}")
        raise


def load_progress_notes(path: Path, max_lines: int = 5) -> list[str] | None:
    """Load the last N lines of progress notes."""
    if not path.exists():
        return None

    try:
        content = path.read_text().strip()
        if not content:
            return None
        lines = [line for line in content.split("\n")[-max_lines:] if line.strip()]
        return lines if lines else None
    except OSError as e:
        logger.error(f"Failed to read progress notes: {e}")
        raise


# ═══════════════════════════════════════════════════════════
# MAIN OUTPUT
# ═══════════════════════════════════════════════════════════


def main() -> None:
    """Main hook execution."""
    console = Console()
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    state_dir = project_dir / ".claude" / "state"
    feature_list_path = state_dir / "feature_list.json"
    progress_txt_path = project_dir / "claude-progress.txt"

    output_sections: list[str] = []

    # 1. Feature list summary
    try:
        feature_list = load_feature_list(feature_list_path)
        if feature_list:
            total = len(feature_list.features)
            passing = sum(1 for f in feature_list.features if f.passes)
            failing = total - passing

            output_sections.append(
                f"📋 Features: {passing}/{total} passing ({failing} remaining)"
            )

            # Find highest-priority incomplete feature
            for f in feature_list.features:
                if not f.passes:
                    output_sections.append(f"⏭️  Next [{f.id}]: {f.description}")
                    break
        else:
            output_sections.append("📝 feature_list.json not yet created")
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        output_sections.append(f"⚠️  feature_list.json error: {e}")

    # 2. Progress notes
    try:
        progress_lines = load_progress_notes(progress_txt_path)
        if progress_lines:
            output_sections.append("📝 Recent progress notes:")
            for line in progress_lines:
                output_sections.append(f"   {line}")
    except OSError as e:
        output_sections.append(f"⚠️  Progress notes error: {e}")

    # 3. Git status
    git_status = get_git_status()
    if git_status:
        output_sections.append(f"📦 Uncommitted changes:\n{git_status}")

    # Output using Rich panel
    if output_sections:
        content = Text("\n".join(output_sections))
        panel = Panel(
            content,
            title="[bold cyan]Horizon Session State[/bold cyan]",
            border_style="cyan",
            expand=True,
        )
        console.print(panel)
    else:
        panel = Panel(
            "🆕 New session - no previous state found",
            title="[bold cyan]Horizon Session State[/bold cyan]",
            border_style="cyan",
            expand=True,
        )
        console.print(panel)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # CRITICAL: Never silently fail per CLAUDE.md directive
        # Log error with full context before exiting with error code
        logger.error(f"SessionStart hook failed: {e}", exc_info=True)
        sys.exit(1)
