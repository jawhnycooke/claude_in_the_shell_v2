#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["rich>=13.0"]
# ///
"""
Stop hook - displays session summary when Claude Code exits.

This hook runs when a Claude Code session ends and provides:
1. Quick summary of feature progress
2. Reminder about uncommitted changes
3. Next steps for future sessions
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

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


def get_git_status() -> tuple[bool, int]:
    """Get git status summary.

    Returns:
        Tuple of (has_uncommitted_changes, count_of_changes)
    """
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            return (len(lines) > 0, len(lines))
        return (False, 0)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Git status failed: {e}")
        return (False, 0)


def load_feature_summary(path: Path) -> tuple[int, int, Optional[str]]:
    """Load feature list summary.

    Returns:
        Tuple of (passing_count, total_count, next_feature_description)
    """
    if not path.exists():
        return (0, 0, None)

    try:
        raw_data = json.loads(path.read_text())
        if not isinstance(raw_data, list):
            return (0, 0, None)

        total = len(raw_data)
        passing = sum(1 for f in raw_data if f.get("passes", False))

        # Find next incomplete feature
        next_feature = None
        for f in raw_data:
            if not f.get("passes", False):
                next_feature = f"{f.get('id', '?')}: {f.get('description', '?')}"
                break

        return (passing, total, next_feature)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse feature_list.json: {e}")
        return (0, 0, None)


def main() -> None:
    """Main hook execution."""
    console = Console()
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    state_dir = project_dir / ".claude" / "state"
    feature_list_path = state_dir / "feature_list.json"

    output_sections: List[str] = []

    # 1. Feature progress summary
    passing, total, next_feature = load_feature_summary(feature_list_path)
    if total > 0:
        remaining = total - passing
        progress_pct = (passing / total) * 100
        output_sections.append(f"📊 Progress: {passing}/{total} features ({progress_pct:.0f}%)")
        if next_feature:
            output_sections.append(f"⏭️  Next: {next_feature}")

    # 2. Git status reminder
    has_changes, change_count = get_git_status()
    if has_changes:
        output_sections.append(f"⚠️  {change_count} uncommitted changes - remember to commit!")
    else:
        output_sections.append("✅ Working directory clean")

    # 3. Session end message
    output_sections.append("")
    output_sections.append("💡 Session ended. Progress saved to claude-progress.txt")

    # Output using Rich panel
    if output_sections:
        content = Text("\n".join(output_sections))
        panel = Panel(
            content,
            title="[bold green]Session Complete[/bold green]",
            border_style="green",
            expand=True,
        )
        console.print(panel)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log error but don't fail - stop hooks should be graceful
        logger.warning(f"Stop hook warning: {e}")
        # Still print a basic message
        print("Session ended.")
