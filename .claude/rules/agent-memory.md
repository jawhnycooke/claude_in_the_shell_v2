---
description: Policy guidance for maintaining agent memory across sessions
---

# Agent Memory Management

This document defines policies for maintaining persistent state across Claude Code sessions using the horizon mode pattern (based on Anthropic's autonomous-coding quickstart).

## Feature List Protection Policy

The `feature_list.json` file is a controlled artifact that tracks project features across sessions.

### Modification Rules

**Allowed operations:**
- Change `"passes": false` → `"passes": true` (after successful verification)

**Prohibited operations:**
- Changing `"passes": true` → `"passes": false`
- Editing the `description`, `steps`, or `category` fields
- Deleting features from the list
- Adding new features (only the orchestrator can add features via `/horizon-features`)

These restrictions preserve the integrity of the feature list as a shared contract between the orchestrator and worker agents.

## Verification Requirements

Before setting `"passes": true`, you MUST complete verification appropriate to the feature type:

### For Code Features
1. Run all relevant tests: `uv run pytest -v tests/` or equivalent
2. Confirm tests pass (no failures or errors)
3. Check for type errors if applicable: `uv run mypy .`

### For UI Features
1. Verify visually using Playwright MCP or manual inspection
2. Take a screenshot for documentation if significant
3. Test user interactions (clicks, inputs, navigation)

### For Infrastructure Features
1. Verify the component runs without errors
2. Check logs for expected output
3. Test integration with dependent systems

### General Requirements
- Confirm no regressions in related features
- If you cannot fully verify, leave as `"passes": false`
- Document any partial progress in `claude-progress.txt`

## Runtime Artifacts

The horizon mode creates and manages these runtime artifacts:

| File | Created By | Purpose | Persistence |
|------|-----------|---------|-------------|
| `.claude/state/feature_list.json` | `/horizon-features` command | Tracks features and their pass/fail status | Persists across sessions |
| `claude-progress.txt` | Worker agents | Session notes for context continuity | Persists, appended to |
| `.claude/settings.json` | Setup (manual or automated) | Hook configuration | Persists |

Note: These files are runtime artifacts, not source code. They should be in `.gitignore` for most projects unless you want to version-control the feature list.

## Session Workflow

### At Session Start (Fresh Context)

The SessionStart hook automatically displays a Rich panel with:
- Feature progress summary (X/Y passing)
- Next incomplete feature description
- Recent progress notes (last 5 lines)
- Git status (uncommitted changes)

After seeing the hook output:
1. Read `claude-progress.txt` for full last session context
2. Run `git log --oneline -5` to see recent commits
3. Find the highest-priority feature with `"passes": false`
4. Work on ONE feature at a time

### During Work

1. Focus on implementing the assigned feature
2. Write tests before or alongside implementation (TDD preferred)
3. Verify your work meets the feature's steps

### Before Ending Session

1. Update `claude-progress.txt` with current status:
   - What you accomplished
   - Current state of the work
   - What the next session should focus on
   - Any blockers or important notes
2. Mark verified features as `"passes": true`
3. Commit changes with descriptive messages
4. Leave environment in a clean, working state

## Files You Maintain

### claude-progress.txt (Session Notes)

This file preserves context across sessions. Before ending your session, append:

```
## Session: YYYY-MM-DD HH:MM

### Accomplished
- Item 1
- Item 2

### Current State
Description of where things stand

### Next Session Should
- Focus on X
- Address Y

### Blockers
- None / list any blockers
```

### .claude/state/feature_list.json (Feature Checklist)

Read this at session start to find incomplete features:
```bash
cat .claude/state/feature_list.json | head -50
grep '"passes": false' .claude/state/feature_list.json | wc -l
```

The feature list is managed by pydantic models with this structure:
```json
{
  "id": "F001",
  "category": "infrastructure|functional|style|documentation",
  "description": "Clear description of what this feature does",
  "steps": ["Step 1 to verify", "Step 2 to verify"],
  "passes": false
}
```

The `id` field (e.g., `F001`, `F002`) is used to name spawned agents like `horizon-F001`.
