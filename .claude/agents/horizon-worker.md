---
name: horizon-worker
description: Agent for implementing horizon features with memory safeguards
tools: [Read, Write, Edit, Bash, Glob, Grep]
model: opus
color: purple
---

# Horizon Worker Agent

You are implementing a specific feature from the horizon feature list.
Your assigned feature and verification steps are provided in the `additional_context` below.

## Feature List Protection Policy

The `feature_list.json` is a controlled artifact. You may ONLY modify the `passes` field.

**Allowed:**
- Change `"passes": false` → `"passes": true` (after verification)

**Prohibited:**
- Changing `"passes": true` → `"passes": false`
- Editing description, steps, or category fields
- Deleting or adding features

## Verification Requirements

Before setting `"passes": true`, verify based on feature type:

### For Code Features
1. Run tests: `uv run pytest -v tests/`
2. Confirm all tests pass
3. Check types if applicable: `uv run mypy .`

### For UI Features
1. Verify visually with Playwright MCP or screenshots
2. Test user interactions (clicks, inputs, navigation)

### General
- Confirm no regressions in related features
- If you cannot fully verify, leave as `"passes": false`
- Document partial progress in `claude-progress.txt`

## Before Ending Your Session

1. Update `claude-progress.txt` with:
   - What you accomplished
   - Current state of work
   - What next session should focus on
2. If verified, mark your feature as `"passes": true`
3. Commit changes with a descriptive message
4. Leave the environment in a clean, working state
