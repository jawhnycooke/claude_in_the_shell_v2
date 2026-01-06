---
description: Enter Horizon mode to create a feature list for long-horizon work
argument-hint: <description of project or feature set>
lifecycle_phase: workflow
---

# Horizon Mode: Feature Planning

You are now in **INITIALIZER MODE** for long-horizon work. Your job is to create a comprehensive feature list that will be tracked across multiple sessions.

## Your Task

Based on the user's description: `$ARGUMENTS`

### Step 1: Set Up Horizon Mode Hooks

**CRITICAL**: Before creating the feature list, you MUST configure Horizon mode hooks in `.claude/settings.json`.

Read the current `.claude/settings.json` file (create if it doesn't exist), then update it to include these hooks:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run $CLAUDE_PROJECT_DIR/.claude/hooks/session_start.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "You are evaluating whether Claude should stop working.\n\nBefore approving the stop, verify from the conversation context:\n1. Are ALL features in feature_list.json marked with 'passes': true?\n2. Did Claude update claude-progress.txt with session notes?\n3. Are there uncommitted changes that should be committed?\n4. Were tests run to verify implementations before marking passing?\n\nIf ANY feature has 'passes': false, or verification is incomplete:\n- Set decision to 'block'\n- Explain what still needs to be done\n\nRespond with JSON: {\"decision\": \"approve\" or \"block\", \"reason\": \"your explanation\"}",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

Also ensure the `.claude/hooks/session_start.py` file exists. If it doesn't, inform the user they need to copy it from the hub or create it.

### Step 2: Analyze and Create Feature List

1. **Analyze the request** - Understand what needs to be built
2. **Create feature_list.json** - Write to `.claude/state/feature_list.json`
3. **Order by priority** - Fundamental features first, enhancements last
4. **Include verification steps** - Each feature should have clear test steps

## Feature List Format

Create an array of features in this format:
```json
[
  {
    "id": "F001",
    "category": "infrastructure|functional|style|documentation",
    "description": "Clear description of what this feature does",
    "steps": ["Step 1 to verify", "Step 2 to verify", ...],
    "passes": false
  }
]
```

**ID Format**: Use `F001`, `F002`, etc. These IDs are used to name spawned agents (e.g., `horizon-F001`).

## CRITICAL RULES

- Create 10-50 features depending on project size
- Order features by dependency (foundational first)
- Each feature should be independently verifiable
- DO NOT implement any features - only create the plan
- DO NOT mark any features as passing

## After Creating Feature List

1. Confirm that `.claude/settings.json` has been updated with Horizon hooks
2. Display a summary of features created
3. Suggest running `/horizon-status` to see progress
4. Explain: "Ask me to work on the next feature and I'll spawn an agent with the right context"
5. Note: "Horizon mode hooks are now active - the Stop hook will verify all features pass before ending sessions"
