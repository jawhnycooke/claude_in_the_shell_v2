---
description: Show current progress on horizon features
lifecycle_phase: workflow
---

# Horizon Status Report

Read `.claude/state/feature_list.json` and display:

1. **Summary**: X/Y features passing (Z% complete)
2. **By Category**: Count per category (infrastructure, functional, etc.)
3. **Next Up**: First feature with `passes: false`
4. **Recent Progress**: Last 5 lines of `claude-progress.txt`

If feature_list.json doesn't exist:
- Inform user no horizon work is in progress
- Suggest running `/horizon-features <description>` to start
