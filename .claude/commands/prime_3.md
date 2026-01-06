---
description: Understand claude_agent_hub codebase and summarize key components
lifecycle_phase: explore
---

# Prime

Execute the `Run`, `Read` and `Report` sections to understand the codebase then summarize your understanding.

## Instructions

- We're focused on apps/claude_agent_hub/*.

## Workflow

Run `git ls-files`
READ @README.md
READ @apps/orchestrator_db/README.md
READ @apps/orchestrator_db/models.py
READ @apps/claude_agent_hub/README.md
READ apps/claude_agent_hub/frontend/src/types.d.ts
READ @apps/claude_agent_hub/backend/modules/orch_database_models.py (should mirror models.py)

### If you're requested to work on the backend, read the following files:

READ apps/claude_agent_hub/backend/main.py
READ apps/claude_agent_hub/backend/modules/orchestrator_service.py
READ apps/claude_agent_hub/backend/modules/websocket_manager.py

### If you're requested to work on the frontend, read the following files:

READ apps/claude_agent_hub/frontend/src/components/AgentList.vue
READ apps/claude_agent_hub/frontend/src/components/EventStream.vue
READ apps/claude_agent_hub/frontend/src/components/OrchestratorChat.vue
READ apps/claude_agent_hub/frontend/src/stores/orchestratorStore.ts
READ apps/claude_agent_hub/frontend/src/services/chatService.ts

## Report
Summarize your understanding of the codebase.