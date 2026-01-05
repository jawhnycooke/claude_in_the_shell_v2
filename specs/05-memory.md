# Memory System Specification

## Overview

The memory system provides semantic memory storage using **ChromaDB only** - no SQLite. Three memory types cover all needs: CONVERSATION (dialog history), FACT (permanent knowledge), and CONTEXT (session data). A built-in conversation window automatically includes recent turns in prompts.

## Design Principles

1. **Single backend** - ChromaDB for everything. No dual-storage complexity.
2. **Three types** - CONVERSATION, FACT, CONTEXT cover all use cases.
3. **Built-in context window** - Last N turns automatically in prompt.
4. **Simple search** - Query + optional type filter. That's it.
5. **Metadata-rich** - Store anything as metadata, filter on it.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Memory Manager                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Memory API                              │   │
│  │  store() │ search() │ get_context_window() │ forget()    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                Context Window                             │   │
│  │            Last N turns (default 5)                       │   │
│  │         Automatically included in prompts                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ChromaDB                               │   │
│  │  Collection: "memories"                                   │   │
│  │  Embeddings: all-MiniLM-L6-v2 (384 dim)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Types

| Type | Purpose | Expiry | Examples |
|------|---------|--------|----------|
| `CONVERSATION` | Dialog history | 30 days | User queries, agent responses |
| `FACT` | Things to remember | Permanent | User preferences, learned info |
| `CONTEXT` | Session data | Same day | Current task, active topic |

### Why Only 3 Types?

v1 had 6 types (CONVERSATION, OBSERVATION, FACT, PREFERENCE, EVENT, TASK). Here's the simplification:

| v1 Type | v2 Mapping | Rationale |
|---------|------------|-----------|
| CONVERSATION | CONVERSATION | Keep as-is |
| OBSERVATION | CONVERSATION | Observations are just context from conversation |
| FACT | FACT | Keep as-is |
| PREFERENCE | FACT | Preferences are facts about the user |
| EVENT | CONTEXT | Events are temporary context |
| TASK | CONTEXT | Active tasks are session context |

## Data Model

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

class MemoryType(Enum):
    CONVERSATION = "conversation"  # 30 day expiry
    FACT = "fact"                  # Permanent
    CONTEXT = "context"            # Same day expiry

@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed properties
    @property
    def expires_at(self) -> datetime | None:
        match self.memory_type:
            case MemoryType.CONVERSATION:
                return self.created_at + timedelta(days=30)
            case MemoryType.CONTEXT:
                return self.created_at.replace(hour=23, minute=59, second=59)
            case MemoryType.FACT:
                return None  # Never expires

@dataclass
class SearchResult:
    """Memory search result with relevance score."""
    memory: Memory
    score: float  # 0-1, higher is more relevant
```

## Memory Manager

```python
import chromadb
from sentence_transformers import SentenceTransformer

class MemoryManager:
    """Manages all memory operations."""

    def __init__(
        self,
        persist_path: str = "~/.reachy/memory",
        context_window_size: int = 5
    ):
        self._persist_path = Path(persist_path).expanduser()
        self._context_window_size = context_window_size
        self._context_window: list[dict] = []

        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(path=str(self._persist_path))
        self._collection = self._client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}
        )

        # Embedding model
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # =========== Core Operations ===========

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None
    ) -> Memory:
        """Store a new memory."""
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Generate embedding
        embedding = self._embedder.encode(content).tolist()

        # Prepare metadata (ChromaDB only stores primitives)
        stored_metadata = {
            "type": memory_type.value,
            "created_at": now.isoformat(),
            **(metadata or {})
        }

        # Store in ChromaDB
        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[stored_metadata]
        )

        return Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            created_at=now,
            metadata=metadata or {}
        )

    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 5,
        min_score: float = 0.3
    ) -> list[SearchResult]:
        """Semantic search over memories."""
        # Generate query embedding
        query_embedding = self._embedder.encode(query).tolist()

        # Build filter
        where_filter = None
        if memory_type:
            where_filter = {"type": memory_type.value}

        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter
        )

        # Parse results
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                # ChromaDB returns distance, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Cosine distance to similarity

                if score < min_score:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                memory = Memory(
                    id=id,
                    content=results["documents"][0][i],
                    memory_type=MemoryType(metadata.get("type", "fact")),
                    created_at=datetime.fromisoformat(metadata.get("created_at", "")),
                    metadata={k: v for k, v in metadata.items()
                              if k not in ("type", "created_at")}
                )
                search_results.append(SearchResult(memory=memory, score=score))

        return search_results

    async def forget(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        try:
            self._collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    async def cleanup_expired(self) -> int:
        """Remove expired memories. Returns count deleted."""
        now = datetime.utcnow()
        deleted = 0

        # Get all memories (ChromaDB doesn't support complex date queries)
        all_results = self._collection.get()

        ids_to_delete = []
        for i, id in enumerate(all_results["ids"]):
            metadata = all_results["metadatas"][i]
            memory_type = MemoryType(metadata.get("type", "fact"))
            created_at = datetime.fromisoformat(metadata.get("created_at", ""))

            # Check expiry based on type
            if memory_type == MemoryType.CONVERSATION:
                if now - created_at > timedelta(days=30):
                    ids_to_delete.append(id)
            elif memory_type == MemoryType.CONTEXT:
                if created_at.date() < now.date():
                    ids_to_delete.append(id)

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            deleted = len(ids_to_delete)

        return deleted

    # =========== Context Window ===========

    def add_to_context_window(self, role: str, content: str):
        """Add a turn to the context window."""
        self._context_window.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep only last N turns
        if len(self._context_window) > self._context_window_size * 2:
            # *2 because each turn has user + assistant
            self._context_window = self._context_window[-self._context_window_size * 2:]

    def get_context_window(self) -> list[dict]:
        """Get recent conversation turns for prompt inclusion."""
        return list(self._context_window)

    def clear_context_window(self):
        """Clear the context window (new session)."""
        self._context_window = []

    # =========== Convenience Methods ===========

    async def remember_fact(self, content: str, **metadata) -> Memory:
        """Shorthand for storing a permanent fact."""
        return await self.store(content, MemoryType.FACT, metadata)

    async def remember_conversation(self, content: str, **metadata) -> Memory:
        """Shorthand for storing conversation."""
        return await self.store(content, MemoryType.CONVERSATION, metadata)

    async def set_context(self, key: str, value: str) -> Memory:
        """Set session context (expires end of day)."""
        return await self.store(
            value,
            MemoryType.CONTEXT,
            {"context_key": key}
        )

    async def get_context(self, key: str) -> str | None:
        """Get session context by key."""
        results = self._collection.get(
            where={"context_key": key}
        )
        if results["documents"]:
            return results["documents"][0]
        return None
```

## MCP Tools (3 tools)

```python
from mcp import FastMCP

app = FastMCP("memory")

@app.tool
async def search_memories(
    query: str,
    memory_type: str | None = None,
    limit: int = 5
) -> list[dict]:
    """
    Search memories semantically.

    Args:
        query: What to search for
        memory_type: Optional filter - "conversation", "fact", or "context"
        limit: Max results to return

    Returns:
        List of matching memories with relevance scores
    """
    type_filter = MemoryType(memory_type) if memory_type else None
    results = await memory_manager.search(query, type_filter, limit)

    return [
        {
            "content": r.memory.content,
            "type": r.memory.memory_type.value,
            "score": r.score,
            "created_at": r.memory.created_at.isoformat(),
            "metadata": r.memory.metadata
        }
        for r in results
    ]

@app.tool
async def store_memory(
    content: str,
    memory_type: str = "fact",
    metadata: dict | None = None
) -> dict:
    """
    Store a new memory.

    Args:
        content: What to remember
        memory_type: "conversation" (30d), "fact" (permanent), or "context" (today)
        metadata: Optional key-value pairs

    Returns:
        The stored memory
    """
    memory = await memory_manager.store(
        content,
        MemoryType(memory_type),
        metadata
    )

    return {
        "id": memory.id,
        "content": memory.content,
        "type": memory.memory_type.value,
        "created_at": memory.created_at.isoformat()
    }

@app.tool
async def forget_memory(memory_id: str) -> dict:
    """
    Delete a specific memory.

    Args:
        memory_id: ID of memory to delete

    Returns:
        Success status
    """
    success = await memory_manager.forget(memory_id)
    return {"success": success, "id": memory_id}
```

## Context Window Integration

The context window automatically provides recent conversation history to Claude:

```python
class ReachyAgentLoop:
    def __init__(self, memory: MemoryManager):
        self._memory = memory

    async def process(self, user_input: str) -> str:
        # Add user message to context window
        self._memory.add_to_context_window("user", user_input)

        # Build prompt with context window
        context = self._memory.get_context_window()
        messages = [
            {"role": "system", "content": self._system_prompt},
            *[{"role": c["role"], "content": c["content"]} for c in context[:-1]],
            {"role": "user", "content": user_input}
        ]

        # Call Claude
        response = await self._client.messages.create(
            model=self._model,
            messages=messages,
            # ...
        )

        assistant_content = response.content[0].text

        # Add assistant response to context window
        self._memory.add_to_context_window("assistant", assistant_content)

        return assistant_content
```

This means Claude always has access to the last ~5 conversation turns without needing to search memory. For older context, Claude can use `search_memories`.

## User Profiles via FACT Memories

User profiles are just FACT memories with specific metadata:

```python
# Store user preference
await memory.store(
    "User prefers to be called Alex",
    MemoryType.FACT,
    {"category": "user_profile", "field": "name"}
)

await memory.store(
    "User is interested in robotics and AI",
    MemoryType.FACT,
    {"category": "user_profile", "field": "interests"}
)

# Search user profile facts
results = await memory.search(
    "user preferences",
    memory_type=MemoryType.FACT
)
```

No separate SQLite table needed - it's all in ChromaDB with metadata filtering.

## Storage Layout

```
~/.reachy/
└── memory/
    └── chroma.sqlite3    # ChromaDB persistence
```

Single file. Simple backup (`cp`). Simple reset (`rm -rf`).

## Configuration

```yaml
# config/default.yaml
memory:
  path: ~/.reachy/memory
  context_window_size: 5
  embedding_model: all-MiniLM-L6-v2
  cleanup_interval: 3600  # seconds, run cleanup hourly
```

## Startup Cleanup

On agent startup, clean up expired memories:

```python
async def startup():
    deleted = await memory_manager.cleanup_expired()
    if deleted > 0:
        log.info("memory_cleanup", deleted=deleted)
```

## What Changed from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Backends | ChromaDB + SQLite | ChromaDB only |
| Memory types | 6 | 3 |
| User profiles | Separate SQLite table | FACT memories with metadata |
| Sessions | Separate SQLite table | CONTEXT memories |
| Context window | Not built-in | Built-in (last N turns) |
| MCP tools | 4 | 3 (removed get_user_profile) |

## Why ChromaDB Only?

v1 used ChromaDB for vectors and SQLite for profiles/sessions. Problems:

1. **Two systems to maintain** - Different backup, migration, debugging
2. **Impedance mismatch** - Had to sync data between systems
3. **Unnecessary complexity** - ChromaDB already stores metadata

ChromaDB stores documents + embeddings + arbitrary metadata. User profiles are just documents with `{"category": "user_profile"}` metadata. Sessions are CONTEXT memories that expire at end of day.

## Related Specs

- [01-overview.md](./01-overview.md) - System architecture
- [02-robot-control.md](./02-robot-control.md) - MCP tools architecture
