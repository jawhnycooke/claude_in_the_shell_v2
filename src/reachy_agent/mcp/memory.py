"""Memory MCP server with 3 tools for semantic memory operations.

This server provides MCP tools for storing, searching, and forgetting memories
using ChromaDB-backed semantic search.

Features:
    - 3 MCP tools for memory management
    - ChromaDB-based semantic search
    - Memory type filtering (FACT, CONVERSATION, CONTEXT)
"""

from typing import Any

import structlog
from fastmcp import FastMCP

from reachy_agent.memory.manager import MemoryManager, MemoryType

# Create FastMCP server
app = FastMCP("memory")
log = structlog.get_logger("mcp.memory")

# Memory manager instance (initialized lazily)
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get or create MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def _parse_memory_type(type_str: str | None) -> MemoryType | None:
    """Parse memory type string to enum."""
    if type_str is None:
        return None
    try:
        return MemoryType(type_str.lower())
    except ValueError:
        log.warning("invalid_memory_type", type=type_str)
        return None


# =============================================================================
# F061: store_memory tool
# =============================================================================


@app.tool
async def store_memory(
    content: str,
    memory_type: str = "fact",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Store a new memory with semantic embedding.

    Args:
        content: What to remember (text content)
        memory_type: One of "conversation" (30 day expiry),
                     "fact" (permanent), or "context" (today only)
        metadata: Optional key-value pairs for additional context

    Returns:
        dict: The stored memory with its generated ID

    Examples:
        store_memory("The user's favorite color is blue", "fact")
        store_memory("User asked about weather", "conversation", {"topic": "weather"})
    """
    manager = get_memory_manager()

    # Parse and validate memory type
    mem_type = _parse_memory_type(memory_type)
    if mem_type is None:
        mem_type = MemoryType.FACT
        log.info("defaulting_to_fact", original_type=memory_type)

    try:
        memory = await manager.store(
            content=content,
            memory_type=mem_type,
            metadata=metadata,
        )

        log.info(
            "memory_stored",
            id=memory.id,
            type=memory.memory_type.value,
            content_preview=content[:50],
        )

        return {
            "success": True,
            "memory_id": memory.id,
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            "created_at": memory.created_at.isoformat(),
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        }

    except Exception as e:
        log.error("store_memory_error", error=str(e))
        return {"success": False, "error": str(e)}


# =============================================================================
# F062: search_memory tool
# =============================================================================


@app.tool
async def search_memory(
    query: str,
    memory_type: str | None = None,
    limit: int = 5,
    min_score: float = 0.3,
) -> dict[str, Any]:
    """
    Search memories semantically using vector similarity.

    Args:
        query: Search query text (will be embedded and compared)
        memory_type: Optional filter - "conversation", "fact", or "context"
        limit: Maximum number of results to return (default: 5)
        min_score: Minimum relevance score 0-1 (default: 0.3)

    Returns:
        dict: Search results with relevance scores

    Examples:
        search_memory("user preferences")
        search_memory("weather discussions", memory_type="conversation", limit=3)
    """
    manager = get_memory_manager()

    # Parse optional memory type filter
    mem_type = _parse_memory_type(memory_type)

    try:
        results = await manager.search(
            query=query,
            memory_type=mem_type,
            limit=limit,
            min_score=min_score,
        )

        log.info(
            "memory_search",
            query=query[:50],
            type_filter=memory_type,
            results_count=len(results),
        )

        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "memory_id": r.memory.id,
                    "content": r.memory.content,
                    "memory_type": r.memory.memory_type.value,
                    "score": round(r.score, 4),
                    "created_at": r.memory.created_at.isoformat(),
                    "metadata": r.memory.metadata,
                }
                for r in results
            ],
        }

    except Exception as e:
        log.error("search_memory_error", error=str(e))
        return {"success": False, "error": str(e), "results": []}


# =============================================================================
# F063: forget_memory tool
# =============================================================================


@app.tool
async def forget_memory(memory_id: str) -> dict[str, Any]:
    """
    Delete a specific memory by ID.

    Args:
        memory_id: The unique ID of the memory to delete

    Returns:
        dict: Success status indicating if memory was deleted

    Examples:
        forget_memory("abc123-def456-...")
    """
    manager = get_memory_manager()

    try:
        success = await manager.forget(memory_id)

        if success:
            log.info("memory_forgotten", id=memory_id)
        else:
            log.warning("memory_not_found", id=memory_id)

        return {
            "success": success,
            "memory_id": memory_id,
            "message": "Memory deleted" if success else "Memory not found",
        }

    except Exception as e:
        log.error("forget_memory_error", id=memory_id, error=str(e))
        return {"success": False, "error": str(e), "memory_id": memory_id}


if __name__ == "__main__":
    # Run MCP server
    app.run()
