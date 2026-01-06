"""ChromaDB-based memory manager for persistent semantic memory."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, cast

import chromadb
from chromadb.types import Where
import structlog


class MemoryType(Enum):
    """
    Memory types with different expiry policies.

    - CONVERSATION: Chat history, expires after 30 days
    - FACT: Permanent facts, never expires
    - CONTEXT: Session context, expires at end of day
    """

    CONVERSATION = "conversation"  # 30 day expiry
    FACT = "fact"  # Permanent
    CONTEXT = "context"  # Same day expiry


@dataclass
class Memory:
    """
    A single memory entry with metadata.

    Attributes:
        id: Unique identifier
        content: Memory content text
        memory_type: Type of memory (affects expiry)
        created_at: Creation timestamp
        metadata: Additional metadata dict
    """

    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> datetime | None:
        """
        Calculate expiry time based on memory type.

        Returns:
            Expiry datetime, or None if never expires (FACT)
        """
        if self.memory_type == MemoryType.CONVERSATION:
            return self.created_at + timedelta(days=30)
        elif self.memory_type == MemoryType.CONTEXT:
            return self.created_at.replace(hour=23, minute=59, second=59)
        else:  # FACT
            return None  # Never expires

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Memory search result with relevance score."""

    memory: Memory
    score: float  # 0-1, higher is more relevant


class MemoryManager:
    """
    Manages all memory operations using ChromaDB.

    Provides semantic search over memories using sentence embeddings
    and maintains a conversation context window for recent turns.

    Examples:
        >>> manager = MemoryManager()
        >>> await manager.store("The user prefers dark mode", MemoryType.FACT)
        >>> results = await manager.search("user preferences")
        >>> for r in results:
        ...     print(f"{r.memory.content} (score: {r.score:.2f})")
    """

    def __init__(
        self,
        persist_path: str = "~/.reachy/memory",
        context_window_size: int = 5,
        collection_name: str = "reachy_memories",
    ):
        """
        Initialize memory manager with ChromaDB backend.

        Args:
            persist_path: Path for ChromaDB persistence
            context_window_size: Number of recent turns to keep in context
            collection_name: Name of the ChromaDB collection
        """
        self._persist_path = Path(persist_path).expanduser()
        self._persist_path.mkdir(parents=True, exist_ok=True)

        self._context_window_size = context_window_size
        self._context_window: list[dict[str, Any]] = []
        self._log = structlog.get_logger("memory")

        # Initialize ChromaDB with persistence
        self._client = chromadb.PersistentClient(path=str(self._persist_path))

        # Get or create collection with default embedding function
        # ChromaDB uses all-MiniLM-L6-v2 by default
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        self._log.info(
            "memory_manager_initialized",
            persist_path=str(self._persist_path),
            collection=collection_name,
        )

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """
        Store a new memory in ChromaDB.

        Args:
            content: Memory content text
            memory_type: Type of memory
            metadata: Optional metadata

        Returns:
            Stored memory object

        Examples:
            >>> memory = await manager.store("User's name is Alice", MemoryType.FACT)
            >>> print(memory.id)
        """
        memory_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        # Build metadata for ChromaDB
        chroma_metadata = {
            "memory_type": memory_type.value,
            "created_at": created_at.isoformat(),
            **(metadata or {}),
        }

        # Store in ChromaDB (embedding generated automatically)
        self._collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[chroma_metadata],
        )

        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            created_at=created_at,
            metadata=metadata or {},
        )

        self._log.debug(
            "memory_stored",
            id=memory_id,
            type=memory_type.value,
            content_length=len(content),
        )

        return memory

    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """
        Semantic search over memories.

        Args:
            query: Search query text
            memory_type: Optional type filter
            limit: Maximum results
            min_score: Minimum relevance score (0-1)

        Returns:
            List of search results sorted by relevance

        Examples:
            >>> results = await manager.search("user preferences", limit=3)
            >>> for r in results:
            ...     print(f"{r.memory.content[:50]}... (score: {r.score:.2f})")
        """
        # Build where clause for optional type filter
        where: Where | None = None
        if memory_type is not None:
            where = cast(Where, {"memory_type": memory_type.value})

        # Query ChromaDB
        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results: list[SearchResult] = []

        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, memory_id in enumerate(ids):
                # Convert distance to similarity score (cosine distance to similarity)
                distance = distances[i] if i < len(distances) else 0
                score = 1.0 - distance  # Cosine distance to similarity

                if score < min_score:
                    continue

                doc = documents[i] if i < len(documents) else ""
                meta = metadatas[i] if i < len(metadatas) else {}

                # Parse memory type
                mem_type_str = meta.get("memory_type", "fact")
                try:
                    mem_type = MemoryType(mem_type_str)
                except ValueError:
                    mem_type = MemoryType.FACT

                # Parse created_at
                created_str = str(meta.get("created_at", ""))
                try:
                    created_at = datetime.fromisoformat(created_str)
                except (ValueError, TypeError):
                    created_at = datetime.utcnow()

                # Extract custom metadata (exclude system fields)
                custom_meta = {
                    k: v
                    for k, v in meta.items()
                    if k not in ("memory_type", "created_at")
                }

                memory = Memory(
                    id=memory_id,
                    content=doc,
                    memory_type=mem_type,
                    created_at=created_at,
                    metadata=custom_meta,
                )

                search_results.append(SearchResult(memory=memory, score=score))

        self._log.debug(
            "memory_search",
            query=query[:50],
            results=len(search_results),
            type_filter=memory_type.value if memory_type else None,
        )

        return search_results

    async def forget(self, memory_id: str) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            # Check if exists first
            existing = self._collection.get(ids=[memory_id])
            if not existing["ids"]:
                return False

            self._collection.delete(ids=[memory_id])
            self._log.info("memory_deleted", id=memory_id)
            return True

        except Exception as e:
            self._log.error("memory_delete_error", id=memory_id, error=str(e))
            return False

    async def get(self, memory_id: str) -> Memory | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: ID of memory to retrieve

        Returns:
            Memory object or None if not found
        """
        try:
            result = self._collection.get(
                ids=[memory_id], include=["documents", "metadatas"]
            )

            if not result["ids"]:
                return None

            doc = result["documents"][0] if result["documents"] else ""
            meta = result["metadatas"][0] if result["metadatas"] else {}

            mem_type_str = meta.get("memory_type", "fact")
            try:
                mem_type = MemoryType(mem_type_str)
            except ValueError:
                mem_type = MemoryType.FACT

            created_str = str(meta.get("created_at", ""))
            try:
                created_at = datetime.fromisoformat(created_str)
            except (ValueError, TypeError):
                created_at = datetime.utcnow()

            custom_meta = {
                k: v for k, v in meta.items() if k not in ("memory_type", "created_at")
            }

            return Memory(
                id=memory_id,
                content=doc,
                memory_type=mem_type,
                created_at=created_at,
                metadata=custom_meta,
            )

        except Exception as e:
            self._log.error("memory_get_error", id=memory_id, error=str(e))
            return None

    # Context window methods

    def add_to_context_window(self, role: str, content: str) -> None:
        """
        Add a turn to the context window.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self._context_window.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Keep only last N turns (* 2 for user+assistant pairs)
        max_items = self._context_window_size * 2
        if len(self._context_window) > max_items:
            self._context_window = self._context_window[-max_items:]

    def get_context_window(self) -> list[dict[str, Any]]:
        """
        Get recent conversation turns for prompt inclusion.

        Returns:
            List of recent turns with role and content (newest first)
        """
        return list(reversed(self._context_window))

    def clear_context_window(self) -> None:
        """Clear the context window (new session)."""
        self._context_window = []

    # Convenience methods

    async def remember_fact(self, content: str, **metadata: Any) -> Memory:
        """Store a permanent fact."""
        return await self.store(content, MemoryType.FACT, metadata)

    async def remember_conversation(self, content: str, **metadata: Any) -> Memory:
        """Store conversation (30 day expiry)."""
        return await self.store(content, MemoryType.CONVERSATION, metadata)

    async def set_context(self, key: str, value: str) -> Memory:
        """Set session context (expires end of day)."""
        return await self.store(value, MemoryType.CONTEXT, {"context_key": key})

    async def cleanup_expired(self) -> int:
        """
        Remove expired memories.

        Returns:
            Count of memories deleted
        """
        now = datetime.utcnow()
        deleted_count = 0

        # Get all memories
        all_results = self._collection.get(include=["metadatas"])

        if not all_results["ids"]:
            return 0

        for i, memory_id in enumerate(all_results["ids"]):
            meta = all_results["metadatas"][i] if all_results["metadatas"] else {}

            mem_type_str = str(meta.get("memory_type", "fact"))
            created_str = str(meta.get("created_at", ""))

            try:
                mem_type = MemoryType(mem_type_str)
            except ValueError:
                continue

            # FACT never expires
            if mem_type == MemoryType.FACT:
                continue

            try:
                created_at = datetime.fromisoformat(created_str)
            except (ValueError, TypeError):
                continue

            # Check expiry
            if mem_type == MemoryType.CONVERSATION:
                expires_at = created_at + timedelta(days=30)
            elif mem_type == MemoryType.CONTEXT:
                expires_at = created_at.replace(hour=23, minute=59, second=59)
            else:
                continue

            if now > expires_at:
                self._collection.delete(ids=[memory_id])
                deleted_count += 1
                self._log.debug("memory_expired", id=memory_id, type=mem_type_str)

        if deleted_count > 0:
            self._log.info("memory_cleanup", deleted=deleted_count)

        return deleted_count

    def count(self, memory_type: MemoryType | None = None) -> int:
        """
        Count memories in the collection.

        Args:
            memory_type: Optional filter by type

        Returns:
            Number of memories
        """
        if memory_type is None:
            return self._collection.count()

        results = self._collection.get(
            where={"memory_type": memory_type.value}, include=[]
        )
        return len(results["ids"]) if results["ids"] else 0
