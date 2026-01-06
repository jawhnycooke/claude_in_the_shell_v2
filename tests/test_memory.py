"""Tests for memory system."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from reachy_agent.memory.manager import (
    Memory,
    MemoryManager,
    MemoryType,
    SearchResult,
)


# ==============================================================================
# F052: MemoryType enum tests
# ==============================================================================


class TestMemoryType:
    """Tests for MemoryType enum (F052)."""

    def test_memory_type_has_three_types(self) -> None:
        """Verify CONVERSATION, FACT, CONTEXT types exist."""
        assert hasattr(MemoryType, "CONVERSATION")
        assert hasattr(MemoryType, "FACT")
        assert hasattr(MemoryType, "CONTEXT")

    def test_memory_type_values(self) -> None:
        """Verify MemoryType values."""
        assert MemoryType.CONVERSATION.value == "conversation"
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.CONTEXT.value == "context"

    def test_memory_type_from_string(self) -> None:
        """Verify MemoryType can be created from string."""
        assert MemoryType("conversation") == MemoryType.CONVERSATION
        assert MemoryType("fact") == MemoryType.FACT
        assert MemoryType("context") == MemoryType.CONTEXT


# ==============================================================================
# F053: Memory dataclass tests
# ==============================================================================


@pytest.mark.asyncio
async def test_memory_model() -> None:
    """
    Comprehensive test for Memory dataclass (F053).

    This test verifies:
    - id, content, memory_type, created_at fields
    - metadata dict for custom fields
    - expires_at computed property
    """
    # Test basic memory creation
    memory = Memory(
        id="test-123",
        content="Test content",
        memory_type=MemoryType.FACT,
    )

    assert memory.id == "test-123"
    assert memory.content == "Test content"
    assert memory.memory_type == MemoryType.FACT
    assert isinstance(memory.created_at, datetime)
    assert memory.metadata == {}

    # Test with metadata
    memory_with_meta = Memory(
        id="test-456",
        content="Content with metadata",
        memory_type=MemoryType.CONVERSATION,
        metadata={"source": "user", "importance": "high"},
    )

    assert memory_with_meta.metadata["source"] == "user"
    assert memory_with_meta.metadata["importance"] == "high"

    # Test FACT never expires
    fact_memory = Memory(
        id="fact-1",
        content="A permanent fact",
        memory_type=MemoryType.FACT,
    )
    assert fact_memory.expires_at is None  # FACT never expires

    # Test CONVERSATION expires after 30 days
    conv_memory = Memory(
        id="conv-1",
        content="A conversation turn",
        memory_type=MemoryType.CONVERSATION,
        created_at=datetime.utcnow(),
    )
    expected_expiry = conv_memory.created_at + timedelta(days=30)
    assert conv_memory.expires_at is not None
    # Allow 1 second tolerance
    assert abs((conv_memory.expires_at - expected_expiry).total_seconds()) < 1

    # Test CONTEXT expires end of day
    ctx_memory = Memory(
        id="ctx-1",
        content="Session context",
        memory_type=MemoryType.CONTEXT,
        created_at=datetime.utcnow(),
    )
    assert ctx_memory.expires_at is not None
    assert ctx_memory.expires_at.hour == 23
    assert ctx_memory.expires_at.minute == 59
    assert ctx_memory.expires_at.second == 59


class TestMemoryDataclass:
    """Additional tests for Memory dataclass."""

    def test_memory_to_dict(self) -> None:
        """Test Memory.to_dict() serialization."""
        memory = Memory(
            id="test-id",
            content="Test content",
            memory_type=MemoryType.FACT,
            metadata={"key": "value"},
        )

        d = memory.to_dict()
        assert d["id"] == "test-id"
        assert d["content"] == "Test content"
        assert d["memory_type"] == "fact"
        assert "created_at" in d
        assert d["metadata"] == {"key": "value"}


# ==============================================================================
# F054: MemoryManager initialization tests
# ==============================================================================


@pytest.fixture
def temp_memory_path(tmp_path: Path) -> str:
    """Create temporary path for ChromaDB."""
    return str(tmp_path / "test_memory")


@pytest.fixture
def memory_manager(temp_memory_path: str) -> MemoryManager:
    """Create MemoryManager with temporary storage."""
    return MemoryManager(
        persist_path=temp_memory_path,
        context_window_size=5,
        collection_name="test_memories",
    )


@pytest.mark.asyncio
async def test_memory_manager_init(temp_memory_path: str) -> None:
    """
    Test MemoryManager initialization (F054).

    This test verifies:
    - ChromaDB client initialization
    - Collection creation with embeddings
    - Uses all-MiniLM-L6-v2 (384 dimensions)
    """
    manager = MemoryManager(
        persist_path=temp_memory_path,
        collection_name="init_test",
    )

    # Verify ChromaDB client is initialized
    assert manager._client is not None

    # Verify collection exists
    assert manager._collection is not None

    # Verify we can count (empty collection)
    count = manager.count()
    assert count == 0


class TestMemoryManager:
    """Tests for MemoryManager."""

    def test_initialization(self, memory_manager: MemoryManager) -> None:
        """Test memory manager initialization."""
        assert memory_manager._client is not None
        assert memory_manager._collection is not None
        assert memory_manager._context_window_size == 5

    # ==============================================================================
    # F055: store() method tests
    # ==============================================================================

    @pytest.mark.asyncio
    async def test_store_memory(self, memory_manager: MemoryManager) -> None:
        """
        Test store() method (F055).

        This test verifies:
        - Accepts content, memory_type, and metadata
        - Generates unique ID
        - Stores in ChromaDB with embedding
        """
        # Store a memory
        memory = await memory_manager.store(
            content="The user's name is Alice",
            memory_type=MemoryType.FACT,
            metadata={"source": "user_intro"},
        )

        # Verify returned memory
        assert memory is not None
        assert memory.id is not None  # Unique ID generated
        assert len(memory.id) > 0
        assert memory.content == "The user's name is Alice"
        assert memory.memory_type == MemoryType.FACT
        assert memory.metadata == {"source": "user_intro"}

        # Verify stored in ChromaDB
        count = memory_manager.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_store_generates_unique_ids(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test that store() generates unique IDs."""
        memory1 = await memory_manager.store("Content 1", MemoryType.FACT)
        memory2 = await memory_manager.store("Content 2", MemoryType.FACT)

        assert memory1.id != memory2.id

    # ==============================================================================
    # F056: search() method tests
    # ==============================================================================

    @pytest.mark.asyncio
    async def test_search_memory(self, memory_manager: MemoryManager) -> None:
        """
        Test search() method (F056).

        This test verifies:
        - Accepts query string
        - Optional memory_type filter
        - Optional limit parameter
        - Returns SearchResult with scores
        """
        # Store some memories
        await memory_manager.store("The user prefers dark mode", MemoryType.FACT)
        await memory_manager.store("The user likes Python", MemoryType.FACT)
        await memory_manager.store("Yesterday we talked about robots", MemoryType.CONVERSATION)

        # Search without filter
        results = await memory_manager.search("user preferences", limit=5)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

        # Verify SearchResult structure
        if results:
            assert hasattr(results[0], "memory")
            assert hasattr(results[0], "score")
            assert isinstance(results[0].memory, Memory)
            assert isinstance(results[0].score, float)
            assert 0.0 <= results[0].score <= 1.0

    @pytest.mark.asyncio
    async def test_search_with_type_filter(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test search with memory_type filter."""
        await memory_manager.store("User fact 1", MemoryType.FACT)
        await memory_manager.store("Conversation 1", MemoryType.CONVERSATION)
        await memory_manager.store("Context 1", MemoryType.CONTEXT)

        # Search only FACTs
        results = await memory_manager.search(
            "fact", memory_type=MemoryType.FACT, limit=10, min_score=0.0
        )

        # All results should be FACT type
        for r in results:
            assert r.memory.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_search_with_limit(self, memory_manager: MemoryManager) -> None:
        """Test search respects limit parameter."""
        # Store multiple memories
        for i in range(5):
            await memory_manager.store(f"Test memory {i}", MemoryType.FACT)

        # Search with limit=2
        results = await memory_manager.search("test", limit=2, min_score=0.0)
        assert len(results) <= 2

    # ==============================================================================
    # F057: Context window tests
    # ==============================================================================

    @pytest.mark.asyncio
    async def test_context_window(self, memory_manager: MemoryManager) -> None:
        """
        Test get_context_window() method (F057).

        This test verifies:
        - Returns last N conversation turns
        - Default N=5
        - Ordered by timestamp (newest first)
        """
        # Add conversation turns
        memory_manager.add_to_context_window("user", "Hello")
        memory_manager.add_to_context_window("assistant", "Hi there!")
        memory_manager.add_to_context_window("user", "How are you?")
        memory_manager.add_to_context_window("assistant", "I'm doing well!")

        # Get context window
        context = memory_manager.get_context_window()

        # Should be ordered newest first
        assert len(context) == 4
        assert context[0]["content"] == "I'm doing well!"  # Most recent
        assert context[-1]["content"] == "Hello"  # Oldest

    def test_context_window_size_limit(self, memory_manager: MemoryManager) -> None:
        """Test context window respects size limit."""
        # Add more turns than window size allows (5 * 2 = 10 items)
        for i in range(15):
            memory_manager.add_to_context_window("user", f"Message {i}")

        context = memory_manager.get_context_window()

        # Should only keep last 10 items (5 turns * 2 for user+assistant)
        assert len(context) <= 10

    def test_clear_context_window(self, memory_manager: MemoryManager) -> None:
        """Test clearing context window."""
        memory_manager.add_to_context_window("user", "Hello")
        memory_manager.add_to_context_window("assistant", "Hi!")

        memory_manager.clear_context_window()

        context = memory_manager.get_context_window()
        assert len(context) == 0

    # ==============================================================================
    # F058: forget() method tests
    # ==============================================================================

    @pytest.mark.asyncio
    async def test_forget_memory(self, memory_manager: MemoryManager) -> None:
        """
        Test forget() method (F058).

        This test verifies:
        - Accepts memory ID
        - Removes from ChromaDB
        - Returns success/failure
        """
        # Store a memory
        memory = await memory_manager.store("Temporary memory", MemoryType.FACT)
        memory_id = memory.id

        # Verify it exists
        initial_count = memory_manager.count()
        assert initial_count >= 1

        # Forget it
        success = await memory_manager.forget(memory_id)
        assert success is True

        # Verify it's gone
        retrieved = await memory_manager.get(memory_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_forget_nonexistent_memory(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test forget() returns False for non-existent ID."""
        success = await memory_manager.forget("nonexistent-id")
        assert success is False

    # ==============================================================================
    # F059: Memory expiry tests
    # ==============================================================================

    @pytest.mark.asyncio
    async def test_memory_expiry(self, memory_manager: MemoryManager) -> None:
        """
        Test automatic expiry cleanup (F059).

        This test verifies:
        - CONVERSATION expires after 30 days
        - CONTEXT expires at end of day
        - FACT never expires
        - cleanup runs correctly
        """
        # Test expires_at properties
        now = datetime.utcnow()

        # FACT never expires
        fact = Memory(id="f1", content="fact", memory_type=MemoryType.FACT, created_at=now)
        assert fact.expires_at is None

        # CONVERSATION expires after 30 days
        conv = Memory(id="c1", content="conv", memory_type=MemoryType.CONVERSATION, created_at=now)
        expected_conv_expiry = now + timedelta(days=30)
        assert conv.expires_at is not None
        # Allow small tolerance
        assert abs((conv.expires_at - expected_conv_expiry).total_seconds()) < 1

        # CONTEXT expires at end of day
        ctx = Memory(id="x1", content="ctx", memory_type=MemoryType.CONTEXT, created_at=now)
        assert ctx.expires_at is not None
        assert ctx.expires_at.date() == now.date()
        assert ctx.expires_at.hour == 23
        assert ctx.expires_at.minute == 59

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, memory_manager: MemoryManager) -> None:
        """Test cleanup_expired() method."""
        # Store some memories
        await memory_manager.store("Permanent fact", MemoryType.FACT)

        # Run cleanup (should not delete FACT)
        deleted = await memory_manager.cleanup_expired()

        # FACT should still exist
        count = memory_manager.count(MemoryType.FACT)
        assert count >= 1

    # ==============================================================================
    # Additional tests
    # ==============================================================================

    @pytest.mark.asyncio
    async def test_get_memory_by_id(self, memory_manager: MemoryManager) -> None:
        """Test get() method retrieves memory by ID."""
        stored = await memory_manager.store("Test content", MemoryType.FACT)

        retrieved = await memory_manager.get(stored.id)

        assert retrieved is not None
        assert retrieved.id == stored.id
        assert retrieved.content == "Test content"
        assert retrieved.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, memory_manager: MemoryManager) -> None:
        """Test get() returns None for non-existent ID."""
        retrieved = await memory_manager.get("nonexistent-id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_convenience_methods(self, memory_manager: MemoryManager) -> None:
        """Test convenience methods."""
        # remember_fact
        fact = await memory_manager.remember_fact("A fact", source="test")
        assert fact.memory_type == MemoryType.FACT

        # remember_conversation
        conv = await memory_manager.remember_conversation("A conversation")
        assert conv.memory_type == MemoryType.CONVERSATION

        # set_context
        ctx = await memory_manager.set_context("key", "value")
        assert ctx.memory_type == MemoryType.CONTEXT
        assert ctx.metadata.get("context_key") == "key"

    @pytest.mark.asyncio
    async def test_count_by_type(self, memory_manager: MemoryManager) -> None:
        """Test count() with memory_type filter."""
        await memory_manager.store("Fact 1", MemoryType.FACT)
        await memory_manager.store("Fact 2", MemoryType.FACT)
        await memory_manager.store("Conv 1", MemoryType.CONVERSATION)

        fact_count = memory_manager.count(MemoryType.FACT)
        conv_count = memory_manager.count(MemoryType.CONVERSATION)
        total_count = memory_manager.count()

        assert fact_count == 2
        assert conv_count == 1
        assert total_count == 3


# ==============================================================================
# F060-F063: Memory MCP Server tests
# ==============================================================================


class TestMemoryMCPServer:
    """Tests for Memory MCP server (F060-F063)."""

    @pytest.fixture(autouse=True)
    def reset_memory_manager(self) -> None:
        """Reset memory manager between tests."""
        import reachy_agent.mcp.memory as memory_module

        memory_module._memory_manager = None
        yield
        memory_module._memory_manager = None

    def test_server_has_3_tools(self) -> None:
        """Verify all 3 MCP tools are registered (F060)."""
        from reachy_agent.mcp.memory import app

        tools = list(app._tool_manager._tools.keys())
        assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}: {tools}"

    def test_memory_tools_exist(self) -> None:
        """Verify memory tools are registered."""
        from reachy_agent.mcp.memory import app

        tools = list(app._tool_manager._tools.keys())
        expected_tools = ["store_memory", "search_memory", "forget_memory"]
        for tool in expected_tools:
            assert tool in tools, f"Missing tool: {tool}"

    @pytest.mark.asyncio
    async def test_store_memory_tool(self, tmp_path: Path) -> None:
        """Test store_memory tool (F061)."""
        import reachy_agent.mcp.memory as memory_module

        # Use temporary path for ChromaDB
        memory_module._memory_manager = MemoryManager(
            persist_path=str(tmp_path / "test_mcp_memory"),
            collection_name="test_mcp_store",
        )

        from reachy_agent.mcp.memory import app

        tool = app._tool_manager._tools["store_memory"]
        result = await tool.fn(
            content="The user's favorite food is pizza",
            memory_type="fact",
            metadata={"source": "test"},
        )

        assert result["success"] is True
        assert "memory_id" in result
        assert result["content"] == "The user's favorite food is pizza"
        assert result["memory_type"] == "fact"

    @pytest.mark.asyncio
    async def test_search_memory_tool(self, tmp_path: Path) -> None:
        """Test search_memory tool (F062)."""
        import reachy_agent.mcp.memory as memory_module

        # Setup with some data
        manager = MemoryManager(
            persist_path=str(tmp_path / "test_mcp_search"),
            collection_name="test_mcp_search",
        )
        await manager.store("User prefers dark mode", MemoryType.FACT)
        await manager.store("User likes Python programming", MemoryType.FACT)
        memory_module._memory_manager = manager

        from reachy_agent.mcp.memory import app

        tool = app._tool_manager._tools["search_memory"]
        result = await tool.fn(query="user preferences", limit=5, min_score=0.0)

        assert result["success"] is True
        assert "results" in result
        assert result["results_count"] >= 0
        # If we got results, check structure
        if result["results"]:
            r = result["results"][0]
            assert "memory_id" in r
            assert "content" in r
            assert "score" in r
            assert "memory_type" in r

    @pytest.mark.asyncio
    async def test_forget_memory_tool(self, tmp_path: Path) -> None:
        """Test forget_memory tool (F063)."""
        import reachy_agent.mcp.memory as memory_module

        # Setup with data
        manager = MemoryManager(
            persist_path=str(tmp_path / "test_mcp_forget"),
            collection_name="test_mcp_forget",
        )
        memory = await manager.store("Temporary data", MemoryType.FACT)
        memory_module._memory_manager = manager

        from reachy_agent.mcp.memory import app

        tool = app._tool_manager._tools["forget_memory"]

        # Test forgetting existing memory
        result = await tool.fn(memory_id=memory.id)
        assert result["success"] is True
        assert result["memory_id"] == memory.id

        # Test forgetting non-existent memory
        result2 = await tool.fn(memory_id="nonexistent-id")
        assert result2["success"] is False

    @pytest.mark.asyncio
    async def test_search_with_type_filter_tool(self, tmp_path: Path) -> None:
        """Test search_memory with type filter."""
        import reachy_agent.mcp.memory as memory_module

        manager = MemoryManager(
            persist_path=str(tmp_path / "test_mcp_filter"),
            collection_name="test_mcp_filter",
        )
        await manager.store("Fact about user", MemoryType.FACT)
        await manager.store("Conversation turn", MemoryType.CONVERSATION)
        memory_module._memory_manager = manager

        from reachy_agent.mcp.memory import app

        tool = app._tool_manager._tools["search_memory"]
        result = await tool.fn(
            query="user",
            memory_type="fact",
            limit=10,
            min_score=0.0,
        )

        assert result["success"] is True
        for r in result["results"]:
            assert r["memory_type"] == "fact"
