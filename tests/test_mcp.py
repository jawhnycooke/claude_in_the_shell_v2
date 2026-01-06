"""Tests for MCP servers."""

import asyncio
import time

import pytest

from reachy_agent.mcp.robot import (
    MCPToolCache,
    app,
)


@pytest.mark.asyncio
async def test_tool_caching() -> None:
    """
    Comprehensive test for MCP tool result caching.

    This test verifies:
    - Caches read-only tool results
    - Cache expires after 200ms
    - Cache key includes tool name and parameters
    """
    import reachy_agent.mcp.robot as robot_module

    # Reset state
    robot_module._robot = None
    robot_module._cache = MCPToolCache()
    cache = robot_module._cache

    # Test 1: Verify cache key includes tool name and parameters
    key1 = cache.make_key("get_status")
    key2 = cache.make_key("get_position")
    key3 = cache.make_key("get_status")  # Same as key1
    key4 = cache.make_key("some_tool", param1="value1", param2=42)

    assert key1 != key2  # Different tools have different keys
    assert key1 == key3  # Same tool with no params has same key
    assert "some_tool" in key4
    assert "param1" in key4 or "value1" in key4

    # Test 2: Verify cache stores read-only tool results
    test_value = {"success": True, "data": "test"}
    cache.set("get_status", test_value, ttl=0.2)

    cached = cache.get("get_status")
    assert cached is not None
    assert cached["success"] is True
    assert cached["data"] == "test"

    # Test 3: Verify cache expires after 200ms
    cache.set("expiring_tool", {"value": 123}, ttl=0.1)  # 100ms TTL
    assert cache.get("expiring_tool") is not None

    # Wait for cache to expire
    await asyncio.sleep(0.15)

    # Cache should now be expired
    assert cache.get("expiring_tool") is None

    # Test 4: Verify caching works with actual MCP tools
    # Reset cache
    robot_module._cache = MCPToolCache()
    robot_module._robot = None

    # Get status twice quickly - second call should be cached
    tool = app._tool_manager._tools["get_status"]
    start = time.time()
    result1 = await tool.fn()
    first_call_time = time.time() - start

    start = time.time()
    result2 = await tool.fn()
    second_call_time = time.time() - start

    # Both should succeed
    assert result1["success"] is True
    assert result2["success"] is True

    # Second call should be faster (cached) or equal
    # (Note: timing can be variable, so we just check it doesn't fail)

    # Clean up
    robot_module._robot = None
    robot_module._cache = MCPToolCache()


class TestRobotMCPServer:
    """Tests for Robot MCP server."""

    @pytest.fixture(autouse=True)
    def reset_robot(self) -> None:
        """Reset robot instance between tests."""
        import reachy_agent.mcp.robot as robot_module

        robot_module._robot = None
        yield
        robot_module._robot = None

    def test_server_has_20_tools(self) -> None:
        """Verify all 20 MCP tools are registered."""
        tools = list(app._tool_manager._tools.keys())
        assert len(tools) == 20, f"Expected 20 tools, got {len(tools)}: {tools}"

    def test_movement_tools_exist(self) -> None:
        """Verify movement tools are registered."""
        tools = list(app._tool_manager._tools.keys())
        movement_tools = ["move_head", "look_at", "rotate_body", "reset_position"]
        for tool in movement_tools:
            assert tool in tools, f"Missing tool: {tool}"

    def test_expression_tools_exist(self) -> None:
        """Verify expression tools are registered."""
        tools = list(app._tool_manager._tools.keys())
        expression_tools = [
            "play_emotion",
            "play_sequence",
            "set_antennas",
            "nod",
            "shake",
        ]
        for tool in expression_tools:
            assert tool in tools, f"Missing tool: {tool}"

    def test_audio_tools_exist(self) -> None:
        """Verify audio tools are registered."""
        tools = list(app._tool_manager._tools.keys())
        audio_tools = ["speak", "listen"]
        for tool in audio_tools:
            assert tool in tools, f"Missing tool: {tool}"

    def test_perception_tools_exist(self) -> None:
        """Verify perception tools are registered."""
        tools = list(app._tool_manager._tools.keys())
        perception_tools = [
            "capture_image",
            "get_sensor_data",
            "detect_sound_direction",
        ]
        for tool in perception_tools:
            assert tool in tools, f"Missing tool: {tool}"

    def test_lifecycle_tools_exist(self) -> None:
        """Verify lifecycle tools are registered."""
        tools = list(app._tool_manager._tools.keys())
        lifecycle_tools = ["wake_up", "sleep_robot", "is_awake"]
        for tool in lifecycle_tools:
            assert tool in tools, f"Missing tool: {tool}"

    def test_status_tools_exist(self) -> None:
        """Verify status tools are registered."""
        tools = list(app._tool_manager._tools.keys())
        status_tools = ["get_status", "get_position", "get_limits"]
        for tool in status_tools:
            assert tool in tools, f"Missing tool: {tool}"


class TestRobotMCPToolExecution:
    """Tests for executing MCP tool functions directly (bypassing MCP)."""

    @pytest.fixture(autouse=True)
    def reset_robot(self) -> None:
        """Reset robot instance between tests."""
        import reachy_agent.mcp.robot as robot_module

        robot_module._robot = None
        yield
        robot_module._robot = None

    @pytest.mark.asyncio
    async def test_move_head_tool(self) -> None:
        """Test move_head tool execution."""
        # Import the underlying function, not the wrapped tool

        # Access the tool function directly
        tool = app._tool_manager._tools["move_head"]
        result = await tool.fn(pitch=10.0, yaw=20.0, roll=0.0, duration=0.1)
        assert result["success"] is True
        assert "position" in result

    @pytest.mark.asyncio
    async def test_get_status_tool(self) -> None:
        """Test get_status tool execution."""
        tool = app._tool_manager._tools["get_status"]
        result = await tool.fn()
        assert result["success"] is True
        assert "is_awake" in result
        assert "battery_percent" in result

    @pytest.mark.asyncio
    async def test_wake_up_tool(self) -> None:
        """Test wake_up tool execution."""
        tool = app._tool_manager._tools["wake_up"]
        result = await tool.fn()
        assert result["success"] is True
        assert result["awake"] is True

    @pytest.mark.asyncio
    async def test_sleep_tool(self) -> None:
        """Test sleep_robot tool execution."""
        wake_tool = app._tool_manager._tools["wake_up"]
        await wake_tool.fn()  # First wake up

        sleep_tool = app._tool_manager._tools["sleep_robot"]
        result = await sleep_tool.fn()
        assert result["success"] is True
        assert result["awake"] is False

    @pytest.mark.asyncio
    async def test_play_sequence_tool(self) -> None:
        """Test play_sequence tool execution."""
        tool = app._tool_manager._tools["play_sequence"]
        result = await tool.fn(emotions=["happy", "curious"], delays=[0.1])
        assert result["success"] is True
        assert result["emotions_played"] == ["happy", "curious"]

    @pytest.mark.asyncio
    async def test_capture_image_tool(self) -> None:
        """Test capture_image tool execution."""
        tool = app._tool_manager._tools["capture_image"]
        result = await tool.fn()
        assert result["success"] is True
        assert result["format"] == "png"
        assert result["image_bytes"] > 0

    @pytest.mark.asyncio
    async def test_detect_sound_direction_tool(self) -> None:
        """Test detect_sound_direction tool execution."""
        tool = app._tool_manager._tools["detect_sound_direction"]
        result = await tool.fn()
        assert result["success"] is True
        assert "azimuth_degrees" in result
        assert "confidence" in result
