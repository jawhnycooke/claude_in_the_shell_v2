"""Tests for robot client."""

import pytest

from reachy_agent.robot.client import AntennaState, HeadPose, RobotStatus
from reachy_agent.robot.mock import MockClient
from reachy_agent.robot.sdk import (
    CachedResult,
    MotorError,
    NotAwakeError,
    RobotConnectionError,
    SDKClient,
    ToolCache,
)


@pytest.mark.asyncio
async def test_mock_client() -> None:
    """
    Comprehensive test for MockClient implementing ReachyClient protocol.

    This test verifies:
    - MockClient implements ReachyClient protocol
    - Simulates motor movements with delays
    - Returns realistic sensor data
    """
    client = MockClient()

    # Test 1: Connection
    await client.connect()

    # Test 2: Wake up and check state
    await client.wake_up()
    assert await client.is_awake() is True

    # Test 3: Movement simulation with realistic delays
    await client.move_head(pitch=10.0, yaw=20.0, roll=5.0, duration=0.1)
    position = await client.get_position()
    assert position["head_pitch"] == 10.0
    assert position["head_yaw"] == 20.0
    assert position["head_roll"] == 5.0

    # Test 4: Returns realistic sensor data
    sensor_data = await client.get_sensor_data()
    assert "accel_x" in sensor_data
    assert "accel_y" in sensor_data
    assert "accel_z" in sensor_data
    # Gravity is approximately 9.8 m/s^2
    assert 9.0 < sensor_data["accel_z"] < 10.5

    # Test 5: Sleep and disconnect
    await client.sleep()
    assert await client.is_awake() is False
    await client.disconnect()


@pytest.fixture
def mock_robot() -> MockClient:
    """Create a fresh mock robot for each test."""
    return MockClient()


class TestMockClient:
    """Tests for MockClient."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, mock_robot: MockClient) -> None:
        """Test mock robot connection and disconnection."""
        await mock_robot.connect()
        assert mock_robot._connected is True

        await mock_robot.disconnect()
        assert mock_robot._connected is False

    @pytest.mark.asyncio
    async def test_wake_up_sleep(self, mock_robot: MockClient) -> None:
        """Test wake up and sleep."""
        await mock_robot.wake_up()
        assert await mock_robot.is_awake() is True

        await mock_robot.sleep()
        assert await mock_robot.is_awake() is False

    @pytest.mark.asyncio
    async def test_move_head(self, mock_robot: MockClient) -> None:
        """Test head movement with new API."""
        await mock_robot.wake_up()
        await mock_robot.move_head(pitch=10.0, yaw=20.0, roll=5.0, duration=0.1)

        position = await mock_robot.get_position()
        assert position["head_pitch"] == 10.0
        assert position["head_yaw"] == 20.0
        assert position["head_roll"] == 5.0

    @pytest.mark.asyncio
    async def test_move_head_clamps_limits(self, mock_robot: MockClient) -> None:
        """Test head movement respects joint limits."""
        await mock_robot.wake_up()
        # Try to move beyond limits
        await mock_robot.move_head(pitch=100.0, yaw=200.0, roll=-100.0, duration=0.1)

        position = await mock_robot.get_position()
        # Should be clamped to max limits
        assert position["head_pitch"] == 35.0  # max pitch
        assert position["head_yaw"] == 60.0  # max yaw
        assert position["head_roll"] == -35.0  # min roll

    @pytest.mark.asyncio
    async def test_move_head_requires_awake(self, mock_robot: MockClient) -> None:
        """Test head movement fails when robot is asleep."""
        # Robot starts asleep
        with pytest.raises(RuntimeError, match="Robot is asleep"):
            await mock_robot.move_head(pitch=10.0, yaw=0.0, roll=0.0, duration=0.1)

    @pytest.mark.asyncio
    async def test_look_at(self, mock_robot: MockClient) -> None:
        """Test look_at with 3D coordinates."""
        await mock_robot.wake_up()
        # Look at point to the right and slightly down
        await mock_robot.look_at(x=1.0, y=0.5, z=1.0, duration=0.1)

        position = await mock_robot.get_position()
        # Should have turned yaw right and pitch down
        assert position["head_yaw"] > 0  # Turned right
        assert position["head_pitch"] < 0  # Tilted down

    @pytest.mark.asyncio
    async def test_rotate_body(self, mock_robot: MockClient) -> None:
        """Test body rotation."""
        await mock_robot.wake_up()
        await mock_robot.rotate_body(angle=90.0, duration=0.1)

        position = await mock_robot.get_position()
        assert position["body_rotation"] == 90.0

    @pytest.mark.asyncio
    async def test_rotate_body_wraps_360(self, mock_robot: MockClient) -> None:
        """Test body rotation wraps at 360 degrees."""
        await mock_robot.wake_up()
        await mock_robot.rotate_body(angle=450.0, duration=0.1)

        position = await mock_robot.get_position()
        assert position["body_rotation"] == 90.0  # 450 % 360

    @pytest.mark.asyncio
    async def test_reset_position(self, mock_robot: MockClient) -> None:
        """Test reset_position returns to neutral."""
        await mock_robot.wake_up()
        await mock_robot.move_head(pitch=20.0, yaw=30.0, roll=10.0, duration=0.1)
        await mock_robot.rotate_body(angle=90.0, duration=0.1)

        await mock_robot.reset_position(duration=0.1)

        position = await mock_robot.get_position()
        assert position["head_pitch"] == 0.0
        assert position["head_yaw"] == 0.0
        assert position["head_roll"] == 0.0
        assert position["body_rotation"] == 0.0

    @pytest.mark.asyncio
    async def test_set_antennas(self, mock_robot: MockClient) -> None:
        """Test antenna positioning."""
        await mock_robot.set_antennas(left=45.0, right=-30.0)

        position = await mock_robot.get_position()
        assert position["antenna_left"] == 45.0
        assert position["antenna_right"] == -30.0

    @pytest.mark.asyncio
    async def test_set_antennas_clamps(self, mock_robot: MockClient) -> None:
        """Test antenna positioning respects limits."""
        await mock_robot.set_antennas(left=200.0, right=-200.0)

        position = await mock_robot.get_position()
        assert position["antenna_left"] == 150.0  # max
        assert position["antenna_right"] == -150.0  # min

    @pytest.mark.asyncio
    async def test_nod(self, mock_robot: MockClient) -> None:
        """Test nod gesture."""
        await mock_robot.wake_up()
        # Should not raise
        await mock_robot.nod(intensity=1.0)

    @pytest.mark.asyncio
    async def test_shake(self, mock_robot: MockClient) -> None:
        """Test shake gesture."""
        await mock_robot.wake_up()
        # Should not raise
        await mock_robot.shake(intensity=1.0)

    @pytest.mark.asyncio
    async def test_speak(self, mock_robot: MockClient) -> None:
        """Test speak simulation."""
        # Should not raise
        await mock_robot.speak("Hello world!", voice="default")

    @pytest.mark.asyncio
    async def test_listen(self, mock_robot: MockClient) -> None:
        """Test listen simulation."""
        result = await mock_robot.listen(timeout=0.1)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_capture_image(self, mock_robot: MockClient) -> None:
        """Test image capture returns bytes."""
        data = await mock_robot.capture_image()
        assert isinstance(data, bytes)
        # Should start with PNG magic bytes
        assert data[:4] == bytes([0x89, 0x50, 0x4E, 0x47])

    @pytest.mark.asyncio
    async def test_get_sensor_data(self, mock_robot: MockClient) -> None:
        """Test sensor data returns dict with expected keys."""
        data = await mock_robot.get_sensor_data()
        assert isinstance(data, dict)
        assert "accel_x" in data
        assert "accel_y" in data
        assert "accel_z" in data
        # Gravity should be around 9.8
        assert 9.0 < data["accel_z"] < 10.5

    @pytest.mark.asyncio
    async def test_detect_sound_direction(self, mock_robot: MockClient) -> None:
        """Test sound direction detection."""
        direction, confidence = await mock_robot.detect_sound_direction()
        assert isinstance(direction, float)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_get_status(self, mock_robot: MockClient) -> None:
        """Test status retrieval."""
        await mock_robot.wake_up()
        status = await mock_robot.get_status()

        assert isinstance(status, RobotStatus)
        assert status.is_awake is True
        assert 0.0 <= status.battery_percent <= 100.0
        assert isinstance(status.head_pose, HeadPose)
        assert isinstance(status.antenna_state, AntennaState)

    @pytest.mark.asyncio
    async def test_get_limits(self, mock_robot: MockClient) -> None:
        """Test joint limits retrieval."""
        limits = await mock_robot.get_limits()

        assert isinstance(limits, dict)
        assert "head_pitch" in limits
        assert "head_yaw" in limits
        assert limits["head_pitch"] == (-45.0, 35.0)
        assert limits["head_yaw"] == (-60.0, 60.0)

    @pytest.mark.asyncio
    async def test_play_emotion(self, mock_robot: MockClient) -> None:
        """Test emotion playback."""
        # Should not raise
        await mock_robot.play_emotion("happy")


# ==============================================================================
# SDKClient Tests (with mock flag - testing without hardware)
# ==============================================================================


@pytest.mark.asyncio
async def test_sdk_client() -> None:
    """
    Comprehensive test for SDKClient implementing ReachyClient protocol.

    This test runs in mock mode (--mock) without hardware and verifies:
    - SDKClient implements ReachyClient protocol
    - Connects to Reachy SDK via Zenoh (or raises clear error if SDK unavailable)
    - Handles connection errors gracefully

    Note: This test expects the Reachy SDK to NOT be installed,
    and verifies that the client handles this gracefully.
    """
    client = SDKClient(connect_timeout=1.0)

    # Test 1: Verify initialization
    assert client._robot is None
    assert client._connected is False
    assert client._awake is False

    # Test 2: Test connection raises RobotConnectionError without SDK
    # (Since reachy_mini is likely not installed in test environment)
    try:
        await client.connect()
        # If we get here, SDK is actually installed - that's fine
        client._connected = True
    except RobotConnectionError as e:
        # This is expected - verify error message is helpful
        assert "SDK" in str(e) or "reachy" in str(e).lower() or "mock" in str(e).lower()

    # Test 3: Test graceful error handling for operations without connection
    if not client._connected:
        with pytest.raises(RobotConnectionError):
            await client.wake_up()

        with pytest.raises(RobotConnectionError):
            await client.move_head(pitch=10, yaw=0, roll=0, duration=0.5)

    # Test 4: Test get_limits works without connection
    limits = await client.get_limits()
    assert isinstance(limits, dict)
    assert "head_pitch" in limits
    assert "head_yaw" in limits
    assert limits["head_pitch"] == (-45.0, 35.0)

    # Test 5: Test is_awake returns correct state
    assert await client.is_awake() is False

    # Test 6: Test disconnect is safe when not connected
    await client.disconnect()
    assert client._connected is False


# Re-import for compatibility with existing tests


class TestToolCache:
    """Tests for the ToolCache class."""

    def test_cache_init(self) -> None:
        """Test cache initialization."""
        cache = ToolCache()
        assert cache._cache == {}

    def test_cache_set_and_get(self) -> None:
        """Test setting and getting cached values."""
        cache = ToolCache()
        cache.set("test_key", "test_value", ttl=1.0)

        result = cache.get("test_key")
        assert result == "test_value"

    def test_cache_get_missing(self) -> None:
        """Test getting non-existent key returns None."""
        cache = ToolCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_invalidate_all(self) -> None:
        """Test invalidating all cache entries."""
        cache = ToolCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.invalidate("*")

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_invalidate_pattern(self) -> None:
        """Test invalidating cache entries by pattern."""
        cache = ToolCache()
        cache.set("pose_head", "head_value")
        cache.set("pose_body", "body_value")
        cache.set("status", "status_value")

        cache.invalidate("pose*")

        assert cache.get("pose_head") is None
        assert cache.get("pose_body") is None
        assert cache.get("status") == "status_value"


@pytest.fixture
def sdk_client() -> SDKClient:
    """Create a fresh SDK client for each test."""
    return SDKClient(connect_timeout=1.0)


class TestSDKClient:
    """
    Tests for SDKClient.

    These tests verify the SDKClient implementation works correctly
    in mock mode (without actual hardware). The SDK will raise
    RobotConnectionError when reachy_mini package is not available.
    """

    @pytest.mark.asyncio
    async def test_sdk_client_connection_error_without_sdk(
        self, sdk_client: SDKClient
    ) -> None:
        """
        Test that SDKClient raises RobotConnectionError when SDK not available.

        This verifies the connection error handling is graceful.
        """
        with pytest.raises(RobotConnectionError) as exc_info:
            await sdk_client.connect()

        # Should have a helpful error message
        assert "SDK" in str(exc_info.value) or "reachy" in str(exc_info.value).lower()

    def test_sdk_client_init(self, sdk_client: SDKClient) -> None:
        """Test SDK client initialization."""
        assert sdk_client._robot is None
        assert sdk_client._connected is False
        assert sdk_client._awake is False
        assert sdk_client._connect_timeout == 1.0
        assert isinstance(sdk_client._cache, ToolCache)

    def test_sdk_client_joint_limits(self, sdk_client: SDKClient) -> None:
        """Test SDK client has correct joint limits."""
        assert sdk_client.HEAD_PITCH_LIMITS == (-45.0, 35.0)
        assert sdk_client.HEAD_YAW_LIMITS == (-60.0, 60.0)
        assert sdk_client.HEAD_ROLL_LIMITS == (-35.0, 35.0)
        assert sdk_client.HEAD_Z_LIMITS == (0.0, 50.0)
        assert sdk_client.ANTENNA_LIMITS == (-150.0, 150.0)

    def test_sdk_client_clamp(self, sdk_client: SDKClient) -> None:
        """Test clamping helper function."""
        assert sdk_client._clamp(50.0, (-10.0, 10.0)) == 10.0
        assert sdk_client._clamp(-50.0, (-10.0, 10.0)) == -10.0
        assert sdk_client._clamp(5.0, (-10.0, 10.0)) == 5.0

    @pytest.mark.asyncio
    async def test_sdk_client_wake_up_requires_connection(
        self, sdk_client: SDKClient
    ) -> None:
        """Test wake_up requires connection."""
        with pytest.raises(RobotConnectionError, match="Not connected"):
            await sdk_client.wake_up()

    @pytest.mark.asyncio
    async def test_sdk_client_move_head_requires_connection(
        self, sdk_client: SDKClient
    ) -> None:
        """Test move_head requires connection."""
        with pytest.raises(RobotConnectionError, match="Not connected"):
            await sdk_client.move_head(pitch=0, yaw=0, roll=0, duration=1.0)

    @pytest.mark.asyncio
    async def test_sdk_client_disconnect_safe_when_not_connected(
        self, sdk_client: SDKClient
    ) -> None:
        """Test disconnect is safe when not connected."""
        # Should not raise
        await sdk_client.disconnect()
        assert sdk_client._connected is False

    @pytest.mark.asyncio
    async def test_sdk_client_get_limits(self, sdk_client: SDKClient) -> None:
        """Test get_limits returns correct values."""
        limits = await sdk_client.get_limits()

        assert isinstance(limits, dict)
        assert limits["head_pitch"] == (-45.0, 35.0)
        assert limits["head_yaw"] == (-60.0, 60.0)
        assert limits["head_roll"] == (-35.0, 35.0)
        assert limits["head_z"] == (0.0, 50.0)
        assert limits["body_rotation"] == (0.0, 360.0)

    @pytest.mark.asyncio
    async def test_sdk_client_is_awake(self, sdk_client: SDKClient) -> None:
        """Test is_awake returns correct state."""
        assert await sdk_client.is_awake() is False

        # Manually set awake state for testing
        sdk_client._awake = True
        assert await sdk_client.is_awake() is True

    @pytest.mark.asyncio
    async def test_sdk_client_speak_does_not_raise(self, sdk_client: SDKClient) -> None:
        """Test speak does not raise (placeholder implementation)."""
        # Should not raise
        await sdk_client.speak("Hello world", voice="default")

    @pytest.mark.asyncio
    async def test_sdk_client_listen_returns_empty(self, sdk_client: SDKClient) -> None:
        """Test listen returns empty string (placeholder implementation)."""
        result = await sdk_client.listen(timeout=1.0)
        assert result == ""

    @pytest.mark.asyncio
    async def test_sdk_client_get_sensor_data_default(
        self, sdk_client: SDKClient
    ) -> None:
        """Test get_sensor_data returns default values when not connected."""
        data = await sdk_client.get_sensor_data()

        assert isinstance(data, dict)
        assert "accel_x" in data
        assert "accel_y" in data
        assert "accel_z" in data
        assert data["accel_z"] == 9.8  # Default gravity value

    @pytest.mark.asyncio
    async def test_sdk_client_detect_sound_direction_default(
        self, sdk_client: SDKClient
    ) -> None:
        """Test detect_sound_direction returns default when not connected."""
        direction, confidence = await sdk_client.detect_sound_direction()

        assert direction == 0.0
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_sdk_client_get_status_without_connection(
        self, sdk_client: SDKClient
    ) -> None:
        """Test get_status returns basic status when not connected."""
        status = await sdk_client.get_status()

        assert isinstance(status, RobotStatus)
        assert status.is_awake is False
        assert status.battery_percent == 100.0

    @pytest.mark.asyncio
    async def test_sdk_client_get_position_without_connection(
        self, sdk_client: SDKClient
    ) -> None:
        """Test get_position returns default positions when not connected."""
        position = await sdk_client.get_position()

        assert isinstance(position, dict)
        assert position["head_pitch"] == 0.0
        assert position["head_yaw"] == 0.0
        assert position["head_roll"] == 0.0
        assert position["body_rotation"] == 0.0


class TestSDKClientExceptions:
    """Tests for SDKClient exception classes."""

    def test_robot_connection_error(self) -> None:
        """Test RobotConnectionError exception."""
        error = RobotConnectionError("Test connection error")
        assert str(error) == "Test connection error"

    def test_not_awake_error(self) -> None:
        """Test NotAwakeError exception."""
        error = NotAwakeError("Robot is asleep")
        assert str(error) == "Robot is asleep"

    def test_motor_error(self) -> None:
        """Test MotorError exception."""
        error = MotorError("Motor stalled")
        assert str(error) == "Motor stalled"


class TestCachedResult:
    """Tests for CachedResult dataclass."""

    def test_cached_result_default_ttl(self) -> None:
        """Test CachedResult has default TTL of 200ms."""
        import time

        result = CachedResult(value="test", timestamp=time.time())
        assert result.ttl == 0.2

    def test_cached_result_custom_ttl(self) -> None:
        """Test CachedResult with custom TTL."""
        import time

        result = CachedResult(value="test", timestamp=time.time(), ttl=1.0)
        assert result.ttl == 1.0
