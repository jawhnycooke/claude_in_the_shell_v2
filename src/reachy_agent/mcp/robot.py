"""Robot MCP server with 20 tools for controlling Reachy Mini.

This server provides MCP tools for robot control, including movement,
expressions, audio, perception, and status queries.

Features:
    - 20 MCP tools for robot control
    - Tool result caching with 200ms TTL for read-only operations
    - Automatic connection management
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog
from fastmcp import FastMCP

from reachy_agent.robot.client import HeadPose, RobotStatus
from reachy_agent.robot.mock import MockClient


# ============================================================================
# Tool Result Caching
# ============================================================================


@dataclass
class CachedToolResult:
    """A cached MCP tool result with timestamp and TTL."""

    value: Any
    timestamp: float
    ttl: float = 0.2  # 200ms default


class MCPToolCache:
    """
    Cache for read-only MCP tool results.

    Caches results to avoid redundant robot queries when Claude
    calls status tools multiple times in one turn.

    The cache key is generated from the tool name and parameters.
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._cache: dict[str, CachedToolResult] = {}
        self._log = structlog.get_logger("mcp.cache")

    def make_key(self, tool_name: str, **params: Any) -> str:
        """
        Generate cache key from tool name and parameters.

        Args:
            tool_name: Name of the MCP tool
            **params: Tool parameters

        Returns:
            Unique cache key string
        """
        # Sort params for consistent key generation
        param_str = json.dumps(params, sort_keys=True, default=str)
        key = f"{tool_name}:{param_str}"
        return key

    def get(self, tool_name: str, **params: Any) -> Any | None:
        """
        Get cached result if not expired.

        Args:
            tool_name: Name of the MCP tool
            **params: Tool parameters

        Returns:
            Cached result if present and not expired, None otherwise
        """
        key = self.make_key(tool_name, **params)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry.timestamp < entry.ttl:
                self._log.debug("cache_hit", tool=tool_name, key=key)
                return entry.value
            # Expired, remove from cache
            del self._cache[key]
            self._log.debug("cache_expired", tool=tool_name, key=key)
        return None

    def set(
        self, tool_name: str, value: Any, ttl: float = 0.2, **params: Any
    ) -> None:
        """
        Store tool result in cache.

        Args:
            tool_name: Name of the MCP tool
            value: Result to cache
            ttl: Time-to-live in seconds (default 200ms)
            **params: Tool parameters
        """
        key = self.make_key(tool_name, **params)
        self._cache[key] = CachedToolResult(value, time.time(), ttl)
        self._log.debug("cache_set", tool=tool_name, key=key, ttl=ttl)

    def invalidate(self, pattern: str = "*") -> None:
        """
        Invalidate cache entries.

        Args:
            pattern: "*" clears all, "tool:*" clears specific tool
        """
        if pattern == "*":
            self._cache.clear()
            self._log.debug("cache_invalidated_all")
        else:
            prefix = pattern.rstrip("*")
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                del self._cache[k]
            self._log.debug("cache_invalidated_pattern", pattern=pattern, count=len(keys))


# Create FastMCP server
app = FastMCP("robot")
log = structlog.get_logger("mcp.robot")

# Robot client instance (default to mock for development)
_robot: Optional[MockClient] = None

# Tool result cache for read-only operations
_cache = MCPToolCache()


def get_robot() -> MockClient:
    """Get or create robot client instance."""
    global _robot
    if _robot is None:
        _robot = MockClient()
    return _robot


async def ensure_connected() -> MockClient:
    """Ensure robot is connected and return client."""
    robot = get_robot()
    if not robot._connected:
        await robot.connect()
    return robot


# === MOVEMENT TOOLS ===


@app.tool
async def move_head(
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    duration: float = 1.0,
) -> dict:
    """
    Move head to absolute position.

    Args:
        pitch: Head pitch in degrees (-45 to +35, down to up)
        yaw: Head yaw in degrees (-60 to +60, right to left)
        roll: Head roll in degrees (-35 to +35, tilt)
        duration: Movement duration in seconds

    Returns:
        dict: Success status with final position
    """
    # Invalidate position cache since we're moving
    _cache.invalidate("get_status*")
    _cache.invalidate("get_position*")

    robot = await ensure_connected()
    await robot.wake_up()
    await robot.move_head(pitch=pitch, yaw=yaw, roll=roll, duration=duration)
    position = await robot.get_position()
    return {"success": True, "position": position}


@app.tool
async def look_at(
    x: float,
    y: float,
    z: float,
    duration: float = 1.0,
) -> dict:
    """
    Look at a 3D point in robot frame.

    Args:
        x: Left/right distance in meters (positive = right)
        y: Up/down distance in meters (positive = down)
        z: Forward distance in meters (positive = forward)
        duration: Movement duration in seconds

    Returns:
        dict: Success status
    """
    # Invalidate position cache since we're moving
    _cache.invalidate("get_status*")
    _cache.invalidate("get_position*")

    robot = await ensure_connected()
    await robot.wake_up()
    await robot.look_at(x=x, y=y, z=z, duration=duration)
    return {"success": True}


@app.tool
async def rotate_body(
    angle: float,
    duration: float = 1.0,
) -> dict:
    """
    Rotate body base to specified angle.

    Args:
        angle: Target angle in degrees (0-360, continuous rotation)
        duration: Movement duration in seconds

    Returns:
        dict: Success status with final angle
    """
    # Invalidate position cache since we're moving
    _cache.invalidate("get_status*")
    _cache.invalidate("get_position*")

    robot = await ensure_connected()
    await robot.wake_up()
    await robot.rotate_body(angle=angle, duration=duration)
    position = await robot.get_position()
    return {"success": True, "body_rotation": position["body_rotation"]}


@app.tool
async def reset_position(duration: float = 1.0) -> dict:
    """
    Return to neutral pose (head centered, body at 0°).

    Args:
        duration: Movement duration in seconds

    Returns:
        dict: Success status
    """
    # Invalidate position cache since we're moving
    _cache.invalidate("get_status*")
    _cache.invalidate("get_position*")

    robot = await ensure_connected()
    await robot.wake_up()
    await robot.reset_position(duration=duration)
    return {"success": True}


# === EXPRESSION TOOLS ===


@app.tool
async def set_antennas(left: float, right: float) -> dict:
    """
    Set antenna positions for expressiveness.

    Args:
        left: Left antenna angle in degrees (-150 to +150)
        right: Right antenna angle in degrees (-150 to +150)

    Returns:
        dict: Success status with final positions
    """
    robot = await ensure_connected()
    await robot.set_antennas(left=left, right=right)
    position = await robot.get_position()
    return {
        "success": True,
        "left": position["antenna_left"],
        "right": position["antenna_right"],
    }


@app.tool
async def nod(intensity: float = 1.0) -> dict:
    """
    Perform nod gesture (affirmative/agreement).

    Args:
        intensity: Motion amplitude multiplier (0.0-2.0)

    Returns:
        dict: Success status
    """
    robot = await ensure_connected()
    await robot.wake_up()
    await robot.nod(intensity=intensity)
    return {"success": True}


@app.tool
async def shake(intensity: float = 1.0) -> dict:
    """
    Perform head shake gesture (negative/disagreement).

    Args:
        intensity: Motion amplitude multiplier (0.0-2.0)

    Returns:
        dict: Success status
    """
    robot = await ensure_connected()
    await robot.wake_up()
    await robot.shake(intensity=intensity)
    return {"success": True}


@app.tool
async def play_emotion(emotion_name: str) -> dict:
    """
    Play an emotion animation.

    Args:
        emotion_name: Name of emotion (happy, curious, surprised, thinking, sad)

    Returns:
        dict: Success status
    """
    robot = await ensure_connected()
    await robot.play_emotion(name=emotion_name)
    return {"success": True, "emotion": emotion_name}


@app.tool
async def play_sequence(
    emotions: list[str],
    delays: list[float] | None = None,
) -> dict:
    """
    Play a sequence of emotions with optional delays between them.

    Args:
        emotions: List of emotion names to play in order
        delays: Optional list of delays (seconds) between emotions.
                If None or shorter than emotions-1, uses 0.5s default.

    Returns:
        dict: Success status with emotions played

    Examples:
        play_sequence(["curious", "happy", "nod"])
        play_sequence(["thinking", "surprised"], delays=[1.0])
    """
    import asyncio

    robot = await ensure_connected()

    # Default delays
    if delays is None:
        delays = []

    played = []
    for i, emotion in enumerate(emotions):
        await robot.play_emotion(name=emotion)
        played.append(emotion)

        # Add delay between emotions (not after last one)
        if i < len(emotions) - 1:
            delay = delays[i] if i < len(delays) else 0.5
            await asyncio.sleep(delay)

    return {"success": True, "emotions_played": played}


# === AUDIO TOOLS ===


@app.tool
async def speak(text: str, voice: str = "default") -> dict:
    """
    Speak text using text-to-speech.

    Args:
        text: Text to speak
        voice: Voice preset (default, enthusiastic, calm)

    Returns:
        dict: Success status
    """
    robot = await ensure_connected()
    await robot.speak(text=text, voice=voice)
    return {"success": True, "spoken": text[:50] + "..." if len(text) > 50 else text}


@app.tool
async def listen(timeout: float = 5.0) -> dict:
    """
    Listen for speech and transcribe.

    Args:
        timeout: Maximum listen time in seconds

    Returns:
        dict: Transcription result
    """
    robot = await ensure_connected()
    transcription = await robot.listen(timeout=timeout)
    return {"success": True, "transcription": transcription}


# === PERCEPTION TOOLS ===


@app.tool
async def capture_image() -> dict:
    """
    Capture image from camera.

    Returns:
        dict: Image data (bytes length in mock mode)
    """
    robot = await ensure_connected()
    image_data = await robot.capture_image()
    return {
        "success": True,
        "image_bytes": len(image_data),
        "format": "png",
    }


@app.tool
async def get_sensor_data() -> dict:
    """
    Read IMU/accelerometer sensor data.

    Returns:
        dict: Accelerometer and gyroscope readings
    """
    robot = await ensure_connected()
    data = await robot.get_sensor_data()
    return {"success": True, "sensors": data}


@app.tool
async def detect_sound_direction() -> dict:
    """
    Detect direction of loudest sound using mic array.

    Returns:
        dict: Azimuth angle and confidence
    """
    robot = await ensure_connected()
    direction, confidence = await robot.detect_sound_direction()
    return {
        "success": True,
        "azimuth_degrees": direction,
        "confidence": confidence,
    }


# === LIFECYCLE TOOLS ===


@app.tool
async def wake_up() -> dict:
    """
    Wake up robot (enable motors).

    Returns:
        dict: Success status
    """
    robot = await ensure_connected()
    await robot.wake_up()
    return {"success": True, "awake": True}


@app.tool
async def sleep_robot() -> dict:
    """
    Put robot to sleep (disable motors, low power).

    Returns:
        dict: Success status
    """
    robot = await ensure_connected()
    await robot.sleep()
    return {"success": True, "awake": False}


# === STATUS TOOLS (with caching) ===


@app.tool
async def is_awake() -> dict:
    """
    Check if robot motors are enabled.

    Returns:
        dict: Awake status
    """
    # Check cache first
    cached = _cache.get("is_awake")
    if cached is not None:
        return cached

    robot = await ensure_connected()
    awake = await robot.is_awake()
    result: dict[str, Any] = {"awake": awake}

    # Cache for 200ms
    _cache.set("is_awake", result, ttl=0.2)
    return result


@app.tool
async def get_status() -> dict:
    """
    Get comprehensive robot status (cached for 200ms).

    Returns:
        dict: Status including battery, pose, motor states
    """
    # Check cache first
    cached = _cache.get("get_status")
    if cached is not None:
        return cached

    robot = await ensure_connected()
    status = await robot.get_status()
    result: dict[str, Any] = {
        "success": True,
        "is_awake": status.is_awake,
        "battery_percent": status.battery_percent,
        "head_pose": {
            "pitch": status.head_pose.pitch,
            "yaw": status.head_pose.yaw,
            "roll": status.head_pose.roll,
            "z": status.head_pose.z,
        },
        "body_angle": status.body_angle,
        "antennas": {
            "left": status.antenna_state.left,
            "right": status.antenna_state.right,
        },
    }

    # Cache for 200ms
    _cache.set("get_status", result, ttl=0.2)
    return result


@app.tool
async def get_position() -> dict:
    """
    Get current joint positions (cached for 200ms).

    Returns:
        dict: All joint angles
    """
    # Check cache first
    cached = _cache.get("get_position")
    if cached is not None:
        return cached

    robot = await ensure_connected()
    position = await robot.get_position()
    result: dict[str, Any] = {"success": True, "position": position}

    # Cache for 200ms
    _cache.set("get_position", result, ttl=0.2)
    return result


@app.tool
async def get_limits() -> dict:
    """
    Get joint angle limits (cached for longer since static).

    Returns:
        dict: Min/max for each joint
    """
    # Check cache first - limits are static, cache longer
    cached = _cache.get("get_limits")
    if cached is not None:
        return cached

    robot = await ensure_connected()
    limits = await robot.get_limits()
    result: dict[str, Any] = {"success": True, "limits": limits}

    # Cache for 60 seconds since limits are static
    _cache.set("get_limits", result, ttl=60.0)
    return result


if __name__ == "__main__":
    # Run MCP server
    app.run()
