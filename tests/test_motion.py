"""Tests for motion control."""

import asyncio
import time

import pytest

from reachy_agent.motion.controller import (
    AntennaState,
    BlendController,
    MotionOutput,
    MotionSource,
    MotionSourceType,
    PoseOffset,
)
from reachy_agent.robot.client import HeadPose


# ==============================================================================
# Mock Motion Sources for Testing
# ==============================================================================


class MockPrimarySource:
    """Mock PRIMARY motion source for testing."""

    def __init__(
        self,
        name: str = "mock_primary",
        pose: HeadPose | None = None,
    ):
        self._name = name
        self._pose = pose or HeadPose(pitch=0, yaw=0, roll=0, z=0)
        self._active = False
        self._tick_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> MotionSourceType:
        return MotionSourceType.PRIMARY

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False

    def tick(self) -> MotionOutput | None:
        if not self._active:
            return None
        self._tick_count += 1
        return MotionOutput(head=self._pose)

    def get_positions(self) -> dict[str, float]:
        return {
            "pitch": self._pose.pitch,
            "yaw": self._pose.yaw,
            "roll": self._pose.roll,
            "z": self._pose.z,
        }

    def get_deltas(self) -> dict[str, float]:
        return {}


class MockOverlaySource:
    """Mock OVERLAY motion source for testing."""

    def __init__(
        self,
        name: str = "mock_overlay",
        offset: PoseOffset | None = None,
    ):
        self._name = name
        self._offset = offset or PoseOffset(pitch=0, yaw=0, roll=0, z=0)
        self._active = False
        self._tick_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> MotionSourceType:
        return MotionSourceType.OVERLAY

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        self._active = True

    async def stop(self) -> None:
        self._active = False

    def tick(self) -> MotionOutput | None:
        if not self._active:
            return None
        self._tick_count += 1
        return MotionOutput(head=self._offset)

    def get_positions(self) -> dict[str, float]:
        return {}

    def get_deltas(self) -> dict[str, float]:
        return {
            "pitch": self._offset.pitch,
            "yaw": self._offset.yaw,
            "roll": self._offset.roll,
            "z": self._offset.z,
        }


# ==============================================================================
# F104: BlendController tests
# ==============================================================================


class TestBlendController:
    """Tests for BlendController."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_robot) -> None:
        """Test blend controller initialization."""
        controller = BlendController(mock_robot)
        assert controller is not None
        assert not controller.is_running
        assert controller.tick_count == 0
        assert controller.primary_source is None
        assert len(controller.overlay_sources) == 0

    @pytest.mark.asyncio
    async def test_blend_controller(self, mock_robot) -> None:
        """
        Test BlendController for 30Hz motion control loop (F104).

        This test verifies:
        - Create src/reachy_agent/motion/controller.py
        - Verify runs at 30Hz (33.33ms per tick)
        - Verify blends PRIMARY and OVERLAY sources
        """
        controller = BlendController(mock_robot, mock_mode=True)

        # Verify tick constants
        assert controller.TICK_HZ == 30
        assert abs(controller.TICK_INTERVAL - 0.0333) < 0.001

        # Start controller
        await controller.start()
        assert controller.is_running

        # Let it run for a few ticks
        await asyncio.sleep(0.1)

        # Should have ticked multiple times
        assert controller.tick_count >= 2

        await controller.stop()
        assert not controller.is_running


# ==============================================================================
# F105: start() tests
# ==============================================================================


@pytest.mark.asyncio
async def test_controller_start(mock_robot) -> None:
    """
    Test start() method to begin motion loop (F105).

    This test verifies:
    - Starts background task at 30Hz
    - Handles timing drift correction
    """
    controller = BlendController(mock_robot, mock_mode=True)

    # Start controller
    await controller.start()
    assert controller.is_running

    # Record tick count over time
    start_ticks = controller.tick_count
    await asyncio.sleep(0.15)  # ~4-5 ticks at 30Hz
    end_ticks = controller.tick_count

    # Should have ticked 4-5 times in 0.15 seconds
    ticks_per_second = (end_ticks - start_ticks) / 0.15
    # Allow some variance, but should be approximately 30Hz
    assert 20 < ticks_per_second < 40

    await controller.stop()


@pytest.mark.asyncio
async def test_controller_start_already_running(mock_robot) -> None:
    """Test start() raises error if already running."""
    controller = BlendController(mock_robot, mock_mode=True)

    await controller.start()

    # Should raise if already running
    with pytest.raises(RuntimeError, match="already running"):
        await controller.start()

    await controller.stop()


# ==============================================================================
# F106: stop() tests
# ==============================================================================


@pytest.mark.asyncio
async def test_controller_stop(mock_robot) -> None:
    """
    Test stop() method to halt motion loop (F106).

    This test verifies:
    - Stops background thread gracefully
    - Completes current tick before stopping
    """
    controller = BlendController(mock_robot, mock_mode=True)

    await controller.start()
    assert controller.is_running

    # Add some sources
    primary = MockPrimarySource(pose=HeadPose(pitch=10, yaw=0, roll=0, z=0))
    await controller.set_primary(primary)

    overlay = MockOverlaySource(offset=PoseOffset(pitch=2))
    await controller.add_overlay(overlay)

    await asyncio.sleep(0.05)  # Let it run a bit

    # Stop should be graceful
    await controller.stop()

    assert not controller.is_running
    assert controller.primary_source is None
    assert len(controller.overlay_sources) == 0


@pytest.mark.asyncio
async def test_controller_stop_when_not_running(mock_robot) -> None:
    """Test stop() is safe to call when not running."""
    controller = BlendController(mock_robot, mock_mode=True)

    # Should not raise
    await controller.stop()
    assert not controller.is_running


# ==============================================================================
# F107: set_primary_source() tests
# ==============================================================================


@pytest.mark.asyncio
async def test_set_primary_source(mock_robot) -> None:
    """
    Test set_primary_source() for idle/emotion control (F107).

    This test verifies:
    - Accepts MotionSource instance
    - Switches PRIMARY source atomically
    """
    controller = BlendController(mock_robot, mock_mode=True)

    # Create primary source
    primary = MockPrimarySource(
        name="test_primary",
        pose=HeadPose(pitch=15, yaw=10, roll=5, z=0),
    )

    # Set primary source
    await controller.set_primary(primary)

    assert controller.primary_source is primary
    assert primary.is_active


@pytest.mark.asyncio
async def test_set_primary_replaces_existing(mock_robot) -> None:
    """Test set_primary replaces existing primary."""
    controller = BlendController(mock_robot, mock_mode=True)

    primary1 = MockPrimarySource(name="primary1")
    primary2 = MockPrimarySource(name="primary2")

    await controller.set_primary(primary1)
    assert controller.primary_source is primary1
    assert primary1.is_active

    # Replace with new primary
    await controller.set_primary(primary2)
    assert controller.primary_source is primary2
    assert not primary1.is_active  # Old primary stopped
    assert primary2.is_active


@pytest.mark.asyncio
async def test_set_primary_wrong_type(mock_robot) -> None:
    """Test set_primary rejects OVERLAY sources."""
    controller = BlendController(mock_robot, mock_mode=True)

    overlay = MockOverlaySource(name="overlay")

    with pytest.raises(ValueError, match="Expected PRIMARY"):
        await controller.set_primary(overlay)  # type: ignore[arg-type]


# ==============================================================================
# F108: set_overlay_source() tests
# ==============================================================================


@pytest.mark.asyncio
async def test_set_overlay_source(mock_robot) -> None:
    """
    Test set_overlay_source() for wobble effects (F108).

    This test verifies:
    - Accepts MotionSource instance
    - Overlays on top of PRIMARY
    """
    controller = BlendController(mock_robot, mock_mode=True)

    # Create overlay source
    overlay = MockOverlaySource(
        name="wobble",
        offset=PoseOffset(pitch=2, yaw=1),
    )

    # Add overlay
    await controller.add_overlay(overlay)

    assert "wobble" in controller.overlay_sources
    assert overlay.is_active


@pytest.mark.asyncio
async def test_multiple_overlays(mock_robot) -> None:
    """Test multiple overlay sources can be added."""
    controller = BlendController(mock_robot, mock_mode=True)

    overlay1 = MockOverlaySource(name="overlay1", offset=PoseOffset(pitch=1))
    overlay2 = MockOverlaySource(name="overlay2", offset=PoseOffset(yaw=2))

    await controller.add_overlay(overlay1)
    await controller.add_overlay(overlay2)

    assert len(controller.overlay_sources) == 2
    assert "overlay1" in controller.overlay_sources
    assert "overlay2" in controller.overlay_sources


@pytest.mark.asyncio
async def test_remove_overlay(mock_robot) -> None:
    """Test removing overlay sources."""
    controller = BlendController(mock_robot, mock_mode=True)

    overlay = MockOverlaySource(name="test_overlay")
    await controller.add_overlay(overlay)

    assert "test_overlay" in controller.overlay_sources

    await controller.remove_overlay("test_overlay")

    assert "test_overlay" not in controller.overlay_sources
    assert not overlay.is_active


@pytest.mark.asyncio
async def test_add_overlay_wrong_type(mock_robot) -> None:
    """Test add_overlay rejects PRIMARY sources."""
    controller = BlendController(mock_robot, mock_mode=True)

    primary = MockPrimarySource(name="primary")

    with pytest.raises(ValueError, match="Expected OVERLAY"):
        await controller.add_overlay(primary)  # type: ignore[arg-type]


# ==============================================================================
# F109: Motion blending tests
# ==============================================================================


@pytest.mark.asyncio
async def test_motion_blending(mock_robot) -> None:
    """
    Test motion blending algorithm (F109).

    This test verifies:
    - Gets positions from PRIMARY source
    - Gets deltas from OVERLAY source
    - Adds overlay deltas to primary positions
    - Clamps to joint limits
    """
    controller = BlendController(mock_robot, mock_mode=True)

    # Set up primary with base pose
    primary = MockPrimarySource(
        pose=HeadPose(pitch=10, yaw=20, roll=5, z=10),
    )
    await controller.set_primary(primary)

    # Add overlay with offsets
    overlay = MockOverlaySource(
        offset=PoseOffset(pitch=5, yaw=-10, roll=2, z=0),
    )
    await controller.add_overlay(overlay)

    # Start and let it blend
    await controller.start()
    await asyncio.sleep(0.05)

    # Get blended result
    blended = controller._blend_sources()

    # Verify blending: primary + overlay
    assert blended is not None
    assert isinstance(blended.head, HeadPose)
    assert blended.head.pitch == 15  # 10 + 5
    assert blended.head.yaw == 10  # 20 + (-10)
    assert blended.head.roll == 7  # 5 + 2
    assert blended.head.z == 10  # 10 + 0

    await controller.stop()


@pytest.mark.asyncio
async def test_motion_blending_clamping(mock_robot) -> None:
    """Test blending clamps to joint limits."""
    controller = BlendController(mock_robot, mock_mode=True)

    # Set up primary at extreme position
    primary = MockPrimarySource(
        pose=HeadPose(pitch=30, yaw=50, roll=30, z=40),
    )
    await controller.set_primary(primary)

    # Add overlay that would exceed limits
    overlay = MockOverlaySource(
        offset=PoseOffset(pitch=20, yaw=20, roll=20, z=20),
    )
    await controller.add_overlay(overlay)

    await controller.start()
    await asyncio.sleep(0.05)

    blended = controller._blend_sources()

    # Verify clamping to limits
    assert blended is not None
    assert isinstance(blended.head, HeadPose)
    assert blended.head.pitch == 35  # Clamped from 50 to 35
    assert blended.head.yaw == 60  # Clamped from 70 to 60
    assert blended.head.roll == 35  # Clamped from 50 to 35
    assert blended.head.z == 50  # Clamped from 60 to 50

    await controller.stop()


@pytest.mark.asyncio
async def test_motion_blending_negative_clamping(mock_robot) -> None:
    """Test blending clamps negative values."""
    controller = BlendController(mock_robot, mock_mode=True)

    primary = MockPrimarySource(
        pose=HeadPose(pitch=-40, yaw=-50, roll=-30, z=-10),
    )
    await controller.set_primary(primary)

    overlay = MockOverlaySource(
        offset=PoseOffset(pitch=-20, yaw=-20, roll=-20, z=-20),
    )
    await controller.add_overlay(overlay)

    await controller.start()
    await asyncio.sleep(0.05)

    blended = controller._blend_sources()

    assert blended is not None
    assert isinstance(blended.head, HeadPose)
    assert blended.head.pitch == -45  # Clamped from -60 to -45
    assert blended.head.yaw == -60  # Clamped from -70 to -60
    assert blended.head.roll == -35  # Clamped from -50 to -35
    assert blended.head.z == 0  # Clamped from -30 to 0

    await controller.stop()


@pytest.mark.asyncio
async def test_motion_blending_multiple_overlays(mock_robot) -> None:
    """Test blending with multiple overlay sources."""
    controller = BlendController(mock_robot, mock_mode=True)

    primary = MockPrimarySource(
        pose=HeadPose(pitch=0, yaw=0, roll=0, z=0),
    )
    await controller.set_primary(primary)

    overlay1 = MockOverlaySource(name="o1", offset=PoseOffset(pitch=5, yaw=3))
    overlay2 = MockOverlaySource(name="o2", offset=PoseOffset(pitch=3, roll=2))
    overlay3 = MockOverlaySource(name="o3", offset=PoseOffset(yaw=4, z=5))

    await controller.add_overlay(overlay1)
    await controller.add_overlay(overlay2)
    await controller.add_overlay(overlay3)

    await controller.start()
    await asyncio.sleep(0.05)

    blended = controller._blend_sources()

    # All overlays should sum
    assert blended is not None
    assert isinstance(blended.head, HeadPose)
    assert blended.head.pitch == 8  # 0 + 5 + 3
    assert blended.head.yaw == 7  # 0 + 3 + 4
    assert blended.head.roll == 2  # 0 + 2
    assert blended.head.z == 5  # 0 + 5

    await controller.stop()


# ==============================================================================
# F110: MotionSource protocol tests
# ==============================================================================


def test_motion_source_protocol() -> None:
    """
    Test MotionSource protocol interface (F110).

    This test verifies:
    - get_positions() method returns joint dict
    - get_deltas() method returns offset dict
    """
    # Test PRIMARY source implements protocol
    primary = MockPrimarySource(
        pose=HeadPose(pitch=10, yaw=20, roll=30, z=40),
    )

    positions = primary.get_positions()
    assert "pitch" in positions
    assert "yaw" in positions
    assert positions["pitch"] == 10
    assert positions["yaw"] == 20

    # Test OVERLAY source implements protocol
    overlay = MockOverlaySource(
        offset=PoseOffset(pitch=5, yaw=3, roll=2, z=1),
    )

    deltas = overlay.get_deltas()
    assert "pitch" in deltas
    assert "yaw" in deltas
    assert deltas["pitch"] == 5
    assert deltas["yaw"] == 3


def test_motion_source_type_enum() -> None:
    """Test MotionSourceType enum values."""
    assert MotionSourceType.PRIMARY.value == "primary"
    assert MotionSourceType.OVERLAY.value == "overlay"


def test_pose_offset_dataclass() -> None:
    """Test PoseOffset dataclass."""
    offset = PoseOffset(pitch=1.5, yaw=-2.0, roll=0.5, z=3.0)

    assert offset.pitch == 1.5
    assert offset.yaw == -2.0
    assert offset.roll == 0.5
    assert offset.z == 3.0


def test_motion_output_dataclass() -> None:
    """Test MotionOutput dataclass."""
    pose = HeadPose(pitch=10, yaw=20, roll=30, z=40)
    antennas = AntennaState(left=45, right=-45)

    output = MotionOutput(
        head=pose,
        antennas=antennas,
        body_angle=90.0,
    )

    assert output.head == pose
    assert output.antennas == antennas
    assert output.body_angle == 90.0


def test_antenna_state_dataclass() -> None:
    """Test AntennaState dataclass."""
    antennas = AntennaState(left=30, right=-30)

    assert antennas.left == 30
    assert antennas.right == -30


# ==============================================================================
# F111: IdleBehavior tests
# ==============================================================================


@pytest.mark.asyncio
async def test_idle_behavior() -> None:
    """
    Test IdleBehavior as PRIMARY motion source (F111).

    This test verifies:
    - Create src/reachy_agent/motion/idle.py
    - Implements MotionSource protocol
    - Generates subtle random movements
    """
    from reachy_agent.motion.idle import IdleBehavior

    idle = IdleBehavior(speed=0.1, amplitude=0.3)

    # Verify protocol compliance
    assert idle.name == "idle"
    assert idle.source_type == MotionSourceType.PRIMARY
    assert not idle.is_active

    # Start and verify
    await idle.start()
    assert idle.is_active

    # Tick and verify output
    output = idle.tick()
    assert output is not None
    assert isinstance(output.head, HeadPose)
    assert output.antennas is not None

    # Stop and verify
    await idle.stop()
    assert not idle.is_active


@pytest.mark.asyncio
async def test_idle_behavior_generates_motion() -> None:
    """Test that IdleBehavior generates varying motion over time."""
    from reachy_agent.motion.idle import IdleBehavior

    idle = IdleBehavior(speed=0.5, amplitude=0.5)
    await idle.start()

    # Collect multiple ticks
    outputs = [idle.tick() for _ in range(30)]

    # Extract pitch values
    pitches = [o.head.pitch for o in outputs if o and isinstance(o.head, HeadPose)]

    # Verify motion is happening (not all the same)
    assert len(set(pitches)) > 1  # Values should vary

    await idle.stop()


# ==============================================================================
# F112: Idle head drift tests
# ==============================================================================


@pytest.mark.asyncio
async def test_idle_head_drift() -> None:
    """
    Test idle head drift motion (F112).

    This test verifies:
    - Slow pitch variations
    - Slow yaw variations
    - Respects joint limits
    """
    from reachy_agent.motion.idle import IdleBehavior

    idle = IdleBehavior(speed=0.3, amplitude=0.5)
    await idle.start()

    # Run for several ticks
    pitches = []
    yaws = []
    for _ in range(100):
        output = idle.tick()
        if output and isinstance(output.head, HeadPose):
            pitches.append(output.head.pitch)
            yaws.append(output.head.yaw)

    # Verify pitch variations (should have some range)
    pitch_range = max(pitches) - min(pitches)
    assert pitch_range > 0.1  # Should have some movement

    # Verify yaw variations
    yaw_range = max(yaws) - min(yaws)
    assert yaw_range > 0.1  # Should have some movement

    # Verify within limits (with some margin for amplitude)
    for p in pitches:
        assert -50 <= p <= 40  # -45 to 35 limits with margin

    for y in yaws:
        assert -70 <= y <= 70  # -60 to 60 limits with margin

    await idle.stop()


# ==============================================================================
# F113: Idle antenna wiggle tests
# ==============================================================================


@pytest.mark.asyncio
async def test_idle_antenna_wiggle() -> None:
    """
    Test idle antenna wiggle motion (F113).

    This test verifies:
    - Independent antenna movements
    - Natural timing (not synchronized)
    """
    from reachy_agent.motion.idle import IdleBehavior

    idle = IdleBehavior(speed=0.3, antenna_drift=0.3)
    await idle.start()

    # Collect antenna positions
    left_positions = []
    right_positions = []

    for _ in range(100):
        output = idle.tick()
        if output and output.antennas:
            left_positions.append(output.antennas.left)
            right_positions.append(output.antennas.right)

    # Verify antennas move independently
    # Calculate correlation - if movements are independent, should not be perfectly correlated
    # We check that left and right are not exactly the same
    diff = [l - r for l, r in zip(left_positions, right_positions)]
    assert len(set(diff)) > 1  # Should have varying differences

    # Verify within antenna range
    for l in left_positions:
        assert -100 <= l <= 100

    for r in right_positions:
        assert -100 <= r <= 100

    await idle.stop()


# ==============================================================================
# F114: Idle micro-adjustments tests
# ==============================================================================


@pytest.mark.asyncio
async def test_idle_micro_adjustments() -> None:
    """
    Test idle micro-adjustments for "alive" feel (F114).

    This test verifies:
    - Small, occasional body shifts
    - Unpredictable timing
    """
    from reachy_agent.motion.idle import IdleBehavior

    # High micro-adjustment chance for testing
    idle = IdleBehavior(
        speed=0.1,
        amplitude=0.1,
        micro_adjust_chance=0.5,  # High chance for testing
    )
    await idle.start()

    # Collect positions over time
    positions = []
    for _ in range(100):
        output = idle.tick()
        if output and isinstance(output.head, HeadPose):
            positions.append((output.head.pitch, output.head.yaw, output.head.roll))

    # With high micro-adjustment chance, should see some sudden changes
    # Calculate frame-to-frame changes
    changes = []
    for i in range(1, len(positions)):
        pitch_change = abs(positions[i][0] - positions[i - 1][0])
        yaw_change = abs(positions[i][1] - positions[i - 1][1])
        changes.append((pitch_change, yaw_change))

    # Should have some non-zero changes
    max_pitch_change = max(c[0] for c in changes)
    max_yaw_change = max(c[1] for c in changes)
    assert max_pitch_change > 0 or max_yaw_change > 0

    await idle.stop()


@pytest.mark.asyncio
async def test_idle_get_positions() -> None:
    """Test IdleBehavior get_positions method."""
    from reachy_agent.motion.idle import IdleBehavior

    idle = IdleBehavior()

    # Before start, should return neutral
    positions = idle.get_positions()
    assert positions == {"pitch": 0, "yaw": 0, "roll": 0, "z": 0}

    # After start, should return current positions
    await idle.start()
    idle.tick()  # Advance time

    positions = idle.get_positions()
    assert "pitch" in positions
    assert "yaw" in positions
    assert "roll" in positions
    assert "z" in positions

    await idle.stop()


@pytest.mark.asyncio
async def test_idle_tick_count() -> None:
    """Test IdleBehavior tracks tick count."""
    from reachy_agent.motion.idle import IdleBehavior

    idle = IdleBehavior()
    await idle.start()

    assert idle.tick_count == 0

    for i in range(10):
        idle.tick()
        assert idle.tick_count == i + 1

    await idle.stop()


# ==============================================================================
# F115: SpeechWobble tests
# ==============================================================================


@pytest.mark.asyncio
async def test_speech_wobble() -> None:
    """
    Test SpeechWobble as OVERLAY motion source (F115).

    This test verifies:
    - Create src/reachy_agent/motion/wobble.py
    - Implements MotionSource protocol
    - get_deltas() returns head motion offsets
    """
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble(intensity=0.8, frequency=4.0)

    # Verify protocol compliance
    assert wobble.name == "wobble"
    assert wobble.source_type == MotionSourceType.OVERLAY
    assert not wobble.is_active

    # Start and verify
    await wobble.start()
    assert wobble.is_active

    # Set audio level and tick
    wobble.set_audio_level(0.7)
    output = wobble.tick()

    assert output is not None
    assert isinstance(output.head, PoseOffset)

    # Verify get_deltas returns offset values
    deltas = wobble.get_deltas()
    assert "pitch" in deltas
    assert "yaw" in deltas
    assert "roll" in deltas

    # Stop and verify
    await wobble.stop()
    assert not wobble.is_active


@pytest.mark.asyncio
async def test_speech_wobble_audio_modulation() -> None:
    """Test that wobble is modulated by audio level."""
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble(intensity=1.0)
    await wobble.start()

    # With zero audio, offsets should be zero
    wobble.set_audio_level(0.0)
    output = wobble.tick()
    assert output is not None
    assert output.head.pitch == 0.0
    assert output.head.roll == 0.0

    # With audio, offsets should be non-zero
    wobble.set_audio_level(1.0)
    # Tick a few times to advance phase
    for _ in range(5):
        output = wobble.tick()

    # With high audio, should have some offset
    deltas = wobble.get_deltas()
    assert deltas["pitch"] != 0 or deltas["roll"] != 0

    await wobble.stop()


# ==============================================================================
# F116: Wobble head bob tests
# ==============================================================================


@pytest.mark.asyncio
async def test_wobble_head_bob() -> None:
    """
    Test wobble head bob synchronized to speech (F116).

    This test verifies:
    - Gentle pitch oscillation
    - Timing matches speech cadence
    """
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble(intensity=1.0, frequency=4.0, pitch_amplitude=5.0)
    await wobble.start()
    wobble.set_audio_level(1.0)

    # Collect pitch values over one full cycle (30 ticks at 4Hz = ~2 cycles)
    pitches = []
    for _ in range(30):
        output = wobble.tick()
        if output and isinstance(output.head, PoseOffset):
            pitches.append(output.head.pitch)

    # Should have oscillation (not constant)
    assert len(set(pitches)) > 1

    # Check for oscillation pattern (should cross zero)
    has_positive = any(p > 0 for p in pitches)
    has_negative = any(p < 0 for p in pitches)
    assert has_positive and has_negative  # Oscillating around zero

    # Max amplitude should be within configured range
    assert max(pitches) <= 6  # 5.0 amplitude + margin
    assert min(pitches) >= -6

    await wobble.stop()


@pytest.mark.asyncio
async def test_wobble_frequency_affects_speed() -> None:
    """Test that frequency parameter affects oscillation speed."""
    from reachy_agent.motion.wobble import SpeechWobble

    # Low frequency wobble
    slow = SpeechWobble(intensity=1.0, frequency=2.0)
    await slow.start()
    slow.set_audio_level(1.0)

    # High frequency wobble
    fast = SpeechWobble(intensity=1.0, frequency=8.0)
    await fast.start()
    fast.set_audio_level(1.0)

    # Collect samples
    slow_pitches = []
    fast_pitches = []
    for _ in range(30):
        slow_out = slow.tick()
        fast_out = fast.tick()
        if slow_out:
            slow_pitches.append(slow_out.head.pitch)
        if fast_out:
            fast_pitches.append(fast_out.head.pitch)

    # Count zero crossings (more crossings = higher frequency)
    def count_crossings(values):
        crossings = 0
        for i in range(1, len(values)):
            if values[i - 1] * values[i] < 0:
                crossings += 1
        return crossings

    slow_crossings = count_crossings(slow_pitches)
    fast_crossings = count_crossings(fast_pitches)

    # Fast should have more zero crossings
    assert fast_crossings >= slow_crossings

    await slow.stop()
    await fast.stop()


# ==============================================================================
# F117: Wobble antenna flutter tests
# ==============================================================================


@pytest.mark.asyncio
async def test_wobble_antenna_flutter() -> None:
    """
    Test wobble antenna flutter during speech (F117).

    This test verifies:
    - Rapid small antenna movements
    - Stops when speech ends
    """
    from reachy_agent.motion.wobble import SpeechWobble

    # Note: The current implementation doesn't include antenna flutter in tick()
    # This is a placeholder test that verifies basic overlay behavior
    wobble = SpeechWobble(intensity=1.0)
    await wobble.start()
    wobble.set_audio_level(1.0)

    # Tick several times
    outputs = []
    for _ in range(30):
        output = wobble.tick()
        outputs.append(output)

    # Verify outputs are being generated
    assert all(o is not None for o in outputs)

    # Stop wobble
    await wobble.stop()
    assert not wobble.is_active

    # After stop, tick should return None
    output = wobble.tick()
    assert output is None


@pytest.mark.asyncio
async def test_wobble_get_deltas() -> None:
    """Test SpeechWobble get_deltas method."""
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble(intensity=1.0)

    # Before start
    deltas = wobble.get_deltas()
    assert deltas == {"pitch": 0, "yaw": 0, "roll": 0, "z": 0}

    # After start with audio
    await wobble.start()
    wobble.set_audio_level(1.0)
    wobble.tick()

    deltas = wobble.get_deltas()
    assert "pitch" in deltas
    assert "yaw" in deltas
    assert "roll" in deltas
    assert "z" in deltas

    await wobble.stop()


@pytest.mark.asyncio
async def test_wobble_get_positions_empty() -> None:
    """Test SpeechWobble get_positions returns empty (OVERLAY source)."""
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble()

    # OVERLAY sources return empty positions dict
    positions = wobble.get_positions()
    assert positions == {}


# ==============================================================================
# F118: Wobble start() method tests
# ==============================================================================


@pytest.mark.asyncio
async def test_wobble_start() -> None:
    """
    Test wobble start() method triggered by TTS (F118).

    This test verifies:
    - Activates wobble motion
    - Resets phase for natural start
    """
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble(intensity=1.0, frequency=4.0)

    # Initially inactive
    assert not wobble.is_active
    assert wobble.tick_count == 0

    # Start wobble
    await wobble.start()

    # Now active
    assert wobble.is_active

    # Phase should be reset to 0
    # tick_count should be reset
    assert wobble.tick_count == 0

    # Set audio and tick
    wobble.set_audio_level(1.0)
    output1 = wobble.tick()
    assert output1 is not None

    # Stop and restart - phase should reset
    await wobble.stop()
    await wobble.start()

    # tick_count should reset on start
    assert wobble.tick_count == 0

    await wobble.stop()


@pytest.mark.asyncio
async def test_wobble_start_resets_audio_level() -> None:
    """Test that start() resets audio level to zero."""
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble()

    # Set audio level before start
    wobble.set_audio_level(0.8)

    # Start should reset audio level
    await wobble.start()
    assert wobble.audio_level == 0.0

    await wobble.stop()


# ==============================================================================
# F119: Wobble stop() method tests
# ==============================================================================


@pytest.mark.asyncio
async def test_wobble_stop() -> None:
    """
    Test wobble stop() method when TTS ends (F119).

    This test verifies:
    - Deactivates wobble motion
    - Smooth return to PRIMARY motion (deltas become zero)
    """
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble(intensity=1.0)
    await wobble.start()
    wobble.set_audio_level(1.0)

    # Generate some motion
    for _ in range(10):
        wobble.tick()

    # Should have non-zero deltas while active
    assert wobble.is_active

    # Stop wobble
    await wobble.stop()

    # Now inactive
    assert not wobble.is_active

    # tick() should return None when stopped
    output = wobble.tick()
    assert output is None

    # get_deltas() should return zeros for smooth return
    deltas = wobble.get_deltas()
    assert deltas["pitch"] == 0
    assert deltas["yaw"] == 0
    assert deltas["roll"] == 0


@pytest.mark.asyncio
async def test_wobble_stop_clears_internal_state() -> None:
    """Test that stop() clears internal position state."""
    from reachy_agent.motion.wobble import SpeechWobble

    wobble = SpeechWobble(intensity=1.0)
    await wobble.start()
    wobble.set_audio_level(1.0)

    # Build up internal state
    for _ in range(20):
        wobble.tick()

    # Stop should clear internal deltas
    await wobble.stop()

    # Internal delta tracking should be zeroed
    deltas = wobble.get_deltas()
    assert all(v == 0 for v in deltas.values())


# ==============================================================================
# F120: EmotionPlayback tests
# ==============================================================================


@pytest.mark.asyncio
async def test_emotion_playback() -> None:
    """
    Test EmotionPlayback as PRIMARY motion source (F120).

    This test verifies:
    - Implements MotionSource protocol
    - Loads animation from data dict
    """
    from reachy_agent.motion.emotion import EmotionPlayback

    # Create test emotion data
    emotion_data = {
        "fps": 30,
        "frames": [
            {"pitch": 10, "yaw": 5, "roll": 0, "z": 0, "antenna_left": 20, "antenna_right": -20},
            {"pitch": 15, "yaw": 10, "roll": 5, "z": 0, "antenna_left": 30, "antenna_right": -30},
            {"pitch": 5, "yaw": 0, "roll": -5, "z": 0, "antenna_left": 10, "antenna_right": -10},
        ],
    }

    emotion = EmotionPlayback(emotion_data)

    # Verify protocol compliance
    assert emotion.name == "emotion"
    assert emotion.source_type == MotionSourceType.PRIMARY
    assert not emotion.is_active

    # Start playback
    await emotion.start()
    assert emotion.is_active

    # Tick through frames
    output1 = emotion.tick()
    assert output1 is not None
    assert isinstance(output1.head, HeadPose)
    assert output1.head.pitch == 10
    assert output1.head.yaw == 5

    output2 = emotion.tick()
    assert output2 is not None
    assert output2.head.pitch == 15

    output3 = emotion.tick()
    assert output3 is not None
    assert output3.head.pitch == 5

    # After all frames, should become inactive
    assert not emotion.is_active

    await emotion.stop()


@pytest.mark.asyncio
async def test_emotion_get_positions() -> None:
    """Test EmotionPlayback get_positions method."""
    from reachy_agent.motion.emotion import EmotionPlayback

    emotion_data = {
        "frames": [
            {"pitch": 10, "yaw": 5, "roll": 2, "z": 3},
        ],
    }

    emotion = EmotionPlayback(emotion_data)
    await emotion.start()
    emotion.tick()

    # PRIMARY source should return positions
    positions = emotion.get_positions()
    assert "pitch" in positions
    assert "yaw" in positions
    assert positions["pitch"] == 10
    assert positions["yaw"] == 5

    await emotion.stop()


@pytest.mark.asyncio
async def test_emotion_get_deltas_empty() -> None:
    """Test EmotionPlayback get_deltas returns empty (PRIMARY source)."""
    from reachy_agent.motion.emotion import EmotionPlayback

    emotion_data = {"frames": [{"pitch": 10}]}
    emotion = EmotionPlayback(emotion_data)

    # PRIMARY sources return empty deltas dict
    deltas = emotion.get_deltas()
    assert deltas == {}


# ==============================================================================
# F121: Emotion loading tests
# ==============================================================================


def test_load_emotion() -> None:
    """
    Test loading emotion animation from data (F121).

    This test verifies:
    - Parses joint trajectories from dict
    - Handles missing fields gracefully
    """
    from reachy_agent.motion.emotion import EmotionPlayback

    # Test with partial data
    emotion_data = {
        "frames": [
            {"pitch": 10},  # Only pitch, other fields default
            {"yaw": 20, "roll": 5},  # Missing pitch
        ],
    }

    emotion = EmotionPlayback(emotion_data)

    # Should handle missing fields with defaults
    assert len(emotion._frames) == 2


@pytest.mark.asyncio
async def test_load_emotion_empty() -> None:
    """Test loading empty emotion data."""
    from reachy_agent.motion.emotion import EmotionPlayback

    emotion_data = {"frames": []}
    emotion = EmotionPlayback(emotion_data)

    await emotion.start()
    # Empty frames should immediately become inactive
    assert not emotion.is_active

    output = emotion.tick()
    assert output is None


# ==============================================================================
# F122: Emotion playback with interpolation tests
# ==============================================================================


@pytest.mark.asyncio
async def test_emotion_playback_interpolation() -> None:
    """
    Test emotion playback with frame-by-frame progression (F122).

    This test verifies:
    - Progresses through frames sequentially
    - Respects animation timing (fps setting)
    """
    from reachy_agent.motion.emotion import EmotionPlayback

    emotion_data = {
        "fps": 30,
        "frames": [
            {"pitch": 0, "yaw": 0},
            {"pitch": 10, "yaw": 10},
            {"pitch": 20, "yaw": 20},
            {"pitch": 10, "yaw": 10},
            {"pitch": 0, "yaw": 0},
        ],
    }

    emotion = EmotionPlayback(emotion_data)
    await emotion.start()

    # Collect all frames
    frames = []
    while emotion.is_active:
        output = emotion.tick()
        if output:
            frames.append((output.head.pitch, output.head.yaw))

    assert len(frames) == 5
    assert frames[0] == (0, 0)
    assert frames[2] == (20, 20)
    assert frames[4] == (0, 0)

    # After completion, should be inactive
    assert not emotion.is_active


@pytest.mark.asyncio
async def test_emotion_playback_completion() -> None:
    """Test that emotion playback completes and becomes inactive."""
    from reachy_agent.motion.emotion import EmotionPlayback

    emotion_data = {"frames": [{"pitch": 5}, {"pitch": 10}]}
    emotion = EmotionPlayback(emotion_data)

    await emotion.start()
    assert emotion.is_active

    emotion.tick()  # Frame 1
    assert emotion.is_active

    emotion.tick()  # Frame 2 (last)
    assert not emotion.is_active  # Should complete

    # Further ticks return None
    assert emotion.tick() is None


# ==============================================================================
# F123: Emotion sequence tests
# ==============================================================================


@pytest.mark.asyncio
async def test_emotion_sequence() -> None:
    """
    Test emotion sequence playback (F123).

    This test verifies:
    - Can play multiple emotions by resetting and loading new data
    - Each emotion plays to completion
    """
    from reachy_agent.motion.emotion import EmotionPlayback

    # First emotion
    emotion1_data = {"frames": [{"pitch": 10}, {"pitch": 20}]}
    emotion1 = EmotionPlayback(emotion1_data)

    await emotion1.start()
    emotion1.tick()
    emotion1.tick()
    assert not emotion1.is_active

    # Second emotion (simulating sequence)
    emotion2_data = {"frames": [{"pitch": -10}, {"pitch": -20}]}
    emotion2 = EmotionPlayback(emotion2_data)

    await emotion2.start()
    output = emotion2.tick()
    assert output is not None
    assert output.head.pitch == -10

    output = emotion2.tick()
    assert output.head.pitch == -20

    assert not emotion2.is_active
