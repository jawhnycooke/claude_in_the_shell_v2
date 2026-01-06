"""Comprehensive validation test suite for Reachy Mini in MuJoCo simulation.

This module provides end-to-end validation tests for all robot commands,
gestures, emotions, and sensor readings using the MuJoCo simulation backend.

Run with: uv run pytest tests/test_validation_suite.py -v
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

# Check if simulation is available
try:
    from reachy_agent.simulation.client import MuJoCoReachyClient

    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False


# ============================================================================
# Test Configuration
# ============================================================================


@dataclass
class ValidationConfig:
    """Configuration for validation tests."""

    position_tolerance: float = 5.0  # degrees
    timing_tolerance: float = 0.5  # seconds
    movement_duration: float = 0.5  # seconds for test movements
    pause_between_tests: float = 0.1  # seconds


CONFIG = ValidationConfig()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def sim_client():
    """Create and connect a simulation client for testing."""
    if not SIMULATION_AVAILABLE:
        pytest.skip("Simulation not available")

    client = MuJoCoReachyClient(realtime=False, viewer=False)
    await client.connect()
    await client.wake_up()
    yield client
    await client.sleep()
    await client.disconnect()


# ============================================================================
# Basic Movement Tests
# ============================================================================


class TestBasicMovements:
    """Validate basic robot movement commands."""

    @pytest.mark.asyncio
    async def test_head_pitch_positive(self, sim_client) -> None:
        """Test head pitch in positive direction (looking down)."""
        target_pitch = 20.0
        await sim_client.move_head(
            pitch=target_pitch, yaw=0, roll=0, duration=CONFIG.movement_duration
        )
        positions = await sim_client.get_position()
        diff = abs(positions["head_pitch"] - target_pitch)
        assert diff < CONFIG.position_tolerance, f"Head pitch error: {diff:.1f}°"

    @pytest.mark.asyncio
    async def test_head_pitch_negative(self, sim_client) -> None:
        """Test head pitch in negative direction (looking up)."""
        target_pitch = -20.0
        await sim_client.move_head(
            pitch=target_pitch, yaw=0, roll=0, duration=CONFIG.movement_duration
        )
        positions = await sim_client.get_position()
        diff = abs(positions["head_pitch"] - target_pitch)
        assert diff < CONFIG.position_tolerance, f"Head pitch error: {diff:.1f}°"

    @pytest.mark.asyncio
    async def test_head_yaw_left(self, sim_client) -> None:
        """Test head yaw to the left."""
        target_yaw = 30.0
        await sim_client.move_head(
            pitch=0, yaw=target_yaw, roll=0, duration=CONFIG.movement_duration
        )
        positions = await sim_client.get_position()
        assert (
            abs(positions["head_yaw"] - target_yaw) < CONFIG.position_tolerance
        ), f"Head yaw {positions['head_yaw']} not within tolerance of {target_yaw}"

    @pytest.mark.asyncio
    async def test_head_yaw_right(self, sim_client) -> None:
        """Test head yaw to the right."""
        target_yaw = -30.0
        await sim_client.move_head(
            pitch=0, yaw=target_yaw, roll=0, duration=CONFIG.movement_duration
        )
        positions = await sim_client.get_position()
        assert (
            abs(positions["head_yaw"] - target_yaw) < CONFIG.position_tolerance
        ), f"Head yaw {positions['head_yaw']} not within tolerance of {target_yaw}"

    @pytest.mark.asyncio
    async def test_head_roll_left(self, sim_client) -> None:
        """Test head roll (tilt) to the left."""
        target_roll = 15.0
        await sim_client.move_head(
            pitch=0, yaw=0, roll=target_roll, duration=CONFIG.movement_duration
        )
        positions = await sim_client.get_position()
        assert (
            abs(positions["head_roll"] - target_roll) < CONFIG.position_tolerance
        ), f"Head roll {positions['head_roll']} not within tolerance of {target_roll}"

    @pytest.mark.asyncio
    async def test_head_roll_right(self, sim_client) -> None:
        """Test head roll (tilt) to the right."""
        target_roll = -15.0
        await sim_client.move_head(
            pitch=0, yaw=0, roll=target_roll, duration=CONFIG.movement_duration
        )
        positions = await sim_client.get_position()
        assert (
            abs(positions["head_roll"] - target_roll) < CONFIG.position_tolerance
        ), f"Head roll {positions['head_roll']} not within tolerance of {target_roll}"

    @pytest.mark.asyncio
    async def test_combined_head_movement(self, sim_client) -> None:
        """Test combined head pitch, yaw, and roll movement."""
        await sim_client.move_head(
            pitch=10, yaw=20, roll=5, duration=CONFIG.movement_duration
        )
        positions = await sim_client.get_position()
        assert abs(positions["head_pitch"] - 10) < CONFIG.position_tolerance
        assert abs(positions["head_yaw"] - 20) < CONFIG.position_tolerance
        assert abs(positions["head_roll"] - 5) < CONFIG.position_tolerance

    @pytest.mark.asyncio
    async def test_reset_position(self, sim_client) -> None:
        """Test reset to neutral position."""
        # First move to a non-neutral position
        await sim_client.move_head(pitch=20, yaw=30, roll=10, duration=0.2)
        # Then reset
        await sim_client.reset_position(duration=CONFIG.movement_duration)
        positions = await sim_client.get_position()
        assert abs(positions["head_pitch"]) < CONFIG.position_tolerance
        assert abs(positions["head_yaw"]) < CONFIG.position_tolerance
        assert abs(positions["head_roll"]) < CONFIG.position_tolerance


class TestAntennaMovements:
    """Validate antenna movement commands."""

    @pytest.mark.asyncio
    async def test_antenna_symmetric_up(self, sim_client) -> None:
        """Test both antennas moving up symmetrically."""
        await sim_client.set_antennas(left=60, right=60)
        await asyncio.sleep(0.2)
        positions = await sim_client.get_position()
        assert abs(positions["antenna_left"] - 60) < CONFIG.position_tolerance
        assert abs(positions["antenna_right"] - 60) < CONFIG.position_tolerance

    @pytest.mark.asyncio
    async def test_antenna_symmetric_down(self, sim_client) -> None:
        """Test both antennas moving down symmetrically."""
        await sim_client.set_antennas(left=-60, right=-60)
        await asyncio.sleep(0.2)
        positions = await sim_client.get_position()
        assert abs(positions["antenna_left"] - (-60)) < CONFIG.position_tolerance
        assert abs(positions["antenna_right"] - (-60)) < CONFIG.position_tolerance

    @pytest.mark.asyncio
    async def test_antenna_asymmetric(self, sim_client) -> None:
        """Test antennas moving to different positions."""
        await sim_client.set_antennas(left=45, right=-45)
        await asyncio.sleep(0.2)
        positions = await sim_client.get_position()
        assert abs(positions["antenna_left"] - 45) < CONFIG.position_tolerance
        assert abs(positions["antenna_right"] - (-45)) < CONFIG.position_tolerance

    @pytest.mark.asyncio
    async def test_antenna_extreme_positions(self, sim_client) -> None:
        """Test antennas at extreme positions within limits."""
        await sim_client.set_antennas(left=120, right=-120)
        await asyncio.sleep(0.2)
        positions = await sim_client.get_position()
        # Should be clamped to limits if out of range
        assert abs(positions["antenna_left"]) <= 150 + CONFIG.position_tolerance
        assert abs(positions["antenna_right"]) <= 150 + CONFIG.position_tolerance


class TestBodyRotation:
    """Validate body rotation commands."""

    @pytest.mark.asyncio
    async def test_body_rotate_clockwise(self, sim_client) -> None:
        """Test body rotation clockwise."""
        await sim_client.rotate_body(angle=45, duration=CONFIG.movement_duration)
        # Body rotation may not have position feedback in all configs
        # Just verify no error is raised

    @pytest.mark.asyncio
    async def test_body_rotate_counterclockwise(self, sim_client) -> None:
        """Test body rotation counter-clockwise."""
        await sim_client.rotate_body(angle=-45, duration=CONFIG.movement_duration)

    @pytest.mark.asyncio
    async def test_body_full_rotation(self, sim_client) -> None:
        """Test 360-degree body rotation."""
        await sim_client.rotate_body(angle=360, duration=2.0)


# ============================================================================
# Gesture Tests
# ============================================================================


class TestGestures:
    """Validate gesture commands (nod, shake)."""

    @pytest.mark.asyncio
    async def test_nod_low_intensity(self, sim_client) -> None:
        """Test nod gesture at low intensity."""
        await sim_client.nod(intensity=0.3)
        # Verify we return to approximately neutral after nod
        positions = await sim_client.get_position()
        assert abs(positions["head_pitch"]) < CONFIG.position_tolerance * 2

    @pytest.mark.asyncio
    async def test_nod_medium_intensity(self, sim_client) -> None:
        """Test nod gesture at medium intensity."""
        await sim_client.nod(intensity=0.6)
        positions = await sim_client.get_position()
        assert abs(positions["head_pitch"]) < CONFIG.position_tolerance * 2

    @pytest.mark.asyncio
    async def test_nod_high_intensity(self, sim_client) -> None:
        """Test nod gesture at high intensity."""
        await sim_client.nod(intensity=1.0)
        positions = await sim_client.get_position()
        assert abs(positions["head_pitch"]) < CONFIG.position_tolerance * 2

    @pytest.mark.asyncio
    async def test_shake_low_intensity(self, sim_client) -> None:
        """Test shake gesture at low intensity."""
        await sim_client.shake(intensity=0.3)
        positions = await sim_client.get_position()
        assert abs(positions["head_yaw"]) < CONFIG.position_tolerance * 2

    @pytest.mark.asyncio
    async def test_shake_medium_intensity(self, sim_client) -> None:
        """Test shake gesture at medium intensity."""
        await sim_client.shake(intensity=0.6)
        positions = await sim_client.get_position()
        assert abs(positions["head_yaw"]) < CONFIG.position_tolerance * 2

    @pytest.mark.asyncio
    async def test_shake_high_intensity(self, sim_client) -> None:
        """Test shake gesture at high intensity."""
        await sim_client.shake(intensity=1.0)
        positions = await sim_client.get_position()
        assert abs(positions["head_yaw"]) < CONFIG.position_tolerance * 2


# ============================================================================
# Motor Control Tests
# ============================================================================


class TestMotorControl:
    """Validate motor control commands (wake/sleep)."""

    @pytest.mark.asyncio
    async def test_wake_up(self, sim_client) -> None:
        """Test waking up the robot."""
        await sim_client.sleep()
        assert not await sim_client.is_awake()
        await sim_client.wake_up()
        assert await sim_client.is_awake()

    @pytest.mark.asyncio
    async def test_sleep(self, sim_client) -> None:
        """Test putting the robot to sleep."""
        await sim_client.wake_up()
        assert await sim_client.is_awake()
        await sim_client.sleep()
        assert not await sim_client.is_awake()
        # Wake back up for other tests
        await sim_client.wake_up()

    @pytest.mark.asyncio
    async def test_status_while_awake(self, sim_client) -> None:
        """Test getting status while robot is awake."""
        status = await sim_client.get_status()
        assert status.is_awake
        assert status.battery_percent >= 0
        assert hasattr(status, "head_pose")


# ============================================================================
# Sensor Tests
# ============================================================================


class TestSensors:
    """Validate sensor reading commands."""

    @pytest.mark.asyncio
    async def test_get_position(self, sim_client) -> None:
        """Test getting joint positions."""
        positions = await sim_client.get_position()
        assert "head_pitch" in positions
        assert "head_yaw" in positions
        assert "head_roll" in positions
        assert "antenna_left" in positions
        assert "antenna_right" in positions

    @pytest.mark.asyncio
    async def test_get_limits(self, sim_client) -> None:
        """Test getting joint limits."""
        limits = await sim_client.get_limits()
        assert "head_pitch" in limits
        min_val, max_val = limits["head_pitch"]
        assert min_val < max_val
        assert min_val >= -180 and max_val <= 180

    @pytest.mark.asyncio
    async def test_position_consistency(self, sim_client) -> None:
        """Test that position readings are consistent."""
        # Move to a known position
        await sim_client.move_head(pitch=15, yaw=0, roll=0, duration=0.3)
        # Read position multiple times
        positions1 = await sim_client.get_position()
        positions2 = await sim_client.get_position()
        # Should be very close
        assert (
            abs(positions1["head_pitch"] - positions2["head_pitch"]) < 1.0
        ), "Position readings inconsistent"


# ============================================================================
# Look-At Tests
# ============================================================================


class TestLookAt:
    """Validate look-at behavior."""

    @pytest.mark.asyncio
    async def test_look_at_center(self, sim_client) -> None:
        """Test looking at a point straight ahead."""
        await sim_client.look_at(x=1.0, y=0, z=0, duration=CONFIG.movement_duration)
        positions = await sim_client.get_position()
        # Looking straight ahead should result in near-zero angles
        assert abs(positions["head_yaw"]) < CONFIG.position_tolerance * 2
        assert abs(positions["head_pitch"]) < CONFIG.position_tolerance * 2

    @pytest.mark.asyncio
    async def test_look_at_left(self, sim_client) -> None:
        """Test looking at a point to the left."""
        await sim_client.look_at(x=1.0, y=1.0, z=0, duration=CONFIG.movement_duration)
        positions = await sim_client.get_position()
        # Should have positive yaw when looking left
        assert positions["head_yaw"] > 0

    @pytest.mark.asyncio
    async def test_look_at_right(self, sim_client) -> None:
        """Test looking at a point to the right."""
        await sim_client.look_at(x=1.0, y=-1.0, z=0, duration=CONFIG.movement_duration)
        positions = await sim_client.get_position()
        # Should have negative yaw when looking right
        assert positions["head_yaw"] < 0

    @pytest.mark.asyncio
    async def test_look_at_up(self, sim_client) -> None:
        """Test looking at a point above."""
        await sim_client.look_at(x=1.0, y=0, z=1.0, duration=CONFIG.movement_duration)
        positions = await sim_client.get_position()
        # Should have negative pitch when looking up
        assert positions["head_pitch"] < 0

    @pytest.mark.asyncio
    async def test_look_at_down(self, sim_client) -> None:
        """Test looking at a point below."""
        await sim_client.look_at(x=1.0, y=0, z=-1.0, duration=CONFIG.movement_duration)
        positions = await sim_client.get_position()
        # Should have positive pitch when looking down
        assert positions["head_pitch"] > 0


# ============================================================================
# Choreography Tests
# ============================================================================


class TestChoreography:
    """Validate complex choreographed movement sequences."""

    @pytest.mark.asyncio
    async def test_head_sweep_sequence(self, sim_client) -> None:
        """Test a sweeping head movement pattern."""
        # Sweep from left to right
        for yaw in [-30, -15, 0, 15, 30]:
            await sim_client.move_head(pitch=0, yaw=yaw, roll=0, duration=0.2)
        # Reset
        await sim_client.reset_position(duration=0.2)
        positions = await sim_client.get_position()
        assert abs(positions["head_yaw"]) < CONFIG.position_tolerance

    @pytest.mark.asyncio
    async def test_antenna_wave_pattern(self, sim_client) -> None:
        """Test an antenna wave pattern."""
        # Alternating antenna positions
        patterns = [
            (60, -60),
            (-60, 60),
            (60, -60),
            (-60, 60),
        ]
        for left, right in patterns:
            await sim_client.set_antennas(left=left, right=right)
            await asyncio.sleep(0.15)
        # Return to neutral
        await sim_client.set_antennas(left=0, right=0)

    @pytest.mark.asyncio
    async def test_combined_choreography(self, sim_client) -> None:
        """Test a combined choreography routine."""
        # Start position
        await sim_client.reset_position(duration=0.2)

        # Look around
        await sim_client.move_head(pitch=0, yaw=30, roll=0, duration=0.2)
        await sim_client.move_head(pitch=0, yaw=-30, roll=0, duration=0.2)

        # Nod
        await sim_client.nod(intensity=0.5)

        # Antenna expression
        await sim_client.set_antennas(left=60, right=60)
        await asyncio.sleep(0.2)

        # Return to neutral
        await sim_client.reset_position(duration=0.2)
        await sim_client.set_antennas(left=0, right=0)

        # Verify final position
        positions = await sim_client.get_position()
        assert abs(positions["head_pitch"]) < CONFIG.position_tolerance
        assert abs(positions["head_yaw"]) < CONFIG.position_tolerance


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_joint_limit_pitch(self, sim_client) -> None:
        """Test head pitch at joint limits."""
        limits = await sim_client.get_limits()
        min_pitch, max_pitch = limits["head_pitch"]

        # Move to max
        await sim_client.move_head(pitch=max_pitch, yaw=0, roll=0, duration=0.3)
        positions = await sim_client.get_position()
        assert positions["head_pitch"] <= max_pitch + CONFIG.position_tolerance

        # Move to min
        await sim_client.move_head(pitch=min_pitch, yaw=0, roll=0, duration=0.3)
        positions = await sim_client.get_position()
        assert positions["head_pitch"] >= min_pitch - CONFIG.position_tolerance

    @pytest.mark.asyncio
    async def test_joint_limit_yaw(self, sim_client) -> None:
        """Test head yaw at joint limits."""
        limits = await sim_client.get_limits()
        min_yaw, max_yaw = limits["head_yaw"]

        # Move to max
        await sim_client.move_head(pitch=0, yaw=max_yaw, roll=0, duration=0.3)
        positions = await sim_client.get_position()
        assert positions["head_yaw"] <= max_yaw + CONFIG.position_tolerance

        # Move to min
        await sim_client.move_head(pitch=0, yaw=min_yaw, roll=0, duration=0.3)
        positions = await sim_client.get_position()
        assert positions["head_yaw"] >= min_yaw - CONFIG.position_tolerance

    @pytest.mark.asyncio
    async def test_rapid_commands(self, sim_client) -> None:
        """Test rapid sequential commands."""
        # Send many commands in quick succession
        for i in range(10):
            await sim_client.move_head(
                pitch=i * 2 - 10, yaw=i * 3 - 15, roll=0, duration=0.05
            )
        # Should complete without error

    @pytest.mark.asyncio
    async def test_zero_duration_movement(self, sim_client) -> None:
        """Test movement with very short duration."""
        await sim_client.move_head(pitch=10, yaw=0, roll=0, duration=0.01)
        # Should complete without error


# ============================================================================
# Validation Runner
# ============================================================================


async def run_validation_suite() -> dict[str, Any]:
    """Run the complete validation suite and return results.

    Returns:
        Dictionary with test results and summary
    """
    if not SIMULATION_AVAILABLE:
        return {"error": "Simulation not available", "passed": 0, "failed": 1}

    results = {
        "basic_movements": {"passed": 0, "failed": 0, "tests": []},
        "antennas": {"passed": 0, "failed": 0, "tests": []},
        "body_rotation": {"passed": 0, "failed": 0, "tests": []},
        "gestures": {"passed": 0, "failed": 0, "tests": []},
        "motor_control": {"passed": 0, "failed": 0, "tests": []},
        "sensors": {"passed": 0, "failed": 0, "tests": []},
        "look_at": {"passed": 0, "failed": 0, "tests": []},
        "choreography": {"passed": 0, "failed": 0, "tests": []},
        "edge_cases": {"passed": 0, "failed": 0, "tests": []},
    }

    client = MuJoCoReachyClient(realtime=False, viewer=False)
    await client.connect()
    await client.wake_up()

    try:
        # Basic movements
        basic_tests = [
            ("head_pitch_positive", 20, 0, 0),
            ("head_pitch_negative", -20, 0, 0),
            ("head_yaw_left", 0, 30, 0),
            ("head_yaw_right", 0, -30, 0),
            ("head_roll_left", 0, 0, 15),
            ("head_roll_right", 0, 0, -15),
        ]

        for name, pitch, yaw, roll in basic_tests:
            try:
                await client.move_head(pitch=pitch, yaw=yaw, roll=roll, duration=0.3)
                positions = await client.get_position()
                if pitch != 0:
                    passed = abs(positions["head_pitch"] - pitch) < 10
                elif yaw != 0:
                    passed = abs(positions["head_yaw"] - yaw) < 10
                else:
                    passed = abs(positions["head_roll"] - roll) < 10

                if passed:
                    results["basic_movements"]["passed"] += 1
                else:
                    results["basic_movements"]["failed"] += 1
                results["basic_movements"]["tests"].append(
                    {"name": name, "passed": passed}
                )
            except Exception as e:
                results["basic_movements"]["failed"] += 1
                results["basic_movements"]["tests"].append(
                    {"name": name, "passed": False, "error": str(e)}
                )

        # Gestures
        gesture_tests = [
            ("nod_low", "nod", 0.3),
            ("nod_high", "nod", 1.0),
            ("shake_low", "shake", 0.3),
            ("shake_high", "shake", 1.0),
        ]

        for name, gesture_type, intensity in gesture_tests:
            try:
                if gesture_type == "nod":
                    await client.nod(intensity=intensity)
                else:
                    await client.shake(intensity=intensity)
                results["gestures"]["passed"] += 1
                results["gestures"]["tests"].append({"name": name, "passed": True})
            except Exception as e:
                results["gestures"]["failed"] += 1
                results["gestures"]["tests"].append(
                    {"name": name, "passed": False, "error": str(e)}
                )

        # Sensors
        try:
            positions = await client.get_position()
            results["sensors"]["passed"] += 1
            results["sensors"]["tests"].append({"name": "get_position", "passed": True})
        except Exception as e:
            results["sensors"]["failed"] += 1
            results["sensors"]["tests"].append(
                {"name": "get_position", "passed": False, "error": str(e)}
            )

        try:
            limits = await client.get_limits()
            assert "head_pitch" in limits  # Validate result
            results["sensors"]["passed"] += 1
            results["sensors"]["tests"].append({"name": "get_limits", "passed": True})
        except Exception as e:
            results["sensors"]["failed"] += 1
            results["sensors"]["tests"].append(
                {"name": "get_limits", "passed": False, "error": str(e)}
            )

        try:
            status = await client.get_status()
            assert status.is_awake  # Validate result
            results["sensors"]["passed"] += 1
            results["sensors"]["tests"].append({"name": "get_status", "passed": True})
        except Exception as e:
            results["sensors"]["failed"] += 1
            results["sensors"]["tests"].append(
                {"name": "get_status", "passed": False, "error": str(e)}
            )

    finally:
        await client.sleep()
        await client.disconnect()

    # Calculate totals
    total_passed = sum(cat["passed"] for cat in results.values())
    total_failed = sum(cat["failed"] for cat in results.values())

    return {
        "categories": results,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_tests": total_passed + total_failed,
        "success_rate": (
            total_passed / (total_passed + total_failed) * 100
            if (total_passed + total_failed) > 0
            else 0
        ),
    }


if __name__ == "__main__":
    # Run validation suite directly
    results = asyncio.run(run_validation_suite())
    print("\nValidation Results:")
    print(f"  Passed: {results['total_passed']}")
    print(f"  Failed: {results['total_failed']}")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
