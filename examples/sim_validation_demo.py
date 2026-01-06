#!/usr/bin/env python3
"""Visual validation demo for Reachy Mini in MuJoCo simulation.

This demo runs a comprehensive visual showcase of all robot capabilities
with the MuJoCo viewer enabled for visual feedback.

Usage:
    uv run python examples/sim_validation_demo.py
    uv run python examples/sim_validation_demo.py --no-viewer  # Headless mode
    uv run python examples/sim_validation_demo.py --record     # Record video

Features demonstrated:
    - Basic head movements (pitch, yaw, roll)
    - Body rotation
    - Antenna expressions
    - Gestures (nod, shake)
    - Look-at behavior
    - Emotion expressions
    - Choreographed dance sequence
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime

# Rich for beautiful console output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Simulation imports
try:
    from reachy_agent.simulation.client import MuJoCoReachyClient

    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


@dataclass
class TestResult:
    """Result of a validation test."""

    name: str
    category: str
    passed: bool
    duration: float
    error: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report."""

    tests: list[TestResult]
    start_time: datetime
    end_time: datetime
    recording_path: str | None = None

    @property
    def total_passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def total_failed(self) -> int:
        return sum(1 for t in self.tests if not t.passed)

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.total_passed / self.total_tests * 100

    @property
    def duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    def categories(self) -> dict[str, list[TestResult]]:
        """Group tests by category."""
        cats: dict[str, list[TestResult]] = {}
        for test in self.tests:
            if test.category not in cats:
                cats[test.category] = []
            cats[test.category].append(test)
        return cats


def print_header() -> None:
    """Print validation demo header."""
    if RICH_AVAILABLE:
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]Reachy Mini MuJoCo Validation Demo[/]\n"
                "[dim]Comprehensive robot command validation with visual feedback[/]",
                border_style="cyan",
            )
        )
        console.print()
    else:
        print("\n" + "=" * 60)
        print("Reachy Mini MuJoCo Validation Demo")
        print("=" * 60 + "\n")


def print_status(message: str, status: str = "info") -> None:
    """Print a status message."""
    if RICH_AVAILABLE:
        if status == "success":
            console.print(f"  [green]✓[/] {message}")
        elif status == "error":
            console.print(f"  [red]✗[/] {message}")
        elif status == "info":
            console.print(f"  [blue]→[/] {message}")
        else:
            console.print(f"    {message}")
    else:
        prefix = {"success": "✓", "error": "✗", "info": "→"}.get(status, " ")
        print(f"  {prefix} {message}")


def print_section(title: str) -> None:
    """Print a section header."""
    if RICH_AVAILABLE:
        console.print()
        console.print(f"[bold yellow]▶ {title}[/]")
        console.print("[dim]" + "─" * 50 + "[/]")
    else:
        print(f"\n▶ {title}")
        print("-" * 50)


def print_report(report: ValidationReport) -> None:
    """Print the validation report."""
    if RICH_AVAILABLE:
        # Create results table
        table = Table(title="Test Results by Category", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Tests", justify="center")
        table.add_column("Passed", justify="center", style="green")
        table.add_column("Failed", justify="center", style="red")
        table.add_column("Status", justify="center")

        for category, tests in report.categories().items():
            passed = sum(1 for t in tests if t.passed)
            failed = sum(1 for t in tests if not t.passed)
            status = "[green]✓ PASS[/]" if failed == 0 else "[red]✗ FAIL[/]"
            table.add_row(
                category.replace("_", " ").title(),
                str(len(tests)),
                str(passed),
                str(failed),
                status,
            )

        console.print()
        console.print(table)

        # Summary panel
        summary_text = (
            f"[bold]Total Tests:[/] {report.total_tests}\n"
            f"[green]Passed:[/] {report.total_passed}\n"
            f"[red]Failed:[/] {report.total_failed}\n"
            f"[bold]Success Rate:[/] {report.success_rate:.1f}%\n"
            f"[dim]Duration:[/] {report.duration:.1f}s"
        )
        if report.recording_path:
            summary_text += f"\n[dim]Recording:[/] {report.recording_path}"

        console.print()
        console.print(
            Panel(
                summary_text,
                title="[bold]Validation Summary[/]",
                border_style="green" if report.total_failed == 0 else "red",
            )
        )
    else:
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        for category, tests in report.categories().items():
            passed = sum(1 for t in tests if t.passed)
            failed = sum(1 for t in tests if not t.passed)
            status = "PASS" if failed == 0 else "FAIL"
            name = category.replace("_", " ").title()
            print(f"  {name}: {passed}/{len(tests)} [{status}]")
        print("-" * 60)
        print(f"Total: {report.total_passed}/{report.total_tests} passed")
        print(f"Success Rate: {report.success_rate:.1f}%")
        print(f"Duration: {report.duration:.1f}s")


class ValidationDemo:
    """Validation demo runner."""

    def __init__(
        self,
        viewer: bool = True,
        realtime: bool = True,
        record: bool = False,
    ) -> None:
        """Initialize validation demo.

        Args:
            viewer: Enable MuJoCo viewer
            realtime: Run in real-time mode
            record: Record video of validation
        """
        self.viewer = viewer
        self.realtime = realtime
        self.record = record
        self.client: MuJoCoReachyClient | None = None
        self.tests: list[TestResult] = []
        self.start_time: datetime | None = None
        self.recording_path: str | None = None

    async def setup(self) -> None:
        """Set up the simulation client."""
        print_status("Creating MuJoCo simulation client...")
        self.client = MuJoCoReachyClient(
            realtime=self.realtime,
            viewer=self.viewer,
        )
        await self.client.connect()
        print_status("Client connected", "success")

    async def teardown(self) -> None:
        """Clean up the simulation client."""
        if self.client:
            await self.client.sleep()
            await self.client.disconnect()
            print_status("Client disconnected", "success")

    async def run_test(
        self,
        name: str,
        category: str,
        test_fn,
    ) -> TestResult:
        """Run a single test and record result.

        Args:
            name: Test name
            category: Test category
            test_fn: Async test function

        Returns:
            TestResult with pass/fail and timing
        """
        start = time.perf_counter()
        try:
            await test_fn()
            duration = time.perf_counter() - start
            result = TestResult(
                name=name,
                category=category,
                passed=True,
                duration=duration,
            )
            print_status(f"{name} ({duration:.2f}s)", "success")
        except Exception as e:
            duration = time.perf_counter() - start
            result = TestResult(
                name=name,
                category=category,
                passed=False,
                duration=duration,
                error=str(e),
            )
            print_status(f"{name}: {e}", "error")

        self.tests.append(result)
        return result

    # ========================================================================
    # Test Implementations
    # ========================================================================

    async def test_startup_sequence(self) -> None:
        """Test startup sequence: wake up and reset to neutral."""
        print_section("Startup Sequence")

        async def wake_up():
            await self.client.wake_up()
            assert await self.client.is_awake()

        async def reset_position():
            await self.client.reset_position(duration=1.0)
            positions = await self.client.get_position()
            assert abs(positions["head_pitch"]) < 5
            assert abs(positions["head_yaw"]) < 5

        await self.run_test("Wake up robot", "startup", wake_up)
        await self.run_test("Reset to neutral", "startup", reset_position)

    async def test_basic_movements(self) -> None:
        """Test basic head movements."""
        print_section("Basic Movements")

        # Head pitch sweep
        async def pitch_sweep():
            for pitch in range(-30, 35, 10):
                await self.client.move_head(pitch=pitch, yaw=0, roll=0, duration=0.3)
            await self.client.reset_position(duration=0.3)

        await self.run_test(
            "Head pitch sweep (-30° to +30°)", "basic_movements", pitch_sweep
        )

        # Head yaw sweep
        async def yaw_sweep():
            for yaw in range(-60, 65, 15):
                await self.client.move_head(pitch=0, yaw=yaw, roll=0, duration=0.3)
            await self.client.reset_position(duration=0.3)

        await self.run_test(
            "Head yaw sweep (-60° to +60°)", "basic_movements", yaw_sweep
        )

        # Head roll sweep
        async def roll_sweep():
            for roll in range(-30, 35, 10):
                await self.client.move_head(pitch=0, yaw=0, roll=roll, duration=0.3)
            await self.client.reset_position(duration=0.3)

        await self.run_test(
            "Head roll sweep (-30° to +30°)", "basic_movements", roll_sweep
        )

        # Body rotation
        async def body_spin():
            await self.client.rotate_body(angle=360, duration=3.0)

        await self.run_test("Body rotation (360° spin)", "basic_movements", body_spin)

        # Antenna wiggle
        async def antenna_wiggle():
            patterns = [(60, -60), (-60, 60), (60, -60), (-60, 60), (0, 0)]
            for left, right in patterns:
                await self.client.set_antennas(left=left, right=right)
                await asyncio.sleep(0.3)

        await self.run_test("Antenna wiggle pattern", "basic_movements", antenna_wiggle)

    async def test_gestures(self) -> None:
        """Test gesture commands."""
        print_section("Gesture Showcase")

        # Nods at different intensities
        async def nod_low():
            await self.client.nod(intensity=0.3)
            await asyncio.sleep(0.3)

        async def nod_medium():
            await self.client.nod(intensity=0.6)
            await asyncio.sleep(0.3)

        async def nod_high():
            await self.client.nod(intensity=1.0)
            await asyncio.sleep(0.3)

        await self.run_test("Nod (low intensity)", "gestures", nod_low)
        await self.run_test("Nod (medium intensity)", "gestures", nod_medium)
        await self.run_test("Nod (high intensity)", "gestures", nod_high)

        # Shakes at different intensities
        async def shake_low():
            await self.client.shake(intensity=0.3)
            await asyncio.sleep(0.3)

        async def shake_medium():
            await self.client.shake(intensity=0.6)
            await asyncio.sleep(0.3)

        async def shake_high():
            await self.client.shake(intensity=1.0)
            await asyncio.sleep(0.3)

        await self.run_test("Shake (low intensity)", "gestures", shake_low)
        await self.run_test("Shake (medium intensity)", "gestures", shake_medium)
        await self.run_test("Shake (high intensity)", "gestures", shake_high)

    async def test_emotions(self) -> None:
        """Test emotion expressions through head and antenna movements."""
        print_section("Emotion Library Demo")

        # Happy - perky antennas, slight head tilt
        async def emotion_happy():
            await self.client.set_antennas(left=60, right=60)
            await self.client.move_head(pitch=-5, yaw=0, roll=5, duration=0.5)
            await asyncio.sleep(0.5)
            await self.client.reset_position(duration=0.3)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Emotion: Happy", "emotions", emotion_happy)

        # Sad - droopy antennas, head down
        async def emotion_sad():
            await self.client.set_antennas(left=-60, right=-60)
            await self.client.move_head(pitch=20, yaw=0, roll=0, duration=0.5)
            await asyncio.sleep(0.5)
            await self.client.reset_position(duration=0.3)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Emotion: Sad", "emotions", emotion_sad)

        # Surprised - antennas up, head back
        async def emotion_surprised():
            await self.client.set_antennas(left=90, right=90)
            await self.client.move_head(pitch=-15, yaw=0, roll=0, duration=0.3)
            await asyncio.sleep(0.5)
            await self.client.reset_position(duration=0.3)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Emotion: Surprised", "emotions", emotion_surprised)

        # Curious - head tilt, one antenna up
        async def emotion_curious():
            await self.client.set_antennas(left=45, right=-15)
            await self.client.move_head(pitch=5, yaw=15, roll=10, duration=0.5)
            await asyncio.sleep(0.5)
            await self.client.reset_position(duration=0.3)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Emotion: Curious", "emotions", emotion_curious)

        # Confused - alternating antennas, head tilt
        async def emotion_confused():
            await self.client.move_head(pitch=0, yaw=0, roll=-15, duration=0.3)
            await self.client.set_antennas(left=30, right=-30)
            await asyncio.sleep(0.3)
            await self.client.set_antennas(left=-30, right=30)
            await asyncio.sleep(0.3)
            await self.client.reset_position(duration=0.3)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Emotion: Confused", "emotions", emotion_confused)

        # Excited - rapid movements
        async def emotion_excited():
            for _ in range(3):
                await self.client.set_antennas(left=60, right=60)
                await self.client.move_head(pitch=-10, yaw=0, roll=0, duration=0.1)
                await asyncio.sleep(0.1)
                await self.client.set_antennas(left=30, right=30)
                await self.client.move_head(pitch=0, yaw=0, roll=0, duration=0.1)
                await asyncio.sleep(0.1)
            await self.client.reset_position(duration=0.3)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Emotion: Excited", "emotions", emotion_excited)

        # Sleepy - slow movements, droopy
        async def emotion_sleepy():
            await self.client.set_antennas(left=-30, right=-30)
            await self.client.move_head(pitch=15, yaw=0, roll=5, duration=1.0)
            await asyncio.sleep(0.5)
            await self.client.move_head(pitch=20, yaw=5, roll=8, duration=0.5)
            await asyncio.sleep(0.5)
            await self.client.reset_position(duration=0.5)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Emotion: Sleepy", "emotions", emotion_sleepy)

        # Attentive - alert posture
        async def emotion_attentive():
            await self.client.set_antennas(left=45, right=45)
            await self.client.move_head(pitch=-10, yaw=0, roll=0, duration=0.3)
            await asyncio.sleep(0.5)
            await self.client.reset_position(duration=0.3)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Emotion: Attentive", "emotions", emotion_attentive)

    async def test_choreography(self) -> None:
        """Test choreographed dance sequence."""
        print_section("Choreographed Dance Sequence")

        async def dance_routine():
            # Introduction - wake up and look around
            await self.client.reset_position(duration=0.5)
            await asyncio.sleep(0.3)

            # Look around curiously
            await self.client.move_head(pitch=0, yaw=40, roll=0, duration=0.5)
            await asyncio.sleep(0.2)
            await self.client.move_head(pitch=0, yaw=-40, roll=0, duration=0.7)
            await asyncio.sleep(0.2)
            await self.client.move_head(pitch=0, yaw=0, roll=0, duration=0.4)

            # Excited antenna wave
            for _ in range(2):
                await self.client.set_antennas(left=60, right=-60)
                await asyncio.sleep(0.2)
                await self.client.set_antennas(left=-60, right=60)
                await asyncio.sleep(0.2)
            await self.client.set_antennas(left=0, right=0)

            # Body spin with head tracking
            await self.client.move_head(pitch=-10, yaw=0, roll=0, duration=0.3)
            await self.client.rotate_body(angle=180, duration=1.5)
            await self.client.rotate_body(angle=-180, duration=1.5)
            await self.client.reset_position(duration=0.3)

            # Wave pattern - head figure 8
            for i in range(2):
                await self.client.move_head(pitch=-15, yaw=30, roll=10, duration=0.4)
                await self.client.move_head(pitch=10, yaw=0, roll=-10, duration=0.4)
                await self.client.move_head(pitch=-15, yaw=-30, roll=-10, duration=0.4)
                await self.client.move_head(pitch=10, yaw=0, roll=10, duration=0.4)

            # Happy finale
            await self.client.set_antennas(left=60, right=60)
            await self.client.nod(intensity=0.8)
            await asyncio.sleep(0.3)
            await self.client.reset_position(duration=0.5)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("30-second dance routine", "choreography", dance_routine)

    async def test_look_at(self) -> None:
        """Test look-at behavior."""
        print_section("Look-At Behavior")

        # Look at various points
        points = [
            ("center", 1.0, 0, 0),
            ("left", 1.0, 0.5, 0),
            ("right", 1.0, -0.5, 0),
            ("up", 1.0, 0, 0.5),
            ("down", 1.0, 0, -0.5),
        ]

        for name, x, y, z in points:

            async def look_at_point(x=x, y=y, z=z):
                await self.client.look_at(x=x, y=y, z=z, duration=0.5)
                await asyncio.sleep(0.3)

            await self.run_test(f"Look at {name}", "look_at", look_at_point)

        # Track a moving target
        async def track_target():
            # Simulate a target moving in a circle
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                x = 1.0
                y = 0.5 * math.sin(rad)
                z = 0.3 * math.cos(rad)
                await self.client.look_at(x=x, y=y, z=z, duration=0.15)
            await self.client.reset_position(duration=0.3)

        await self.run_test("Track circular target", "look_at", track_target)

    async def test_sensors(self) -> None:
        """Test sensor readings."""
        print_section("Sensor Validation")

        async def read_positions():
            positions = await self.client.get_position()
            assert "head_pitch" in positions
            assert "head_yaw" in positions
            assert "head_roll" in positions
            assert "antenna_left" in positions
            assert "antenna_right" in positions
            print_status(
                f"Positions: pitch={positions['head_pitch']:.1f}°, "
                f"yaw={positions['head_yaw']:.1f}°, roll={positions['head_roll']:.1f}°",
                "info",
            )

        await self.run_test("Read joint positions", "sensors", read_positions)

        async def read_limits():
            limits = await self.client.get_limits()
            assert "head_pitch" in limits
            pitch_min, pitch_max = limits["head_pitch"]
            print_status(f"Head pitch limits: [{pitch_min}°, {pitch_max}°]", "info")

        await self.run_test("Read joint limits", "sensors", read_limits)

        async def read_status():
            status = await self.client.get_status()
            assert status.is_awake
            print_status(
                f"Status: awake={status.is_awake}, battery={status.battery_percent}%",
                "info",
            )

        await self.run_test("Read robot status", "sensors", read_status)

    async def test_shutdown_sequence(self) -> None:
        """Test shutdown sequence."""
        print_section("Shutdown Sequence")

        async def return_to_neutral():
            await self.client.reset_position(duration=1.0)
            await self.client.set_antennas(left=0, right=0)

        await self.run_test("Return to neutral", "shutdown", return_to_neutral)

        async def sleep_mode():
            await self.client.sleep()
            assert not await self.client.is_awake()

        await self.run_test("Enter sleep mode", "shutdown", sleep_mode)

    async def run(self) -> ValidationReport:
        """Run the complete validation demo.

        Returns:
            ValidationReport with all test results
        """
        print_header()
        self.start_time = datetime.now()
        self.tests = []

        try:
            await self.setup()

            # Run all test sections
            await self.test_startup_sequence()
            await self.test_basic_movements()
            await self.test_gestures()
            await self.test_emotions()
            await self.test_choreography()
            await self.test_look_at()
            await self.test_sensors()
            await self.test_shutdown_sequence()

        except Exception as e:
            print_status(f"Fatal error: {e}", "error")
            self.tests.append(
                TestResult(
                    name="Fatal error",
                    category="system",
                    passed=False,
                    duration=0,
                    error=str(e),
                )
            )
        finally:
            try:
                await self.teardown()
            except Exception:
                pass

        end_time = datetime.now()
        report = ValidationReport(
            tests=self.tests,
            start_time=self.start_time,
            end_time=end_time,
            recording_path=self.recording_path,
        )

        print_report(report)
        return report


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Reachy Mini MuJoCo Validation Demo")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without MuJoCo viewer (headless mode)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run in fast-forward mode (not real-time)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record video of validation",
    )

    args = parser.parse_args()

    if not SIMULATION_AVAILABLE:
        print(
            "Error: Simulation not available. Install with: uv pip install -e '.[sim]'"
        )
        return

    demo = ValidationDemo(
        viewer=not args.no_viewer,
        realtime=not args.no_realtime,
        record=args.record,
    )

    report = asyncio.run(demo.run())

    # Exit with error code if any tests failed
    if report.total_failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
