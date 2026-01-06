"""CLI entry point for Reachy Agent."""

import asyncio
import sys

import structlog
import typer
from rich.console import Console
from rich.panel import Panel

from reachy_agent.agent.loop import ReachyAgentLoop
from reachy_agent.agent.options import AgentConfig

app = typer.Typer(name="reachy-agent", help="Reachy Mini embodied AI agent")
console = Console()
log = structlog.get_logger()


@app.command()
def run(
    voice: bool = typer.Option(False, "--voice", help="Enable voice mode"),
    mock: bool = typer.Option(False, "--mock", help="Use mock hardware"),
    sim: bool = typer.Option(False, "--sim", help="Use MuJoCo simulation"),
    sim_viewer: bool = typer.Option(
        False, "--sim-viewer", help="Enable simulation viewer"
    ),
    sim_realtime: bool = typer.Option(
        True, "--sim-realtime/--no-sim-realtime", help="Run simulation in real-time"
    ),
    debug_voice: bool = typer.Option(False, "--debug-voice", help="Debug voice events"),
    persona: str | None = typer.Option(
        None, "--persona", help="Persona to use (jarvis, motoko, batou)"
    ),
) -> None:
    """Run the Reachy agent.

    Backend selection (mutually exclusive):
    - Default: Use real Reachy hardware via SDK
    - --mock: Use in-memory mock (no physics, fastest)
    - --sim: Use MuJoCo physics simulation

    Simulation options (only with --sim):
    - --sim-viewer: Open 3D visualization window
    - --no-sim-realtime: Run as fast as possible (for testing)
    """
    # Validate mutually exclusive options
    if mock and sim:
        console.print("[red]Error: Cannot use both --mock and --sim[/red]")
        sys.exit(1)

    try:
        config = AgentConfig(
            enable_voice=voice,
            enable_motion=not mock and not sim,
            mock_hardware=mock,
            simulation_mode=sim,
            simulation_viewer=sim_viewer,
            simulation_realtime=sim_realtime,
            persona_path=f"prompts/personas/{persona}.md" if persona else None,
        )
        asyncio.run(_run_agent(config, debug_voice))
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


async def _run_agent(config: AgentConfig, debug_voice: bool) -> None:
    """Run agent with given configuration."""
    agent = ReachyAgentLoop(config)

    try:
        await agent.start()

        if config.enable_voice:
            await _run_voice_mode(agent)
        else:
            await _run_text_mode(agent)

    finally:
        await agent.stop()


async def _run_text_mode(agent: ReachyAgentLoop) -> None:
    """Run agent in text REPL mode."""
    console.print(
        Panel(
            "🤖 [bold cyan]Reachy Agent[/bold cyan] (type 'quit' to exit)",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            response = await agent.process(user_input)
            console.print(f"\n[green]{response}[/green]")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    console.print("\n[yellow]Goodbye![/yellow]")


async def _run_voice_mode(agent: ReachyAgentLoop) -> None:
    """Run agent in voice mode."""
    console.print(
        Panel(
            "🎤 [bold cyan]Voice mode active[/bold cyan] (Ctrl+C to exit)",
            border_style="cyan",
        )
    )

    try:
        # Voice pipeline handles the interaction loop
        while agent._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass


@app.command()
def check() -> None:
    """Check system health and component availability."""
    from pathlib import Path

    console.print("[cyan]Checking system health...[/cyan]\n")

    checks_passed = 0
    checks_total = 0

    # Check config files
    checks_total += 1
    config_path = Path("config/default.yaml")
    if config_path.exists():
        console.print("[green]✓[/green] Configuration file found")
        checks_passed += 1
    else:
        console.print("[red]✗[/red] Configuration file missing")

    # Check prompts directory
    checks_total += 1
    prompts_path = Path("prompts")
    if prompts_path.exists():
        console.print("[green]✓[/green] Prompts directory found")
        checks_passed += 1
    else:
        console.print("[red]✗[/red] Prompts directory missing")

    # Check MuJoCo availability
    checks_total += 1
    try:
        import mujoco  # noqa: F401

        console.print("[green]✓[/green] MuJoCo available (simulation supported)")
        checks_passed += 1

        # Check MJCF model
        checks_total += 1
        model_path = Path("data/models/reachy_mini/reachy_mini.xml")
        if model_path.exists():
            console.print("[green]✓[/green] Reachy Mini MJCF model found")
            checks_passed += 1
        else:
            console.print("[yellow]![/yellow] Reachy Mini MJCF model missing")
    except ImportError:
        console.print(
            "[yellow]![/yellow] MuJoCo not installed (install with: pip install mujoco)"
        )

    # Check ChromaDB availability
    checks_total += 1
    try:
        import chromadb  # noqa: F401

        console.print("[green]✓[/green] ChromaDB available (memory supported)")
        checks_passed += 1
    except ImportError:
        console.print("[red]✗[/red] ChromaDB not installed")

    # Check Anthropic API key
    checks_total += 1
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[green]✓[/green] ANTHROPIC_API_KEY set")
        checks_passed += 1
    else:
        console.print("[yellow]![/yellow] ANTHROPIC_API_KEY not set")

    # Summary
    console.print(f"\n[cyan]Health check: {checks_passed}/{checks_total} passed[/cyan]")
    if checks_passed == checks_total:
        console.print("[green]✓ System OK[/green]")
    elif checks_passed >= checks_total - 2:
        console.print("[yellow]! System partially ready[/yellow]")
    else:
        console.print("[red]✗ System not ready[/red]")


@app.command()
def validate(
    viewer: bool = typer.Option(True, "--viewer/--no-viewer", help="Enable viewer"),
    realtime: bool = typer.Option(
        True, "--realtime/--no-realtime", help="Real-time mode"
    ),
    record: bool = typer.Option(False, "--record", help="Record validation video"),
    quick: bool = typer.Option(False, "--quick", help="Quick validation (fewer tests)"),
) -> None:
    """Run MuJoCo simulation validation suite.

    This command runs a comprehensive validation of all robot commands
    in MuJoCo simulation, with optional visual feedback and recording.

    Examples:
        reachy-agent validate              # Full validation with viewer
        reachy-agent validate --no-viewer  # Headless validation
        reachy-agent validate --record     # Record video of validation
        reachy-agent validate --quick      # Quick smoke test
    """
    from datetime import datetime

    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    # Check MuJoCo availability
    try:
        import mujoco  # noqa: F401
    except ImportError:
        console.print(
            "[red]Error: MuJoCo not installed.[/red]\n"
            "Install with: uv pip install -e '.[sim]'"
        )
        sys.exit(1)

    # Import validation demo
    try:
        from reachy_agent.simulation.client import MuJoCoReachyClient
    except ImportError as e:
        console.print(f"[red]Error: Could not load simulation: {e}[/red]")
        sys.exit(1)

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Reachy Mini MuJoCo Validation[/]\n"
            "[dim]Comprehensive robot command validation[/]",
            border_style="cyan",
        )
    )

    # Run validation
    async def run_validation() -> dict:
        """Run validation tests."""
        results = {
            "categories": {},
            "total_passed": 0,
            "total_failed": 0,
            "start_time": datetime.now(),
        }

        client = MuJoCoReachyClient(realtime=realtime, viewer=viewer)

        try:
            await client.connect()
            await client.wake_up()

            test_categories = {
                "Basic Movements": [
                    (
                        "Head pitch up",
                        lambda: client.move_head(
                            pitch=-20, yaw=0, roll=0, duration=0.3
                        ),
                    ),
                    (
                        "Head pitch down",
                        lambda: client.move_head(pitch=20, yaw=0, roll=0, duration=0.3),
                    ),
                    (
                        "Head yaw left",
                        lambda: client.move_head(pitch=0, yaw=30, roll=0, duration=0.3),
                    ),
                    (
                        "Head yaw right",
                        lambda: client.move_head(
                            pitch=0, yaw=-30, roll=0, duration=0.3
                        ),
                    ),
                    (
                        "Head roll",
                        lambda: client.move_head(pitch=0, yaw=0, roll=15, duration=0.3),
                    ),
                    ("Reset position", lambda: client.reset_position(duration=0.3)),
                ],
                "Antennas": [
                    ("Antennas up", lambda: client.set_antennas(left=60, right=60)),
                    ("Antennas down", lambda: client.set_antennas(left=-60, right=-60)),
                    (
                        "Antennas asymmetric",
                        lambda: client.set_antennas(left=45, right=-45),
                    ),
                    ("Antennas neutral", lambda: client.set_antennas(left=0, right=0)),
                ],
                "Gestures": [
                    ("Nod (low)", lambda: client.nod(intensity=0.3)),
                    ("Nod (high)", lambda: client.nod(intensity=1.0)),
                    ("Shake (low)", lambda: client.shake(intensity=0.3)),
                    ("Shake (high)", lambda: client.shake(intensity=1.0)),
                ],
                "Sensors": [
                    ("Get position", client.get_position),
                    ("Get limits", client.get_limits),
                    ("Get status", client.get_status),
                ],
            }

            if not quick:
                test_categories["Body Rotation"] = [
                    ("Rotate 90°", lambda: client.rotate_body(angle=90, duration=1.0)),
                    (
                        "Rotate -90°",
                        lambda: client.rotate_body(angle=-90, duration=1.0),
                    ),
                ]
                test_categories["Look At"] = [
                    (
                        "Look center",
                        lambda: client.look_at(x=1, y=0, z=0, duration=0.3),
                    ),
                    (
                        "Look left",
                        lambda: client.look_at(x=1, y=0.5, z=0, duration=0.3),
                    ),
                    (
                        "Look right",
                        lambda: client.look_at(x=1, y=-0.5, z=0, duration=0.3),
                    ),
                ]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for category, tests in test_categories.items():
                    results["categories"][category] = {
                        "passed": 0,
                        "failed": 0,
                        "tests": [],
                    }
                    task = progress.add_task(f"[cyan]{category}...", total=len(tests))

                    for test_name, test_fn in tests:
                        try:
                            await test_fn()
                            results["categories"][category]["passed"] += 1
                            results["total_passed"] += 1
                            results["categories"][category]["tests"].append(
                                {"name": test_name, "passed": True}
                            )
                        except Exception as e:
                            results["categories"][category]["failed"] += 1
                            results["total_failed"] += 1
                            results["categories"][category]["tests"].append(
                                {"name": test_name, "passed": False, "error": str(e)}
                            )
                        progress.advance(task)

                    await asyncio.sleep(0.1)  # Brief pause between categories

        finally:
            await client.sleep()
            await client.disconnect()

        results["end_time"] = datetime.now()
        return results

    try:
        results = asyncio.run(run_validation())
    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        sys.exit(1)

    # Display results
    console.print()

    # Results table
    table = Table(title="Validation Results", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Tests", justify="center")
    table.add_column("Passed", justify="center", style="green")
    table.add_column("Failed", justify="center", style="red")
    table.add_column("Status", justify="center")

    for category, data in results["categories"].items():
        total = data["passed"] + data["failed"]
        status = "[green]✓ PASS[/]" if data["failed"] == 0 else "[red]✗ FAIL[/]"
        table.add_row(
            category,
            str(total),
            str(data["passed"]),
            str(data["failed"]),
            status,
        )

    console.print(table)

    # Summary
    total = results["total_passed"] + results["total_failed"]
    duration = (results["end_time"] - results["start_time"]).total_seconds()
    success_rate = results["total_passed"] / total * 100 if total > 0 else 0

    console.print()
    summary_panel = Panel(
        f"[bold]Total Tests:[/] {total}\n"
        f"[green]Passed:[/] {results['total_passed']}\n"
        f"[red]Failed:[/] {results['total_failed']}\n"
        f"[bold]Success Rate:[/] {success_rate:.1f}%\n"
        f"[dim]Duration:[/] {duration:.1f}s",
        title="[bold]Validation Summary[/]",
        border_style="green" if results["total_failed"] == 0 else "red",
    )
    console.print(summary_panel)

    if results["total_failed"] > 0:
        sys.exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from reachy_agent import __version__

    console.print(f"reachy-agent version {__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
