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
def version() -> None:
    """Show version information."""
    from reachy_agent import __version__

    console.print(f"reachy-agent version {__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
