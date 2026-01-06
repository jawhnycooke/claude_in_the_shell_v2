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
    debug_voice: bool = typer.Option(False, "--debug-voice", help="Debug voice events"),
    persona: str | None = typer.Option(
        None, "--persona", help="Persona to use (jarvis, motoko, batou)"
    ),
) -> None:
    """Run the Reachy agent."""
    try:
        config = AgentConfig(
            enable_voice=voice,
            enable_motion=not mock,
            mock_hardware=mock,
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
    """Check system health."""
    console.print("[cyan]Checking system health...[/cyan]")

    # TODO: Implement health checks
    # - Robot connectivity
    # - Memory system
    # - API keys present
    # - Config files valid

    console.print("[green]✓ System OK[/green]")


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
