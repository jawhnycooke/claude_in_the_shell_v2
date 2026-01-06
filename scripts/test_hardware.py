#!/usr/bin/env python3
"""Test robot hardware connectivity."""

import asyncio
import sys

import structlog
from rich.console import Console
from rich.table import Table

from reachy_agent.robot.sdk import SDKClient

console = Console()
log = structlog.get_logger()


async def test_connection() -> bool:
    """Test robot connection."""
    console.print("[cyan]Testing robot connection...[/cyan]")
    try:
        client = SDKClient()
        await client.connect()
        console.print("[green]✓ Connection successful[/green]")
        await client.disconnect()
        return True
    except Exception as e:
        console.print(f"[red]✗ Connection failed: {e}[/red]")
        return False


async def test_motors() -> bool:
    """Test motor control."""
    console.print("[cyan]Testing motor control...[/cyan]")
    try:
        client = SDKClient()
        await client.connect()
        await client.wake_up()
        console.print("[green]✓ Motors enabled[/green]")

        # TODO: Test movements
        # await client.move_head(...)

        await client.sleep()
        await client.disconnect()
        return True
    except Exception as e:
        console.print(f"[red]✗ Motor test failed: {e}[/red]")
        return False


async def test_sensors() -> bool:
    """Test sensor reading."""
    console.print("[cyan]Testing sensors...[/cyan]")
    try:
        client = SDKClient()
        await client.connect()

        # Test IMU
        sensor_data = await client.get_sensor_data()
        console.print(f"[green]✓ IMU data: {sensor_data}[/green]")

        # Test battery
        battery = await client.get_battery()
        console.print(f"[green]✓ Battery: {battery}%[/green]")

        await client.disconnect()
        return True
    except Exception as e:
        console.print(f"[red]✗ Sensor test failed: {e}[/red]")
        return False


async def main() -> None:
    """Run all hardware tests."""
    console.print("[bold cyan]Reachy Hardware Test Suite[/bold cyan]\n")

    results = []
    results.append(("Connection", await test_connection()))
    results.append(("Motors", await test_motors()))
    results.append(("Sensors", await test_sensors()))

    # Summary table
    table = Table(title="\nTest Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")

    for test_name, passed in results:
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        table.add_row(test_name, status)

    console.print(table)

    # Exit code
    all_passed = all(passed for _, passed in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
