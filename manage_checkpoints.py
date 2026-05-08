#!/usr/bin/env python3
"""
Command-line tool to manage checkpoints.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evaluation_utils.checkpoint_manager import CheckpointManager
from evaluation_utils.commons import GAME_DATA_DIR

app = typer.Typer(help="Manage agent checkpoints")
console = Console()


@app.command()
def list(
    game_name: str = typer.Option(
        None, "--game", "-g", help="Game name (for directory path, e.g. twenty_fourty_eight)"
    ),
    agent: str = typer.Option(None, "--agent", "-a", help="Filter by agent name"),
    run_id: str = typer.Option(None, "--run-id", help="Run ID"),
    checkpoint_dir: str = typer.Option(
        None, "--dir", "-d", help="Full checkpoint directory path (overrides game/run-id)"
    ),
):
    """List all available checkpoints."""
    # Determine checkpoint directory
    if checkpoint_dir is None:
        if game_name and run_id:
            checkpoint_dir = f"{GAME_DATA_DIR}/{game_name}/{run_id}/checkpoints"
        else:
            console.print("[red]Error: Either provide --dir or both --game and --run-id[/red]")
            raise typer.Exit(1)

    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    # Don't filter by game_name when listing since the directory already specifies the game
    checkpoints = manager.list_checkpoints(game_name=None, agent_name=agent)

    if not checkpoints:
        console.print("[yellow]No checkpoints found.[/yellow]")
        return

    table = Table(title="Available Checkpoints")
    table.add_column("Game", style="cyan")
    table.add_column("Agent", style="green")
    table.add_column("Timestamp", style="magenta")
    table.add_column("Episode", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Total Steps", justify="right")
    table.add_column("Path", style="dim")

    for cp in checkpoints:
        table.add_row(
            cp.get("game_name", "N/A"),
            cp.get("agent_name", "N/A"),
            cp.get("timestamp", "N/A"),
            str(cp.get("game_state", {}).get("episode", "N/A")),
            str(cp.get("game_state", {}).get("score", "N/A")),
            str(cp.get("metadata", {}).get("total_steps", "N/A")),
            str(cp.get("checkpoint_path", "N/A")),
        )

    console.print(table)
    console.print(f"\nTotal: {len(checkpoints)} checkpoints")


@app.command()
def info(
    checkpoint_path: str = typer.Argument(..., help="Path to checkpoint file"),
):
    """Show detailed information about a checkpoint."""
    manager = CheckpointManager()

    try:
        checkpoint = manager.load_checkpoint(Path(checkpoint_path))

        console.print("\n[bold cyan]Checkpoint Information[/bold cyan]")
        console.print(f"Game: [green]{checkpoint['game_name']}[/green]")
        console.print(f"Agent: [green]{checkpoint['agent_name']}[/green]")
        console.print(f"Timestamp: [magenta]{checkpoint['timestamp']}[/magenta]")

        console.print("\n[bold cyan]Game State[/bold cyan]")
        for key, value in checkpoint.get("game_state", {}).items():
            console.print(f"  {key}: {value}")

        console.print("\n[bold cyan]Metadata[/bold cyan]")
        for key, value in checkpoint.get("metadata", {}).items():
            console.print(f"  {key}: {value}")

        console.print("\n[bold cyan]Agent State Keys[/bold cyan]")
        agent_state = checkpoint.get("agent_state", {})
        for key in agent_state.keys():
            console.print(f"  • {key}")

    except Exception as e:
        console.print(f"[red]Error loading checkpoint: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def delete(
    checkpoint_path: str = typer.Argument(..., help="Path to checkpoint file"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a checkpoint."""
    manager = CheckpointManager()
    path = Path(checkpoint_path)

    if not yes:
        confirm = typer.confirm(f"Delete checkpoint {path}?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    try:
        manager.delete_checkpoint(path)
        console.print(f"[green]Deleted checkpoint: {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error deleting checkpoint: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def cleanup(
    game_name: str = typer.Argument(..., help="Game name (e.g. twenty_fourty_eight)"),
    agent_name: str = typer.Argument(..., help="Agent name"),
    keep: int = typer.Option(5, "--keep", "-k", help="Number of checkpoints to keep"),
    run_id: str = typer.Option(None, "--run-id", help="Run ID"),
    checkpoint_dir: str = typer.Option(
        None, "--dir", "-d", help="Full checkpoint directory path (overrides game/run-id)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clean up old checkpoints, keeping only the N most recent."""
    # Determine checkpoint directory
    if checkpoint_dir is None:
        if game_name and run_id:
            checkpoint_dir = f"{GAME_DATA_DIR}/{game_name}/{run_id}/checkpoints"
        else:
            console.print("[red]Error: Either provide --dir or both game_name and --run-id[/red]")
            raise typer.Exit(1)

    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

    if not yes:
        confirm = typer.confirm(
            f"Keep only the {keep} most recent checkpoints for {game_name}/{agent_name}?"
        )
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    try:
        manager.cleanup_old_checkpoints(agent_name, game_name=None, keep_last_n=keep)
        console.print(f"[green]Cleanup complete. Kept {keep} most recent checkpoints.[/green]")
    except Exception as e:
        console.print(f"[red]Error during cleanup: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
