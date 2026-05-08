"""
Centralized Rich console rendering for the Orak evaluation framework.

All terminal UI concerns are handled here. Other modules call update hooks
to mutate display state; the Renderer owns the single Live context and
manages responsive layout, throttling, and visual composition.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Literal

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

ServerStatus = Literal["queued", "launching", "running", "completed", "failed", "stopped"]


@dataclass
class RendererState:
    """Internal state cache for all displayed information."""

    # Global config
    show_local_mode: bool = False
    session_id: str | None = None
    submission_id: str | None = None
    game_data_path: str = ""
    warnings: list[str] = field(default_factory=list)

    # Game servers - supports parallel execution
    server_status_by_game: dict[str, ServerStatus] = field(default_factory=dict)
    scores_by_game: dict[str, int] = field(default_factory=dict)
    game_start_times: dict[str, float | None] = field(default_factory=dict)
    elapsed_times: dict[str, float | None] = field(default_factory=dict)

    # Evaluation completion
    evaluation_completed: bool = False
    evaluation_failed: bool = False


class Renderer:
    """
    Centralized Rich rendering engine.

    Owns a single Live context for the entire program lifetime.
    Provides update hooks that mutate cached state and trigger repaints.
    Handles responsive layout based on terminal width.
    """

    # Status emoji mapping with spinner indicator
    STATUS_MAP = {
        "queued": ("⏳", "[yellow]Queued[/yellow]", False),
        "launching": ("", "[cyan]Launching[/cyan]", True),
        "running": ("", "[blue]Running[/blue]", True),
        "completed": ("✅", "[green]Completed[/green]", False),
        "failed": ("💥", "[red]Failed[/red]", False),
        "stopped": ("⛔", "[dim]Stopped[/dim]", False),
    }

    def __init__(self):
        self.console = Console()
        self.state = RendererState()
        self.live: Live | None = None
        self.last_render_time = 0.0
        self.throttle_ms = 50  # Minimum time between renders
        self._started = False
        # Plain logs mode (disables Rich Live UI) controlled via env var ORAK_PLAIN_LOGS
        self.headless = os.getenv("ORAK_PLAIN_LOGS", "").lower() in ("1", "true", "yes", "y")

    def start(
        self,
        local: bool = False,
        session_id: str | None = None,
        game_data_path: str = "",
        submission_id: str | None = None,
    ):
        """Initialize the Live display and show header."""
        if self._started:
            return

        self.state.show_local_mode = local
        self.state.session_id = session_id
        self.state.game_data_path = game_data_path
        self.state.submission_id = submission_id
        # Respect plain logs mode: skip Live initialization
        if self.headless:
            self._started = True
            return
        # Start Live context (header is now part of the layout)
        layout = self._build_layout()
        self.live = Live(layout, console=self.console, refresh_per_second=10, screen=False)
        self.live.start()
        self._started = True

    def stop(self):
        """Stop the Live display."""
        if self.live and self._started:
            self.live.stop()
            self._started = False

    def set_session_info(self, session_id: str | None = None, submission_id: str | None = None):
        """Update session/submission identifiers and refresh UI."""
        if session_id is not None:
            self.state.session_id = session_id
        if submission_id is not None:
            self.state.submission_id = submission_id
        self._refresh()

    def _should_render(self) -> bool:
        """Check if enough time has passed since last render (throttling)."""
        now = time.time() * 1000  # milliseconds
        if now - self.last_render_time >= self.throttle_ms:
            self.last_render_time = now
            return True
        return False

    def _refresh(self):
        """Update the Live display with current state."""
        if self.headless or not self.live or not self._started:
            return

        # Don't refresh if evaluation is completed
        if self.state.evaluation_completed:
            return

        if self._should_render():
            layout = self._build_layout()
            self.live.update(layout)

    def _build_layout(self) -> Layout:
        """Build the responsive layout based on terminal width."""
        layout = Layout()

        # Build sections
        parts = []

        # Always show banner (full width)
        parts.append(Layout(self._build_banner(), name="banner", size=3))

        # Always show config section
        parts.append(Layout(self._build_config(), name="config", size=7))

        # Show table with fixed size based on number of games + header + total row if completed
        num_games = len(self.state.server_status_by_game)
        table_rows = num_games + 1  # header
        if self.state.evaluation_completed:
            table_rows += 1  # total row
        table_size = table_rows + 4  # +4 for padding and spacing
        parts.append(
            Padding(Layout(self._build_merged_table(), name="table", size=table_size), (1, 0, 1, 0))
        )

        # Events panel takes all remaining space (no size specified)
        parts.append(Layout(self._build_messages_panel(), name="messages"))

        layout.split_column(*parts)
        return layout

    def _build_banner(self) -> Panel:
        """Build the full-width banner with title only."""
        title = Text("AIcrowd Orak 2025 Evaluation", style="bold", justify="center")
        title.stylize("#fffafa", 0, len(title))
        return Panel(title, border_style="#fffafa", padding=(0, 1))

    def _build_config(self) -> Panel:
        """Build the game config panel."""
        from rich.table import Table as ConfigTable

        config_table = ConfigTable(show_header=False, box=None, padding=(0, 2), show_edge=False)
        config_table.add_column("Key", style="dim", no_wrap=True)
        config_table.add_column("Value", style="bold")

        # Mode
        mode_value = "LOCAL" if self.state.show_local_mode else "Remote"
        mode_style = "bold yellow" if self.state.show_local_mode else "bold cyan"
        config_table.add_row("Mode:", Text(mode_value, style=mode_style))

        # Game Data Path (relative and clickable)
        if self.state.game_data_path:
            # Convert to relative path
            try:
                rel_path = os.path.relpath(self.state.game_data_path)
            except ValueError:
                # If paths are on different drives (Windows), use absolute
                rel_path = self.state.game_data_path

            # Make it clickable with file:// protocol
            abs_path = os.path.abspath(self.state.game_data_path)
            file_url = f"file://{abs_path}"

            # Create clickable text
            path_text = Text(rel_path, style="bold link")
            path_text.stylize(f"link {file_url}")

            config_table.add_row("Game Data Path:", path_text)
        else:
            config_table.add_row("Game Data Path:", "N/A")

        # Submission # (if not local)
        if not self.state.show_local_mode:
            config_table.add_row("Submission #:", self.state.submission_id or "N/A")
            config_table.add_row("Session #:", self.state.session_id or "N/A")

        return Panel(
            config_table, title="[bold]Game Config[/bold]", border_style="dim", padding=(0, 1)
        )

    def _build_merged_table(self) -> Table:
        """Build the merged game servers and scores table."""
        table = Table(show_header=True, box=box.SIMPLE_HEAD, show_edge=False, padding=(0, 1))
        table.add_column("Game", style="green", no_wrap=True, header_style="bold green")
        table.add_column(
            "Status", justify="center", style="bright_black", header_style="bold bright_black"
        )
        table.add_column("Score", justify="right", style="blue", header_style="bold blue")
        table.add_column("Elapsed", justify="right", style="cyan", header_style="bold cyan")

        for game in self.state.server_status_by_game.keys():
            status = self.state.server_status_by_game.get(game, "queued")
            score = self.state.scores_by_game.get(game, 0)

            emoji, text, has_spinner = self.STATUS_MAP.get(status, ("", status, False))
            text = emoji + " " + text

            # Add spinner for active states
            if has_spinner:
                spinner = Spinner("dots", text=text, style="bright_black")
                status_display = spinner
            else:
                status_display = text

            # Calculate elapsed time
            elapsed = self.state.elapsed_times.get(game)
            if elapsed is not None:
                elapsed_str = self._format_elapsed(elapsed)
            else:
                elapsed_str = "-"

            name = game.replace("_", " ").title()
            table.add_row(name, status_display, str(score), elapsed_str)

        # Add total row if evaluation is completed
        if self.state.evaluation_completed:
            total_score = sum(self.state.scores_by_game.values())
            table.add_row("[bold]TOTAL[/bold]", "", f"[bold]{total_score}[/bold]", "", style="bold")

        return table

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time in a readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _build_messages_panel(self) -> Panel:
        """Build the messages/warnings panel."""
        if not self.state.warnings:
            content = Text("No events", style="dim", justify="center")
            return Panel(content, title="[dim]Events[/dim]", border_style="bright_black")

        content = Text()
        # Show all events (panel is now fluid and will expand) in the reverse order
        for msg in reversed(self.state.warnings):
            # Display all messages in default text color
            content.append(msg + "\n")

        return Panel(content, title="Events", border_style="bright_black")

    # Public API: Update hooks

    def warn(self, message: str):
        """Add a warning message to the events panel."""
        formatted = f"[dim]{time.strftime('%H:%M:%S')}[/dim] ⚠ {message}"
        self.state.warnings.append(formatted)
        if self.headless:
            self.console.print(formatted)
        else:
            self._refresh()

    def event(self, message: str):
        """Add an info event to the events panel."""
        formatted = f"{time.strftime('%H:%M:%S')} {message}"
        self.state.warnings.append(formatted)
        if self.headless:
            self.console.print(formatted)
        else:
            self._refresh()

    def info(self, message: str):
        """Print an info message outside the live area (for console logs)."""
        if self.live and self._started:
            self.console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {message}")
        else:
            self.console.print(f"[dim]{time.strftime('%H:%M:%S')}[/dim] {message}")

    def confirm(self, message: str, default: bool = True) -> bool:
        """
        Display a confirmation prompt to the user.

        Temporarily stops the Live context to allow interactive input,
        then restarts it after receiving the user's response.

        Args:
            message: The confirmation message to display
            default: The default answer if user just presses Enter

        Returns:
            True if user confirms, False otherwise
        """
        from rich.prompt import Confirm

        # Temporarily stop the Live context to allow input
        was_started = self._started
        if self.live and was_started:
            self.live.stop()

        try:
            # Get user confirmation
            result = Confirm.ask(message, default=default, console=self.console)
            return result
        finally:
            # Restart the Live context
            if self.live and was_started:
                self.live.start()

    def set_server_status(self, game: str, status: ServerStatus):
        """Update a game server's status."""
        self.state.server_status_by_game[game] = status
        self._refresh()

    def set_score(self, game: str, score: int):
        """Update a game's score."""
        self.state.scores_by_game[game] = score
        self._refresh()

    def set_scores(self, scores: dict[str, int]):
        """Batch update scores."""
        self.state.scores_by_game.update(scores)
        self._refresh()

    def start_game_timer(self, game: str):
        """Start the timer for a game when it begins execution."""
        self.state.game_start_times[game] = time.time()
        self.state.elapsed_times[game] = 0.0
        self._refresh()

    def update_game_elapsed(self, game: str):
        """Update the elapsed time for a specific game."""
        start_time = self.state.game_start_times.get(game)
        if start_time is not None:
            self.state.elapsed_times[game] = time.time() - start_time
        self._refresh()

    def update_game_progress(self, game: str, score: int):
        """Update a game's score and elapsed time during execution."""
        self.set_score(game, score)
        self.update_game_elapsed(game)

    def complete_game(self, game: str, final_score: int):
        """Mark a game as completed with its final score."""
        self.set_server_status(game, "completed")
        self.set_score(game, final_score)
        self.update_game_elapsed(game)

    def complete_evaluation(self, success: bool = True):
        """Mark the entire evaluation as completed."""
        self.state.evaluation_completed = True
        self.state.evaluation_failed = not success

        # Set all incomplete games to completed or failed
        for game in self.state.server_status_by_game.keys():
            status = self.state.server_status_by_game[game]
            if status not in ["completed", "failed", "stopped"]:
                self.state.server_status_by_game[game] = "completed" if success else "failed"

        # Force one final update
        if self.live and self._started:
            layout = self._build_layout()
            self.live.update(layout)

    def show_final_summary(self, game: str, score: int):
        """Show the final summary after game completion."""
        # Just mark evaluation as complete, don't print anything new
        self.complete_evaluation(success=True)


# Global renderer instance
_renderer: Renderer | None = None


def get_renderer() -> Renderer:
    """Get or create the global renderer instance."""
    global _renderer
    if _renderer is None:
        _renderer = Renderer()
    return _renderer
