import requests
import os
import asyncio
import time
import logging
import argparse
import json
import logging
from mcp.types import LoggingLevel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install as rich_traceback_install
from rich.prompt import Confirm
from fastmcp import Client  # type: ignore

from agents.config import UserAgent


from fastmcp.client.logging import LogMessage

# In a real app, you might configure this in your main entry point
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get a logger for the module where the client is used
logger = logging.getLogger(__name__)

# This mapping is useful for converting MCP level strings to Python's levels
LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()

async def log_handler(message: LogMessage):
    """
    Handles incoming logs from the MCP server and forwards them
    to the standard Python logging system.
    """
    msg = message.data.get('msg')
    extra = message.data.get('extra')

    # Convert the MCP log level to a Python log level
    level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)

    # Log the message using the standard logging library
    logger.log(level, msg, extra=extra)


console = Console()
rich_traceback_install(show_locals=False)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)],
    )


API_TOKEN = os.getenv("AICROWD_API_TOKEN")
assert API_TOKEN is not None, "AICROWD_API_TOKEN is not set"

BASE_URL = os.getenv("AICROWD_API_BASE_URL", "https://orak-game-api.aicrowd.com")


class Session:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id
    
    def create(self):
        with console.status("[bold]Creating session...[/bold]"):
            response = requests.post(
                f"{BASE_URL}/sessions",
                headers={"Authorization": f"Token {API_TOKEN}"}
            )
            response.raise_for_status()
            self.session_id = response.json()["task_id"]
            submission_id = response.json()["submission_id"]
        console.print(Panel.fit(Text(f"Session created: {self.session_id}\nSubmission ID: {submission_id}", style="green"), title="Session"))
    
    def get(self):
        response = requests.get(
            f"{BASE_URL}/sessions/{self.session_id}",
            headers={"Authorization": f"Token {API_TOKEN}"}
        )
        return response.json()
    
    def stop(self):
        response = requests.delete(
            f"{BASE_URL}/sessions/{self.session_id}",
            headers={"Authorization": f"Token {API_TOKEN}"}
        )
        return response.json()
    
    def wait_for_start(self, poll_interval: float = 1.0, timeout: float = 300.0):
        start = time.time()
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Waiting for game server instance to start...", start=True)
            while True:
                status = self.get()["last_status"]
                progress.update(task, description=f"Game server instance: [bold]{status}[/bold]")
                if status == "RUNNING":
                    break
                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for task to start")
                time.sleep(poll_interval)
        console.print(Panel.fit(Text("Game server has started", style="green"), title="Status"))


class GameEnv:
    def __init__(self, mcp_url: str):
        self.client = Client(mcp_url, log_handler=log_handler)
    
    async def wait_for_ping(self):
        await self.client.ping()
    
    async def _call_tool(self, tool_name: str, *args, **kwargs):
        logging.getLogger(__name__).debug("Calling tool %s with args=%s kwargs=%s", tool_name, args, kwargs)
        result = await self.client.call_tool(tool_name, *args, **kwargs)
        logging.getLogger(__name__).debug("Tool %s result: %s", tool_name, result)
        return json.loads(result.structured_content["result"])

    async def load_obs(self):
        return await self._call_tool("load-obs")
    
    async def dispatch_final_action(self, action: str):
        return await self._call_tool("dispatch-final-action", {"action_str": action})


class Runner:
    def __init__(self, session_id: str | None = None):
        self.session = Session(session_id=session_id)

        session_dir = os.path.join(os.getcwd(), ".aicrowd")
        session_file = os.path.join(session_dir, "session_id")
        self.session_file = session_file
        os.makedirs(session_dir, exist_ok=True)

        # If no session-id provided, check persisted session file
        if self.session.session_id is None and os.path.exists(session_file):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    previous_session_id = f.read().strip()
            except Exception:
                previous_session_id = ""

            if previous_session_id:
                if Confirm.ask(
                    f"Found previous session [bold]{previous_session_id}[/bold]. Continue it?",
                    default=True,
                ):
                    self.session.session_id = previous_session_id
                else:
                    # Stop previous session before creating a new one
                    try:
                        temp = Session(previous_session_id)
                        temp.stop()
                    except Exception:
                        pass

        # Create a new session if we still don't have one
        if self.session.session_id is None:
            self.session.create()
            try:
                with open(session_file, "w", encoding="utf-8") as f:
                    f.write(self.session.session_id)
            except Exception:
                pass
        else:
            # Persist provided/continued session id
            try:
                with open(session_file, "w", encoding="utf-8") as f:
                    f.write(self.session.session_id)
            except Exception:
                pass

        self.session.wait_for_start()
        self.scores = {
            "twenty_fourty_eight": 0,
            "street_fighter": 0,
            "super_mario": 0,
            "pokemon": 0,
            "starcraft2": 0,
        }
    
    async def start_game(self, game_name: str):
        console.rule(f"[bold]Starting game[/bold] [cyan]{game_name}[/cyan]")
        mcp_url = self.session.get()["mcp_urls"][game_name]
        agent = UserAgent()

        env = GameEnv(mcp_url)
        async with env.client:
            await env.wait_for_ping()

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Playing...[/bold blue] | Score: [bold]{task.completed}[/bold] | Finished: [bold]{task.fields[finished]}[/bold]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("game_loop", total=None, finished=False)
                while True:
                    obs = await env.load_obs()
                    action = agent.act(obs)
                    result = await env.dispatch_final_action(action)
                    finished = bool(result.get("is_finished"))
                    progress.update(task, advance=0, completed=result.get("cumulative_score"), finished=finished)

                    if finished:
                        self.scores[game_name] = result.get("cumulative_score")
                        break

        table = Table(title=f"Results: {game_name}")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Score", str(self.scores[game_name]))
        console.print(table)
        console.rule()

        os.remove(self.session_file)

def main():
    parser = argparse.ArgumentParser(description="Orak Starter Kit Runner")
    parser.add_argument("--game", default="twenty_fourty_eight", help="Game key to run (matches mcp_urls key)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--session-id", default=None, help="Use existing session id instead of creating a new session")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    console.print(Panel.fit(Text("Orak Runner", style="bold cyan")))

    runner = Runner(session_id=args.session_id)
    asyncio.run(runner.start_game(args.game))
    console.print(Panel.fit(Text(f"Score: {runner.scores[args.game]}", style="bold green"), title="Final"))


if __name__ == "__main__":
    main()
