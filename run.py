import asyncio
from typing import Any, Annotated
import typer
from enum import StrEnum
from evaluation_utils.runner import Runner
from evaluation_utils.commons import setup_logging, GAME_DATA_DIR, GAME_SERVER_PORTS
from evaluation_utils.renderer import get_renderer
from dotenv import load_dotenv
from config.utils import load_hydra_settings
from loguru import logger
import weave

app = typer.Typer(pretty_exceptions_enable=False)


class ExperimentConfigName(StrEnum):
    GEMINI = "gemini"
    OPENAI = "openai"


load_dotenv()


@app.command()
def main(
    config_name: Annotated[
        ExperimentConfigName,
        typer.Option(
            "--config-name",
            "-c",
            help="Hydra config name for evaluation setup.",
            case_sensitive=False,
        ),
    ] = ExperimentConfigName.GEMINI,
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Use existing session id instead of creating a new session",
    ),
    local: bool = typer.Option(False, "--local", help="Run in local mode"),
    games: Annotated[list[str], typer.Option(
        "--games",
        help="Only run these games (space-separated list). Only supported in LOCAL mode.",
    )] =list(GAME_SERVER_PORTS.keys()),   
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Run evaluation for Orak 2025 games."""

    # Enforce that game selection is only supported in local mode
    if games and not local:
        raise typer.BadParameter("--games can only be used together with --local")
    setup_logging(verbose=verbose)

    settings = load_hydra_settings(config_name=config_name.value)
    logger.info(f"Loading Hydra settings {config_name}...")

    # Initialize Weave if enabled (uses same W&B credentials)
    if settings.wandb.weave_enabled:
        try:
            weave.init(settings.wandb.project_name)
            logger.info(f"Weave initialized for project: {settings.wandb.project_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize Weave: {e}")

    # Initialize the centralized renderer
    renderer = get_renderer()
    renderer.start(local=local, session_id=session_id, game_data_path=GAME_DATA_DIR)

    try:
        # Only pass a game subset in local mode; remote mode always runs all games
        selected_games = games if local else None
        renderer.event("Starting evaluation run ...")
        renderer.event(f"Settings: {settings.model_dump()}...")
        runner = Runner(
            session_id=session_id,
            local=local,
            renderer=renderer,
            games=selected_games,
            settings=settings,
        )
        
        asyncio.run(runner.evaluate_all_games())

        # Show final summary with total score
        total_score = sum(runner.scores.values())
        renderer.show_final_summary("all_games", total_score)
    except Exception:
        # Mark evaluation as failed
        renderer.complete_evaluation(success=False)
        raise
    finally:
        renderer.stop()
        # Finish Weave tracking
        if settings.wandb.weave_enabled:
            try:
                weave.finish()
            except:
                pass


if __name__ == "__main__":
    app()
