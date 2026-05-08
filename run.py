import asyncio
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import weave
from dotenv import load_dotenv
from loguru import logger

from config.utils import load_hydra_settings
from evaluation_utils.commons import GAME_DATA_DIR, GAME_SERVER_PORTS, setup_logging
from evaluation_utils.renderer import get_renderer
from evaluation_utils.runner import Runner

app = typer.Typer(pretty_exceptions_enable=False)


class ExperimentConfigName(StrEnum):
    GEMINI = "gemini"
    OPENAI = "openai"
    POETIQ = "poetiq"
    MACLA = "macla"
    UNIFIED_MACLA = "unified_macla"
    LOCAL = "local"
    LOCAL_TEST = "local_test"
    GEMMA = "gemma"
    GEMMA_TEST = "gemma_test"
    GEMMA_STAGE_A = "gemma_stage_a"


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
    games: Annotated[
        list[str] | None,
        typer.Option(
            "--games",
            help="Only run these games (space-separated list). Only supported in LOCAL mode.",
        ),
    ] = None,
    experiment_description: str | None = typer.Option(
        None,
        "--experiment-description",
        "-d",
        help="Description for the experiment (logged to W&B notes)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    # Checkpoint options
    save_checkpoints: bool = typer.Option(
        True, "--save-checkpoints", help="Save agent checkpoints during training"
    ),
    load_checkpoint: bool = typer.Option(
        False, "--load-checkpoint", help="Load from latest checkpoint if available"
    ),
    checkpoint_frequency: int = typer.Option(
        10, "--checkpoint-freq", help="Save checkpoint every N steps (default: 10)"
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Custom run ID for organising outputs (default: timestamp)"
    ),
    prev_run_id: str | None = typer.Option(
        None,
        "--prev-run-id",
        help="If set with --load-checkpoint, load the latest checkpoint from "
        "game_logs/<game>/<prev_run_id>/checkpoints/ instead of the "
        "current run's (empty) checkpoint dir. Used by autoresearch to "
        "carry MACLA's learned procedures across iterations.",
    ),
):
    """Run evaluation for Orak 2025 games."""

    # Enforce that game selection is only supported in local mode
    if games is not None and not local:
        raise typer.BadParameter("--games can only be used together with --local")

    # If no games specified in local mode, default to all games
    if local and games is None:
        games = list(GAME_SERVER_PORTS.keys())

    setup_logging(verbose=verbose)

    logger.info(f"Loading Hydra settings {config_name}...")
    settings = load_hydra_settings(config_name=config_name.value)

    # Override W&B notes if provided
    if experiment_description:
        settings.wandb.notes = experiment_description
        # Also update game-specific wandb configs
        if settings.pokemon_red and settings.pokemon_red.wandb:
            settings.pokemon_red.wandb.notes = experiment_description
        if settings.super_mario and settings.super_mario.wandb:
            settings.super_mario.wandb.notes = experiment_description
        if settings.twenty_fourty_eight and settings.twenty_fourty_eight.wandb:
            settings.twenty_fourty_eight.wandb.notes = experiment_description
        if settings.star_craft and settings.star_craft.wandb:
            settings.star_craft.wandb.notes = experiment_description
        logger.info(f"Experiment description set: {experiment_description}")

    # Note: weave is auto-initialized by wandb.init() per game project.
    # Do NOT call weave.init() globally — it overrides per-game initialization.

    # If loading checkpoint and neither run_id nor prev_run_id are provided,
    # try to find the latest run to resume. Skipped when prev_run_id is set —
    # autoresearch wants to LOAD FROM prev_run_id but WRITE to a fresh run_id,
    # so resuming the latest run would clobber game_logs.
    if run_id is None and load_checkpoint and local and games and not prev_run_id:
        # Check the first game's directory for runs
        first_game_dir = Path(GAME_DATA_DIR) / games[0]
        if first_game_dir.exists():
            # Get all subdirectories that look like potential runs (ignore non-dirs)
            runs = [d.name for d in first_game_dir.iterdir() if d.is_dir()]
            if runs:
                # Sort to find the latest (assuming timestamp or lexicographical order)
                runs.sort()
                latest_run = runs[-1]
                run_id = latest_run
                logger.info(f"Auto-detected latest run: {latest_run}")
                logger.info(f"Resuming Run ID: {run_id}")
                logger.info(
                    f"Looking for checkpoints in: {first_game_dir / latest_run / 'checkpoints'}"
                )

    # Create run ID (timestamp-based if not provided)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting new run with ID: {run_id}")

    settings.wandb.run_id = run_id

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
            run_id=run_id,
            save_checkpoints=save_checkpoints,
            load_checkpoint=load_checkpoint,
            checkpoint_frequency=checkpoint_frequency,
            prev_run_id=prev_run_id,
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
