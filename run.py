import asyncio
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import weave
from dotenv import load_dotenv
from loguru import logger

from config.agent_config import LocalConfig
from config.base import Settings
from config.utils import load_hydra_settings
from evaluation_utils.commons import GAME_DATA_DIR, GAME_SERVER_PORTS, setup_logging
from evaluation_utils.renderer import get_renderer
from evaluation_utils.runner import Runner

GAME_KEYS = ("pokemon_red", "super_mario", "twenty_fourty_eight", "star_craft")

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
    GEMMA_26B = "gemma_26b"
    GEMMA_26B_NO_PROCEDURES = "gemma_26b_no_procedures"
    QWEN35_A3B_INT4 = "qwen35_a3b_int4"
    QWEN3_THINKING = "qwen3_thinking"


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
    # Paired-rollout / agentic-RL collection. Same harness invocation
    # serves the offline DPO/GSPO pool (collect N rollouts per seed) and
    # the online LoRA loop (trainer subprocess-spawns this between train
    # steps with a fresh --adapter-name). See feat/agnetic_rl_research for
    # the trainer side.
    n_rollouts: int = typer.Option(
        1,
        "--n-rollouts",
        help="Number of rollouts to collect under a shared rollout_group_id. "
        "Default 1 = current single-run behaviour (no group). When N>1 each "
        "iteration writes its own run dir suffixed _rollout_<i>.",
    ),
    rollout_group_id: str | None = typer.Option(
        None,
        "--rollout-group-id",
        help="Shared id for paired rollouts. Auto-generated (uuid4 hex) when "
        "--n-rollouts >1 and not provided. Surfaces as a wandb tag and is "
        "written to wandb.config + each raw_requests.jsonl record.",
    ),
    rollout_idx: int = typer.Option(
        0,
        "--rollout-idx",
        help="Index of this rollout within its group. Used by external "
        "orchestrators (trainer / autoresearch) when calling --n-rollouts=1 "
        "in a loop. Overridden per-iteration when --n-rollouts >1.",
    ),
    adapter_name: str | None = typer.Option(
        None,
        "--adapter-name",
        help="LoRA adapter name registered with vLLM (via "
        "/v1/load_lora_adapter). When set, overrides each game's local-agent "
        "`model` field — vLLM routes requests to that adapter. Use null/base "
        "model when omitted.",
    ),
    capture_logprobs: bool = typer.Option(
        False,
        "--capture-logprobs/--no-capture-logprobs",
        help="Request per-token logprobs from the local serving backend "
        "(vLLM). Used as π_θ_old by online GSPO/PPO; offline DPO/GRPO can "
        "ignore and recompute via teacher-forcing.",
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

    if n_rollouts < 1:
        raise typer.BadParameter("--n-rollouts must be >= 1")

    # Generate a group_id only when paired-rollout mode is on. Keeps the
    # default invocation (n=1, no explicit id) free of group metadata so
    # legacy runs aren't tagged as part of an RL collection.
    if n_rollouts > 1 and rollout_group_id is None:
        rollout_group_id = uuid.uuid4().hex[:12]
        logger.info(f"Auto-generated rollout_group_id: {rollout_group_id}")

    # Initialize the centralized renderer
    renderer = get_renderer()
    renderer.start(local=local, session_id=session_id, game_data_path=GAME_DATA_DIR)

    try:
        # Only pass a game subset in local mode; remote mode always runs all games
        selected_games = games if local else None
        renderer.event("Starting evaluation run ...")
        renderer.event(f"Settings: {settings.model_dump()}...")
        if n_rollouts > 1:
            renderer.event(
                f"Paired-rollout mode: collecting {n_rollouts} rollouts under "
                f"group {rollout_group_id} (adapter={adapter_name or 'base'})"
            )

        for i in range(n_rollouts):
            iter_idx = i if n_rollouts > 1 else rollout_idx
            iter_run_id = f"{run_id}_rollout_{i}" if n_rollouts > 1 else run_id

            _apply_rollout_metadata(
                settings,
                run_id=iter_run_id,
                rollout_group_id=rollout_group_id,
                rollout_idx=iter_idx,
                adapter_name=adapter_name,
                capture_logprobs=capture_logprobs,
            )

            runner = Runner(
                session_id=session_id,
                local=local,
                renderer=renderer,
                games=selected_games,
                settings=settings,
                run_id=iter_run_id,
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


def _apply_rollout_metadata(
    settings: Settings,
    *,
    run_id: str,
    rollout_group_id: str | None,
    rollout_idx: int,
    adapter_name: str | None,
    capture_logprobs: bool,
) -> None:
    """Stamp paired-rollout fields onto top-level + per-game wandb configs
    and apply adapter/logprobs overrides to local-model agent configs."""
    settings.wandb.run_id = run_id
    settings.wandb.rollout_group_id = rollout_group_id
    settings.wandb.rollout_idx = rollout_idx
    settings.wandb.adapter_name = adapter_name

    for key in GAME_KEYS:
        game_settings = getattr(settings, key, None)
        if game_settings is None:
            continue
        if game_settings.wandb is not None:
            game_settings.wandb.run_id = run_id
            game_settings.wandb.rollout_group_id = rollout_group_id
            game_settings.wandb.rollout_idx = rollout_idx
            game_settings.wandb.adapter_name = adapter_name
        # Adapter + logprobs only apply to local (vLLM/Ollama/MLX) agents —
        # Gemini/OpenAI don't expose LoRA endpoints or per-call logprobs.
        agent_cfg = getattr(game_settings, "agent", None)
        if isinstance(agent_cfg, LocalConfig):
            if adapter_name:
                agent_cfg.model = adapter_name
            agent_cfg.capture_logprobs = capture_logprobs


if __name__ == "__main__":
    app()
