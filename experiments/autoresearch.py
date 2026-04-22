"""
Autoresearch loop for MACLA parameter optimisation.

Iteratively runs experiments, adjusting per-game theta/warmup params
based on results. Logs to experiment tracker and regenerates plots.

Usage:
    # Run optimisation loop (max 5 iterations)
    python experiments/autoresearch.py run --max-iterations 5

    # Dry run: propose next params without running
    python experiments/autoresearch.py run --dry-run

    # Log results from a completed run
    python experiments/autoresearch.py log-run --run-id 20260422_213143
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import typer
import yaml

from experiments.experiment_progress import (
    ALL_GAMES,
    extract_run_results,
    load_results,
    log_experiment,
    plot_progress,
)

ROOT = Path(__file__).parent.parent
CONFIGS_DIR = ROOT / "configs"

# Per-game parameter search bounds
PARAM_BOUNDS = {
    "super_mario": {
        "macla_theta_base": (0.10, 0.30),
        "macla_max_theta": (0.15, 0.35),
        "macla_min_theta": (0.03, 0.10),
        "macla_warmup_steps": (0, 10),
    },
    "twenty_fourty_eight": {
        "macla_theta_base": (0.15, 0.35),
        "macla_max_theta": (0.20, 0.45),
        "macla_min_theta": (0.05, 0.15),
        "macla_warmup_steps": (0, 10),
    },
    "pokemon_red": {
        "macla_theta_base": (0.25, 0.45),
        "macla_max_theta": (0.35, 0.55),
        "macla_min_theta": (0.10, 0.25),
        "macla_warmup_steps": (5, 20),
    },
}

STEP_SIZES = {
    "macla_theta_base": 0.05,
    "macla_max_theta": 0.05,
    "macla_min_theta": 0.02,
    "macla_warmup_steps": 3,
}

app = typer.Typer(help="MACLA autoresearch parameter optimisation")


def read_yaml_config(game: str, config_type: str = "unified_macla") -> dict:
    """Read current YAML config for a game."""
    path = CONFIGS_DIR / game / "agent" / f"{config_type}.yaml"
    return yaml.safe_load(path.read_text())


def write_yaml_config(game: str, config: dict, config_type: str = "unified_macla"):
    """Write updated YAML config for a game."""
    path = CONFIGS_DIR / game / "agent" / f"{config_type}.yaml"
    # Preserve comments by reading, updating macla_ fields only
    lines = path.read_text().splitlines()
    macla_keys = {k for k in config if k.startswith("macla_")}

    # Remove existing macla_ lines
    new_lines = [l for l in lines if not any(l.strip().startswith(f"{k}:") for k in macla_keys)]

    # Append updated macla_ values
    for k in sorted(macla_keys):
        v = config[k]
        new_lines.append(f"{k}: {v}")

    path.write_text("\n".join(new_lines) + "\n")


def get_best_scores(tag: str = "macla") -> dict[str, float]:
    """Get current best evaluation_score per game from results."""
    results = load_results(tag=tag)
    best = {}
    for r in results:
        game = r["game"]
        score = r["evaluation_score"]
        if r["status"] in ("KEEP", "BASELINE"):
            best[game] = max(best.get(game, 0), score)
    return best


def get_current_params(game: str, config_type: str = "unified_macla") -> dict[str, float]:
    """Extract current macla_ params from YAML config."""
    config = read_yaml_config(game, config_type)
    return {k: v for k, v in config.items() if k.startswith("macla_")}


def propose_next_params(game: str, results: list[dict], config_type: str = "unified_macla") -> dict[str, float]:
    """Propose next theta params for a game based on experiment history.

    Strategy: look at last 2 experiments for this game.
    - If last improved over previous: continue in same direction
    - If last regressed: reverse direction
    - If first run: use current config as-is
    """
    game_results = [r for r in results if r["game"] == game]
    current = get_current_params(game, config_type)
    bounds = PARAM_BOUNDS.get(game, {})

    if len(game_results) < 2:
        return current

    last = game_results[-1]
    prev = game_results[-2]
    improved = last["evaluation_score"] > prev["evaluation_score"]

    new_params = dict(current)
    for param, (lo, hi) in bounds.items():
        step = STEP_SIZES.get(param, 0.05)
        cur_val = current.get(param, (lo + hi) / 2)

        if improved:
            # Continue in same direction (decrease theta = more bayesian)
            new_val = cur_val - step
        else:
            # Try opposite direction (increase theta = more fallback)
            new_val = cur_val + step

        # Clamp to bounds
        if param == "macla_warmup_steps":
            new_val = int(max(lo, min(hi, round(new_val))))
        else:
            new_val = round(max(lo, min(hi, new_val)), 3)

        new_params[param] = new_val

    return new_params


def run_experiment(config_name: str, games: list[str]) -> str:
    """Run an experiment and return the run_id."""
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "run.py"),
        f"--config-name={config_name}",
        "--local",
    ]
    for g in games:
        cmd.extend(["--games", g])

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = (time.time() - start) / 60

    if result.returncode != 0:
        print(f"Run failed with exit code {result.returncode}")
        return ""

    # Find latest run_id from game_logs
    game_log_dir = ROOT / "game_logs" / games[0]
    run_dirs = sorted(game_log_dir.iterdir(), key=lambda p: p.name, reverse=True)
    run_id = run_dirs[0].name if run_dirs else ""
    print(f"\nRun completed in {elapsed:.1f}min — run_id: {run_id}")
    return run_id


def log_run_results(
    run_id: str,
    games: list[str],
    description: str,
    tag: str = "macla",
    best_scores: dict[str, float] | None = None,
):
    """Extract results from a run and log to experiment tracker."""
    results = extract_run_results(run_id, games)
    best_scores = best_scores or {}

    for game, data in results.items():
        eval_score = data["max_eval"]
        game_score = data["game_score"]
        steps = data["steps"]
        best = best_scores.get(game, 0)
        status = "KEEP" if eval_score > best else "DISCARD"

        wandb_project = {
            "super_mario": "orak-super-mario",
            "twenty_fourty_eight": "orak-2048",
            "pokemon_red": "orak-pokemon-red",
        }.get(game, game)
        wandb_url = f"https://wandb.ai/chaleong/{wandb_project}/runs/{run_id}_{wandb_project}"

        notes = f"max_eval={eval_score:.2f}, {data['episodes']} episodes, {steps} steps"
        if status == "KEEP":
            notes += f". Improved from {best:.2f}"
        else:
            notes += f". Below best {best:.2f}"

        log_experiment(
            game=game,
            score=eval_score,
            steps=steps,
            status=status,
            description=description,
            wandb_url=wandb_url,
            notes=notes,
            game_score=game_score,
            tags=[tag],
        )

    return results


@app.command()
def log_run(
    run_id: str = typer.Option(..., help="Run ID from game_logs (e.g. 20260422_213143)"),
    description: str = typer.Option(..., "-d", help="Experiment description"),
    tag: str = typer.Option("macla", help="Experiment tag"),
    games: list[str] = typer.Option(ALL_GAMES, "--games", help="Games to log"),
):
    """Log results from a completed run to the experiment tracker."""
    best = get_best_scores(tag)
    log_run_results(run_id, games, description, tag, best)
    plot_progress(tag=tag)


@app.command()
def run(
    config: str = typer.Option("unified_macla", help="Hydra config name"),
    tag: str = typer.Option("macla", help="Experiment tag"),
    max_iterations: int = typer.Option(5, help="Max optimisation iterations"),
    games: list[str] = typer.Option(ALL_GAMES, "--games", help="Games to optimise"),
    dry_run: bool = typer.Option(False, help="Only propose params, don't run"),
    config_type: str = typer.Option("unified_macla", help="YAML config type to modify"),
):
    """Run the autoresearch optimisation loop."""
    print(f"Autoresearch loop: config={config}, tag={tag}, max_iterations={max_iterations}")
    print(f"Games: {games}\n")

    all_results = load_results(tag=tag)

    for iteration in range(max_iterations):
        print(f"\n{'#'*60}")
        print(f"# Iteration {iteration + 1}/{max_iterations}")
        print(f"{'#'*60}")

        best = get_best_scores(tag)
        print(f"\nCurrent best scores: {best}")

        # Propose and apply new params per game
        descriptions = []
        for game in games:
            new_params = propose_next_params(game, all_results, config_type)
            current = get_current_params(game, config_type)

            changed = {k: v for k, v in new_params.items() if current.get(k) != v}
            if changed:
                print(f"\n  {game}: {changed}")
                descriptions.append(f"{game}: {', '.join(f'{k}={v}' for k, v in changed.items())}")

                if not dry_run:
                    full_config = read_yaml_config(game, config_type)
                    full_config.update(new_params)
                    write_yaml_config(game, full_config, config_type)
            else:
                print(f"\n  {game}: no changes (at boundary)")

        description = f"autoresearch iter {iteration + 1}: " + "; ".join(descriptions) if descriptions else f"autoresearch iter {iteration + 1}: no param changes"
        print(f"\nDescription: {description}")

        if dry_run:
            print("\n[DRY RUN] Skipping experiment execution")
            continue

        # Run experiment
        run_id = run_experiment(config, games)
        if not run_id:
            print("Run failed, stopping loop")
            break

        # Log results
        run_results = log_run_results(run_id, games, description, tag, best)

        # Reload results for next iteration
        all_results = load_results(tag=tag)

        # Check if any game improved
        any_improved = False
        for game, data in run_results.items():
            if data["max_eval"] > best.get(game, 0):
                any_improved = True
                print(f"  {game}: IMPROVED {best.get(game, 0):.2f} → {data['max_eval']:.2f}")
            else:
                print(f"  {game}: no improvement (best={best.get(game, 0):.2f})")

        # Regenerate plot
        plot_progress(tag=tag)

        if not any_improved:
            print(f"\nNo improvements in iteration {iteration + 1}. Continuing to explore...")

    print(f"\nAutoresearch complete after {min(iteration + 1, max_iterations)} iterations")
    plot_progress(tag=tag)


if __name__ == "__main__":
    app()
