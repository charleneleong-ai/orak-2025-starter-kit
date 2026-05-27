"""Replay the existing PR3 smoke through the new StarCraftShaper.

Re-runs game_logs/star_craft/stagnation_pr3_star_craft_smoke_20260527T094639Z/
game_states.jsonl (2500 iterations, already on disk) through the new shaper
without needing SC2. Reports cumulative reward per episode and prints the
decision gate: reward at iter 51 (productive state) MUST be > reward at iter
201 (floated + supply-blocked failure state).

Run:
    .venv/bin/python -m experiments.sc2_replay_shaper
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import typer

from agents.macla.online_evaluator import DEFAULT_SHAPING, StarCraftShaper

app = typer.Typer(add_completion=False)

SMOKE_PATH = Path(
    "game_logs/star_craft/stagnation_pr3_star_craft_smoke_20260527T094639Z/game_states.jsonl"
)


@app.command()
def replay(
    path: Path = typer.Option(SMOKE_PATH, "--path", "-p", help="game_states.jsonl"),
    spot_iters: list[int] = typer.Option(
        [1, 51, 201, 500, 1000, 2000], "--spot", help="iterations to print"
    ),
) -> None:
    if not path.exists():
        typer.echo(f"ERROR: {path} not found", err=True)
        raise typer.Exit(1)

    shaper = StarCraftShaper(DEFAULT_SHAPING["star_craft"])
    prev_metrics: dict = {}
    spot_rewards: dict[int, float] = {}
    per_episode_reward: dict[int, float] = defaultdict(float)
    n_idleness = 0
    n_supply_block = 0
    n_building_built = 0
    cur_episode = 1

    with path.open() as f:
        prev_iter = -1
        for line in f:
            obj = json.loads(line)
            it = obj.get("iteration", 0)
            # Episode boundary: iteration resets to 1 after the first step.
            if it == 1 and prev_iter != -1:
                cur_episode += 1
                shaper.reset_episode()  # also clear shaper-internal episode state
                prev_metrics = {}  # don't bleed metrics across episodes
            prev_iter = it
            obs_str = obj.get("obs", {}).get("obs_str", "")
            cur = shaper.extract_metrics(obs_str)

            reward = shaper.compute_reward(prev_metrics, cur, success=False, is_fatal=False)

            mineral_delta = cur.get("mineral", 0) - prev_metrics.get("mineral", 0)
            supply_delta = cur.get("supply_used", 0) - prev_metrics.get("supply_used", 0)
            building_delta = cur.get("building_count", 0) - prev_metrics.get("building_count", 0)
            if mineral_delta > 0 and supply_delta == 0:
                n_idleness += 1
            if cur.get("supply_left", 1) <= 0:
                n_supply_block += 1
            if building_delta > 0:
                n_building_built += building_delta

            per_episode_reward[cur_episode] += reward

            if it in spot_iters:
                spot_rewards[it] = reward
                typer.echo(
                    f"  iter {it:>4}  reward={reward:+.3f}  "
                    f"mineral={cur.get('mineral'):>4}  "
                    f"supply_used={cur.get('supply_used'):>3}  "
                    f"supply_left={cur.get('supply_left'):>3}  "
                    f"buildings={cur.get('building_count')}"
                )

            prev_metrics = cur

    typer.echo("")
    typer.echo("=== Per-episode cumulative reward ===")
    for ep, r in sorted(per_episode_reward.items()):
        typer.echo(f"  episode {ep}: {r:+.3f}")

    typer.echo("")
    typer.echo("=== Aggregate penalty fires ===")
    typer.echo(f"  idleness:   {n_idleness}")
    typer.echo(f"  supply_blk: {n_supply_block}")
    typer.echo(f"  buildings:  {n_building_built}")

    typer.echo("")
    typer.echo("=== Decision gate ===")
    r_51 = spot_rewards.get(51, float("nan"))
    r_201 = spot_rewards.get(201, float("nan"))
    typer.echo(f"  reward[iter 51]  (productive) = {r_51:+.3f}")
    typer.echo(f"  reward[iter 201] (floated)    = {r_201:+.3f}")
    if r_201 < r_51:
        typer.echo("  PASS — failure state is lower-rewarded than productive state")
    else:
        typer.echo("  FAIL — magnitudes need tuning before running a fresh smoke")
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
