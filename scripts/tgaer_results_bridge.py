"""Bridge: orak `evaluation_summary.json` → autoresearch `results.jsonl`.

Walks both worktrees' game_logs/ dirs for `tgaer_(baseline|detector)_*`
rollouts, extracts per-rollout aggregate (mean across episodes), and
appends rows to per-side `experiments/<tag>/results.jsonl` via
`autoresearch.log_experiment`. Once written, the canonical autoresearch
CLI flow (autoresearch-report, autoresearch-pr-updater, plot_milestone_bars)
works directly.

Tag layout:
  experiments/tgaer_pr1_baseline/results.jsonl  (12 rows: 3 × 4 games)
  experiments/tgaer_pr1_detector/results.jsonl  (12 rows: 3 × 4 games)

The `game` field per row differentiates within each tag.

Idempotent: log_experiment auto-assigns experiment index based on
existing rows, so re-running this script on the same rollouts will
duplicate entries. To avoid that we check existing experiment numbers
per (tag, game, run_id) and skip already-bridged rows.

Run:
    uv run python /workspace/orak-futile-detector/scripts/tgaer_results_bridge.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from glob import glob
from pathlib import Path

# autoresearch is installed in both worktree venvs — use the one we're running under
sys.path.insert(0, "/workspace/autoresearch/src")
from autoresearch.results import load_results, log_experiment

REPO_FOR_TAG = "/workspace/orak-futile-detector"  # any worktree works for storing results.jsonl
EXPERIMENTS_DIR = f"{REPO_FOR_TAG}/experiments"

WORKTREES = {
    "baseline": "/workspace/orak-master-baselines",
    "detector": "/workspace/orak-futile-detector",
}

WANDB_PROJECT = {
    "pokemon_red": "orak-pokemon-red",
    "super_mario": "orak-super-mario",
    "twenty_fourty_eight": "orak-2048",
    "star_craft": "orak-star-craft",
}


def wandb_url(game: str, run_id: str) -> str:
    return f"https://wandb.ai/chaleong/{WANDB_PROJECT[game]}/runs/{run_id}_{WANDB_PROJECT[game]}"


def parse_run(d: Path) -> dict | None:
    """Read evaluation_summary.json — port the full payload + add derived aggregates.

    The `summary` field carries the raw evaluation_summary.json content verbatim
    (including per-episode `episodes` list). Derived aggregates (best/mean/std)
    are added at top level for fast access without re-iterating episodes.
    """
    sj = d / "evaluation_summary.json"
    if not sj.exists():
        return None
    s = json.loads(sj.read_text())
    scores = [e["final_score"] for e in s["episodes"]]
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    return {
        "run_id": d.name,
        "summary": s,  # full evaluation_summary.json verbatim — preserves episodes + all fields
        "best": max(scores),
        "mean": mean,
        "std": (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5,
        "n_episodes": len(scores),
        "total_calls": s["total_inference_calls"],
        "total_tokens": s["total_tokens"],
        "ctime": sj.stat().st_mtime,
    }


def already_bridged(tag: str, game: str, run_id: str) -> bool:
    """Check if a row for this (tag, game, run_id) already exists.

    Note: autoresearch's log_experiment FLATTENS the `extra` dict into top-level
    row fields, so run_id ends up at row["run_id"], not row["extra"]["run_id"].
    """
    rows = load_results(experiments_dir=EXPERIMENTS_DIR, tag=tag)
    return any(r.get("run_id") == run_id and r.get("game") == game for r in rows)


def main() -> None:
    written = 0
    skipped = 0
    for side, wt in WORKTREES.items():
        tag = f"tgaer_pr1_{side}"
        for game in WANDB_PROJECT:
            pattern = f"{wt}/game_logs/{game}/tgaer_{side}_{game}_*"
            for d in sorted(glob(pattern)):
                p = Path(d)
                if not (p / "evaluation_summary.json").exists():
                    continue
                rec = parse_run(p)
                if rec is None:
                    continue
                if already_bridged(tag, game, rec["run_id"]):
                    skipped += 1
                    continue
                # Status: KEEP for completed rollouts (we'll let autoresearch verdict
                # tooling assign comparative status later); first per (tag, game) gets BASELINE.
                existing_for_game = [r for r in load_results(experiments_dir=EXPERIMENTS_DIR, tag=tag) if r.get("game") == game]
                status = "BASELINE" if not existing_for_game else "KEEP"
                log_experiment(
                    experiments_dir=EXPERIMENTS_DIR,
                    tag=tag,
                    game=game,
                    score=rec["mean"],
                    game_score=rec["best"],
                    steps=rec["total_calls"],
                    status=status,
                    description=f"TGAER PR1 {side} on {game} (run_id={rec['run_id']})",
                    wandb_url=wandb_url(game, rec["run_id"]),
                    extra={
                        "run_id": rec["run_id"],
                        "mean": rec["mean"],
                        "best": rec["best"],
                        "std": rec["std"],
                        "n_episodes": rec["n_episodes"],
                        "total_tokens": rec["total_tokens"],
                        # Full evaluation_summary.json content (per-episode breakdown + totals).
                        # ~50 bytes/episode × ~50 episodes = ~2.5KB per row, negligible at our scale.
                        "evaluation_summary": rec["summary"],
                    },
                )
                written += 1
                print(f"  + {tag} / {game} / n={existing_for_game and len(existing_for_game) + 1 or 1}  best={rec['best']:.2f} mean={rec['mean']:.2f} status={status}")
    print(f"\nBridged {written} new rows, skipped {skipped} already-bridged.")
    print(f"  experiments/{('tgaer_pr1_baseline', 'tgaer_pr1_detector')[0]}/results.jsonl")
    print(f"  experiments/{('tgaer_pr1_baseline', 'tgaer_pr1_detector')[1]}/results.jsonl")


if __name__ == "__main__":
    main()
