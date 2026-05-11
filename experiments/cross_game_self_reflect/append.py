"""Append a cross-game self-reflection row to results.jsonl.

Compares Stage D + self-reflection vs the PR #31 Stage D baselines.
Reads evaluation_summary.json and picks the max final_score across
episodes (matches the per-game append.py shape used by PR #31).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO / "experiments/cross_game_self_reflect/gemma_26b"
RESULTS = SWEEP_DIR / "results.jsonl"


def append(game: str, game_logs_dir: str, runtime_min: float) -> dict:
    summary_path = Path(game_logs_dir) / "evaluation_summary.json"
    if not summary_path.exists():
        sys.exit(f"summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    eps = summary.get("episodes", [])
    scores = [float(e.get("final_score", 0.0)) for e in eps] if eps else [0.0]
    final = max(scores)
    n_eps = len(eps)
    steps = int(summary.get("total_inference_calls", 0))

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    rows = (
        [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
        if RESULTS.exists()
        else []
    )
    row = {
        "experiment": len(rows) + 1,
        "variant": f"stage_d_self_reflect_{game}",
        "game": game,
        "evaluation_score": final,
        "game_score": final,
        "steps": steps,
        "runtime_min": runtime_min,
        "status": "KEEP",
        "description": (
            f"stage_d_self_reflect_{game}: Stage D + self-reflection ({game}) — "
            "vmem ON, planner ON, reflect_every=10. PR #62 cross-game test."
        ),
        "notes": f"max_eval={final:.2f}, {n_eps} ep, {steps} steps",
        "tags": ["cross_game_self_reflect", f"stage_d_self_reflect_{game}"],
        "wandb_url": "",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "config_name": "gemma_26b",
    }
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"Appended {row['experiment']}: {row['variant']} score={final:.2f} "
        f"({n_eps} eps, {steps} steps)"
    )
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--game-logs", required=True)
    ap.add_argument("--runtime-min", type=float, required=True)
    a = ap.parse_args()
    append(game=a.game, game_logs_dir=a.game_logs, runtime_min=a.runtime_min)
