"""Append a mario rerun row to results.jsonl.

Mirrors experiments/pr31_2048_rerun/append.py — picks final_score
from evaluation_summary.json (max across episodes when multi-ep).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO / "experiments/pr31_mario_rerun/gemma_26b"
RESULTS = SWEEP_DIR / "results.jsonl"


def _existing() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]


def append(
    variant: str,
    description: str,
    game_logs_dir: str,
    runtime_min: float | None = None,
    status_override: str | None = None,
) -> dict:
    summary_path = Path(game_logs_dir) / "evaluation_summary.json"
    if not summary_path.exists():
        sys.exit(f"summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    eps = summary.get("episodes", [])
    if eps:
        scores = [float(e.get("final_score", 0.0)) for e in eps]
        final = max(scores) if scores else 0.0
        n_eps = len(eps)
    else:
        final = 0.0
        n_eps = 0
    steps = int(summary.get("total_inference_calls", 0))

    if runtime_min is None:
        log = Path(game_logs_dir) / "evaluation.log"
        runtime_min = (
            (log.stat().st_mtime - Path(game_logs_dir).stat().st_mtime) / 60.0
            if log.exists()
            else 0.0
        )

    rows = _existing()
    best = max((r.get("evaluation_score", 0) for r in rows), default=0.0)
    status = status_override or ("KEEP" if final > best else "DISCARD")
    if not rows and not status_override:
        status = "BASELINE"

    row = {
        "experiment": len(rows) + 1,
        "variant": variant,
        "game": "super_mario",
        "evaluation_score": final,
        "game_score": final,
        "steps": steps,
        "runtime_min": runtime_min,
        "status": status,
        "description": f"{variant}: {description}",
        "notes": f"max_eval={final:.2f}, {n_eps} ep, {steps} steps",
        "tags": ["pr31_mario_rerun", variant],
        "wandb_url": "",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "config_name": "gemma_26b",
    }
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"Appended {row['experiment']}: {variant} score={final:.2f} "
        f"({n_eps} eps, {steps} steps) status={status}"
    )
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--game-logs", required=True)
    ap.add_argument("--runtime-min", type=float, default=None)
    ap.add_argument("--status", dest="status_override", default=None)
    a = ap.parse_args()
    append(
        variant=a.variant,
        description=a.description,
        game_logs_dir=a.game_logs,
        runtime_min=a.runtime_min,
        status_override=a.status_override,
    )
