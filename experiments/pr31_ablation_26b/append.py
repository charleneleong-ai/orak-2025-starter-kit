"""Append a PR #31 ablation run to results.jsonl + re-render plot.

Usage:
    uv run python experiments/pr31_ablation_26b/append.py \
        --variant stage_d_26b \
        --description "Stage D 26B (vmem on, planner on, baseline rerun)" \
        --game-logs /tmp/orak-planner-prompt/game_logs/pokemon_red/<run_id>

Reads the run's evaluation_summary.json for final_score + step count,
then appends a row in the autoresearch shape so render() can plot it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO / "experiments/pr31_ablation_26b/gemma_26b"
RESULTS = SWEEP_DIR / "results.jsonl"
PNG_OUT = REPO / "experiments/pr31_ablation_26b/progress.png"
PNG_COMMIT = REPO / "docs/experiments/gemma/plots/pr31_ablation_26b.png"


def _existing() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]


def append(
    variant: str,
    description: str,
    game_logs_dir: str,
    runtime_min: float | None = None,
    wandb_url: str | None = None,
    status_override: str | None = None,
) -> dict:
    summary_path = Path(game_logs_dir) / "evaluation_summary.json"
    if not summary_path.exists():
        sys.exit(f"summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    ep = summary.get("episodes", [{}])[-1] if summary.get("episodes") else {}
    final = float(ep.get("final_score", 0.0))
    steps = int(ep.get("inference_calls", summary.get("total_inference_calls", 0)))
    eval_pct = (final / 7.0) * 100.0

    if runtime_min is None:
        log = Path(game_logs_dir) / "evaluation.log"
        if log.exists():
            runtime_min = (log.stat().st_mtime - Path(game_logs_dir).stat().st_mtime) / 60.0
        else:
            runtime_min = 0.0

    rows = _existing()
    best = max((r.get("evaluation_score", 0) for r in rows), default=0.0)
    status = status_override or ("KEEP" if eval_pct > best else "DISCARD")
    if not rows and not status_override:
        status = "BASELINE"

    row = {
        "experiment": len(rows) + 1,
        "variant": variant,
        "game": "pokemon_red",
        "evaluation_score": eval_pct,
        "game_score": final,
        "steps": steps,
        "runtime_min": runtime_min,
        "status": status,
        "description": f"{variant}: {description}",
        "notes": (
            f"max_eval={eval_pct:.2f} ({final}/7), "
            f"{summary.get('evaluation_episodes', 1)} ep, {steps} steps"
        ),
        "tags": ["pr31_ablation_26b", variant],
        "wandb_url": wandb_url or "",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_name": "gemma_26b",
    }
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"Appended {row['experiment']}: {variant} score={final}/7 ({eval_pct:.2f}%) status={status}")
    return row


def render() -> Path:
    from autoresearch import render as ar
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    out = ar.render(
        experiments_dir=str(REPO / "experiments"),
        tag="pr31_ablation_26b",
        config_name="gemma_26b",
        out=str(PNG_OUT),
        title="PR #31 ablation — pokemon Stage D rerun + 26B ablations",
        score_field="evaluation_score",
        score_label="evaluation_score (% of 7)",
    )
    PNG_COMMIT.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(PNG_OUT, PNG_COMMIT)
    print(f"Rendered: {out}\n  Committed mirror: {PNG_COMMIT}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--game-logs", required=True)
    ap.add_argument("--runtime-min", type=float, default=None)
    ap.add_argument("--wandb-url", default=None)
    ap.add_argument("--status", dest="status_override", default=None)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    append(
        variant=a.variant,
        description=a.description,
        game_logs_dir=a.game_logs,
        runtime_min=a.runtime_min,
        wandb_url=a.wandb_url,
        status_override=a.status_override,
    )
    if not a.no_render:
        render()
