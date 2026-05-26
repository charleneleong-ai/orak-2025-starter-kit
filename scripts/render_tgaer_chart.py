"""Render TGAER PR1 cross-game baseline-vs-detector bar chart.

Reads `experiments/tgaer_pr1_{baseline,detector}/results.jsonl`, computes per-game
mean ± std across all n rolls per side, and renders one `Milestone` per game
via `autoresearch.compare.plot_milestone_bars`. Output: a PR-body PNG at
`experiments/progress/tgaer_pr1/cross_game_lift.png`.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from autoresearch.compare import Milestone, plot_milestone_bars

REPO = Path("/workspace/orak-futile-detector")
OUT = REPO / "experiments/progress/tgaer_pr1/cross_game_lift.png"

GAME_ORDER = ("pokemon_red", "super_mario", "twenty_fourty_eight", "star_craft")
GAME_LABEL = {
    "pokemon_red": "pokemon_red\n(score / 7)",
    "super_mario": "super_mario\n(% world progress)",
    "twenty_fourty_eight": "twenty_fourty_eight\n(log2 max-tile %)",
    "star_craft": "star_craft\n(win %)",
}


def load(tag: str) -> list[dict]:
    return [json.loads(l) for l in (REPO / f"experiments/{tag}/results.jsonl").read_text().splitlines()]


def per_game(rows: list[dict], game: str) -> list[float]:
    return [r["mean"] for r in rows if r["game"] == game]


def verdict_for(delta: float, base_scores: list[float], det_scores: list[float]) -> str:
    if not det_scores or all(s == 0 for s in base_scores + det_scores):
        return "DISCARD" if not det_scores else "BASELINE"
    if abs(delta) < 1.0:
        return "BASELINE"
    return "KEEP" if delta > 0 else "DISCARD"


def main() -> None:
    base = load("tgaer_pr1_baseline")
    det = load("tgaer_pr1_detector")

    milestones: list[Milestone] = []
    for g in GAME_ORDER:
        b = per_game(base, g)
        d = per_game(det, g)
        b_mean = statistics.mean(b) if b else 0.0
        d_mean = statistics.mean(d) if d else 0.0
        b_std = statistics.stdev(b) if len(b) > 1 else 0.0
        d_std = statistics.stdev(d) if len(d) > 1 else 0.0
        delta = d_mean - b_mean
        milestones.append(
            Milestone(
                label=GAME_LABEL[g],
                metrics={"baseline (no detector)": b_mean, "detector (PR1)": d_mean},
                metric_stds={"baseline (no detector)": b_std, "detector (PR1)": d_std},
                metric_scores={"baseline (no detector)": b, "detector (PR1)": d},
                description=f"Δ={delta:+.2f}  base n={len(b)} / det n={len(d)}",
                verdict=verdict_for(delta, b, d),
                n=len(d),
            )
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plot_milestone_bars(
        milestones,
        primary_metric="detector (PR1)",
        out_path=OUT,
        title="TGAER PR1 — universal futile-action detector vs baseline (n=3 per cell)",
        ylabel="Mean score (game-native units)",
        show_descriptions=True,
        figsize=(13.0, 6.5),
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
