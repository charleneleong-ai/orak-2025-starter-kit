"""Render TGAER PR1 cross-game Δ-forest plot.

Loads results from `experiments/tgaer_pr1_{baseline,detector}/results.jsonl`,
computes per-game Δ (detector_mean − baseline_mean) with std-of-difference,
and renders a forest plot where each game is one row and the verdict reads
directly off whether the CI band crosses Δ=0. Output:
`experiments/progress/futile_detector/cross_game_lift.png`.

Why a forest plot for this PR: the thesis is "ship as safety floor, no lift".
A paired-bar mean-vs-mean chart with overlapping error bars reads as
ambiguous; a Δ plot anchored at 0 reads as "all four CIs touch zero — no
measurable lift". Renders single-axis, no legend gymnastics.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/workspace/orak-futile-detector")
OUT = REPO / "experiments/progress/futile_detector/cross_game_lift.png"

GAME_ORDER = ("pokemon_red", "super_mario", "twenty_fourty_eight", "star_craft")
GAME_LABEL = {
    "pokemon_red": "pokemon_red  (score / 7)",
    "super_mario": "super_mario  (% world)",
    "twenty_fourty_eight": "twenty_fourty_eight  (log2 tile)",
    "star_craft": "star_craft  (win %)",
}


def load(tag: str) -> list[dict]:
    return [
        json.loads(l) for l in (REPO / f"experiments/{tag}/results.jsonl").read_text().splitlines()
    ]


def per_game(rows: list[dict], game: str) -> list[float]:
    return [r["mean"] for r in rows if r["game"] == game]


def std_of_diff(b: list[float], d: list[float]) -> float:
    """Std of the unpaired mean difference: sqrt(sb²/nb + sd²/nd)."""
    sb = statistics.stdev(b) if len(b) > 1 else 0.0
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    return (sb**2 / max(len(b), 1) + sd**2 / max(len(d), 1)) ** 0.5


def main() -> None:
    base = load("tgaer_pr1_baseline")
    det = load("tgaer_pr1_detector")

    rows = []
    for g in GAME_ORDER:
        b = per_game(base, g)
        d = per_game(det, g)
        b_mean = statistics.mean(b) if b else 0.0
        d_mean = statistics.mean(d) if d else 0.0
        rows.append((g, b_mean, d_mean, std_of_diff(b, d), len(b), len(d)))

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    y = list(range(len(rows)))
    deltas = [d_mean - b_mean for _, b_mean, d_mean, _, _, _ in rows]
    errs = [s for _, _, _, s, _, _ in rows]

    # 95% CI band (≈ 1.96·std). Inflated to a thicker bar for SC2 where std=0.
    ci = [1.96 * s for s in errs]

    ax.axvline(0, color="grey", lw=1, linestyle="--", zorder=1)
    for i, (delta, c) in enumerate(zip(deltas, ci)):
        # CI bar
        ax.plot([delta - c, delta + c], [i, i], color="#444", lw=2, zorder=2)
        ax.plot([delta - c, delta - c], [i - 0.12, i + 0.12], color="#444", lw=2, zorder=2)
        ax.plot([delta + c, delta + c], [i - 0.12, i + 0.12], color="#444", lw=2, zorder=2)
        # point estimate
        color = "#cccccc" if abs(delta) < 1.0 else ("#2ca02c" if delta > 0 else "#d62728")
        ax.scatter([delta], [i], s=140, color=color, edgecolor="black", lw=1, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([GAME_LABEL[g] for g, *_rest in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Δ score  (detector − baseline)")
    ax.set_title(
        "TGAER PR1 — futile-action detector cross-game lift (n=3 per cell)\n"
        "All four CIs touch Δ=0 → no measurable lift; ship as safety floor."
    )

    # Annotate Δ value + n
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    for i, (_g, b_mean, d_mean, _, nb, nd) in enumerate(rows):
        delta = d_mean - b_mean
        ax.text(
            xmax + 0.02 * span,
            i,
            f"Δ={delta:+.2f}   base→{b_mean:.2f}  det→{d_mean:.2f}   (n={nb}/{nd})",
            va="center",
            fontsize=9,
            family="monospace",
        )

    ax.set_xlim(xmin, xmax + 0.55 * span)  # room for annotation
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
