"""Custom matplotlib plot for the PR #31 ablation rolling comment.

Shows stages in conceptual order (A → C → B → D → D++) as horizontal bars
with the prior 14.29% plateau as a reference line. Replaces the autoresearch
default chart for this sweep — it's a wider/clearer comparison view.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments/pr31_ablation_26b/gemma_26b/results.jsonl"
PNG_LOCAL = REPO / "experiments/pr31_ablation_26b/progress.png"
PNG_COMMIT = REPO / "docs/experiments/gemma/plots/pr31_ablation_26b.png"
PRIOR_PLATEAU = 14.29  # PR #31 v14b
ORDER = ["stage_a_26b", "stage_c_26b", "stage_b_26b", "stage_d_26b", "stage_d_plus_26b"]
LABELS = {
    "stage_a_26b": "Stage A\n(model only)",
    "stage_c_26b": "Stage C\n(+vmem)",
    "stage_b_26b": "Stage B\n(+planner)",
    "stage_d_26b": "Stage D\n(both, 300st)",
    "stage_d_plus_26b": "Stage D++\n(both, 600st + grace)",
}
COLOURS = {
    "stage_a_26b": "#9CA3AF",
    "stage_c_26b": "#60A5FA",
    "stage_b_26b": "#A78BFA",
    "stage_d_26b": "#10B981",
    "stage_d_plus_26b": "#059669",
}


def render() -> Path:
    rows = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["variant"]] = r

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ys = list(range(len(ORDER)))
    completed = []
    pending = []
    for y, v in zip(ys, ORDER):
        r = rows.get(v)
        if r is None:
            ax.barh(y, 100, color="#F3F4F6", edgecolor="#D1D5DB", linewidth=1)
            ax.text(2, y, "pending", va="center", color="#9CA3AF",
                    fontsize=10, style="italic")
            pending.append(v)
        else:
            score_pct = r["evaluation_score"]
            game = r["game_score"]
            steps = r["steps"]
            ax.barh(y, score_pct, color=COLOURS[v], edgecolor="#1F2937",
                    linewidth=1, alpha=0.92)
            label = f"  {game:.0f}/7  ·  {score_pct:.2f}%  ·  {steps} steps"
            text_x = score_pct + 1
            ax.text(text_x, y, label, va="center", fontsize=11,
                    color="#1F2937", fontweight="bold")
            completed.append(v)

    # Prior plateau reference line
    ax.axvline(PRIOR_PLATEAU, color="#DC2626", linestyle=":", linewidth=1.5,
               alpha=0.8)
    ax.text(PRIOR_PLATEAU + 0.6, -0.55,
            f"prior PR #31 plateau ({PRIOR_PLATEAU}%)",
            color="#DC2626", fontsize=9, va="center")

    # Cosmetics
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[v] for v in ORDER], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("evaluation_score (% of 7 milestones)", fontsize=11)
    ax.set_title(
        f"PR #31 ablation — pokemon Stage D rerun + 26B ablations\n"
        f"{len(completed)}/{len(ORDER)} stages complete",
        fontsize=13, pad=14, loc="left",
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    PNG_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    PNG_COMMIT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_LOCAL, dpi=140, bbox_inches="tight")
    fig.savefig(PNG_COMMIT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered: {PNG_LOCAL}\n  Mirror: {PNG_COMMIT}")
    return PNG_LOCAL


if __name__ == "__main__":
    render()
