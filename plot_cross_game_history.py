"""Cross-game self-reflect: historical baselines (PR #31 Stage A->D) +
PR #62/#64 self-reflect runs (v2/v3) side by side, per game.

Pulls from the existing results.jsonl files; no rerun required.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def pick(rows, variant=None, game=None, tag_substr=None):
    for r in rows:
        if variant and r.get("variant") == variant and (game is None or r.get("game") == game):
            return r
        if tag_substr and any(tag_substr in t for t in r.get("tags", [])):
            return r
    return None


# Per-game series: (label, score, color). PR #31 ablation = baselines; v2/v3 = cross_game self-reflect runs.
GAME_SERIES: dict[str, list[tuple[str, float, str]]] = {}

pr31_pokemon = load("experiments/pr31_ablation_26b/gemma_26b/results.jsonl")
pr31_mario = load("experiments/pr31_mario_rerun/gemma_26b/results.jsonl")
pr31_2048 = load("experiments/pr31_2048_rerun/gemma_26b/results.jsonl")
crossgame = load("experiments/cross_game_self_reflect/gemma_26b/results.jsonl")


def score_of(row, default=float("nan")):
    return row.get("evaluation_score", default) if row else default


# Pokemon. Stage C scored 0 (vmem-only collapse); drawn as a real bar to keep the story.
pok = pr31_pokemon
pok_v2 = pick(crossgame, variant="stage_d_self_reflect_pokemon_red")  # first occurrence = v2
pok_v3 = None
for r in crossgame:
    if r.get("variant") == "stage_d_self_reflect_pokemon_red":
        pok_v3 = r  # last occurrence = v3
GAME_SERIES["pokemon_red"] = [
    ("Stage A\n(model only)", score_of(pick(pok, variant="stage_a_26b")), "#cccccc"),
    ("Stage B\n(planner)", score_of(pick(pok, variant="stage_b_26b")), "#cccccc"),
    ("Stage C\n(vmem)", score_of(pick(pok, variant="stage_c_26b")), "#cccccc"),
    ("Stage D\n(full stack)", score_of(pick(pok, variant="stage_d_26b")), "#4c8dbf"),
    ("D + reflect\nv2 n=1", score_of(pok_v2), "#f0a868"),
    ("D + reflect\nv3 (adapter)", score_of(pok_v3), "#d96b3a"),
    ("Stage E\n(LG+verify)", 4.0 / 7.0 * 100, "#7d4d9e"),
    ("D + reflect\n(600 steps)", 4.0 / 7.0 * 100, "#8b5cf6"),
    ("Stage F\n(plan-do-check)", 2.0 / 7.0 * 100, "#a78bfa"),
    ("Stage B'\n(no proc) n=3", 42.857142857142854, "#fb923c"),
    ("Stage G\n(proc-escape) n=3", 47.61666666666667, "#16a34a"),
]

mar = pr31_mario
mar_v2 = pick(crossgame, variant="stage_d_self_reflect_super_mario")
mar_v3 = None
for r in crossgame:
    if r.get("variant") == "stage_d_self_reflect_super_mario":
        mar_v3 = r
GAME_SERIES["super_mario"] = [
    ("Stage A\n(model only)", score_of(pick(mar, variant="stage_a_mario")), "#cccccc"),
    ("Stage B\n(planner)", score_of(pick(mar, variant="stage_b_mario")), "#cccccc"),
    ("Stage C\n(vmem)", score_of(pick(mar, variant="stage_c_mario")), "#cccccc"),
    ("Stage D\n(full stack)", score_of(pick(mar, variant="stage_d_mario")), "#4c8dbf"),
    ("D + reflect\nv2 n=1", score_of(mar_v2), "#f0a868"),
    ("D + reflect\nv3 (adapter,\nevery 30)", score_of(mar_v3), "#d96b3a"),
    ("Stage B'\n(no proc) n=3", 27.49, "#fb923c"),
]

t48 = pr31_2048
t48_v2 = pick(crossgame, variant="stage_d_self_reflect_twenty_fourty_eight")
t48_v3 = None
for r in crossgame:
    if r.get("variant") == "stage_d_self_reflect_twenty_fourty_eight":
        t48_v3 = r
GAME_SERIES["twenty_fourty_eight"] = [
    ("Stage A\n(model only)", score_of(pick(t48, variant="stage_a_2048")), "#cccccc"),
    ("Stage B\n(planner)", score_of(pick(t48, variant="stage_b_2048")), "#cccccc"),
    ("Stage C\n(vmem)", score_of(pick(t48, variant="stage_c_2048")), "#cccccc"),
    ("Stage D\n(full stack)", score_of(pick(t48, variant="stage_d_2048")), "#4c8dbf"),
    ("D + reflect\nv2 n=1", score_of(t48_v2), "#f0a868"),
    ("D + reflect\nv3 (adapter\nOFF)", score_of(t48_v3), "#d96b3a"),
    ("Stage B'\n(no proc) n=3", 60.61, "#fb923c"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

titles = {
    "pokemon_red": "Pokemon Red (max_tile→% of 7 milestones)",
    "super_mario": "Super Mario (eval % of route)",
    "twenty_fourty_eight": "2048 (log2 max_tile %)",
}

for ax, (game, series) in zip(axes, GAME_SERIES.items(), strict=True):
    labels = [s[0] for s in series]
    scores = [s[1] for s in series]
    colors = [s[2] for s in series]
    bars = ax.bar(labels, scores, color=colors, edgecolor="#333", linewidth=0.6)
    for bar, score in zip(bars, scores, strict=True):
        if not isinstance(score, float) or score != score:  # NaN check
            continue
        ax.annotate(
            f"{score:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_title(titles[game], fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)

axes[0].set_ylabel("Evaluation score (%, higher is better)", fontsize=10)
fig.suptitle(
    "Cross-game self-reflect + Stage B' (no procedure cache) vs PR #31 Stage A→D baselines "
    "(gemma-4-26B-A4B AWQ, 300 steps)",
    fontsize=13,
)

# Footer legend
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor="#cccccc", edgecolor="#333", label="PR #31 ablation baseline"),
    Patch(facecolor="#4c8dbf", edgecolor="#333", label="Stage D (full stack) — the comparison anchor"),
    Patch(facecolor="#f0a868", edgecolor="#333", label="D + reflect v2 (always-on, every 10)"),
    Patch(facecolor="#d96b3a", edgecolor="#333", label="D + reflect v3 (per-game adapter recommendation)"),
    Patch(facecolor="#7d4d9e", edgecolor="#333", label="Stage E (PR #66) LangGraph + verify_action — pokemon only"),
    Patch(facecolor="#8b5cf6", edgecolor="#333", label="D + reflect 600 steps (PR #64 follow-up #3) — pokemon only"),
    Patch(facecolor="#a78bfa", edgecolor="#333", label="Stage F (PR #67) plan-do-check — pokemon only"),
    Patch(facecolor="#fb923c", edgecolor="#333", label="Stage B' (PR #69) no procedure cache, n=3 mean"),
    Patch(facecolor="#16a34a", edgecolor="#333", label="Stage G (PR #70) procedure-escape K=5/N=50, n=3 mean — pokemon only"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
    frameon=False,
)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])

out = Path("docs/experiments/gemma/plots/pr64_v3_crossgame.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
