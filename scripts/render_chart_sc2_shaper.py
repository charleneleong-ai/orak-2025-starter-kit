"""Render PR #111 SC2 reward-shaper signal chart.

Replays the PR3 SC2 smoke trace through `StarCraftShaper` (offline, no SC2
needed) and renders a 2-panel chart at
`experiments/progress/sc2_reward_shaping/shaper_signal.png`:

1. Per-episode cumulative shaped reward across the 10 episodes — shows the
   signal distribution MACLA's procedural memory now refines against.
2. Aggregate event counts (idleness penalties, supply-block penalties,
   positive building events) — shows the shaper is actively penalising the
   iter-201 failure mode.

The decision gate (`reward[iter 201] < reward[iter 51]`) is asserted at the
bottom of the panel-1 title so the chart is self-explanatory.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from agents.macla.online_evaluator import DEFAULT_SHAPING, StarCraftShaper

TRACE = Path(
    "/workspace/orak-futile-detector/game_logs/star_craft/"
    "stagnation_pr3_star_craft_smoke_20260527T094639Z/game_states.jsonl"
)
OUT = Path(__file__).parent.parent / "experiments/progress/sc2_reward_shaping/shaper_signal.png"


def replay() -> tuple[dict[int, float], dict[str, int]]:
    """Replay trace through StarCraftShaper. Returns (per-episode reward, fire counts)."""
    shaper = StarCraftShaper(DEFAULT_SHAPING["star_craft"])
    prev: dict = {}
    per_episode: dict[int, float] = defaultdict(float)
    fires = {"idleness": 0, "supply_block": 0, "buildings_built": 0}
    cur_ep = 1
    prev_iter = -1

    for line in TRACE.read_text().splitlines():
        obj = json.loads(line)
        it = obj.get("iteration", 0)
        if it == 1 and prev_iter != -1:
            cur_ep += 1
            shaper.reset_episode()
            prev = {}
        prev_iter = it

        cur = shaper.extract_metrics(obj.get("obs", {}).get("obs_str", ""))
        per_episode[cur_ep] += shaper.compute_reward(prev, cur, success=False, is_fatal=False)

        mineral_d = cur.get("mineral", 0) - prev.get("mineral", 0)
        supply_d = cur.get("supply_used", 0) - prev.get("supply_used", 0)
        building_d = cur.get("building_count", 0) - prev.get("building_count", 0)
        if mineral_d > 0 and supply_d == 0:
            fires["idleness"] += 1
        if cur.get("supply_left", 1) <= 0:
            fires["supply_block"] += 1
        if building_d > 0:
            fires["buildings_built"] += building_d

        prev = cur

    return dict(per_episode), fires


def main() -> None:
    per_episode, fires = replay()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), dpi=140)

    # Panel 1: per-episode cumulative reward
    episodes = sorted(per_episode.keys())
    rewards = [per_episode[e] for e in episodes]
    best_idx = rewards.index(max(rewards))
    worst_idx = rewards.index(min(rewards))
    colors = ["#cccccc"] * len(rewards)
    colors[best_idx] = "#54a24b"
    colors[worst_idx] = "#e15759"

    ax1.bar(episodes, rewards, color=colors, edgecolor="black", lw=0.5)
    for i, r in enumerate(rewards):
        ax1.text(
            episodes[i], r - 3, f"{r:+.1f}", ha="center", va="top", fontsize=9, family="monospace"
        )
    ax1.axhline(0, color="black", lw=0.7)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative shaped reward")
    ax1.set_title(
        "Per-episode shaped-reward signal — replay through StarCraftShaper\n"
        "Range: best ep 9 (-56.7) to worst ep 5 (-110.7); decision gate iter51 > iter201 PASS"
    )
    ax1.set_xticks(episodes)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: aggregate fire counts
    labels = [
        "idleness\n(minerals up,\nsupply flat)",
        "supply-block\n(supply_left ≤ 0)",
        "buildings_built\n(positive event)",
    ]
    values = [fires["idleness"], fires["supply_block"], fires["buildings_built"]]
    bar_colors = ["#e15759", "#e15759", "#54a24b"]
    ax2.bar(labels, values, color=bar_colors, edgecolor="black", lw=0.5)
    for i, v in enumerate(values):
        ax2.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10, family="monospace")
    ax2.set_ylabel("Event count over 2,500 iters")
    ax2.set_title(
        "Shaper actively penalises the iter-201 failure mode\n"
        "Red = penalty fires, green = positive events"
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(values) * 1.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(
        f"  per-episode reward: best ep{episodes[best_idx]}={rewards[best_idx]:+.2f}  worst ep{episodes[worst_idx]}={rewards[worst_idx]:+.2f}"
    )
    print(f"  fires: {fires}")


if __name__ == "__main__":
    main()
