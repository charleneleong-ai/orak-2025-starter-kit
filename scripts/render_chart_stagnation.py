"""Render PR3 progress-stagnation detector charts.

Two outputs, both in `experiments/progress/stagnation_detector/`:

1. `detector_selectivity.png` — retrospective replay of the three universal
   detectors (PR1 futile-action, PR2 repeated-plan, PR3 progress-stagnation)
   over an existing SC2 trace. Shows fires + distinct streaks per detector;
   PR3's macro-selectivity advantage reads directly off the streak counts.

2. `sc2_smoke_comparison.png` — cross-smoke comparison of the MACLA progress
   metrics (`procedures_learned`, `procedures_refined`, `avg_success_rate`,
   `successful_executions`) extracted from the final MACLA Stats line of
   each of the three SC2 smoke logs (PR2 detector, PR3 detector, PR111
   reward shaper). At n=1 per config the differences are within seed noise —
   the chart's role is to show that PR3 doesn't regress the procedural
   dynamics while still adding the selectivity advantage in chart (1).
"""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt

WORKTREE = Path("/workspace/orak-futile-detector")
TRACE = (
    WORKTREE / "game_logs/star_craft/stagnation_pr3_star_craft_smoke_20260527T094639Z/"
    "game_states.jsonl"
)
OUT_DIR = Path(__file__).parent.parent / "experiments/progress/stagnation_detector"
OUT_SELECTIVITY = OUT_DIR / "detector_selectivity.png"
OUT_COMPARISON = OUT_DIR / "sc2_smoke_comparison.png"

# Final-step MACLA stats line is the source of cross-smoke perf metrics.
SC2_SMOKES = [
    ("PR2  (PR1+PR2)", WORKTREE / "logs/futile_pr2_star_craft_smoke_20260526T163642Z.log"),
    ("PR3  (PR1+PR2+PR3)", WORKTREE / "logs/stagnation_pr3_star_craft_smoke_20260527T094639Z.log"),
    ("PR111 (+ shaper)", WORKTREE / "logs/sc2_reward_shaping_smoke_20260527T153806Z.log"),
]

FUTILE_WINDOW = 3
REPEATED_WINDOW = 4
STAGNATION_WINDOW = 20
SC2_STAGNATION_RE = re.compile(r"Supply used:?\s*(\d+)")
MACLA_STATS_RE = re.compile(r"MACLA Stats.*Step (\d+)\): (\{.*\})")


def replay(jsonl: Path) -> tuple[list[bool], list[bool], list[bool]]:
    """Re-run PR1/PR2/PR3 detector logic offline over the trace.

    Returns three per-iter fire bitmasks (same length as the trace).
    """
    obs_window: deque[int] = deque(maxlen=FUTILE_WINDOW)
    plan_window: deque[str] = deque(maxlen=REPEATED_WINDOW)
    progress_window: deque[float] = deque(maxlen=STAGNATION_WINDOW)
    pr1_fires: list[bool] = []
    pr2_fires: list[bool] = []
    pr3_fires: list[bool] = []

    for line in jsonl.read_text().splitlines():
        row = json.loads(line)
        obs = row.get("obs", {}).get("obs_str", "") or ""
        plan = row.get("action", "") or ""

        # PR1: byte-equality on last FUTILE_WINDOW observations
        obs_window.append(hash(obs))
        pr1_fires.append(len(obs_window) == FUTILE_WINDOW and len(set(obs_window)) == 1)

        # PR2: equality on last REPEATED_WINDOW chosen plans
        plan_window.append(plan)
        pr2_fires.append(len(plan_window) == REPEATED_WINDOW and len(set(plan_window)) == 1)

        # PR3: zero-variance on extracted progress signal (SC2 = Supply used)
        m = SC2_STAGNATION_RE.search(obs)
        if m:
            progress_window.append(float(m.group(1)))
            pr3_fires.append(
                len(progress_window) == STAGNATION_WINDOW and len(set(progress_window)) == 1
            )
        else:
            pr3_fires.append(False)

    return pr1_fires, pr2_fires, pr3_fires


def streak_count(fires: list[bool]) -> int:
    """Count contiguous True runs — each streak is one macro 'stuck' signal."""
    return sum(1 for i, f in enumerate(fires) if f and (i == 0 or not fires[i - 1]))


def parse_final_macla_stats(log_path: Path) -> dict[str, float]:
    """Extract the final 'MACLA Stats & Optimisation' dict from a smoke log.

    The stats are emitted as a Python-repr dict every 10 steps. We grab the
    last one and pull the four metrics the cross-smoke comparison cares about.
    """
    last_dict_text: str | None = None
    for m in MACLA_STATS_RE.finditer(log_path.read_text()):
        last_dict_text = m.group(2)
    if last_dict_text is None:
        return {}
    stats = eval(last_dict_text, {"__builtins__": {}}, {})
    return {
        "successful_executions": float(stats["agent_stats"]["successful_executions"]),
        "procedures_learned": float(stats["agent_stats"]["procedures_learned"]),
        "procedures_refined": float(stats["agent_stats"]["procedures_refined"]),
        "avg_success_rate": float(stats["optimisation"]["avg_procedure_success_rate"]),
    }


def render_selectivity() -> None:
    pr1, pr2, pr3 = replay(TRACE)
    n_iters = len(pr1)

    rows = [
        ("PR1  byte-equality (K=3)", sum(pr1), streak_count(pr1)),
        ("PR2  plan-equality  (K=4)", sum(pr2), streak_count(pr2)),
        ("PR3  progress-stagnation (K=20)", sum(pr3), streak_count(pr3)),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.0), dpi=140)
    y = list(range(len(rows)))
    fires = [r[1] for r in rows]
    streaks = [r[2] for r in rows]

    # Fires bar (thin, behind)
    ax.barh(y, fires, height=0.62, color="#4c78a8", edgecolor="black", lw=0.5, label="Fires")
    # Streak count as inset marker — thicker so it pops against the fires bar
    for i, s in enumerate(streaks):
        if s == 0:
            continue
        ax.scatter([s], [i], s=180, color="#f58518", edgecolor="black", lw=1, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], family="monospace", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(f"Count over {n_iters:,} SC2 iters  (blue = fires, orange = distinct streaks)")
    ax.set_title(
        "Universal pathology-guard family — retrospective replay on SC2 trace\n"
        "PR3 fires ~2x PR2 but in ~8x fewer streaks → macro-level (longer-horizon) signal"
    )

    # Annotate counts + avg streak length
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    for i, (_label, f, s) in enumerate(rows):
        avg = (f / s) if s else 0.0
        ax.text(
            xmax + 0.02 * span,
            i,
            f"fires={f:<4d}  streaks={s:<4d}  avg_len={avg:>5.1f}",
            va="center",
            fontsize=9,
            family="monospace",
        )

    ax.set_xlim(xmin, xmax + 0.55 * span)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_SELECTIVITY, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_SELECTIVITY}  ({n_iters} iters replayed)")
    for label, f, s in rows:
        print(f"  {label:40s}  fires={f:<4d}  streaks={s:<4d}")


def render_smoke_comparison() -> None:
    """Bar chart of MACLA progress metrics across the three SC2 smokes.

    Goal is honesty about overall perf: all three smokes show 0 game-wins
    (SC2 ceiling = base-model capability, not detector quality). The chart
    confirms PR3 doesn't regress the procedural dynamics it inherits from
    PR1+PR2 — flat-to-slightly-up vs the prior smoke at n=1.
    """
    rows = [(label, parse_final_macla_stats(log)) for label, log in SC2_SMOKES]

    metrics = [
        ("successful_executions", "successful\nexecutions"),
        ("procedures_learned", "procedures\nlearned"),
        ("procedures_refined", "procedures\nrefined"),
        ("avg_success_rate", "avg success\nrate"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), dpi=140)
    palette = ["#9e9e9e", "#4c78a8", "#54a24b"]

    for ax, (key, title) in zip(axes, metrics):
        values = [stats.get(key, 0.0) for _, stats in rows]
        ax.bar(range(len(rows)), values, color=palette, edgecolor="black", lw=0.5)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        # Annotate value on top of each bar
        for i, v in enumerate(values):
            ax.text(
                i,
                v,
                f"{v:.3g}" if v < 10 else f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                family="monospace",
            )
        # Pad y-axis to give label room
        ymax = max(values) if max(values) > 0 else 1.0
        ax.set_ylim(0, ymax * 1.18)

    fig.suptitle(
        "SC2 smoke comparison — 2,500 steps each, n=1, gemma-26B-AWQ, Protoss vs Zerg D4\n"
        "All three smokes show 0 game-wins (SC2 ceiling is base-model capability). "
        "PR3 holds the procedural-dynamics line vs PR2 — selectivity advantage is the lift.",
        fontsize=10,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_COMPARISON, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_COMPARISON}")
    for label, stats in rows:
        print(f"  {label:25s}  {stats}")


def main() -> None:
    render_selectivity()
    render_smoke_comparison()


if __name__ == "__main__":
    main()
