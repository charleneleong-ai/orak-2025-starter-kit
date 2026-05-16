"""Stage K live learning-curve chart.

Scans /tmp/orak-planner-prompt/pokemon_red/pr_stage_j_cumulative_pokemon_iter*/
for evaluation_summary.json files, plots per-iter score across n=5.
Writes both HTML (served at localhost:9000) and PNG (committed to PR body).

Run as a daemon: regenerates every 60s while the sweep is in flight.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

GAME_DIR = Path("/tmp/orak-planner-prompt/pokemon_red")
ITER_PATTERN = re.compile(r"pr_stage_j_cumulative_pokemon_iter(\d+)_")
OUT_DIR = Path(__file__).parent / "gemma_26b"
HTML_OUT = OUT_DIR / "progress.html"
PNG_OUT = Path(__file__).parent.parent / "progress" / "stage_k_cumulative_memory" / "progress.png"

# Reference baselines for the chart (from cross-stage diagnosis doc).
BASELINES = {
    "Stage D (n=1)": 57.14,
    "Stage H Qwen3.5-Int4 (n=3)": 57.14,
    "Stage J Qwen3-Thinking (n=3)": 28.57,
    "Stage G procedure-escape (n=3)": 47.62,
    "Stage B' no procedures (n=3)": 42.86,
}


def collect_iters() -> list[tuple[int, str, float | None, str]]:
    """Return (iter_num, run_id, score_pct_or_none, status) sorted by iter."""
    if not GAME_DIR.exists():
        return []
    rows: list[tuple[int, str, float | None, str]] = []
    for d in sorted(GAME_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = ITER_PATTERN.match(d.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        summary = d / "evaluation_summary.json"
        if summary.exists():
            try:
                payload = json.loads(summary.read_text())
                eps = payload.get("episodes", [])
                raw = max((float(e.get("final_score", 0.0)) for e in eps), default=0.0)
                score = (raw / 7.0) * 100
                rows.append((iter_num, d.name, score, "done"))
                continue
            except (json.JSONDecodeError, ValueError):
                pass
        # In-flight: peek at game_states.jsonl to know how far it's gotten
        gs = d / "game_states.jsonl"
        if gs.exists():
            try:
                steps = sum(1 for _ in gs.open())
                rows.append((iter_num, d.name, None, f"running ({steps} steps)"))
                continue
            except OSError:
                pass
        rows.append((iter_num, d.name, None, "starting"))
    return sorted(rows, key=lambda r: r[0])


def render_plotly(iters: list[tuple[int, str, float | None, str]]) -> str:
    """Return HTML string for the Plotly chart."""
    xs = [r[0] for r in iters]
    scores = [r[2] if r[2] is not None else None for r in iters]
    statuses = [r[3] for r in iters]
    hover = [
        f"iter {n}<br>{run_id}<br>{f'{s:.2f}%' if s is not None else status}"
        for n, run_id, s, status in iters
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=scores,
            mode="lines+markers+text",
            text=[f"{s:.1f}%" if s is not None else st for s, st in zip(scores, statuses)],
            textposition="top center",
            hovertext=hover,
            hoverinfo="text",
            line={"width": 3, "color": "#16a34a"},
            marker={"size": 14, "color": "#16a34a", "line": {"width": 2, "color": "#0a4a1f"}},
            name="Stage K cumulative",
        )
    )
    colors = ["#999", "#666", "#d96b3a", "#7d4d9e", "#fb923c"]
    for (label, score), color in zip(BASELINES.items(), colors):
        fig.add_hline(
            y=score,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label} = {score:.2f}%",
            annotation_position="right",
            annotation={"font": {"size": 10, "color": color}},
        )
    fig.update_layout(
        title=(
            "Stage K — cumulative cross-episode memory (n=5, Gemma-26B AWQ, pokemon Stage D)<br>"
            "<sub>iter N inherits iter N-1's EnhancedHierarchicalMemorySystem via --load-checkpoint --prev-run-id</sub>"
        ),
        xaxis={"title": "iter", "tickmode": "array", "tickvals": list(range(1, 6))},
        yaxis={"title": "evaluation score (%)", "range": [0, 100]},
        height=520,
        margin={"l": 70, "r": 220, "t": 90, "b": 60},
        plot_bgcolor="#fafafa",
        showlegend=False,
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


def render_matplotlib(iters: list[tuple[int, str, float | None, str]]) -> None:
    """Write a static PNG (for committed PR body)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = [r[0] for r in iters]
    ys = [r[2] for r in iters]

    # Done iters get the colored line; in-flight iters get hollow markers.
    done_x = [x for x, y in zip(xs, ys) if y is not None]
    done_y = [y for y in ys if y is not None]
    pending_x = [x for x, y in zip(xs, ys) if y is None]
    pending_y_floor = [5] * len(pending_x)

    if done_x:
        ax.plot(
            done_x,
            done_y,
            marker="o",
            markersize=12,
            linewidth=3,
            color="#16a34a",
            markeredgecolor="#0a4a1f",
            markeredgewidth=2,
            label="Stage K (done)",
            zorder=5,
        )
        for x, y in zip(done_x, done_y):
            ax.annotate(
                f"{y:.1f}%",
                xy=(x, y),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
    if pending_x:
        ax.scatter(
            pending_x,
            pending_y_floor,
            marker="o",
            s=120,
            facecolors="none",
            edgecolors="#999",
            linewidths=2,
            label="Stage K (pending)",
            zorder=4,
        )

    colors = ["#999", "#666", "#d96b3a", "#7d4d9e", "#fb923c"]
    for (label, score), color in zip(BASELINES.items(), colors):
        ax.axhline(score, linestyle="--", color=color, alpha=0.55, linewidth=1.2)
        ax.text(5.15, score, f"{label} = {score:.2f}%", fontsize=9, color=color, va="center")

    ax.set_xlabel("iter (each inherits previous iter's memory)", fontsize=11)
    ax.set_ylabel("evaluation score (%, higher = more milestones)", fontsize=11)
    ax.set_title(
        "Stage K — cumulative cross-episode memory (n=5)\n"
        "Gemma-26B AWQ, pokemon Stage D, 300 steps/iter",
        fontsize=12,
    )
    ax.set_ylim(0, 100)
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks(list(range(1, 6)))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PNG_OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_chart() -> tuple[int, int]:
    """Write HTML + PNG; return (done_count, total_seen)."""
    iters = collect_iters()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(render_plotly(iters))
    render_matplotlib(iters)
    done = sum(1 for _, _, s, _ in iters if s is not None)
    return done, len(iters)


def loop(interval_s: int = 60, max_done: int = 5) -> None:
    while True:
        try:
            done, total = write_chart()
            ts = time.strftime("%H:%M:%SZ", time.gmtime())
            print(f"[{ts}] chart: done={done}/{total} seen", flush=True)
            if done >= max_done:
                print(f"[{ts}] all {max_done} iters done — exiting loop.", flush=True)
                return
        except Exception as e:  # pragma: no cover — daemon must not die
            print(f"[chart] error: {e}", flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    import sys

    if "--once" in sys.argv:
        done, total = write_chart()
        print(f"chart: done={done}/{total}")
    else:
        loop()
