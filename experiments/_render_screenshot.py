"""Standalone matplotlib renderer for the autoresearch progress PNG.

Reads `experiments/<TAG>/results.jsonl` and renders a static PNG that mirrors
the Plotly chart's visual encoding (status colour + best-run halo +
kill_reason inline). Produces per-game subplots when the JSONL contains
multiple games (orak: 2048, super_mario, pokemon_red). Runs without Plotly
+ kaleido + Chrome — clean fallback when the browser-based exporter
deadlocks or in headless CI.

Usage:
    .venv/bin/python experiments/_render_screenshot.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from autoresearch.results import (
    KILL_GPU_SLOW,
    KILL_GPU_SPIKE,
    KILL_LOSS_BLOWUP,
    KILL_NO_LEARNING,
    KILL_POLICY_DIVERGENCE,
    categorize_kill_reason,
)

# ────────────────────────── EDIT FOR YOUR PROJECT ──────────────────────────
TAG = "unified_macla"
SCORE_FIELD = "evaluation_score"  # 0-100 normalised in orak
SCORE_LABEL = "Evaluation score (higher is better)"
TITLE = "Orak MACLA Autoresearch — gemma-4-E4B-it / A100-40GB"
ROOT = Path(__file__).resolve().parent.parent


def _paths_for(config_name: str | None) -> tuple[Path, Path, str]:
    """Resolve (results.jsonl, output.png, title_suffix) for a given config.
    Pass `config_name=None` for flat layout (single sweep per TAG).
    Per-config layout: experiments/<TAG>/<config_name>/results.jsonl."""
    if config_name:
        results = ROOT / "experiments" / TAG / config_name / "results.jsonl"
        out = ROOT / "experiments" / TAG / config_name / "progress_static.png"
        suffix = f" — {config_name}"
    else:
        results = ROOT / "experiments" / TAG / "results.jsonl"
        out = ROOT / "experiments" / TAG / "progress_static.png"
        suffix = ""
    return results, out, suffix
# ──────────────────────────────────────────────────────────────────────────


_STATUS_STYLE = {
    "DISCARD":    {"color": "#cccccc", "line_color": "#999",    "text_color": "#777"},
    "KEEP":       {"color": "#2ecc71", "line_color": "black",   "text_color": "#1a7a3a"},
    "BASELINE":   {"color": "#2ecc71", "line_color": "black",   "text_color": "#1a7a3a"},
    "RUNNING":    {"color": "#f1c40f", "line_color": "#9a7d0a", "text_color": "#7d6608"},
    "EARLY_KILL": {"color": "#7f8c8d", "line_color": "#34495e", "text_color": "#34495e"},
    "CRASH":      {"color": "#e74c3c", "line_color": "#922b21", "text_color": "#922b21"},
}


def _kill_tag(kill_reason: str) -> str:
    """Map a long triage reason to a short category for the inline label.

    Orak-specific triage labels (plateau / below-baseline / iter-timeout /
    no-improvement) take precedence — they describe orak's *specific*
    triage triggers, which the upstream `categorize_kill_reason` would
    otherwise collapse into the more-generic ``no_learning`` bucket
    (eg. "below baseline gate" matches upstream's ``"baseline"`` rule).

    Anything that doesn't match an orak trigger falls through to the
    upstream categoriser, which handles the gemma4-style KL/loss/GPU
    patterns that orak doesn't emit but might see if a sweep ever
    bridges the two projects.
    """
    kr = (kill_reason or "").lower()
    if not kr:
        return "killed early"
    if "no improvement" in kr or "no_learn" in kr:
        return "killed: no learning"
    if "plateau" in kr:
        m = re.search(r"\(([\d.]+)%\)", kr)
        return f"killed: plateau {m.group(1)}%" if m else "killed: plateau"
    if "below baseline" in kr or "baseline gate" in kr:
        return "killed: below baseline"
    if "iteration timeout" in kr or "iter timeout" in kr:
        return "killed: iter timeout"

    # Fall back to upstream classifier for anything orak doesn't recognise
    # (gemma4 KL/loss divergence, GPU spike/slow/hang/wasted/undersized).
    category, extras = categorize_kill_reason(kill_reason)
    if category == KILL_POLICY_DIVERGENCE:
        return f"killed: kl={extras['kl']} (policy)" if extras else "killed: policy divergence"
    if category == KILL_LOSS_BLOWUP:
        return f"killed: |loss|={extras['loss']}" if extras else "killed: loss blow-up"
    if category == KILL_GPU_SPIKE:
        return f"killed: {extras['step_time']}s GPU spike" if extras else "killed: GPU spike"
    if category == KILL_GPU_SLOW:
        return f"killed: {extras['step_time']}s/step (slow)" if extras else "killed: GPU slow"
    if category == KILL_NO_LEARNING:
        return "killed: no learning"
    return f"killed: {kill_reason[:30]}"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _render_game_axis(ax, rows: list[dict], game: str) -> None:
    """Draw one game's experiment timeline onto the given axis."""
    if not rows:
        ax.set_title(f"{game} — no results yet", fontsize=11, color="#999")
        ax.set_axis_off()
        return

    score = lambda r: r.get(SCORE_FIELD, 0)
    best_exp = max(rows, key=score).get("experiment", 0)

    for r in rows:
        cfg = _STATUS_STYLE.get(r.get("status", "DISCARD"), _STATUS_STYLE["DISCARD"])
        is_best = r.get("experiment") == best_exp
        ax.scatter(
            r.get("experiment", 0), score(r),
            s=300 if is_best else 160,
            c=cfg["color"],
            edgecolors="#27ae60" if is_best else cfg["line_color"],
            linewidths=2.5 if is_best else 1.0,
            zorder=3,
        )

    kept = [r for r in rows if r.get("status") in ("KEEP", "BASELINE")]
    if kept:
        xs = [r["experiment"] for r in kept]
        ys, best = [], float("-inf")
        for r in kept:
            best = max(best, score(r))
            ys.append(best)
        ax.step(xs, ys, where="post", color="#27ae60", lw=2, alpha=0.6, zorder=2)

    for j, r in enumerate(rows):
        cfg = _STATUS_STYLE.get(r.get("status", "DISCARD"), _STATUS_STYLE["DISCARD"])
        is_best = r.get("experiment") == best_exp
        status = r.get("status", "DISCARD")
        if status == "EARLY_KILL":
            tag = _kill_tag(r.get("notes", ""))
        else:
            tag = status.lower()
        runtime = f"{int(r.get('runtime_min', 0))}min" if r.get("runtime_min") else ""
        head = f"E{r.get('experiment', '?')} · {runtime} · {tag}".strip(" ·").replace("·  ·", "·")
        bits = [f"eval={score(r):.2f}"]
        if r.get("steps"):
            bits.append(f"{r['steps']}st")
        text = f"{head}\n{' · '.join(bits)}"

        y_off = 1.4 if j % 2 == 0 else -1.6
        ax.annotate(
            text, xy=(r.get("experiment", 0), score(r)),
            xytext=(0, y_off * 16), textcoords="offset points",
            ha="center", va="center",
            fontsize=8 if not is_best else 9,
            fontweight="bold" if is_best else "normal",
            color=("#1a7a3a" if is_best else cfg["text_color"]),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=("#f0fff0" if is_best else "white"),
                edgecolor=("#27ae60" if is_best else cfg["color"]),
                linewidth=1.5 if is_best else 1,
            ),
        )

    n = len(rows)
    n_kept = sum(1 for r in rows if r.get("status") in ("KEEP", "BASELINE"))
    n_kill = sum(1 for r in rows if r.get("status") == "EARLY_KILL")
    runtime_total = sum(r.get("runtime_min", 0) for r in rows)
    pretty = game.replace("_", " ").title()
    ax.set_title(
        f"{pretty} — {n} exp · {n_kept} kept · {n_kill} killed · {runtime_total:.0f}min",
        fontsize=11, color="#222",
    )
    ax.set_xlabel("Experiment #", fontsize=10)
    ax.set_ylabel(SCORE_LABEL, fontsize=9)
    ax.grid(True, color="#eee", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks(range(0, n))
    ax.set_xlim(-0.5, n - 0.5)
    ymin = max(0, min(score(r) for r in rows) - 5)
    ymax = max(score(r) for r in rows) + 8
    ax.set_ylim(ymin, ymax)


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else None
    results_path, out_path, title_suffix = _paths_for(config_name)

    rows = _load(results_path)
    if not rows:
        raise SystemExit(f"no results to plot at {results_path}")

    games = sorted({r.get("game", "unknown") for r in rows})
    n_games = len(games)

    fig, axes = plt.subplots(n_games, 1, figsize=(14, 5 * n_games), dpi=140,
                             squeeze=False)
    fig.patch.set_facecolor("white")

    for idx, game in enumerate(games):
        game_rows = sorted(
            (r for r in rows if r.get("game") == game),
            key=lambda r: r.get("experiment", 0),
        )
        _render_game_axis(axes[idx][0], game_rows, game)

    fig.suptitle(f"{TITLE}{title_suffix}", fontsize=15, color="#222", y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight")
    print(f"wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
