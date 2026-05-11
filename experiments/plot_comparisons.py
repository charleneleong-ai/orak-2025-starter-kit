"""Multi-sweep comparison plots for autoresearch.

Built on top of ``experiment_progress.load_results``. Two flavours:

1. ``plot_multi_tag_overlay`` — overlay multiple sweeps on the same axes,
   per-iter delta annotations between the first sweep (baseline) and the
   rest. Best for "did adding feature X lift scores?" questions.

2. ``plot_cross_game_scoreboard`` — bar chart per game, one bar per sweep,
   shows best evaluation score. Best for "where does feature X help?"
   summary questions.

Both render to matplotlib PNG (the per-tag ``progress.html`` plotly chart
in ``experiment_progress.plot_progress`` is fine for single sweeps but
gets sparse and hard-to-read with only 2-4 iters; this module is for the
cross-sweep comparison story).

Examples:

    # CLI — overlay 3 sweeps on 2048
    python -m experiments.plot_comparisons overlay \\
        --tag harness_check --tag cognitive_check --tag cognitive_check_v2 \\
        --label 'Stage A baseline' --label 'Stage C v1' --label 'Stage C v2' \\
        --config-name gemma --game twenty_fourty_eight \\
        --out docs/experiments/gemma/plots/stage_a_vs_c_2048.png

    # CLI — cross-game scoreboard
    python -m experiments.plot_comparisons scoreboard \\
        --game twenty_fourty_eight --tag harness_check --tag cognitive_check \\
        --game super_mario --tag mario_check \\
        --game pokemon_red --tag pokemon_check --tag pokemon_check_v3 \\
        --config-name gemma \\
        --out docs/experiments/gemma/plots/cross_game_scoreboard.png

    # Python — overlay
    from experiments.plot_comparisons import plot_multi_tag_overlay
    plot_multi_tag_overlay(
        sweeps=[
            ("harness_check", "Stage A baseline (no vmem)"),
            ("cognitive_check", "Stage C v1 (vmem on, uniform)"),
            ("cognitive_check_v2", "Stage C v2 (richer template)"),
        ],
        config_name="gemma",
        game="twenty_fourty_eight",
        out_path="docs/.../stage_a_vs_c_2048.png",
    )
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import typer

from experiments.experiment_progress import load_results

app = typer.Typer(help="Cross-sweep comparison plots for autoresearch.")


# ── helpers ────────────────────────────────────────────────────────────


def _filter_game(rows: list[dict], game: str | None) -> list[dict]:
    if not game:
        return rows
    return [r for r in rows if r.get("game") == game]


def _best_score(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return max(r.get("evaluation_score", r.get("score", 0)) for r in rows)


# ── multi-tag overlay ──────────────────────────────────────────────────


def plot_multi_tag_overlay(
    sweeps: Sequence[tuple[str, str]],
    *,
    config_name: str | None = None,
    game: str | None = None,
    out_path: str | Path,
    title: str | None = None,
    annotate_deltas: bool = True,
    figsize: tuple[float, float] = (13.0, 6.5),
    dpi: int = 140,
) -> Path:
    """Overlay multiple sweep results on the same iteration axis.

    The first sweep is treated as the baseline; per-iter percentage deltas
    against it are annotated when ``annotate_deltas=True``.

    ``sweeps`` is a list of ``(tag, display_label)`` tuples. Tags are
    looked up via ``experiment_progress.load_results``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    palette = ["#777", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    by_label: dict[str, dict[int, float]] = {}

    for i, (tag, label) in enumerate(sweeps):
        rows = _filter_game(load_results(tag=tag, config_name=config_name), game)
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r.get("experiment", 0))
        xs = [r.get("experiment", j) for j, r in enumerate(rows)]
        ys = [r.get("evaluation_score", r.get("score", 0)) for r in rows]
        statuses = [r.get("status", "?") for r in rows]
        sizes = [220 if s in ("KEEP", "BASELINE") else 100 for s in statuses]
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]

        ax.plot(xs, ys, "-", color=color, alpha=0.4, linewidth=2)
        ax.scatter(
            xs,
            ys,
            c=color,
            s=sizes,
            marker=marker,
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
            label=label,
        )
        for x, y, s in zip(xs, ys, statuses):
            ax.annotate(
                f"{y:.2f}\n{s.lower()}",
                xy=(x, y),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                fontweight="bold" if s in ("KEEP", "BASELINE") else "normal",
            )
        by_label[label] = dict(zip(xs, ys))

    # Delta annotations between baseline (first sweep) and second sweep
    if annotate_deltas and len(sweeps) >= 2:
        baseline_label = sweeps[0][1]
        compare_label = sweeps[1][1]
        if baseline_label in by_label and compare_label in by_label:
            for exp in sorted(set(by_label[baseline_label]) & set(by_label[compare_label])):
                base = by_label[baseline_label][exp]
                comp = by_label[compare_label][exp]
                if base <= 0:
                    continue
                delta = (comp - base) / base * 100
                ax.annotate(
                    f"{'+' if delta >= 0 else ''}{delta:.0f}%",
                    xy=(exp, (base + comp) / 2),
                    xytext=(40, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="#2ca02c" if delta > 0 else "#d62728",
                    arrowprops=dict(
                        arrowstyle="-[", color="#2ca02c" if delta > 0 else "#d62728", lw=1.5
                    ),
                )

    # Axes / title
    all_x: list[int] = []
    all_y: list[float] = []
    for d in by_label.values():
        all_x.extend(d.keys())
        all_y.extend(d.values())
    if all_x:
        ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.7)
        ax.set_xticks(range(min(all_x), max(all_x) + 1))
    if all_y:
        ax.set_ylim(0, max(all_y) * 1.2)

    ax.set_xlabel("Iteration #", fontsize=11)
    ax.set_ylabel("Evaluation Score (higher is better)", fontsize=11)
    if title is None:
        scope = f"{game}" if game else "all games"
        title = f"Sweep comparison — {scope}"
    ax.set_title(title, fontsize=13, color="#222", pad=15)
    ax.grid(True, color="#eee", linewidth=0.7, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── cross-game scoreboard ──────────────────────────────────────────────


def plot_cross_game_scoreboard(
    games_to_sweeps: dict[str, list[tuple[str, str]]],
    *,
    config_name: str | None = None,
    out_path: str | Path,
    title: str | None = None,
    game_titles: dict[str, str] | None = None,
    game_verdicts: dict[str, str] | None = None,
    figsize_per_panel: tuple[float, float] = (5.5, 6.0),
    dpi: int = 140,
) -> Path:
    """Per-game scoreboard — bar chart of best score per sweep.

    ``games_to_sweeps`` maps game name → list of ``(tag, display_label)``.
    Each game becomes one subplot panel; bars are best ``evaluation_score``
    per sweep. The best bar per panel gets a green border.

    Optional ``game_titles`` / ``game_verdicts`` add a richer title /
    footer per panel ("✓ vmem helps (+58%)" etc).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_games = len(games_to_sweeps)
    fig, axes = plt.subplots(
        1,
        n_games,
        figsize=(figsize_per_panel[0] * n_games, figsize_per_panel[1]),
        dpi=dpi,
    )
    if n_games == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    palette = ["#888", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]

    for ax, (game, sweeps) in zip(axes, games_to_sweeps.items()):
        labels = [s[1] for s in sweeps]
        values = [
            _best_score(_filter_game(load_results(tag=s[0], config_name=config_name), game))
            for s in sweeps
        ]
        n = len(labels)
        colors = [palette[i % len(palette)] for i in range(n)]
        bars = ax.bar(range(n), values, color=colors, edgecolor="white", linewidth=2, zorder=3)

        # Highlight best
        if any(v > 0 for v in values):
            best_idx = max(range(n), key=lambda i: values[i])
            bars[best_idx].set_edgecolor("#2ca02c")
            bars[best_idx].set_linewidth(3)
        else:
            best_idx = -1

        # Score labels
        max_v = max(values) if any(v > 0 for v in values) else 1
        for i, v in enumerate(values):
            ax.text(
                i,
                v + max_v * 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold" if i == best_idx else "normal",
                color="#2ca02c" if i == best_idx else "#333",
            )

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        panel_title = (game_titles or {}).get(game, game)
        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=10)
        if ax is axes[0]:
            ax.set_ylabel("Best Evaluation Score")
        ax.grid(True, color="#eee", linewidth=0.7, axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max_v * 1.25)

        verdict = (game_verdicts or {}).get(game)
        if verdict:
            ax.text(
                0.5,
                -0.32,
                verdict,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

    if title:
        fig.suptitle(title, fontsize=13, color="#222", y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────


@app.command()
def overlay(
    tag: list[str] = typer.Option(..., "--tag", "-t", help="Sweep tag (repeat for multi-sweep)"),
    label: list[str] = typer.Option(
        ..., "--label", "-l", help="Display label per tag (same order)"
    ),
    out: str = typer.Option(..., help="Output PNG path"),
    config_name: str | None = typer.Option(
        None, help="Per-config sub-dir under experiments/<tag>/"
    ),
    game: str | None = typer.Option(None, help="Filter to a specific game"),
    title: str | None = typer.Option(None, help="Plot title (default auto)"),
    no_deltas: bool = typer.Option(False, help="Disable per-iter delta annotations"),
):
    """Overlay multiple sweeps on the same iter axis with delta annotations."""
    if len(tag) != len(label):
        raise typer.BadParameter(
            f"--tag count ({len(tag)}) must match --label count ({len(label)})"
        )
    p = plot_multi_tag_overlay(
        sweeps=list(zip(tag, label)),
        config_name=config_name,
        game=game,
        out_path=out,
        title=title,
        annotate_deltas=not no_deltas,
    )
    typer.echo(f"wrote {p}")


@app.command()
def scoreboard(
    out: str = typer.Option(..., help="Output PNG path"),
    config_name: str | None = typer.Option(None, help="Per-config sub-dir"),
    game: list[str] = typer.Option(
        ...,
        "--game",
        "-g",
        help="Game name (repeat for multiple games — interleave with --tag/--label per game)",
    ),
    tag: list[str] = typer.Option(
        ..., "--tag", "-t", help="Sweep tag — these are MATCHED to games positionally via --sep"
    ),
    label: list[str] = typer.Option(
        ..., "--label", "-l", help="Display label per tag, same order as --tag"
    ),
    sep: list[int] = typer.Option(
        ..., help="Number of (tag, label) pairs per game — must sum to len(tag)"
    ),
    title: str | None = typer.Option(None, help="Top-level title"),
):
    """Cross-game scoreboard. Pass --game once per panel; --tag/--label
    multiple times overall; --sep tells where each game's tags end.

    Example for 3 games × varying number of sweeps each:

        --game 2048 --tag a1 --tag a2 --label "A" --label "B" --sep 2
        --game mario --tag m1 --label "C" --sep 1
        --game pokemon --tag p1 --tag p2 --label "D" --label "E" --sep 2
    """
    if sum(sep) != len(tag) or len(tag) != len(label):
        raise typer.BadParameter(
            f"--sep ({sep}, sum={sum(sep)}) must add up to len(tag)={len(tag)} "
            f"== len(label)={len(label)}"
        )
    if len(sep) != len(game):
        raise typer.BadParameter(
            f"--sep entries ({len(sep)}) must match --game entries ({len(game)})"
        )

    games_to_sweeps: dict[str, list[tuple[str, str]]] = {}
    cursor = 0
    for g, n in zip(game, sep):
        games_to_sweeps[g] = list(zip(tag[cursor : cursor + n], label[cursor : cursor + n]))
        cursor += n

    p = plot_cross_game_scoreboard(
        games_to_sweeps=games_to_sweeps,
        config_name=config_name,
        out_path=out,
        title=title,
    )
    typer.echo(f"wrote {p}")


# ── scoreboard-from-index (canonical alternative to scoreboard) ────────


def _best_score_for_variant_from_index(rows: list[dict], *, game: str, variant: str) -> float:
    """Best ``evaluation_score`` across all rows matching (game, variant)."""
    matching = [r for r in rows if r.get("game") == game and r.get("variant") == variant]
    return _best_score(matching)


def _load_index(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@app.command("scoreboard-from-index")
def scoreboard_from_index(
    from_file: str = typer.Option(..., help="Path to a consolidated results.jsonl"),
    out: str = typer.Option(..., help="Output PNG path"),
    game: list[str] = typer.Option(..., "--game", "-g", help="Game name (repeat per panel)"),
    variant: list[str] = typer.Option(
        ...,
        "--variant",
        "-v",
        help="Variant to render as one bar (matched against the 'variant' field). "
        "Repeat per bar; interleave with --label/--sep like the scoreboard subcommand.",
    ),
    label: list[str] = typer.Option(..., "--label", "-l", help="Display label per variant"),
    sep: list[int] = typer.Option(
        ...,
        help="Number of (variant, label) pairs per game — must sum to len(variant)",
    ),
    title: str | None = typer.Option(None, help="Top-level title"),
):
    """Render a cross-game scoreboard from a single consolidated index file.

    Unlike ``scoreboard``, this command doesn't need one tag-dir per bar —
    it reads from a single ``--from-file`` and filters by ``(game, variant)``
    to produce each bar. Build the index with ``python -m
    experiments._consolidate ...``.
    """
    if sum(sep) != len(variant) or len(variant) != len(label):
        raise typer.BadParameter(
            f"--sep ({sep}, sum={sum(sep)}) must add up to len(variant)={len(variant)} "
            f"== len(label)={len(label)}"
        )
    if len(sep) != len(game):
        raise typer.BadParameter(
            f"--sep entries ({len(sep)}) must match --game entries ({len(game)})"
        )

    index_rows = _load_index(Path(from_file))

    # Build games_to_sweeps in the shape plot_cross_game_scoreboard expects,
    # but the "tag" we pass is actually a placeholder — we precompute scores
    # via the index and feed them via game_titles is a hack; better to render
    # directly here.
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_games = len(game)
    fig, axes = plt.subplots(
        1,
        n_games,
        figsize=(5.5 * n_games, 6.0),
        dpi=140,
    )
    if n_games == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    palette = ["#888", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]

    cursor = 0
    for ax, (g, n) in zip(axes, zip(game, sep)):
        variants_for_game = variant[cursor : cursor + n]
        labels_for_game = label[cursor : cursor + n]
        cursor += n
        values = [
            _best_score_for_variant_from_index(index_rows, game=g, variant=v)
            for v in variants_for_game
        ]
        colors = [palette[i % len(palette)] for i in range(len(labels_for_game))]
        bars = ax.bar(
            range(len(labels_for_game)),
            values,
            color=colors,
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        if any(v > 0 for v in values):
            best_idx = max(range(len(values)), key=lambda i: values[i])
            bars[best_idx].set_edgecolor("#2ca02c")
            bars[best_idx].set_linewidth(3)
        else:
            best_idx = -1
        max_v = max(values) if any(v > 0 for v in values) else 1
        for i, v in enumerate(values):
            ax.text(
                i,
                v + max_v * 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold" if i == best_idx else "normal",
                color="#2ca02c" if i == best_idx else "#333",
            )
        ax.set_xticks(range(len(labels_for_game)))
        ax.set_xticklabels(labels_for_game, rotation=20, ha="right", fontsize=9)
        ax.set_title(g, fontsize=13, fontweight="bold", pad=10)
        if ax is axes[0]:
            ax.set_ylabel("Best Evaluation Score")
        ax.grid(True, color="#eee", linewidth=0.7, axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max_v * 1.25)

    if title:
        fig.suptitle(title, fontsize=13, color="#222", y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    typer.echo(f"wrote {out_path}")


if __name__ == "__main__":
    app()
