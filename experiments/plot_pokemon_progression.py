"""Render the canonical pokemon_red cross-stage progression chart.

Thin wrapper around :func:`autoresearch.compare.plot_milestone_bars`.
Reads ``experiments/milestones/pokemon_progression.yaml`` and writes
``experiments/progress/pokemon_stage_progression.png``. The YAML is the
single source of truth — append new milestones there, then re-run.

Usage:
    uv run python experiments/plot_pokemon_progression.py
"""

from __future__ import annotations

# Drop the script's directory from sys.path so the local
# ``experiments/autoresearch.py`` (orak's sweep-loop module) doesn't
# shadow the installed ``autoresearch`` package. Has to happen before
# the autoresearch import below.
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p != _SCRIPT_DIR and p != ""]

import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402
from autoresearch.compare import (  # noqa: E402
    VERDICT_PALETTE,
    load_milestones_yaml,
    plot_milestone_bars,
)

# Extend the upstream palette: MIXED and KEEP aren't in autoresearch's
# default and both fall through to the same #34495e navy fallback, which
# makes the chart unreadable when both verdicts appear. Give KEEP a strong
# "ship-it" orange and MIXED its own muted shade so they're visually
# distinct from each other and from FLAT/NEUTRAL+.
_LOCAL_PALETTE = {
    **VERDICT_PALETTE,
    "KEEP": "#e67e22",  # ship-verdict orange
    "MIXED": "#34495e",  # partial-success navy
}

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "experiments" / "milestones" / "pokemon_progression.yaml"
DEFAULT_OUT = REPO / "experiments" / "progress" / "pokemon_stage_progression.png"

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    milestones_yaml: Path = typer.Option(DEFAULT_YAML, "--milestones-yaml", "-m"),
    out: Path = typer.Option(DEFAULT_OUT, "--out"),
) -> None:
    milestones, meta = load_milestones_yaml(milestones_yaml)
    fig = plot_milestone_bars(
        milestones,
        primary_metric=meta["primary_metric"],
        out_path=out,
        title=meta.get("title"),
        ylabel=f"{meta['primary_metric']} (%)",
        thresholds=meta.get("thresholds"),
        palette=_LOCAL_PALETTE,
        return_fig=True,
    )
    # Stage labels collide when rendered horizontally — rotate diagonally
    # so every stage's text is readable at the chart's default width.
    ax = fig.axes[0]
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
