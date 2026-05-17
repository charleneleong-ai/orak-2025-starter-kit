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

import typer  # noqa: E402
from autoresearch.compare import load_milestones_yaml, plot_milestone_bars  # noqa: E402

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
    plot_milestone_bars(
        milestones,
        primary_metric=meta["primary_metric"],
        out_path=out,
        title=meta.get("title"),
        ylabel=f"{meta['primary_metric']} (%)",
        thresholds=meta.get("thresholds"),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
