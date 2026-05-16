"""Render the canonical pokemon_red cross-stage progression chart.

Reads `experiments/milestones/pokemon_progression.yaml` and writes
`experiments/progress/pokemon_stage_progression.png`. The YAML is the
single source of truth — append new milestones there, then re-run.

Verdict colour map matches the verdict tag the launcher / writeups use:
  BASELINE / FLAT / NEUTRAL+ / REGRESS / LIFT.

Usage:
    uv run python experiments/plot_pokemon_progression.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import typer
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "experiments" / "milestones" / "pokemon_progression.yaml"
DEFAULT_OUT = REPO / "experiments" / "progress" / "pokemon_stage_progression.png"

VERDICT_COLOR = {
    "BASELINE": "#7f8c8d",
    "FLAT": "#3498db",
    "NEUTRAL+": "#2ecc71",
    "REGRESS": "#e74c3c",
    "LIFT": "#9b59b6",
}


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    milestones_yaml: Path = typer.Option(DEFAULT_YAML, "--milestones-yaml", "-m"),
    out: Path = typer.Option(DEFAULT_OUT, "--out"),
) -> None:
    data = yaml.safe_load(milestones_yaml.read_text())
    metric = data["primary_metric"]
    title = data["title"]
    thresholds = data.get("thresholds", [])

    milestones = data["milestones"]
    labels = [m["label"] for m in milestones]
    means = [m["metrics"][metric]["mean"] for m in milestones]
    stds = [m["metrics"][metric]["std"] for m in milestones]
    verdicts = [m.get("verdict", "FLAT") for m in milestones]
    descriptions = [m.get("description", "") for m in milestones]
    ns = [m.get("n") for m in milestones]
    colors = [VERDICT_COLOR.get(v, "#34495e") for v in verdicts]

    fig, ax = plt.subplots(figsize=(13, 7))
    xs = list(range(len(labels)))
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=1.2)

    for x, m, s, v, n in zip(xs, means, stds, verdicts, ns):
        ax.text(
            x, m + (s if s else 0) + 1.8, f"{m:.2f}%", ha="center", fontsize=10, fontweight="bold"
        )
        n_label = f"  n={n}" if n else ""
        ax.text(
            x,
            m / 2,
            f"{v}{n_label}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            rotation=90,
        )

    for t in thresholds:
        ax.axhline(t["value"], linestyle="--", color=t["color"], linewidth=1.0, alpha=0.7)
        ax.text(
            -0.45,
            t["value"] + 0.6,
            f"{t['label']} ({t['value']:.2f}%)",
            ha="left",
            fontsize=8,
            color=t["color"],
        )

    ax.set_ylim(0, max(78, max(means) + max(stds) + 10))
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel("evaluation_score (%)", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_labels = sorted(
        {v for v in verdicts},
        key=lambda v: ["BASELINE", "FLAT", "NEUTRAL+", "LIFT", "REGRESS"].index(v),
    )
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[v]) for v in legend_labels],
        legend_labels,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        ncol=len(legend_labels),
        bbox_to_anchor=(1.0, 1.0),
    )

    footer = "\n".join(f"  {lab:<22s} {desc}" for lab, desc in zip(labels, descriptions))
    fig.text(
        0.06,
        -0.02,
        "Levers tested:\n" + footer,
        ha="left",
        va="top",
        fontsize=8,
        color="#444",
        family="monospace",
    )

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
