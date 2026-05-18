"""Per-iter Reflexion summary built from the prior iter's trajectory.

At iter start (after checkpoint load), the planner sees a 5-line summary of
what the previous iter actually did — score reached, milestones hit, final
zone, perseveration rate — and a prompt to hypothesise why and try
differently. Built on top of ``autoresearch.trajectory.extract_iter_metrics``
so per-game adapters that ship ``TRAJECTORY_*`` constants get this for free.

The summary is a string prepended to the subtask planner's history block.
No new LLM call — same call, more context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch.trajectory import extract_iter_metrics
from loguru import logger


def build_reflexion_summary(run_dir: Path, adapter: Any) -> str:
    """Build a Reflexion summary of the iter whose game_states.jsonl lives in
    ``run_dir``. Returns "" when there's nothing useful to say (run_dir
    missing, adapter doesn't expose TRAJECTORY_* constants, etc).
    """
    if not run_dir.exists():
        return ""

    milestone_specs = getattr(adapter, "TRAJECTORY_MILESTONES", None)
    score_extractor = getattr(adapter, "TRAJECTORY_SCORE_EXTRACTOR", None)
    zone_extractor = getattr(adapter, "TRAJECTORY_ZONE_EXTRACTOR", None)
    score_max = getattr(adapter, "TRAJECTORY_SCORE_MAX", None)
    if not (milestone_specs and score_extractor and zone_extractor and score_max):
        return ""

    try:
        metrics = extract_iter_metrics(
            run_dir,
            milestone_specs=milestone_specs,
            dwell_specs=getattr(adapter, "TRAJECTORY_DWELL_SPECS", None),
            action_spec=getattr(adapter, "TRAJECTORY_ACTION_SPEC", None),
            score_extractor=score_extractor,
            zone_extractor=zone_extractor,
            score_max=score_max,
        )
    except Exception as e:
        logger.warning(f"[Reflexion] extract_iter_metrics failed for {run_dir}: {e}")
        return ""

    if metrics.error:
        return ""

    hit_milestones = [
        f"{name}@step{step}"
        for name, step in metrics.first_milestone_step.items()
        if step is not None
    ]
    milestones_str = ", ".join(hit_milestones) if hit_milestones else "none"

    dwell_str = ", ".join(
        f"{name}={count}"
        for name, count in metrics.dwell_counts.items()
    ) if metrics.dwell_counts else ""

    lines = [
        "### Previous iter (Reflexion)",
        f"Final score: {metrics.score_pct:.2f}% — milestones reached: {milestones_str}.",
        f"Final zone: {metrics.final_zone}.",
    ]
    if dwell_str:
        lines.append(f"Zone dwell: {dwell_str}.")
    if metrics.action_count > 1:
        lines.append(
            f"move_to perseveration: {metrics.perseveration_pct:.1f}% "
            f"({metrics.action_count} actions sampled)."
        )
    lines.append(
        "Hypothesise why the previous iter stalled where it did, "
        "and try a different approach this iter."
    )

    return "\n".join(lines)
