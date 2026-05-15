"""Stage L vs Stage K introspection: compare cumulative-memory mechanics.

For each iter dir, computes the metrics that distinguish "cumulative memory
helps" from "negative transfer":
  - first_M{1..7}_step: step index when each milestone was first banked
  - route1_steps: how many steps the agent spent on Route1 (the gate to M5)
  - move_to_count: total `move_to(...)` calls (movement attempts)
  - perseveration_rate: fraction of consecutive `move_to(...)` calls that
    repeat coordinates (proxy for thrashing)
  - final_map: where the agent ended up at step 300

Usage:
    uv run python experiments/stage_l_map_aware/introspect.py \
        --stage-k /tmp/orak-post-asm-rerun/pokemon_red \
        --stage-l /tmp/orak-stage-l-map-aware/pokemon_red
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import typer

app = typer.Typer(pretty_exceptions_enable=False)

MOVE_TO_RE = re.compile(
    r"move_to[^()]*\(\s*[^,]*?(-?\d+)\s*,\s*[^,]*?(-?\d+)\s*\)"
)


def _iter_metrics(run_dir: Path) -> dict:
    """Pull the comparison metrics from one iter dir."""
    gs = run_dir / "game_states.jsonl"
    if not gs.exists():
        return {"run_id": run_dir.name, "error": "no game_states.jsonl"}

    lines = gs.read_text().splitlines()
    total_steps = len(lines)
    first_score_step: dict[int, int] = {}
    map_count: Counter[str] = Counter()
    move_targets: list[tuple[int, int]] = []
    final_map = "?"
    final_score = 0.0

    for i, raw in enumerate(lines):
        try:
            row = json.loads(raw)
        except Exception:
            continue
        gi = row.get("obs", {}).get("game_info", {})
        raw_score = gi.get("score", 0)
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            score = 0
        map_name = gi.get("map_name", "?") or "?"
        map_count[map_name] += 1
        if score >= 1:
            for milestone in range(1, score + 1):
                first_score_step.setdefault(milestone, i)
        action = row.get("action", "")
        action_str = action if isinstance(action, str) else json.dumps(action)
        m = MOVE_TO_RE.search(action_str)
        if m:
            move_targets.append((int(m.group(1)), int(m.group(2))))
        final_map = map_name
        final_score = float(score)

    summ = run_dir / "evaluation_summary.json"
    if summ.exists():
        try:
            ep = json.load(summ.open()).get("episodes", [{}])[0]
            fs = ep.get("final_score")
            if fs is not None:
                final_score = float(fs)
        except Exception:
            pass

    # Perseveration: consecutive identical move_to targets
    if len(move_targets) > 1:
        repeats = sum(1 for i in range(1, len(move_targets)) if move_targets[i] == move_targets[i - 1])
        perseveration = repeats / max(len(move_targets) - 1, 1)
    else:
        perseveration = 0.0

    return {
        "run_id": run_dir.name,
        "total_steps": total_steps,
        "final_score": final_score,
        "final_score_pct": round((final_score / 7.0) * 100, 2),
        "final_map": final_map,
        "first_M1_step": first_score_step.get(1),
        "first_M2_step": first_score_step.get(2),
        "first_M3_step": first_score_step.get(3),
        "first_M4_step": first_score_step.get(4),
        "first_M5_step": first_score_step.get(5),
        "route1_steps": map_count.get("Route1", 0),
        "viridian_steps": sum(v for k, v in map_count.items() if "Viridian" in k),
        "move_to_count": len(move_targets),
        "perseveration_pct": round(perseveration * 100, 1),
        "top_maps": map_count.most_common(3),
    }


def _classify(scores: list[float], deltas: list[int]) -> str:
    """LIFT / NEUTRAL+ / STILL-NEGATIVE.

    deltas = list of (iter_n_steps_to_M4 - iter_1_steps_to_M4) for iters >= 2.
    """
    if any(s > 57.14 for s in scores):
        return "LIFT"
    # min bar: no iter took >20% more steps to M4 than iter 1
    if deltas and max(deltas) <= 26:  # 20% of 129 ≈ 26
        return "NEUTRAL+"
    return "STILL-NEGATIVE"


@app.command()
def main(
    stage_k: Path = typer.Option(
        Path("/tmp/orak-post-asm-rerun/pokemon_red"),
        "--stage-k",
        help="Stage K (baseline) GAME_DATA_DIR/pokemon_red root",
    ),
    stage_l: Path = typer.Option(
        Path("/tmp/orak-stage-l-map-aware/pokemon_red"),
        "--stage-l",
        help="Stage L (treatment) GAME_DATA_DIR/pokemon_red root",
    ),
    glob_k: str = typer.Option("post_asm_stage_k_post_asm_iter*", "--glob-k"),
    glob_l: str = typer.Option("stage_l_map_aware_iter*", "--glob-l"),
):
    """Print a side-by-side table of Stage K vs Stage L iter metrics."""
    for tag, base, glob in (("K", stage_k, glob_k), ("L", stage_l, glob_l)):
        print(f"\n══════ Stage {tag} ({base}) ══════")
        rows = []
        for d in sorted(base.glob(glob)):
            if not d.is_dir():
                continue
            rows.append(_iter_metrics(d))
        if not rows:
            print("  (no iter dirs found yet)")
            continue
        for r in rows:
            if "error" in r:
                print(f"  {r['run_id']}: {r['error']}")
                continue
            print(
                f"  iter {r['run_id'].split('iter')[1].split('_')[0]:>2}: "
                f"final={r['final_score_pct']:5.2f}% "
                f"M4@step={r['first_M4_step'] or 'n/a':>3} "
                f"Route1={r['route1_steps']:>3} "
                f"Viridian={r['viridian_steps']:>3} "
                f"move_to={r['move_to_count']:>3} "
                f"persev={r['perseveration_pct']:>4.1f}% "
                f"final_map={r['final_map']}"
            )

        scores = [r["final_score_pct"] for r in rows if "error" not in r]
        m4_steps = [r["first_M4_step"] for r in rows if "error" not in r and r["first_M4_step"]]
        if m4_steps and len(m4_steps) >= 2:
            iter1_m4 = m4_steps[0]
            deltas = [s - iter1_m4 for s in m4_steps[1:]]
            verdict = _classify(scores, deltas)
            print(
                f"\n  Summary: scores={scores} "
                f"iter1_M4_step={iter1_m4} deltas_vs_iter1={deltas} "
                f"→ verdict={verdict}"
            )


if __name__ == "__main__":
    app()
