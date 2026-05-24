"""Post-hoc results extractor — parses openevolve's internal log + cross-references
rollout dirs to write a clean append-only `results.jsonl` per the project convention.

Why post-hoc not in-evaluator: the daemon is already running. Modifying evaluator.py
would risk restart contamination. The openevolve log + rollout dirs already contain
everything we need.

Schema per line:
    {
        "iter": int,                # 0 = baseline, 1..N = mutations
        "program_id": str,          # openevolve uuid
        "parent_id": str | None,    # parent uuid (None for baseline)
        "completed_at": str,        # ISO timestamp from log
        "elapsed_s_evolve": float,  # time openevolve attributes to this iter
        "elapsed_s_rollout": float, # time evaluator's rollout took
        "score": float,             # normalized 0-1
        "raw_score": float,         # 0-7 from pokemon_red
        "runs_successfully": float, # 1.0 unless the rollout crashed
        "final_map": str,           # terminal map_name (debug Viridian loops)
        "run_id": str,              # rollout dir name in /workspace/orak-evolve-rollouts/
        "status": "ok" | "failed",
    }

Usage:
    python extract_results.py \\
        --log /workspace/orak-stage-s-evolve/logs/openevolve_real_<ts>.log \\
        --rollouts /workspace/orak-evolve-rollouts/pokemon_red \\
        --out /workspace/orak-stage-s-evolve/experiments/openevolve_milestones/results.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import typer

# OpenEvolve log line patterns. The internal `openevolve_<ts>.log` (in
# evolve_output/logs/) is the authoritative event stream.
_EVAL_RE = re.compile(
    r"^(\S+ \S+) - openevolve\.evaluator - INFO - "
    r"Evaluated program (?P<pid>[a-f0-9-]+) in (?P<elapsed>[\d.]+)s: (?P<metrics>.+)$"
)
_ITER_RE = re.compile(
    r"^(\S+ \S+) - openevolve\.process_parallel - INFO - "
    r"Iteration (?P<iter>\d+): Program (?P<pid>[a-f0-9-]+) "
    r"\(parent: (?P<parent>[a-f0-9-]+)\) completed in (?P<elapsed>[\d.]+)s$"
)
_ERR_RE = re.compile(r"^(\S+ \S+) - \S+ - WARNING - Iteration (?P<iter>\d+) error: (?P<msg>.+)$")
_METRIC_KV = re.compile(r"(\w+)=([\d.eE+-]+)")


def _parse_metrics(s: str) -> dict[str, float]:
    return {k: float(v) for k, v in _METRIC_KV.findall(s)}


def _ts(s: str) -> str:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f").isoformat() + "Z"
    except ValueError:
        return s


def _find_run_id(rollouts_dir: Path, completion_ts: float) -> str | None:
    """OpenEvolve emits program_id but our evaluator spawns rollouts with a
    different `evolve_<sha>_<unix>` ID. Match by completion time — the rollout
    whose evaluation_summary.json was written closest to openevolve's
    eval-completion timestamp (typically within ~1s)."""
    best: tuple[float, str] | None = None
    for d in rollouts_dir.glob("evolve_*"):
        summary = d / "evaluation_summary.json"
        if not summary.exists():
            continue
        diff = abs(summary.stat().st_mtime - completion_ts)
        if diff < 30 and (best is None or diff < best[0]):
            best = (diff, d.name)
    return best[1] if best else None


def _final_map(run_dir: Path) -> str | None:
    states = run_dir / "game_states.jsonl"
    if not states.exists():
        return None
    last_map = None
    for line in states.read_text().splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line).get("obs", {}).get("game_info", {}).get("map_name")
        except json.JSONDecodeError:
            continue
        if m:
            last_map = m
    return last_map


def main(
    log: Path = typer.Option(..., help="Path to openevolve_real_<ts>.log"),
    rollouts: Path = typer.Option(
        Path("/workspace/orak-evolve-rollouts/pokemon_red"),
        help="Directory containing per-rollout artefacts",
    ),
    out: Path = typer.Option(..., help="results.jsonl output path"),
) -> None:
    if not log.exists():
        typer.echo(f"ERROR: log not found: {log}", err=True)
        raise typer.Exit(2)

    # Pass 1: collect evaluation metrics keyed by program_id.
    evals: dict[str, dict] = {}
    iters: dict[str, dict] = {}  # iter_n -> {pid, parent, elapsed_s_evolve, ts}
    errors: list[dict] = []

    for raw in log.read_text().splitlines():
        if m := _EVAL_RE.match(raw):
            pid = m["pid"]
            evals[pid] = {
                "ts_str": m.group(1),
                "elapsed_s_rollout": float(m["elapsed"]),
                "metrics": _parse_metrics(m["metrics"]),
            }
        elif m := _ITER_RE.match(raw):
            iters[m["iter"]] = {
                "pid": m["pid"],
                "parent": m["parent"],
                "elapsed_s_evolve": float(m["elapsed"]),
                "ts_str": m.group(1),
            }
        elif m := _ERR_RE.match(raw):
            errors.append({"iter": int(m["iter"]), "msg": m["msg"], "ts_str": m.group(1)})

    # Iter 0 = baseline ("initial program"). It's logged via an Evaluated line
    # without a corresponding Iteration line. Synthesize an iter=0 entry.
    if evals and not any(i["pid"] in iters for i in [{"pid": list(evals)[0]}]):
        first_pid = list(evals.keys())[0]
        iters["0"] = {
            "pid": first_pid,
            "parent": None,
            "elapsed_s_evolve": evals[first_pid]["elapsed_s_rollout"],
            "ts_str": evals[first_pid]["ts_str"],
        }

    rows: list[dict] = []
    for iter_n in sorted(iters, key=int):
        info = iters[iter_n]
        pid = info["pid"]
        eval_info = evals.get(pid, {})
        metrics = eval_info.get("metrics", {})
        ts_iso = _ts(info["ts_str"])
        ts_unix = datetime.strptime(info["ts_str"], "%Y-%m-%d %H:%M:%S,%f").timestamp()
        run_id = _find_run_id(rollouts, ts_unix)
        run_dir = rollouts / run_id if run_id else None

        rows.append(
            {
                "iter": int(iter_n),
                "program_id": pid,
                "parent_id": info["parent"],
                "completed_at": ts_iso,
                "elapsed_s_evolve": info["elapsed_s_evolve"],
                "elapsed_s_rollout": eval_info.get("elapsed_s_rollout", 0.0),
                "score": metrics.get("score", 0.0),
                "raw_score": metrics.get("raw_score", 0.0),
                "runs_successfully": metrics.get("runs_successfully", 0.0),
                "final_map": _final_map(run_dir) if run_dir else None,
                "run_id": run_id,
                "status": "ok" if metrics.get("runs_successfully", 0) > 0.5 else "failed",
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))

    print(f"Wrote {len(rows)} rows to {out}", file=sys.stderr)
    if errors:
        print(f"Iter errors (not in jsonl, see log): {len(errors)}", file=sys.stderr)
        for e in errors[:5]:
            print(f"  iter {e['iter']}: {e['msg']}", file=sys.stderr)


if __name__ == "__main__":
    typer.run(main)
