"""OpenEvolve evaluator — runs one 300-step pokemon_red rollout per candidate.

Contract (per OpenEvolve docs):
    evaluate(program_path: str) -> EvaluationResult
        program_path: path to a candidate `_POKEMON_MILESTONE_LIBRARY` module
                      (a mutation of initial_program.py).
        returns: EvaluationResult(metrics, artifacts).

The candidate is hot-swapped into the running agent via
POKEMON_MILESTONE_LIBRARY_PATH (read by agents/pokemon_red/game_adapter.py).
We spawn a fresh `python run.py -c gemma_26b ...` subprocess so the override
fires at import-time and there's no module-cache contamination across evals.

Score = final_score from evaluation_summary.json, normalized to [0, 1] by
dividing by TRAJECTORY_SCORE_MAX=7. Stage S v1/v2 baselines: 0.571 (4/7),
0.714 (5/7), 0.857 (6/7), 1.0 (7/7).

Artifacts captured (visible to mutation LLM as feedback):
- milestones_hit: bitmask of which gates fired
- final_map: terminal map_name (debugging Viridian-loop candidates)
- inference_calls, tokens: cost telemetry
- stderr_tail: last 50 lines of run.py stderr on failure
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

REPO = Path("/workspace/orak-stage-s-evolve")
ROLLOUT_TIMEOUT_S = (
    2 * 60 * 60
)  # 2h hard cap. v1 measured 50min for 300 steps; v2 doubles to 600 steps ≈ 100min nominal.
GAME_DATA_DIR = Path("/workspace/orak-evolve-rollouts")
GAME_DATA_DIR.mkdir(parents=True, exist_ok=True)

SCORE_MAX = 7.0


def _spawn_rollout(program_path: str, run_id: str) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["POKEMON_MILESTONE_LIBRARY_PATH"] = str(Path(program_path).resolve())
    env["GAME_DATA_DIR"] = str(GAME_DATA_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        str(REPO / ".venv/bin/python"),
        "run.py",
        "-c",
        "gemma_26b",
        "--local",
        "--games",
        "pokemon_red",
        "--run-id",
        run_id,
        "-d",
        f"openevolve candidate {run_id}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=ROLLOUT_TIMEOUT_S,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_summary(run_id: str) -> dict | None:
    summary_path = GAME_DATA_DIR / "pokemon_red" / run_id / "evaluation_summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None


def _parse_final_map(run_id: str) -> str | None:
    states_path = GAME_DATA_DIR / "pokemon_red" / run_id / "game_states.jsonl"
    if not states_path.exists():
        return None
    last_map = None
    for line in states_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line).get("obs", {}).get("game_info", {}).get("map_name")
        except json.JSONDecodeError:
            continue
        if m:
            last_map = m
    return last_map


def _fail(reason: str, started: float, **extra: object) -> EvaluationResult:
    return EvaluationResult(
        metrics={"score": 0.0, "runs_successfully": 0.0},
        artifacts={"failure": reason, "elapsed_s": time.time() - started, **extra},
    )


def evaluate(program_path: str) -> EvaluationResult:
    sha = hashlib.sha1(Path(program_path).read_bytes()).hexdigest()[:10]
    run_id = f"evolve_{sha}_{int(time.time())}"
    started = time.time()

    try:
        rc, stdout, stderr = _spawn_rollout(program_path, run_id)
    except subprocess.TimeoutExpired:
        return _fail("rollout_timeout", started)

    summary = _parse_summary(run_id)
    if summary is None or rc != 0:
        return _fail(
            "no_summary" if summary is None else f"nonzero_exit_{rc}",
            started,
            stderr_tail="\n".join(stderr.splitlines()[-50:]),
            stdout_tail="\n".join(stdout.splitlines()[-20:]),
        )

    episode = summary["episodes"][0]
    raw_score = float(episode["final_score"])

    return EvaluationResult(
        metrics={
            "score": raw_score / SCORE_MAX,
            "raw_score": raw_score,
            "runs_successfully": 1.0,
        },
        artifacts={
            "final_map": _parse_final_map(run_id) or "?",
            "inference_calls": episode.get("inference_calls", 0),
            "tokens": episode.get("tokens", 0),
            "elapsed_s": time.time() - started,
            "run_id": run_id,
        },
    )
