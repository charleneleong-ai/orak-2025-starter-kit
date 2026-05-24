"""Smoke-test evaluator — proves the OpenEvolve install + LLM connection
without burning a real pokemon rollout.

Scores candidates by:
1. Does the candidate import + execute cleanly?       (+0.4)
2. Does it define `_POKEMON_MILESTONE_LIBRARY` dict?  (+0.3)
3. How many MilestoneSpec entries does it have?       (+0.05 each, cap 0.3)

So a valid baseline scores 0.85. Garbage candidates score 0. The mutation
LLM sees a real gradient and OpenEvolve gets to run end-to-end without us
spending 17min per rollout.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

# OpenEvolve spawns the evaluator in a worker process whose sys.path doesn't
# include the worktree root. The candidate `initial_program.py` does
# `from agents.macla.macla_lib import MilestoneSpec` — that resolves only
# when /workspace/orak-stage-s-evolve is importable.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def evaluate(program_path: str) -> EvaluationResult:
    artifacts: dict[str, str | int | float] = {"program_path": program_path}

    try:
        spec = importlib.util.spec_from_file_location("_candidate", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create spec from {program_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        artifacts["failure"] = f"import_error: {e}"
        artifacts["traceback"] = traceback.format_exc()
        return EvaluationResult(
            metrics={"score": 0.0, "imports_cleanly": 0.0, "runs_successfully": 0.0},
            artifacts=artifacts,
        )

    library = getattr(mod, "_POKEMON_MILESTONE_LIBRARY", None)
    if not isinstance(library, dict):
        artifacts["failure"] = "missing _POKEMON_MILESTONE_LIBRARY dict"
        return EvaluationResult(
            metrics={"score": 0.4, "imports_cleanly": 1.0, "runs_successfully": 0.0},
            artifacts=artifacts,
        )

    artifacts["library_keys"] = str(sorted(library.keys()))
    artifacts["program_size_bytes"] = Path(program_path).stat().st_size

    return EvaluationResult(
        metrics={
            "score": 0.7 + min(len(library) * 0.05, 0.3),
            "imports_cleanly": 1.0,
            "has_library_dict": 1.0,
            "library_entries": float(len(library)),
            "runs_successfully": 1.0,
        },
        artifacts=artifacts,
    )
