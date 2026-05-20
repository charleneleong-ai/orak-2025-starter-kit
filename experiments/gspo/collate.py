"""GSPO data collation.

Reads a sweep iter's ``game_states.jsonl`` + ``evaluation_summary.json`` and
emits one ``GSPOSample`` per env step. Sequence-level reward: every step in
the same iter shares the trajectory's final score (normalised to 0-1 by
``score_max``), matching GSPO's sequence-level advantage formulation.

Notes for the GSPO trainer downstream:
  * ``group_id`` defaults to ``run_id`` so a single sweep flows through the
    pipeline. Real multi-rollout groups (K trajectories from one checkpoint
    state, used for group-relative advantage estimation) need a separate
    re-roll launcher that fixes the same ``group_id`` across the K rollouts.
  * ``prompt`` is the env observation string. Full planner prompts (system +
    observation + history + active_subgoal) live in Weave traces — query
    ``weave.Client.get_calls`` and join by ``(run_id, iter_step)`` once
    Weave joining lands.
  * ``completion`` is the parsed tool call string. For raw policy tokens
    (``<think>...</think>`` + tool call) join with Weave as above.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass(frozen=True, slots=True)
class GSPOSample:
    run_id: str
    iter_step: int
    prompt: str
    completion: str
    reward: float
    group_id: str


def collate_iter(run_dir: Path, *, score_max: float = 7.0) -> list[GSPOSample]:
    states_path = run_dir / "game_states.jsonl"
    summary_path = run_dir / "evaluation_summary.json"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    final_score = float(json.loads(summary_path.read_text())["episodes"][0]["final_score"])
    reward = final_score / score_max
    run_id = run_dir.name
    samples: list[GSPOSample] = []
    for line in states_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(
            GSPOSample(
                run_id=run_id,
                iter_step=int(row.get("iteration", len(samples) + 1)),
                prompt=row.get("obs", {}).get("obs_str", ""),
                completion=row.get("action", ""),
                reward=reward,
                group_id=run_id,
            )
        )
    return samples


@app.command()
def collate(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="iter run dir with game_states.jsonl"
    ),
    out: Path = typer.Option(Path("gspo_dataset.jsonl"), "--out", "-o", help="output jsonl"),
    score_max: float = typer.Option(
        7.0, "--score-max", help="env score ceiling for reward normalisation"
    ),
) -> None:
    samples = collate_iter(run_dir, score_max=score_max)
    with out.open("w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")
    typer.echo(f"wrote {len(samples)} samples (reward={samples[0].reward:.3f}) -> {out}")


if __name__ == "__main__":
    app()
