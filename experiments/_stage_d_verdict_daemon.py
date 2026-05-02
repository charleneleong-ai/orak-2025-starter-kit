"""Stage D ablation verdict daemon — posts the cross-game scoreboard to PR #28.

Polls every N seconds. When both `experiments/stage_d_ablation_2048/gemma/results.jsonl`
and `experiments/stage_d_ablation_mario/gemma/results.jsonl` have at least
TARGET_ITERS rows (filtered by their respective game), computes per-game best
score deltas vs Stage A baseline + Stage C vmem, posts a verdict comment to
PR #28, and exits.

Detach with:
    setsid nohup .venv/bin/python -u experiments/_stage_d_verdict_daemon.py \
        > logs/stage_d_verdict.log 2>&1 < /dev/null & disown

Mirrors `experiments/_pr_updater.py` style — inline JSONL load + git-free
(comment-only via gh; no commits/pushes from this daemon).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PR_NUMBER = 28
REPO = "charleneleong-ai/orak-2025-starter-kit"
TARGET_ITERS = 2          # both ablation sweeps run with --iters 2
POLL_S = 300              # 5 min
MAX_WAIT_S = 4 * 60 * 60  # 4 hours; daemon exits + comments TIMEOUT after this

# (tag, game-field-value, label)  — order: Stage A, Stage C vmem, Stage D
_2048_SWEEPS = [
    ("harness_check",         "twenty_fourty_eight", "Stage A baseline"),
    ("cognitive_check_v2",    "twenty_fourty_eight", "Stage C (vmem)"),
    ("stage_d_ablation_2048", "twenty_fourty_eight", "Stage D (vmem+planner)"),
]
_MARIO_SWEEPS = [
    ("harness_check",          "super_mario", "Stage A baseline"),
    ("mario_check",            "super_mario", "Stage C (vmem)"),
    ("stage_d_ablation_mario", "super_mario", "Stage D (vmem+planner)"),
]
CONFIG_DIR = "gemma"


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _results_path(tag: str) -> Path:
    """Try per-config layout first, then flat."""
    p = ROOT / "experiments" / tag / CONFIG_DIR / "results.jsonl"
    if p.exists():
        return p
    return ROOT / "experiments" / tag / "results.jsonl"


def _load(tag: str) -> list[dict]:
    p = _results_path(tag)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _get_score(row: dict) -> float:
    if "evaluation_score" in row:
        return row["evaluation_score"]
    return row.get("score", 0.0)


def _filter_game(rows: list[dict], game: str) -> list[dict]:
    return [r for r in rows if r.get("game") == game]


def _best(rows: list[dict]) -> float:
    return max((_get_score(r) for r in rows), default=0.0)


def _ablation_ready() -> bool:
    """Both Stage D ablation sweeps have at least TARGET_ITERS rows for their game."""
    rows_2048 = _filter_game(_load("stage_d_ablation_2048"), "twenty_fourty_eight")
    rows_mario = _filter_game(_load("stage_d_ablation_mario"), "super_mario")
    return len(rows_2048) >= TARGET_ITERS and len(rows_mario) >= TARGET_ITERS


def _verdict(stage_d: float, stage_c: float) -> str:
    if stage_c <= 0:
        return "?"
    delta_pct = (stage_d - stage_c) / stage_c * 100
    if delta_pct >= 10:
        return "HELPS"
    if delta_pct <= -10:
        return "REGRESSES"
    return "NEUTRAL"


def _row(game_label: str, sweeps: list[tuple[str, str, str]]) -> tuple[str, str]:
    """Return (markdown_row, verdict) for the given game's sweeps."""
    cells = []
    scores = {}
    for tag, game, label in sweeps:
        rows = _filter_game(_load(tag), game)
        if not rows:
            cells.append("_(missing)_")
            scores[label] = None
        else:
            best = _best(rows)
            cells.append(f"{best:.2f} _({len(rows)} iters)_")
            scores[label] = best

    a = scores["Stage A baseline"]
    c = scores["Stage C (vmem)"]
    d = scores["Stage D (vmem+planner)"]
    if d is None:
        return f"| {game_label} | {' | '.join(cells)} | ? | ? | _(no Stage D data)_ |", "?"

    delta_c = f"{(d - c) / c * 100:+.0f}%" if c and c > 0 else "?"
    delta_a = f"{(d - a) / a * 100:+.0f}%" if a and a > 0 else "?"
    verdict = _verdict(d, c) if c else "?"
    return (
        f"| {game_label} | {' | '.join(cells)} | {delta_c} | {delta_a} | **{verdict}** |",
        verdict,
    )


def _build_comment(timed_out: bool = False) -> str:
    row_2048, v_2048 = _row("2048", _2048_SWEEPS)
    row_mario, v_mario = _row("mario", _MARIO_SWEEPS)

    verdicts = {v_2048, v_mario}
    if "HELPS" in verdicts:
        implication = (
            "Cross-game scoreboard mis-attributes — Stage D substrate is empirically "
            "validated on at least one game. Update `training_plan.md` + "
            "`agentic_rl_options.md` to claim Stage D substrate empirically, not "
            "theoretically."
        )
    elif "REGRESSES" in verdicts:
        implication = (
            "Stage D regresses scores on at least one game — inference cost from "
            "subtask injection is real and the model is being distracted by "
            "sub-goals it can't act on. Treat Stage D as a partial-fit primitive "
            "(pokemon-specific) in the docs."
        )
    elif verdicts == {"NEUTRAL"}:
        implication = (
            "Picked path stands as written. Stage D adds no measurable lift on "
            "mario/2048 — confirms the prior bottleneck-driven theory."
        )
    else:
        implication = "?"

    header = "## Stage D ablation results — verdict"
    if timed_out:
        header += " (timed out)"

    return (
        f"{header}\n\n"
        f"_(automated by `experiments/_stage_d_verdict_daemon.py` at {_ts()})_\n\n"
        f"| Game | Stage A | Stage C (vmem) | Stage D (vmem+planner) | Δ vs C | Δ vs A | Verdict |\n"
        f"|---|---|---|---|---|---|---|\n"
        f"{row_2048}\n"
        f"{row_mario}\n\n"
        f"**Implication for the picked path** "
        f"(`docs/experiments/gemma/training_plan.md` / `agentic_rl_options.md`): "
        f"{implication}\n"
    )


def _post_comment(body: str) -> bool:
    """Post a comment to PR #28 via gh. Returns True on success."""
    proc = subprocess.run(
        ["gh", "pr", "comment", str(PR_NUMBER), "--repo", REPO, "--body", body],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if proc.returncode != 0:
        print(f"[{_ts()}] gh pr comment failed: {proc.stderr.strip()[:300]}", flush=True)
        return False
    print(f"[{_ts()}] posted to PR #{PR_NUMBER}: {proc.stdout.strip()}", flush=True)
    return True


def main() -> int:
    started = time.time()
    print(f"[{_ts()}] daemon up, polling every {POLL_S}s, timeout {MAX_WAIT_S}s", flush=True)
    while True:
        if _ablation_ready():
            print(f"[{_ts()}] both ablation sweeps reached {TARGET_ITERS} iters — posting", flush=True)
            ok = _post_comment(_build_comment())
            return 0 if ok else 1

        elapsed = time.time() - started
        if elapsed > MAX_WAIT_S:
            print(f"[{_ts()}] timeout — posting partial verdict", flush=True)
            ok = _post_comment(_build_comment(timed_out=True))
            return 2 if ok else 1

        print(f"[{_ts()}] not ready yet, sleeping {POLL_S}s", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
