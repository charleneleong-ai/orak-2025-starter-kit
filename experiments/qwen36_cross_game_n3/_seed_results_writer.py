#!/usr/bin/env python3
"""Watch the qwen36 n=3 sweep log + emit one results.jsonl row per completed seed.

PPID=1 daemon. Polls logs/qwen36_n3_seed*.log every 60s. When a log contains
both "All games completed successfully" and a per-game completion table with
final scores, it appends one row to
experiments/qwen36_cross_game_n3/qwen36_a3b_reasoning/results.jsonl matching
the schema autoresearch-pr-updater expects (`evaluation_score` headline,
per-game `scores` dict, `status`).

The row gets emitted once per (seed_run_id); re-runs are idempotent — already-
written run_ids are skipped on rescan.

Launch (from /workspace/orak-qwen36-n3):
    setsid nohup .venv/bin/python -u \
        experiments/qwen36_cross_game_n3/_seed_results_writer.py \
        </dev/null >>logs/seed_results_writer_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/workspace/orak-qwen36-n3")
LOG_DIR = ROOT / "logs"
RESULTS_PATH = ROOT / "experiments/qwen36_cross_game_n3/qwen36_a3b_reasoning/results.jsonl"
POLL_S = 60
MODEL = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"

# rich.Console renders the per-game completion table with these markers
GAME_SCORE_RE = re.compile(r"(Pokemon Red|Super Mario|2048|StarCraft)\s+✅\s+Completed\s+([0-9.]+)")
SEED_DONE_RE = re.compile(r"All games completed successfully")
SEED_ID_RE = re.compile(r"run_id=(qwen36_n3_seed(\d+)_[\dT_Z]+)")
GAME_KEY = {
    "Pokemon Red": "pokemon_red",
    "Super Mario": "super_mario",
    "2048": "twenty_fourty_eight",
    "StarCraft": "star_craft",
}


def already_written(seed_run_id: str) -> bool:
    if not RESULTS_PATH.exists():
        return False
    for line in RESULTS_PATH.open():
        try:
            if json.loads(line).get("run_id") == seed_run_id:
                return True
        except json.JSONDecodeError:
            continue
    return False


def emit_row(seed: int, seed_run_id: str, scores: dict[str, float]) -> None:
    score_values = list(scores.values())
    mean = sum(score_values) / len(score_values) if score_values else 0.0
    row = {
        "experiment": seed,
        "variant": "qwen36_cross_game_n3",
        "run_id": seed_run_id,
        "model": MODEL,
        "seed": seed,
        "evaluation_score": mean,
        "scores": scores,
        "n_episodes": len(scores),
        "status": "KEEP",
        "timestamp": datetime.now(UTC).isoformat(),
        "tags": ["qwen", "qwen3.6-35b-a3b-reasoning", "cross_game_n3"],
        "description": (
            f"Qwen3.6-35B-A3B-Reasoning cross-game n=3 seed {seed} — "
            f"mean {mean:.2f}% across {len(scores)} games"
        ),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"[results_writer] emitted seed {seed} score {mean:.2f}% — {seed_run_id}",
        flush=True,
    )


def scan_log_for_seed(log_path: Path) -> None:
    text = log_path.read_text(errors="ignore")
    if not SEED_DONE_RE.search(text):
        return
    seed_match = SEED_ID_RE.search(text)
    if not seed_match:
        return
    seed_run_id = seed_match.group(1)
    if already_written(seed_run_id):
        return
    seed = int(seed_match.group(2))
    scores = {GAME_KEY[game]: float(score) for game, score in GAME_SCORE_RE.findall(text)}
    if scores:
        emit_row(seed, seed_run_id, scores)


def main() -> None:
    print(
        f"[results_writer] watching {LOG_DIR}/qwen36_n3_seed*.log every {POLL_S}s → {RESULTS_PATH}",
        flush=True,
    )
    while True:
        try:
            for log_path in LOG_DIR.glob("qwen36_n3_seed*.log"):
                scan_log_for_seed(log_path)
        except Exception as e:
            print(f"[results_writer] error: {e}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
