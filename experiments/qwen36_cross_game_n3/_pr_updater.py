#!/usr/bin/env python
"""Live PR status-table refresher for the qwen36 cross-game n=3 sweep.

Every ``--poll-s`` seconds, rebuild a per-game status table from the live
sweep artefacts and PATCH it into PR #113's body between the markers::

    <!-- SWEEP_STATUS_START -->
    (table goes here)
    <!-- SWEEP_STATUS_END -->

Both markers must already exist in the body (one-time setup). The body is
only re-edited when the rendered section actually changes, so idle ticks
are silent.

This sweep is bespoke (no autoresearch ``results.jsonl``); live data comes
straight from the per-game logs:

- ``game_logs/<game>/<run_id>/game_states.jsonl`` — step count (line count)
  and the latest ``evaluation_score`` / ``game_score``.
- StarCraft is rescored 0-100 via ``star_craft.progress.run_progress``
  (PR #115) rather than the flat binary the live process still writes.
- ``/proc/<pid>/status`` + ``nvidia-smi`` — thread count and GPU.

Run detached so it survives SSH / agent-session death; verify ``PPID=1``::

    setsid nohup python -u experiments/qwen36_cross_game_n3/_pr_updater.py \\
        </dev/null >>logs/pr_updater_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import typer

from evaluation_utils.mcp_game_servers.star_craft.progress import extract_metrics, run_progress

MARKER_START = "<!-- SWEEP_STATUS_START -->"
MARKER_END = "<!-- SWEEP_STATUS_END -->"

# Display order + step cap (max_steps) per game.
GAMES: list[tuple[str, int]] = [
    ("super_mario", 1000),
    ("pokemon_red", 1200),
    ("twenty_fourty_eight", 1000),
    ("star_craft", 2500),
]
PRETTY = {
    "super_mario": "super_mario",
    "pokemon_red": "pokemon_red",
    "twenty_fourty_eight": "twenty_fourty_eight",
    "star_craft": "star_craft",
}


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%MZ")


def _latest_run_dir(game_logs: Path, game: str) -> Path | None:
    """Most-recently-written run dir for a game (follows seed transitions)."""
    candidates = [d for d in (game_logs / game).glob("*") if (d / "game_states.jsonl").exists()]
    return max(candidates, key=lambda d: (d / "game_states.jsonl").stat().st_mtime, default=None)


# Per-run-dir incremental cache so each tick parses only the new tail of
# game_states.jsonl instead of re-reading the whole (60 MB+) file.
#   offset     — bytes of complete lines already consumed
#   n_lines    — running step count
#   last_line  — most recent complete line (for the latest eval score)
#   sc2_steps  — accumulated extract_metrics() dicts (StarCraft only; tiny)
_CACHE: dict[str, dict] = {}


def _ingest(run_dir: Path, want_sc2: bool) -> dict:
    """Read only the new complete lines since the last tick; update the cache.

    Binary-mode byte offsets keep unicode-safe; a trailing partial line (the
    writer mid-append) is left unconsumed until it is newline-terminated, so a
    half-written JSON row never reaches ``json.loads``.
    """
    path = run_dir / "game_states.jsonl"
    key = str(run_dir)
    size = path.stat().st_size
    cache = _CACHE.get(key)
    if cache is None or size < cache["offset"]:  # first sight, or truncated/rotated
        cache = {"offset": 0, "n_lines": 0, "last_line": "", "sc2_steps": []}
        _CACHE[key] = cache
    with path.open("rb") as fh:
        fh.seek(cache["offset"])
        chunk = fh.read()
    nl = chunk.rfind(b"\n")
    if nl == -1:  # no newly-completed line yet
        return cache
    complete = chunk[: nl + 1]
    cache["offset"] += len(complete)
    for raw in complete.decode("utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        cache["n_lines"] += 1
        cache["last_line"] = raw
        if want_sc2:
            try:
                obs = json.loads(raw).get("obs", "")
            except json.JSONDecodeError:
                obs = raw
            cache["sc2_steps"].append(extract_metrics(str(obs)))
    return cache


def _seed_of(run_dir: Path) -> str:
    m = re.search(r"seed(\d+)", run_dir.name)
    return m.group(1) if m else "?"


def _last_eval(last_line: str) -> float:
    try:
        return float(json.loads(last_line).get("evaluation_score", 0.0))
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return 0.0


def _game_row(game: str, cap: int, game_logs: Path) -> tuple[str, str, int]:
    """Return (markdown_row, seed, steps) for one game."""
    run_dir = _latest_run_dir(game_logs, game)
    if run_dir is None:
        return f"| **{PRETTY[game]}** | — | {cap} | — | ⏳ not started |", "?", 0
    cache = _ingest(run_dir, want_sc2=(game == "star_craft"))
    steps = cache["n_lines"]
    seed = _seed_of(run_dir)
    done = steps >= cap
    if game == "star_craft":
        score = f"progress **{run_progress(cache['sc2_steps'])['starcraft_progress']:.1f}** / 100"
    else:
        score = f"eval **{_last_eval(cache['last_line']):.1f}**"
    state = "✅ done" if done else "▶ running"
    step_cell = "done" if done else str(steps)
    return f"| **{PRETTY[game]}** | {step_cell} | {cap} | {score} | {state} |", seed, steps


def _proc_threads(pattern: str) -> int | None:
    try:
        pid = subprocess.check_output(["pgrep", "-f", pattern], text=True).split()[0]
        status = Path(f"/proc/{pid}/status").read_text()
        m = re.search(r"^Threads:\s*(\d+)", status, re.M)
        return int(m.group(1)) if m else None
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def _gpu() -> str | None:
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
        util, used, total = (x.strip() for x in out.split(","))
        return f"{util}% util · {int(used) / 1024:.1f}/{int(total) / 1024:.0f} GB"
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def render_section(game_logs: Path, run_pattern: str) -> str:
    rows, seeds = [], set()
    for game, cap in GAMES:
        row, seed, _ = _game_row(game, cap, game_logs)
        rows.append(row)
        if seed != "?":
            seeds.add(seed)
    seed_label = f"seed {min(seeds)} of 3" if seeds else "seed ? of 3"
    health = []
    if (threads := _proc_threads(run_pattern)) is not None:
        health.append(f"threads **{threads}**")
    if (gpu := _gpu()) is not None:
        health.append(gpu)
    health_line = " · ".join(health) if health else "process not found"
    return (
        f"### Live sweep status — {seed_label} @ {_ts()}\n\n"
        "| Game | Step | Cap | Score | State |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(rows)
        + f"\n\n**System health:** {health_line} · _auto-refreshed by "
        "[`_pr_updater.py`](../tree/feat/qwen36-cross-game-n3/experiments/qwen36_cross_game_n3/_pr_updater.py)._"
    )


def patch_body(pr: int, repo: str, section: str) -> bool:
    """PATCH the marker section into the PR body. Returns True if it changed."""
    body = subprocess.check_output(
        ["gh", "pr", "view", str(pr), "--repo", repo, "--json", "body", "--jq", ".body"],
        text=True,
    )
    if MARKER_START not in body or MARKER_END not in body:
        raise SystemExit(f"PR #{pr} body is missing the {MARKER_START} / {MARKER_END} markers")
    replacement = f"{MARKER_START}\n{section}\n{MARKER_END}"
    new_body = re.sub(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
        lambda _: replacement,
        body,
        flags=re.S,
    )
    if new_body == body:
        return False
    tmp = Path("/tmp/_pr_body.md")
    tmp.write_text(new_body)
    subprocess.run(
        ["gh", "pr", "edit", str(pr), "--repo", repo, "--body-file", str(tmp)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return True


def main(
    pr: int = 113,
    repo: str = "charleneleong-ai/orak-2025-starter-kit",
    game_logs: Path = Path("game_logs"),
    run_pattern: str = "run.py -c qwen36",
    poll_s: int = 600,
    once: bool = False,
) -> None:
    while True:
        try:
            section = render_section(game_logs, run_pattern)
            changed = patch_body(pr, repo, section)
            print(f"[{_ts()}] {'patched' if changed else 'no change'}", flush=True)
        except Exception as exc:  # noqa: BLE001 — daemon must never die on a transient error
            print(f"[{_ts()}] error: {exc}", flush=True)
        if once:
            break
        time.sleep(poll_s)


if __name__ == "__main__":
    typer.run(main)
