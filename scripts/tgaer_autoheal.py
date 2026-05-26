#!/usr/bin/env python3
"""TGAER PR1 autoheal — keep trying to fill n=3 each-side slots until satisfied.

Strategy:
- Every POLL_INTERVAL seconds, scan completed + running rollouts
- For any (side, game, n) slot that isn't done AND isn't currently running,
  launch a fresh attempt (with retryN suffix) if active count < MAX_PARALLEL
- Cap per-slot at MAX_ATTEMPTS so we don't loop forever on broken slots
- Exit when every slot is either DONE or maxed-out on retries

Designed to be launched as PPID=1 daemon and survive SSH disconnects.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

MAX_PARALLEL = 12
MAX_ATTEMPTS = 4
POLL_INTERVAL = 300  # 5 min
LAUNCH_STAGGER = 60  # 1 min between launches within a sweep

GAMES = ("pokemon_red", "super_mario", "twenty_fourty_eight", "star_craft")
NS = ("1", "2", "3")
SIDES = ("baseline", "detector")

LOG_FILE = f"/workspace/orak-futile-detector/logs/tgaer_autoheal_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)


def worktree_for(side: str) -> str:
    return "orak-master-baselines" if side == "baseline" else "orak-futile-detector"


def get_completed() -> set[tuple[str, str, str]]:
    done = set()
    for side in SIDES:
        wt = worktree_for(side)
        for game in GAMES:
            for d in glob.glob(f"/workspace/{wt}/game_logs/{game}/tgaer_{side}_{game}_n*"):
                if (Path(d) / "evaluation_summary.json").exists():
                    name = os.path.basename(d)
                    m = re.match(rf"tgaer_{side}_{game}_n(\d)", name)
                    if m:
                        done.add((side, game, m.group(1)))
    return done


def get_running() -> set[tuple[str, str, str]]:
    ps = subprocess.run(["ps", "-ef"], capture_output=True, text=True).stdout
    running = set()
    for line in ps.split("\n"):
        if "tgaer_" not in line or "run.py" not in line:
            continue
        m = re.search(r"--run-id\s+(\S+)", line)
        if not m:
            continue
        m2 = re.match(
            r"tgaer_(baseline|detector)_(pokemon_red|super_mario|twenty_fourty_eight|star_craft)_n(\d)",
            m.group(1),
        )
        if m2:
            running.add((m2.group(1), m2.group(2), m2.group(3)))
    return running


def launch(side: str, game: str, n: str, attempt: int) -> str:
    wt = worktree_for(side)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    runid = f"tgaer_{side}_{game}_n{n}_heal{attempt}_{ts}"
    cmd = (
        f"cd /workspace/{wt} && "
        f"setsid nohup ./.venv/bin/python run.py -c gemma_26b --local --games {game} "
        f"--run-id '{runid}' "
        f"-d 'TGAER PR1 autoheal {attempt} for {side} {game} n={n}' "
        f"</dev/null >>logs/{runid}.log 2>&1 & disown"
    )
    subprocess.run(["bash", "-c", cmd], check=False)
    return runid


def main() -> None:
    log(f"autoheal started (max_parallel={MAX_PARALLEL}, max_attempts={MAX_ATTEMPTS}, poll={POLL_INTERVAL}s)")
    all_slots = [(side, g, n) for side in SIDES for g in GAMES for n in NS]
    attempts = {s: 0 for s in all_slots}

    while True:
        done = get_completed()
        running = get_running()
        active = len(running)

        needed = [
            s for s in all_slots
            if s not in done and s not in running and attempts[s] < MAX_ATTEMPTS
        ]

        # Status report
        done_count = sum(1 for s in all_slots if s in done)
        running_count = sum(1 for s in all_slots if s in running)
        gave_up = sum(1 for s in all_slots if attempts[s] >= MAX_ATTEMPTS and s not in done and s not in running)
        log(f"sweep: done={done_count}/24 running={running_count} needed={len(needed)} gave_up={gave_up} active_proc={active}")

        if not needed:
            log("queue empty — all slots done or maxed out")
            break

        # Launch what we can, respecting MAX_PARALLEL
        launched_this_sweep = 0
        for slot in needed:
            current_active = len(get_running())
            if current_active >= MAX_PARALLEL:
                log(f"  pause: active={current_active} >= max={MAX_PARALLEL}")
                break
            side, game, n = slot
            attempts[slot] += 1
            runid = launch(side, game, n, attempts[slot])
            log(f"  launch attempt {attempts[slot]}/{MAX_ATTEMPTS}: {runid}")
            launched_this_sweep += 1
            time.sleep(LAUNCH_STAGGER)

        log(f"sweep complete: launched {launched_this_sweep}, sleeping {POLL_INTERVAL}s")
        time.sleep(POLL_INTERVAL)

    log("autoheal exiting")


if __name__ == "__main__":
    main()
