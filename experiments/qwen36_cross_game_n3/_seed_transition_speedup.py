#!/usr/bin/env python3
"""After seed 1 finishes, swap vLLM to MTP speculative-decoding + max-model-len=8192,
then relaunch run_sweep.sh for seeds 2-3.

Expected gain on seeds 2 + 3: ~1.5-2.5× per-step throughput
  - MTP speculative-decoding: 1.8-2.5× on accepted tokens (HF docs: 70-89% acceptance)
  - max-model-len 12288 → 8192: more KV slots → better concurrent batching across
    the 4 games (direct attack on 2048's vLLM-contention slowness)

Trigger: watches the sweep wrapper log for "Seed 1 OK". Fires once, then exits.
  1. Kills the sweep wrapper bash + run.py child (before bash auto-starts seed 2)
  2. Stops vLLM
  3. Relaunches vLLM with --speculative-config + --max-model-len 8192
  4. Waits for vLLM /v1/models to respond
  5. Launches a new run_sweep.sh with START_SEED=2

Caveats:
  - MTP may produce subtly different outputs than the seed-1 config (the draft
    model isn't bit-identical to the main model). Seeds 1 vs 2-3 are no longer
    a perfectly-matched n=3 — there's a small "config drift" confound.
  - If seed 1 fails ("Seed 1 FAILED"), this script does NOT fire — the sweep
    continues on the original config.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/orak-qwen36-n3")
LOG_DIR = ROOT / "logs"
SWEEP_WRAPPER_PROC = "run_sweep.sh"
VLLM_PROC_MATCH = "vllm.entrypoints.openai.api_server"
POLL_S = 30
VLLM_READY_TIMEOUT_S = 600  # 10 min for model reload
VLLM_HEALTH_URL = "http://localhost:8000/v1/models"

SEED1_DONE_RE = re.compile(r"Seed 1 OK\b")


def pids_matching(needle: str) -> list[int]:
    out = subprocess.check_output(["pgrep", "-f", needle], text=True).strip().splitlines()
    return [int(p) for p in out if p]


def kill_pids(pids: list[int], sig: int = signal.SIGTERM) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def vllm_ready() -> bool:
    try:
        with urllib.request.urlopen(VLLM_HEALTH_URL, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_vllm() -> bool:
    deadline = time.time() + VLLM_READY_TIMEOUT_S
    while time.time() < deadline:
        if vllm_ready():
            return True
        time.sleep(5)
    return False


def relaunch_vllm_fast() -> None:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = LOG_DIR / f"vllm_qwen36_fast_{ts}.log"
    spec = '{"method": "mtp", "num_speculative_tokens": 2}'
    cmd = [
        "setsid",
        "nohup",
        "env",
        "QWEN_GPU_UTIL=0.85",
        "QWEN_MAX_MODEL_LEN=8192",
        "bash",
        "-c",
        f"./serving/qwen_serve.sh palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 "
        f"--speculative-config '{spec}'",
    ]
    print(f"[transition] relaunching vLLM (MTP, max-len=8192) → {log_path}", flush=True)
    with open(log_path, "ab") as logf:
        subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def relaunch_sweep_from_seed_2() -> None:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = LOG_DIR / f"qwen36_n3_sweep_fast_{ts}.log"
    print(f"[transition] launching seeds 2-3 sweep → {log_path}", flush=True)
    with open(log_path, "ab") as logf:
        subprocess.Popen(
            [
                "setsid",
                "nohup",
                "env",
                "START_SEED=2",
                "N_SEEDS=3",
                "./experiments/qwen36_cross_game_n3/run_sweep.sh",
            ],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def seed1_done(sweep_log: Path) -> bool:
    if not sweep_log.exists():
        return False
    return bool(SEED1_DONE_RE.search(sweep_log.read_text(errors="ignore")))


def find_active_sweep_log() -> Path | None:
    candidates = sorted(LOG_DIR.glob("qwen36_n3_sweep_*.log"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> None:
    print(f"[transition] watching for 'Seed 1 OK' every {POLL_S}s", flush=True)
    while True:
        sweep_log = find_active_sweep_log()
        if sweep_log and seed1_done(sweep_log):
            print(f"[transition] seed 1 done — sweep log: {sweep_log}", flush=True)
            break
        time.sleep(POLL_S)

    # 1. Kill the original sweep wrapper + child run.py before seed 2 auto-starts
    sweep_pids = pids_matching(SWEEP_WRAPPER_PROC)
    print(f"[transition] killing sweep wrapper pids: {sweep_pids}", flush=True)
    kill_pids(sweep_pids, signal.SIGTERM)
    # Give bash + python time to exit cleanly
    time.sleep(5)
    kill_pids(pids_matching(SWEEP_WRAPPER_PROC), signal.SIGKILL)
    # Also catch any orphan run.py children
    runpy_pids = pids_matching("run.py.*qwen36_a3b_reasoning")
    if runpy_pids:
        print(f"[transition] killing residual run.py: {runpy_pids}", flush=True)
        kill_pids(runpy_pids, signal.SIGTERM)
        time.sleep(3)
        kill_pids(pids_matching("run.py.*qwen36_a3b_reasoning"), signal.SIGKILL)

    # 2. Stop vLLM
    vllm_pids = pids_matching(VLLM_PROC_MATCH)
    print(f"[transition] stopping vLLM pids: {vllm_pids}", flush=True)
    kill_pids(vllm_pids, signal.SIGTERM)
    time.sleep(8)
    kill_pids(pids_matching(VLLM_PROC_MATCH), signal.SIGKILL)
    time.sleep(3)

    # 3. Restart vLLM with MTP + lower max-model-len
    relaunch_vllm_fast()

    # 4. Wait for vLLM to come up
    print(f"[transition] waiting up to {VLLM_READY_TIMEOUT_S}s for vLLM ready ...", flush=True)
    if not wait_for_vllm():
        print("[transition] vLLM did NOT come up in time — manual intervention needed", flush=True)
        return
    print("[transition] vLLM ready (MTP, max-len=8192)", flush=True)

    # 5. Launch seeds 2-3
    relaunch_sweep_from_seed_2()
    print("[transition] done — seeds 2-3 launched on fast config", flush=True)


if __name__ == "__main__":
    main()
