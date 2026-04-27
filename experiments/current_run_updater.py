"""In-flight RUNNING-dot daemon for the autoresearch chart.

Watches the latest `logs/autoresearch_*.log` and keeps
`experiments/<TAG>/current_run.json` in sync with whatever iteration is
currently in flight. When the chart re-renders, plot_progress reads this
sidecar and adds an extra RUNNING-status point at the rightmost x position
so reviewers see "this iter is mid-flight" instead of waiting for it to
finish + commit.

Runs as a detached daemon (setsid + nohup) so it survives Claude / SSH
disconnects.

Logic:
  * Most recent `Iteration N/M` line with no matching `Autoresearch
    complete` or `Iteration N+1/M` below it → write current_run.json.
  * Otherwise → delete current_run.json.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

# ────────────────────────── EDIT FOR YOUR PROJECT ──────────────────────────
TAG = "unified_macla"
CONFIG_NAME = "gemma"
PER_CONFIG = False  # True → experiments/<TAG>/<CONFIG_NAME>/, False → flat
POLL_S = 30
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

_BASE = ROOT / "experiments" / TAG / (CONFIG_NAME if PER_CONFIG else "")
SIDECAR = _BASE / "current_run.json"
RESULTS = _BASE / "results.jsonl"

# Orak's autoresearch.py emits these lines (no leading timestamp wrapper):
#   "# Iteration 3/30"
#   "  Run ID: 20260427_171234 — triage monitoring active"
#   "Autoresearch complete after N iterations"
ITER_START_RE = re.compile(r"# Iteration (?P<n>\d+)/(?P<m>\d+)")
ITER_END_RE = re.compile(r"Autoresearch complete after (?P<n>\d+) iterations")
RUN_ID_RE = re.compile(r"Run ID: (?P<rid>\d{8}_\d{6})")
DESC_RE = re.compile(r"Description: (?P<desc>.+)")
WANDB_RE = re.compile(r"https://wandb\.ai/[\w\-./]+/runs/[\w\-]+")
# ──────────────────────────────────────────────────────────────────────────


def _latest_log() -> Path | None:
    logs = sorted(LOG_DIR.glob("autoresearch_*T*Z.log"))
    return logs[-1] if logs else None


def _experiment_count() -> int:
    if not RESULTS.exists():
        return 0
    return sum(1 for line in RESULTS.read_text().splitlines() if line.strip())


def _tick() -> None:
    log = _latest_log()
    if log is None:
        return
    text = log.read_text(errors="replace")

    starts = list(ITER_START_RE.finditer(text))
    ends = list(ITER_END_RE.finditer(text))
    if not starts:
        return

    last = starts[-1]
    iter_n = int(last.group("n"))
    iter_m = int(last.group("m"))

    # If "Autoresearch complete" appears AFTER the last Iter line, sweep done
    if ends and ends[-1].start() > last.start():
        if SIDECAR.exists():
            SIDECAR.unlink()
            print(f"[updater] sweep complete after iter {iter_n}/{iter_m} — sidecar removed")
        return

    after_iter = text[last.end():]
    m_rid = RUN_ID_RE.search(after_iter)
    run_id = m_rid.group("rid") if m_rid else ""

    m_desc = DESC_RE.search(after_iter)
    desc = m_desc.group("desc").strip() if m_desc else f"iter {iter_n}/{iter_m}"

    chunk = text[last.start():]
    urls = WANDB_RE.findall(chunk)
    wandb_url = urls[-1] if urls else ""

    payload = {
        "experiment": _experiment_count(),
        "config_name": CONFIG_NAME,
        "description": desc,
        "notes": desc,
        "iter_marker": f"Iteration {iter_n}/{iter_m}",
        "run_id": run_id,
        "log_path": str(log),
        "wandb_url": wandb_url,
    }

    if SIDECAR.exists():
        try:
            cur = json.loads(SIDECAR.read_text())
            if cur == payload:
                return
        except json.JSONDecodeError:
            pass
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(payload, indent=2))
    print(f"[updater] sidecar → iter {iter_n}/{iter_m} run_id={run_id or 'pending'} "
          f"E{payload['experiment']}")


def main() -> None:
    print(f"[updater] starting — poll every {POLL_S}s, log dir={LOG_DIR}")
    while True:
        try:
            _tick()
        except Exception as e:
            print(f"[updater] tick error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
