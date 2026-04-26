"""Periodic PR refresher for the MACLA autoresearch sweep.

Polls every N seconds and:
1. Re-renders `experiments/{TAG}/progress.png` and `progress.html` from
   `results.jsonl` via experiment_progress.plot_progress().
2. If the PNG changed: git add + commit + push so the embedded image
   in the PR body refreshes (GitHub serves it via `?raw=true`).
3. Re-builds the sweep-narrative table from `results.jsonl` and PATCHes
   PR #PR_NUMBER's body between the markers
   `<!-- SWEEP_NARRATIVE_START -->` … `<!-- SWEEP_NARRATIVE_END -->`.

Detach with:
    setsid nohup .venv/bin/python -u experiments/_pr_updater.py \\
        > logs/pr_updater.log 2>&1 < /dev/null & disown

Adapted from charleneleong-ai/gemma4-rlvr#4 _pr_updater.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TAG = "unified_macla"
CONFIG_TYPE = "gemma"  # configs/<game>/agent/<CONFIG_TYPE>.yaml — embedded in chart metadata
RESULTS = ROOT / "experiments" / TAG / "results.jsonl"
PNG = ROOT / "experiments" / TAG / "progress.png"
HTML = ROOT / "experiments" / TAG / "progress.html"
POLL_S = 600  # 10 min
PR_NUMBER = 20
REPO = "charleneleong-ai/orak-2025-starter-kit"
BRANCH = "feat/macla-sweep-live"
MARKER_START = "<!-- SWEEP_NARRATIVE_START -->"
MARKER_END = "<!-- SWEEP_NARRATIVE_END -->"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _build_narrative() -> str:
    if not RESULTS.exists():
        return "_(no results yet)_"
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    if not rows:
        return "_(no results yet)_"
    n_keep = sum(1 for r in rows if r.get("status") in ("KEEP", "BASELINE"))
    n_disc = sum(1 for r in rows if r.get("status") == "DISCARD")
    n_kill = sum(1 for r in rows if r.get("status") == "EARLY_KILL")
    runtime = sum(r.get("runtime_min", 0) for r in rows)
    best = max(rows, key=lambda r: r.get("evaluation_score", r.get("score", 0)))
    best_score = best.get("evaluation_score", best.get("score", 0))

    lines = [
        f"_Last refresh: {_ts()}._ "
        f"**{len(rows)}** experiments · {n_keep} kept · {n_disc} discarded · {n_kill} killed"
        f" · {runtime:.0f}min total · best so far: **{best_score:.2f}** "
        f"({best.get('game', '?')} #{best.get('experiment', '?')})\n",
        "| # | game | status | score | steps | runtime | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        status = r.get("status", "?")
        score = r.get("evaluation_score", r.get("score", 0))
        steps = r.get("steps", "")
        rt = f"{r.get('runtime_min', 0):.0f}min" if r.get("runtime_min") else ""
        notes = (r.get("notes") or r.get("description") or "").replace("|", "\\|")[:80]
        link = ""
        if r.get("wandb_url"):
            link = f" [↗]({r['wandb_url']})"
        lines.append(
            f"| #{r.get('experiment', '?')} | {r.get('game', '?')} | {status.lower()} | "
            f"{score:.2f} | {steps} | {rt} | {notes}{link} |"
        )
    return "\n".join(lines)


def _refresh_chart() -> bool:
    """Re-render progress.html + progress.png. Returns True if PNG changed."""
    before = PNG.stat().st_mtime if PNG.exists() else -1
    venv_py = ROOT / ".venv" / "bin" / "python"
    proc = subprocess.run(
        [str(venv_py), "-c",
         f"import sys; sys.path.insert(0, '{ROOT}'); "
         f"from experiments.experiment_progress import plot_progress; "
         f"plot_progress(tag='{TAG}', config_type='{CONFIG_TYPE}')"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"[pr_updater] chart render failed: {proc.stderr.strip()[:200]}")
        return False
    if not PNG.exists():
        return False
    return PNG.stat().st_mtime > before


def _git_push_png_if_changed() -> bool:
    """Stage png+html, commit + push only if working-tree differs from HEAD."""
    subprocess.run(["git", "add", str(PNG), str(HTML)], cwd=str(ROOT), check=False)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=str(ROOT),
    )
    if diff.returncode == 0:
        return False
    subprocess.run(
        ["git", "commit", "-m", f"docs: refresh autoresearch progress ({_ts()})"],
        cwd=str(ROOT), check=False,
    )
    push = subprocess.run(
        ["git", "push", "origin", BRANCH], cwd=str(ROOT),
        capture_output=True, text=True,
    )
    if push.returncode != 0:
        print(f"[pr_updater] push failed: {push.stderr.strip()[:200]}")
    return push.returncode == 0


def _patch_pr_body(narrative: str) -> bool:
    body_proc = subprocess.run(
        ["gh", "api", f"repos/{REPO}/pulls/{PR_NUMBER}", "--jq", ".body"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if body_proc.returncode != 0:
        print(f"[pr_updater] gh api failed: {body_proc.stderr.strip()[:200]}")
        return False
    body = body_proc.stdout
    if MARKER_START not in body or MARKER_END not in body:
        print(f"[pr_updater] markers missing in PR #{PR_NUMBER} body — manual setup needed")
        return False
    pre, _, rest = body.partition(MARKER_START)
    _, _, post = rest.partition(MARKER_END)
    new = pre + MARKER_START + "\n" + narrative + "\n" + MARKER_END + post
    if new == body:
        return False
    payload = json.dumps({"body": new})
    proc = subprocess.run(
        ["gh", "api", f"repos/{REPO}/pulls/{PR_NUMBER}",
         "--method", "PATCH", "--input", "-"],
        input=payload, text=True, capture_output=True, cwd=str(ROOT),
    )
    if proc.returncode != 0:
        print(f"[pr_updater] PATCH failed: {proc.stderr.strip()[:200]}")
        return False
    return True


def main() -> None:
    print(f"[pr_updater] starting — poll every {POLL_S}s, PR #{PR_NUMBER} on {REPO}, tag={TAG}")
    while True:
        try:
            png_changed = _refresh_chart()
            pushed = _git_push_png_if_changed() if png_changed else False
            narrative = _build_narrative()
            patched = _patch_pr_body(narrative)
            print(f"[pr_updater] {_ts()} — png_changed={png_changed} pushed={pushed} pr_patched={patched}", flush=True)
        except Exception as e:
            print(f"[pr_updater] tick error: {e}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
