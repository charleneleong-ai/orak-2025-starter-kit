"""Maintain a single rolling comparative retrospective comment on PR #31.

Each invocation rebuilds the full comparative report from results.jsonl
(all stages completed so far), then PATCHes the existing comment if one
is tracked, else POSTs a new one and saves the ID.

Also PATCHes the PR #31 body's "Rerun results" table between
<!-- PR31_RERUN_TABLE_START --> markers.

Usage (called by postprocess_runs.sh after each append+render):
    uv run python experiments/pr31_ablation_26b/post_retro.py \
        --variant <variant> --game-logs <run_dir>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO / "experiments/pr31_ablation_26b"
RESULTS = SWEEP_DIR / "gemma_26b/results.jsonl"
COMMENT_ID_FILE = SWEEP_DIR / ".retro_comment_id"
PNG_PATH = SWEEP_DIR / "progress.png"
PR_NUM = 31
GH_REPO = "charleneleong-ai/orak-2025-starter-kit"

STAGE_LABELS = {
    "stage_a_26b": "Stage A 26B — vmem OFF · planner OFF",
    "stage_c_26b": "Stage C 26B — vmem ON · planner OFF",
    "stage_b_26b": "Stage B 26B — vmem OFF · planner ON",
    "stage_d_26b": "Stage D 26B — vmem ON · planner ON (300st)",
    "stage_d_plus_26b": "Stage D++ 26B — vmem ON · planner ON (600st + grace=10)",
}

ALL_STAGES = ["stage_a_26b", "stage_c_26b", "stage_b_26b", "stage_d_26b", "stage_d_plus_26b"]


def load_rows() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def stats(run_dir: Path) -> dict:
    eval_log = run_dir / "evaluation.log"
    raw = run_dir / "logs" / "raw_requests.jsonl"
    if not eval_log.exists():
        return {"n_inferences": 0, "progression": [], "actions": {}, "stuck_firings": 0,
                "exhaust_firings": 0, "stuck_pct": 0}
    text = eval_log.read_text(errors="ignore")
    n_inferences = text.count("is_finished:")
    progression, seen, step = [], set(), 0
    for line in text.splitlines():
        if "is_finished:" in line:
            step += 1
            m = re.search(r"Score: (\d+),", line)
            if not m:
                continue
            s = int(m.group(1))
            if s not in seen and s > 0:
                seen.add(s)
                progression.append((s, step))
    actions = Counter(m.group(1) for m in re.finditer(r"use_tool\(([a-z_]+)", text))
    stuck = exhaust = 0
    if raw.exists():
        rt = raw.read_text(errors="ignore")
        stuck = rt.count("Stuck Detector")
        exhaust = rt.count("Exhaust Interactables")
    return {
        "n_inferences": n_inferences,
        "progression": progression,
        "actions": dict(actions.most_common(8)),
        "stuck_firings": stuck,
        "exhaust_firings": exhaust,
        "stuck_pct": (stuck / n_inferences * 100) if n_inferences else 0,
    }


def find_run_dir(variant: str) -> Path | None:
    """Find the latest worktree run-dir for a given variant."""
    base = Path("/tmp/orak-planner-prompt/game_logs/pokemon_red")
    if variant == "stage_d_26b":
        # Stage D ran before the *_stage_d_26b_ naming convention — match the bare timestamp.
        candidates = sorted(base.glob("pr31_rerun_pokemon_2026*"), reverse=True)
        return candidates[0] if candidates else None
    candidates = sorted(base.glob(f"pr31_rerun_pokemon_{variant}_*"), reverse=True)
    return candidates[0] if candidates else None


def results_table(rows: list[dict]) -> str:
    out = ["| Variant | Switches | Score | % | Steps | Stuck-fire % | Status |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        v = r["variant"]
        switches = STAGE_LABELS.get(v, v).split("—", 1)[1].strip() if "—" in STAGE_LABELS.get(v, "") else "?"
        # Reload stats from run dir for stuck %
        rd = find_run_dir(v)
        stuck_pct = "—"
        if rd:
            s = stats(rd)
            if s["n_inferences"]:
                stuck_pct = f"{s['stuck_pct']:.0f}%"
        out.append(
            f"| {v} | {switches} | {r['game_score']:.0f}/7 | "
            f"{r['evaluation_score']:.2f}% | {r['steps']} | {stuck_pct} | {r['status']} |"
        )
    return "\n".join(out)


def md_progression(s: dict) -> str:
    if not s.get("progression"):
        return "_no score gain logged_"
    rows = ["| Score | Step |", "|---|---|"]
    for sc, step in s["progression"]:
        rows.append(f"| {sc} | {step} |")
    return "\n".join(rows)


def md_actions(s: dict) -> str:
    if not s.get("actions"):
        return "_no actions logged_"
    rows = ["| Action | Count |", "|---|---|"]
    for a, c in s["actions"].items():
        rows.append(f"| `{a}` | {c} |")
    return "\n".join(rows)


def stage_section(variant: str, rows: list[dict]) -> str:
    label = STAGE_LABELS.get(variant, variant)
    cur = next((r for r in rows if r["variant"] == variant), None)
    if not cur:
        return f"### {label}\n\n_pending_\n"
    rd = find_run_dir(variant)
    s = stats(rd) if rd else {"progression": [], "actions": {}, "stuck_firings": 0,
                              "exhaust_firings": 0, "stuck_pct": 0, "n_inferences": 0}
    head = f"### {label}\n\n**Result: {cur['game_score']:.0f}/7 ({cur['evaluation_score']:.2f}%)** · {cur['steps']} steps · stuck-fire {s['stuck_pct']:.0f}% · exhaust-fire {s['exhaust_firings']}"
    return (
        f"{head}\n\n"
        f"<details><summary>Score progression + action mix</summary>\n\n"
        f"**Score progression**\n\n{md_progression(s)}\n\n"
        f"**Action mix (top 8)**\n\n{md_actions(s)}\n\n"
        f"</details>\n"
    )


def verdict(rows: list[dict]) -> str:
    by_v = {r["variant"]: r["evaluation_score"] for r in rows}
    a = by_v.get("stage_a_26b")
    c = by_v.get("stage_c_26b")
    b = by_v.get("stage_b_26b")
    d = by_v.get("stage_d_26b")
    dpp = by_v.get("stage_d_plus_26b")

    lines = ["## Interim verdict"]
    if d is not None:
        lines.append(f"- **Stage D baseline**: {d:.2f}% — anchors the 'with everything' result.")
    if a is not None:
        lines.append(f"- **Stage A**: {a:.2f}% — 26B model alone with all recent fixes.")
    if c is not None and a is not None:
        lift = c - a
        lines.append(f"- **vmem-only lift (C − A)**: {lift:+.2f} pp")
    if b is not None and a is not None:
        lift = b - a
        lines.append(f"- **planner-only lift (B − A)**: {lift:+.2f} pp")
    if c is not None and b is not None:
        if abs(c - b) < 5:
            lines.append(f"- C ≈ B: planner and vmem contribute roughly equally in isolation")
        elif b > c:
            lines.append(f"- B > C ({b - c:+.2f} pp): planner is the load-bearing lever")
        else:
            lines.append(f"- C > B ({c - b:+.2f} pp): vmem is the load-bearing lever")
    if d is not None and a is not None and (c is not None or b is not None):
        synergy = d - a - max((c or 0) - (a or 0), 0) - max((b or 0) - (a or 0), 0)
        lines.append(f"- **Synergy (D − A − vmem-lift − planner-lift)**: {synergy:+.2f} pp — positive means the levers compound; near-zero means they're additive")
    if dpp is not None and d is not None:
        lines.append(f"- **Stage D++ vs D (budget+grace)**: {dpp - d:+.2f} pp at 600 steps")
    if not all(by_v.get(v) is not None for v in ALL_STAGES):
        missing = [v for v in ALL_STAGES if by_v.get(v) is None]
        lines.append(f"\n_Pending: {', '.join(missing)}_")
    else:
        lines.append("\n**All stages complete.** See above for the full ablation matrix.")
    return "\n".join(lines)


def build_comment(rows: list[dict]) -> str:
    import datetime as dt
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    completed = [r["variant"] for r in rows]
    pending = [v for v in ALL_STAGES if v not in completed]
    plot_url = f"https://github.com/{GH_REPO}/blob/feat/pokemon-planner-prompt/docs/experiments/gemma/plots/pr31_ablation_26b.png?raw=true"
    body = [
        "## PR #31 ablation — comparative retrospective (rolling)",
        "",
        f"_Auto-updated: **{ts}** · {len(completed)}/{len(ALL_STAGES)} stages complete_",
        "",
        f"**Status:** completed → `{', '.join(completed) or 'none'}`; pending → `{', '.join(pending) or 'none'}`",
        "",
        "### Results table",
        "",
        results_table(rows),
        "",
        verdict(rows),
        "",
        "### Per-stage details",
        "",
    ]
    for v in ALL_STAGES:
        body.append(stage_section(v, rows))
    body += [
        "",
        "### Plot",
        "",
        f"![PR #31 ablation progress]({plot_url})",
        "",
        "_(File: `experiments/pr31_ablation_26b/progress.png` — re-rendered after every appended run.)_",
    ]
    return "\n".join(body)


def upsert_comment(body_md: str) -> int:
    body_path = Path("/tmp/pr31_retro_rolling.md")
    body_path.write_text(body_md)
    if COMMENT_ID_FILE.exists():
        cid = COMMENT_ID_FILE.read_text().strip()
        try:
            subprocess.check_output(
                ["gh", "api", f"repos/{GH_REPO}/issues/comments/{cid}",
                 "-X", "PATCH", "-F", f"body=@{body_path}", "--jq", ".id"],
                text=True,
            )
            print(f"PATCHed comment {cid}")
            return int(cid)
        except subprocess.CalledProcessError:
            print(f"PATCH failed for {cid}; will POST a fresh comment")
    out = subprocess.check_output(
        ["gh", "api", f"repos/{GH_REPO}/issues/{PR_NUM}/comments",
         "-X", "POST", "-F", f"body=@{body_path}", "--jq", ".id"],
        text=True,
    ).strip()
    COMMENT_ID_FILE.write_text(out)
    print(f"POSTed new rolling comment {out}")
    return int(out)


def update_pr_body(rows: list[dict]) -> None:
    body = subprocess.check_output(
        ["gh", "api", f"repos/{GH_REPO}/pulls/{PR_NUM}", "--jq", ".body"],
        text=True,
    )
    table = results_table(rows)
    block = f"<!-- PR31_RERUN_TABLE_START -->\n\n{table}\n\n<!-- PR31_RERUN_TABLE_END -->"
    if "<!-- PR31_RERUN_TABLE_START -->" in body:
        body = re.sub(
            r"<!-- PR31_RERUN_TABLE_START -->.*?<!-- PR31_RERUN_TABLE_END -->",
            block,
            body,
            count=1,
            flags=re.DOTALL,
        )
    else:
        anchor = body.find("### Rerun results")
        if anchor != -1:
            body = body[:anchor] + f"### Rerun results\n\n{block}\n\n" + body[anchor + len("### Rerun results"):].lstrip()
    body_path = Path("/tmp/pr31_body_autoupdate.md")
    body_path.write_text(body)
    subprocess.run(
        ["gh", "api", f"repos/{GH_REPO}/pulls/{PR_NUM}", "-X", "PATCH",
         "-F", f"body=@{body_path}"],
        check=True, stdout=subprocess.DEVNULL,
    )
    print(f"PATCHed PR #{PR_NUM} body")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=False, help="(unused; kept for compatibility)")
    ap.add_argument("--game-logs", required=False, help="(unused; kept for compatibility)")
    a = ap.parse_args()
    rows = load_rows()
    if not rows:
        sys.exit("results.jsonl is empty — append before running this")
    md = build_comment(rows)
    upsert_comment(md)
    update_pr_body(rows)
