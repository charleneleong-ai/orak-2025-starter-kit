"""Maintain a single rolling comparative retrospective comment on PR #31.

Each invocation rebuilds the full comparative report from results.jsonl,
PATCHes the existing comment if one is tracked (else POSTs new + saves ID),
and refreshes the PR body's results table.

Layout: plot first, scannable overview table, single-line verdict,
collapsed per-stage drilldowns.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
PR_NUM = 31
GH_REPO = "charleneleong-ai/orak-2025-starter-kit"
PLOT_URL = f"https://github.com/{GH_REPO}/blob/feat/pokemon-planner-prompt/docs/experiments/gemma/plots/pr31_ablation_26b.png?raw=true"

ORDER = ["stage_a_26b", "stage_c_26b", "stage_b_26b", "stage_d_26b", "stage_d_plus_26b"]
LABELS = {
    "stage_a_26b":      ("Stage A",   "model only",            "vmem OFF · planner OFF"),
    "stage_c_26b":      ("Stage C",   "+ vmem",                "vmem ON · planner OFF"),
    "stage_b_26b":      ("Stage B",   "+ planner",             "vmem OFF · planner ON"),
    "stage_d_26b":      ("Stage D",   "both (300st)",          "vmem ON · planner ON"),
    "stage_d_plus_26b": ("Stage D++", "both (600st + grace)",  "vmem ON · planner ON · 600st · grace=10"),
}


def load_rows() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def find_run_dir(variant: str) -> Path | None:
    base = Path("/tmp/orak-planner-prompt/game_logs/pokemon_red")
    pat = "pr31_rerun_pokemon_2026*" if variant == "stage_d_26b" else f"pr31_rerun_pokemon_{variant}_*"
    cands = sorted(base.glob(pat), reverse=True)
    return cands[0] if cands else None


def stats(run_dir: Path | None) -> dict:
    base = {"n_inferences": 0, "progression": [], "actions": {},
            "stuck_firings": 0, "exhaust_firings": 0, "stuck_pct": 0.0}
    if run_dir is None:
        return base
    eval_log = run_dir / "evaluation.log"
    raw = run_dir / "logs" / "raw_requests.jsonl"
    if not eval_log.exists():
        return base
    text = eval_log.read_text(errors="ignore")
    n = text.count("is_finished:")
    progression, seen, step = [], set(), 0
    for line in text.splitlines():
        if "is_finished:" not in line:
            continue
        step += 1
        m = re.search(r"Score: (\d+),", line)
        if not m:
            continue
        s = int(m.group(1))
        if s and s not in seen:
            seen.add(s)
            progression.append((s, step))
    actions = Counter(m.group(1) for m in re.finditer(r"use_tool\(([a-z_]+)", text))
    stuck = exhaust = 0
    if raw.exists():
        rt = raw.read_text(errors="ignore")
        stuck = rt.count("Stuck Detector")
        exhaust = rt.count("Exhaust Interactables")
    return {
        "n_inferences": n,
        "progression": progression,
        "actions": dict(actions.most_common(6)),
        "stuck_firings": stuck,
        "exhaust_firings": exhaust,
        "stuck_pct": (stuck / n * 100) if n else 0.0,
    }


def overview_table(rows_by_v: dict, stats_by_v: dict) -> str:
    out = ["| Stage | Switches | Score | % | Steps | Stuck-fire | Status |",
           "|---|---|---|---|---|---|---|"]
    for v in ORDER:
        name, tag, switches = LABELS[v]
        r = rows_by_v.get(v)
        if r is None:
            out.append(f"| **{name}** _{tag}_ | {switches} | — | — | — | — | _pending_ |")
        else:
            sf = stats_by_v.get(v, {})
            stuck = f"{sf.get('stuck_pct', 0):.0f}%" if sf.get("n_inferences") else "—"
            out.append(
                f"| **{name}** _{tag}_ | {switches} | {r['game_score']:.0f}/7 "
                f"| **{r['evaluation_score']:.2f}%** | {r['steps']} | {stuck} | {r['status']} |"
            )
    return "\n".join(out)


def md_progression(s: dict) -> str:
    if not s.get("progression"):
        return "_no score gain logged_"
    parts = [f"`{sc}` @ step {step}" for sc, step in s["progression"]]
    return " · ".join(parts)


def md_actions(s: dict) -> str:
    if not s.get("actions"):
        return "_no actions logged_"
    parts = [f"`{a}`={c}" for a, c in s["actions"].items()]
    return " · ".join(parts)


def stage_drilldown(v: str, rows_by_v: dict, stats_by_v: dict) -> str:
    name, tag, switches = LABELS[v]
    r = rows_by_v.get(v)
    s = stats_by_v.get(v, {})
    head = f"#### {name} — _{tag}_"
    if r is None:
        return f"{head}\n\n_pending_\n"
    score_line = f"**{r['game_score']:.0f}/7 ({r['evaluation_score']:.2f}%)** · {r['steps']} steps · stuck-fire {s.get('stuck_pct', 0):.0f}% · exhaust-fire {s.get('exhaust_firings', 0)}"
    return (
        f"{head}\n\n"
        f"{score_line}\n\n"
        f"**Score progression:** {md_progression(s)}\n\n"
        f"**Top actions:** {md_actions(s)}\n"
    )


def verdict(rows_by_v: dict) -> str:
    score = lambda v: rows_by_v[v]["evaluation_score"] if v in rows_by_v else None
    a, c, b, d, dpp = (score(v) for v in ORDER)
    completed = [v for v in ORDER if v in rows_by_v]
    if len(completed) < 2:
        if d is not None and a is not None:
            return f"> _**Headline so far:** Stage D ({d:.2f}%) is **{d - a:+.2f} pp** above Stage A ({a:.2f}%) — vmem+planner together unlock the score 2→4 jump. Awaiting C/B to attribute the lift._"
        if d is not None:
            return f"> _**Headline so far:** Stage D rerun lands at **{d:.2f}%** — {d - 14.29:+.2f} pp above PR #31's prior 14.29% plateau. Awaiting Stage A baseline._"
        return "> _Awaiting first results._"

    lines = ["> **Headline so far:**"]
    if d is not None and a is not None:
        lines.append(f"> - Stage D vs Stage A: **{d - a:+.2f} pp** — full stack adds this much over model-only.")
    if c is not None and a is not None:
        lines.append(f"> - vmem-only lift (C − A): **{c - a:+.2f} pp**")
    if b is not None and a is not None:
        lines.append(f"> - planner-only lift (B − A): **{b - a:+.2f} pp**")
    if c is not None and b is not None:
        if b > c + 5:
            lines.append(f"> - **Planner is the load-bearing lever** ({b:.2f}% vs vmem's {c:.2f}%).")
        elif c > b + 5:
            lines.append(f"> - **Vmem is the load-bearing lever** ({c:.2f}% vs planner's {b:.2f}%).")
        else:
            lines.append(f"> - **Planner ≈ vmem** in isolation ({b:.2f}% vs {c:.2f}%) — the levers may compose multiplicatively.")
    if dpp is not None and d is not None:
        lines.append(f"> - Stage D++ vs Stage D (600st + grace): **{dpp - d:+.2f} pp**.")
    return "\n".join(lines)


def build_comment(rows: list[dict]) -> str:
    rows_by_v = {r["variant"]: r for r in rows}
    stats_by_v = {v: stats(find_run_dir(v)) for v in ORDER if v in rows_by_v}
    completed = [v for v in ORDER if v in rows_by_v]
    pending = [v for v in ORDER if v not in rows_by_v]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    body = [
        "## PR #31 ablation — comparative retrospective",
        "",
        f"![Progress]({PLOT_URL})",
        "",
        f"_Auto-updated **{ts}** · **{len(completed)}/{len(ORDER)}** stages complete_  ",
        f"_Pending: {', '.join(LABELS[v][0] for v in pending) or 'none'}_",
        "",
        "### Overview",
        "",
        overview_table(rows_by_v, stats_by_v),
        "",
        verdict(rows_by_v),
        "",
        "<details>",
        "<summary><b>Per-stage drilldown</b> — score progression + action mix</summary>",
        "",
    ]
    for v in ORDER:
        body.append(stage_drilldown(v, rows_by_v, stats_by_v))
    body += [
        "</details>",
        "",
        f"_Plot regenerated by `experiments/pr31_ablation_26b/render_progress.py` after each completed run; rolling comment maintained by `post_retro.py`._",
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
            print(f"PATCH failed for {cid}; will POST fresh")
    out = subprocess.check_output(
        ["gh", "api", f"repos/{GH_REPO}/issues/{PR_NUM}/comments",
         "-X", "POST", "-F", f"body=@{body_path}", "--jq", ".id"],
        text=True,
    ).strip()
    COMMENT_ID_FILE.write_text(out)
    print(f"POSTed comment {out}")
    return int(out)


def update_pr_body(rows: list[dict]) -> None:
    rows_by_v = {r["variant"]: r for r in rows}
    stats_by_v = {v: stats(find_run_dir(v)) for v in ORDER if v in rows_by_v}
    table = overview_table(rows_by_v, stats_by_v)
    body = subprocess.check_output(
        ["gh", "api", f"repos/{GH_REPO}/pulls/{PR_NUM}", "--jq", ".body"], text=True,
    )
    block = f"<!-- PR31_RERUN_TABLE_START -->\n\n{table}\n\n<!-- PR31_RERUN_TABLE_END -->"
    if "<!-- PR31_RERUN_TABLE_START -->" in body:
        body = re.sub(
            r"<!-- PR31_RERUN_TABLE_START -->.*?<!-- PR31_RERUN_TABLE_END -->",
            block, body, count=1, flags=re.DOTALL,
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
    ap.add_argument("--variant", required=False)
    ap.add_argument("--game-logs", required=False)
    ap.parse_args()
    rows = load_rows()
    if not rows:
        sys.exit("results.jsonl is empty")
    md = build_comment(rows)
    upsert_comment(md)
    update_pr_body(rows)
