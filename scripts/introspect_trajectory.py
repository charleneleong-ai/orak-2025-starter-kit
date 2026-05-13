"""Introspect a single pokemon run dir to surface where the agent got stuck,
what it was thinking at the plateau, and how its critique/sub-goal evolved.

Reads:
  - game_states.jsonl   (per-step state: score, map, position)
  - logs/raw_requests.jsonl  (per-step LLM prompt + response)

Writes:
  - a markdown report to stdout

Example
-------
    python scripts/introspect_trajectory.py \\
        --run-dir /tmp/orak-planner-prompt/pokemon_red/pr_stage_h_qwen35_a3b_pokemon_iter1_20260513T130435Z
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(pretty_exceptions_enable=False)

MAP_RE = re.compile(r"Map Name: ([\w-]+)")
POS_RE = re.compile(r"Your position \(x, y\): \(([\d-]+),\s*([\d-]+)\)")
SUBGOAL_RE = re.compile(r"\[Current sub-goal.*?\]\s*(.*?)(?=\[|$)", re.DOTALL)
CRITIQUE_RE = re.compile(r"\[Recent critique\](.*?)(?=\*Critique:|$)", re.DOTALL)
ACTION_RE = re.compile(r'"action":\s*\[([^\]]*)\]')


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_obs(obs_str: str) -> dict:
    m_map = MAP_RE.search(obs_str)
    m_pos = POS_RE.search(obs_str)
    badges_match = re.search(r"\[Badge List\]\s*(.*?)(?=\[|$)", obs_str, re.DOTALL)
    party_match = re.search(r"\[Current Party\]\s*(.*?)(?=\[|$)", obs_str, re.DOTALL)
    return {
        "map": m_map.group(1) if m_map else "?",
        "pos": (int(m_pos.group(1)), int(m_pos.group(2))) if m_pos else None,
        "badges": (badges_match.group(1).strip()[:60] if badges_match else ""),
        "party": (party_match.group(1).strip()[:80] if party_match else ""),
    }


def extract_prompt_pieces(prompt: str) -> dict:
    sg = SUBGOAL_RE.search(prompt)
    cr = CRITIQUE_RE.search(prompt)
    return {
        "subgoal": (sg.group(1).strip()[:400] if sg else ""),
        "critique": (cr.group(1).strip()[:600] if cr else ""),
    }


def first_line(s: str, n: int = 180) -> str:
    return s.strip().split("\n", 1)[0][:n]


@app.command()
def main(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="Path to a single run dir")],
    plateau_window: Annotated[int, typer.Option(help="N consecutive steps with no score change = plateau")] = 30,
    show_responses: Annotated[bool, typer.Option(help="Dump full LLM responses at plateau (verbose)")] = False,
):
    states = load_jsonl(run_dir / "game_states.jsonl")
    requests = load_jsonl(run_dir / "logs" / "raw_requests.jsonl")
    summary_p = run_dir / "evaluation_summary.json"
    summary = json.loads(summary_p.read_text()) if summary_p.exists() else {}

    if not states:
        print(f"# {run_dir.name}\n\nNo game_states.jsonl found.")
        return

    final_score = (summary.get("episodes") or [{}])[0].get("final_score", "?")

    # Build timeline of (step, score, map, pos)
    timeline = []
    last_score = -1
    score_bank_events = []
    for i, s in enumerate(states):
        obs = s.get("obs", {})
        obs_str = obs.get("obs_str", "")
        parsed = parse_obs(obs_str)
        score = s.get("info", {}).get("score") or s.get("score") or last_score
        # Some envs report score inside obs.info; fall back to evaluator
        if isinstance(score, (int, float)) and score != last_score and score >= 0:
            score_bank_events.append({"step": i, "from": last_score, "to": score, "map": parsed["map"]})
            last_score = score
        timeline.append({"step": i, "score": last_score, **parsed})

    # Match requests by step (raw_requests has explicit step field)
    req_by_step = {r.get("step"): r for r in requests if "step" in r}

    # Find plateau windows (windows of plateau_window+ steps with no score change)
    plateaus = []
    cur_start = 0
    cur_score = timeline[0]["score"]
    for i, t in enumerate(timeline + [{"score": "END"}]):
        if t["score"] != cur_score:
            if i - cur_start >= plateau_window:
                plateaus.append({"start": cur_start, "end": i - 1, "score": cur_score, "length": i - cur_start})
            cur_start = i
            cur_score = t["score"]

    # Final plateau (if run ended stuck)
    if len(timeline) - cur_start >= plateau_window:
        plateaus.append({"start": cur_start, "end": len(timeline) - 1, "score": cur_score, "length": len(timeline) - cur_start})

    # ── Report ──
    print(f"# Trajectory: {run_dir.name}")
    print()
    print(f"- **final_score:** {final_score} (raw); **eval %:** {(final_score / 7.0) * 100 if isinstance(final_score, (int, float)) else '?'}")
    print(f"- **steps recorded:** {len(timeline)}")
    print(f"- **total tokens:** {summary.get('total_tokens', '?')}")
    print()

    print("## Score-bank events")
    if not score_bank_events:
        print("_No score increments observed in this trace._\n")
    else:
        print("| step | prev | new | map |")
        print("|------|------|-----|-----|")
        for ev in score_bank_events:
            print(f"| {ev['step']} | {ev['from']} | {ev['to']} | `{ev['map']}` |")
        print()

    print(f"## Plateaus (≥{plateau_window} consecutive steps with no score change)")
    if not plateaus:
        print(f"_No plateaus of {plateau_window}+ steps found._\n")
    else:
        print()
        for p in plateaus:
            print(f"### Plateau at score={p['score']} — steps {p['start']}–{p['end']} ({p['length']} steps)")
            print()
            # Map distribution during plateau
            map_counts: dict[str, int] = {}
            for t in timeline[p["start"]:p["end"] + 1]:
                map_counts[t["map"]] = map_counts.get(t["map"], 0) + 1
            top_maps = sorted(map_counts.items(), key=lambda kv: -kv[1])[:5]
            print(f"**Maps visited:** {', '.join(f'`{m}`×{c}' for m, c in top_maps)}")
            print()
            # Sample prompts at plateau start, middle, end
            samples_at = [p["start"], (p["start"] + p["end"]) // 2, p["end"]]
            for label, step in zip(("start", "mid", "end"), samples_at):
                req = req_by_step.get(step)
                if req is None:
                    continue
                prompt = req.get("prompt", "")
                response = req.get("response", "") or req.get("completion", "")
                parts = extract_prompt_pieces(prompt)
                print(f"#### step {step} ({label})")
                print(f"- **map:** `{timeline[step]['map']}` pos={timeline[step]['pos']}")
                print(f"- **subgoal:** {first_line(parts['subgoal'], 200)}")
                if parts["critique"]:
                    print(f"- **critique excerpt:** {first_line(parts['critique'], 200)}")
                resp_first = first_line(str(response), 200)
                if resp_first:
                    print(f"- **agent reply (head):** {resp_first}")
                if show_responses and response:
                    print()
                    print("```")
                    print(str(response)[:1500])
                    print("```")
                print()

    # Final 20-step burst analysis (interesting for iter 2 which banked late)
    if len(timeline) >= 30:
        burst_start = max(0, len(timeline) - 30)
        last_score_at_burst_start = timeline[burst_start]["score"]
        final_score_t = timeline[-1]["score"]
        if final_score_t > last_score_at_burst_start:
            print(f"## Late burst — last 30 steps banked {final_score_t - last_score_at_burst_start} milestones")
            print(f"score went **{last_score_at_burst_start} → {final_score_t}** in steps {burst_start}–{len(timeline) - 1}")
            print()
            # show subgoals + critique at the start of the burst
            for step in [burst_start, burst_start + 10, len(timeline) - 5]:
                req = req_by_step.get(step)
                if req is None:
                    continue
                parts = extract_prompt_pieces(req.get("prompt", ""))
                print(f"- step {step} subgoal: {first_line(parts['subgoal'], 200)}")
            print()


if __name__ == "__main__":
    app()
