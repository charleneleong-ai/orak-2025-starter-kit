"""
Autoresearch loop for MACLA parameter optimisation.

Iteratively runs experiments, analyzes trajectories for failure patterns,
and proposes targeted changes (prompts, params, success detection) based
on what's actually blocking improvement.

Usage:
    # Run optimisation loop (max 5 iterations)
    python experiments/autoresearch.py run --max-iterations 5

    # Analyze a past run
    python experiments/autoresearch.py analyze --run-id 20260422_221353

    # Log results from a completed run
    python experiments/autoresearch.py log-run --run-id 20260422_213143
"""
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import sys

import typer
import yaml

# Allow running as both `python experiments/autoresearch.py` and `python -m experiments.autoresearch`
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.experiment_progress import (
    ALL_GAMES,
    GAME_LOG_DIR,
    extract_run_results,
    load_results,
    log_experiment,
    normalize_eval_score,
    plot_progress,
)

# Post-iter retrospective hook (autoresearch >= 0.8.0). Detectors run after each
# iter writes its results.jsonl row and surface failure-mode findings into the
# next iter's description (warn) or stop the sweep (block). See PR #28 v6 audit
# for the orak-side findings that motivated the package detectors.
from autoresearch.retrospective import (
    BUILTIN_DETECTORS,
    Finding,
    attach_findings_to_row,
    audit_iter,
    filter_by_severity,
    format_markdown,
)
# Sweep-loop helpers extracted into the package in v0.11.0 (autoresearch#20 PR 1).
# These wrap the package APIs to preserve orak's existing call-site signatures.
from autoresearch import (
    clear_sidecar as _ar_clear_sidecar,
    relabel_last_as_early_kill as _ar_relabel_last_as_early_kill,
    write_sidecar as _ar_write_sidecar,
)

ROOT = Path(__file__).parent.parent
CONFIGS_DIR = ROOT / "configs"
AGENTS_DIR = ROOT / "agents"

# Detectors relevant to orak's game-agent sweep loop. Skipped from the
# package's BUILTIN_DETECTORS:
#   - bucketed_failure: orak doesn't emit per-row JSONL rubric dumps
#   - gradient_collapse: orak doesn't run an RL training loop (LLM is frozen)
ORAK_RETROSPECTIVE_DETECTORS = [
    BUILTIN_DETECTORS["silent_kill"],
    BUILTIN_DETECTORS["triage_threshold_mismatch"],
    BUILTIN_DETECTORS["eval_score_plateau"],
]
# Per-game tuning of detector thresholds. Pokemon's first scoring event lands
# at step 100-150 so the global plateau-at-80 default would false-positive
# every iter; bump it to 200 for pokemon. Other games keep the defaults.
ORAK_RETROSPECTIVE_KWARGS = {
    "triage_threshold_mismatch": {"min_first_score_step": 150},
}

# Marker to track auto-appended prompt hints (avoid duplication)
AUTORESEARCH_MARKER = "# [autoresearch]"


# ── Trajectory Analysis ─────────────────────────────────────────────


def load_game_states(run_id: str, game: str) -> list[dict]:
    """Load all entries from game_states.jsonl for a run."""
    path = GAME_LOG_DIR / game / run_id / "game_states.jsonl"
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().strip().split("\n"):
        if line:
            entries.append(json.loads(line))
    return entries


def analyze_trajectory(run_id: str, game: str) -> dict:
    """Analyze game_states.jsonl for failure patterns.

    Returns a dict with signals that propose_changes() maps to fixes.
    """
    entries = load_game_states(run_id, game)
    if not entries:
        # Game was triage-killed (or crashed) before any state got logged.
        # Return a complete stub so downstream consumers (the run-loop print,
        # propose_changes, the analyze CLI) don't KeyError on missing keys.
        # Hit on iter 8 of PR #20 sweep (wandb run
        # https://wandb.ai/chaleong/orak-pokemon-red/runs/20260427_183055_orak-pokemon-red)
        # when pokemon_red was triage-killed at 0% with no game_states.jsonl
        # written, leading to KeyError: 'episodes'.
        return {
            "game": game,
            "total_steps": 0,
            "episodes": 0,
            "max_eval": 0,
            "episode_scores": [],
            "action_distribution": {},
            "top_action": "",
            "top_action_pct": 0,
            "repeated_actions": False,
            "score_plateau": False,
            "error": "no data",
        }

    actions = [e.get("action", "") for e in entries]
    action_counts = Counter(actions)
    total = len(actions)
    top_action, top_count = action_counts.most_common(1)[0] if action_counts else ("", 0)

    # Episode boundaries (iteration resets)
    episode_scores = []
    current_max = 0
    for i, e in enumerate(entries):
        current_max = max(current_max, e.get("evaluation_score", 0))
        if e.get("result", {}).get("is_finished") or (i > 0 and e["iteration"] <= entries[i - 1]["iteration"]):
            episode_scores.append(current_max)
            current_max = 0
    if current_max > 0:
        episode_scores.append(current_max)

    max_eval = max(e.get("evaluation_score", 0) for e in entries)

    # Score plateau: last 3+ episodes at same score
    plateau = len(episode_scores) >= 3 and len(set(round(s, 1) for s in episode_scores[-3:])) == 1

    analysis = {
        "game": game,
        "total_steps": total,
        "episodes": len(episode_scores),
        "max_eval": max_eval,
        "episode_scores": episode_scores,
        "action_distribution": dict(action_counts.most_common(5)),
        "top_action": top_action,
        "top_action_pct": top_count / total if total else 0,
        "repeated_actions": top_count / total > 0.5 if total else False,
        "score_plateau": plateau,
    }

    # Game-specific analysis
    if game == "super_mario":
        _analyze_mario(entries, analysis)
    elif game == "pokemon_red":
        _analyze_pokemon(entries, analysis)
    elif game == "twenty_fourty_eight":
        _analyze_2048(entries, analysis)

    return analysis


def _analyze_mario(entries: list[dict], analysis: dict):
    """Mario-specific: find death clustering by x_pos."""
    death_xpos = []
    for e in entries:
        if e.get("result", {}).get("is_finished"):
            gi = e.get("obs", {}).get("game_info", {})
            x = gi.get("x_pos", 0)
            if isinstance(x, str):
                try:
                    x = int(x)
                except ValueError:
                    x = 0
            death_xpos.append(x)

    analysis["death_positions"] = death_xpos

    if death_xpos:
        # Cluster deaths into 100-unit bins
        bins = Counter(x // 100 * 100 for x in death_xpos)
        most_common_bin, count = bins.most_common(1)[0]
        if count >= 3:  # At least 3 deaths in same zone
            analysis["failure_zone"] = f"x={most_common_bin}-{most_common_bin + 100}"
            analysis["failure_zone_deaths"] = count
            analysis["failure_zone_total_deaths"] = len(death_xpos)


def _analyze_pokemon(entries: list[dict], analysis: dict):
    """Pokemon-specific: check map diversity and flag progress."""
    maps = []
    for e in entries:
        gi = e.get("obs", {}).get("game_info", {})
        m = gi.get("map_name", "")
        if m:
            maps.append(m)

    unique_maps = set(maps)
    analysis["maps_visited"] = list(unique_maps)
    analysis["map_count"] = len(unique_maps)
    analysis["map_stuck"] = len(unique_maps) <= 2 and len(maps) > 50

    # Check if score ever changes
    scores = [e.get("game_score", 0) for e in entries]
    analysis["score_changes"] = len(set(scores))
    analysis["max_flags"] = max(scores) if scores else 0


def _analyze_2048(entries: list[dict], analysis: dict):
    """2048-specific: check action balance and max tile."""
    max_tiles = []
    for e in entries:
        gi = e.get("obs", {}).get("game_info", {})
        mt = gi.get("max_tile", 0)
        if isinstance(mt, str):
            try:
                mt = int(mt)
            except ValueError:
                mt = 0
        max_tiles.append(mt)

    analysis["max_tile"] = max(max_tiles) if max_tiles else 0

    # Action balance: check if one direction dominates
    direction_counts = {d: 0 for d in ["up", "down", "left", "right"]}
    for a in analysis["action_distribution"]:
        if a in direction_counts:
            direction_counts[a] = analysis["action_distribution"][a]
    total_dir = sum(direction_counts.values())
    if total_dir > 0:
        max_dir_pct = max(direction_counts.values()) / total_dir
        analysis["action_imbalance"] = max_dir_pct > 0.4
        analysis["direction_balance"] = {k: round(v / total_dir, 2) for k, v in direction_counts.items()}
    else:
        analysis["action_imbalance"] = False


# ── Change Proposal ──────────────────────────────────────────────────


def propose_changes(analysis: dict) -> list[dict]:
    """Map failure patterns to structural MACLA changes (not prompt appends).

    Targets: refinement thresholds, theta/decay, temperature, success detection.
    """
    changes = []
    game = analysis["game"]
    episodes = analysis.get("episodes", 0)
    total_steps = analysis.get("total_steps", 0)

    # ── Refinement thresholds ──────────────────────────────────────
    # If procedures are learned but never refined (not enough data),
    # lower n_min_s/n_min_f so refinement kicks in faster
    if episodes >= 3 and analysis.get("score_plateau"):
        changes.append({
            "type": "param",
            "target": "macla_n_min_s",
            "action": "decrease",
            "step": 1,
            "min": 1,
            "reason": f"Score plateau {episodes} eps — refine procedures faster",
        })
        changes.append({
            "type": "param",
            "target": "macla_n_min_f",
            "action": "decrease",
            "step": 1,
            "min": 1,
            "reason": f"Score plateau {episodes} eps — need fewer failures to refine",
        })

    # ── Theta decay ────────────────────────────────────────────────
    # If procedures exist but score doesn't improve, decay faster
    # so agent explores new procedure/fallback combinations
    if episodes >= 5 and analysis.get("score_plateau"):
        changes.append({
            "type": "param",
            "target": "macla_theta_decay",
            "action": "increase",
            "step": 0.001,
            "max": 0.01,
            "reason": f"Stagnation — decay theta faster for exploration",
        })

    # ── Temperature ────────────────────────────────────────────────
    # Action repetition signals LLM is too deterministic
    if analysis.get("repeated_actions") and analysis["top_action_pct"] > 0.6:
        changes.append({
            "type": "param",
            "target": "temperature",
            "action": "increase",
            "step": 0.1,
            "max": 1.5,
            "reason": f"{analysis['top_action']} at {analysis['top_action_pct']:.0%} — increase diversity",
        })

    # ── Game-specific structural changes ───────────────────────────

    if game == "super_mario":
        # Death clustering → lower theta so bayesian avoids fatal zones
        if analysis.get("failure_zone"):
            deaths = analysis["failure_zone_deaths"]
            total = analysis.get("failure_zone_total_deaths", deaths)
            if deaths / max(total, 1) > 0.3:
                changes.append({
                    "type": "param",
                    "target": "macla_max_theta",
                    "action": "decrease",
                    "step": 0.03,
                    "min": 0.10,
                    "reason": f"Deaths at {analysis['failure_zone']} ({deaths}/{total}) — tighten procedure selection",
                })

    elif game == "twenty_fourty_eight":
        # Action imbalance → increase theta_base to force more fallback diversity
        if analysis.get("action_imbalance"):
            changes.append({
                "type": "param",
                "target": "macla_theta_base",
                "action": "increase",
                "step": 0.05,
                "max": 0.45,
                "reason": f"Action imbalance {analysis.get('direction_balance', {})} — more fallback for diversity",
            })
        # Low max tile → faster theta decay to let procedures develop
        if analysis.get("max_tile", 0) < 128 and total_steps > 50:
            changes.append({
                "type": "param",
                "target": "macla_theta_decay",
                "action": "increase",
                "step": 0.001,
                "max": 0.008,
                "reason": f"Max tile {analysis.get('max_tile', 0)} — faster procedure activation",
            })

    elif game == "pokemon_red":
        # Map stagnation → lower warmup so procedures activate sooner
        if analysis.get("map_stuck"):
            changes.append({
                "type": "param",
                "target": "macla_warmup_steps",
                "action": "decrease",
                "step": 3,
                "min": 0,
                "reason": f"Stuck on {analysis.get('map_count', 0)} maps — reduce warmup",
            })
        # No flags → increase theta base (more LLM fallback, less bad procedures)
        if analysis.get("max_flags", 0) <= 1 and total_steps > 100:
            changes.append({
                "type": "param",
                "target": "macla_theta_base",
                "action": "increase",
                "step": 0.05,
                "max": 0.50,
                "reason": f"Only {analysis.get('max_flags', 0)} flags — prefer LLM fallback",
            })

    return changes


# ── Apply Changes ────────────────────────────────────────────────────


def apply_changes(game: str, changes: list[dict], config_type: str = "unified_macla") -> list[str]:
    """Apply proposed changes to game adapter and/or YAML config."""
    applied = []
    for change in changes:
        if change["type"] == "prompt":
            success = _apply_prompt_change(game, change)
            if success:
                applied.append(f"prompt: {change['reason']}")
        elif change["type"] == "param":
            success = _apply_param_change(game, change, config_type)
            if success:
                applied.append(f"param: {change['reason']}")
    return applied


def _apply_prompt_change(game: str, change: dict) -> bool:
    """Append text to a game adapter constant (DEFAULT_GOAL, etc)."""
    adapter_path = AGENTS_DIR / game / "game_adapter.py"
    if not adapter_path.exists():
        print(f"  [SKIP] No adapter at {adapter_path}")
        return False

    content = adapter_path.read_text()
    target = change["target"]  # e.g. "DEFAULT_GOAL"
    text = change["text"]

    # Check if this exact hint was already appended
    if text.strip() in content:
        print(f"  [SKIP] Already applied: {change['reason']}")
        return False

    # Find the target constant and append before closing quote
    # Pattern: DEFAULT_GOAL = "...existing text..."
    pattern = rf'({target}\s*=\s*")(.*?)(")'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        # Try single quotes
        pattern = rf"({target}\s*=\s*')(.*?)(')"
        match = re.search(pattern, content, re.DOTALL)

    if match:
        new_content = content[:match.end(2)] + text + content[match.end(2):]
        adapter_path.write_text(new_content)
        print(f"  [APPLIED] {target} += '{text[:60]}...'")
        return True

    print(f"  [SKIP] Could not find {target} in {adapter_path}")
    return False


def _apply_param_change(game: str, change: dict, config_type: str) -> bool:
    """Update a YAML config parameter."""
    config = read_yaml_config(game, config_type)
    target = change["target"]
    current = config.get(target)

    if change["action"] == "increase":
        new_val = (current or 0) + change.get("step", 0.05)
        if "max" in change:
            new_val = min(new_val, change["max"])
    elif change["action"] == "decrease":
        new_val = (current or 0) - change.get("step", 0.05)
        if "min" in change:
            new_val = max(new_val, change["min"])
    else:
        new_val = change.get("value", current)

    if new_val == current:
        print(f"  [SKIP] {target} already at {current}")
        return False

    config[target] = round(new_val, 3) if isinstance(new_val, float) else new_val
    write_yaml_config(game, config, config_type)
    print(f"  [APPLIED] {target}: {current} → {new_val}")
    return True


# ── Per-game parameter search bounds
PARAM_BOUNDS = {
    "super_mario": {
        "macla_theta_base": (0.10, 0.30),
        "macla_max_theta": (0.15, 0.35),
        "macla_min_theta": (0.03, 0.10),
        "macla_warmup_steps": (0, 10),
    },
    "twenty_fourty_eight": {
        # 2048 has the worst procedure-reuse profile of the three games:
        # boards almost never repeat, so memorised "in board X do Y" procedures
        # rarely fire. Optimal play depends on global structure (corner-anchor,
        # snake pattern) which doesn't fit MACLA's context-matching abstraction
        # cleanly. Lever: push max_theta high so the agent leans on LLM
        # reasoning rather than brittle procedure fallback. With checkpoint
        # carry-over (this PR), procedures that DO transfer accumulate over
        # iters and θ can drop later.
        # Bounds were shifted up once already after PR #20 sweep (the 6.02%
        # best at iter 6 hit 3 of 4 OLD ceilings); pushing max_theta further
        # up gives the search room to find LLM-heavy variants.
        "macla_theta_base": (0.20, 0.50),
        "macla_max_theta":  (0.40, 0.75),
        "macla_min_theta":  (0.08, 0.25),
        "macla_warmup_steps": (5, 15),
    },
    "pokemon_red": {
        "macla_theta_base": (0.25, 0.45),
        "macla_max_theta": (0.35, 0.55),
        "macla_min_theta": (0.10, 0.25),
        # Lowered floor 5 -> 0: the iter 3 breakthrough on PR #20 sweep
        # (max_eval=14.29%, wandb run
        # https://wandb.ai/chaleong/orak-pokemon-red/runs/20260427_172124_orak-pokemon-red)
        # hit warmup=5 exactly at the old bounds floor, so propose_next_params
        # couldn't explore any lower. Subsequent iters drove warmup UP to 8
        # then 11 and lost the gain. Letting it try warmup={0,1,2,3} preserves
        # the high-theta / low-warmup combination that's actually working.
        "macla_warmup_steps": (0, 15),
    },
}

STEP_SIZES = {
    "macla_theta_base": 0.05,
    "macla_max_theta": 0.05,
    "macla_min_theta": 0.02,
    "macla_warmup_steps": 3,
}

app = typer.Typer(help="MACLA autoresearch parameter optimisation")


def read_yaml_config(game: str, config_type: str = "unified_macla") -> dict:
    """Read current YAML config for a game."""
    path = CONFIGS_DIR / game / "agent" / f"{config_type}.yaml"
    return yaml.safe_load(path.read_text())


def write_yaml_config(game: str, config: dict, config_type: str = "unified_macla"):
    """Write updated YAML config for a game."""
    path = CONFIGS_DIR / game / "agent" / f"{config_type}.yaml"
    # Preserve comments by reading, updating macla_ fields only
    lines = path.read_text().splitlines()
    macla_keys = {k for k in config if k.startswith("macla_")}

    # Remove existing macla_ lines
    new_lines = [l for l in lines if not any(l.strip().startswith(f"{k}:") for k in macla_keys)]

    # Append updated macla_ values
    for k in sorted(macla_keys):
        v = config[k]
        new_lines.append(f"{k}: {v}")

    path.write_text("\n".join(new_lines) + "\n")


def get_best_scores(tag: str = "macla", config_name: str | None = None) -> dict[str, float]:
    """Get current best evaluation_score per game from results."""
    results = load_results(tag=tag, config_name=config_name)
    best = {}
    for r in results:
        game = r["game"]
        score = r["evaluation_score"]
        if r["status"] in ("KEEP", "BASELINE"):
            best[game] = max(best.get(game, 0), score)
    return best


def get_current_params(game: str, config_type: str = "unified_macla") -> dict[str, float]:
    """Extract current macla_ params from YAML config."""
    config = read_yaml_config(game, config_type)
    return {k: v for k, v in config.items() if k.startswith("macla_")}


def propose_next_params(game: str, results: list[dict], config_type: str = "unified_macla") -> dict[str, float]:
    """Propose next theta params for a game based on experiment history.

    Strategy: look at last 2 experiments for this game.
    - If last improved over previous: continue in same direction
    - If last regressed: reverse direction
    - If first run: use current config as-is
    """
    game_results = [r for r in results if r["game"] == game]
    current = get_current_params(game, config_type)
    bounds = PARAM_BOUNDS.get(game, {})

    if len(game_results) < 2:
        return current

    last = game_results[-1]
    prev = game_results[-2]
    improved = last["evaluation_score"] > prev["evaluation_score"]

    new_params = dict(current)
    for param, (lo, hi) in bounds.items():
        step = STEP_SIZES.get(param, 0.05)
        cur_val = current.get(param, (lo + hi) / 2)

        if improved:
            # Continue in same direction (decrease theta = more bayesian)
            new_val = cur_val - step
        else:
            # Try opposite direction (increase theta = more fallback)
            new_val = cur_val + step

        # Clamp to bounds
        if param == "macla_warmup_steps":
            new_val = int(max(lo, min(hi, round(new_val))))
        else:
            new_val = round(max(lo, min(hi, new_val)), 3)

        new_params[param] = new_val

    return new_params


# ── Triage Thresholds ──────────────────────────────────────────────

# Per-game plateau threshold. Pokemon's first scoring event (1st flag) lands
# around step 100-150 in successful runs (`pokemon_check` 2026-04-30 sweep),
# so the global 80 was killing Stage A and D before they had a chance to score
# (PR #28 v6 audit). Other games hit their first score in <20 steps so 80 is
# fine. Default applies to games not explicitly listed.
TRIAGE_SCORE_PLATEAU_STEPS_PER_GAME = {"pokemon_red": 200}
TRIAGE_SCORE_PLATEAU_STEPS_DEFAULT = 80
TRIAGE_SCORE_PLATEAU_STEPS = 80    # Kept as fallback for code paths that don't pass game
TRIAGE_NO_LEARN_EPISODES = 8       # Kill if no episode score improvement for N episodes.
                                   # Bumped 5 -> 8 after iter 4 of PR #20 sweep
                                   # (wandb run https://wandb.ai/chaleong/orak-super-mario/runs/20260427_174648_orak-super-mario):
                                   # super_mario set a new best of 51.87% in episode 1,
                                   # then was killed in episodes 2-6 before MACLA's
                                   # procedure-learning could compound — even though
                                   # the iter was healthy. 5 was too tight; 8 gives
                                   # one-shot bests room to consolidate.
TRIAGE_BASELINE_FACTOR = 0.5       # Kill if max_eval < baseline * factor after 100 steps
TRIAGE_POLL_INTERVAL = 5           # Seconds between game_states.jsonl checks
ITER_TIMEOUT_MIN = 30              # Hard wall-clock cap per iteration; SIGINT subprocess if exceeded


def _find_run_id(games: list[str]) -> str:
    """Find the latest run_id from game_logs by mtime.

    Lexicographic sort would pick e.g. `pokemon_stage_a_direct_*` over a fresh
    timestamp-named dir like `20260503_093804`, since 'p' (0x70) > '2' (0x30).
    Old / abandoned runs would then bleed stale game_states.jsonl entries
    into the new sweep's triage check. Sort by mtime to always pick the
    actually-most-recent dir regardless of naming scheme.
    """
    game_log_dir = GAME_LOG_DIR / games[0]
    if not game_log_dir.exists():
        return ""
    run_dirs = sorted(
        (p for p in game_log_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0].name if run_dirs else ""


def _read_game_states(run_id: str, game: str) -> list[dict]:
    """Read current game_states.jsonl for a running experiment."""
    path = GAME_LOG_DIR / game / run_id / "game_states.jsonl"
    if not path.exists():
        return []
    try:
        return [json.loads(l) for l in path.read_text().strip().split("\n") if l]
    except Exception:
        return []


def _triage_check(
    run_id: str,
    games: list[str],
    baseline_scores: dict[str, float],
) -> str | None:
    """Check if any game triggers a triage kill. Returns kill_reason or None."""
    for game in games:
        entries = _read_game_states(run_id, game)
        if not entries:
            continue

        total = len(entries)
        evals = [e.get("evaluation_score", 0) for e in entries]
        max_eval_raw = max(evals) if evals else 0
        max_eval = normalize_eval_score(game, max_eval_raw, max(e.get("game_score", 0) for e in entries))

        # Don't kill an iter that's already produced a new PR best for this
        # game. Otherwise propose_next_params advances past the kill (which
        # ends with a partial run.py and partial flushed state) and the gain
        # is lost. Hit on iter 4 of PR #20 sweep — super_mario reached 77.83%
        # (vs prior best 61.13%) then got triage-killed by no_learn=8 before
        # the new best could be consolidated.
        if max_eval > baseline_scores.get(game, 0):
            continue

        # Count episodes
        episode_scores = []
        cur_max = 0
        for i, e in enumerate(entries):
            cur_max = max(cur_max, e.get("evaluation_score", 0))
            if i > 0 and e["iteration"] <= entries[i - 1]["iteration"]:
                episode_scores.append(cur_max)
                cur_max = 0
        if cur_max > 0:
            episode_scores.append(cur_max)

        # Triage 1: Score plateau — max eval unchanged for N steps (per-game)
        plateau_steps = TRIAGE_SCORE_PLATEAU_STEPS_PER_GAME.get(game, TRIAGE_SCORE_PLATEAU_STEPS_DEFAULT)
        if total >= plateau_steps:
            recent = evals[-plateau_steps:]
            if max(recent) == min(recent):
                return f"{game}: score plateau ({max_eval:.2f}%) for {plateau_steps} steps"

        # Triage 2: No episode improvement for N episodes
        if len(episode_scores) >= TRIAGE_NO_LEARN_EPISODES:
            last_n = episode_scores[-TRIAGE_NO_LEARN_EPISODES:]
            best_before = max(episode_scores[:-TRIAGE_NO_LEARN_EPISODES]) if len(episode_scores) > TRIAGE_NO_LEARN_EPISODES else 0
            if max(last_n) <= best_before:
                return f"{game}: no improvement for {TRIAGE_NO_LEARN_EPISODES} episodes (best={normalize_eval_score(game, best_before, 0):.2f}%)"

        # Triage 3: Baseline gate — below baseline*factor after 100 steps
        baseline = baseline_scores.get(game, 0)
        if total >= 100 and baseline > 0 and max_eval < baseline * TRIAGE_BASELINE_FACTOR:
            return f"{game}: below baseline gate ({max_eval:.2f}% < {baseline * TRIAGE_BASELINE_FACTOR:.2f}%)"

    return None


def _relabel_last_as_early_kill(
    tag: str, kill_reason: str, games: list[str], triggered_game: str | None = None
):
    """Patch the triggering game's result row to EARLY_KILL (orak signature).

    Thin wrapper over `autoresearch.relabel_last_as_early_kill` (v0.11.0+).
    Preserves the orak call-site shape: a filter on `game` field restricts
    relabelling to the triggered game, leaving other games' rows in the
    same iter untouched. Pass `triggered_game=None` to relabel all games.
    """
    games_to_kill = [triggered_game] if triggered_game else games
    n = _ar_relabel_last_as_early_kill(
        experiments_dir=ROOT / "experiments",
        tag=tag,
        kill_reason=kill_reason,
        filter_field="game",
        filter_values=games_to_kill,
        last_n=len(games),
    )
    print(
        f"  Relabelled {n} entr{'y' if n == 1 else 'ies'} as EARLY_KILL "
        f"({', '.join(games_to_kill)}): {kill_reason}"
    )


def _write_sidecar(tag: str, run_id: str, description: str, games: list[str]):
    """Write current_run.json sidecar for live chart updates (orak schema).

    Thin wrapper over `autoresearch.write_sidecar` (v0.11.0+). Keeps orak's
    historic payload schema (run_id / games array, no log_path / wandb_url
    fields) so existing chart renderers see the same JSON keys.
    """
    _ar_write_sidecar(
        {
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
            "games": games,
            "description": description,
        },
        tag=tag,
        experiments_dir=ROOT / "experiments",
    )


def _clear_sidecar(tag: str):
    """Remove current_run.json after run completes (thin wrapper, v0.11.0+)."""
    _ar_clear_sidecar(tag=tag, experiments_dir=ROOT / "experiments")


def _cleanup_threads():
    """Kill leaked wandb-core / game-server processes that exhaust threads.

    wandb 0.26+ spawns dozens of wandb-core helper processes that hold OS
    threads. Without cleanup, asyncio shutdown in subsequent runs hits
    'RuntimeError: can't start new thread'.
    """
    patterns = ["wandb-core", "wandb-internal", "game_server", "grpc"]
    killed = 0
    for pat in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pat],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().split("\n"):
                if not pid_str.strip():
                    continue
                try:
                    pid = int(pid_str)
                    # Don't kill our own process or its parent
                    if pid in (os.getpid(), os.getppid()):
                        continue
                    os.kill(pid, signal.SIGTERM)
                    killed += 1
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    if killed:
        print(f"  Cleaned up {killed} leaked processes")
        time.sleep(2)  # Give OS time to reclaim threads


def run_experiment(
    config_name: str,
    games: list[str],
    baseline_scores: dict[str, float] | None = None,
    tag: str = "macla",
    description: str = "",
    prev_run_id: str | None = None,
) -> tuple[str, float]:
    """Run an experiment with live triage monitoring. Returns (run_id, elapsed_min).

    When prev_run_id is set, the subprocess is launched with --load-checkpoint
    --prev-run-id <prev_run_id> so MACLA's learned procedures from the previous
    iter are restored at agent init. Without this, every iter starts from
    procedures=0 and re-learns from scratch — root cause of the inconsistency
    where one-shot bests (mario 77.83%, pokemon 14.29%, 2048 6.02%) couldn't
    be reproduced in subsequent iters.
    """
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "run.py"),
        f"--config-name={config_name}",
        "--local",
    ]
    for g in games:
        cmd.extend(["--games", g])
    if prev_run_id:
        cmd.extend(["--load-checkpoint", "--prev-run-id", prev_run_id])
        print(f"  Loading MACLA checkpoint from prev run: {prev_run_id}")

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Triage: plateau_default={TRIAGE_SCORE_PLATEAU_STEPS_DEFAULT}steps overrides={TRIAGE_SCORE_PLATEAU_STEPS_PER_GAME}, no_learn={TRIAGE_NO_LEARN_EPISODES}eps, baseline_gate={TRIAGE_BASELINE_FACTOR}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    # Disable Weave tracing in subprocess: project doesn't exist on wandb,
    # every trace upload returns 403, httpx retries forever, leaks CLOSE-WAIT
    # sockets, eventually run.py deadlocks.
    env["WEAVE_ENABLED"] = "false"
    env.setdefault("WEAVE_DISABLED", "true")
    baseline_scores = baseline_scores or {}

    # Mark the wall-clock instant just before Popen so we only accept run dirs
    # created AFTER this iteration started. Comparing run_id strings (e.g.
    # `candidate >= "20260503_095500"`) breaks for non-timestamp dir names
    # (e.g. `pokemon_stage_a_direct_*`) where the lexicographic order doesn't
    # match creation order — fall through to the most-recent-by-mtime dir
    # from a previous run instead.
    start_mtime = time.time()
    start = start_mtime
    iter_timeout_s = ITER_TIMEOUT_MIN * 60
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    kill_reason = None

    # Wait for run_id to appear in game_logs
    run_id = ""
    for _ in range(60):
        time.sleep(2)
        candidate = _find_run_id(games)
        if not candidate:
            continue
        candidate_dir = GAME_LOG_DIR / games[0] / candidate
        if candidate_dir.stat().st_mtime >= start_mtime:
            run_id = candidate
            break

    if run_id:
        _write_sidecar(tag, run_id, description, games)
        print(f"  Run ID: {run_id} — triage monitoring active")

    # Monitor loop: poll game_states.jsonl for triage signals
    while proc.poll() is None:
        time.sleep(TRIAGE_POLL_INTERVAL)

        # Wall-clock timeout: a single iteration must not stall the whole sweep
        if (time.time() - start) > iter_timeout_s:
            kill_reason = f"iteration timeout ({ITER_TIMEOUT_MIN}min wall-clock)"
            print(f"\n  TIMEOUT KILL: {kill_reason}")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            break

        if not run_id:
            candidate = _find_run_id(games)
            if candidate:
                candidate_dir = GAME_LOG_DIR / games[0] / candidate
                if candidate_dir.stat().st_mtime >= start_mtime:
                    run_id = candidate
                    _write_sidecar(tag, run_id, description, games)
            continue

        kill_reason = _triage_check(run_id, games, baseline_scores)
        if kill_reason:
            print(f"\n  TRIAGE KILL: {kill_reason}")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=30)
            break

    elapsed = (time.time() - start) / 60
    _clear_sidecar(tag)

    # Clean up wandb-core / game-server processes that hold OS threads
    _cleanup_threads()

    if not run_id:
        run_id = _find_run_id(games)

    if kill_reason:
        print(f"\n  EARLY KILL after {elapsed:.1f}min — {kill_reason}")
        return run_id, elapsed, kill_reason
    elif proc.returncode and proc.returncode != 0:
        print(f"  Run failed with exit code {proc.returncode}")
        return "", 0.0, None
    else:
        print(f"\n  Run completed in {elapsed:.1f}min — run_id: {run_id}")
        return run_id, elapsed, None


def log_run_results(
    run_id: str,
    games: list[str],
    description: str,
    tag: str = "macla",
    best_scores: dict[str, float] | None = None,
    runtime_min: float = 0.0,
    config_name: str | None = None,
):
    """Extract results from a run and log to experiment tracker.

    When config_name is set, results are written to
    experiments/<tag>/<config_name>/results.jsonl so multiple parallel sweeps
    don't trample each other.
    """
    results = extract_run_results(run_id, games)
    best_scores = best_scores or {}

    for game, data in results.items():
        eval_score = data["max_eval"]
        game_score = data["game_score"]
        steps = data["steps"]
        best = best_scores.get(game, 0)
        status = "KEEP" if eval_score > best else "DISCARD"

        wandb_project = {
            "super_mario": "orak-super-mario",
            "twenty_fourty_eight": "orak-2048",
            "pokemon_red": "orak-pokemon-red",
        }.get(game, game)
        wandb_url = f"https://wandb.ai/chaleong/{wandb_project}/runs/{run_id}_{wandb_project}"

        notes = f"max_eval={eval_score:.2f}, {data['episodes']} episodes, {steps} steps"
        if status == "KEEP":
            notes += f". Improved from {best:.2f}"
        else:
            notes += f". Below best {best:.2f}"

        log_experiment(
            game=game,
            score=eval_score,
            steps=steps,
            status=status,
            description=description,
            wandb_url=wandb_url,
            notes=notes,
            game_score=game_score,
            runtime_min=runtime_min,
            tags=[tag],
            config_name=config_name,
        )

    return results


def _run_post_iter_retrospective(
    *,
    tag: str,
    iteration: int,
    games: list[str],
    config_name: str | None,
    log_path: Path | None,
) -> list[Finding]:
    """Run autoresearch retrospective detectors against the iter we just logged.

    Per-game: load the row we just appended (last entry filtered by game) and
    its prior history, run the orak-tuned detector panel, write findings into
    `experiments/<tag>[/<config>]/retrospective_E<iter>_<game>.md`, mutate the
    row in `results.jsonl` to attach summaries.

    Returns the union of findings across all games so the caller can decide
    whether to fold warns into the next iter's description (self-correcting
    loop) or stop the sweep on a `block`.
    """
    findings_all: list[Finding] = []
    rows = load_results(tag=tag, config_name=config_name)
    out_dir = ROOT / "experiments" / tag
    if config_name:
        out_dir = out_dir / config_name

    for game in games:
        # Latest row for this game = the one we just logged.
        per_game = [r for r in rows if r.get("game") == game]
        if not per_game:
            continue
        cur_row = per_game[-1]
        history = per_game[:-1]

        findings = audit_iter(
            results_row=cur_row,
            log_path=log_path,
            per_row_jsonl_path=None,  # orak doesn't emit per-row rubric dumps
            history=history,
            detectors=ORAK_RETROSPECTIVE_DETECTORS,
            detector_kwargs=ORAK_RETROSPECTIVE_KWARGS,
        )
        if not findings:
            continue

        findings_all.extend(findings)
        attach_findings_to_row(cur_row, findings)

        # Write the markdown sibling for human review.
        md_path = out_dir / f"retrospective_E{iteration}_{game}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_markdown(findings, iter_id=iteration))

        # Re-write the row in results.jsonl with the attached findings. We
        # have to rewrite the whole file since results are append-only JSONL.
        results_file = out_dir / "results.jsonl"
        if results_file.exists():
            all_rows = [
                json.loads(line) for line in results_file.read_text().splitlines() if line.strip()
            ]
            # Match by (game, experiment) — the most recent row for this game
            # was just appended by log_experiment.
            target_iter = cur_row.get("experiment")
            for i, r in enumerate(all_rows):
                if r.get("game") == game and r.get("experiment") == target_iter:
                    all_rows[i] = cur_row
                    break
            results_file.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")

    return findings_all


@app.command()
def log_run(
    run_id: str = typer.Option(..., help="Run ID from game_logs (e.g. 20260422_213143)"),
    description: str = typer.Option(..., "-d", help="Experiment description"),
    tag: str = typer.Option("macla", help="Experiment tag"),
    games: list[str] = typer.Option(ALL_GAMES, "--games", help="Games to log"),
):
    """Log results from a completed run to the experiment tracker."""
    best = get_best_scores(tag)
    log_run_results(run_id, games, description, tag, best)
    plot_progress(tag=tag)


@app.command()
def run(
    config: str = typer.Option("unified_macla", help="Hydra config name"),
    tag: str = typer.Option("macla", help="Experiment tag"),
    max_iterations: int = typer.Option(5, help="Max optimisation iterations"),
    games: list[str] = typer.Option(ALL_GAMES, "--games", help="Games to optimise"),
    dry_run: bool = typer.Option(False, help="Only propose params, don't run"),
    config_type: str = typer.Option("unified_macla", help="YAML config type to modify"),
    note: str = typer.Option("", help="Extra context to prepend to experiment description (e.g. 'OnlineAgentEvaluator enabled')"),
    patience: int = typer.Option(5, help="Stop sweep if no game improves over best-so-far for N consecutive iterations (0 = disable)"),
    time_budget_min: float = typer.Option(0.0, help="Stop sweep after this many wall-clock minutes (0 = disable)"),
    config_name: str = typer.Option("", help="Per-config sub-dir (e.g. 'gemma' or 'qwen'). Empty = flat experiments/<TAG>/. Set to isolate parallel sweeps in experiments/<TAG>/<CONFIG_NAME>/."),
    log_path: str = typer.Option("", "--log-path", help="Path to the sweep's stdout log file. Enables silent_kill / triage_threshold_mismatch retrospective detectors that need to grep log lines."),
):
    """Run the autoresearch optimisation loop."""
    cfg = config_name or None
    print(f"Autoresearch loop: config={config}, tag={tag}, max_iterations={max_iterations}")
    print(f"Games: {games}")
    if cfg:
        print(f"Per-config sub-dir: experiments/{tag}/{cfg}/")
    print(f"Early stopping: patience={patience} iters, time_budget={time_budget_min}min")
    log_path_obj = Path(log_path) if log_path else None
    print(f"Retrospective: detectors={[d.name for d in ORAK_RETROSPECTIVE_DETECTORS]}, log_path={log_path_obj}\n")

    all_results = load_results(tag=tag, config_name=cfg)
    sweep_start = time.time()
    no_improve_streak = 0
    # Track the last successful run_id so the next iter can load its MACLA
    # checkpoint and carry procedures/meta-procedures forward — biggest lever
    # for cross-iter consistency.
    prev_run_id: str | None = None
    # Warn-level findings from the previous iter; prepended to this iter's
    # description so the autoresearch parameter proposer (and any downstream
    # plot reader) sees the context. Self-correcting loop pattern.
    pending_warns: list[str] = []

    for iteration in range(max_iterations):
        # Budget stop: hit the wall-clock cap
        if time_budget_min > 0:
            elapsed = (time.time() - sweep_start) / 60
            if elapsed >= time_budget_min:
                print(f"\nEarly stop: wall-clock budget reached ({elapsed:.1f}min >= {time_budget_min}min)")
                break
        print(f"\n{'#'*60}")
        print(f"# Iteration {iteration + 1}/{max_iterations}")
        print(f"{'#'*60}")

        best = get_best_scores(tag, config_name=cfg)
        print(f"\nCurrent best scores: {best}")

        # Propose and apply new params per game
        param_summaries = []
        for game in games:
            new_params = propose_next_params(game, all_results, config_type)
            current = get_current_params(game, config_type)

            changed = {k: v for k, v in new_params.items() if current.get(k) != v}
            if changed:
                print(f"\n  {game}: {changed}")
                # Short param summary: theta=0.20, warmup=5
                short = ", ".join(
                    f"{k.replace('macla_', '').replace('theta_', 'θ_')}={v}"
                    for k, v in sorted(changed.items())
                )
                param_summaries.append(f"{game.split('_')[0]}: {short}")

                if not dry_run:
                    full_config = read_yaml_config(game, config_type)
                    full_config.update(new_params)
                    write_yaml_config(game, full_config, config_type)
            else:
                print(f"\n  {game}: no changes (at boundary)")

        desc_parts = []
        if note:
            desc_parts.append(note)
        desc_parts.append(f"iter {iteration + 1}")
        if param_summaries:
            desc_parts.append("; ".join(param_summaries))
        else:
            desc_parts.append("no param changes")
        if pending_warns:
            # Surface previous iter's retrospective warnings into this iter's
            # description so reviewers see them inline on the chart / PR.
            desc_parts.append("retrospective: " + "; ".join(pending_warns[:3]))
            pending_warns = []
        description = " | ".join(desc_parts)
        print(f"\nDescription: {description}")

        if dry_run:
            print("\n[DRY RUN] Skipping experiment execution")
            continue

        # Run experiment with triage monitoring; carry MACLA procedures
        # forward by loading the previous iter's checkpoint.
        result = run_experiment(
            config, games, baseline_scores=best, tag=tag,
            description=description, prev_run_id=prev_run_id,
        )
        # run_experiment now returns a 3-tuple (run_id, elapsed_min, kill_reason).
        # Tolerate older 2-tuple returns from any third-party fork.
        if isinstance(result, tuple):
            run_id = result[0]
            elapsed_min = result[1] if len(result) > 1 else 0.0
            run_kill_reason = result[2] if len(result) > 2 else None
        else:
            run_id, elapsed_min, run_kill_reason = result, 0.0, None
        if not run_id:
            print("Run failed, stopping loop")
            break
        # This run becomes the source for the next iter's checkpoint load.
        prev_run_id = run_id

        # Analyze trajectories and apply targeted changes for NEXT iteration
        print(f"\n--- Trajectory Analysis ---")
        change_summaries = []
        for game in games:
            analysis = analyze_trajectory(run_id, game)
            print(
                f"\n  {game}: {analysis.get('total_steps', 0)} steps, "
                f"{analysis.get('episodes', 0)} episodes, "
                f"max_eval={analysis.get('max_eval', 0):.2f}"
            )
            if analysis.get("failure_zone"):
                print(f"    Death cluster: {analysis['failure_zone']} ({analysis.get('failure_zone_deaths', 0)} deaths)")
            if analysis.get("repeated_actions"):
                print(f"    Action repetition: {analysis.get('top_action', '?')} at {analysis.get('top_action_pct', 0):.0%}")
            if analysis.get("map_stuck"):
                print(f"    Map stuck: {analysis.get('maps_visited', [])}")

            changes = propose_changes(analysis)
            if changes:
                applied = apply_changes(game, changes, config_type)
                change_summaries.extend(applied)

        if change_summaries:
            description += " | " + "; ".join(change_summaries[:3])  # Cap for plot readability

        # Log results
        run_results = log_run_results(run_id, games, description, tag, best, runtime_min=elapsed_min, config_name=cfg)

        # Relabel the iter as EARLY_KILL when either:
        #  1. Triage check (plateau / no_learn / baseline_gate) flagged a game, OR
        #  2. The wall-clock iter timeout fired (run_kill_reason from run_experiment).
        # Without (2) the row stays as DISCARD and the silent_kill retrospective
        # detector can't tell a timeout from an honest low score.
        triage_reason = _triage_check(run_id, games, best) if run_id else None
        kill_reason = triage_reason or run_kill_reason
        if kill_reason:
            # Triage reasons are formatted as "<game>: <why>" — split out the game
            # so we only relabel the offender. Wall-clock timeouts aren't game-
            # specific, so relabel all games for that iter.
            triggered_game = (
                kill_reason.split(":")[0].strip() if triage_reason else None
            )
            _relabel_last_as_early_kill(tag, kill_reason, games, triggered_game=triggered_game)

        # Post-iter retrospective: run failure-mode detectors against the row
        # we just logged. Warn-level findings get prepended to next iter's
        # description; block-level stops the sweep entirely.
        findings = _run_post_iter_retrospective(
            tag=tag,
            iteration=iteration + 1,
            games=games,
            config_name=cfg,
            log_path=log_path_obj,
        )
        if findings:
            print(f"\n--- Retrospective ({len(findings)} finding(s)) ---")
            for f in findings:
                print(f"  [{f.severity.upper()}] [{f.detector}] {f.summary}")
            blocks = filter_by_severity(findings, "block")
            if blocks:
                print(f"\nEarly stop: retrospective produced {len(blocks)} block-level finding(s).")
                break
            for f in filter_by_severity(findings, "warn"):
                pending_warns.append(f"{f.detector}={f.suggested_action}")

        # Reload results for next iteration
        all_results = load_results(tag=tag, config_name=cfg)

        # Check if any game improved
        any_improved = False
        for game, data in run_results.items():
            if data["max_eval"] > best.get(game, 0):
                any_improved = True
                print(f"  {game}: IMPROVED {best.get(game, 0):.2f} → {data['max_eval']:.2f}")
            else:
                print(f"  {game}: no improvement (best={best.get(game, 0):.2f})")

        # Regenerate plot
        plot_progress(tag=tag, config_type=config_type, config_name=cfg)

        # Convergence stop: track consecutive iterations with no improvement
        if any_improved:
            no_improve_streak = 0
        else:
            no_improve_streak += 1
            print(f"\nNo improvements in iteration {iteration + 1} (streak={no_improve_streak}/{patience}).")
            if patience > 0 and no_improve_streak >= patience:
                print(f"\nEarly stop: no improvement for {patience} consecutive iterations.")
                break

    elapsed_total = (time.time() - sweep_start) / 60
    print(f"\nAutoresearch complete after {min(iteration + 1, max_iterations)} iterations ({elapsed_total:.1f}min)")
    plot_progress(tag=tag, config_type=config_type, config_name=cfg)


@app.command()
def analyze(
    run_id: str = typer.Option(..., help="Run ID to analyze"),
    games: list[str] = typer.Option(ALL_GAMES, "--games", help="Games to analyze"),
    propose: bool = typer.Option(False, help="Also propose changes (don't apply)"),
):
    """Analyze trajectories from a completed run and report failure patterns."""
    for game in games:
        analysis = analyze_trajectory(run_id, game)
        if analysis.get("error"):
            print(f"\n{game}: {analysis['error']}")
            continue

        print(f"\n{'='*50}")
        print(f"  {game.upper()}")
        print(f"{'='*50}")
        print(f"  Steps: {analysis['total_steps']}, Episodes: {analysis['episodes']}")
        print(f"  Max eval: {analysis['max_eval']:.2f}")
        print(f"  Top action: {analysis['top_action']} ({analysis['top_action_pct']:.0%})")
        print(f"  Action dist: {analysis['action_distribution']}")
        print(f"  Score plateau: {analysis.get('score_plateau', False)}")
        print(f"  Repeated actions: {analysis.get('repeated_actions', False)}")

        if game == "super_mario":
            if analysis.get("failure_zone"):
                print(f"  Death cluster: {analysis['failure_zone']} ({analysis['failure_zone_deaths']}/{analysis['failure_zone_total_deaths']} deaths)")
            if analysis.get("death_positions"):
                print(f"  Death positions: {analysis['death_positions'][:10]}...")

        elif game == "pokemon_red":
            print(f"  Maps visited: {analysis.get('maps_visited', [])}")
            print(f"  Map stuck: {analysis.get('map_stuck', False)}")
            print(f"  Max flags: {analysis.get('max_flags', 0)}")

        elif game == "twenty_fourty_eight":
            print(f"  Max tile: {analysis.get('max_tile', 0)}")
            print(f"  Action imbalance: {analysis.get('action_imbalance', False)}")
            if analysis.get("direction_balance"):
                print(f"  Direction balance: {analysis['direction_balance']}")

        if propose:
            changes = propose_changes(analysis)
            if changes:
                print(f"\n  Proposed changes:")
                for c in changes:
                    print(f"    [{c['type']}] {c['target']}: {c['reason']}")
                    if c["type"] == "prompt":
                        print(f"           → append: '{c['text'][:80]}...'")
            else:
                print(f"\n  No changes proposed")


if __name__ == "__main__":
    app()
