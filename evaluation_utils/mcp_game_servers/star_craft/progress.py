"""Progressive StarCraft II evaluation metric — milestone ladder.

Mirrors ``pokemon_red_env.evaluate()``: ``score = milestones_reached / N * 100``.
Replaces the binary Victory/no-Victory eval (``star_craft_env.evaluate``) with a
graded 0-100 measure of economic/military progress, so the ceiling check can
discriminate models that build economy/army/make-contact but never win a full
SC2 match.

Thresholds are calibrated from Qwen3.6 seed-1's per-episode peak distribution
and frozen (see docs/superpowers/specs/2026-06-03-starcraft-progress-metric-design.md).

Pure module: stdlib only, no SC2/game-server imports, so it stays unit-testable
without the burnysc2 stack.
"""

from __future__ import annotations

import re
from statistics import mean

# (field, threshold): a rung is credited when the episode's PEAK value for
# `field` reaches `threshold`. Rungs are counted independently, so the score is
# monotonic — more peak progress can only add rungs, never remove them.
MILESTONE_RUNGS: tuple[tuple[str, int], ...] = (
    ("building_count", 2),  # M1 built first structure
    ("supply_cap", 23),  # M2 expanded supply (first Pylon)
    ("worker_supply", 16),  # M3 built up economy
    ("supply_used", 20),  # M4 trained army
    ("building_count", 8),  # M5 tech / production
    ("supply_used", 34),  # M6 larger army
    ("enemy_unit_count", 1),  # M7 engaged enemy
)
TOTAL_MILESTONES = len(MILESTONE_RUNGS) + 1  # 7 economic/military rungs + Victory (M8)

# Metric fields the ladder reads — peaked per episode. Matches the field names
# StarCraftShaper.extract_metrics emits, so logged metrics feed straight in.
PEAK_FIELDS: tuple[str, ...] = (
    "building_count",
    "supply_cap",
    "worker_supply",
    "supply_used",
    "enemy_unit_count",
)


# A drop in SC2 game-time of at least this many seconds marks an episode reset.
# Guards against false splits on noisy single-step dips (a 1s wobble is not a
# new game).
GAME_TIME_RESET_GUARD = 2


# Names containing any of these are workers / in-progress markers, not finished
# structures — excluded from building_count.
_BUILDING_EXCLUDE = ("Probe", "Worker", "Producing", "Constructing")


def _find_int(pattern: str, text: str) -> int:
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def extract_metrics(text: str) -> dict:
    """Parse an SC2 obs_str into the metric fields the ladder reads.

    Single source of truth for the obs→metrics regexes (StarCraftShaper delegates
    here). Absent fields default to 0. Multi-`Game time` states use the last match.
    """
    gt = re.findall(r"Game time:\s*(\d+):(\d+)", text)
    game_time_sec = int(gt[-1][0]) * 60 + int(gt[-1][1]) if gt else 0

    building_count = 0
    for name, n in re.findall(r"([\w ]+) count:\s*(\d+)", text):
        if any(excluded in name for excluded in _BUILDING_EXCLUDE):
            continue
        building_count += int(n)

    enemy_unit_count = sum(int(n) for n in re.findall(r"Enemy unittypeid\.\w+:\s*(\d+)", text))

    return {
        "game_time_sec": game_time_sec,
        "mineral": _find_int(r"Mineral:\s*(\d+)", text),
        "supply_used": _find_int(r"Supply used:\s*(\d+)", text),
        "supply_cap": _find_int(r"Supply cap:\s*(\d+)", text),
        "supply_left": _find_int(r"Supply left:\s*(-?\d+)", text),
        "worker_supply": _find_int(r"Worker supply:\s*(\d+)", text),
        "building_count": building_count,
        "enemy_unit_count": enemy_unit_count,
    }


def milestone_score(peaks: dict, victory: bool) -> float:
    """0-100 fraction of SC2 milestones reached given an episode's peak metrics."""
    reached = sum(1 for field, thr in MILESTONE_RUNGS if peaks.get(field, 0) >= thr)
    if victory:
        reached += 1
    return reached / TOTAL_MILESTONES * 100.0


def split_episodes(steps: list[dict]) -> list[list[dict]]:
    """Split a run's per-step metric dicts into episodes on game_time resets."""
    episodes: list[list[dict]] = []
    cur: list[dict] = []
    prev_t: int | None = None
    for step in steps:
        t = step.get("game_time_sec", 0)
        if prev_t is not None and t <= prev_t - GAME_TIME_RESET_GUARD and cur:
            episodes.append(cur)
            cur = []
        cur.append(step)
        prev_t = t
    if cur:
        episodes.append(cur)
    return episodes


def merge_peaks(running: dict, metrics: dict) -> dict:
    """Element-wise max of a running peak dict with one step's metrics.

    Lets the live env keep peaks incrementally (O(1) per step) instead of
    re-scanning the whole episode each step.
    """
    return {f: max(running.get(f, 0), metrics.get(f, 0)) for f in PEAK_FIELDS}


def episode_peaks(episode: list[dict]) -> dict:
    """Peak (max) value of each ladder field over an episode's steps."""
    peaks = dict.fromkeys(PEAK_FIELDS, 0)
    for step in episode:
        peaks = merge_peaks(peaks, step)
    return peaks


def run_progress(steps: list[dict]) -> dict:
    """Aggregate a full run's per-step metrics into the headline progress score.

    Each step is a metric dict (PEAK_FIELDS + a truthy ``victory`` when that step
    ended in a win). Splits into episodes, scores each by its peak state, and
    returns the mean (headline), best episode, binary win-rate (secondary), and
    episode count.
    """
    episodes = split_episodes(steps)
    if not episodes:
        return {
            "starcraft_progress": 0.0,
            "starcraft_progress_best": 0.0,
            "star_craft_victory": 0.0,
            "n_episodes": 0,
        }
    victories = [any(s.get("victory") for s in ep) for ep in episodes]
    per_ep = [milestone_score(episode_peaks(ep), v) for ep, v in zip(episodes, victories)]
    return {
        "starcraft_progress": mean(per_ep),
        "starcraft_progress_best": max(per_ep),
        "star_craft_victory": sum(victories) / len(episodes),
        "n_episodes": len(episodes),
    }
