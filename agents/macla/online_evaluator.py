"""
OnlineAgentEvaluator: per-step reward shaping for MACLA.

Replaces binary success/fail with continuous rewards based on
game-specific score deltas, position progress, and metric changes.

Architecture: each game has its own `RewardShaper` subclass that owns its
metric extraction, reward computation, and per-episode state. The evaluator
itself is a thin coordinator. Add a new game by writing a `<Game>Shaper`
subclass and registering it in `SHAPERS`.

Shaping params per game live in DEFAULT_SHAPING. Override per-agent via the
agent yaml `reward_shaping:` block, e.g.:

    # configs/<game>/agent/<variant>.yaml
    reward_shaping:
      repeat_visit_bonus: 0.5
      flag_bonus: 5.0

Missing keys fall back to the per-game DEFAULT_SHAPING entry. Useful for
ablation sweeps over shaping values without editing source.
"""

import re
from collections import deque

from loguru import logger

# ── Shaping defaults ────────────────────────────────────────

# Per-game default shaping params. See module docstring for override syntax.
# Game-specific rationale lives next to the parameters that encode it.
DEFAULT_SHAPING: dict[str, dict[str, float]] = {
    "super_mario": {
        "fatal_penalty": -2.0,
        "x_progress_divisor": 100.0,
        "score_delta_divisor": 200.0,
        "lives_lost_penalty": -1.5,
        "stagnation_threshold_steps": 3,
        "stagnation_penalty": -0.3,
        "stagnation_x_threshold": 3.0,
        "reward_min": -2.0,
        "reward_max": 3.0,
    },
    "twenty_fourty_eight": {
        "fatal_penalty": -1.5,
        "score_delta_divisor": 200.0,
        "tile_double_bonus": 1.5,
        "free_cell_bonus": 0.3,
        "crowding_penalty": -0.2,
        # Corner-anchoring is the dominant 2048 strategy. Densifying its signal
        # gives MACLA procedure-update events for "max-tile in corner → GOOD"
        # and "anchor disturbed → BAD" without waiting for terminal reward.
        "corner_anchor_bonus": 0.4,
        "anchor_disturbed_penalty": -0.5,
        "stagnation_threshold_steps": 3,
        "stagnation_penalty": -0.5,
        "reward_min": -2.0,
        "reward_max": 3.0,
    },
    "pokemon_red": {
        "fatal_penalty": -1.5,
        # Reward only the *discovery* of a new map; default 0 for re-entries.
        # Rewarding every transition unconditionally (repeat_visit_bonus > 0)
        # creates a known reward-hack: a 2-map loop (e.g. a staircase) becomes
        # infinite reward and the agent oscillates instead of exploring. Set
        # > 0 only when back-and-forth motion is genuinely desirable.
        "map_discovery_bonus": 1.5,
        "repeat_visit_bonus": 0.0,
        "flag_bonus": 3.0,
        "score_delta_divisor": 2.0,
        "stagnation_threshold_steps": 3,
        "stagnation_penalty": -0.4,
        "reward_min": -2.0,
        "reward_max": 3.0,
    },
    "star_craft": {
        # Terminal rewards
        "fatal_penalty": -2.0,
        "victory_bonus": 3.0,
        # Per-step positive deltas
        "supply_used_weight": 0.2,
        "building_built_weight": 0.5,
        "survival_increment": 0.05,
        "first_enemy_bonus": 0.5,
        # State-based penalties (the load-bearing fix)
        #
        # Floated-minerals penalty fires when mineral grows but supply_used
        # is flat — i.e. the agent is collecting resources without spending
        # them on units. Symptom from PR3 smoke iter 201: 3980 minerals + only
        # 1 Pylon + supply-blocked. Without this penalty, a naive Δ-mineral
        # term would reward the same state. Mirrors the repeat_visit_bonus
        # warning above: same class of reward hack.
        "floated_minerals_penalty": -0.3,
        # Supply-block fires when supply_left <= 0 — the agent cannot train
        # new units regardless of mineral. Critical state to penalize.
        "supply_block_penalty": -0.5,
        # Stagnation (no game_time progress for N steps in a row)
        "stagnation_threshold_steps": 3,
        "stagnation_penalty": -0.3,
        "reward_min": -2.0,
        "reward_max": 3.0,
    },
}


# ── Regex helpers (stateless) ───────────────────────────────


def _find_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _find_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _find_str(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None


# ── Shaper protocol ─────────────────────────────────────────


class RewardShaper:
    """Base class — game shapers extract metrics and compute rewards.

    Subclasses must implement `extract_metrics` and `compute_reward`. They
    inherit a per-episode `_stagnation_count` and a `_shaping` dict (defaults
    overlaid with per-agent overrides). Override `reset_episode` to clear
    any additional per-episode state (e.g. visited-maps set).
    """

    def __init__(self, shaping: dict):
        self._shaping = shaping
        self._stagnation_count: int = 0

    def extract_metrics(self, state: str) -> dict:
        raise NotImplementedError

    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        raise NotImplementedError

    def reset_episode(self) -> None:
        self._stagnation_count = 0

    def _clamp(self, reward: float) -> float:
        s = self._shaping
        return max(s["reward_min"], min(s["reward_max"], reward))


# ── Mario ───────────────────────────────────────────────────


class MarioShaper(RewardShaper):
    def extract_metrics(self, state: str) -> dict:
        return {
            "x_pos": _find_float(r"x_pos:?\s*(\d+\.?\d*)", state) or 0,
            "score": _find_float(r"[Ss]core:?\s*(\d+)", state) or 0,
            "lives": _find_int(r"[Ll]ives:?\s*(\d+)", state),
        }

    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping
        if is_fatal:
            return s["fatal_penalty"]

        reward = 0.0
        x_delta = cur.get("x_pos", 0) - prev.get("x_pos", 0)
        reward += x_delta / s["x_progress_divisor"]

        score_delta = cur.get("score", 0) - prev.get("score", 0)
        if score_delta > 0:
            reward += score_delta / s["score_delta_divisor"]

        if prev.get("lives") is not None and cur.get("lives") is not None:
            if cur["lives"] < prev["lives"]:
                reward += s["lives_lost_penalty"]

        if abs(x_delta) < s["stagnation_x_threshold"]:
            self._stagnation_count += 1
            if self._stagnation_count >= s["stagnation_threshold_steps"]:
                reward += s["stagnation_penalty"]
        else:
            self._stagnation_count = 0

        return self._clamp(reward)


# ── 2048 ────────────────────────────────────────────────────


class TwentyFortyEightShaper(RewardShaper):
    _BOARD_RE = re.compile(
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
        r"\s*,\s*"
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
        r"\s*,\s*"
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
        r"\s*,\s*"
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
    )

    def extract_metrics(self, state: str) -> dict:
        score = _find_float(r"[Ss]core:?\s*(\d+\.?\d*)", state) or 0
        max_tile = _find_int(r"[Mm]ax.?[Tt]ile:?\s*(\d+)", state) or 0
        empty_cells = (
            state.count(", 0,") + state.count("[0,") + state.count(", 0]") + state.count("[0]")
        )
        return {
            "score": score,
            "max_tile": max_tile,
            "empty_cells": empty_cells,
            "max_pos": self._max_position(state, max_tile),
        }

    def _max_position(self, state: str, max_tile: int) -> str:
        """Return 'corner' / 'edge' / 'center' / 'unknown' for the max tile."""
        if max_tile <= 0:
            return "unknown"
        m = self._BOARD_RE.search(state)
        if not m:
            return "unknown"
        cells = [int(x) for x in m.groups()]
        try:
            idx = cells.index(max_tile)
        except ValueError:
            return "unknown"
        r, c = divmod(idx, 4)
        if (r, c) in {(0, 0), (0, 3), (3, 0), (3, 3)}:
            return "corner"
        if r in (0, 3) or c in (0, 3):
            return "edge"
        return "center"

    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping
        if is_fatal:
            return s["fatal_penalty"]

        reward = 0.0
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        reward += score_delta / s["score_delta_divisor"]

        prev_tile = prev.get("max_tile", 0)
        cur_tile = cur.get("max_tile", 0)
        if cur_tile > prev_tile and prev_tile > 0:
            reward += s["tile_double_bonus"]

        prev_empty = prev.get("empty_cells", 0)
        cur_empty = cur.get("empty_cells", 0)
        if cur_empty > prev_empty:
            reward += s["free_cell_bonus"]
        elif cur_empty < prev_empty - 1:
            reward += s["crowding_penalty"]

        prev_pos = prev.get("max_pos")
        cur_pos = cur.get("max_pos")
        if cur_pos == "corner":
            reward += s["corner_anchor_bonus"]
        if prev_pos == "corner" and cur_pos in ("edge", "center"):
            reward += s["anchor_disturbed_penalty"]

        if score_delta == 0 and not is_fatal:
            self._stagnation_count += 1
            if self._stagnation_count >= s["stagnation_threshold_steps"]:
                reward += s["stagnation_penalty"]
        else:
            self._stagnation_count = 0

        return self._clamp(reward)


# ── Pokemon Red ─────────────────────────────────────────────


class PokemonShaper(RewardShaper):
    def __init__(self, shaping: dict):
        super().__init__(shaping)
        self._visited_maps: set[str] = set()

    def reset_episode(self) -> None:
        super().reset_episode()
        self._visited_maps = set()

    def extract_metrics(self, state: str) -> dict:
        return {
            "score": _find_float(r"[Ss]core:?\s*(\d+\.?\d*)", state) or 0,
            "flags": _find_int(r"[Ff]lags?:?\s*(\d+)", state) or 0,
            "map_name": _find_str(r"[Mm]ap.?[Nn]ame:?\s*(\S+)", state) or "",
        }

    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping
        if is_fatal:
            return s["fatal_penalty"]

        reward = 0.0
        prev_map = prev.get("map_name", "")
        cur_map = cur.get("map_name", "")
        map_changed = bool(cur_map and prev_map and cur_map != prev_map)
        if map_changed:
            if cur_map not in self._visited_maps:
                reward += s["map_discovery_bonus"]
            else:
                reward += s["repeat_visit_bonus"]
        if cur_map:
            self._visited_maps.add(cur_map)

        flag_delta = cur.get("flags", 0) - prev.get("flags", 0)
        if flag_delta > 0:
            reward += s["flag_bonus"] * flag_delta

        score_delta = cur.get("score", 0) - prev.get("score", 0)
        if score_delta > 0:
            reward += score_delta / s["score_delta_divisor"]

        if score_delta == 0 and flag_delta == 0 and not map_changed:
            self._stagnation_count += 1
            if self._stagnation_count >= s["stagnation_threshold_steps"]:
                reward += s["stagnation_penalty"]
        else:
            self._stagnation_count = 0

        return self._clamp(reward)


# ── StarCraft II ────────────────────────────────────────────


class StarCraftShaper(RewardShaper):
    """Per-step shaped reward for the SC2 adapter.

    Reads structured fields from the obs_str text summary emitted by
    star_craft_env.obs2text: `Game time`, `Mineral`, `Supply used/cap/left`,
    `Worker supply`, building counts, and enemy-unit counts. Race-agnostic
    by construction — the regexes don't reference Pylon/SupplyDepot/Overlord
    specifically.

    The load-bearing signal is the idleness + supply-block penalties: PR3 smoke
    showed avg_procedure_success_rate=0.51 across 2500 steps with zero
    successful_executions — procedural memory had nothing to refine against.
    Without these penalties, mere mineral accumulation would still earn
    positive reward, teaching the wrong lesson.
    """

    # Buildings list — sum of all `X count: N` matches excluding workers
    # and in-progress markers (Probe/Worker/Producing/Constructing).
    _BUILDING_EXCLUDE = ("Probe", "Worker", "Producing", "Constructing")

    def __init__(self, shaping: dict):
        super().__init__(shaping)
        self._seen_enemy_unit: bool = False

    def reset_episode(self) -> None:
        super().reset_episode()
        self._seen_enemy_unit = False

    def extract_metrics(self, state: str) -> dict:
        # Multi-summary obs_str states (multiple "Summary N:" blocks concatenated)
        # do not currently occur in the SC2 adapter — empirical check on the PR3
        # smoke (2500 iters) found zero such cases. game_time_sec uses LAST-match
        # defensively for future-proofing; all other scalars use _find_int (FIRST
        # match). If the adapter ever starts emitting multi-summary states, switch
        # all scalars to LAST-match for delta-consistency.
        gt_matches = re.findall(r"Game time:\s*(\d+):(\d+)", state)
        if gt_matches:
            mm, ss = gt_matches[-1]
            game_time_sec = int(mm) * 60 + int(ss)
        else:
            game_time_sec = 0

        # Building count: sum all "X count: N" matches except worker/in-progress
        # markers. The regex is greedy over `[\w ]+` and relies on the closed
        # _BUILDING_EXCLUDE list — if the SC2 adapter ever adds new aggregate
        # fields like "Total army count:" or "Attack count:" they would inflate
        # this sum and need explicit exclusion.
        building_count = 0
        for name, n in re.findall(r"([\w ]+) count:\s*(\d+)", state):
            if any(excluded in name for excluded in self._BUILDING_EXCLUDE):
                continue
            building_count += int(n)

        # Enemy unit count: sum all "Enemy unittypeid.X: N" matches.
        enemy_unit_count = sum(int(n) for n in re.findall(r"Enemy unittypeid\.\w+:\s*(\d+)", state))

        return {
            "game_time_sec": game_time_sec,
            "mineral": _find_int(r"Mineral:\s*(\d+)", state) or 0,
            "supply_used": _find_int(r"Supply used:\s*(\d+)", state) or 0,
            "supply_cap": _find_int(r"Supply cap:\s*(\d+)", state) or 0,
            "supply_left": _find_int(r"Supply left:\s*(-?\d+)", state) or 0,
            "worker_supply": _find_int(r"Worker supply:\s*(\d+)", state) or 0,
            "building_count": building_count,
            "enemy_unit_count": enemy_unit_count,
        }

    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping

        if is_fatal:
            return s["fatal_penalty"]
        if success:
            return s["victory_bonus"]

        reward = 0.0

        # Survival baseline: tiny constant when game_time advances.
        if cur.get("game_time_sec", 0) > prev.get("game_time_sec", 0):
            reward += s["survival_increment"]

        # Supply_used delta — army / worker built.
        supply_delta = cur.get("supply_used", 0) - prev.get("supply_used", 0)
        if supply_delta > 0:
            reward += s["supply_used_weight"] * supply_delta

        # Building delta — structure built.
        building_delta = cur.get("building_count", 0) - prev.get("building_count", 0)
        if building_delta > 0:
            reward += s["building_built_weight"] * building_delta

        # Floated-minerals penalty: mineral grows but supply_used flat → idle.
        mineral_delta = cur.get("mineral", 0) - prev.get("mineral", 0)
        if mineral_delta > 0 and supply_delta == 0:
            reward += s["floated_minerals_penalty"]

        # Supply-block penalty: cannot train new units.
        if cur.get("supply_left", 1) <= 0:
            reward += s["supply_block_penalty"]

        return self._clamp(reward)


# ── Generic fallback ────────────────────────────────────────


class GenericShaper(RewardShaper):
    def extract_metrics(self, state: str) -> dict:
        return {"score": _find_float(r"[Ss]core:?\s*(\d+\.?\d*)", state) or 0}

    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        if is_fatal:
            return -1.0
        if success:
            score_delta = cur.get("score", 0) - prev.get("score", 0)
            return max(0.5, min(3.0, score_delta / 100.0 + 0.5))
        return 0.0


# ── Registry + evaluator ────────────────────────────────────

SHAPERS: dict[str, type[RewardShaper]] = {
    "super_mario": MarioShaper,
    "twenty_fourty_eight": TwentyFortyEightShaper,
    "pokemon_red": PokemonShaper,
    "star_craft": StarCraftShaper,
}


class OnlineAgentEvaluator:
    """Coordinates a per-game `RewardShaper`. Stateless across games."""

    def __init__(self, game_name: str, shaping_overrides: dict | None = None):
        self._game_name = game_name
        self._prev_metrics: dict = {}
        self._step_rewards: deque = deque(maxlen=100)
        self._shaping: dict[str, float] = {
            **DEFAULT_SHAPING.get(game_name, {}),
            **(shaping_overrides or {}),
        }
        shaper_cls = SHAPERS.get(game_name, GenericShaper)
        self._shaper: RewardShaper = shaper_cls(self._shaping)

    def evaluate_step(
        self, prev_state: str, cur_state: str, success: bool, is_fatal: bool
    ) -> float:
        """Compute shaped reward in [reward_min, reward_max] (defaults: -2.0 to 3.0)."""
        cur_metrics = self._shaper.extract_metrics(cur_state)
        prev_metrics = self._prev_metrics or self._shaper.extract_metrics(prev_state)

        reward = self._shaper.compute_reward(prev_metrics, cur_metrics, success, is_fatal)

        self._prev_metrics = cur_metrics
        self._step_rewards.append(reward)
        logger.debug(
            f"[Evaluator] {self._game_name} shaped_reward={reward:.3f} metrics={cur_metrics}"
        )
        return reward

    def mean_reward(self) -> float:
        return sum(self._step_rewards) / len(self._step_rewards) if self._step_rewards else 0.0

    def last_reward(self) -> float:
        return self._step_rewards[-1] if self._step_rewards else 0.0

    def reset_episode(self) -> None:
        self._prev_metrics = {}
        self._shaper.reset_episode()

    # Backward-compat shim — some callers/tests reach for the visited-maps set
    # directly. Forward to the pokemon shaper when applicable; empty set otherwise.
    @property
    def _visited_maps(self) -> set[str]:
        return getattr(self._shaper, "_visited_maps", set())
