"""
OnlineAgentEvaluator: per-step reward shaping for MACLA.

Replaces binary success/fail with continuous rewards based on
game-specific score deltas, position progress, and metric changes.

Shaping params per game live in DEFAULT_SHAPING below and can be overridden
via the agent yaml (block: `reward_shaping:`). This lets ablations sweep over
shaping values without editing source. See PR #28 v6 for context.

TODO(refactor, post-PR-#28): pull each `_reward_<game>` into its own
`RewardShaper` strategy class with a registry, mirroring how UnifiedMaclaAgent
dispatches by game. The current single-class dict-dispatch is workable but
will become unwieldy as shaping grows.
"""
import re
from collections import deque

from loguru import logger


# Per-game default shaping params. Override via agent yaml `reward_shaping:`
# block; missing keys fall back to these defaults.
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
        "corner_anchor_bonus": 0.4,
        "anchor_disturbed_penalty": -0.5,
        "stagnation_threshold_steps": 3,
        "stagnation_penalty": -0.5,
        "reward_min": -2.0,
        "reward_max": 3.0,
    },
    "pokemon_red": {
        "fatal_penalty": -1.5,
        # PR #28 v6 fix: only the FIRST visit to a new map is rewarded
        # (`map_discovery_bonus`); re-entering an already-visited map gives
        # `repeat_visit_bonus` (default 0). Set repeat_visit_bonus > 0 only
        # if you specifically want to reward back-and-forth movement (the old
        # behavior that caused the warp-loop reward hack).
        "map_discovery_bonus": 1.5,
        "repeat_visit_bonus": 0.0,
        "flag_bonus": 3.0,            # per flag delta
        "score_delta_divisor": 2.0,
        "stagnation_threshold_steps": 3,
        "stagnation_penalty": -0.4,
        "reward_min": -2.0,
        "reward_max": 3.0,
    },
}


class OnlineAgentEvaluator:
    """Computes continuous per-step rewards from game state deltas."""

    def __init__(self, game_name: str, shaping_overrides: dict | None = None):
        self._game_name = game_name
        self._prev_metrics: dict = {}
        self._step_rewards: deque = deque(maxlen=100)
        self._stagnation_count: int = 0
        # Per-episode visited-map set for pokemon. Without this, the warp tile
        # between RedsHouse1f ↔ RedsHouse2f becomes an infinite +1.5/step reward
        # loop and the agent never explores the world (PR #28 v6 diagnosis).
        self._visited_maps: set[str] = set()
        # Shaping = defaults overlaid with per-agent overrides. Unknown game
        # name yields {} defaults; reward methods then use .get(key, fallback).
        self._shaping: dict[str, float] = {
            **DEFAULT_SHAPING.get(game_name, {}),
            **(shaping_overrides or {}),
        }

    def evaluate_step(
        self, prev_state: str, cur_state: str, success: bool, is_fatal: bool
    ) -> float:
        """Compute shaped reward in [-2.0, 3.0] range."""
        cur_metrics = self._extract_metrics(cur_state)
        prev_metrics = self._prev_metrics or self._extract_metrics(prev_state)

        reward = self._compute_reward(prev_metrics, cur_metrics, success, is_fatal)

        self._prev_metrics = cur_metrics
        self._step_rewards.append(reward)
        logger.debug(f"[Evaluator] {self._game_name} shaped_reward={reward:.3f} metrics={cur_metrics}")
        return reward

    def mean_reward(self) -> float:
        return sum(self._step_rewards) / len(self._step_rewards) if self._step_rewards else 0.0

    def last_reward(self) -> float:
        return self._step_rewards[-1] if self._step_rewards else 0.0

    def reset_episode(self):
        self._prev_metrics = {}
        self._stagnation_count = 0
        self._visited_maps = set()

    def _extract_metrics(self, state: str) -> dict:
        extractors = {
            "super_mario": self._extract_mario,
            "twenty_fourty_eight": self._extract_2048,
            "pokemon_red": self._extract_pokemon,
        }
        extractor = extractors.get(self._game_name, self._extract_generic)
        return extractor(state)

    def _compute_reward(
        self, prev: dict, cur: dict, success: bool, is_fatal: bool
    ) -> float:
        computers = {
            "super_mario": self._reward_mario,
            "twenty_fourty_eight": self._reward_2048,
            "pokemon_red": self._reward_pokemon,
        }
        computer = computers.get(self._game_name, self._reward_generic)
        return computer(prev, cur, success, is_fatal)

    # ── Mario ───────────────────────────────────────────────

    def _extract_mario(self, state: str) -> dict:
        x = self._find_float(r"x_pos:?\s*(\d+\.?\d*)", state) or 0
        score = self._find_float(r"[Ss]core:?\s*(\d+)", state) or 0
        lives = self._find_int(r"[Ll]ives:?\s*(\d+)", state)
        return {"x_pos": x, "score": score, "lives": lives}

    def _reward_mario(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping
        if is_fatal:
            return s["fatal_penalty"]

        reward = 0.0
        # Position progress (main signal)
        x_delta = cur.get("x_pos", 0) - prev.get("x_pos", 0)
        reward += x_delta / s["x_progress_divisor"]

        # Score delta
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        if score_delta > 0:
            reward += score_delta / s["score_delta_divisor"]

        # Lives lost
        if prev.get("lives") is not None and cur.get("lives") is not None:
            if cur["lives"] < prev["lives"]:
                reward += s["lives_lost_penalty"]

        # Stagnation penalty
        if abs(x_delta) < s["stagnation_x_threshold"]:
            self._stagnation_count += 1
            if self._stagnation_count >= s["stagnation_threshold_steps"]:
                reward += s["stagnation_penalty"]
        else:
            self._stagnation_count = 0

        return max(s["reward_min"], min(s["reward_max"], reward))

    # ── 2048 ────────────────────────────────────────────────

    def _extract_2048(self, state: str) -> dict:
        score = self._find_float(r"[Ss]core:?\s*(\d+\.?\d*)", state) or 0
        max_tile = self._find_int(r"[Mm]ax.?[Tt]ile:?\s*(\d+)", state) or 0
        # Count empty cells from board representation
        empty_cells = state.count(", 0,") + state.count("[0,") + state.count(", 0]") + state.count("[0]")
        # Locate max-tile position (corner / edge / center) — corner-anchoring
        # is the dominant 2048 strategy, so densifying its signal helps the
        # agent learn it from procedure feedback rather than from terminal
        # game-over reward alone.
        max_pos = self._extract_2048_max_position(state, max_tile)
        return {"score": score, "max_tile": max_tile, "empty_cells": empty_cells, "max_pos": max_pos}

    def _extract_2048_max_position(self, state: str, max_tile: int) -> str:
        """Return 'corner' / 'edge' / 'center' / 'unknown' for the max tile.

        Parses the 4×4 board from the state string. The board is typically
        rendered as nested lists, e.g. '[[0, 2, 4, 0], [0, 0, 8, 0], ...]'.
        """
        if max_tile <= 0:
            return "unknown"
        # Match 4 consecutive rows of 4 numbers each
        m = re.search(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
            r"\s*,\s*"
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
            r"\s*,\s*"
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
            r"\s*,\s*"
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
            state,
        )
        if not m:
            return "unknown"
        cells = [int(x) for x in m.groups()]
        # Find max-tile cell index (0-15, row-major)
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

    def _reward_2048(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping
        if is_fatal:
            return s["fatal_penalty"]

        reward = 0.0
        # Score delta (main signal — reward merges proportionally)
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        reward += score_delta / s["score_delta_divisor"]

        # Max tile increase (big bonus for doubling)
        prev_tile = prev.get("max_tile", 0)
        cur_tile = cur.get("max_tile", 0)
        if cur_tile > prev_tile and prev_tile > 0:
            reward += s["tile_double_bonus"]

        # Empty cells: reward freeing space, penalize filling board
        prev_empty = prev.get("empty_cells", 0)
        cur_empty = cur.get("empty_cells", 0)
        if cur_empty > prev_empty:
            reward += s["free_cell_bonus"]
        elif cur_empty < prev_empty - 1:
            reward += s["crowding_penalty"]

        # Corner-anchoring (the dominant 2048 strategy). Densifies the strategic
        # signal so procedures get update events for "max-tile in corner →
        # GOOD" and "anchor disturbed → BAD" without needing terminal game-over
        # feedback. Doesn't bias toward any specific corner (all 4 are
        # symmetric); it's the *invariant* that's rewarded.
        prev_pos = prev.get("max_pos")
        cur_pos = cur.get("max_pos")
        if cur_pos == "corner":
            reward += s["corner_anchor_bonus"]
        if prev_pos == "corner" and cur_pos in ("edge", "center"):
            reward += s["anchor_disturbed_penalty"]

        # Board stuck (no score change = no merges happened)
        if score_delta == 0 and not is_fatal:
            self._stagnation_count += 1
            if self._stagnation_count >= s["stagnation_threshold_steps"]:
                reward += s["stagnation_penalty"]
        else:
            self._stagnation_count = 0

        return max(s["reward_min"], min(s["reward_max"], reward))

    # ── Pokemon Red ─────────────────────────────────────────

    def _extract_pokemon(self, state: str) -> dict:
        score = self._find_float(r"[Ss]core:?\s*(\d+\.?\d*)", state) or 0
        flags = self._find_int(r"[Ff]lags?:?\s*(\d+)", state) or 0
        map_name = self._find_str(r"[Mm]ap.?[Nn]ame:?\s*(\S+)", state) or ""
        return {"score": score, "flags": flags, "map_name": map_name}

    def _reward_pokemon(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping
        if is_fatal:
            return s["fatal_penalty"]

        reward = 0.0
        # Map transition reward — first visit gets `map_discovery_bonus`,
        # subsequent re-entries get `repeat_visit_bonus` (default 0).
        # PR #28 v6 fix: prior code always rewarded transitions, which the
        # agent exploited via the RedsHouse1f↔2f warp loop. Visited set
        # is reset per episode in reset_episode().
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

        # Flag collected (big bonus — these are the actual eval scoring units)
        flag_delta = cur.get("flags", 0) - prev.get("flags", 0)
        if flag_delta > 0:
            reward += s["flag_bonus"] * flag_delta

        # Score increase (intermediate)
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        if score_delta > 0:
            reward += score_delta / s["score_delta_divisor"]

        # Stuck in same area with no progress
        if score_delta == 0 and flag_delta == 0 and not map_changed:
            self._stagnation_count += 1
            if self._stagnation_count >= s["stagnation_threshold_steps"]:
                reward += s["stagnation_penalty"]
        else:
            self._stagnation_count = 0

        return max(s["reward_min"], min(s["reward_max"], reward))

    # ── Generic fallback ────────────────────────────────────

    def _extract_generic(self, state: str) -> dict:
        score = self._find_float(r"[Ss]core:?\s*(\d+\.?\d*)", state) or 0
        return {"score": score}

    def _reward_generic(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        if is_fatal:
            return -1.0
        if success:
            score_delta = cur.get("score", 0) - prev.get("score", 0)
            return max(0.5, min(3.0, score_delta / 100.0 + 0.5))
        return 0.0

    # ── Helpers ─────────────────────────────────────────────

    def _find_float(self, pattern: str, text: str) -> float | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    def _find_int(self, pattern: str, text: str) -> int | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _find_str(self, pattern: str, text: str) -> str | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1) if m else None
