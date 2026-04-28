"""
OnlineAgentEvaluator: per-step reward shaping for MACLA.

Replaces binary success/fail with continuous rewards based on
game-specific score deltas, position progress, and metric changes.
"""
import re
from collections import deque

from loguru import logger


class OnlineAgentEvaluator:
    """Computes continuous per-step rewards from game state deltas."""

    def __init__(self, game_name: str):
        self._game_name = game_name
        self._prev_metrics: dict = {}
        self._step_rewards: deque = deque(maxlen=100)
        self._stagnation_count: int = 0

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
        if is_fatal:
            return -2.0

        reward = 0.0
        # Position progress (main signal)
        x_delta = cur.get("x_pos", 0) - prev.get("x_pos", 0)
        reward += x_delta / 100.0  # ~0.0 to 2.0 for good progress

        # Score delta
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        if score_delta > 0:
            reward += score_delta / 200.0

        # Lives lost
        if prev.get("lives") is not None and cur.get("lives") is not None:
            if cur["lives"] < prev["lives"]:
                reward -= 1.5

        # Stagnation penalty
        if abs(x_delta) < 3:
            self._stagnation_count += 1
            if self._stagnation_count >= 3:
                reward -= 0.3
        else:
            self._stagnation_count = 0

        return max(-2.0, min(3.0, reward))

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
        if is_fatal:
            return -1.5

        reward = 0.0
        # Score delta (main signal — reward merges proportionally)
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        reward += score_delta / 200.0  # More sensitive (was /500)

        # Max tile increase (big bonus for doubling)
        prev_tile = prev.get("max_tile", 0)
        cur_tile = cur.get("max_tile", 0)
        if cur_tile > prev_tile and prev_tile > 0:
            reward += 1.5  # Strong signal for tile doubling

        # Empty cells: reward freeing space, penalize filling board
        prev_empty = prev.get("empty_cells", 0)
        cur_empty = cur.get("empty_cells", 0)
        if cur_empty > prev_empty:
            reward += 0.3  # Freed a cell via merge
        elif cur_empty < prev_empty - 1:
            reward -= 0.2  # Board getting crowded

        # Corner-anchoring (the dominant 2048 strategy). Densifies the strategic
        # signal so procedures get update events for "max-tile in corner →
        # GOOD" and "anchor disturbed → BAD" without needing terminal game-over
        # feedback. Doesn't bias toward any specific corner (all 4 are
        # symmetric); it's the *invariant* that's rewarded.
        prev_pos = prev.get("max_pos")
        cur_pos = cur.get("max_pos")
        if cur_pos == "corner":
            reward += 0.4  # max-tile is anchored — keep it there
        if prev_pos == "corner" and cur_pos in ("edge", "center"):
            reward -= 0.5  # anchor was disturbed — strong negative signal

        # Board stuck (no score change = no merges happened)
        if score_delta == 0 and not is_fatal:
            self._stagnation_count += 1
            if self._stagnation_count >= 3:
                reward -= 0.5  # Stronger stagnation penalty
        else:
            self._stagnation_count = 0

        return max(-2.0, min(3.0, reward))

    # ── Pokemon Red ─────────────────────────────────────────

    def _extract_pokemon(self, state: str) -> dict:
        score = self._find_float(r"[Ss]core:?\s*(\d+\.?\d*)", state) or 0
        flags = self._find_int(r"[Ff]lags?:?\s*(\d+)", state) or 0
        map_name = self._find_str(r"[Mm]ap.?[Nn]ame:?\s*(\S+)", state) or ""
        return {"score": score, "flags": flags, "map_name": map_name}

    def _reward_pokemon(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        if is_fatal:
            return -1.5

        reward = 0.0
        # Map transition bonus — reward moving between rooms/maps. This is the
        # primary navigation signal before any flags are earned. Pokemon's main
        # blocker in the PR #20 / #22 sweeps was getting stuck on the starting
        # map, so the map-change reward is the densest progress signal we have.
        prev_map = prev.get("map_name", "")
        cur_map = cur.get("map_name", "")
        map_changed = bool(cur_map and prev_map and cur_map != prev_map)
        if map_changed:
            reward += 1.5  # consolidated (was +0.8 + duplicate +1.0 below)

        # Flag collected (big bonus — these are the actual eval scoring units)
        flag_delta = cur.get("flags", 0) - prev.get("flags", 0)
        if flag_delta > 0:
            reward += 3.0 * flag_delta

        # Score increase (intermediate)
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        if score_delta > 0:
            reward += score_delta / 2.0

        # Stuck in same area with no progress — tightened from 5 → 3 turns and
        # penalty from -0.2 → -0.4. Pokemon iters in the prior sweep showed
        # 50+ steps with the agent mashing the same direction without leaving
        # the starting map; the prior penalty was too weak to discourage it.
        if score_delta == 0 and flag_delta == 0 and not map_changed:
            self._stagnation_count += 1
            if self._stagnation_count >= 3:
                reward -= 0.4
        else:
            self._stagnation_count = 0

        return max(-2.0, min(3.0, reward))

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
