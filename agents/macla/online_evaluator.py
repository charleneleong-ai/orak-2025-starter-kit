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
        # Try to extract board state
        board_str = state if "board" in state.lower() else ""
        return {"score": score, "max_tile": max_tile, "board": board_str}

    def _reward_2048(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        if is_fatal:
            return -1.0

        reward = 0.0
        # Score delta (main signal for 2048)
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        reward += score_delta / 500.0

        # Max tile increase (big bonus for doubling)
        prev_tile = prev.get("max_tile", 0)
        cur_tile = cur.get("max_tile", 0)
        if cur_tile > prev_tile and prev_tile > 0:
            reward += 1.0

        # Board stuck (no score change)
        if score_delta == 0 and not is_fatal:
            self._stagnation_count += 1
            if self._stagnation_count >= 3:
                reward -= 0.3
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
        # Flag collected (big bonus)
        flag_delta = cur.get("flags", 0) - prev.get("flags", 0)
        if flag_delta > 0:
            reward += 3.0 * flag_delta

        # Score increase
        score_delta = cur.get("score", 0) - prev.get("score", 0)
        if score_delta > 0:
            reward += score_delta / 2.0

        # New area entered
        if cur.get("map_name") and cur["map_name"] != prev.get("map_name", ""):
            reward += 1.0

        # Stuck in same area with no progress
        if score_delta == 0 and flag_delta == 0 and cur.get("map_name") == prev.get("map_name"):
            self._stagnation_count += 1
            if self._stagnation_count >= 5:
                reward -= 0.2
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
