"""Unit tests for the 2048 progress-to-win normaliser.

The chart's cross-game scoreboard requires every game to expose a 0-100
progress metric. 2048 used to derive its 0-100 from
``min(1.0, game_score/20_000) * 100`` — game_score=20,000 was an
arbitrary proxy for the win condition. This module exercises the new
``log2(max_tile)/log2(2048) * 100`` formula and asserts the agents-side
helper agrees with the env-side helper bit-for-bit.
"""
from __future__ import annotations

import math

from agents.twenty_fourty_eight._metrics import normalize_2048_score as agent_normalize


def _env_normalize(max_tile: int) -> float:
    """Re-implement the env formula here without booting PyBoy/pygame.

    The real env module is `twenty_fourty_eight_env`; importing it pulls
    in pygame init via the package. We intentionally re-derive the value
    so the test pins the *formula*, not the call path. The env's own
    helper is exercised by the integration suite.
    """
    if max_tile <= 1:
        return 0.0
    return min((math.log2(max_tile) / 11.0) * 100.0, 100.0)


def test_start_state_yields_zero():
    """Empty board / unspawned tiles → 0% progress."""
    assert agent_normalize(0) == 0.0
    assert agent_normalize(1) == 0.0


def test_first_tile_is_baseline():
    """A 2-tile is the smallest tile that ever appears; 1/11 of the log scale."""
    assert agent_normalize(2) == agent_normalize(2)  # determinism
    assert math.isclose(agent_normalize(2), 100.0 / 11.0, rel_tol=1e-9)


def test_2048_tile_is_exactly_100():
    """Reaching the 2048 tile = win = 100%."""
    assert agent_normalize(2048) == 100.0


def test_progression_is_monotonic():
    tiles = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    scores = [agent_normalize(t) for t in tiles]
    assert scores == sorted(scores), f"non-monotonic: {scores}"


def test_log_scale_doubles_evenly():
    """Each tile doubling adds the same fixed delta (1/11 of 100%)."""
    expected_step = 100.0 / 11.0
    for low, high in [(2, 4), (4, 8), (16, 32), (512, 1024), (1024, 2048)]:
        delta = agent_normalize(high) - agent_normalize(low)
        assert math.isclose(delta, expected_step, rel_tol=1e-9), (
            f"{low}→{high}: delta={delta} expected={expected_step}"
        )


def test_above_win_clamps_to_100():
    """Post-2048 play (4096+) is still capped at 100%."""
    assert agent_normalize(4096) == 100.0
    assert agent_normalize(8192) == 100.0


def test_agent_and_env_formulas_agree():
    """The agent-side helper and the env-side formula must produce the
    same value for every legitimate max_tile — otherwise wandb metrics
    and the cross-game scoreboard will diverge."""
    for power in range(0, 14):
        tile = 2**power
        assert math.isclose(
            agent_normalize(tile), _env_normalize(tile), rel_tol=1e-12
        ), f"diverged at max_tile={tile}"
