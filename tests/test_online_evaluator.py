"""Tests for OnlineAgentEvaluator reward shaping.

PR #28 v6 introduced two changes covered here:
1. Pokemon map-transition reward only fires on first visit per episode.
2. Shaping params are overridable via the constructor's `shaping_overrides`.
"""
from agents.macla.online_evaluator import OnlineAgentEvaluator, DEFAULT_SHAPING


def _state(map_name: str, score: int = 0, flags: int = 0) -> str:
    return f"Map Name: {map_name},\nScore: {score}\nFlags: {flags}"


def test_pokemon_warp_loop_does_not_compound_reward():
    """Warping A→B→A→B should reward only the first visits, not every transition."""
    ev = OnlineAgentEvaluator(game_name="pokemon_red")

    # Bootstrap prev_metrics from initial map.
    ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse2f"), success=True, is_fatal=False)
    ev._step_rewards.clear()  # ignore the bootstrap step

    # Discover RedsHouse1f for the first time → +1.5.
    r1 = ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse1f"), True, False)
    # Warp back to RedsHouse2f — already visited → 0.
    r2 = ev.evaluate_step(_state("RedsHouse1f"), _state("RedsHouse2f"), True, False)
    # Warp back to RedsHouse1f — already visited → 0.
    r3 = ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse1f"), True, False)
    # Warp back to RedsHouse2f again — already visited → 0.
    r4 = ev.evaluate_step(_state("RedsHouse1f"), _state("RedsHouse2f"), True, False)

    # First map discovery rewards 1.5; subsequent re-entries get 0 (the
    # default repeat_visit_bonus). The warp-loop reward hack is gone.
    assert r1 >= 1.5, f"first visit to new map should reward >= 1.5, got {r1}"
    assert r2 < r1, f"re-visit should not match first-visit reward, got r2={r2} vs r1={r1}"
    assert r3 < r1, f"re-visit should not match first-visit reward, got r3={r3} vs r1={r1}"
    assert r4 < r1, f"re-visit should not match first-visit reward, got r4={r4} vs r1={r1}"


def test_pokemon_new_map_after_warp_loop_still_rewards():
    """Discovering a third map after a warp loop should still trigger discovery bonus."""
    ev = OnlineAgentEvaluator(game_name="pokemon_red")
    ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse2f"), True, False)

    ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse1f"), True, False)  # discover 1f
    ev.evaluate_step(_state("RedsHouse1f"), _state("RedsHouse2f"), True, False)  # back to 2f, no reward
    r_outside = ev.evaluate_step(_state("RedsHouse2f"), _state("PalletTown"), True, False)
    assert r_outside >= 1.5, f"discovering PalletTown should still reward >= 1.5, got {r_outside}"


def test_pokemon_reset_episode_clears_visited_maps():
    """A new episode should let the same map discovery reward fire again."""
    ev = OnlineAgentEvaluator(game_name="pokemon_red")
    ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse2f"), True, False)
    r_first = ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse1f"), True, False)

    ev.reset_episode()
    ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse2f"), True, False)
    r_after_reset = ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse1f"), True, False)
    assert r_after_reset == r_first, (
        "after reset_episode the same map should re-reward as a discovery, "
        f"got {r_after_reset} vs original {r_first}"
    )


def test_shaping_override_repeat_visit_bonus_restores_old_behavior():
    """Setting repeat_visit_bonus=1.5 reproduces the pre-fix warp-loop reward."""
    ev = OnlineAgentEvaluator(
        game_name="pokemon_red",
        shaping_overrides={"repeat_visit_bonus": 1.5},
    )
    ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse2f"), True, False)

    ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse1f"), True, False)  # discover
    r2 = ev.evaluate_step(_state("RedsHouse1f"), _state("RedsHouse2f"), True, False)
    r3 = ev.evaluate_step(_state("RedsHouse2f"), _state("RedsHouse1f"), True, False)
    assert r2 >= 1.5 and r3 >= 1.5, (
        f"with repeat_visit_bonus=1.5 every transition should reward >= 1.5, "
        f"got r2={r2}, r3={r3}"
    )


def test_shaping_override_partial_keeps_other_defaults():
    """Override one key — other shaping defaults should remain intact."""
    ev = OnlineAgentEvaluator(
        game_name="pokemon_red",
        shaping_overrides={"flag_bonus": 99.0},
    )
    assert ev._shaping["flag_bonus"] == 99.0
    assert ev._shaping["map_discovery_bonus"] == DEFAULT_SHAPING["pokemon_red"]["map_discovery_bonus"]


def test_unknown_game_with_overrides_does_not_crash():
    """A game without a DEFAULT_SHAPING entry can still receive overrides."""
    ev = OnlineAgentEvaluator(
        game_name="unknown_game",
        shaping_overrides={"reward_min": -10.0, "reward_max": 10.0},
    )
    assert ev._shaping == {"reward_min": -10.0, "reward_max": 10.0}
