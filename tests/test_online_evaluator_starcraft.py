"""Tests for StarCraftShaper — per-step reward shaping for the SC2 adapter.

Fixtures lift canonical obs_str snippets from a real PR3 smoke run
(stagnation_pr3_star_craft_smoke_20260527T094639Z) at iterations 1 / 51 / 201,
covering empty-init / productive-state / floated-supply-blocked respectively.
"""

import pytest

from agents.macla.online_evaluator import (
    DEFAULT_SHAPING,
    SHAPERS,
    OnlineAgentEvaluator,
    StarCraftShaper,
)

# ── Canonical obs strings lifted from real smoke iterations ─────────────────


# Module-scoped: used across TestExtractMetrics (added in Task 2) and
# TestComputeReward; pure data, never mutated.
@pytest.fixture(scope="module")
def obs_strings() -> dict[str, str]:
    return {
        "iter_1_empty": "",
        "iter_51_productive": (
            "Summary 1: At 01:29 game time, our current StarCraft II situation is as follows:\n"
            "\n"
            "Resources:\n"
            "- Game time: 01:29\n"
            "- Worker supply: 20\n"
            "- Mineral: 515\n"
            "- Supply left: 2\n"
            "- Supply cap: 23\n"
            "- Supply used: 21\n"
            "\n"
            "Buildings:\n"
            "- Nexus count: 1\n"
            "- Pylon count: 1\n"
            "\n"
            "Units:\n"
            "- Probe count: 20\n"
            "\n"
            "In Progress:\n"
            "Building constructing:\n"
            "- Constructing gateway count: 1\n"
            "Unit producing:\n"
            "- Producing probe count: 1\n"
        ),
        "iter_201_floated": (
            "Summary 1: At 05:56 game time, our current StarCraft II situation is as follows:\n"
            "\n"
            "Resources:\n"
            "- Game time: 05:56\n"
            "- Worker supply: 23\n"
            "- Mineral: 3980\n"
            "- Supply left: -15\n"
            "- Supply cap: 8\n"
            "- Supply used: 23\n"
            "\n"
            "Buildings:\n"
            "- Pylon count: 1\n"
            "- Gateway count: 2\n"
            "\n"
            "Units:\n"
            "- Probe count: 23\n"
            "\n"
            "Enemy:\n"
            "\n"
            "Unit:\n"
            "- Enemy unittypeid.zergling: 3\n"
            "- Enemy unittypeid.ravager: 3\n"
            "- Enemy unittypeid.roach: 4\n"
        ),
    }


@pytest.fixture
def shaper() -> StarCraftShaper:
    return StarCraftShaper(DEFAULT_SHAPING["star_craft"])


class TestRegistry:
    def test_star_craft_registered_in_SHAPERS(self):
        assert SHAPERS["star_craft"] is StarCraftShaper

    def test_default_shaping_has_star_craft_entry(self):
        assert "star_craft" in DEFAULT_SHAPING

    def test_evaluator_routes_star_craft_to_shaper(self):
        ev = OnlineAgentEvaluator("star_craft")
        assert isinstance(ev._shaper, StarCraftShaper)


@pytest.mark.parametrize(
    "key",
    [
        "reward_min",
        "reward_max",
        "fatal_penalty",
        "victory_bonus",
        "supply_used_weight",
        "building_built_weight",
        "floated_minerals_penalty",
        "supply_block_penalty",
        "first_enemy_bonus",
        "survival_increment",
    ],
)
def test_default_shaping_star_craft_has_key(key):
    assert key in DEFAULT_SHAPING["star_craft"], f"missing DEFAULT_SHAPING['star_craft']['{key}']"


class TestExtractMetrics:
    def test_empty_state_returns_zero_defaults(self, shaper, obs_strings):
        m = shaper.extract_metrics(obs_strings["iter_1_empty"])
        assert m["game_time_sec"] == 0
        assert m["mineral"] == 0
        assert m["supply_used"] == 0
        assert m["supply_cap"] == 0
        assert m["supply_left"] == 0
        assert m["worker_supply"] == 0
        assert m["building_count"] == 0
        assert m["enemy_unit_count"] == 0

    def test_productive_state_extracts_all_fields(self, shaper, obs_strings):
        m = shaper.extract_metrics(obs_strings["iter_51_productive"])
        assert m["game_time_sec"] == 89  # 01:29 = 89s
        assert m["mineral"] == 515
        assert m["supply_used"] == 21
        assert m["supply_cap"] == 23
        assert m["supply_left"] == 2
        assert m["worker_supply"] == 20
        # building_count: Nexus(1) + Pylon(1) = 2  (Probe/Constructing/Producing excluded)
        assert m["building_count"] == 2
        assert m["enemy_unit_count"] == 0

    def test_floated_state_extracts_negative_supply_left(self, shaper, obs_strings):
        m = shaper.extract_metrics(obs_strings["iter_201_floated"])
        assert m["game_time_sec"] == 356  # 05:56 = 356s
        assert m["mineral"] == 3980
        assert m["supply_left"] == -15
        # building_count: Pylon(1) + Gateway(2) = 3
        assert m["building_count"] == 3
        # enemy_unit_count: zergling(3) + ravager(3) + roach(4) = 10
        assert m["enemy_unit_count"] == 10

    @pytest.mark.parametrize(
        "field,patched,expected_key,expected_value",
        [
            ("Game time: 01:29", "Game time: 12:34", "game_time_sec", 754),
            ("Mineral: 515", "Mineral: 9999", "mineral", 9999),
            ("Supply left: 2", "Supply left: -1", "supply_left", -1),
        ],
    )
    def test_extract_handles_value_variations(
        self, shaper, obs_strings, field, patched, expected_key, expected_value
    ):
        state = obs_strings["iter_51_productive"].replace(field, patched)
        m = shaper.extract_metrics(state)
        assert m[expected_key] == expected_value


class TestTerminal:
    def test_is_fatal_returns_fatal_penalty(self, shaper):
        r = shaper.compute_reward(prev={}, cur={}, success=False, is_fatal=True)
        assert r == DEFAULT_SHAPING["star_craft"]["fatal_penalty"]

    def test_success_returns_victory_bonus(self, shaper):
        r = shaper.compute_reward(prev={}, cur={}, success=True, is_fatal=False)
        assert r == DEFAULT_SHAPING["star_craft"]["victory_bonus"]

    def test_is_fatal_takes_precedence_over_success(self, shaper):
        # Defensive: if both flags somehow set, defeat wins (no false positives).
        r = shaper.compute_reward(prev={}, cur={}, success=True, is_fatal=True)
        assert r == DEFAULT_SHAPING["star_craft"]["fatal_penalty"]


class TestPositiveDeltas:
    def _metrics(self, **overrides):
        """Helper: build a fully-populated metrics dict with overrides."""
        base = {
            "game_time_sec": 100,
            "mineral": 500,
            "supply_used": 20,
            "supply_cap": 23,
            "supply_left": 3,
            "worker_supply": 18,
            "building_count": 2,
            "enemy_unit_count": 0,
        }
        return {**base, **overrides}

    def test_supply_used_delta_rewards_unit_built(self, shaper):
        prev = self._metrics(supply_used=20)
        cur = self._metrics(supply_used=22, game_time_sec=110)  # 2 supply built + time advanced
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # 2 * 0.2 (army) + 0.05 (survival) = 0.45
        assert r == pytest.approx(0.45)

    def test_building_count_delta_rewards_structure_built(self, shaper):
        prev = self._metrics(building_count=2)
        cur = self._metrics(building_count=3, game_time_sec=110)  # +1 structure, time advanced
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # 1 * 0.5 (building) + 0.05 (survival) = 0.55
        assert r == pytest.approx(0.55)

    def test_survival_increment_when_only_time_advances(self, shaper):
        prev = self._metrics(game_time_sec=100)
        cur = self._metrics(game_time_sec=110)
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # Just survival baseline.
        assert r == pytest.approx(0.05)

    def test_no_survival_increment_if_time_did_not_advance(self, shaper):
        prev = self._metrics(game_time_sec=100)
        cur = self._metrics(game_time_sec=100)
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        assert r == pytest.approx(0.0)
