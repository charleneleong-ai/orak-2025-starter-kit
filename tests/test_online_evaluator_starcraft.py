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
