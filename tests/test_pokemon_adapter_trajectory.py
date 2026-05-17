"""Tests for the pokemon_red adapter's TRAJECTORY_* introspection constants."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("autoresearch.trajectory") is None:
    pytest.skip("autoresearch.trajectory not available", allow_module_level=True)

from agents.pokemon_red.game_adapter import (
    TRAJECTORY_ACTION_SPEC,
    TRAJECTORY_DWELL_SPECS,
    TRAJECTORY_MILESTONES,
    TRAJECTORY_SCORE_EXTRACTOR,
    TRAJECTORY_SCORE_MAX,
    TRAJECTORY_ZONE_EXTRACTOR,
)


def _row(action: str = "", score: int = 0, map_name: str = "PalletTown") -> dict:
    return {
        "action": action,
        "obs": {
            "game_info": {
                "score": str(score),
                "map_name": map_name,
                "evaluation_score": str(score),
            }
        },
    }


class TestScoreExtractor:
    def test_returns_int_score_from_game_info(self):
        assert TRAJECTORY_SCORE_EXTRACTOR(_row(score=4)) == 4.0

    def test_returns_zero_when_score_missing(self):
        assert TRAJECTORY_SCORE_EXTRACTOR({}) == 0.0

    def test_returns_zero_when_score_non_numeric(self):
        row = {"obs": {"game_info": {"score": "?"}}}
        assert TRAJECTORY_SCORE_EXTRACTOR(row) == 0.0

    def test_score_max_is_seven(self):
        assert TRAJECTORY_SCORE_MAX == 7.0


class TestZoneExtractor:
    def test_returns_map_name(self):
        assert TRAJECTORY_ZONE_EXTRACTOR(_row(map_name="Route1")) == "Route1"

    def test_returns_question_mark_when_missing(self):
        assert TRAJECTORY_ZONE_EXTRACTOR({}) == "?"

    def test_returns_question_mark_when_none(self):
        row = {"obs": {"game_info": {"map_name": None}}}
        assert TRAJECTORY_ZONE_EXTRACTOR(row) == "?"


class TestMoveTargetExtractor:
    def test_extracts_xy_from_actual_action_syntax(self):
        """Real game_states.jsonl rows use the x_dest=/y_dest= keyword form."""
        row = _row(action="use_tool(move_to, (x_dest=6, y_dest=4))")
        assert TRAJECTORY_ACTION_SPEC.extract_target(row) == (6, 4)

    def test_handles_negative_coords(self):
        row = _row(action="use_tool(move_to, (x_dest=-1, y_dest=-2))")
        assert TRAJECTORY_ACTION_SPEC.extract_target(row) == (-1, -2)

    def test_returns_none_for_non_move_action(self):
        assert TRAJECTORY_ACTION_SPEC.extract_target(_row(action="press_a")) is None

    def test_returns_none_for_empty_action(self):
        assert TRAJECTORY_ACTION_SPEC.extract_target(_row(action="")) is None


class TestMilestones:
    def test_seven_milestones(self):
        assert [m.name for m in TRAJECTORY_MILESTONES] == [f"M{i}" for i in range(1, 8)]

    def test_milestone_predicates_threshold_correctly(self):
        row3 = _row(score=3)
        fired = {m.name: m.predicate(row3) for m in TRAJECTORY_MILESTONES}
        assert fired == {
            "M1": True,
            "M2": True,
            "M3": True,
            "M4": False,
            "M5": False,
            "M6": False,
            "M7": False,
        }


class TestDwellSpecs:
    def test_route1_dwell_fires_on_route1(self):
        spec = next(s for s in TRAJECTORY_DWELL_SPECS if s.name == "Route1")
        assert spec.predicate(_row(map_name="Route1")) is True
        assert spec.predicate(_row(map_name="PalletTown")) is False

    def test_viridian_dwell_fires_on_any_viridian_map(self):
        spec = next(s for s in TRAJECTORY_DWELL_SPECS if s.name == "Viridian")
        assert spec.predicate(_row(map_name="ViridianCity")) is True
        assert spec.predicate(_row(map_name="ViridianPokecenter")) is True
        assert spec.predicate(_row(map_name="Route1")) is False
