"""Hierarchical subgoal stack + Reflexion-loop summary.

Covers three pieces:

1. A persistent subgoal stack on the memory system. Each subgoal carries
   a completion predicate (Callable[[dict], bool]). Per step the top
   subgoal's predicate is checked; if true, it pops. Stack persists via
   pickle across iters.

2. Per-game adapter SUBGOAL_TEMPLATES dict — pokemon ships
   NavigateToMap / TalkTo / DefeatTrainer with their completion
   predicates. Mario / 2048 opt out via absence.

3. Reflexion summary at iter start (after checkpoint load): build a
   5-line summary of the prior iter from autoresearch.trajectory
   IterMetrics and prepend to the planner's prompt — "Iter N-1 you hit
   M2 at step 49 then stalled in PalletTown for 250 steps."
"""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path

import pytest

from agents.macla.macla_lib import (
    EnhancedHierarchicalMemorySystem,
    Subgoal,
)


def _map_matches(target_map: str, obs: dict) -> bool:
    """Top-level picklable predicate (lambdas/closures don't pickle)."""
    return obs.get("map_name") == target_map


def _navigate_to(target_map: str) -> Subgoal:
    return Subgoal(
        name=f"NavigateToMap({target_map})",
        description=f"Walk until the current map is {target_map}.",
        completion=partial(_map_matches, target_map),
        suggested_tools=["move_to"],
    )


class TestSubgoalDataclass:
    def test_required_fields(self):
        s = Subgoal(
            name="N",
            description="D",
            completion=lambda _: False,
        )
        assert s.name == "N"
        assert s.description == "D"
        assert s.completion({}) is False
        assert s.suggested_tools == []

    def test_picklable(self):
        """Stack persists via the existing checkpoint pickle path —
        Subgoal must round-trip cleanly (callables included)."""
        s = _navigate_to("Route1")
        round_tripped = pickle.loads(pickle.dumps(s))
        assert round_tripped.name == s.name
        assert round_tripped.completion({"map_name": "Route1"}) is True
        assert round_tripped.completion({"map_name": "PalletTown"}) is False


class TestSubgoalStack:
    def test_stack_initially_empty(self):
        mem = EnhancedHierarchicalMemorySystem()
        assert mem.peek_subgoal() is None
        assert mem.subgoal_depth() == 0

    def test_push_and_peek(self):
        mem = EnhancedHierarchicalMemorySystem()
        s = _navigate_to("Route1")
        mem.push_subgoal(s)
        assert mem.peek_subgoal() is s
        assert mem.subgoal_depth() == 1

    def test_push_multiple_lifo_ordering(self):
        mem = EnhancedHierarchicalMemorySystem()
        a = _navigate_to("Route1")
        b = _navigate_to("ViridianCity")
        mem.push_subgoal(a)
        mem.push_subgoal(b)
        # LIFO — top of stack is the most recently pushed
        assert mem.peek_subgoal() is b
        assert mem.subgoal_depth() == 2

    def test_pop_returns_and_removes(self):
        mem = EnhancedHierarchicalMemorySystem()
        s = _navigate_to("Route1")
        mem.push_subgoal(s)
        popped = mem.pop_subgoal()
        assert popped is s
        assert mem.peek_subgoal() is None

    def test_pop_empty_returns_none(self):
        mem = EnhancedHierarchicalMemorySystem()
        assert mem.pop_subgoal() is None

    def test_extend_replaces_stack(self):
        """When the planner emits a fresh stack at iter start it should
        replace, not augment, the existing stack."""
        mem = EnhancedHierarchicalMemorySystem()
        mem.push_subgoal(_navigate_to("Old"))
        mem.set_subgoal_stack([_navigate_to("New1"), _navigate_to("New2")])
        assert mem.subgoal_depth() == 2
        # First element of list is the bottom (executed last); top is "New2"
        assert mem.peek_subgoal().name == "NavigateToMap(New2)"


class TestSubgoalCompletion:
    def test_check_completion_pops_when_predicate_true(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.push_subgoal(_navigate_to("Route1"))
        completed = mem.check_active_subgoal_completion({"map_name": "Route1"})
        assert completed is not None
        assert completed.name == "NavigateToMap(Route1)"
        assert mem.peek_subgoal() is None

    def test_check_completion_no_op_when_predicate_false(self):
        mem = EnhancedHierarchicalMemorySystem()
        s = _navigate_to("Route1")
        mem.push_subgoal(s)
        completed = mem.check_active_subgoal_completion({"map_name": "PalletTown"})
        assert completed is None
        assert mem.peek_subgoal() is s

    def test_check_completion_cascades_on_chain_completion(self):
        """When popping subgoal A reveals subgoal B and B is already done,
        B should pop too. Returns the deepest-popped subgoal."""
        mem = EnhancedHierarchicalMemorySystem()
        # Bottom: be in Route1. Top: be in Route1 (same — pops as cascade).
        mem.set_subgoal_stack([_navigate_to("Route1"), _navigate_to("Route1")])
        completed = mem.check_active_subgoal_completion({"map_name": "Route1"})
        # Both should pop
        assert completed is not None
        assert mem.subgoal_depth() == 0

    def test_check_completion_no_op_on_empty_stack(self):
        mem = EnhancedHierarchicalMemorySystem()
        assert mem.check_active_subgoal_completion({"map_name": "Route1"}) is None


class TestSubgoalStackPersistence:
    def test_stack_survives_pickle(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.push_subgoal(_navigate_to("Route1"))
        mem.push_subgoal(_navigate_to("ViridianCity"))
        restored = pickle.loads(pickle.dumps(mem))
        assert restored.subgoal_depth() == 2
        assert restored.peek_subgoal().name == "NavigateToMap(ViridianCity)"
        # Predicates round-trip and still fire
        completed = restored.check_active_subgoal_completion({"map_name": "ViridianCity"})
        assert completed is not None
        assert restored.subgoal_depth() == 1


class TestPokemonAdapterSubgoalTemplates:
    """Per-game adapter ships templates; mario/2048 opt out via absence."""

    def test_pokemon_adapter_exposes_templates(self):
        from agents.pokemon_red.game_adapter import SUBGOAL_TEMPLATES

        assert "NavigateToMap" in SUBGOAL_TEMPLATES
        assert "TalkTo" in SUBGOAL_TEMPLATES

    def test_navigate_to_map_template_builds_subgoal(self):
        from agents.pokemon_red.game_adapter import SUBGOAL_TEMPLATES

        builder = SUBGOAL_TEMPLATES["NavigateToMap"]
        sg = builder("ViridianCity")
        assert "ViridianCity" in sg.name
        # Fires when obs has matching map_name
        assert sg.completion({"map_name": "ViridianCity"}) is True
        assert sg.completion({"map_name": "PalletTown"}) is False

    def test_talk_to_template_fires_on_dialog_keyword(self):
        from agents.pokemon_red.game_adapter import SUBGOAL_TEMPLATES

        builder = SUBGOAL_TEMPLATES["TalkTo"]
        sg = builder("OAK")
        # Completion: obs shows the NPC's name in recent dialog text
        assert sg.completion({"recent_dialog": "OAK: Hello!"}) is True
        assert sg.completion({"recent_dialog": ""}) is False


class TestReflexionSummary:
    """Builds a per-iter introspect summary string for the planner prompt."""

    def test_summary_includes_score_and_milestones(self, tmp_path: Path):
        from agents.macla.reflexion import build_reflexion_summary

        run_dir = tmp_path / "iter_prev"
        run_dir.mkdir()
        # Minimal game_states.jsonl with 3 rows mirroring the real schema
        gs = run_dir / "game_states.jsonl"
        rows = [
            '{"iteration":1,"action":"a","obs":{"game_info":{"score":"0","map_name":"PalletTown","evaluation_score":"0.0"}}}',
            '{"iteration":2,"action":"a","obs":{"game_info":{"score":"1","map_name":"PalletTown","evaluation_score":"14.3"}}}',
            '{"iteration":3,"action":"a","obs":{"game_info":{"score":"2","map_name":"PalletTown","evaluation_score":"28.6"}}}',
        ]
        gs.write_text("\n".join(rows) + "\n")

        from agents.pokemon_red import game_adapter

        summary = build_reflexion_summary(run_dir, adapter=game_adapter)
        # Summary should mention final score, hit milestones, and final zone
        assert "M2" in summary  # 2/7 reached
        assert "PalletTown" in summary
        # Self-critique prompt phrase ("hypothesise why" or similar) lets the
        # planner know this is feedback, not a fact list
        assert "previous iter" in summary.lower() or "prior iter" in summary.lower()

    def test_summary_no_op_when_run_dir_missing(self, tmp_path: Path):
        from agents.macla.reflexion import build_reflexion_summary
        from agents.pokemon_red import game_adapter

        summary = build_reflexion_summary(tmp_path / "does_not_exist", adapter=game_adapter)
        assert summary == ""

    def test_summary_no_op_when_adapter_lacks_trajectory_constants(self, tmp_path: Path):
        from agents.macla.reflexion import build_reflexion_summary

        run_dir = tmp_path / "iter_prev"
        run_dir.mkdir()
        (run_dir / "game_states.jsonl").write_text('{"iteration":1}\n')

        class StubAdapter:
            pass  # no TRAJECTORY_* exports

        assert build_reflexion_summary(run_dir, adapter=StubAdapter()) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
