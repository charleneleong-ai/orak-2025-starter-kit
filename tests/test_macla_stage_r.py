"""Stage R end-to-end: planner-side soft phrasing + memory-side counters
+ pokemon subgoal ladder.

Consolidates per-feature blocks for v3 (F1/F2) and v4 (0/1/4/5/6).
Base Subgoal dataclass + stack mechanics live in ``test_macla_subgoals``.

  v3 (F1) — soft "Currently pursuing" planner block (no HARD CONSTRAINT)
  v3 (F2) — stagnation counter + escape valve (drops subgoal at 30 steps)
  v4 (0) — unified.py uses adapter.graph_hint (not the hand-authored path)
  v4 (1) — anti-perseveration: position visit counter + looped hint
  v4 (4) — __setstate__ zeros per-episode counters on checkpoint load
  v4 (5) — record_episode_end writes mem.last_iter_score (raw, no norm)
  v4 (6) — pokemon initial_subgoal_stack extends to M5-M7 via the
           generic ``build_score_milestone_stack`` framework helper
"""

from __future__ import annotations

import inspect
import pickle

import pytest

from agents._cognitive.subtask_planner import (
    DEFAULT_SYSTEM_PROMPT,
    _render_active_subgoal_block,
)
from agents.macla import macla_lib, unified
from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem, Subgoal
from agents.pokemon_red.game_adapter import initial_subgoal_stack


def _never(_obs: dict) -> bool:
    """Module-level completion predicate; picklable (unlike a lambda)."""
    return False


def _sub(name: str) -> Subgoal:
    return Subgoal(name=name, description=f"desc_{name}", completion=_never)


@pytest.fixture
def mem() -> EnhancedHierarchicalMemorySystem:
    return EnhancedHierarchicalMemorySystem()


@pytest.fixture(scope="module")
def unified_src() -> str:
    return inspect.getsource(unified)


# ── v3 (F1): soft planner-prompt phrasing ───────────────────────────


class TestSoftPhrasing:
    """v2's HARD CONSTRAINT phrasing → v3 soft "Currently pursuing / prefer"."""

    def test_user_block_uses_soft_phrasing_only(self):
        block = _render_active_subgoal_block("NavigateToMap(Route1): walk north")
        assert "HARD CONSTRAINT" not in block and "MUST" not in block
        assert "Currently pursuing" in block and "NavigateToMap(Route1)" in block
        assert "prefer" in block.lower()
        assert "blocked" in block.lower() or "evidence" in block.lower()

    @pytest.mark.parametrize("empty", [None, ""])
    def test_user_block_empty_passthrough(self, empty):
        assert _render_active_subgoal_block(empty) == ""

    def test_system_prompt_no_hard_constraint_lockout(self):
        txt = DEFAULT_SYSTEM_PROMPT
        assert "HARD CONSTRAINT" not in txt
        assert "MUST be a concrete step that advances" not in txt
        assert "overrides every heuristic below" not in txt
        assert "active subgoal" in txt.lower() and "prefer" in txt.lower()


# ── v3 (F2): stagnation counter + escape-valve wiring ───────────────


class TestStagnationCounter:
    def test_increments_when_top_unchanged(self, mem):
        assert mem.subgoal_stagnation_steps == 0  # initial-state invariant
        mem.push_subgoal(_sub("A"))
        for _ in range(3):
            mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 3

    def test_resets_when_top_changes_via_push(self, mem):
        mem.push_subgoal(_sub("A"))
        mem.record_subgoal_step()
        mem.record_subgoal_step()
        mem.push_subgoal(_sub("B"))
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 1

    def test_resets_when_top_changes_via_pop(self, mem):
        mem.push_subgoal(_sub("A"))
        mem.push_subgoal(_sub("B"))
        for _ in range(2):
            mem.record_subgoal_step()
        mem.pop_subgoal()
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 1

    def test_goes_zero_when_stack_drained(self, mem):
        mem.push_subgoal(_sub("A"))
        mem.record_subgoal_step()
        mem.pop_subgoal()
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 0

    def test_resets_when_set_subgoal_stack_replaces_top(self, mem):
        mem.push_subgoal(_sub("A"))
        for _ in range(2):
            mem.record_subgoal_step()
        mem.set_subgoal_stack([_sub("X"), _sub("Y")])
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 1


class TestEscapeValveWiring:
    def test_threshold_is_30(self):
        from agents.macla.unified import SUBGOAL_STAGNATION_THRESHOLD

        assert SUBGOAL_STAGNATION_THRESHOLD == 30

    @pytest.mark.parametrize(
        "needle",
        [
            "record_subgoal_step()",
            "subgoal_stagnation_steps",
            "SUBGOAL_STAGNATION_THRESHOLD",
        ],
    )
    def test_unified_threads_escape_valve(self, unified_src, needle):
        assert needle in unified_src


# ── v4 (0): adapter graph_hint dispatch ─────────────────────────────


class TestAdapterGraphHint:
    """Hand-authored MAP_GRAPH + map_graph_hint deleted; unified.py uses
    ``getattr(self._adapter, "graph_hint", None)(map, mem.visited_maps)``."""

    def test_hand_authored_artefacts_deleted(self):
        assert getattr(macla_lib, "MAP_GRAPH", None) is None
        assert not hasattr(macla_lib.EnhancedHierarchicalMemorySystem, "map_graph_hint")

    def test_unified_uses_adapter_dispatch(self, unified_src):
        # Adapter-optional dispatch + the signature swap (pass visited_maps in).
        assert 'getattr(self._adapter, "graph_hint"' in unified_src
        assert "mem.visited_maps" in unified_src
        # Old hand-authored call is fully deleted, not left alongside.
        assert "mem.map_graph_hint" not in unified_src


# ── v4 (1): anti-perseveration position counter ─────────────────────


class TestAntiPerseveration:
    """Memory tracks (map, x, y) visits; unified.py prepends a hint when ≥5."""

    def test_record_position_increments(self, mem):
        mem.record_position("PalletTown", 7, 10)
        mem.record_position("PalletTown", 7, 10)
        mem.record_position("OaksLab", 6, 4)
        assert mem.position_visits[("PalletTown", 7, 10)] == 2
        assert mem.position_visits[("OaksLab", 6, 4)] == 1

    @pytest.mark.parametrize("bad_map", [None, "unknown", ""])
    def test_record_position_ignores_unknown_map(self, mem, bad_map):
        mem.record_position(bad_map, 1, 1)
        assert sum(mem.position_visits.values()) == 0

    def test_hint_none_below_threshold(self, mem):
        for _ in range(4):
            mem.record_position("PalletTown", 7, 10)
        assert mem.looped_positions_hint() is None  # default threshold=5

    def test_hint_lists_offenders_ordered_by_visit_count(self, mem):
        for _ in range(20):
            mem.record_position("PalletTown", 12, 12)
        for _ in range(44):
            mem.record_position("PalletTown", 7, 10)
        for _ in range(8):
            mem.record_position("OaksLab", 6, 4)
        for _ in range(3):
            mem.record_position("PalletTown", 1, 1)  # below threshold

        hint = mem.looped_positions_hint()
        assert hint is not None
        assert "### Recently looped" in hint
        assert "PalletTown(7, 10): visited 44" in hint
        assert "PalletTown(1, 1)" not in hint  # below threshold

        # Most-visited first so the planner sees the worst loops at the top.
        lines = hint.splitlines()
        order = [
            next(i for i, line in enumerate(lines) if frag in line)
            for frag in ("(7, 10)", "(12, 12)", "(6, 4)")
        ]
        assert order == sorted(order)

    def test_hint_caps_display_count(self, mem):
        for i in range(10):
            for _ in range(5 + i):
                mem.record_position("Map", i, 0)
        hint = mem.looped_positions_hint(threshold=5, max_display=5)
        assert hint.count("\n- ") == 5

    @pytest.mark.parametrize(
        "needle",
        ["mem.record_position(", "looped_positions_hint"],
    )
    def test_unified_wires_perseveration(self, unified_src, needle):
        assert needle in unified_src


# ── v4 (4): __setstate__ resets per-episode state ───────────────────


class TestSetstateReset:
    """Per-episode counters (stagnation + position visits) zero on unpickle;
    cumulative state (subgoal stack, procedural memory) survives."""

    def test_per_episode_counters_zero_on_unpickle(self, mem):
        mem.push_subgoal(_sub("ViridianCity"))
        mem.push_subgoal(_sub("Route1"))
        for _ in range(35):
            mem.record_subgoal_step()
        for _ in range(20):
            mem.record_position("PalletTown", 7, 10)

        loaded = pickle.loads(pickle.dumps(mem))

        # Per-episode signals reset.
        assert loaded.subgoal_stagnation_steps == 0
        assert loaded._subgoal_stagnation_key is None
        assert sum(loaded.position_visits.values()) == 0
        # Cumulative state preserved.
        assert loaded.subgoal_depth() == 2
        assert loaded.peek_subgoal().name == "Route1"
        # And the freshly-loaded counter climbs again on the next step.
        loaded.record_subgoal_step()
        assert loaded.subgoal_stagnation_steps == 1


# ── v4 (5): perf-prune write site ───────────────────────────────────


class TestPerfPruneWrite:
    """``prune_low_score_iter`` was a no-op forever because nothing wrote
    ``mem.last_iter_score``. ``record_episode_end`` writes it raw."""

    @pytest.fixture(scope="class")
    def src(self) -> str:
        return inspect.getsource(unified.UnifiedMaclaAgent.record_episode_end)

    def test_record_episode_end_writes_last_iter_score_raw(self, src):
        # Threshold is on the raw 0-7 scale — no /7 or *100 mangling on the write.
        assert "last_iter_score" in src
        assert "last_iter_score = score / " not in src
        assert "last_iter_score = score *" not in src

    def test_prune_no_ops_on_fresh_mem(self, mem):
        assert mem.last_iter_score is None
        assert mem.prune_low_score_iter(4.0) == []


# ── v4 (6): pokemon M5-M7 subgoal ladder ────────────────────────────


class TestPokemonSubgoalLadder:
    """Stack: Route1 (top) → M5 EnterViridian → M6 GetOaksParcel
    → M7 DeliverOaksParcel (bottom). M1-M4 are cutscene-paced and
    don't need stack entries. Built via the generic
    ``build_score_milestone_stack`` framework helper."""

    EXPECTED_NAMES = [
        "DeliverOaksParcel",  # bottom: M7 — parcel delivery
        "NavigateToMap(OaksLab)",  # Stage S v2 bridge — exposes M7
        "GetOaksParcel",  # M6 — Mart clerk
        "NavigateToMap(ViridianMart)",  # Stage S v2 bridge — exposes M6
        "EnterViridian",  # M5 — score-based, fires on env tick
        "NavigateToMap(ViridianCity)",  # Stage S v1 bridge — exposes M5
        "NavigateToMap(Route1)",  # top: next-from-Pallet
    ]

    def test_initial_stack_full_m7_ladder(self):
        stack = initial_subgoal_stack()
        assert [sg.name for sg in stack] == self.EXPECTED_NAMES

    @pytest.mark.parametrize(
        "name, threshold",
        [("EnterViridian", 5), ("GetOaksParcel", 6), ("DeliverOaksParcel", 7)],
    )
    def test_milestone_completion_matches_env_score_trigger(self, name, threshold):
        stack = {sg.name: sg for sg in initial_subgoal_stack()}
        assert stack[name].completion({"score": threshold}) is True
        assert stack[name].completion({"score": threshold - 1}) is False
        assert stack[name].completion({"score": threshold + 1}) is True  # at-least

    def test_stack_is_picklable(self):
        stack = initial_subgoal_stack()
        roundtripped = pickle.loads(pickle.dumps(stack))
        assert [sg.name for sg in roundtripped] == [sg.name for sg in stack]
        m5 = next(sg for sg in roundtripped if sg.name == "EnterViridian")
        assert m5.completion({"score": 5}) is True

    @pytest.mark.parametrize(
        "name",
        ["EnterViridian", "GetOaksParcel", "DeliverOaksParcel"],
    )
    def test_each_milestone_has_descriptive_metadata(self, name):
        sg = next(s for s in initial_subgoal_stack() if s.name == name)
        assert isinstance(sg, Subgoal)
        assert sg.description and sg.suggested_tools

    @pytest.mark.parametrize("name", ["GetOaksParcel", "DeliverOaksParcel"])
    def test_parcel_quest_subgoals_include_dialog_tools(self, name):
        sg = next(s for s in initial_subgoal_stack() if s.name == name)
        assert {"interact_with_object", "continue_dialog"} & set(sg.suggested_tools)

    @pytest.mark.parametrize("garbage", [None, "not a number", {}, []])
    def test_completion_robust_to_garbage_score(self, garbage):
        m5 = next(sg for sg in initial_subgoal_stack() if sg.name == "EnterViridian")
        assert m5.completion({"score": garbage}) is False
