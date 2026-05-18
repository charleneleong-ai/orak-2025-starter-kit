"""Stage R v3: F1 softens the planner-prompt phrasing and F2 adds a
stagnation escape valve. When the top of the subgoal stack hasn't changed
for ≥ SUBGOAL_STAGNATION_THRESHOLD steps, unified.py stops threading it
into the planner so the planner can reason freely until the stack mutates.

Background: PR #93 (v2 introspection) — v2's HARD CONSTRAINT phrasing
locked the planner in PalletTown when move_to could not cross the
Pallet→Route1 edge."""

from __future__ import annotations

from pathlib import Path

from agents._cognitive.subtask_planner import (
    DEFAULT_SYSTEM_PROMPT,
    _render_active_subgoal_block,
)
from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem, Subgoal


def _sub(name: str) -> Subgoal:
    return Subgoal(name=name, description=f"desc_{name}", completion=lambda _obs: False)


# ── F1: soft phrasing ──────────────────────────────────────────────


def test_user_block_drops_hard_constraint_phrasing():
    block = _render_active_subgoal_block("NavigateToMap(Route1): walk north")
    assert "HARD CONSTRAINT" not in block
    assert "MUST" not in block
    assert "NavigateToMap(Route1)" in block


def test_user_block_uses_soft_phrasing():
    block = _render_active_subgoal_block("X: y")
    assert "Currently pursuing" in block
    # Soft preference language — invites override when blocked
    assert "Prefer" in block or "prefer" in block
    assert "blocked" in block.lower() or "evidence" in block.lower()


def test_user_block_still_empty_when_none():
    assert _render_active_subgoal_block(None) == ""
    assert _render_active_subgoal_block("") == ""


def test_system_prompt_no_hard_constraint_lockout():
    txt = DEFAULT_SYSTEM_PROMPT
    assert "HARD CONSTRAINT" not in txt
    assert "MUST be a concrete step that advances" not in txt
    assert "overrides every heuristic below" not in txt
    # but still teaches the concept
    assert "active subgoal" in txt.lower()
    assert "prefer" in txt.lower()


# ── F2: stagnation counter ─────────────────────────────────────────


class TestStagnationCounter:
    def test_starts_zero_with_empty_stack(self):
        assert EnhancedHierarchicalMemorySystem().subgoal_stagnation_steps == 0

    def test_increments_when_top_unchanged(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.push_subgoal(_sub("A"))
        mem.record_subgoal_step()
        mem.record_subgoal_step()
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 3

    def test_resets_when_top_changes_via_push(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.push_subgoal(_sub("A"))
        mem.record_subgoal_step()
        mem.record_subgoal_step()
        mem.push_subgoal(_sub("B"))
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 1  # B's first recorded step

    def test_resets_when_top_changes_via_pop(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.push_subgoal(_sub("A"))
        mem.push_subgoal(_sub("B"))
        mem.record_subgoal_step()
        mem.record_subgoal_step()
        mem.pop_subgoal()  # back to A
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 1

    def test_goes_zero_when_stack_drained(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.push_subgoal(_sub("A"))
        mem.record_subgoal_step()
        mem.pop_subgoal()
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 0

    def test_resets_when_set_subgoal_stack_replaces_top(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.push_subgoal(_sub("A"))
        mem.record_subgoal_step()
        mem.record_subgoal_step()
        mem.set_subgoal_stack([_sub("X"), _sub("Y")])  # new top = Y
        mem.record_subgoal_step()
        assert mem.subgoal_stagnation_steps == 1


# ── F2: unified.py wiring ──────────────────────────────────────────


class TestEscapeValveWiring:
    def test_threshold_constant_is_30(self):
        from agents.macla.unified import SUBGOAL_STAGNATION_THRESHOLD
        assert SUBGOAL_STAGNATION_THRESHOLD == 30

    def test_unified_records_subgoal_step_per_act(self):
        src = Path("agents/macla/unified.py").read_text()
        assert "record_subgoal_step()" in src

    def test_unified_drops_subgoal_at_threshold(self):
        src = Path("agents/macla/unified.py").read_text()
        # The act loop must consult stagnation to gate active_subgoal_str
        assert "subgoal_stagnation_steps" in src
        assert "SUBGOAL_STAGNATION_THRESHOLD" in src
