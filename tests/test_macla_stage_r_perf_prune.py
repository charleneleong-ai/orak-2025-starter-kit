"""Stage R: performance-gated proc-cache prune on checkpoint load.

Tags each procedure with its origin iter; on checkpoint load, drops all
procedures added during a prior iter that scored below the per-game M4
threshold. Stops bad iters from poisoning the cumulative-memory chain.

Diagnosis: Stage Q n=5 introspect showed iter 1 lifted past M5 (Pallet →
Route1 → Viridian) but iters 2-5 collapsed back to PalletTown and never
escaped, even with the exit-tile hint. Cause: Stage L's age-based prune
(``prune_stale_procedures``) only retires unused procs — bad iters with
plenty of selected-but-useless procs survive intact and trap late iters.

Stage R rule: if prev_iter_score < threshold, drop every proc with
``origin_iter == prev_iter``. Keeps only procedures from iters that
actually made progress.
"""

from __future__ import annotations

import pytest

from agents.macla.macla_lib import (
    EnhancedHierarchicalMemorySystem,
    Procedure,
)


def _proc(name: str, steps: tuple = ("up",), map_name: str = "PalletTown") -> Procedure:
    """Tiny helper — Procedure with distinct steps so add_procedural_entry doesn't merge.

    The ``name`` arg is folded into the steps list to force distinct hash keys
    across test procedures while keeping the test code readable.
    """
    return Procedure(
        goal=f"goal_{name}",
        preconditions=[],
        steps=[*steps, f"_marker_{name}"],
        map_name=map_name,
    )


def _add_proc_in_iter(mem, name: str, map_name: str = "PalletTown") -> str:
    """Add a proc and tag it with the current_iter — what Stage R requires."""
    return mem.add_procedural_entry(
        procedure=_proc(name, map_name=map_name),
        contexts={f"ctx_{name}"},
        goals={f"goal_{name}"},
        performance=0.5,
    )


class TestOriginIterTagging:
    def test_new_proc_is_tagged_with_current_iter(self):
        """Each ProceduralMemoryEntry remembers which iter introduced it."""
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 3
        key = _add_proc_in_iter(mem, "p1")
        assert mem.procedural_memory[key].origin_iter == 3

    def test_origin_iter_zero_for_pre_stage_r_checkpoints(self):
        """Procs without explicit origin (legacy load) default to 0."""
        mem = EnhancedHierarchicalMemorySystem()
        # Don't set current_iter — defaults to 0
        key = _add_proc_in_iter(mem, "p1")
        assert mem.procedural_memory[key].origin_iter == 0


class TestLastIterScore:
    def test_last_iter_score_defaults_none(self):
        mem = EnhancedHierarchicalMemorySystem()
        assert mem.last_iter_score is None

    def test_last_iter_score_persists_via_attribute(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.last_iter_score = 4.0
        assert mem.last_iter_score == 4.0


class TestPruneLowScoreIter:
    def test_drops_procs_from_iter_that_scored_below_threshold(self):
        """The iter being pruned is current_iter (most-recently-completed)."""
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 1
        keep_key = _add_proc_in_iter(mem, "good_p1")
        mem.current_iter = 2  # simulate next iter ran and added more
        drop_key = _add_proc_in_iter(mem, "bad_p2")
        # Most-recent iter (2) scored 1/7 — well below 4/7
        mem.last_iter_score = 1.0
        removed = mem.prune_low_score_iter(score_threshold=4.0)
        assert drop_key in removed
        assert keep_key not in removed
        assert keep_key in mem.procedural_memory
        assert drop_key not in mem.procedural_memory

    def test_keeps_procs_when_iter_scored_at_or_above_threshold(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 2
        key = _add_proc_in_iter(mem, "good_p2")
        mem.last_iter_score = 4.0  # exactly at threshold — keep
        removed = mem.prune_low_score_iter(score_threshold=4.0)
        assert removed == []
        assert key in mem.procedural_memory

    def test_no_op_when_last_iter_score_unknown(self):
        """Without a recorded score, refuse to prune — fail-safe."""
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 2
        key = _add_proc_in_iter(mem, "p2")
        mem.last_iter_score = None
        removed = mem.prune_low_score_iter(score_threshold=4.0)
        assert removed == []
        assert key in mem.procedural_memory

    def test_cleans_up_context_and_goal_indices(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 2
        key = _add_proc_in_iter(mem, "p2")
        mem.last_iter_score = 0.0
        mem.prune_low_score_iter(score_threshold=4.0)
        assert key not in mem.context_index["ctx_p2"]
        assert key not in mem.goal_index["goal_p2"]

    def test_pruned_count_recorded_in_stats(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 2
        _add_proc_in_iter(mem, "p2a")
        _add_proc_in_iter(mem, "p2b", map_name="Route1")
        mem.last_iter_score = 0.0
        mem.prune_low_score_iter(score_threshold=4.0)
        assert mem.stats.get("procedures_pruned_low_score", 0) == 2

    def test_preserves_procs_from_earlier_high_scoring_iters(self):
        """The Stage Q failure mode we're targeting: iter 1 lifted (keep),
        iter 2 collapsed (drop only iter 2's procs)."""
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 1
        iter1_keys = [_add_proc_in_iter(mem, f"good_{i}") for i in range(3)]
        mem.current_iter = 2
        iter2_keys = [_add_proc_in_iter(mem, f"bad_{i}", map_name="Route1") for i in range(3)]
        mem.last_iter_score = 2.0
        removed = mem.prune_low_score_iter(score_threshold=4.0)
        assert set(removed) == set(iter2_keys)
        assert all(k in mem.procedural_memory for k in iter1_keys)


class TestThresholdConstant:
    def test_pokemon_adapter_exposes_proc_cache_threshold(self):
        """The per-game threshold (M4 raw score) lives in the adapter so
        mario/2048 can set their own when they add cumulative memory."""
        from agents.pokemon_red.game_adapter import PROC_CACHE_MIN_ITER_SCORE
        assert PROC_CACHE_MIN_ITER_SCORE == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
