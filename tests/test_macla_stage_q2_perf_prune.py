"""Stage Q2: performance-gated proc-cache prune on checkpoint load.

Tags each procedure with the iter that added it; on load, drops procs whose
``origin_iter`` matches a prior iter that scored below the per-game M4
threshold. Stops bad iters from poisoning the cumulative-memory chain.

Background / diagnosis: PR #92."""

from __future__ import annotations

from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem, Procedure


def _add_proc(mem, name: str, map_name: str = "PalletTown") -> str:
    # Distinct steps per call so add_procedural_entry doesn't merge entries.
    return mem.add_procedural_entry(
        procedure=Procedure(
            goal=f"goal_{name}",
            preconditions=[],
            steps=["up", f"_marker_{name}"],
            map_name=map_name,
        ),
        contexts={f"ctx_{name}"},
        goals={f"goal_{name}"},
        performance=0.5,
    )


class TestOriginIterTagging:
    def test_proc_tagged_with_current_iter(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 3
        assert mem.procedural_memory[_add_proc(mem, "p1")].origin_iter == 3

    def test_origin_iter_defaults_zero_for_legacy_checkpoints(self):
        mem = EnhancedHierarchicalMemorySystem()
        assert mem.procedural_memory[_add_proc(mem, "p1")].origin_iter == 0


class TestLastIterScore:
    def test_defaults_none(self):
        assert EnhancedHierarchicalMemorySystem().last_iter_score is None

    def test_persists_via_attribute(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.last_iter_score = 4.0
        assert mem.last_iter_score == 4.0


class TestPruneLowScoreIter:
    def test_drops_procs_from_iter_below_threshold(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 1
        keep = _add_proc(mem, "good_p1")
        mem.current_iter = 2
        drop = _add_proc(mem, "bad_p2")
        mem.last_iter_score = 1.0

        removed = mem.prune_low_score_iter(score_threshold=4.0)

        assert removed == [drop]
        assert keep in mem.procedural_memory
        assert drop not in mem.procedural_memory

    def test_keeps_procs_when_iter_at_threshold(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 2
        key = _add_proc(mem, "good_p2")
        mem.last_iter_score = 4.0

        assert mem.prune_low_score_iter(score_threshold=4.0) == []
        assert key in mem.procedural_memory

    def test_no_op_when_score_unknown(self):
        # Fail-safe: refuse to prune without a recorded score.
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 2
        key = _add_proc(mem, "p2")
        mem.last_iter_score = None

        assert mem.prune_low_score_iter(score_threshold=4.0) == []
        assert key in mem.procedural_memory

    def test_cleans_context_and_goal_indices(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 2
        key = _add_proc(mem, "p2")
        mem.last_iter_score = 0.0

        mem.prune_low_score_iter(score_threshold=4.0)

        assert key not in mem.context_index["ctx_p2"]
        assert key not in mem.goal_index["goal_p2"]

    def test_prune_count_recorded_in_stats(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 2
        _add_proc(mem, "p2a")
        _add_proc(mem, "p2b", map_name="Route1")
        mem.last_iter_score = 0.0

        mem.prune_low_score_iter(score_threshold=4.0)

        assert mem.stats.get("procedures_pruned_low_score") == 2

    def test_preserves_procs_from_earlier_high_scoring_iters(self):
        # Stage Q failure mode: iter 1 lifted (keep), iter 2 collapsed (drop only iter 2's procs).
        mem = EnhancedHierarchicalMemorySystem()
        mem.current_iter = 1
        iter1 = [_add_proc(mem, f"good_{i}") for i in range(3)]
        mem.current_iter = 2
        iter2 = [_add_proc(mem, f"bad_{i}", map_name="Route1") for i in range(3)]
        mem.last_iter_score = 2.0

        removed = mem.prune_low_score_iter(score_threshold=4.0)

        assert set(removed) == set(iter2)
        assert all(k in mem.procedural_memory for k in iter1)


def test_pokemon_adapter_exposes_proc_cache_threshold():
    # Per-game threshold lives in the adapter so mario/2048 can set their own.
    from agents.pokemon_red.game_adapter import PROC_CACHE_MIN_ITER_SCORE
    assert PROC_CACHE_MIN_ITER_SCORE == 4.0
