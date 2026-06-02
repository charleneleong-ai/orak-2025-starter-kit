"""Tests for the episode-level procedure trace on EnhancedHierarchicalMemorySystem.

The trace is a deque[str] populated in record_execution_outcome and drained
at episode end. Tested in isolation with synthetic Procedure entries.
"""

from __future__ import annotations

import pytest

from agents.macla.macla_lib import (
    ContrastiveContext,
    EnhancedHierarchicalMemorySystem,
    ProceduralMemoryEntry,
    Procedure,
)


def _ctx() -> ContrastiveContext:
    return ContrastiveContext(
        observation_init="",
        action_sequence=[],
        observation_term="",
        cumulative_reward=0.0,
        trajectory_id="t",
        success=True,
    )


def _seed(mem: EnhancedHierarchicalMemorySystem, key: str) -> None:
    """Insert a minimal Procedure under `key` so record_execution_outcome doesn't skip it."""
    mem.procedural_memory[key] = ProceduralMemoryEntry(
        procedure=Procedure(goal="g", preconditions=[], steps=[]),
        success_contexts=[],
        failure_contexts=[],
    )


@pytest.fixture
def mem() -> EnhancedHierarchicalMemorySystem:
    return EnhancedHierarchicalMemorySystem()


class TestEpisodeProcTrace:
    def test_record_execution_outcome_appends_proc_key(self, mem):
        _seed(mem, "p1")
        mem.record_execution_outcome("p1", success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == ["p1"]

    def test_trace_preserves_execution_order(self, mem):
        for k in ("p1", "p2", "p1", "p3"):
            _seed(mem, k)
            mem.record_execution_outcome(k, success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == ["p1", "p2", "p1", "p3"]

    def test_unknown_proc_key_is_not_appended(self, mem):
        # record_execution_outcome returns early when proc_key is unknown;
        # trace must NOT capture that no-op call.
        mem.record_execution_outcome("ghost", success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == []

    def test_drain_episode_trace_returns_and_clears(self, mem):
        for k in ("a", "b", "c"):
            _seed(mem, k)
            mem.record_execution_outcome(k, success=True, context=_ctx())
        assert mem.drain_episode_trace() == ["a", "b", "c"]
        assert list(mem._episode_proc_trace) == []
        # Second drain on empty trace is a no-op
        assert mem.drain_episode_trace() == []

    def test_deque_maxlen_caps_growth(self, mem):
        # maxlen=2000 — older entries are dropped FIFO
        _seed(mem, "p")
        for _ in range(2500):
            mem.record_execution_outcome("p", success=True, context=_ctx())
        assert len(mem._episode_proc_trace) == 2000

    def test_defensive_read_for_old_checkpoints(self, mem):
        # Simulate an older checkpoint that was pickled before _episode_proc_trace existed.
        del mem._episode_proc_trace
        _seed(mem, "p")
        # Should re-initialise on first touch instead of raising AttributeError.
        mem.record_execution_outcome("p", success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == ["p"]

    def test_setstate_resets_trace_on_checkpoint_restore(self, mem):
        # Simulate a checkpoint restore where the previous iter had populated
        # the trace but didn't drain it (eg. crashed mid-episode). __setstate__
        # should zero the trace so the new iter starts clean.
        _seed(mem, "p_stale")
        mem.record_execution_outcome("p_stale", success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == ["p_stale"]

        # Mimic pickle round-trip via __getstate__/__setstate__ directly.
        # __setstate__ is the per-episode reset hook used by checkpoint restore.
        mem.__setstate__(mem.__getstate__())

        assert list(mem._episode_proc_trace) == []
