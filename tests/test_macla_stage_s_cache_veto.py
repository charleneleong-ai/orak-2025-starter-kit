"""Stage S — cache veto under escape-valve fire.

Stage R v5 introspection (docs/experiments/stage_r_subgoals/v5_n5_introspection.md)
proved the perf-prune write-side gate works correctly; iter1's procs are
the *only* ones being passed forward across iters. The wall is the read
side: at fresh boot the cached PalletTown-tagged procs from iter1 match
the observation, the selector picks them at theta=0.050, and the agent
loops on iter1's mid-trajectory moves. The escape valve (drop subgoal
from planner prompt at stagnation >= 30) already fires but loses the
arbitration to the cache.

Fix: when the escape valve fires, *also* veto cached-proc selection for
the next K steps so fresh planning wins. Veto is per-episode (reset on
__setstate__) and decrements by 1 each step, matching the existing
subgoal-stagnation counter and anti-perseveration position counter
which are also per-episode signals.
"""

from __future__ import annotations

import inspect
import pickle

import pytest

from agents.macla import unified
from agents.macla.macla_lib import BayesianProcedureSelector, EnhancedHierarchicalMemorySystem

# ── memory-side state ──────────────────────────────────────────────────


class TestCacheVetoState:
    """Per-episode veto counter on the memory system. Decremented per
    step alongside subgoal_stagnation_steps; cleared on __setstate__."""

    def test_cache_vetoed_defaults_false(self):
        mem = EnhancedHierarchicalMemorySystem()
        assert mem.cache_vetoed is False

    @pytest.mark.parametrize("k", [1, 5, 30])
    def test_set_cache_veto_activates(self, k: int):
        mem = EnhancedHierarchicalMemorySystem()
        mem.set_cache_veto(k)
        assert mem.cache_vetoed is True

    def test_tick_cache_veto_decrements_until_cleared(self):
        mem = EnhancedHierarchicalMemorySystem()
        mem.set_cache_veto(3)
        assert mem.cache_vetoed is True
        for expected_active in (True, True, False):
            mem.tick_cache_veto()
            assert mem.cache_vetoed is expected_active

    def test_tick_below_zero_stays_at_zero(self):
        """Defensive: tick is safe to call when no veto is active."""
        mem = EnhancedHierarchicalMemorySystem()
        for _ in range(5):
            mem.tick_cache_veto()
        assert mem.cache_vetoed is False

    def test_set_cache_veto_replaces_remaining(self):
        """Re-arming with a new K replaces the remaining countdown rather
        than stacking — escape valve fires once per stagnation event, the
        intent is "veto for K more steps", not "veto for the sum of all
        prior calls"."""
        mem = EnhancedHierarchicalMemorySystem()
        mem.set_cache_veto(3)
        mem.set_cache_veto(10)
        for _ in range(4):
            mem.tick_cache_veto()
        # 10 - 4 = 6 left
        assert mem.cache_vetoed is True

    def test_setstate_clears_veto(self):
        """Per-episode signal: zeroed on checkpoint load alongside the
        stagnation counter and position counter (Stage R v4 fix). Without
        this, iter N+1 starts boot with iter N's tail veto window leaking
        forward and the escape signal becomes meaningless."""
        mem = EnhancedHierarchicalMemorySystem()
        mem.set_cache_veto(30)
        blob = pickle.dumps(mem)
        restored = pickle.loads(blob)
        assert restored.cache_vetoed is False


# ── selector-side enforcement ─────────────────────────────────────────


class TestSelectorRespectsVeto:
    """``BayesianProcedureSelector.select_procedure`` must short-circuit
    to (None, 0.0) when the memory's cache_vetoed flag is set, regardless
    of what candidates / theta_conf would otherwise produce."""

    def test_returns_none_when_vetoed(self):
        mem = EnhancedHierarchicalMemorySystem()
        sel = BayesianProcedureSelector(mem)
        mem.set_cache_veto(30)
        # Even with the default theta_conf, the veto wins. We don't need
        # populated procedural memory because the early-return must fire
        # before candidate retrieval.
        pk, conf = sel.select_procedure(observation="anything", goal="x")
        assert pk is None
        assert conf == 0.0

    def test_returns_none_when_no_procs_and_not_vetoed(self):
        """Control: even without a veto, an empty cache returns (None, 0)
        — confirms the veto check isn't the only path to None."""
        mem = EnhancedHierarchicalMemorySystem()
        sel = BayesianProcedureSelector(mem)
        pk, conf = sel.select_procedure(observation="anything", goal="x")
        assert pk is None
        assert conf == 0.0


# ── unified.py wiring ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def unified_src() -> str:
    """Once-per-session source read — these are static-grep tests, no
    need to re-read on every assertion."""
    return inspect.getsource(unified)


class TestEscapeValveSetsCacheVeto:
    """Source-grep wiring tests: when the escape valve fires, unified.py
    must also call set_cache_veto on the memory. Tick must happen once
    per step alongside record_subgoal_step."""

    def test_cache_veto_k_steps_constant_defined(self, unified_src: str):
        assert "CACHE_VETO_K_STEPS" in unified_src

    def test_escape_valve_calls_set_cache_veto(self, unified_src: str):
        """When the escape-valve branch fires, set_cache_veto must be
        called with the K-steps constant. Pragmatic source-grep — the
        runtime behaviour is asserted indirectly via the selector test
        above + the state tests."""
        assert "set_cache_veto" in unified_src
        assert "CACHE_VETO_K_STEPS" in unified_src

    def test_per_step_tick_wired(self, unified_src: str):
        """Veto must decrement per step or it would never clear once set.
        unified.py is the natural site since it already calls
        ``record_subgoal_step`` on every act-loop iteration."""
        assert "tick_cache_veto" in unified_src
