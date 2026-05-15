"""Stage L: map-aware procedure keys + iter-based TTL/decay for cumulative memory.

Background
----------
Stage K cumulative-memory rerun (PR #75) under the asm fix showed **negative
transfer** in the procedure cache: iter 2 took +91 steps to bank M4 (220 vs
iter 1's 129) and never reached Route1, because procedures captured against
transient OaksLab states kept firing in the wrong map. iter 3 increased
perseveration to 22.0% (worst of three). All scored 57.14% — asm fix prevents
the catastrophic floor regression seen pre-fix but doesn't fix the underlying
context-blindness in the procedure key.

The current procedure cache (``agents/macla/macla_lib.py``) keys procedures on
``hash(str(procedure.steps))`` only. Two design gaps:

1. **Procedure key is not map-aware** — an OaksLab procedure can fire when
   the agent is in Route1 or PalletTown because the cache match doesn't
   condition on ``map_name``.

2. **No staleness retirement** — Bayesian acquisition adds procedures but
   never retires them based on usage staleness or context-mismatch.

This file specifies the new behaviour:

* ``Procedure`` carries a ``map_name`` field (default ``"unknown"``).
* ``add_procedural_entry`` keys on ``(steps, map_name)`` so same-steps in
  different maps don't collide / merge.
* ``_retrieve_candidates`` filters out procedures whose ``map_name`` is
  neither the current map nor ``"unknown"``.
* ``ProceduralMemoryEntry`` tracks ``last_used_iter``; ``prune_stale_procedures``
  removes entries unused for ≥ ``max_age`` (default 2) full iters.
* ``EnhancedHierarchicalMemorySystem.bump_iter()`` increments the iter
  counter; called when a checkpoint is loaded (i.e. each new iter under
  ``--load-checkpoint --prev-run-id`` chaining).
"""

from __future__ import annotations


def _mk_procedure(steps=("a",), map_name=None):
    from agents.macla.macla_lib import Procedure

    if map_name is None:
        return Procedure(goal="g", preconditions=[], steps=list(steps))
    return Procedure(goal="g", preconditions=[], steps=list(steps), map_name=map_name)


# ─── Procedure dataclass surface ──────────────────────────────────────────


def test_procedure_has_map_name_field_default_unknown():
    p = _mk_procedure(["walk_n"])
    assert p.map_name == "unknown"


def test_procedure_map_name_settable():
    p = _mk_procedure(["walk_n"], map_name="Route1")
    assert p.map_name == "Route1"


def test_procedural_memory_entry_has_last_used_iter_default_zero():
    from agents.macla.macla_lib import ProceduralMemoryEntry

    entry = ProceduralMemoryEntry(procedure=_mk_procedure(["a"]))
    assert entry.last_used_iter == 0


# ─── add_procedural_entry: map-aware keys ─────────────────────────────────


def test_add_procedural_entry_keys_separate_maps_distinctly():
    """Two procedures with identical steps but different map_name must NOT
    collide — they need separate cache entries."""
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()

    p_oaks = _mk_procedure(["interact"], map_name="OaksLab")
    p_route1 = _mk_procedure(["interact"], map_name="Route1")

    pk_oaks = mem.add_procedural_entry(p_oaks, contexts={"ctx"}, goals={"g"}, performance=1.0)
    pk_route1 = mem.add_procedural_entry(p_route1, contexts={"ctx"}, goals={"g"}, performance=1.0)

    assert pk_oaks != pk_route1
    assert pk_oaks in mem.procedural_memory
    assert pk_route1 in mem.procedural_memory


def test_add_procedural_entry_merges_same_map_same_steps():
    """Same steps in same map collapse to one entry (existing behaviour)."""
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()

    p1 = _mk_procedure(["walk_n"], map_name="Route1")
    p2 = _mk_procedure(["walk_n"], map_name="Route1")

    pk1 = mem.add_procedural_entry(p1, contexts={"a"}, goals={"g"}, performance=1.0)
    pk2 = mem.add_procedural_entry(p2, contexts={"b"}, goals={"g"}, performance=1.0)

    assert pk1 == pk2
    assert len(mem.procedural_memory) == 1


# ─── _retrieve_candidates: map filter ─────────────────────────────────────


def test_retrieve_candidates_filters_out_wrong_map():
    """Procedures keyed on a different map must not be returned for the
    current map."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
    )

    mem = EnhancedHierarchicalMemorySystem()
    pk_oaks = mem.add_procedural_entry(
        _mk_procedure(["interact"], map_name="OaksLab"),
        contexts={"oaks_ctx"},
        goals={"talk_to_oak"},
        performance=1.0,
    )
    pk_route1 = mem.add_procedural_entry(
        _mk_procedure(["walk_n"], map_name="Route1"),
        contexts={"route1_ctx"},
        goals={"talk_to_oak"},
        performance=1.0,
    )

    selector = BayesianProcedureSelector(memory_system=mem)

    obs_route1 = "Map Name: Route1, (x_max, y_max): (10, 36)"
    cands = selector._retrieve_candidates(obs_route1, "talk_to_oak", k=10)
    assert pk_route1 in cands
    assert pk_oaks not in cands


def test_retrieve_candidates_includes_unknown_map_procedures():
    """Procedures with map_name=='unknown' match any current map
    (backwards-compat for procedures captured before this change)."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
    )

    mem = EnhancedHierarchicalMemorySystem()
    pk_unknown = mem.add_procedural_entry(
        _mk_procedure(["walk_n"], map_name="unknown"),
        contexts={"any_ctx"},
        goals={"explore"},
        performance=1.0,
    )

    selector = BayesianProcedureSelector(memory_system=mem)

    obs_route1 = "Map Name: Route1, (x_max, y_max): (10, 36)"
    cands = selector._retrieve_candidates(obs_route1, "explore", k=10)
    assert pk_unknown in cands


def test_retrieve_candidates_backwards_compat_missing_map_name_attr():
    """A procedure object missing the ``map_name`` attribute (loaded from a
    pre-Stage-L checkpoint) is treated as ``'unknown'`` — matches any map."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    old_proc = _mk_procedure(["walk_n"])
    # Simulate a pickle from a pre-Stage-L checkpoint that didn't have the field
    del old_proc.__dict__["map_name"]
    entry = ProceduralMemoryEntry(procedure=old_proc)
    entry.contexts = {"any"}
    entry.goals = {"explore"}
    mem.procedural_memory["old_proc"] = entry
    mem.goal_index["explore"].add("old_proc")

    selector = BayesianProcedureSelector(memory_system=mem)
    obs_route1 = "Map Name: Route1, (x_max, y_max): (10, 36)"
    cands = selector._retrieve_candidates(obs_route1, "explore", k=10)
    assert "old_proc" in cands


# ─── prune_stale_procedures ───────────────────────────────────────────────


def test_prune_stale_procedures_removes_entries_older_than_max_age():
    from agents.macla.macla_lib import (
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    mem.current_iter = 5

    fresh = ProceduralMemoryEntry(procedure=_mk_procedure(["fresh"]))
    fresh.last_used_iter = 4
    stale = ProceduralMemoryEntry(procedure=_mk_procedure(["stale"]))
    stale.last_used_iter = 2

    mem.procedural_memory["fresh"] = fresh
    mem.procedural_memory["stale"] = stale

    removed = mem.prune_stale_procedures(max_age=2)
    assert "stale" not in mem.procedural_memory
    assert "fresh" in mem.procedural_memory
    assert "stale" in removed


def test_prune_stale_procedures_default_max_age_two():
    """Default max_age=2 retires procedures unused for >=2 full iters."""
    from agents.macla.macla_lib import (
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    mem.current_iter = 3
    e = ProceduralMemoryEntry(procedure=_mk_procedure(["old"]))
    e.last_used_iter = 0  # 3 iters ago
    mem.procedural_memory["old"] = e

    removed = mem.prune_stale_procedures()
    assert "old" in removed
    assert "old" not in mem.procedural_memory


def test_bump_iter_increments_current_iter():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    assert mem.current_iter == 0
    mem.bump_iter()
    assert mem.current_iter == 1
    mem.bump_iter()
    assert mem.current_iter == 2


# ─── End-to-end: select_procedure respects map filter ─────────────────────


def test_select_procedure_skips_wrong_map_procedure():
    """Even if an out-of-map procedure has high EU, select_procedure must
    not return it for the current map."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
    )

    mem = EnhancedHierarchicalMemorySystem()
    pk_oaks = mem.add_procedural_entry(
        _mk_procedure(["high_eu_action"], map_name="OaksLab"),
        contexts={"oaks_ctx"},
        goals={"explore"},
        performance=1.0,
    )

    selector = BayesianProcedureSelector(memory_system=mem)
    # Force everything to look high EU — the filter must still exclude wrong-map.
    selector._compute_expected_utility = lambda *a, **kw: 0.99

    obs_route1 = "Map Name: Route1, (x_max, y_max): (10, 36)"
    best_pk, conf = selector.select_procedure(obs_route1, "explore", theta_conf=0.05)
    assert best_pk != pk_oaks
