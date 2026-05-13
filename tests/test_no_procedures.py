"""Stage B' baseline: ``use_procedure_layer`` master switch on MACLA.

Disambiguates whether the score-plateau ceiling we see across pokemon
Stage D / D+reflect (PR #62/#64) / D+reflect 600 / LangGraph verify (PR #66) /
plan-do-check (PR #67) is caused by the procedure layer locking the agent
into a failing procedure, or by an LLM/planner ceiling upstream of MACLA.

Flipping ``use_procedure_layer = False`` makes ``select_procedure`` short-
circuit to ``(None, 0.0)`` so every step takes the LLM fallback path —
but vmem retrieval, subtask planning, and self-reflection all still run
(they live on the agent, not on the selector). It is NOT a "raw model
only" Stage A baseline.

Default ``True`` preserves the current Stage D behaviour, so this PR
is a pure no-op for any existing config that doesn't set the field.
"""

from __future__ import annotations


def _mk_procedure(steps=("a",)):
    from agents.macla.macla_lib import Procedure

    return Procedure(goal="g", preconditions=[], steps=list(steps))


# ── LocalConfig wiring ─────────────────────────────────────────────────


def test_localconfig_use_procedure_layer_defaults_to_true():
    """Default preserves current Stage D behaviour — knob is opt-out."""
    from config.agent_config import LocalConfig

    c = LocalConfig(class_name="x", model="m", temperature=0.0)
    assert c.use_procedure_layer is True


def test_localconfig_use_procedure_layer_accepts_false():
    from config.agent_config import LocalConfig

    c = LocalConfig(class_name="x", model="m", temperature=0.0, use_procedure_layer=False)
    assert c.use_procedure_layer is False


# ── BayesianProcedureSelector master switch ────────────────────────────


def test_selector_use_procedure_layer_defaults_to_true():
    """Selector defaults to current behaviour even without the agent wiring."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
    )

    selector = BayesianProcedureSelector(memory_system=EnhancedHierarchicalMemorySystem())
    assert selector.use_procedure_layer is True


def test_select_procedure_short_circuits_when_layer_disabled():
    """``use_procedure_layer = False`` makes ``select_procedure`` return
    ``(None, 0.0)`` without consulting candidates, EU, or theta."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    mem.procedural_memory["proc_high"] = ProceduralMemoryEntry(procedure=_mk_procedure(["a"]))

    selector = BayesianProcedureSelector(memory_system=mem)
    selector.use_procedure_layer = False
    # Even with a high-EU candidate visible, the master switch wins.
    selector._retrieve_candidates = lambda obs, goal, k=10: ["proc_high"]
    selector._compute_expected_utility = lambda *args, **kw: 0.99

    best_pk, conf = selector.select_procedure("obs", "goal", theta_conf=0.05)
    assert best_pk is None
    assert conf == 0.0


def test_select_procedure_runs_normally_when_layer_enabled():
    """Layer-on path keeps original behaviour — picks the highest-EU candidate
    that clears ``theta_conf``."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    mem.procedural_memory["proc_top"] = ProceduralMemoryEntry(procedure=_mk_procedure(["a"]))

    selector = BayesianProcedureSelector(memory_system=mem)
    selector._retrieve_candidates = lambda obs, goal, k=10: ["proc_top"]
    selector._compute_expected_utility = lambda *args, **kw: 0.9

    best_pk, conf = selector.select_procedure("obs", "goal", theta_conf=0.05)
    assert best_pk == "proc_top"
    assert conf > 0


# ── EnhancedMACLAAgent wiring ──────────────────────────────────────────


def test_enhanced_macla_agent_forwards_use_procedure_layer():
    from agents.macla.macla_lib import EnhancedMACLAAgent

    agent_default = EnhancedMACLAAgent()
    assert agent_default.bayesian_selector.use_procedure_layer is True

    agent_off = EnhancedMACLAAgent(use_procedure_layer=False)
    assert agent_off.bayesian_selector.use_procedure_layer is False
