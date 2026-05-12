"""Procedure-layer escape: failure-streak retirement + stuck-state forced LLM.

Two diagnosed-and-targeted fixes after the 600-step + Stage E (PR #66) +
Stage F (PR #67) negative results all plateaued at pokemon score=4 (57.14%):

1. **Failure-streak retirement**: per-procedure ``consecutive_failures`` counter;
   when a procedure fails K times in a row, the Bayesian selector filters it
   from the candidate set so the agent falls through to LLM fallback. Resets
   on success.

2. **Stuck-state forced LLM**: per-agent ``steps_since_step_success`` counter;
   when the agent goes N steps without ANY procedure-success, set the next
   theta above 1.0 so ``select_procedure`` rejects all candidates (best_eu is
   clamped to [0,1]) and the agent uses LLM fallback for at least one step.

Both fixes are gated by configurable thresholds (``K=5``, ``N=50`` defaults)
exposed via ``LocalConfig``. Setting either to 0 disables that fix entirely.
"""

from __future__ import annotations


def _mk_procedure(steps=("a",)):
    from agents.macla.macla_lib import Procedure

    return Procedure(goal="g", preconditions=[], steps=list(steps))


def _mk_context(success: bool = False):
    from agents.macla.macla_lib import ContrastiveContext

    return ContrastiveContext(
        observation_init="",
        action_sequence=[],
        observation_term="",
        cumulative_reward=0.0,
        trajectory_id="t",
        success=success,
    )


# ── ProceduralMemoryEntry.consecutive_failures ─────────────────────────


def test_procedural_entry_defaults_consecutive_failures_to_zero():
    from agents.macla.macla_lib import ProceduralMemoryEntry

    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    assert entry.consecutive_failures == 0


def test_record_execution_outcome_increments_streak_on_failure():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem, ProceduralMemoryEntry

    mem = EnhancedHierarchicalMemorySystem()
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    mem.procedural_memory["proc_1"] = entry

    mem.record_execution_outcome("proc_1", success=False, context=_mk_context())
    assert entry.consecutive_failures == 1
    mem.record_execution_outcome("proc_1", success=False, context=_mk_context())
    mem.record_execution_outcome("proc_1", success=False, context=_mk_context())
    assert entry.consecutive_failures == 3


def test_record_execution_outcome_resets_streak_on_success():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem, ProceduralMemoryEntry

    mem = EnhancedHierarchicalMemorySystem()
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    mem.procedural_memory["proc_1"] = entry
    entry.consecutive_failures = 4

    mem.record_execution_outcome("proc_1", success=True, context=_mk_context(success=True))
    assert entry.consecutive_failures == 0


# ── BayesianProcedureSelector failure-streak filter ────────────────────


def test_select_procedure_filters_out_at_or_above_failure_streak_max():
    """Procedure with ``consecutive_failures >= failure_streak_max`` must
    be excluded from candidates. Behavioral contract:

    - default ``failure_streak_max = 5`` rejects entries at 5+ failures
    - an entry at 4 is still selectable
    - if ALL candidates are retired, selector returns (None, 0.0)
    """
    import inspect

    from agents.macla.macla_lib import BayesianProcedureSelector

    sig = inspect.signature(BayesianProcedureSelector.__init__)
    assert "failure_streak_max" in sig.parameters
    assert sig.parameters["failure_streak_max"].default == 5


def test_select_procedure_skips_retired_entry_when_returning_top_candidate():
    """End-to-end: a procedure at the streak threshold should not be the
    selected ``best_pk``, even if its EU would otherwise be highest."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    high_eu_retired = ProceduralMemoryEntry(procedure=_mk_procedure(["a"]))
    high_eu_retired.consecutive_failures = 5
    mem.procedural_memory["proc_retired"] = high_eu_retired
    fresh = ProceduralMemoryEntry(procedure=_mk_procedure(["b"]))
    mem.procedural_memory["proc_fresh"] = fresh

    selector = BayesianProcedureSelector(memory_system=mem, failure_streak_max=5)
    selector._retrieve_candidates = lambda obs, goal, k=10: ["proc_retired", "proc_fresh"]
    selector._compute_expected_utility = lambda entry, obs, goal: (
        0.9 if entry is high_eu_retired else 0.1
    )

    best_pk, _ = selector.select_procedure("obs", "goal", theta_conf=0.05)
    assert best_pk != "proc_retired"


def test_select_procedure_returns_none_when_all_candidates_retired():
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    a = ProceduralMemoryEntry(procedure=_mk_procedure(["a"]))
    b = ProceduralMemoryEntry(procedure=_mk_procedure(["b"]))
    a.consecutive_failures = 5
    b.consecutive_failures = 7
    mem.procedural_memory["a"] = a
    mem.procedural_memory["b"] = b

    selector = BayesianProcedureSelector(memory_system=mem, failure_streak_max=5)
    selector._retrieve_candidates = lambda obs, goal, k=10: ["a", "b"]
    selector._compute_expected_utility = lambda *args, **kw: 0.9

    best_pk, conf = selector.select_procedure("obs", "goal", theta_conf=0.05)
    assert best_pk is None
    assert conf == 0.0


# ── Stuck-state forced LLM fallback ────────────────────────────────────


def test_macla_agent_tracks_steps_since_step_success():
    """Agent exposes ``note_step_outcome(success: bool)`` that increments
    ``steps_since_step_success`` on failure and resets on success."""
    from agents.macla.macla_lib import EnhancedMACLAAgent

    agent = EnhancedMACLAAgent(force_llm_after_stuck_steps=50)
    assert agent._steps_since_step_success == 0

    agent.note_step_outcome(success=False)
    agent.note_step_outcome(success=False)
    agent.note_step_outcome(success=False)
    assert agent._steps_since_step_success == 3

    agent.note_step_outcome(success=True)
    assert agent._steps_since_step_success == 0


def test_macla_agent_should_force_llm_returns_true_after_threshold():
    """When ``steps_since_step_success >= force_llm_after_stuck_steps``,
    ``should_force_llm_fallback()`` returns True."""
    from agents.macla.macla_lib import EnhancedMACLAAgent

    agent = EnhancedMACLAAgent(force_llm_after_stuck_steps=5)
    for _ in range(4):
        agent.note_step_outcome(success=False)
    assert agent.should_force_llm_fallback() is False
    agent.note_step_outcome(success=False)
    assert agent.should_force_llm_fallback() is True


def test_macla_agent_force_llm_zero_disables_feature():
    """``force_llm_after_stuck_steps=0`` disables the feature entirely;
    ``should_force_llm_fallback()`` never returns True."""
    from agents.macla.macla_lib import EnhancedMACLAAgent

    agent = EnhancedMACLAAgent(force_llm_after_stuck_steps=0)
    for _ in range(100):
        agent.note_step_outcome(success=False)
    assert agent.should_force_llm_fallback() is False


# ── Compute adaptive theta returns force-LLM sentinel when stuck ───────


def test_compute_adaptive_theta_returns_above_one_when_stuck():
    """When the agent should force LLM, ``_compute_adaptive_theta`` returns
    a value strictly greater than 1.0 (above the EU clamp ceiling), so
    ``select_procedure``'s existing ``best_eu < theta_conf`` reject path
    fires for ALL candidates without further changes to the selector."""
    from agents.macla.macla_lib import EnhancedMACLAAgent

    agent = EnhancedMACLAAgent(force_llm_after_stuck_steps=3)
    for _ in range(3):
        agent.note_step_outcome(success=False)
    theta = agent._compute_adaptive_theta()
    assert theta > 1.0, f"expected theta > 1.0 when forcing LLM, got {theta}"


# ── LocalConfig wiring ─────────────────────────────────────────────────


def test_localconfig_declares_procedure_escape_fields_with_defaults():
    """Both knobs default to the recommended K=5 / N=50."""
    from config.agent_config import LocalConfig

    c = LocalConfig(class_name="x", model="m", temperature=0.0)
    assert c.procedure_failure_streak_max == 5
    assert c.force_llm_after_stuck_steps == 50


def test_localconfig_accepts_explicit_yaml_values():
    from config.agent_config import LocalConfig

    c = LocalConfig(
        class_name="x",
        model="m",
        temperature=0.0,
        procedure_failure_streak_max=3,
        force_llm_after_stuck_steps=30,
    )
    assert c.procedure_failure_streak_max == 3
    assert c.force_llm_after_stuck_steps == 30
