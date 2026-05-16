"""Stage N: bootstrap-neutral signals + planner-side novelty.

Background
----------
Stage M (PR #86 / `docs/experiments/stage_m_multi_signal/n5_rerun.md`)
introduced three multiplicative selector signals (state-delta confidence ×
map-novelty theta-bump × logprob percentile-rank). The n=5 sweep was
score-FLAT vs Stage L but trajectory introspection revealed two structural
problems that Stage N corrects:

1. **Multiplicative damping trap.** Both ``_state_delta_confidence`` and
   ``_logprob_confidence`` returned ``0.5`` when uncalibrated, mapping to a
   ``0.75×`` damping multiplier each — combined ``0.5625×`` on every
   brand-new procedure. New procs need to fire to refine, but they couldn't
   fire because they were damped before they fired. Stage M ended with
   1 successful execution and 13 procedure refinements across 5 iters vs
   Stage L's 4 / 289 (22× less refinement).

   **Fix:** bootstrap returns ``1.0`` (neutral) instead of ``0.5``. The signal
   only "speaks" once it has enough evidence. ``_state_delta_confidence``
   gains a ``_SDC_BOOTSTRAP_N = 3`` threshold matching ``_LOGPROB_BOOTSTRAP_N = 10``.

2. **Novelty theta-bump fired zero times** across 420 selector events in
   the Stage M n=5 sweep. ``select_procedure`` early-returns when there are
   no candidate procs in the current map, before the ``new_map`` log line is
   reached. With the cache being OaksLab-only, no new-map call ever had
   candidates. The bonus is dead code as designed.

   **Fix:** strip the theta-bump and visit-marking from
   ``BayesianProcedureSelector.select_procedure``. Add a planner-side
   ``map_visit_status`` accessor on ``EnhancedHierarchicalMemorySystem``.
   ``agents/macla/unified.py`` consults it before each
   ``LLMSubtaskPlanner.plan()`` call, injects a novelty hint into the
   history string when the current map is unvisited, and only THEN marks
   the map as visited (so the hint fires exactly once per new map).
"""

from __future__ import annotations

# ─── Helpers (copied from Stage M test file for isolation) ─────────────────


def _mk_procedure(steps=("a",), map_name=None, mean_logprob=None):
    from agents.macla.macla_lib import Procedure

    if map_name is None:
        p = Procedure(goal="g", preconditions=[], steps=list(steps))
    else:
        p = Procedure(goal="g", preconditions=[], steps=list(steps), map_name=map_name)
    p.mean_logprob = mean_logprob
    return p


def _mk_context(state_delta_observed=None, success=True):
    from agents.macla.macla_lib import ContrastiveContext

    ctx = ContrastiveContext(
        observation_init="",
        action_sequence=[],
        observation_term="",
        cumulative_reward=1.0 if success else -1.0,
        trajectory_id="t",
        success=success,
    )
    ctx.state_delta_observed = state_delta_observed
    return ctx


# ─── (Fix 1a) _state_delta_confidence: bootstrap-neutral ───────────────────


def test_sdc_bootstrap_neutral_when_no_contexts():
    """No data → 1.0 (no damping) instead of 0.5 (which silenced the proc)."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    assert sel._state_delta_confidence(entry) == 1.0


def test_sdc_bootstrap_neutral_below_threshold():
    """Stage N: < _SDC_BOOTSTRAP_N observations → bootstrap 1.0.
    A new proc with 1 or 2 observations must not be damped — it needs
    to fire to accumulate enough evidence to score."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    # 2 observations < _SDC_BOOTSTRAP_N (3) → still bootstrap, even though
    # both observations say "no state delta"
    entry.success_contexts = [_mk_context(state_delta_observed=False) for _ in range(2)]
    assert sel._state_delta_confidence(entry) == 1.0


def test_sdc_bootstrap_neutral_when_all_observations_are_none():
    """Unstructured-observation game (e.g. non-pokemon) → all-None →
    bootstrap 1.0. Without state-delta extraction we cannot score, so we
    must not damp."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    entry.success_contexts = [_mk_context(state_delta_observed=None) for _ in range(5)]
    assert sel._state_delta_confidence(entry) == 1.0


def test_sdc_calibrated_when_enough_observations():
    """≥ _SDC_BOOTSTRAP_N (3) observations → real rate."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    entry.success_contexts = [
        _mk_context(state_delta_observed=True),
        _mk_context(state_delta_observed=True),
        _mk_context(state_delta_observed=False),
    ]
    # 2/3 ≈ 0.667
    assert abs(sel._state_delta_confidence(entry) - 2.0 / 3.0) < 1e-9


def test_sdc_calibrated_full_high():
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    entry.success_contexts = [_mk_context(state_delta_observed=True) for _ in range(5)]
    assert sel._state_delta_confidence(entry) == 1.0


def test_sdc_calibrated_full_low():
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    entry.success_contexts = [_mk_context(state_delta_observed=False) for _ in range(5)]
    assert sel._state_delta_confidence(entry) == 0.0


# ─── (Fix 1b) _logprob_confidence: bootstrap-neutral ───────────────────────


def test_lpc_bootstrap_neutral_when_mean_logprob_is_none():
    """Pre-Stage-M procedure (no logprob captured) → 1.0, not 0.5."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure(mean_logprob=None))
    # Fill the deque so the OTHER bootstrap (sample-count) wouldn't fire
    mem._recent_logprobs.extend([-0.5] * 20)
    assert sel._logprob_confidence(entry) == 1.0


def test_lpc_bootstrap_neutral_when_deque_below_threshold():
    """< _LOGPROB_BOOTSTRAP_N samples → 1.0 (deque too small to rank)."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure(mean_logprob=-0.3))
    mem._recent_logprobs.extend([-0.5] * 5)  # 5 < 10
    assert sel._logprob_confidence(entry) == 1.0


def test_lpc_calibrated_returns_percentile_rank():
    """Once both gates pass, signal is real rank/N."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure(mean_logprob=-0.1))
    # 10 samples: nine at -1.0 (worse) and one at -0.05 (better)
    mem._recent_logprobs.extend([-1.0] * 9 + [-0.05])
    # mlp=-0.1: 9 of 10 are ≤ -0.1 → rank = 9/10 = 0.9
    assert sel._logprob_confidence(entry) == 0.9


# ─── (Fix 1c) EU compounding: brand-new procs are NOT damped ──────────────


def test_eu_not_damped_for_uncalibrated_procedure():
    """The point of the fix: a brand-new proc (no contexts, no logprob,
    empty deque) should have base EU unchanged, not 0.5625× damped."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
        Procedure,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    obs = "Map Name: Route1\nScore: 0\n"
    # High-alpha proc with matching context but zero history
    proc = Procedure(goal="g", preconditions=[], steps=["s"], alpha=10, beta=1, map_name="Route1")
    entry = ProceduralMemoryEntry(procedure=proc, contexts={"general"}, goals={"g"})

    eu_with_fix = sel._compute_expected_utility(entry, obs, "g")

    # Compare against base (no signals applied). Construct expected base
    # using the same helpers.
    relevance = sel._compute_relevance(entry, obs, "g")
    rho_mean = proc.alpha / (proc.alpha + proc.beta)
    risk = sel._compute_failure_risk(entry, obs, theta_risk=0.85)
    info_gain = sel._compute_information_gain(proc)
    base = (relevance * rho_mean * 1.0) - (risk * (1 - rho_mean) * 0.5) + 0.1 * info_gain
    base = max(0.0, base)

    assert abs(eu_with_fix - base) < 1e-9, (
        f"Stage N fix failed: uncalibrated proc damped from base={base:.4f} "
        f"to {eu_with_fix:.4f}. Bootstrap multipliers must return 1.0 (neutral)."
    )


def test_eu_damped_once_calibrated_with_negative_signal():
    """Sanity: once a proc has enough history showing no state-delta, EU
    should drop. The fix mustn't disable damping entirely."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
        Procedure,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    obs = "Map Name: Route1\nScore: 0\n"
    proc = Procedure(goal="g", preconditions=[], steps=["s"], alpha=10, beta=1, map_name="Route1")
    entry = ProceduralMemoryEntry(procedure=proc, contexts={"general"}, goals={"g"})
    # 5 success contexts all reporting NO state delta
    entry.success_contexts = [_mk_context(state_delta_observed=False) for _ in range(5)]

    base_uncalibrated = sel._compute_expected_utility(
        ProceduralMemoryEntry(procedure=proc, contexts={"general"}, goals={"g"}),
        obs,
        "g",
    )
    damped = sel._compute_expected_utility(entry, obs, "g")
    assert damped < base_uncalibrated, (
        f"Stage N: calibrated all-False sdc should still damp EU. "
        f"base={base_uncalibrated:.4f} damped={damped:.4f}"
    )


# ─── (Fix 2a) Selector no longer touches visited_maps ──────────────────────


def test_select_procedure_does_not_record_visit():
    """The visit-marking moves to the planner site so a planner that
    sees the novelty hint marks the map; the selector stays orthogonal."""
    from agents.macla.macla_lib import BayesianProcedureSelector, EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    sel.select_procedure("Map Name: PalletTown\nScore: 0\n", goal="g")
    assert mem.visited_maps == set(), (
        "Stage N: select_procedure must not mark maps as visited. "
        "That responsibility moves to the planner site."
    )


def test_select_procedure_no_new_map_theta_bump():
    """A high-EU proc must NOT be rejected on a new map purely because the
    map is unvisited. Stage M's theta-bump fired 0 times in 420 selector
    events; the bonus is moved to the planner."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        Procedure,
    )

    mem = EnhancedHierarchicalMemorySystem()
    proc = Procedure(goal="g", preconditions=[], steps=["s"], alpha=10, beta=1, map_name="Viridian")
    pk = mem.add_procedural_entry(proc, contexts={"general"}, goals={"g"}, performance=0.9)
    entry = mem.procedural_memory[pk]
    entry.contexts = {"general"}
    entry.goals = {"g"}

    sel = BayesianProcedureSelector(mem)
    obs_new_map = "Map Name: Viridian\nScore: 0\n"
    # Mark a different map as visited so Viridian is genuinely "new"
    mem.record_map_visit("Route1")

    selected, _ = sel.select_procedure(obs_new_map, goal="g", theta_conf=0.05)
    # We don't assert selected is not None — EU may legitimately fall below
    # 0.05 from the base formula. What we DO assert is that the rejection,
    # if any, comes from base EU vs theta_conf, NOT from a bumped 0.6 floor.
    # Equivalently: the selector must not look at is_new_map for its theta.
    # We assert this structurally by removing the attribute.
    assert not hasattr(sel, "_NEW_MAP_THETA"), (
        "Stage N: _NEW_MAP_THETA constant should be removed from selector."
    )


# ─── (Fix 2b) Memory exposes a planner-friendly novelty hint ───────────────


def test_map_visit_status_returns_hint_for_unvisited():
    """Planner asks the memory system whether the current map is novel."""
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    mem.record_map_visit("Route1")
    hint = mem.map_visit_status("Viridian")
    assert hint is not None
    assert "Viridian" in hint
    # Should communicate novelty in plain language for the LLM
    assert any(token in hint.lower() for token in ("new", "never", "first")), (
        f"hint should signal novelty in natural language for the LLM: {hint!r}"
    )


def test_map_visit_status_returns_none_for_visited():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    mem.record_map_visit("Route1")
    assert mem.map_visit_status("Route1") is None


def test_map_visit_status_returns_none_for_unknown_empty():
    """No information ≠ novelty."""
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    assert mem.map_visit_status("unknown") is None
    assert mem.map_visit_status("") is None
    assert mem.map_visit_status(None) is None


# ─── (Fix 2c) End-to-end planner wiring: hint reaches the planner ─────────


def test_planner_receives_novelty_hint_on_new_map():
    """Regression test for the smoke-run finding.

    The unified.py wiring must inject ``### Novelty`` into the history
    string passed to ``LLMSubtaskPlanner.plan()`` whenever the current map
    is unvisited. Mock the planner's plan() to capture its history kwarg,
    then drive a memory system with a fresh visited_maps set.
    """
    from unittest.mock import MagicMock

    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem, _extract_map_name

    # Simulate the relevant unified.py block in isolation. This mirrors
    # the wiring at agents/macla/unified.py:493-510 — if the structure of
    # that block changes, this test must change to follow.
    mem = EnhancedHierarchicalMemorySystem()
    planner = MagicMock()
    planner.plan.return_value = "explore north"

    observation = "Map Name: Route1\nScore: 0\nPosition: (3, 7)\n"
    history_str = "### Recent steps\n(no history yet)"

    # Replicate the unified.py block verbatim
    current_map = _extract_map_name(observation)
    novelty_hint = mem.map_visit_status(current_map)
    if novelty_hint:
        history_str = f"### Novelty\n{novelty_hint}\n\n{history_str}"
        mem.record_map_visit(current_map)
    planner.plan(goal="g", observation=observation, history=history_str)

    # The hint MUST have fired (Route1 was unvisited)
    assert current_map == "Route1"
    assert novelty_hint is not None
    # The planner MUST have seen the hint in its history kwarg
    call_kwargs = planner.plan.call_args.kwargs
    assert "### Novelty" in call_kwargs["history"]
    assert "Route1" in call_kwargs["history"]
    assert "NEW MAP" in call_kwargs["history"]
    # The map must now be marked visited so subsequent calls don't re-fire
    assert "Route1" in mem.visited_maps


def test_planner_does_not_re_fire_hint_on_revisit():
    """Second call to the same map must NOT inject the hint — fires once."""
    from unittest.mock import MagicMock

    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem, _extract_map_name

    mem = EnhancedHierarchicalMemorySystem()
    mem.record_map_visit("Route1")  # pretend prior call already fired
    planner = MagicMock()

    observation = "Map Name: Route1\nScore: 0\n"
    history_str = "### Recent steps\n(none)"

    current_map = _extract_map_name(observation)
    novelty_hint = mem.map_visit_status(current_map)
    if novelty_hint:
        history_str = f"### Novelty\n{novelty_hint}\n\n{history_str}"
        mem.record_map_visit(current_map)
    planner.plan(goal="g", observation=observation, history=history_str)

    assert novelty_hint is None
    call_kwargs = planner.plan.call_args.kwargs
    assert "### Novelty" not in call_kwargs["history"]


# ─── (Fix 2c) Module-level _extract_map_name is importable ─────────────────


def test_module_level_extract_map_name():
    """Stage N hoisted ``_extract_map_name`` to module-level so
    ``agents/macla/unified.py`` can resolve the current map name without
    coupling through the selector. The class method now delegates."""
    from agents.macla.macla_lib import BayesianProcedureSelector, _extract_map_name

    assert _extract_map_name("Map Name: Route1, (x_max, y_max): (10, 20)") == "Route1"
    assert _extract_map_name("No map info here") == "unknown"
    assert _extract_map_name("") == "unknown"
    assert _extract_map_name(None) == "unknown"

    # Class method still works (call sites at macla_lib.py:576/1447 unchanged)
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    sel = BayesianProcedureSelector(EnhancedHierarchicalMemorySystem())
    assert sel._extract_map_name("Map Name: PalletTown\n") == "PalletTown"
