"""Stage M: multi-signal procedure quality + exploration novelty.

Background
----------
Stage L (PR #85, `docs/experiments/stage_l_map_aware/n5_rerun.md`) confirmed
map-aware procedure keys + iter-based TTL deliver monotonic M4 banking
speedup (259 → 229 → 172 → 140 steps across the four passing iters) but
do NOT lift the 57.14% ceiling — zero Viridian (M5) steps across all 5 iters
× 300 steps. The remaining ceiling is past M4 in the M5 entry gate, not
in procedure-cache context.

Two cognitive failures observed in iter-1 / iter-3 game_states.jsonl
deep-dives:

1. Wild-encounter loop on Route1 drains the step budget — agent repeats
   actions that produced zero salient state delta.
2. Battle policy thrash — LEECH SEED spam (doesn't model once-per-target
   debuff), RUN from full-HP lv3 wilds.

Generalised across pokemon / mario / 2048 / starcraft, the failure is:
agent does not reason about which action is producing forward progress.

Stage M is two-pronged:

(a) **Multi-signal procedure quality** via `state_delta_confidence`:
    score = base_posterior × state_delta_confidence. When a procedure's
    historical executions produced no salient state change (state delta
    observed = False), the procedure is downweighted in selection. Calibrated
    from `Procedure.state_delta_rate` over the entry's success_contexts.

(b) **Exploration novelty bonus** for unvisited maps:
    `EnhancedHierarchicalMemorySystem` tracks `visited_maps`. When the
    current map is new, `select_procedure` raises the effective theta_conf
    so cached procedures rarely fire, letting the LLM explore. Persists
    via the existing pickle checkpoint.

Both signals are generalisable. Salient-state extraction parses `Key: Value`
lines from the observation for keys typically tracked across games (score,
hp, position, map, in battle, minerals, gas, supply, lives, x, y, board).
When extraction yields nothing (non-structured observation), state_delta is
recorded as `None` and the procedure's confidence falls back to neutral 0.5.
"""

from __future__ import annotations


def _mk_procedure(steps=("a",), map_name=None):
    from agents.macla.macla_lib import Procedure

    if map_name is None:
        return Procedure(goal="g", preconditions=[], steps=list(steps))
    return Procedure(goal="g", preconditions=[], steps=list(steps), map_name=map_name)


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


# ─── Salient state extractor ──────────────────────────────────────────────


def test_salient_state_empty_observation_returns_empty():
    from agents.macla.macla_lib import _extract_salient_state

    assert _extract_salient_state("") == ()
    assert _extract_salient_state(None) == ()


def test_salient_state_extracts_pokemon_keys():
    from agents.macla.macla_lib import _extract_salient_state

    obs = "Some prose\nScore: 4\nHP: 18/24\nMap Name: Route1\nPosition: (9, 28)\nIn Battle: False\n"
    extracted = _extract_salient_state(obs)
    # Lines containing salient keys are extracted in order
    assert any("Score: 4" in line for line in extracted)
    assert any("HP: 18/24" in line for line in extracted)
    assert any("Map Name: Route1" in line for line in extracted)
    assert any("Position: (9, 28)" in line for line in extracted)
    assert any("In Battle: False" in line for line in extracted)


def test_salient_state_detects_change_when_score_increments():
    from agents.macla.macla_lib import _extract_salient_state

    before = "Score: 4\nMap Name: Route1\nPosition: (9, 28)\n"
    after = "Score: 5\nMap Name: Route1\nPosition: (9, 28)\n"
    assert _extract_salient_state(before) != _extract_salient_state(after)


def test_salient_state_detects_no_change_when_only_prose_changes():
    from agents.macla.macla_lib import _extract_salient_state

    before = "Just walked.\nScore: 4\nMap Name: Route1\nPosition: (9, 28)\n"
    after = "Bumped into a wall.\nScore: 4\nMap Name: Route1\nPosition: (9, 28)\n"
    assert _extract_salient_state(before) == _extract_salient_state(after)


def test_salient_state_handles_unstructured_observation():
    from agents.macla.macla_lib import _extract_salient_state

    # Non-key-value observation (e.g. battle dialog) yields empty
    assert _extract_salient_state("WILD RATTATA APPEARS!\nWhat will TACKLE do?") == ()


# ─── ContrastiveContext.state_delta_observed ──────────────────────────────


def test_contrastive_context_has_state_delta_observed_default_none():
    from agents.macla.macla_lib import ContrastiveContext

    ctx = ContrastiveContext(
        observation_init="a",
        action_sequence=[],
        observation_term="b",
        cumulative_reward=0.0,
        trajectory_id="t",
        success=True,
    )
    assert ctx.state_delta_observed is None


# ─── Procedure.state_delta_rate / state_delta_confidence ──────────────────


def test_state_delta_confidence_neutral_when_no_contexts():
    """No data → 0.5 (neutral)."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    assert sel._state_delta_confidence(entry) == 0.5


def test_state_delta_confidence_high_when_all_contexts_observe_delta():
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    entry.success_contexts = [_mk_context(state_delta_observed=True) for _ in range(4)]
    assert sel._state_delta_confidence(entry) == 1.0


def test_state_delta_confidence_low_when_no_contexts_observe_delta():
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    entry.success_contexts = [_mk_context(state_delta_observed=False) for _ in range(4)]
    assert sel._state_delta_confidence(entry) == 0.0


def test_state_delta_confidence_ignores_none_contexts():
    """Contexts captured without salient-state info (non-structured obs)
    do not count in the rate — they contribute to fallback bootstrap."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    # 1 True + 1 None → rate = 1/1 (None excluded)
    entry.success_contexts = [
        _mk_context(state_delta_observed=True),
        _mk_context(state_delta_observed=None),
    ]
    assert sel._state_delta_confidence(entry) == 1.0


def test_state_delta_confidence_neutral_when_all_none():
    """All-None contexts (non-structured game) → 0.5 (neutral fallback)."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    entry = ProceduralMemoryEntry(procedure=_mk_procedure())
    entry.success_contexts = [_mk_context(state_delta_observed=None) for _ in range(4)]
    assert sel._state_delta_confidence(entry) == 0.5


def test_state_delta_confidence_partial():
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
        _mk_context(state_delta_observed=False),
    ]
    assert sel._state_delta_confidence(entry) == 0.5


# ─── Visited maps tracking ────────────────────────────────────────────────


def test_visited_maps_empty_on_init():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    assert mem.visited_maps == set()


def test_record_map_visit_adds_to_set():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    mem.record_map_visit("Route1")
    mem.record_map_visit("OaksLab")
    assert mem.visited_maps == {"Route1", "OaksLab"}


def test_record_map_visit_ignores_unknown_and_empty():
    """'unknown' / empty / None must not pollute visited_maps — they
    represent absence-of-info, not a new map."""
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    mem.record_map_visit("unknown")
    mem.record_map_visit("")
    mem.record_map_visit(None)
    assert mem.visited_maps == set()


def test_is_new_map_true_for_unvisited():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    mem.record_map_visit("Route1")
    assert mem.is_new_map("Viridian") is True


def test_is_new_map_false_for_visited():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    mem.record_map_visit("Route1")
    assert mem.is_new_map("Route1") is False


def test_is_new_map_false_for_unknown_and_empty():
    """'unknown' is information absence, not a new map — never treat as novel."""
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    assert mem.is_new_map("unknown") is False
    assert mem.is_new_map("") is False
    assert mem.is_new_map(None) is False


# ─── select_procedure: novelty bonus suppresses cached procs on new maps ──


def test_select_procedure_raises_theta_on_new_map():
    """On an unvisited map, cached procedures with marginal EU should be
    rejected so the LLM explores. A high-EU candidate that would pass the
    default theta should be filtered out by the bumped theta on a new map.
    """
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        Procedure,
    )

    mem = EnhancedHierarchicalMemorySystem()
    # Seed a high-prior procedure that would normally fire
    proc = Procedure(goal="g", preconditions=[], steps=["s"], alpha=10, beta=1, map_name="unknown")
    pk = mem.add_procedural_entry(proc, contexts={"general"}, goals={"g"}, performance=0.9)
    entry = mem.procedural_memory[pk]
    entry.contexts = {"general"}
    entry.goals = {"g"}

    sel = BayesianProcedureSelector(mem)
    obs_known_map = "Map Name: Route1\nScore: 0\n"
    obs_new_map = "Map Name: Viridian\nScore: 0\n"

    mem.record_map_visit("Route1")  # mark Route1 as visited

    # On a known map: procedure may fire (we just need it to not be blocked
    # by the novelty path — the actual selection depends on EU which we
    # don't assert here; just check that the new-map branch is the one
    # bumping theta).
    sel.select_procedure(obs_known_map, goal="g", theta_conf=0.25)

    # On the NEW map, raising effective theta to >= 0.6 should reject
    # the marginal procedure even though its EU passed the default 0.25.
    selected_new, conf_new = sel.select_procedure(obs_new_map, goal="g", theta_conf=0.25)
    assert selected_new is None, (
        f"Stage M (b) failed: cached procedure fired on new map ({selected_new=}, "
        f"{conf_new=}). On unvisited maps theta should rise to >= 0.6 to bias "
        f"the agent toward LLM-driven exploration."
    )


def test_select_procedure_records_visit():
    """After select_procedure runs, the current map should be in visited_maps."""
    from agents.macla.macla_lib import BayesianProcedureSelector, EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    sel.select_procedure("Map Name: PalletTown\nScore: 0\n", goal="g")
    assert "PalletTown" in mem.visited_maps


def test_visited_maps_survive_pickle_roundtrip():
    """Cumulative memory checkpoint must carry visited_maps across iters."""
    import pickle

    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    mem.record_map_visit("Route1")
    mem.record_map_visit("PalletTown")

    blob = pickle.dumps(mem)
    restored = pickle.loads(blob)
    assert restored.visited_maps == {"Route1", "PalletTown"}


# ─── End-to-end: state_delta_confidence enters expected-utility ───────────


def test_expected_utility_penalised_when_state_delta_zero():
    """A procedure with all-False state_delta in its success history
    should have lower EU than one with all-True, holding everything else
    equal."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        ContrastiveContext,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
        Procedure,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)

    obs = "Map Name: Route1\nScore: 0\n"

    def _mk_entry(delta_observed: bool) -> ProceduralMemoryEntry:
        p = Procedure(goal="g", preconditions=[], steps=["s"], alpha=5, beta=1, map_name="Route1")
        e = ProceduralMemoryEntry(procedure=p, contexts={"general"}, goals={"g"})
        for _ in range(4):
            ctx = ContrastiveContext(
                observation_init=obs,
                action_sequence=[],
                observation_term=obs,
                cumulative_reward=1.0,
                trajectory_id="t",
                success=True,
            )
            ctx.state_delta_observed = delta_observed
            e.success_contexts.append(ctx)
        return e

    eu_high = sel._compute_expected_utility(_mk_entry(True), obs, "g")
    eu_low = sel._compute_expected_utility(_mk_entry(False), obs, "g")
    assert eu_low < eu_high, (
        f"Stage M (a) failed: state_delta=False should downweight EU. "
        f"got eu_high={eu_high:.3f} eu_low={eu_low:.3f}"
    )
