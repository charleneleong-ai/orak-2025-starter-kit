"""Stage O: broaden procedure acquisition by state-delta.

Background
----------
Stage M/N trajectory introspection showed MACLA's procedural memory
plateaus at 4 procedures across 5 cumulative iters × 300 steps — all
rooted at one OaksLab tile (the Charmander nickname dialog). The
acquisition trigger at ``provide_feedback`` line 1381 only fires when
``actual_success=True``, and ``actual_success`` is set by the per-game
``ConfigSuccessDetector`` only on score-increase / progress-threshold /
keyword events. For pokemon_red, score changes only on M1-M7 milestone
crossings — so per 300-step episode the agent has roughly 1-4 procedure-
learning opportunities. That matches exactly what we observed:
procedures_learned = [1, 2, 3, 3, 4] across iters 1-5.

Stage N (PR #87) fixes *selection* (bootstrap-neutral signals + planner
novelty). It cannot grow the cache; it only changes which existing procs
fire and how the planner explores.

Stage O broadens *acquisition*: in addition to the score-increase signal,
learn a procedure whenever the executing step moved the salient game
state forward (``_state_delta_observed(obs, next_observation) is True``).
The delta signal is already computed and stored on every context (used
by ``_state_delta_confidence`` for selection scoring) — we just stop
gating acquisition on the rarer success signal.

Stage N is a *prerequisite* for Stage O: without bootstrap-neutral
signals, every newly-acquired proc would start at 0.5625× damped EU
and never fire to refine, so the cache would grow but stay frozen.
With Stage N, brand-new procs see base EU and earn their calibrated
sdc/lpc through repeated firing.
"""

from __future__ import annotations


def _mk_macla_agent():
    """Construct a minimal EnhancedMACLAAgent with no external deps.

    We test provide_feedback directly — no LLM, env, or game machinery
    is needed for the acquisition gate.
    """
    from agents.macla.macla_lib import EnhancedMACLAAgent

    return EnhancedMACLAAgent()


def _delta_obs():
    """A pair (obs, next_obs) whose salient state HAS moved forward.

    Different position → _state_delta_observed returns True.
    """
    obs = "Map Name: PalletTown\nScore: 0\nPosition: (5, 5)\nIn Battle: False\n"
    next_obs = "Map Name: PalletTown\nScore: 0\nPosition: (5, 4)\nIn Battle: False\n"
    return obs, next_obs


def _no_delta_obs():
    """A pair whose salient state did NOT move (same position, same score).

    Only the surrounding prose changes — _state_delta_observed returns False.
    """
    obs = "Map Name: PalletTown\nScore: 0\nPosition: (5, 5)\nIn Battle: False\n"
    next_obs = "You bumped into a wall.\nMap Name: PalletTown\nScore: 0\nPosition: (5, 5)\nIn Battle: False\n"
    return obs, next_obs


def _exec_result(obs, action_sequence=("up",), goal="explore PalletTown"):
    """Construct an execution_result dict for the fallback branch
    (selected_procedure=None means the LLM-fallback path was taken)."""
    return {
        "selected_procedure": None,
        "action_sequence": list(action_sequence),
        "goal": goal,
        "observation": obs,
        "trajectory_id": "test_traj",
        "obs_image": None,
        "reasoning": "(test)",
    }


# ─── (Acquisition) state_delta gate adds learning opportunities ────────────


def test_acquisition_learns_when_state_delta_observed_even_without_score():
    """The Stage O contract: actual_success=False (no score change) but a
    real game-state move should still capture the procedure. Pre-Stage-O
    this returned without learning anything."""
    agent = _mk_macla_agent()
    obs, next_obs = _delta_obs()
    pre_count = agent.stats["procedures_learned"]

    info = agent.provide_feedback(
        execution_result=_exec_result(obs),
        actual_success=False,
        next_observation=next_obs,
    )

    assert agent.stats["procedures_learned"] == pre_count + 1, (
        f"Stage O failed: state_delta=True but no procedure learned. info={info}"
    )
    assert info["type"] == "procedure_learned"


def test_acquisition_still_learns_on_actual_success_no_regression():
    """Don't break the existing score-increase trigger. actual_success=True
    must still create a procedure (regardless of state_delta value)."""
    agent = _mk_macla_agent()
    obs, next_obs = _delta_obs()
    pre_count = agent.stats["procedures_learned"]

    info = agent.provide_feedback(
        execution_result=_exec_result(obs),
        actual_success=True,
        next_observation=next_obs,
    )

    assert agent.stats["procedures_learned"] == pre_count + 1
    assert info["type"] == "procedure_learned"


def test_acquisition_does_not_learn_when_no_delta_and_no_success():
    """Negative case: no score change AND no salient state move → don't
    capture the procedure (it produced nothing of interest)."""
    agent = _mk_macla_agent()
    obs, next_obs = _no_delta_obs()
    pre_count = agent.stats["procedures_learned"]

    info = agent.provide_feedback(
        execution_result=_exec_result(obs),
        actual_success=False,
        next_observation=next_obs,
    )

    assert agent.stats["procedures_learned"] == pre_count, (
        f"Stage O over-fired: learned a procedure for a no-op step. info={info}"
    )
    assert info["type"] != "procedure_learned"


def test_acquisition_does_not_learn_when_unstructured_observation():
    """If the observation lacks salient key:value lines (non-pokemon games,
    battle screens), _state_delta_observed returns False — and we should
    fall back to the existing actual_success gate, NOT learn from noise."""
    agent = _mk_macla_agent()
    obs = "WILD RATTATA APPEARS!\nWhat will TACKLE do?"
    next_obs = "RATTATA used TAIL WHIP!\nWhat will TACKLE do?"
    pre_count = agent.stats["procedures_learned"]

    info = agent.provide_feedback(
        execution_result=_exec_result(obs),
        actual_success=False,
        next_observation=next_obs,
    )

    # Both observations have empty _extract_salient_state() → no delta
    # observed → no acquisition. Acquisition remains gated on actual_success
    # for non-structured observations.
    assert agent.stats["procedures_learned"] == pre_count
    assert info["type"] != "procedure_learned"


# ─── End-to-end: simulated multi-step trajectory grows the cache ──────────


def test_acquisition_grows_cache_over_simulated_trajectory():
    """The headline effect: a 10-step simulated walk through PalletTown
    (no score changes, only position changes) should grow the cache
    meaningfully under Stage O. Pre-Stage-O: 0 procs. Stage O: ≥ 5 procs
    (one per distinct state-delta step). Bounded by the
    add_procedural_entry dedupe — equivalent procs may merge."""
    agent = _mk_macla_agent()
    positions = [(5, 5), (5, 4), (5, 3), (4, 3), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4)]
    pre_count = agent.stats["procedures_learned"]

    prev_pos = None
    for pos in positions:
        if prev_pos is None:
            prev_pos = pos
            continue
        obs = f"Map Name: PalletTown\nScore: 0\nPosition: {prev_pos}\nIn Battle: False\n"
        next_obs = f"Map Name: PalletTown\nScore: 0\nPosition: {pos}\nIn Battle: False\n"
        action = (
            "up"
            if pos[1] < prev_pos[1]
            else "down"
            if pos[1] > prev_pos[1]
            else "left"
            if pos[0] < prev_pos[0]
            else "right"
        )
        agent.provide_feedback(
            execution_result=_exec_result(obs, action_sequence=(action,)),
            actual_success=False,
            next_observation=next_obs,
        )
        prev_pos = pos

    grown = agent.stats["procedures_learned"] - pre_count
    # With 9 distinct moves of true position delta, we should learn at
    # least 5 procs (allowing some merging by context).
    assert grown >= 5, (
        f"Stage O: expected ≥5 procs learned across 9 delta steps; got {grown}. "
        f"Pre-Stage-O this would have been 0."
    )
