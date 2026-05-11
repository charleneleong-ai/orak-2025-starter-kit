"""Tests for the plan-do-check loop + tool gating.

Two complementary validators run after the action LLM proposes an action:

1. ``ToolGateValidator`` — adapter-side **rule check** (cheap, deterministic).
   Catches obvious hallucinations like ``warp_with_warp_point(5, 4)`` when
   (5, 4) isn't actually a warp tile in the current observation. Game
   adapters opt-in by exporting ``validate_action(action, observation)``.

2. ``LLMPlanValidator`` — **subgoal-alignment check** (1 extra LLM call).
   Asks the LLM "is this action consistent with the current subgoal?"
   Catches actions that pass tool gating but ignore the planner's subgoal
   (the diagnosed failure mode behind PR #31's Stage C / C′++ wedges).

Both feed a shared retry loop in ``UnifiedMaclaAgent._base_fallback`` —
when validation fails, the critique is appended to the user prompt and
the action LLM is re-invoked. Bounded retries (default 2) cap cost.
"""

from __future__ import annotations

import textwrap


def _make_fake_llm(*responses: str):
    """Stub langchain LLM that returns the responses in order."""

    class _FakeMsg:
        def __init__(self, content: str):
            self.content = content

    class _FakeLLM:
        def __init__(self) -> None:
            self._responses = list(responses)
            self.invoke_count = 0

        def invoke(self, messages):
            self.invoke_count += 1
            return _FakeMsg(self._responses.pop(0) if self._responses else "")

    return _FakeLLM()


# ── ToolGateValidator ──────────────────────────────────────────────────


def test_tool_gate_validator_accepts_when_adapter_check_passes():
    from agents._cognitive import ToolGateValidator

    adapter_check = lambda action, obs: (True, "")
    v = ToolGateValidator(adapter_check=adapter_check)
    valid, reason = v.validate(action="warp_with_warp_point(7, 1)", observation="…")
    assert valid is True
    assert reason == ""


def test_tool_gate_validator_rejects_with_reason_for_retry_prompt():
    from agents._cognitive import ToolGateValidator

    def adapter_check(action, obs):
        return False, "Tile (5, 4) is not a WarpPoint — use move_to instead"

    v = ToolGateValidator(adapter_check=adapter_check)
    valid, reason = v.validate(action="warp_with_warp_point(5, 4)", observation="…")
    assert valid is False
    assert "not a WarpPoint" in reason


def test_tool_gate_validator_is_noop_without_adapter_check():
    """Adapters that don't export ``validate_action`` get pass-through behaviour."""
    from agents._cognitive import ToolGateValidator

    v = ToolGateValidator(adapter_check=None)
    valid, reason = v.validate(action="anything", observation="…")
    assert valid is True


# ── LLMPlanValidator ───────────────────────────────────────────────────


def test_llm_plan_validator_accepts_action_aligned_with_subgoal():
    from agents._cognitive import LLMPlanValidator

    llm = _make_fake_llm("### Verdict\nALIGNED\n### Critique\n(none)")
    v = LLMPlanValidator(llm)
    valid, critique = v.validate(
        action="warp_with_warp_point(3, 7)",
        subgoal="Leave the starter house via the exit door at (3, 7)",
        observation="…",
    )
    assert valid is True
    assert llm.invoke_count == 1


def test_llm_plan_validator_rejects_divergent_action_with_critique():
    from agents._cognitive import LLMPlanValidator

    llm = _make_fake_llm(
        "### Verdict\nDIVERGENT\n"
        "### Critique\nThe subgoal is to leave the house via (3, 7); "
        "warping to (7, 1) takes you upstairs, away from the exit."
    )
    v = LLMPlanValidator(llm)
    valid, critique = v.validate(
        action="warp_with_warp_point(7, 1)",
        subgoal="Leave the starter house via the exit door at (3, 7)",
        observation="…",
    )
    assert valid is False
    assert "takes you upstairs" in critique


def test_llm_plan_validator_handles_llm_failure_as_pass_through():
    """LLM exception → don't block the action. (Fail-open: better to let a
    possibly-bad action through than to hang the run on a transient LLM error.)"""
    from agents._cognitive import LLMPlanValidator

    class _FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("network blip")

    v = LLMPlanValidator(_FailingLLM())
    valid, critique = v.validate(action="any", subgoal="any", observation="any")
    assert valid is True
    assert critique == ""
    assert v.stats()["parse_failures"] >= 1


def test_llm_plan_validator_skips_when_subgoal_is_empty():
    """No subgoal to validate against → pass through without an LLM call."""
    from agents._cognitive import LLMPlanValidator

    llm = _make_fake_llm("never called")
    v = LLMPlanValidator(llm)
    valid, critique = v.validate(action="x", subgoal="", observation="o")
    assert valid is True
    assert llm.invoke_count == 0


# ── CompositeValidator ─────────────────────────────────────────────────


def test_composite_validator_short_circuits_on_first_rejection():
    """Tool gate runs first; if it rejects, plan check should NOT be called
    (saves the LLM call when the action is already provably wrong)."""
    from agents._cognitive import CompositeValidator, LLMPlanValidator, ToolGateValidator

    tool_gate = ToolGateValidator(adapter_check=lambda a, o: (False, "tile (5,4) is not walkable"))
    plan_llm = _make_fake_llm("never called")
    plan_check = LLMPlanValidator(plan_llm)
    composite = CompositeValidator([tool_gate, plan_check])
    valid, reason = composite.validate(action="move_to(5, 4)", subgoal="x", observation="o")
    assert valid is False
    assert "not walkable" in reason
    assert plan_llm.invoke_count == 0


def test_composite_validator_returns_valid_when_both_pass():
    from agents._cognitive import CompositeValidator, LLMPlanValidator, ToolGateValidator

    tool_gate = ToolGateValidator(adapter_check=lambda a, o: (True, ""))
    plan_check = LLMPlanValidator(_make_fake_llm("### Verdict\nALIGNED\n"))
    composite = CompositeValidator([tool_gate, plan_check])
    valid, reason = composite.validate(
        action="warp_with_warp_point(3, 7)", subgoal="leave", observation="o"
    )
    assert valid is True
    assert reason == ""


# ── LocalConfig wiring ─────────────────────────────────────────────────


def test_localconfig_declares_validator_fields():
    """pydantic ``extra='forbid'`` — the three new YAML keys must be declared."""
    from config.agent_config import LocalConfig

    c = LocalConfig(
        class_name="x",
        model="m",
        temperature=0.0,
        use_tool_gating=True,
        use_plan_check=True,
        validation_max_retries=3,
    )
    assert c.use_tool_gating is True
    assert c.use_plan_check is True
    assert c.validation_max_retries == 3


def test_localconfig_validator_fields_default_to_none_for_adapter_precedence():
    """Defaults are None so the per-game adapter recommendation can win
    (same precedence pattern as use_self_reflection in PR #64)."""
    from config.agent_config import LocalConfig

    c = LocalConfig(class_name="x", model="m", temperature=0.0)
    assert c.use_tool_gating is None
    assert c.use_plan_check is None
    assert c.validation_max_retries == 2  # concrete fallback; retries is a budget


# ── Game adapter validators (per-game tool gates) ──────────────────────


def _redshouse2f_obs() -> str:
    """Minimal pokemon obs with a known walkability map for testing."""
    return textwrap.dedent(
        """
        Map Name: RedsHouse2f, (x_max , y_max): (7, 7)
        Map type: reds_house
        Your position (x, y): (5, 4)

        Map on Screen:
        ( 1,  1): X	( 2,  1): X	( 3,  1): O	( 4,  1): O	( 5,  1): O	( 6,  1): O	( 7,  1): Warp→RedsHouse1f
        ( 3,  2): O	( 4,  2): O	( 5,  2): O	( 6,  2): O	( 7,  2): O
        """
    ).lstrip()


def test_pokemon_adapter_validates_warp_target():
    """``warp_with_warp_point(x, y)`` on a non-warp tile should be rejected."""
    from agents.pokemon_red import game_adapter

    obs = _redshouse2f_obs()
    # (7, 1) is the staircase WarpPoint → valid
    valid, _ = game_adapter.validate_action(
        "use_tool(warp_with_warp_point, (x_dest=7, y_dest=1))", obs
    )
    assert valid is True
    # (3, 1) is an open floor 'O' tile, not a warp → reject
    valid, reason = game_adapter.validate_action(
        "use_tool(warp_with_warp_point, (x_dest=3, y_dest=1))", obs
    )
    assert valid is False
    assert "warp" in reason.lower() or "WarpPoint" in reason


def test_pokemon_adapter_validates_move_to_walkability():
    """``move_to(x, y)`` to an 'X' (wall) tile should be rejected."""
    from agents.pokemon_red import game_adapter

    obs = _redshouse2f_obs()
    # (3, 1) is 'O' → walkable, accept
    valid, _ = game_adapter.validate_action("use_tool(move_to, (x_dest=3, y_dest=1))", obs)
    assert valid is True
    # (1, 1) is 'X' → wall, reject
    valid, reason = game_adapter.validate_action("use_tool(move_to, (x_dest=1, y_dest=1))", obs)
    assert valid is False
    assert "walkable" in reason.lower() or "wall" in reason.lower()


def test_pokemon_adapter_accepts_dialog_and_battle_tools_without_coord_check():
    """Coord-less tools (continue_dialog, battle moves) pass through."""
    from agents.pokemon_red import game_adapter

    obs = _redshouse2f_obs()
    valid, _ = game_adapter.validate_action("use_tool(continue_dialog, ())", obs)
    assert valid is True
    valid, _ = game_adapter.validate_action(
        "use_tool(interact_with_object, (object_name='SIGN_X'))", obs
    )
    assert valid is True


# ── UnifiedMaclaAgent wiring (source-inspection contract tests) ────────


def test_unified_agent_factory_consults_adapter_for_validator_recommendations():
    """``_maybe_init_action_validator`` reads adapter RECOMMENDED_* constants."""
    import inspect

    from agents.macla import unified

    src = inspect.getsource(unified.UnifiedMaclaAgent._maybe_init_action_validator)
    assert "RECOMMENDED_USE_TOOL_GATING" in src
    assert "RECOMMENDED_USE_PLAN_CHECK" in src


def test_unified_agent_base_fallback_wires_retry_on_validation_failure():
    """``_base_fallback`` reads ``self._action_validator`` and retries when
    validation rejects (up to validation_max_retries)."""
    import inspect

    from agents.macla import unified

    src = inspect.getsource(unified.UnifiedMaclaAgent._base_fallback)
    assert "_action_validator" in src
    assert "validation_max_retries" in src or "max_retries" in src


def test_pokemon_adapter_exports_validator_recommendations():
    """Pokemon adapter publishes the per-game default for plan-do-check."""
    from agents.pokemon_red import game_adapter

    assert game_adapter.RECOMMENDED_USE_TOOL_GATING is True
    assert game_adapter.RECOMMENDED_USE_PLAN_CHECK is True
