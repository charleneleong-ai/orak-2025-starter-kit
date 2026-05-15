"""Tests for the StarCraft game adapter wired into UnifiedMaclaAgent.

StarCraft's contract is unusual: each step emits exactly 5 actions encoded
as ``"1: ACTION_A\n2: ACTION_B\n..."``. The mcp_game_servers env parses
``\\d+: <NAME>`` from the action text and maps each name through its
72-entry Protoss action dictionary, so the adapter must produce that exact
shape and survive UnifiedMaclaAgent._validate_action() unchanged.

These tests pin the contract that:
1. The adapter is registered under "star_craft" in GAME_ADAPTERS
2. extract_action() formats StarCraftAction.actions as numbered multi-line
3. _validate_action() passes the multi-action format through
4. calculate_metrics() lifts the resource/supply fields game_info exposes
5. VALID_ACTIONS includes the Protoss vocabulary that the env can resolve
"""

from __future__ import annotations

import importlib

from agents.macla.unified import GAME_ADAPTERS


def test_starcraft_adapter_registered():
    assert GAME_ADAPTERS["star_craft"] == "agents.starcraft.game_adapter"
    adapter = importlib.import_module(GAME_ADAPTERS["star_craft"])
    assert hasattr(adapter, "SYSTEM_PROMPT")
    assert hasattr(adapter, "USER_PROMPT_TEMPLATE")
    assert hasattr(adapter, "extract_action")
    assert hasattr(adapter, "calculate_metrics")
    assert hasattr(adapter, "VALID_ACTIONS")
    assert hasattr(adapter, "DEFAULT_ACTION")


def test_extract_action_formats_five_actions_as_numbered_lines():
    from agents.starcraft.game_adapter import StarCraftAction, extract_action

    result = StarCraftAction(
        reasoning="Open with worker production then a pylon for supply.",
        current_goal="Expand economy",
        actions=["TRAIN PROBE", "BUILD PYLON", "TRAIN PROBE", "EMPTY ACTION", "EMPTY ACTION"],
    )
    rendered = extract_action(result)
    assert rendered == (
        "1: TRAIN PROBE\n"
        "2: BUILD PYLON\n"
        "3: TRAIN PROBE\n"
        "4: EMPTY ACTION\n"
        "5: EMPTY ACTION"
    )


def test_extract_action_round_trips_through_env_regex():
    """The env parses actions via r"\\d+: <?([^>\\n]+)>?" — verify our output
    survives that regex unchanged so each line maps to a real action_dict key."""
    import re

    from agents.starcraft.game_adapter import StarCraftAction, extract_action

    rendered = extract_action(
        StarCraftAction(
            reasoning="r",
            current_goal="g",
            actions=["BUILD GATEWAY", "TRAIN ZEALOT", "TRAIN ZEALOT", "MULTI-ATTACK", "EMPTY ACTION"],
        )
    )
    parsed = re.findall(r"\d+: <?([^>\n]+)>?", rendered)
    assert parsed == [
        "BUILD GATEWAY",
        "TRAIN ZEALOT",
        "TRAIN ZEALOT",
        "MULTI-ATTACK",
        "EMPTY ACTION",
    ]


def test_validate_action_passes_starcraft_multi_action_through():
    """The 5-action newline format must survive _validate_action so the env's
    text2action() gets the full payload instead of just DEFAULT_ACTION."""
    from agents.macla.unified import _STARCRAFT_MULTI_ACTION_RE

    multi_action = "1: TRAIN PROBE\n2: BUILD PYLON\n3: EMPTY ACTION\n4: EMPTY ACTION\n5: EMPTY ACTION"
    assert _STARCRAFT_MULTI_ACTION_RE.match(multi_action) is not None

    # A leading-whitespace variant — Pydantic structured-output models
    # sometimes emit one — should still match.
    assert _STARCRAFT_MULTI_ACTION_RE.match("  1: TRAIN PROBE\n2: BUILD PYLON") is not None

    # A bare action ("EMPTY ACTION" alone) must NOT match the multi-line
    # gate — DEFAULT_ACTION fallback should still handle that path.
    assert _STARCRAFT_MULTI_ACTION_RE.match("EMPTY ACTION") is None


def test_calculate_metrics_lifts_resource_and_supply_fields():
    from agents.starcraft.game_adapter import calculate_metrics

    game_info = {
        "minerals": 250,
        "vespene": 100,
        "supply_cap": 23,
        "supply_used": 14,
        "supply_left": 9,
        "worker_supply": 12,
        "army_supply": 2,
        "game_time": "02:30",
        "evaluation_score": 0.0,
        "unrelated_field": "should be dropped",
    }
    metrics = calculate_metrics(game_info)
    assert metrics["minerals"] == 250
    assert metrics["vespene"] == 100
    assert metrics["supply_left"] == 9
    assert metrics["worker_supply"] == 12
    assert metrics["game_time"] == "02:30"
    assert metrics["evaluation_score"] == 0.0
    assert "unrelated_field" not in metrics


def test_calculate_metrics_handles_missing_evaluation_score():
    """Pre-victory steps don't have evaluation_score; metrics should still
    populate the fields that ARE present without raising."""
    from agents.starcraft.game_adapter import calculate_metrics

    metrics = calculate_metrics({"minerals": 50, "supply_left": 0})
    assert metrics["minerals"] == 50
    assert metrics["supply_left"] == 0
    assert "evaluation_score" not in metrics


def test_valid_actions_includes_protoss_vocabulary():
    """VALID_ACTIONS must be a superset of the action names the Protoss env
    routes through its action_dict. Failing this would force every action
    through the DEFAULT_ACTION fallback inside _validate_action."""
    from agents.starcraft.game_adapter import VALID_ACTIONS

    expected = {
        "TRAIN PROBE",
        "TRAIN ZEALOT",
        "BUILD PYLON",
        "BUILD GATEWAY",
        "RESEARCH WARPGATERESEARCH",
        "RESEARCH CHARGE",
        "SCOUTING PROBE",
        "MULTI-ATTACK",
        "CHRONOBOOST NEXUS",
        "EMPTY ACTION",
    }
    missing = expected - set(VALID_ACTIONS)
    assert not missing, f"Protoss vocabulary missing from VALID_ACTIONS: {missing}"


def test_user_prompt_template_uses_only_standard_placeholders():
    """UnifiedMaclaAgent.format_map() only fills cur_state_str / last_action /
    task_description / prev_state_str. Any other placeholder would render as
    an empty string via SafeDict — verify the adapter sticks to the four."""
    import re

    from agents.starcraft.game_adapter import USER_PROMPT_TEMPLATE

    placeholders = set(re.findall(r"\{([a-zA-Z_]\w*)\}", USER_PROMPT_TEMPLATE))
    allowed = {"cur_state_str", "last_action", "task_description", "prev_state_str"}
    extra = placeholders - allowed
    assert not extra, f"USER_PROMPT_TEMPLATE has non-standard placeholders: {extra}"
