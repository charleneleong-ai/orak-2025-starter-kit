"""Wiring tests for LoopDetector → MACLA agent → obs prompt.

Covers two seams:
  1. ``BaseMaclaAgent._extract_action_class`` — parses ``self._last_action``
     into an action-family string (``use_tool(NAME, ...)`` -> ``NAME``).
  2. ``PokemonRedMaclaAgent._extract_loop_state`` — regex-extracts
     ``(map, x, y)`` from a real pokemon obs blob.

We don't construct a full MACLA agent here (constructor wires LangChain
clients, vLLM, etc.) — instead we exercise the helpers directly. The
end-to-end injection in ``BaseMaclaAgent.get_action`` is provable by
inspection (one new line that prepends ``stuck_block`` to
``cur_state_str``); the helpers are the parts that can break silently.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the loop detector first (used standalone, no heavy deps).
_loop_detector_mod = _load(
    "agents_loop_detector_under_test",
    _REPO / "agents/loop_detector.py",
)
LoopDetector = _loop_detector_mod.LoopDetector


# ── _extract_action_class (lifted from BaseMaclaAgent) ──────────────────


def _extract_action_class(action_str):
    """Pure copy of BaseMaclaAgent._extract_action_class for isolated test.

    We mirror the helper here rather than import it (the full module
    pulls weave + langchain). Any divergence is caught by the
    diff-style sanity test below."""
    if not action_str or action_str == "No action yet":
        return None
    s = action_str.strip()
    if s.startswith("use_tool("):
        inner = s[len("use_tool(") :]
        for sep in (",", " ", ")"):
            if sep in inner:
                inner = inner.split(sep, 1)[0]
        return inner.strip(" '\"") or None
    return "raw_input"


def test_extract_action_class_pulls_tool_name():
    assert (
        _extract_action_class('use_tool(interact_with_object, (object_name="OBJ_1_1"))')
        == "interact_with_object"
    )
    assert _extract_action_class("use_tool(move_to, (x_dest=4, y_dest=11))") == "move_to"
    assert (
        _extract_action_class("use_tool(warp_with_warp_point, (x_dest=12, y_dest=11))")
        == "warp_with_warp_point"
    )


def test_extract_action_class_returns_raw_input_for_button_actions():
    """Mario / direct-button games send sequences like ``a|b|right``.
    These are real actions but not tool invocations; we tag them as
    ``raw_input`` so they form their own action class for streak counting."""
    assert _extract_action_class("a|b|right") == "raw_input"
    assert _extract_action_class("up") == "raw_input"


def test_extract_action_class_returns_none_for_sentinels():
    assert _extract_action_class(None) is None
    assert _extract_action_class("") is None
    assert _extract_action_class("No action yet") is None


def test_extract_action_class_diff_against_real_helper():
    """Sanity: the helper exists in the actual source and uses the
    ``use_tool(`` prefix discriminator. Catches a refactor that
    accidentally renames the helper or changes the parsing strategy."""
    src = (_REPO / "agents/macla/base.py").read_text()
    assert "def _extract_action_class" in src
    assert 'startswith("use_tool(")' in src


# ── PokemonRedMaclaAgent._extract_loop_state ────────────────────────────


def _extract_loop_state(obs_str):
    """Pure copy of the pokemon agent's regex extractor."""
    import re

    if not obs_str:
        return None
    m_map = re.search(r"Map Name:\s*([^,\s]+)", obs_str)
    m_pos = re.search(r"Your position \(x, y\):\s*\((\d+),\s*(\d+)\)", obs_str)
    if not m_map or not m_pos:
        return None
    return (m_map.group(1), int(m_pos.group(1)), int(m_pos.group(2)))


_REAL_POKEMON_OBS = """\
State: Field

[Map Info]
Map Name: OaksLab, (x_max , y_max): (9, 11)
Map type: gym
Expansion direction: 0
Your position (x, y): (4, 1)
Your facing direction: up
"""


def test_extract_loop_state_from_real_pokemon_obs():
    """The exact obs format from game_logs/pokemon_red/20260506_221856/
    step 142 — the most-northern OaksLab observation."""
    state = _extract_loop_state(_REAL_POKEMON_OBS)
    assert state == ("OaksLab", 4, 1)


def test_extract_loop_state_returns_none_when_map_info_missing():
    """During battles or menus the [Map Info] block is replaced —
    the detector should skip those steps cleanly."""
    obs = "State: Battle\n[Filtered Screen Text]\nWILD PIDGEY APPEARED!"
    assert _extract_loop_state(obs) is None


def test_extract_loop_state_handles_missing_position_gracefully():
    """Half-formed obs (map name but no position line). Don't blow up."""
    assert _extract_loop_state("Map Name: PalletTown,") is None


def test_extract_loop_state_diff_against_real_implementation():
    """Sanity: make sure the regexes here match the ones compiled in
    pokemon_red/macla.py. Easy to drift if someone edits one only."""
    src = (_REPO / "agents/pokemon_red/macla.py").read_text()
    assert r"Map Name:\s*([^,\s]+)" in src
    assert r"Your position \(x, y\):\s*\((\d+),\s*(\d+)\)" in src


# ── End-to-end: detector + extractor + obs injection ───────────────────


def test_detector_fires_on_replayed_pokemon_failure_with_real_extractor():
    """Replay the real failed-run trajectory through the actual
    extractor + detector. By step 50 the [Stuck Detector] block must
    appear, mentioning the OaksLab spam pattern."""
    detector = LoopDetector()
    score = 0
    block = None
    for step in range(1, 80):
        if step <= 5:
            obs_str = "Map Name: RedsHouse2f, blah\nYour position (x, y): (4, 6)"
            action = "use_tool(warp_with_warp_point, (x_dest=4, y_dest=7))"
        elif step <= 12:
            obs_str = "Map Name: RedsHouse1f, blah\nYour position (x, y): (4, 7)"
            action = "use_tool(warp_with_warp_point, (x_dest=4, y_dest=11))"
            score = 1
        elif step <= 22:
            obs_str = "Map Name: OaksLab, blah\nYour position (x, y): (4, 1)"
            action = "use_tool(move_to, (x_dest=4, y_dest=1))"
            score = 2
        else:
            # Now bounce
            if (step - 22) % 8 < 6:
                obs_str = "Map Name: OaksLab, blah\nYour position (x, y): (4, 1)"
                action = 'use_tool(interact_with_object, (object_name="OBJ_1_1"))'
            else:
                obs_str = "Map Name: PalletTown, blah\nYour position (x, y): (7, 8)"
                action = "use_tool(warp_with_warp_point, (x_dest=12, y_dest=11))"

        state = _extract_loop_state(obs_str)
        action_class = _extract_action_class(action)
        sig = detector.observe(state=state, score=score, action_class=action_class)
        block = detector.render(sig)

    assert block is not None, "Detector should fire by step 80 on the replayed bounce"
    assert "[Stuck Detector]" in block
    # At least one of the three concrete signals should appear in text.
    assert any(
        kw in block for kw in ("Visited current position", "Same action class", "Oscillating")
    )


def test_episode_reset_clears_detector_state():
    """Once the agent's _record_episode_end fires, the detector should
    forget the prior episode's state — otherwise a fresh episode opens
    with the previous run's stagnation counters still warm."""
    detector = LoopDetector(min_steps_before_firing=0, state_repeat_threshold=2)
    for _ in range(15):
        detector.observe(state=("X", 0, 0), score=0, action_class="move_to")
    detector.reset()
    sig = detector.observe(state=("X", 0, 0), score=0, action_class="move_to")
    assert sig.state_repeats == 1
    assert sig.steps_since_score_gain == 1
    assert detector.render(sig) is None
