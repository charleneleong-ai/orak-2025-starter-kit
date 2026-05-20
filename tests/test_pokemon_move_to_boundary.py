"""Stage S (F4): ``move_to`` boundary detection.

Symptom from v2-n5 introspection: when the agent calls ``move_to(12, 0)``
to walk off the north edge of PalletTown, the walk DOES transition into
Route1 (the warp tile auto-fires), but the post-walk verdict checks
``(player_pos_x, player_pos_y) == target_coord`` against the new map's
entry coords — which never match the old map's target — so ``move_to``
returns ``(False, "Unable to move to (12, 0) after 3 attempts.")`` even
though the agent has successfully crossed into Route1.

F4 fix: classify the post-walk outcome via a pure helper. If the map
changed during the walk, report it as a structured success with the
old/new map and current position. The executor can then update its
subgoal stack (e.g. cascade-pop NavigateToMap(Route1)) instead of
treating the transition as a failure.

Pure helper means no mocking of the whole ``PokemonTools`` /
``self.agent.memory.state_dict`` graph.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Pokemon_tools.py's top-of-module `from mcp_game_servers...map_utils import *`
# requires the runtime package layout. Stub the broken parts + load the file
# directly (same pattern as test_pokemon_milestones.py).
_REPO = Path(__file__).resolve().parent.parent
_pkg = types.ModuleType("mcp_game_servers")
_pkg.__path__ = []
sys.modules.setdefault("mcp_game_servers", _pkg)
_map_utils_stub = types.ModuleType("mcp_game_servers.pokemon_red.game.utils.map_utils")
sys.modules.setdefault("mcp_game_servers.pokemon_red.game.utils.map_utils", _map_utils_stub)
_TOOLS_PATH = _REPO / "evaluation_utils/mcp_game_servers/pokemon_red/game/utils/pokemon_tools.py"
_spec = importlib.util.spec_from_file_location("pokemon_tools_under_test", _TOOLS_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
classify_post_move_outcome = _module.classify_post_move_outcome


class TestClassifyPostMoveOutcome:
    """``classify_post_move_outcome`` returns one of:
    - ``(True, "Successfully Move to …")`` — landed on target.
    - ``(True, "Crossed map boundary …")`` — F4: walked into adjacent map.
    - ``(False, "Interrupt by …")`` — dialog/battle interrupted the walk.
    - ``None`` — no verdict yet; the caller should retry."""

    def test_landed_on_target_is_success(self):
        ok, msg = classify_post_move_outcome(
            prev_map="PalletTown",
            current_map="PalletTown",
            current_pos=(12, 0),
            target=(12, 0),
            state="Field",
            x_dest=12,
            y_dest=0,
        )
        assert ok is True
        assert "Successfully Move to (12, 0)" in msg

    def test_map_changed_is_boundary_crossing_success(self):
        """v2 symptom: agent calls move_to(12, 0) at the Pallet→Route1
        edge. The walk warps the player to Route1's entry; the old
        verdict said False ("unable to move"). F4 says True with
        boundary-crossing context."""
        ok, msg = classify_post_move_outcome(
            prev_map="PalletTown",
            current_map="Route1",
            current_pos=(12, 18),  # Route1's entry coords, not the target
            target=(12, 0),
            state="Field",
            x_dest=12,
            y_dest=0,
        )
        assert ok is True
        # Structured message mentions both maps + the new position so
        # the executor / planner can react meaningfully.
        for fragment in ("Crossed", "PalletTown", "Route1", "(12, 18)"):
            assert fragment in msg

    def test_boundary_crossing_takes_priority_over_target_match(self):
        """If by coincidence the new map's entry coords equal the original
        target coords, the map-change is still the more informative
        signal — surface it as a boundary crossing, not as 'reached
        target' (the target was meant in prev_map, not current_map)."""
        ok, msg = classify_post_move_outcome(
            prev_map="PalletTown",
            current_map="Route1",
            current_pos=(12, 0),
            target=(12, 0),
            state="Field",
            x_dest=12,
            y_dest=0,
        )
        assert ok is True
        assert "Crossed" in msg

    @pytest.mark.parametrize(
        "state, fragment",
        [
            ("Dialog", "Dialog"),
            ("Battle", "Battle"),
            ("WildBattle", "Battle"),
        ],
    )
    def test_dialog_or_battle_interrupt_is_failure(self, state, fragment):
        ok, msg = classify_post_move_outcome(
            prev_map="PalletTown",
            current_map="PalletTown",
            current_pos=(5, 5),
            target=(12, 0),
            state=state,
            x_dest=12,
            y_dest=0,
        )
        assert ok is False
        assert fragment in msg

    def test_same_map_not_at_target_returns_none_for_retry(self):
        """Verdict 'no resolution yet — retry' is ``None``. The caller's
        ``for attempt in range(max_attempts)`` loop continues."""
        verdict = classify_post_move_outcome(
            prev_map="PalletTown",
            current_map="PalletTown",
            current_pos=(11, 5),  # walked partway, not at target, no interrupt
            target=(12, 0),
            state="Field",
            x_dest=12,
            y_dest=0,
        )
        assert verdict is None


class TestMoveToWiring:
    """Both sync ``PokemonTools.move_to`` and async ``move_to`` in the MCP
    variant must dispatch through ``classify_post_move_outcome`` — not
    re-implement the verdict inline. Source-level check (full move_to
    needs a live PyBoy environment to roundtrip)."""

    @pytest.mark.parametrize(
        "path",
        [
            "evaluation_utils/mcp_game_servers/pokemon_red/game/utils/pokemon_tools.py",
            "evaluation_utils/mcp_game_servers/pokemon_red/game/utils/pokemon_tools_mcp.py",
        ],
    )
    def test_move_to_dispatches_through_helper(self, path):
        src = (_REPO / path).read_text()
        assert "classify_post_move_outcome(" in src, (
            f"{path} must invoke classify_post_move_outcome rather than "
            "re-implementing the post-walk verdict inline."
        )
