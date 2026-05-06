"""Regression tests for the ``_is_warp`` predicate in pokemon_tools.

PR #44 added warp-destination labels (``"Warp→RedsHouse1f"``) to the
rendered map so the agent can tell a staircase from an exit door. The
rendered text is parsed back into ``explored_map`` by
``construct_init_map``, so cells that used to read ``"WarpPoint"`` now
read ``"Warp→<dest>"``. Eight downstream call sites in
``pokemon_tools.py`` were comparing against the bare string and silently
rejecting all warp tiles — the Stage A retry got 0.0 over 300 steps,
spending 119 of those calling ``warp_with_warp_point(7, 1)`` against an
explored_map cell that read ``"Warp→RedsHouse1f"``.

These tests pin the predicate behaviour and prove the regression is
fixed by exercising the actual ``can_land`` / pathfinding code paths
that were broken.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Stub the parent package so importing pokemon_tools doesn't drag in the
# runtime-only mcp_game_servers chain.
_pkg_root = types.ModuleType("mcp_game_servers")
_pkg_root.__path__ = []
sys.modules.setdefault("mcp_game_servers", _pkg_root)

_pokemon_pkg = types.ModuleType("mcp_game_servers.pokemon_red")
_pokemon_pkg.__path__ = []
sys.modules.setdefault("mcp_game_servers.pokemon_red", _pokemon_pkg)
_pokemon_game_pkg = types.ModuleType("mcp_game_servers.pokemon_red.game")
_pokemon_game_pkg.__path__ = []
sys.modules.setdefault("mcp_game_servers.pokemon_red.game", _pokemon_game_pkg)
_utils_pkg = types.ModuleType("mcp_game_servers.pokemon_red.game.utils")
_utils_pkg.__path__ = []
sys.modules.setdefault("mcp_game_servers.pokemon_red.game.utils", _utils_pkg)

# pokemon_tools imports map_utils via `from ... import *`; load that
# real module first so the star-import resolves cleanly.
_MAP_UTILS_PATH = (
    _REPO
    / "evaluation_utils/mcp_game_servers/pokemon_red/game/utils/map_utils.py"
)
_map_spec = importlib.util.spec_from_file_location(
    "mcp_game_servers.pokemon_red.game.utils.map_utils", _MAP_UTILS_PATH
)
_map_mod = importlib.util.module_from_spec(_map_spec)
sys.modules["mcp_game_servers.pokemon_red.game.utils.map_utils"] = _map_mod
_map_spec.loader.exec_module(_map_mod)

_TOOLS_PATH = (
    _REPO
    / "evaluation_utils/mcp_game_servers/pokemon_red/game/utils/pokemon_tools.py"
)
_spec = importlib.util.spec_from_file_location(
    "mcp_game_servers.pokemon_red.game.utils.pokemon_tools", _TOOLS_PATH
)
pokemon_tools = importlib.util.module_from_spec(_spec)
sys.modules["mcp_game_servers.pokemon_red.game.utils.pokemon_tools"] = pokemon_tools
_spec.loader.exec_module(pokemon_tools)

_is_warp = pokemon_tools._is_warp
PokemonToolset = pokemon_tools.PokemonToolset


# ── _is_warp predicate ──


def test_is_warp_accepts_legacy_label():
    assert _is_warp("WarpPoint")


def test_is_warp_accepts_enriched_label():
    assert _is_warp("Warp→RedsHouse1f")
    assert _is_warp("Warp→PalletTown")


def test_is_warp_rejects_walkable_tiles():
    for tile in ("O", "G", "X", "~", "?", "D", "L", "R", "C"):
        assert not _is_warp(tile)


def test_is_warp_rejects_sprite_labels():
    assert not _is_warp("SPRITE_OAK")
    assert not _is_warp("Warpish")  # almost-but-not


def test_is_warp_rejects_non_strings():
    assert not _is_warp(None)
    assert not _is_warp(42)
    assert not _is_warp([])


# ── regression: warp tool precondition accepts enriched cell ──


class _StubRunner:
    quit_flag = False


class _StubMemory:
    def __init__(self, explored_map, map_name="RedsHouse2f", x=5, y=4):
        self.state_dict = {
            "state": "Field",
            "map_info": {
                "map_name": map_name,
                "player_pos_x": x,
                "player_pos_y": y,
                "x_max": len(explored_map[0]) - 1,
                "y_max": len(explored_map) - 1,
                "expansion_direction": "$00",
                "map_type": "reds_house",
            },
        }
        self.map_memory_dict = {map_name: {"explored_map": explored_map, "history": []}}
        self.dialog_buffer: list = []


class _StubAgent:
    def __init__(self, explored_map, **kw):
        self.memory = _StubMemory(explored_map, **kw)
        self.env = types.SimpleNamespace(
            runner=_StubRunner(),
            send_action_set=lambda *_a, **_kw: None,
            _send_action=lambda *_a, **_kw: None,
        )


def _redshouse2f_grid_with_warp_label(label: str) -> list[list[str]]:
    """Mimic the RedsHouse2f explored_map after construct_init_map parses
    a rendered map_screen_raw. Cell (7, 1) is the staircase warp tile."""
    grid = [
        ["X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "O", "O", "O", "O", label],
        ["X", "O", "O", "O", "O", "O", "O", "O"],
        ["X", "O", "O", "O", "O", "O", "O", "O"],
        ["X", "O", "O", "X", "O", "O", "O", "O"],
        ["X", "O", "O", "X", "O", "O", "O", "O"],
        ["X", "O", "O", "O", "O", "O", "X", "O"],
        ["X", "O", "O", "O", "O", "O", "X", "O"],
    ]
    return grid


def test_warp_precondition_accepts_legacy_label():
    """Sanity: the legacy ``WarpPoint`` literal still works."""
    grid = _redshouse2f_grid_with_warp_label("WarpPoint")
    agent = _StubAgent(grid)
    toolset = PokemonToolset(agent)
    explored = grid
    assert _is_warp(explored[1][7])  # precondition the tool checks


def test_warp_precondition_accepts_enriched_label():
    """The bug: PR #44 enriched cell becomes 'Warp→RedsHouse1f'.

    Without the fix, ``warp_with_warp_point`` would return
    ``"(7, 1) is not 'WarpPoint'"`` and the agent gets stuck — exactly
    the Stage A 0.0 failure mode."""
    grid = _redshouse2f_grid_with_warp_label("Warp→RedsHouse1f")
    agent = _StubAgent(grid)
    toolset = PokemonToolset(agent)
    assert _is_warp(grid[1][7])
    # Path search must also treat (7, 1) as walkable; without the fix,
    # the A* in _find_path_inner excluded enriched warp cells from the
    # walkable set so the destination was unreachable.
    success, _ = toolset._find_path_inner(7, 1)
    assert success, "pathfinding should reach the enriched warp tile"


def test_pathfinding_treats_enriched_warp_as_walkable_destination():
    """Reachability test: from (5, 4) to the warp at (7, 1)."""
    grid = _redshouse2f_grid_with_warp_label("Warp→RedsHouse1f")
    agent = _StubAgent(grid, x=5, y=4)
    toolset = PokemonToolset(agent)
    success, directions = toolset._find_path_inner(7, 1)
    assert success
    # We expect a sequence of cardinal moves; sanity-check that the
    # path is non-empty.
    assert any(d in directions for d in ("up", "down", "left", "right"))


def test_move_to_redirects_when_destination_is_enriched_warp():
    """``move_to`` should send the agent to ``warp_with_warp_point``
    when the destination is a warp tile, regardless of label form."""
    grid = _redshouse2f_grid_with_warp_label("Warp→RedsHouse1f")
    agent = _StubAgent(grid)
    toolset = PokemonToolset(agent)
    success, msg = toolset.move_to(7, 1)
    assert success is False
    assert "warp_with_warp_point" in msg
