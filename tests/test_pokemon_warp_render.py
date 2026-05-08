"""Unit tests for the warp-destination rendering fix.

Targets ``PyBoyRunner.get_warp_destinations`` and ``_enrich_warp_label`` —
the two pieces that replace the ambiguous ``WarpPoint`` label with
``Warp→<dest_map>`` so the agent can tell a staircase from an exit.

We don't construct a full PyBoyRunner (that would boot a Game Boy ROM in
a thread). Instead we test ``get_warp_destinations`` against a stub that
mimics the surface PyBoyRunner uses (``self.pyboy.memory`` indexable +
``self.map_names`` dict) and ``_enrich_warp_label`` directly since it's a
pure function.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load pyboy_runner directly without going through the pokemon_red.game
# package __init__, which pulls in mcp_game_servers.base_env (only on the
# server's PYTHONPATH at runtime, not in pytest's import path).
_RUNNER_PATH = (
    Path(__file__).resolve().parent.parent
    / "evaluation_utils/mcp_game_servers/pokemon_red/game/pyboy_runner.py"
)
_spec = importlib.util.spec_from_file_location("pyboy_runner_under_test", _RUNNER_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
PyBoyRunner = _module.PyBoyRunner


class _StubMemory:
    """Mimics PyBoy's memory subscript access against a dict."""

    def __init__(self, layout: dict[int, int]) -> None:
        self._layout = layout

    def __getitem__(self, addr: int) -> int:
        return self._layout.get(addr, 0)


class _StubPyBoy:
    def __init__(self, layout: dict[int, int]) -> None:
        self.memory = _StubMemory(layout)


class _StubRunner:
    """Just enough of PyBoyRunner's surface to exercise ``get_warp_destinations``."""

    def __init__(self, layout: dict[int, int], map_names: dict[str, str]) -> None:
        self.pyboy = _StubPyBoy(layout)
        self.map_names = map_names


def _make_layout(warps: list[tuple[int, int, int, int]]) -> dict[int, int]:
    """Pack ``[(y, x, dest_warp_id, dest_map_id), …]`` into a memory layout
    matching the wWarpEntries format at 0xD3AE/0xD3AF."""
    layout = {0xD3AE: len(warps)}
    for i, (wy, wx, dwarp, dmap) in enumerate(warps):
        base = 0xD3AF + i * 4
        layout[base] = wy
        layout[base + 1] = wx
        layout[base + 2] = dwarp
        layout[base + 3] = dmap
    return layout


def test_get_warp_destinations_resolves_dest_map_id():
    """Each warp entry's dest_map_id resolves via ``map_names``."""
    layout = _make_layout(
        [
            (1, 7, 0, 38),  # RedsHouse2f staircase → RedsHouse1f (map id 38)
            (7, 2, 1, 0),  # exit door → PalletTown (map id 0)
            (7, 3, 2, 0),  # exit door → PalletTown (map id 0)
        ]
    )
    runner = _StubRunner(layout, map_names={"0": "PalletTown", "38": "RedsHouse1f"})
    dests = PyBoyRunner.get_warp_destinations(runner)
    assert dests == {
        (7, 1): "RedsHouse1f",
        (2, 7): "PalletTown",
        (3, 7): "PalletTown",
    }


def test_get_warp_destinations_handles_prevmap_sentinel():
    """``dest_map_id == 0xFF`` means "use last entered map" — label as PrevMap."""
    layout = _make_layout([(0, 5, 0, 0xFF)])
    runner = _StubRunner(layout, map_names={})
    assert PyBoyRunner.get_warp_destinations(runner) == {(5, 0): "PrevMap"}


def test_get_warp_destinations_unknown_id_falls_back():
    """Unknown ids stringify so renders never crash on missing map_names entries."""
    layout = _make_layout([(1, 1, 0, 99)])
    runner = _StubRunner(layout, map_names={})
    assert PyBoyRunner.get_warp_destinations(runner) == {(1, 1): "UNKNOWN_99"}


def test_get_warp_destinations_clamps_garbage_warp_count():
    """Uninitialised RAM can return arbitrary bytes for wNumberOfWarps;
    a count > 32 must clamp to 0 rather than walking off the table."""
    layout = {0xD3AE: 200}
    runner = _StubRunner(layout, map_names={})
    assert PyBoyRunner.get_warp_destinations(runner) == {}


def test_get_warp_destinations_empty_when_no_warps():
    layout = {0xD3AE: 0}
    runner = _StubRunner(layout, map_names={})
    assert PyBoyRunner.get_warp_destinations(runner) == {}


def test_enrich_warp_label_rewrites_warppoint_with_destination():
    dests = {(7, 1): "RedsHouse1f", (2, 7): "PalletTown"}
    assert PyBoyRunner._enrich_warp_label("WarpPoint", (7, 1), dests) == "Warp→RedsHouse1f"
    assert PyBoyRunner._enrich_warp_label("WarpPoint", (2, 7), dests) == "Warp→PalletTown"


def test_enrich_warp_label_passes_through_non_warp_cells():
    """Only 'WarpPoint' cells are enriched — anything else is returned as-is
    so the existing 'O' / 'X' / 'TalkTo*' / 'SIGN_*' tile labels stay intact."""
    dests = {(0, 0): "Whatever"}
    assert PyBoyRunner._enrich_warp_label("O", (0, 0), dests) == "O"
    assert PyBoyRunner._enrich_warp_label("X", (0, 0), dests) == "X"
    assert (
        PyBoyRunner._enrich_warp_label("SIGN_REDSHOUSE1F_TV", (0, 0), dests)
        == "SIGN_REDSHOUSE1F_TV"
    )


def test_enrich_warp_label_leaves_warppoint_alone_if_no_dest():
    """A WarpPoint with no matching memory entry (shouldn't happen on a real
    map, but be defensive) keeps the bare 'WarpPoint' string so the obs
    schema is preserved for downstream consumers like pokemon_tools."""
    assert PyBoyRunner._enrich_warp_label("WarpPoint", (5, 5), {}) == "WarpPoint"
