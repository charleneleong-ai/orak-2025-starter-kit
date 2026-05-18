"""Stage Q: pokemon ``graph_hint`` appends an Exit tiles section with the
exact coordinate (indoor warp) or direction (outdoor connection) for every
unvisited neighbour. Diagnosis + sample output: PR #92."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.macla.pokered_map_extractor import build_exit_tiles
from agents.pokemon_red import game_adapter as pokemon_adapter

POKERED = Path("evaluation_utils/mcp_game_servers/pokemon_red/game/pokered")

pytestmark = pytest.mark.skipif(
    not (POKERED / "data/maps/headers").is_dir()
    or not any((POKERED / "data/maps/headers").glob("*.asm")),
    reason="pokered submodule not present",
)


def test_section_renders_when_unvisited_have_exit_info():
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert hint is not None
    assert "### Map graph" in hint
    assert "### Exit tiles" in hint


def test_indoor_warp_renders_as_coord():
    # objects/PalletTown.asm: warp_event 12, 11, OAKS_LAB, 2
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert "OaksLab: walk to (12, 11)" in hint


def test_outdoor_connection_renders_as_direction():
    # headers/PalletTown.asm: connection north, Route1, ROUTE_1, ...
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert "Route1: walk off the north edge" in hint


def test_route1_exit_for_m5_unblock():
    # The Stage Q intervention point: on Route 1 the agent must walk NORTH to Viridian.
    hint = pokemon_adapter.graph_hint(
        "Route1", {"PalletTown", "Route1", "RedsHouse1f", "RedsHouse2f"}
    )
    assert hint is not None
    assert "ViridianCity: walk off the north edge" in hint


def test_section_omitted_when_all_neighbours_visited():
    visited = {"PalletTown", "OaksLab", "Route1", "RedsHouse1f", "BluesHouse", "Route21"}
    hint = pokemon_adapter.graph_hint("PalletTown", visited)
    if hint is not None:
        assert "### Exit tiles" not in hint


def test_visited_neighbours_excluded_from_exit_tiles():
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown", "OaksLab"})
    exit_section = hint.split("### Exit tiles", 1)[1] if "### Exit tiles" in hint else ""
    assert "OaksLab: walk to" not in exit_section
    assert "Route1: walk off the north edge" in exit_section


def test_map_graph_section_format_preserved():
    # Downstream parsers / historical log diffs depend on byte-stable formatting.
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert hint.splitlines()[0] == "### Map graph"
    assert "Unvisited maps reachable from PalletTown:" in hint
    assert "Visited so far (1): PalletTown" in hint


def test_canonical_map_names_from_asm():
    # Names must match what runtime _extract_map_name emits (lowercase 'c' in Pokecenter etc).
    hint = pokemon_adapter.graph_hint("ViridianCity", {"ViridianCity"})
    assert "ViridianPokecenter" in hint
    assert "ViridianNicknameHouse" in hint


def test_unknown_map_returns_none():
    assert pokemon_adapter.graph_hint("AtlantisGym", set()) is None
    assert pokemon_adapter.graph_hint(None, set()) is None


def test_adapter_output_matches_extractor():
    exits = build_exit_tiles(POKERED)
    assert exits[("PalletTown", "OaksLab")] == (12, 11)
    assert exits[("Route1", "ViridianCity")] == "north"

    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert "OaksLab: walk to (12, 11)" in hint
