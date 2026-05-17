"""Stage Q — pokemon adapter ``graph_hint`` renders exit-tile coordinates
for each unvisited neighbour.

Stage P (PR #90) confirmed FLAT — the map-graph hint reached the planner
1,406 times across n=5 but the agent still couldn't find the Route 1 →
Viridian transition tile. The diagnosis-named follow-up: surface the
*exact* exit tile, not just the destination name.

The new hint shape extends Stage P's:

    ### Map graph
    Unvisited maps reachable from PalletTown: OaksLab, Route1
    Visited so far (1): PalletTown

    ### Exit tiles
      → OaksLab: walk to (12, 11)
      → Route1: walk off the south edge

Exit-tile section appears only when at least one unvisited neighbour has
an exit entry. Indoor warps render as ``walk to (x, y)``; outdoor
connections render as ``walk off the <direction> edge``.

Data source: ``agents.macla.pokered_map_extractor.build_exit_tiles(...)``
parsed from the pokered repo's .asm map metadata (PR #90 commit
``332cc90``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.macla.pokered_map_extractor import build_exit_tiles
from agents.pokemon_red import game_adapter as pokemon_adapter

POKERED = Path("evaluation_utils/mcp_game_servers/pokemon_red/game/pokered")


def _pokered_available() -> bool:
    return (POKERED / "data/maps/headers").is_dir() and any(
        (POKERED / "data/maps/headers").glob("*.asm")
    )


pytestmark = pytest.mark.skipif(
    not _pokered_available(),
    reason="pokered submodule not present (path-symlinked or missing)",
)


# ─── New section header is present when exit info available ────────────────


def test_hint_includes_exit_tiles_section_when_unvisited_have_exit_info():
    """First visit to PalletTown — OaksLab/Route1 are unvisited and both
    have exit info, so the new section must render."""
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert hint is not None
    assert "### Map graph" in hint
    assert "### Exit tiles" in hint


def test_hint_renders_indoor_warp_as_coord():
    """Indoor warps (PalletTown → OaksLab) have an exact (x, y) tile —
    render as ``walk to (x, y)``."""
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert hint is not None
    # From objects/PalletTown.asm: `warp_event 12, 11, OAKS_LAB, 2`
    assert "OaksLab: walk to (12, 11)" in hint


def test_hint_renders_outdoor_connection_as_direction():
    """Outdoor connections (PalletTown → Route1) only have a direction —
    render as ``walk off the <dir> edge``."""
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert hint is not None
    # From headers/PalletTown.asm: `connection north, Route1, ROUTE_1, ...`
    assert "Route1: walk off the north edge" in hint


def test_hint_renders_m5_unblock_exit_for_route1():
    """The whole point: on Route 1, the agent must be told to walk off
    the NORTH edge to reach Viridian. This is the M5 unblock surface."""
    # Agent has visited PalletTown + Route1; Viridian is the next unvisited.
    hint = pokemon_adapter.graph_hint(
        "Route1", {"PalletTown", "Route1", "RedsHouse1f", "RedsHouse2f"}
    )
    assert hint is not None, "Route1 hint must fire — Viridian is unvisited from here"
    assert "ViridianCity" in hint
    # From headers/Route1.asm: `connection north, ViridianCity, ...`
    assert "ViridianCity: walk off the north edge" in hint


# ─── Section omitted when no unvisited neighbours have exit info ───────────


def test_hint_omits_exit_tiles_section_when_all_neighbours_visited():
    """If every neighbour is visited, there are no unvisited→exit lines
    to render — the Exit tiles section must not appear."""
    # Visit everything reachable from PalletTown (auto-graph includes
    # Route21 as a south connection — the unused-in-game outdoor edge).
    visited = {"PalletTown", "OaksLab", "Route1", "RedsHouse1f", "BluesHouse", "Route21"}
    hint = pokemon_adapter.graph_hint("PalletTown", visited)
    if hint is not None:
        # If a hint comes back at all (because visited list is non-empty),
        # it must NOT include the exit-tiles section.
        assert "### Exit tiles" not in hint


# ─── Only unvisited neighbours get exit lines (not visited ones) ───────────


def test_hint_excludes_visited_neighbours_from_exit_tiles():
    """Exit tiles only list ACTIONABLE next-steps — visited neighbours
    don't need a ``walk to`` reminder."""
    # OaksLab visited; Route1 + RedsHouse1f unvisited.
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown", "OaksLab"})
    assert hint is not None
    exit_section = hint.split("### Exit tiles", 1)[1] if "### Exit tiles" in hint else ""
    # OaksLab should NOT have an exit line (already visited).
    assert "OaksLab: walk to" not in exit_section
    # But Route1 should still appear.
    assert "Route1: walk off the north edge" in exit_section


# ─── Map graph section unchanged from Stage P ──────────────────────────────


def test_map_graph_section_format_preserved():
    """The original ``### Map graph`` block format must stay byte-stable
    so downstream parsers (and historical logs) still work — only a new
    section is appended."""
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert hint is not None
    # First line is the original section header.
    assert hint.splitlines()[0] == "### Map graph"
    # Original "Unvisited" / "Visited so far" lines preserved.
    assert "Unvisited maps reachable from PalletTown:" in hint
    assert "Visited so far (1): PalletTown" in hint


# ─── Canonical map names from .asm ─────────────────────────────────────────


def test_graph_uses_canonical_map_names_from_asm():
    """The adapter must use the names that pokered's .asm sources emit
    (e.g. ``ViridianPokecenter`` with lowercase ``c``, ``ViridianNicknameHouse``
    full form) so the hint matches what the runtime ``_extract_map_name``
    parser produces."""
    hint = pokemon_adapter.graph_hint("ViridianCity", {"ViridianCity"})
    assert hint is not None
    assert "ViridianPokecenter" in hint
    assert "ViridianNicknameHouse" in hint


# ─── No exit-tile section if pokered absent (graceful degradation) ─────────


def test_no_crash_when_unknown_map():
    """Maps outside the auto-graph still return None cleanly — no
    KeyError trying to look up exit tiles."""
    assert pokemon_adapter.graph_hint("AtlantisGym", set()) is None
    assert pokemon_adapter.graph_hint(None, set()) is None


# ─── Sanity: extractor matches what the adapter renders ────────────────────


def test_adapter_uses_extractor_exit_tile_data():
    """Cross-check: the adapter's rendering must match what the extractor
    returns. Catches drift if either side changes independently."""
    exits = build_exit_tiles(POKERED)
    # PalletTown → OaksLab indoor warp
    assert exits[("PalletTown", "OaksLab")] == (12, 11)
    # Route1 → ViridianCity outdoor connection (north)
    assert exits[("Route1", "ViridianCity")] == "north"

    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert hint is not None
    assert "OaksLab: walk to (12, 11)" in hint
