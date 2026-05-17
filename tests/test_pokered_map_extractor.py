"""Tests for the auto-extractor that builds MAP_GRAPH + EXIT_TILES from
the pokered repo's .asm map metadata.

Stage P's hand-authored ``MAP_GRAPH`` only covers M1-M6 territory
(14 maps). The pokered submodule has 224 map headers — auto-extraction
gives full-game coverage without hand-authoring.

The extractor consumes two .asm conventions:

1. ``headers/<Map>.asm``::

       map_header Route1, ROUTE_1, OVERWORLD, NORTH | SOUTH
       connection north, ViridianCity, VIRIDIAN_CITY, -5
       connection south, PalletTown, PALLET_TOWN, 0
       end_map_header

   → outdoor map-to-map adjacency, plus the direction.

2. ``objects/<Map>.asm``::

       def_warp_events
       warp_event  5,  5, REDS_HOUSE_1F, 1
       warp_event 13,  5, BLUES_HOUSE, 1

   → indoor warps (doors/stairs), plus the exact ``(x, y)`` tile.

The two files use different naming conventions for the same map:
header file uses CamelCase (``RedsHouse1F``), warp_event uses
SCREAMING_SNAKE (``REDS_HOUSE_1F``), the game's observation uses
mixed case with lowercase floor suffix (``RedsHouse1f``).

Tests use the symlinked pokered repo at the same path the runtime
uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.macla.macla_lib import MAP_GRAPH
from agents.macla.pokered_map_extractor import (
    build_exit_tiles,
    build_map_graph,
    parse_connections,
    parse_warps,
    snake_to_canonical,
)

# The pokered submodule path matches the runtime hard-fail location.
POKERED = Path("evaluation_utils/mcp_game_servers/pokemon_red/game/pokered")
HEADERS = POKERED / "data/maps/headers"
OBJECTS = POKERED / "data/maps/objects"


def _pokered_available() -> bool:
    return HEADERS.is_dir() and any(HEADERS.glob("*.asm"))


pytestmark = pytest.mark.skipif(
    not _pokered_available(),
    reason="pokered submodule not present (path-symlinked or missing)",
)


# ─── Name normalisation ────────────────────────────────────────────────────


def test_snake_to_canonical_simple():
    """SCREAMING_SNAKE → canonical observation name (mixed case)."""
    assert snake_to_canonical("PALLET_TOWN") == "PalletTown"
    assert snake_to_canonical("VIRIDIAN_CITY") == "ViridianCity"
    assert snake_to_canonical("OAKS_LAB") == "OaksLab"
    assert snake_to_canonical("ROUTE_1") == "Route1"


def test_snake_to_canonical_floor_suffixes_lowercase():
    """Floor suffixes like '1F'/'2F' lower-case to '1f'/'2f' — that's
    how the game observation reports them (see map_names.json)."""
    assert snake_to_canonical("REDS_HOUSE_1F") == "RedsHouse1f"
    assert snake_to_canonical("REDS_HOUSE_2F") == "RedsHouse2f"


def test_snake_to_canonical_passthrough_sentinels():
    """LAST_MAP is a runtime sentinel meaning 'whichever map you entered
    from'; it has no canonical mapping. Extractor must signal that by
    returning None so callers can drop the edge."""
    assert snake_to_canonical("LAST_MAP") is None


# ─── Header connection parsing (outdoor adjacency) ─────────────────────────


def test_parse_connections_route1():
    """Route1 connects north→ViridianCity, south→PalletTown."""
    conns = parse_connections(HEADERS / "Route1.asm")
    assert conns == {"north": "ViridianCity", "south": "PalletTown"}


def test_parse_connections_indoor_map_has_none():
    """OaksLab is indoor — no outdoor connections in the header."""
    conns = parse_connections(HEADERS / "OaksLab.asm")
    assert conns == {}


# ─── Object warp parsing (indoor edges + exit tiles) ───────────────────────


def test_parse_warps_pallet_town():
    """PalletTown has 3 warps: RedsHouse1F (5,5), BluesHouse (13,5),
    OaksLab (12,11)."""
    warps = parse_warps(OBJECTS / "PalletTown.asm")
    targets = {(target, (x, y)) for x, y, target in warps}
    assert ("RedsHouse1f", (5, 5)) in targets
    assert ("BluesHouse", (13, 5)) in targets
    assert ("OaksLab", (12, 11)) in targets


def test_parse_warps_drops_last_map_sentinel():
    """RedsHouse1F has 2x warp_event ... LAST_MAP — these are 'exit
    back to whatever map you came from' sentinels. They must NOT
    appear as edges in the parsed warp list."""
    warps = parse_warps(OBJECTS / "RedsHouse1F.asm")
    targets = [target for _x, _y, target in warps]
    assert "LAST_MAP" not in targets
    # Should still get the upstairs warp to RedsHouse2f
    assert "RedsHouse2f" in targets


# ─── Full graph build ──────────────────────────────────────────────────────


def test_build_map_graph_is_superset_of_handauthored():
    """Regression check: every edge in the hand-authored MAP_GRAPH must
    be present in the auto-extracted graph (modulo known typos in the
    hand-authored data).

    Catching these typos is part of the value the auto-extractor adds:
        * ``ViridianPokeCenter`` — game uses ``ViridianPokecenter``
          (lowercase 'c'), see ``map_names.json`` id=41.
        * ``ViridianHouse``     — game uses ``ViridianNicknameHouse``,
          see ``map_names.json`` id=44.

    Both bad keys mean those nodes never matched a runtime map_name
    anyway — the hand-authored hint silently dropped them. Fix the
    hand-authored graph (or swap to the auto graph) to recover.

    This test gates the swap: once both typos are reconciled, the
    whitelist below should be empty.
    """
    known_handauthored_typos = {"ViridianPokeCenter", "ViridianHouse"}
    auto = build_map_graph(POKERED)

    missing: list[tuple[str, str]] = []
    for src, dsts in MAP_GRAPH.items():
        if src in known_handauthored_typos:
            continue
        if src not in auto:
            missing.append((src, "<no src node>"))
            continue
        for dst in dsts:
            if dst in known_handauthored_typos:
                continue
            if dst not in auto[src]:
                missing.append((src, dst))

    assert not missing, (
        f"Auto-extracted graph is missing hand-authored edges: {missing}. "
        f"Either the parser is broken or the hand-authored graph has a "
        f"new typo to add to known_handauthored_typos."
    )


def test_build_map_graph_is_symmetric():
    """Every A→B edge must have a B→A reverse. Pokemon Red map
    transitions are bidirectional."""
    auto = build_map_graph(POKERED)
    asymmetric = [
        (src, dst)
        for src, dsts in auto.items()
        for dst in dsts
        if dst in auto and src not in auto[dst]
    ]
    assert not asymmetric, f"Asymmetric edges in auto graph: {asymmetric[:5]}"


def test_build_map_graph_covers_more_maps_than_handauthored():
    """The whole point: auto graph should cover most of the 224 maps in
    pokered, vastly more than the 14-map hand-authored graph."""
    auto = build_map_graph(POKERED)
    assert len(auto) > 100, (
        f"Auto graph only has {len(auto)} maps; expected >100. Parser is likely dropping most maps."
    )


# ─── Exit tiles ────────────────────────────────────────────────────────────


def test_build_exit_tiles_indoor_has_coords():
    """Indoor warps have exact (x, y) tiles — the warp_event coord."""
    exits = build_exit_tiles(POKERED)
    # PalletTown → RedsHouse1f door is at (5, 5) per
    # objects/PalletTown.asm: `warp_event 5, 5, REDS_HOUSE_1F, 1`
    assert exits.get(("PalletTown", "RedsHouse1f")) == (5, 5)


def test_build_exit_tiles_outdoor_has_direction():
    """Outdoor connections only have direction, no precise tile.
    Represent those as the direction string so consumers can render
    'walk off the north edge'."""
    exits = build_exit_tiles(POKERED)
    # Route1 → ViridianCity is an outdoor north connection
    assert exits.get(("Route1", "ViridianCity")) == "north"
    assert exits.get(("Route1", "PalletTown")) == "south"


def test_build_exit_tiles_has_no_entries_for_nonexistent_edges():
    """A pair that's not in MAP_GRAPH shouldn't have an exit tile."""
    exits = build_exit_tiles(POKERED)
    # PalletTown does NOT connect to ViridianCity directly
    assert ("PalletTown", "ViridianCity") not in exits
