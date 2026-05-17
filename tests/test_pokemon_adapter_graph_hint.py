"""Pokemon game-adapter ``graph_hint`` — the generalisable interface
that lets other games plug in their own layout-graph hint.

Game adapters in this repo are duck-typed modules (see
``agents/macla/unified.py: _load_adapter``). UnifiedMaclaAgent reads
capabilities via ``getattr(self._adapter, NAME, default)``, so games
that don't expose ``graph_hint`` (currently mario, 2048) get None and
no hint is prepended — safe default.

These tests cover the per-game adapter surface:

    pokemon_red.game_adapter.graph_hint(current_map, visited_maps)

Pokemon's implementation reads the auto-extracted graph + exit tiles
from ``agents.macla.pokered_map_extractor`` (221 maps, 404 exits).
"""

from __future__ import annotations

from agents.pokemon_red import game_adapter as pokemon_adapter

# ─── pokemon adapter exports graph_hint ────────────────────────────────────


def test_pokemon_adapter_exports_graph_hint():
    """The graph_hint symbol must exist on the pokemon adapter — that's
    how unified.py will detect it via getattr."""
    assert callable(getattr(pokemon_adapter, "graph_hint", None)), (
        "pokemon_red.game_adapter must export a graph_hint(current_map, "
        "visited_maps) callable for the generalisable adapter pattern."
    )


def test_pokemon_adapter_graph_hint_first_visit():
    """On first visit to PalletTown, all neighbours unvisited."""
    hint = pokemon_adapter.graph_hint("PalletTown", {"PalletTown"})
    assert hint is not None
    assert "### Map graph" in hint
    assert "Route1" in hint and "OaksLab" in hint


def test_pokemon_adapter_graph_hint_returns_none_for_unknown_map():
    """Map not in graph → None (no misleading hint)."""
    assert pokemon_adapter.graph_hint("AtlantisGym", set()) is None
    assert pokemon_adapter.graph_hint(None, set()) is None
    assert pokemon_adapter.graph_hint("", set()) is None
    assert pokemon_adapter.graph_hint("unknown", set()) is None


def test_pokemon_adapter_graph_hint_renders_map_graph_section():
    """The ``### Map graph`` header + visited-maps line must always be
    present when there's something to say. Stage Q's exit-tile section
    is additive — it appends, never rewrites."""
    visited = {"RedsHouse2f", "RedsHouse1f", "PalletTown"}
    hint = pokemon_adapter.graph_hint("PalletTown", visited)
    assert hint is not None
    assert hint.startswith("### Map graph")
    assert f"Visited so far ({len(visited)}): " in hint


# ─── other games don't export it (deliberate) ──────────────────────────────


def test_mario_adapter_does_not_export_graph_hint():
    """Mario level→level adjacency could be hand-authored later. For
    now the absence is deliberate — getattr returns None, which
    unified.py treats as no-op (no hint prepended)."""
    from agents.super_mario import game_adapter as mario_adapter

    assert getattr(mario_adapter, "graph_hint", None) is None, (
        "Mario adapter doesn't have a map graph yet; not exporting "
        "graph_hint is the correct opt-out."
    )


def test_twenty48_adapter_does_not_export_graph_hint():
    """2048 has no spatial map — opting out is permanent."""
    from agents.twenty_fourty_eight import game_adapter as t48_adapter

    assert getattr(t48_adapter, "graph_hint", None) is None
