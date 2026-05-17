"""Pokemon game-adapter ``graph_hint`` — the generalisable interface
that lets other games plug in their own layout-graph hint.

Game adapters in this repo are duck-typed modules (see
``agents/macla/unified.py: _load_adapter``). UnifiedMaclaAgent reads
capabilities via ``getattr(self._adapter, NAME, default)``. The
existing Stage P hint lives in ``EnhancedHierarchicalMemorySystem.
map_graph_hint`` and is hard-wired to ``unified.py`` — fine for
pokemon-only, but other games (Mario level→level, StarCraft
base→expansion, Sokoban room→room) can't reuse it.

These tests cover the per-game adapter surface:

    pokemon_red.game_adapter.graph_hint(current_map, visited_maps)

The pokemon adapter delegates to the same MAP_GRAPH the runtime uses,
so the strings are byte-identical to ``mem.map_graph_hint(...)``.
Mario/2048 adapters simply don't export the symbol — ``getattr(...,
'graph_hint', None)`` returns None for them, which the planned
``unified.py`` swap will treat as a no-op (no hint prepended).

Not yet wired into unified.py — that flip is a one-liner gated on
the Stage P n=5 verdict.
"""

from __future__ import annotations

from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem
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


# ─── Stage P parity check (intentionally relaxed by Stage Q) ───────────────
#
# Stage Q (2026-05-17) deliberately diverges from ``mem.map_graph_hint``
# by appending an ``### Exit tiles`` section + swapping to the auto-
# extracted MAP_GRAPH (which has 221 maps + 2 typo fixes vs the
# hand-authored 14). Byte-equality parity is therefore no longer the
# correct contract — see ``tests/test_macla_stage_q_exit_tiles.py`` for
# the Stage Q assertions. We keep the Stage P sub-string check here so
# the legacy hint shape is still surfaced unchanged inside the new
# adapter output.


def test_pokemon_adapter_graph_hint_preserves_stage_p_section():
    """The original ``### Map graph`` section format must remain
    embedded in the Stage Q adapter output — the new exit-tile section
    is *additive*, not a rewrite."""
    mem = EnhancedHierarchicalMemorySystem()
    for m in ["RedsHouse2f", "RedsHouse1f", "PalletTown"]:
        mem.record_map_visit(m)
    visited = set(mem.visited_maps)

    adapter_hint = pokemon_adapter.graph_hint("PalletTown", visited)
    assert adapter_hint is not None
    # Stage P section header preserved.
    assert adapter_hint.startswith("### Map graph")
    # Visited-maps line still present with same format.
    assert f"Visited so far ({len(visited)}): " in adapter_hint


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
