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


# ─── parity with the existing memory_system hint ───────────────────────────


def test_pokemon_adapter_graph_hint_matches_memory_system():
    """The adapter must produce the SAME string as the existing
    memory_system.map_graph_hint method — that's what guarantees the
    eventual runtime swap is behaviour-preserving."""
    mem = EnhancedHierarchicalMemorySystem()
    for m in ["RedsHouse2f", "RedsHouse1f", "PalletTown"]:
        mem.record_map_visit(m)
    visited = set(mem.visited_maps)

    adapter_hint = pokemon_adapter.graph_hint("PalletTown", visited)
    mem_hint = mem.map_graph_hint("PalletTown")

    assert adapter_hint == mem_hint, (
        f"Adapter / memory_system hints diverged — swap would change runtime.\n"
        f"  adapter: {adapter_hint!r}\n"
        f"  mem    : {mem_hint!r}"
    )


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
