"""Stage R v4 (0): unified.py must consume the per-game adapter's
``graph_hint(current_map, visited_maps)`` instead of the hand-authored
``MAP_GRAPH`` + ``EnhancedHierarchicalMemorySystem.map_graph_hint`` path.

Stage P shipped the hand-authored back-compat scaffold with a note that
it would be deleted in follow-up #92 once unified.py swapped to
``self._adapter.graph_hint``. That swap never landed — the agent ran
Stages P/Q/R on a ~30-map hand-authored graph with no exit-tile
coordinates while the full 221-map + 404-exit-tile adapter sat unused.

These tests assert the swap is done:
- macla_lib no longer exports ``MAP_GRAPH`` or ``map_graph_hint``
- unified.py calls ``self._adapter.graph_hint(...)`` via getattr so
  games without the symbol (mario / 2048) cleanly opt out.
"""

from __future__ import annotations

import inspect

from agents.macla import macla_lib, unified


def test_macla_lib_no_longer_exposes_hand_authored_map_graph():
    assert getattr(macla_lib, "MAP_GRAPH", None) is None, (
        "Stage R v4 deletes the hand-authored MAP_GRAPH from macla_lib. "
        "The auto-extracted graph in pokered_map_extractor (consumed by "
        "agents.pokemon_red.game_adapter.graph_hint) is the source of "
        "truth — 221 maps + 404 exit tiles vs the hand-authored ~30."
    )


def test_memory_system_no_longer_has_map_graph_hint_method():
    cls = macla_lib.EnhancedHierarchicalMemorySystem
    assert not hasattr(cls, "map_graph_hint"), (
        "The hand-authored map_graph_hint method is replaced by the "
        "per-game adapter's graph_hint(current_map, visited_maps). "
        "unified.py reaches it via self._adapter.graph_hint."
    )


def test_unified_py_invokes_resolved_graph_hint_with_visited_maps():
    """The wiring resolves the adapter's graph_hint via getattr (see
    next test) and then must actually *call* it with (current_map,
    mem.visited_maps). The signature swap from the old single-arg
    mem.map_graph_hint(current_map) → adapter.graph_hint(current_map,
    visited_maps) is the whole point of v4."""
    src = inspect.getsource(unified)
    assert "mem.visited_maps" in src, (
        "unified.py must pass mem.visited_maps into the adapter's "
        "graph_hint call so the 'visited so far' line in the hint "
        "reflects the agent's exploration history."
    )


def test_unified_py_no_longer_calls_mem_map_graph_hint():
    src = inspect.getsource(unified)
    assert "mem.map_graph_hint" not in src, (
        "The old hand-authored dispatch must be deleted, not kept "
        "alongside the adapter call — keeping both would re-introduce "
        "the 'which graph is the planner actually seeing?' confusion."
    )


def test_unified_py_uses_getattr_for_adapter_optionality():
    """Games without graph_hint (mario, 2048) must opt out cleanly.
    The wiring must use getattr-with-default, never bare attribute
    access that would raise on those adapters."""
    src = inspect.getsource(unified)
    assert 'getattr(self._adapter, "graph_hint"' in src, (
        "unified.py must use getattr(self._adapter, 'graph_hint', "
        "None) so games without a map graph (mario, 2048) get a clean "
        "None and the wiring no-ops. Bare self._adapter.graph_hint "
        "would AttributeError on those adapters."
    )
