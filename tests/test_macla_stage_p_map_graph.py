"""Stage P: map-graph observation augmentation.

The 2026-05-15 cross-stage diagnosis identified the pokemon_red ceiling
as living at the M5 milestone gate (``'Viridian' in map_name``). Across
all post-asm-fix sweeps (Stage D + H + K + L + M + N+O), 0 runs ever
set foot in Viridian. The agent reaches the Route 1 / Viridian boundary
but never finds the north-exit tile.

Stage P implements the diagnosis's primary recommended intervention:
surface "unvisited adjacent maps" and "maps visited so far" into the
planner's observation string every step.

Mechanism: a hand-authored ``MAP_GRAPH`` adjacency dict for pokemon_red
early-game maps + ``EnhancedHierarchicalMemorySystem.map_graph_hint``
returns a multi-line natural-language hint. ``unified.py`` prepends the
hint to ``observation`` before the planner sees it.

Unlike Stage N's novelty hint (fires once per first-visit, lives in the
history block), Stage P's map-graph hint fires every step and lives in
the observation block — so the planner is reminded of unvisited
neighbours every decision.
"""

from __future__ import annotations

from agents.macla.macla_lib import (
    MAP_GRAPH,
    EnhancedHierarchicalMemorySystem,
)


def _mk_memory() -> EnhancedHierarchicalMemorySystem:
    """Fresh memory with no visited maps."""
    return EnhancedHierarchicalMemorySystem()


# ─── MAP_GRAPH constant ────────────────────────────────────────────────────


def test_map_graph_covers_early_game_corridor():
    """The route from RedsHouse → Viridian (M1-M5+) must be in the graph."""
    required = {
        "RedsHouse2f",
        "RedsHouse1f",
        "PalletTown",
        "OaksLab",
        "Route1",
        "ViridianCity",
        "ViridianMart",
    }
    missing = required - set(MAP_GRAPH.keys())
    assert not missing, f"MAP_GRAPH missing early-game maps: {missing}"


def test_map_graph_route1_borders_viridian():
    """Route1 must list ViridianCity as a neighbour — that's the M5 gate
    we're trying to surface to the planner."""
    assert "ViridianCity" in MAP_GRAPH["Route1"], (
        "Route1 → ViridianCity adjacency is the M5 unblock the entire "
        "stage is designed to surface; missing it makes Stage P a no-op."
    )


def test_map_graph_adjacency_is_symmetric():
    """For every A → B edge, B → A must also exist. Map transitions in
    pokemon_red are bidirectional (the agent can always walk back)."""
    asymmetric = []
    for src, neighbours in MAP_GRAPH.items():
        for dst in neighbours:
            if dst not in MAP_GRAPH:
                continue  # endpoint-only maps are fine
            if src not in MAP_GRAPH[dst]:
                asymmetric.append((src, dst))
    assert not asymmetric, f"Asymmetric edges (one-way only): {asymmetric}"


# ─── map_graph_hint behaviour ──────────────────────────────────────────────


def test_hint_none_on_unknown_map():
    """If the observation didn't carry a map name (battles, menus, parse
    failure), no hint can be constructed — return None."""
    mem = _mk_memory()
    assert mem.map_graph_hint(None) is None
    assert mem.map_graph_hint("") is None
    assert mem.map_graph_hint("unknown") is None


def test_hint_none_on_map_outside_graph():
    """If the agent is on a map we haven't authored neighbours for, return
    None rather than emit a misleading hint. The MAP_GRAPH is meant to
    grow as we tackle later-game stages."""
    mem = _mk_memory()
    assert mem.map_graph_hint("RouteThatDoesNotExist") is None


def test_hint_lists_unvisited_neighbours_first_visit():
    """On the first visit to a map, every neighbour is unvisited and must
    be listed. This is the headline behaviour: when the agent enters
    PalletTown for the first time, the planner sees it can go to
    RedsHouse1f, OaksLab, or Route1."""
    mem = _mk_memory()
    mem.record_map_visit("PalletTown")  # current map counts as visited
    hint = mem.map_graph_hint("PalletTown")

    assert hint is not None
    # All PalletTown neighbours should appear as unvisited
    for nbr in MAP_GRAPH["PalletTown"]:
        assert nbr in hint, f"Neighbour {nbr} of PalletTown missing from hint"


def test_hint_drops_visited_neighbours():
    """If the agent has been to OaksLab and PalletTown but not Route1, the
    Route1 (the M5 gate path) must remain the unvisited-neighbour focus."""
    mem = _mk_memory()
    mem.record_map_visit("PalletTown")
    mem.record_map_visit("OaksLab")
    hint = mem.map_graph_hint("PalletTown")

    assert hint is not None
    assert "Route1" in hint, "Unvisited Route1 must remain in the hint"
    # The visited neighbours should be marked as visited, not omitted
    # entirely — knowing where you've been is also useful — but the
    # *unvisited* call-out should not list OaksLab.
    unvisited_section = hint.split("Visited so far")[0] if "Visited so far" in hint else hint
    assert "OaksLab" not in unvisited_section, (
        f"OaksLab is visited; must not appear in the unvisited-neighbour section. Hint: {hint!r}"
    )


def test_hint_shows_visited_count():
    """The hint should include the visited-maps list so the planner has
    explicit evidence of what's been explored. After visiting 3 maps,
    all 3 should appear."""
    mem = _mk_memory()
    for m in ["RedsHouse2f", "RedsHouse1f", "PalletTown"]:
        mem.record_map_visit(m)
    hint = mem.map_graph_hint("PalletTown")

    assert hint is not None
    for m in ["RedsHouse2f", "RedsHouse1f", "PalletTown"]:
        assert m in hint, f"Visited map {m} should be referenced in hint"


def test_hint_when_all_neighbours_visited():
    """If every neighbour has been visited, still emit a hint with the
    visited list — but no unvisited call-out. This avoids misleading the
    planner with an empty 'unvisited neighbours' section."""
    mem = _mk_memory()
    # Visit OaksLab and all its neighbours
    for m in ["OaksLab", "PalletTown"]:
        mem.record_map_visit(m)
    hint = mem.map_graph_hint("OaksLab")

    # Hint can be None (no unvisited info) OR a hint with no unvisited
    # neighbours mentioned. Either is acceptable; what's NOT acceptable is
    # emitting "Unvisited neighbours: (none)" which adds noise.
    if hint is not None:
        # If we emit a hint, it must not falsely claim there are unvisited
        # neighbours to explore here.
        lower = hint.lower()
        assert "unvisited" not in lower or "none" not in lower, (
            f"Hint claims unvisited where none exist: {hint!r}"
        )


# ─── observation augmentation (end-to-end) ─────────────────────────────────


def test_hint_preserves_original_observation_fields():
    """The augmented observation must keep all original fields (Map Name,
    Position, etc.) intact so downstream parsers (loop detector, novelty
    extractor) still work."""
    mem = _mk_memory()
    mem.record_map_visit("PalletTown")
    obs = (
        "State: Field\n\n[Map Info]\nMap Name: PalletTown\n"
        "(x_max , y_max): (10, 10)\nYour position (x, y): (5, 5)\n"
    )
    hint = mem.map_graph_hint("PalletTown")
    assert hint is not None
    augmented = f"{hint}\n\n{obs}"

    # All original fields must survive
    assert "Map Name: PalletTown" in augmented
    assert "Your position (x, y): (5, 5)" in augmented
    assert "State: Field" in augmented
    # And the hint is prepended, not appended (so it's the first thing
    # the planner reads)
    assert augmented.index("Map graph") < augmented.index("State: Field"), (
        "Hint must come BEFORE the original observation so the planner encounters it first."
    )


def test_hint_idempotent_on_repeated_calls():
    """Calling map_graph_hint twice in a row with the same state returns
    the same hint — no internal mutation, no first-vs-subsequent split.
    This is the key contract that distinguishes Stage P (every-step hint)
    from Stage N's novelty hint (once-per-first-visit)."""
    mem = _mk_memory()
    mem.record_map_visit("PalletTown")
    h1 = mem.map_graph_hint("PalletTown")
    h2 = mem.map_graph_hint("PalletTown")
    assert h1 == h2 and h1 is not None
