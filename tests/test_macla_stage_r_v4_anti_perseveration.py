"""Stage R v4 (1): anti-perseveration position penalty.

v3 introspect (docs/experiments/stage_r_subgoals/v3_n5_introspection.md)
showed iter 5 revisited PalletTown(7,10) **44 times**, OaksLab(6,4) **39
times**, PalletTown(12,12) **20 times** — the planner is provably
failing to notice position-level loops from text history alone.

Stage R v4 (1) injects the count directly: memory tracks a
(map, x, y) → visits counter; unified.py prepends a "### Recently
looped" block to the observation when any cell crosses a threshold.
The block resets per-iter (same semantics as v4(4)'s stagnation
reset) so cumulative procedural memory isn't poisoned by an old
iter's loop trap.
"""

from __future__ import annotations

import inspect
import pickle

from agents.macla import unified
from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

# ── memory-side: counter + hint renderer ─────────────────────────────


def test_memory_has_record_position_method():
    mem = EnhancedHierarchicalMemorySystem()
    assert hasattr(mem, "record_position") and callable(mem.record_position), (
        "Stage R v4 (1) adds EnhancedHierarchicalMemorySystem.record_position("
        "map_name, x, y) — unified.py calls it once per step."
    )


def test_record_position_increments_visit_counter():
    mem = EnhancedHierarchicalMemorySystem()
    mem.record_position("PalletTown", 7, 10)
    mem.record_position("PalletTown", 7, 10)
    mem.record_position("OaksLab", 6, 4)
    # Counter must be queryable — exposed as a property so tests + the
    # hint renderer can introspect it.
    counts = mem.position_visits
    assert counts[("PalletTown", 7, 10)] == 2
    assert counts[("OaksLab", 6, 4)] == 1


def test_record_position_ignores_none_or_unknown_map():
    mem = EnhancedHierarchicalMemorySystem()
    mem.record_position(None, 1, 1)
    mem.record_position("unknown", 1, 1)
    mem.record_position("", 1, 1)
    assert sum(mem.position_visits.values()) == 0


def test_looped_positions_hint_returns_none_below_threshold():
    mem = EnhancedHierarchicalMemorySystem()
    for _ in range(4):
        mem.record_position("PalletTown", 7, 10)
    # default threshold = 5
    assert mem.looped_positions_hint() is None


def test_looped_positions_hint_fires_at_threshold():
    mem = EnhancedHierarchicalMemorySystem()
    for _ in range(8):
        mem.record_position("PalletTown", 7, 10)
    for _ in range(6):
        mem.record_position("OaksLab", 6, 4)
    for _ in range(3):
        mem.record_position("PalletTown", 1, 1)  # below threshold

    hint = mem.looped_positions_hint()
    assert hint is not None
    assert "### Recently looped" in hint
    assert "PalletTown(7, 10): visited 8" in hint
    assert "OaksLab(6, 4): visited 6" in hint
    assert "PalletTown(1, 1)" not in hint, (
        "Cells below threshold must not appear — the block is for "
        "active loops only, not every cell the agent has ever stepped on."
    )


def test_looped_positions_hint_orders_by_visit_count_desc():
    mem = EnhancedHierarchicalMemorySystem()
    for _ in range(20):
        mem.record_position("PalletTown", 12, 12)
    for _ in range(44):
        mem.record_position("PalletTown", 7, 10)
    for _ in range(8):
        mem.record_position("OaksLab", 6, 4)

    hint = mem.looped_positions_hint()
    lines = hint.splitlines()
    # Order: most-visited first so the planner sees the worst loops at the top.
    idx_710 = next(i for i, line in enumerate(lines) if "(7, 10)" in line)
    idx_1212 = next(i for i, line in enumerate(lines) if "(12, 12)" in line)
    idx_64 = next(i for i, line in enumerate(lines) if "(6, 4)" in line)
    assert idx_710 < idx_1212 < idx_64


def test_looped_positions_hint_caps_display_count():
    """Don't dump 50 entries into the prompt — surface the worst N."""
    mem = EnhancedHierarchicalMemorySystem()
    for i in range(10):
        for _ in range(5 + i):
            mem.record_position("Map", i, 0)
    hint = mem.looped_positions_hint(threshold=5, max_display=5)
    assert hint.count("\n- ") == 5, (
        "max_display caps the number of bullet lines so the prompt "
        "stays readable. 10 cells over threshold → 5 in the hint."
    )


# ── per-iter reset on checkpoint load ────────────────────────────────


def test_position_visits_reset_on_unpickle():
    mem = EnhancedHierarchicalMemorySystem()
    for _ in range(20):
        mem.record_position("PalletTown", 7, 10)
    assert sum(mem.position_visits.values()) == 20

    loaded = pickle.loads(pickle.dumps(mem))

    assert sum(loaded.position_visits.values()) == 0, (
        "Position visits are a per-episode signal — they must zero "
        "on checkpoint load just like the stagnation counter. "
        "Otherwise iter 2 starts iteratively biased away from "
        "tiles iter 1 found problematic, which could chill useful "
        "re-exploration."
    )


# ── unified.py wiring ────────────────────────────────────────────────


def test_unified_records_position_per_step():
    """The wiring must call mem.record_position(...) inside the act
    loop so the counter sees every observation."""
    src = inspect.getsource(unified)
    assert "mem.record_position(" in src, (
        "unified.py must call mem.record_position(current_map, x, y) "
        "once per step so the anti-perseveration counter actually "
        "tracks the agent's position."
    )


def test_unified_prepends_looped_positions_hint():
    """The hint block must reach the planner — prepended to the "
    "observation alongside the map_graph_hint."""
    src = inspect.getsource(unified)
    assert "looped_positions_hint" in src, (
        "unified.py must call mem.looped_positions_hint() and "
        "prepend the returned block to the observation. Without "
        "this the counter is dead weight — the planner never sees it."
    )
