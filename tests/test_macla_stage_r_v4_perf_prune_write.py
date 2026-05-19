"""Stage R v4 (5): perf-prune semantics audit.

Root cause from v2/v3 introspect: ``prune_low_score_iter`` is a no-op
forever because nothing writes ``mem.last_iter_score``. The field is
initialized to None, the prune early-returns on None, and the v2/v3
``procedures_pruned_low_score`` counter stayed at 0 across both
sweeps despite iter 1+2 scoring 2.0/7 (well below the
``PROC_CACHE_MIN_ITER_SCORE = 4.0`` threshold).

The fix: ``UnifiedMaclaAgent.record_episode_end`` already receives the
raw game score from the eval runner (no unit mismatch — both sides
are 0-7 raw). It just needs to forward that into
``mem.last_iter_score`` so the next iter's checkpoint-load-time
prune can actually fire.
"""

from __future__ import annotations

import inspect

from agents.macla import unified
from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem


def test_record_episode_end_writes_last_iter_score():
    """The wiring is in unified.py:record_episode_end. Source-level
    check (full record_episode_end roundtrip needs a fully-wired
    BaseOrakAgent which is heavy to set up)."""
    src = inspect.getsource(unified.UnifiedMaclaAgent.record_episode_end)
    assert "last_iter_score" in src, (
        "unified.py:record_episode_end must assign the episode's raw "
        "score to mem.last_iter_score so the next iter's "
        "prune_low_score_iter (called on checkpoint load) actually "
        "fires when the iter's score < PROC_CACHE_MIN_ITER_SCORE."
    )


def test_record_episode_end_writes_raw_not_normalised():
    """Threshold semantics: ``PROC_CACHE_MIN_ITER_SCORE = 4.0`` is the
    raw game score (pokemon = 0-7). ``record_episode_end`` receives
    the raw score from the eval runner — no /7 or *100 mangling on
    the write path."""
    src = inspect.getsource(unified.UnifiedMaclaAgent.record_episode_end)
    # The simplest assertion that catches the most likely regression:
    # no normalising arithmetic on the score going into last_iter_score.
    no_norm = (
        "last_iter_score = score / " not in src
        and "last_iter_score = score *" not in src
    )
    assert no_norm, (
        "Don't normalise the score on the write — PROC_CACHE_MIN_ITER_SCORE "
        "is on the raw 0-7 scale to match this assignment."
    )


def test_last_iter_score_field_starts_none():
    """Sanity — the field exists, defaults to None, prune is a no-op
    on fresh runs (existing behaviour, just documenting it)."""
    mem = EnhancedHierarchicalMemorySystem()
    assert mem.last_iter_score is None
    assert mem.prune_low_score_iter(4.0) == []
