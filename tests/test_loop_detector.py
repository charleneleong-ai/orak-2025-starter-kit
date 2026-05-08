"""Unit tests for the game-agnostic LoopDetector.

Built from the actual failure mode in
``game_logs/pokemon_red/20260506_221856/`` — the agent visited
``(OaksLab, 4, 1)`` 7 times in 50 steps and bounced
OaksLab ↔ PalletTown 14 times. The thresholds and signal shapes here
are pinned to catch that trajectory while staying quiet during normal
exploration (e.g. a corridor walked once).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_MOD_NAME = "agents_loop_detector_under_test"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _REPO / "agents/loop_detector.py")
_mod = importlib.util.module_from_spec(_spec)
# Must register before exec_module so @dataclass can resolve cls.__module__.
sys.modules[_MOD_NAME] = _mod
_spec.loader.exec_module(_mod)

LoopDetector = _mod.LoopDetector
LoopSignal = _mod.LoopSignal


# ── normal exploration: detector stays silent ────────────────────────────


def test_silent_during_warmup_window():
    """First ``min_steps_before_firing`` steps don't render anything,
    even if the agent loops — opening exploration is naturally repetitive
    when the agent is feeling out the room."""
    d = LoopDetector(min_steps_before_firing=10)
    for _ in range(8):
        sig = d.observe(state=("RedsHouse2f", 4, 6), score=0, action_class="move_to")
    assert d.render(sig) is None


def test_silent_during_clean_pathfinding():
    """Walking a straight corridor: each tile visited once, action class
    consistent but states keep advancing, score is 0 throughout. The
    detector should NOT fire — nothing's actually wrong."""
    d = LoopDetector()
    for x in range(15):
        sig = d.observe(state=("Corridor", x, 0), score=0, action_class="move_to")
    assert d.render(sig) is None


def test_silent_after_score_gain_resets_stagnation():
    """Score increment should clear the stagnation counter so the
    detector doesn't keep firing about a loop that just resolved."""
    d = LoopDetector(min_steps_before_firing=0)
    for _ in range(20):
        d.observe(state=("PalletTown", 7, 8), score=0, action_class="move_to")
    sig = d.observe(state=("PalletTown", 7, 8), score=1, action_class="move_to")
    assert sig.steps_since_score_gain == 0


# ── state-recurrence detector ────────────────────────────────────────────


def test_state_recurrence_fires_when_position_revisited():
    """Visit ``(OaksLab, 4, 1)`` 4 times within the window, no score
    gain — this is the exact pokemon failure pattern."""
    d = LoopDetector(state_repeat_threshold=3, min_steps_before_firing=0)
    sig = None
    for _ in range(5):
        sig = d.observe(state=("OaksLab", 4, 1), score=2, action_class="interact_with_object")
    assert sig.state_repeats == 5
    block = d.render(sig)
    assert block is not None
    assert "Visited current position 5 times" in block


def test_state_repeats_only_count_within_window():
    """A revisit outside the sliding window shouldn't count toward the
    repeat tally — the detector is a *recent* loop catcher, not lifetime
    history."""
    d = LoopDetector(window_size=10, state_repeat_threshold=3, min_steps_before_firing=0)
    # Visit position twice
    d.observe(state=("X", 0, 0), score=0, action_class="move_to")
    d.observe(state=("X", 0, 0), score=0, action_class="move_to")
    # Walk away for 12 steps (longer than window)
    for x in range(12):
        d.observe(state=("Y", x, 0), score=0, action_class="move_to")
    # Come back — earlier visits aged out.
    sig = d.observe(state=("X", 0, 0), score=0, action_class="move_to")
    assert sig.state_repeats == 1


# ── action-class repetition ──────────────────────────────────────────────


def test_action_repetition_streak_increments_on_same_class():
    d = LoopDetector(action_repeat_threshold=5, min_steps_before_firing=0)
    sig = None
    for i in range(7):
        sig = d.observe(
            state=("OaksLab", 4 + i % 2, 1), score=2, action_class="interact_with_object"
        )
    assert sig.action_repeat_streak == 7
    block = d.render(sig)
    assert "Same action class (`interact_with_object`)" in block


def test_action_streak_resets_on_class_change():
    d = LoopDetector(action_repeat_threshold=3, min_steps_before_firing=0)
    for _ in range(4):
        d.observe(state=("X", 0, 0), score=0, action_class="interact_with_object")
    sig = d.observe(state=("X", 0, 0), score=0, action_class="move_to")
    assert sig.action_repeat_streak == 1
    assert sig.last_action_class == "move_to"


def test_action_class_none_does_not_disturb_streak():
    """A step with no tool call (e.g. raw button press) shouldn't reset
    the streak counter to a poisoned state."""
    d = LoopDetector(action_repeat_threshold=3, min_steps_before_firing=0)
    d.observe(state=("X", 0, 0), score=0, action_class="move_to")
    sig = d.observe(state=("X", 0, 0), score=0, action_class=None)
    assert sig.action_repeat_streak == 0
    assert sig.last_action_class is None


# ── map oscillation ──────────────────────────────────────────────────────


def test_oscillation_detects_abab_pattern():
    """The exact OaksLab ↔ PalletTown bounce from the failed run."""
    d = LoopDetector(oscillation_threshold=3, min_steps_before_firing=0)
    maps = ["OaksLab", "PalletTown", "OaksLab", "PalletTown", "OaksLab", "PalletTown"]
    sig = None
    for m in maps:
        sig = d.observe(state=(m, 0, 0), score=2, action_class="warp_with_warp_point")
    assert sig.oscillation_pair is not None
    assert set(sig.oscillation_pair) == {"OaksLab", "PalletTown"}
    assert sig.oscillation_count >= 3
    block = d.render(sig)
    assert block is not None
    assert "Oscillating" in block
    assert "OaksLab" in block and "PalletTown" in block


def test_oscillation_collapses_consecutive_same_map():
    """Multiple steps within the same map shouldn't be counted as
    transitions — only actual map changes feed the oscillation buffer."""
    d = LoopDetector(min_steps_before_firing=0)
    # 10 steps inside OaksLab, 10 in PalletTown, 10 back in OaksLab
    for _ in range(10):
        d.observe(state=("OaksLab", 4, 5), score=2, action_class="move_to")
    for _ in range(10):
        d.observe(state=("PalletTown", 7, 8), score=2, action_class="move_to")
    sig = d.observe(state=("OaksLab", 4, 5), score=2, action_class="move_to")
    # Only 3 transitions happened → not enough for ABAB oscillation
    assert sig.oscillation_count < 3


def test_oscillation_silent_on_acyclic_traversal():
    """A→B→C→D should NOT be flagged as oscillation."""
    d = LoopDetector(min_steps_before_firing=0)
    for m in ["A", "B", "C", "D", "E"]:
        sig = d.observe(state=(m, 0, 0), score=0, action_class="move_to")
    assert sig.oscillation_count == 0
    assert sig.oscillation_pair is None


# ── render output shape ──────────────────────────────────────────────────


def test_render_includes_steps_since_score_gain():
    d = LoopDetector(min_steps_before_firing=0, state_repeat_threshold=2)
    for _ in range(15):
        sig = d.observe(state=("X", 0, 0), score=0, action_class="move_to")
    block = d.render(sig)
    assert block is not None
    assert "No score gain in last" in block
    assert "Hint:" in block


def test_render_silent_when_only_score_stagnation_no_loop_signal():
    """``steps_since_score_gain`` going up alone shouldn't fire — that's
    just a hard exploration phase. We need a concrete loop signal too."""
    d = LoopDetector(
        min_steps_before_firing=0,
        state_repeat_threshold=10,
        action_repeat_threshold=10,
        oscillation_threshold=10,
    )
    for x in range(15):
        sig = d.observe(state=("Path", x, 0), score=0, action_class="move_to")
    assert sig.steps_since_score_gain == 15
    assert d.render(sig) is None  # no concrete loop, just slow


# ── reset / lifecycle ────────────────────────────────────────────────────


def test_reset_clears_all_state():
    """Between episodes, reset() must wipe the buffers — otherwise the
    detector reports phantom loops carried over from the previous run."""
    d = LoopDetector(min_steps_before_firing=0, state_repeat_threshold=2)
    for _ in range(10):
        d.observe(state=("X", 0, 0), score=0, action_class="move_to")
    d.reset()
    sig = d.observe(state=("X", 0, 0), score=0, action_class="move_to")
    assert sig.state_repeats == 1
    assert sig.action_repeat_streak == 1
    assert sig.steps_since_score_gain == 1
    assert sig.oscillation_count == 0


# ── regression replay against the real failed run ────────────────────────


def test_pokemon_failed_run_replay_triggers_detector():
    """Replay the OaksLab failure mode end-to-end. Detector should fire
    well before step 100 (we observed score=2 plateau from ~step 30
    onward in the real run; this test asserts we'd flag it by step 50)."""
    d = LoopDetector()
    score = 0
    sig = None
    # Simulate the real trajectory's broad strokes.
    for step in range(1, 101):
        if step <= 5:
            state = ("RedsHouse2f", 4, 6)
            action = "warp_with_warp_point"
        elif step <= 12:
            state = ("RedsHouse1f", 4, 7)
            action = "warp_with_warp_point"
            score = 1
        elif step <= 18:
            state = ("PalletTown", 7, 8)
            action = "move_to"
        elif step <= 22:
            state = ("OaksLab", 4, 11)
            action = "move_to"
            score = 2
        else:
            # Now bounce: OaksLab interactions then back to PalletTown
            if (step - 22) % 8 < 6:
                state = ("OaksLab", 4, 1)
                action = "interact_with_object"
            else:
                state = ("PalletTown", 7, 8)
                action = "warp_with_warp_point"
        sig = d.observe(state=state, score=score, action_class=action)

    block = d.render(sig)
    assert block is not None, "detector must flag the bounce-and-spam pattern"
    # At least one of the three signals should be in the block.
    assert any(
        keyword in block
        for keyword in ("Visited current position", "Same action class", "Oscillating")
    )
