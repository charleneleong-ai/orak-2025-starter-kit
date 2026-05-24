"""Tests for the universal futile-action detector (PR 1 of the MVA harness).

Verifies the detector fires only when the planner-visible observation has
been byte-identical for FUTILE_ACTION_WINDOW consecutive calls, resets on
episode boundaries, and stays game-agnostic.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from agents.macla.unified import (
    FUTILE_ACTION_WINDOW,
    UnifiedMaclaAgent,
)


def _make_detector() -> SimpleNamespace:
    """Stand-in: bind the unbound method to a SimpleNamespace so internal
    `self._obs_hash_window` reads/writes hit the same object the test
    inspects from the outside."""
    obj = SimpleNamespace()
    obj._detect_futile_action = UnifiedMaclaAgent._detect_futile_action.__get__(obj)
    return obj


@pytest.fixture
def detector():
    return _make_detector()


class TestFutileActionDetector:
    """Detector returns a hint string only on K consecutive identical obs."""

    def test_no_hint_below_window(self, detector):
        for _ in range(FUTILE_ACTION_WINDOW - 1):
            assert detector._detect_futile_action("same obs") is None

    def test_fires_on_window(self, detector):
        hints = [detector._detect_futile_action("same obs") for _ in range(FUTILE_ACTION_WINDOW)]
        assert hints[-1] is not None
        assert "no observable change" in hints[-1]
        assert all(h is None for h in hints[:-1])

    def test_silent_when_obs_changes(self, detector):
        assert detector._detect_futile_action("A") is None
        assert detector._detect_futile_action("B") is None
        assert detector._detect_futile_action("C") is None
        assert detector._detect_futile_action("D") is None

    def test_partial_streak_breaks(self, detector):
        for _ in range(FUTILE_ACTION_WINDOW - 1):
            detector._detect_futile_action("X")
        assert detector._detect_futile_action("Y") is None
        for _ in range(FUTILE_ACTION_WINDOW - 1):
            detector._detect_futile_action("Y")
        assert detector._detect_futile_action("Y") is not None

    def test_stays_lit_during_streak(self, detector):
        for _ in range(FUTILE_ACTION_WINDOW):
            detector._detect_futile_action("stuck")
        for _ in range(5):
            assert detector._detect_futile_action("stuck") is not None

    def test_window_clear_resets_state(self, detector):
        for _ in range(FUTILE_ACTION_WINDOW):
            detector._detect_futile_action("stuck")
        detector._obs_hash_window.clear()
        detector._futile_streak_logged = False
        for _ in range(FUTILE_ACTION_WINDOW - 1):
            assert detector._detect_futile_action("stuck") is None
        assert detector._detect_futile_action("stuck") is not None

    @pytest.mark.parametrize(
        "obs",
        [
            "pokemon: map=ViridianCity, pos=(12,5)",  # pokemon-style
            "[[0,0,0,0],[2,0,0,0],[4,2,0,0],[2,16,8,0]]",  # 2048-style
            "mario world=1 stage=1 x_pos=240 status=small",  # mario-style
        ],
    )
    def test_game_agnostic(self, obs):
        det = _make_detector()
        for _ in range(FUTILE_ACTION_WINDOW - 1):
            assert det._detect_futile_action(obs) is None
        assert det._detect_futile_action(obs) is not None


class TestDetectorWiring:
    """Sanity-check the deque + log-flag state model."""

    def test_window_initialized_lazily(self, detector):
        assert not hasattr(detector, "_obs_hash_window")
        detector._detect_futile_action("x")
        assert isinstance(detector._obs_hash_window, deque)
        assert detector._obs_hash_window.maxlen == FUTILE_ACTION_WINDOW

    def test_log_flag_toggles_with_streak(self, detector):
        for _ in range(FUTILE_ACTION_WINDOW):
            detector._detect_futile_action("a")
        assert detector._futile_streak_logged is True
        detector._detect_futile_action("b")
        assert detector._futile_streak_logged is False
