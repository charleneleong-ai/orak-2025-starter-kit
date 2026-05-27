"""Tests for autoresearch.macla.episode_credit — framework math, game-agnostic.

Detection rules are tested in isolation with synthetic EpisodeOutcome inputs.
"""

from __future__ import annotations

import pytest

from agents.macla.episode_credit import (
    EpisodeOutcome,
    _terminal_credit,
)


class TestTerminalCredit:
    @pytest.mark.parametrize(
        "outcome,expected",
        [
            pytest.param(EpisodeOutcome(is_victory=True), 1.0, id="clean_victory"),
            pytest.param(
                EpisodeOutcome(is_fatal_game_over=True, progress_norm=0.0),
                -1.0,
                id="fatal_zero_progress",
            ),
            pytest.param(
                EpisodeOutcome(is_fatal_game_over=True, progress_norm=0.3),
                -0.85,
                id="fatal_with_partial_progress",
            ),
            pytest.param(
                EpisodeOutcome(is_fatal_game_over=True, progress_norm=1.0),
                -0.5,
                id="fatal_with_full_progress",
            ),
            pytest.param(
                EpisodeOutcome(final_score_norm=0.5, time_alive_norm=0.5, progress_norm=0.5),
                0.0,
                id="max_steps_mean_progress",
            ),
            pytest.param(
                EpisodeOutcome(final_score_norm=1.0, time_alive_norm=1.0, progress_norm=1.0),
                0.3,
                id="max_steps_full_progress",
            ),
            pytest.param(
                EpisodeOutcome(),  # all zeros
                -0.3,
                id="max_steps_zero_progress",
            ),
        ],
    )
    def test_credit_mapping(self, outcome, expected):
        assert _terminal_credit(outcome) == pytest.approx(expected)

    def test_victory_overrides_fatal(self):
        # Defensive: if both flags are set, victory wins (full positive credit).
        o = EpisodeOutcome(is_victory=True, is_fatal_game_over=True)
        assert _terminal_credit(o) == pytest.approx(1.0)
