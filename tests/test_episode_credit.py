"""Tests for autoresearch.macla.episode_credit — framework math, game-agnostic.

Detection rules are tested in isolation with synthetic EpisodeOutcome inputs.
"""

from __future__ import annotations

import pytest

from agents.macla.episode_credit import (
    EpisodeCreditConfig,
    EpisodeOutcome,
    _terminal_credit,
    assign_retrospective_credit,
)
from agents.macla.macla_lib import (
    EnhancedHierarchicalMemorySystem,
    ProceduralMemoryEntry,
    Procedure,
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


def _seed_proc(mem: EnhancedHierarchicalMemorySystem, key: str) -> Procedure:
    """Insert a minimal Procedure under `key` and return it."""
    proc = Procedure(goal="g", preconditions=[], steps=[])
    mem.procedural_memory[key] = ProceduralMemoryEntry(
        procedure=proc, success_contexts=[], failure_contexts=[]
    )
    return proc


class TestAssignRetrospectiveCredit:
    def test_empty_trace_is_noop(self):
        mem = EnhancedHierarchicalMemorySystem()
        deltas = assign_retrospective_credit(mem, [], EpisodeOutcome(is_victory=True))
        assert deltas == {}

    def test_victory_distributes_positive_alpha_only(self):
        mem = EnhancedHierarchicalMemorySystem()
        proc = _seed_proc(mem, "p_terminal")
        prev_alpha, prev_beta = proc.alpha, proc.beta

        deltas = assign_retrospective_credit(
            mem,
            trace=["p_terminal"],
            outcome=EpisodeOutcome(is_victory=True, n_steps=1),
        )

        # Single-step trace: weight is td_lambda^0 = 1.0, credit = +1.0,
        # base_alpha_delta = 5.0 → delta_alpha = 5.0.
        assert deltas == {"p_terminal": pytest.approx((5.0, 0.0))}
        assert proc.alpha == pytest.approx(prev_alpha + 5.0)
        assert proc.beta == prev_beta  # unchanged

    def test_fatal_defeat_distributes_negative_beta_only(self):
        mem = EnhancedHierarchicalMemorySystem()
        proc = _seed_proc(mem, "p_only")
        prev_alpha, prev_beta = proc.alpha, proc.beta

        deltas = assign_retrospective_credit(
            mem,
            trace=["p_only"],
            outcome=EpisodeOutcome(is_fatal_game_over=True, progress_norm=0.0, n_steps=1),
        )

        # credit = -1.0, base_beta_delta = 5.0 → delta_beta = 5.0
        assert deltas == {"p_only": pytest.approx((0.0, 5.0))}
        assert proc.alpha == prev_alpha  # unchanged
        assert proc.beta == pytest.approx(prev_beta + 5.0)

    def test_td_lambda_decay_terminal_gets_full_weight(self):
        mem = EnhancedHierarchicalMemorySystem()
        for k in ("p0", "p1", "p2", "p3"):
            _seed_proc(mem, k)

        deltas = assign_retrospective_credit(
            mem,
            trace=["p0", "p1", "p2", "p3"],
            outcome=EpisodeOutcome(is_victory=True, n_steps=4),
        )

        # td_lambda=0.95, n=4 → weights are 0.95^3, 0.95^2, 0.95^1, 0.95^0
        expected = {
            "p0": (5.0 * (0.95**3), 0.0),
            "p1": (5.0 * (0.95**2), 0.0),
            "p2": (5.0 * (0.95**1), 0.0),
            "p3": (5.0 * (0.95**0), 0.0),  # terminal has full weight
        }
        for k, (exp_alpha, exp_beta) in expected.items():
            assert deltas[k] == pytest.approx((exp_alpha, exp_beta))

    def test_frequently_used_procedure_accumulates_weight(self):
        mem = EnhancedHierarchicalMemorySystem()
        _seed_proc(mem, "p_repeated")
        _seed_proc(mem, "p_once")

        deltas = assign_retrospective_credit(
            mem,
            trace=["p_repeated", "p_once", "p_repeated", "p_repeated"],
            outcome=EpisodeOutcome(is_victory=True, n_steps=4),
        )

        # p_repeated appears at positions 0, 2, 3 → weights 0.95^3 + 0.95^1 + 0.95^0
        # p_once at position 1 → weight 0.95^2
        expected_repeated_alpha = 5.0 * (0.95**3 + 0.95**1 + 0.95**0)
        expected_once_alpha = 5.0 * (0.95**2)
        assert deltas["p_repeated"] == pytest.approx((expected_repeated_alpha, 0.0))
        assert deltas["p_once"] == pytest.approx((expected_once_alpha, 0.0))

    def test_evicted_proc_key_silently_skipped(self):
        mem = EnhancedHierarchicalMemorySystem()
        # "p_evicted" was used in the trace but no longer exists in procedural_memory
        # (eg. evicted by _prune_procedural_memory).
        _seed_proc(mem, "p_live")

        deltas = assign_retrospective_credit(
            mem,
            trace=["p_evicted", "p_live"],
            outcome=EpisodeOutcome(is_victory=True, n_steps=2),
        )

        # The function still RETURNS the delta for p_evicted (so the caller can log
        # it), but the memory mutation only happens for p_live. The contract is:
        # "applies credit where it can; silently skips evicted procs."
        assert "p_evicted" in deltas
        assert "p_live" in deltas
        # p_live had no mutation skipped — alpha was updated (default Procedure
        # alpha=1 + 5.0 * 0.95^0 = 6.0)
        assert mem.procedural_memory["p_live"].procedure.alpha == pytest.approx(1 + 5.0 * (0.95**0))

    def test_config_overrides_apply(self):
        mem = EnhancedHierarchicalMemorySystem()
        _seed_proc(mem, "p")
        cfg = EpisodeCreditConfig(base_alpha_delta=10.0, base_beta_delta=10.0, td_lambda=0.5)

        deltas = assign_retrospective_credit(
            mem,
            trace=["p"],
            outcome=EpisodeOutcome(is_victory=True, n_steps=1),
            config=cfg,
        )
        # base_alpha_delta=10, weight=0.5^0=1.0, credit=+1.0 → delta_alpha=10.0
        assert deltas["p"] == pytest.approx((10.0, 0.0))
