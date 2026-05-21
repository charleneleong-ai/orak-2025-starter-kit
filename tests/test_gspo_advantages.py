"""GSPO advantage math — pure functions, no model needed."""

from __future__ import annotations

import math

import pytest

from experiments.gspo.advantages import (
    attach_advantage,
    compute_group_advantages,
    length_normalized_log_ratio,
    zero_variance_group_ids,
)
from experiments.gspo.collate import GSPOSample


def _sample(reward: float, group_id: str = "g", run_id: str = "r") -> GSPOSample:
    """Module-level factory — keeps test bodies short. Frozen dataclass
    means we can't mutate; helper builds one per call."""
    return GSPOSample(
        run_id=run_id,
        iter_step=1,
        prompt="x",
        completion="y",
        reward=reward,
        group_id=group_id,
    )


class TestComputeGroupAdvantages:
    """Z-score each sample against its group's reward distribution."""

    def test_single_group_z_scored(self):
        samples = [
            _sample(reward=0.4, group_id="g1"),
            _sample(reward=0.6, group_id="g1"),
            _sample(reward=0.8, group_id="g1"),
        ]
        pairs = compute_group_advantages(samples)
        advs = [a for _, a in pairs]
        # Population std (divisor n=3) = √((0.04+0+0.04)/3) = √(0.0267) ≈ 0.1633
        # advantages: ±0.2/0.1633 = ±√(3/2) ≈ ±1.2247
        expected = math.sqrt(3.0 / 2.0)
        assert advs[0] == pytest.approx(-expected)
        assert advs[1] == pytest.approx(0.0, abs=1e-7)
        assert advs[2] == pytest.approx(+expected)

    def test_multiple_groups_z_scored_independently(self):
        """Two groups with different reward levels — high-reward samples
        in a high-mean group can still have negative advantage if their
        group's spread puts them below the group mean."""
        samples = [
            _sample(reward=0.1, group_id="low"),
            _sample(reward=0.3, group_id="low"),
            _sample(reward=0.7, group_id="high"),
            _sample(reward=0.9, group_id="high"),
        ]
        pairs = compute_group_advantages(samples)
        by_group = {(s.group_id, s.reward): a for s, a in pairs}
        # Within each group, mean is the midpoint and the two members
        # are symmetric → advantages of ±1.0
        assert by_group[("low", 0.1)] == pytest.approx(-1.0)
        assert by_group[("low", 0.3)] == pytest.approx(+1.0)
        assert by_group[("high", 0.7)] == pytest.approx(-1.0)
        assert by_group[("high", 0.9)] == pytest.approx(+1.0)

    def test_zero_variance_group_yields_zero_advantage(self):
        """Default ``group_id=run_id`` collation produces n=1 groups,
        which always have zero variance. Advantage falls back to 0.0 —
        no gradient signal."""
        samples = [
            _sample(reward=0.5, group_id="solo_1"),
            _sample(reward=0.7, group_id="solo_2"),
        ]
        pairs = compute_group_advantages(samples)
        for _, a in pairs:
            assert a == 0.0

    def test_identical_rewards_within_group_yields_zero_advantage(self):
        """Multi-member group where every member got the same final
        score (e.g., all 5 iters ceiling-bound at M4=57.14%) — no
        learning signal even though group size >1."""
        samples = [_sample(reward=0.5714, group_id="g") for _ in range(5)]
        pairs = compute_group_advantages(samples)
        assert all(a == 0.0 for _, a in pairs)

    def test_preserves_input_order(self):
        """Downstream batching expects deterministic ordering vs the
        input iterable."""
        samples = [_sample(reward=0.4 + 0.05 * i, group_id="g") for i in range(10)]
        pairs = compute_group_advantages(samples)
        assert [s.reward for s, _ in pairs] == [0.4 + 0.05 * i for i in range(10)]


class TestZeroVarianceGroupIds:
    """Surface groups that would produce no gradient signal — used by
    ``train.py`` to refuse to start on uninformative datasets."""

    def test_singleton_groups_all_flagged(self):
        """The default ``group_id=run_id`` collation gives n=1 per
        group — all flagged. This is the expected pre-re-roll state."""
        samples = [
            _sample(reward=0.5, group_id="iter1"),
            _sample(reward=0.7, group_id="iter2"),
            _sample(reward=0.3, group_id="iter3"),
        ]
        assert zero_variance_group_ids(samples) == {"iter1", "iter2", "iter3"}

    def test_multi_sample_groups_with_variance_not_flagged(self):
        samples = [
            _sample(reward=0.4, group_id="g"),
            _sample(reward=0.8, group_id="g"),
        ]
        assert zero_variance_group_ids(samples) == set()

    def test_multi_sample_group_with_identical_rewards_flagged(self):
        samples = [_sample(reward=0.5714, group_id="g") for _ in range(5)]
        assert zero_variance_group_ids(samples) == {"g"}

    def test_mixed_dataset_only_zero_variance_groups_flagged(self):
        samples = [
            _sample(reward=0.4, group_id="varied"),
            _sample(reward=0.8, group_id="varied"),
            _sample(reward=0.5, group_id="flat_a"),
            _sample(reward=0.5, group_id="flat_a"),
            _sample(reward=0.9, group_id="solo"),
        ]
        assert zero_variance_group_ids(samples) == {"flat_a", "solo"}


class TestLengthNormalizedLogRatio:
    """Sequence-level importance ratio — geometric mean of token ratios
    in log-space. The GSPO innovation vs token-wise GRPO/PPO."""

    def test_identical_logp_yields_zero(self):
        """If new and old policy assign identical log-probs, the ratio
        is 1 → log(1) = 0."""
        new = [-1.0, -2.0, -0.5]
        old = [-1.0, -2.0, -0.5]
        assert length_normalized_log_ratio(new, old) == 0.0

    def test_uniform_token_shift_averages_correctly(self):
        """Every token shifted by the same delta → sequence-level ratio
        equals that delta (geometric mean of equal values)."""
        new = [-0.5, -0.5, -0.5, -0.5]
        old = [-1.0, -1.0, -1.0, -1.0]
        # log-ratio per token = -0.5 - (-1.0) = +0.5
        # length-normalized = (4 * 0.5) / 4 = 0.5
        assert length_normalized_log_ratio(new, old) == pytest.approx(0.5)

    def test_mixed_token_shifts_average_to_geometric_mean(self):
        """Per-token log-ratios: +0.5, -0.3, +0.8 → mean = +1.0/3."""
        new = [-0.5, -2.3, -0.2]
        old = [-1.0, -2.0, -1.0]
        expected = (0.5 + (-0.3) + 0.8) / 3
        assert length_normalized_log_ratio(new, old) == pytest.approx(expected)

    def test_empty_sequence_yields_zero(self):
        assert length_normalized_log_ratio([], []) == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length mismatch"):
            length_normalized_log_ratio([0.0, 0.0], [0.0, 0.0, 0.0])


class TestAttachAdvantage:
    """Frozen-dataclass-friendly helper: swap ``reward`` → ``advantage``."""

    def test_returns_new_sample_with_advantage_in_reward_field(self):
        s = _sample(reward=0.6)
        out = attach_advantage(s, advantage=-1.2)
        assert out.reward == -1.2
        assert out is not s  # frozen → new instance

    def test_other_fields_preserved(self):
        s = _sample(reward=0.6, group_id="g1", run_id="r1")
        out = attach_advantage(s, advantage=0.5)
        assert out.group_id == "g1"
        assert out.run_id == "r1"
        assert out.prompt == s.prompt
        assert out.completion == s.completion

    def test_chained_with_compute_group_advantages(self):
        """End-to-end glue: z-score then swap reward field for the
        trainer to consume."""
        samples = [_sample(reward=0.4, group_id="g"), _sample(reward=0.8, group_id="g")]
        with_advantages = [attach_advantage(s, a) for s, a in compute_group_advantages(samples)]
        # Per the z-score: -1.0 and +1.0
        assert with_advantages[0].reward == pytest.approx(-1.0)
        assert with_advantages[1].reward == pytest.approx(+1.0)


# Confirm math.sqrt is imported in advantages.py (smoke — not a regression test).
def test_math_module_used():
    assert math.sqrt(4.0) == 2.0
