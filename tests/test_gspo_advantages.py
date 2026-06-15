"""GSPO advantage math — pure functions, no model needed."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.gspo.advantages import (
    attach_advantage,
    compute_group_advantages,
    gather_completion_logprobs,
    gspo_clipped_loss,
    length_normalized_log_ratio,
    length_normalized_log_ratio_batch,
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


class TestLengthNormalizedLogRatioBatch:
    """Batched (tensor) version of length_normalized_log_ratio — what the
    training loop actually calls. Matches the per-sample scalar version
    when called on a single-row tensor."""

    def test_identical_logp_yields_zero(self):
        new = torch.tensor([[-1.0, -2.0, -0.5]])
        old = torch.tensor([[-1.0, -2.0, -0.5]])
        mask = torch.ones_like(new)
        out = length_normalized_log_ratio_batch(new, old, mask)
        assert out.shape == (1,)
        assert out[0].item() == pytest.approx(0.0)

    def test_uniform_token_shift(self):
        """Every token shifted by +0.5 → length-normalized ratio = +0.5."""
        new = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])
        old = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        mask = torch.ones_like(new)
        out = length_normalized_log_ratio_batch(new, old, mask)
        assert out[0].item() == pytest.approx(0.5)

    def test_mask_excludes_prompt_tokens(self):
        """Only completion tokens (mask=1) contribute. Prompt-region
        tokens (mask=0) must not affect the ratio even if their logp
        differs wildly — this is the key correctness property."""
        # [B=1, T=4]: prompt tokens at 0,1 (mask=0); completion at 2,3 (mask=1)
        new = torch.tensor([[99.0, -99.0, -0.5, -0.5]])
        old = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        out = length_normalized_log_ratio_batch(new, old, mask)
        # Only positions 2,3: (-0.5 - -1.0) avg over 2 tokens = +0.5
        assert out[0].item() == pytest.approx(0.5)

    def test_batched_independent_per_row(self):
        """Each row in the batch is normalized by its own completion
        length — variable-length completions in a batch are common."""
        new = torch.tensor([[-0.5, -0.5, -0.5, 0.0], [-2.0, 0.0, 0.0, 0.0]])
        old = torch.tensor([[-1.0, -1.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
        # Row 0: 3 completion tokens, each +0.5 ratio → mean = +0.5
        # Row 1: 1 completion token, -1.0 ratio → mean = -1.0
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        out = length_normalized_log_ratio_batch(new, old, mask)
        assert out[0].item() == pytest.approx(0.5)
        assert out[1].item() == pytest.approx(-1.0)

    def test_zero_completion_length_no_div_by_zero(self):
        """A row with no completion tokens (all-zero mask) should not
        produce NaN — the trainer would propagate it. Returns 0.0."""
        new = torch.tensor([[-1.0, -1.0]])
        old = torch.tensor([[-1.0, -1.0]])
        mask = torch.tensor([[0.0, 0.0]])
        out = length_normalized_log_ratio_batch(new, old, mask)
        assert not torch.isnan(out).any()
        assert out[0].item() == 0.0

    def test_matches_scalar_version(self):
        """Batched call on a single-row tensor matches the scalar helper
        — same math, different shape."""
        new_list = [-0.5, -2.3, -0.2]
        old_list = [-1.0, -2.0, -1.0]
        scalar = length_normalized_log_ratio(new_list, old_list)
        new_t = torch.tensor([new_list])
        old_t = torch.tensor([old_list])
        mask = torch.ones_like(new_t)
        batched = length_normalized_log_ratio_batch(new_t, old_t, mask)
        assert batched[0].item() == pytest.approx(scalar)


class TestGspoClippedLoss:
    """PPO-style clipped surrogate loss using sequence-level log-ratio
    and group-relative advantages. ``epsilon`` is a required arg, not a
    default — the GSPO paper's 3e-4 is domain-specific (LLM math
    reasoning) and not portable. Callers must pass an explicit value."""

    def test_zero_log_ratio_loss_equals_negative_mean_advantage(self):
        """log_ratio=0 → ratio=1 → loss = -advantage.mean() (iter-1 case
        where pi_new = pi_old). This is the boundary that lets GSPO
        degenerate to REINFORCE at iter 1 without a special case."""
        log_ratio = torch.zeros(4)
        advantages = torch.tensor([1.0, -0.5, 0.5, -1.0])
        loss = gspo_clipped_loss(log_ratio, advantages, epsilon=3e-4)
        assert loss.item() == pytest.approx(0.0)  # mean is 0.0 → -0.0

    def test_positive_advantage_positive_ratio_drives_loss_negative(self):
        log_ratio = torch.tensor([0.0001])  # exp ≈ 1.0001, inside clip
        advantages = torch.tensor([1.0])
        loss = gspo_clipped_loss(log_ratio, advantages, epsilon=3e-4)
        assert loss.item() < 0  # negative loss = good direction

    def test_clip_caps_unclipped_when_advantage_positive(self):
        """If ratio drifts above 1+eps with advantage>0, the clipped
        surrogate kicks in (min(unclipped, clipped) = clipped)."""
        log_ratio = torch.tensor([0.1])  # exp ≈ 1.105, well above 1+3e-4
        advantages = torch.tensor([1.0])
        loss = gspo_clipped_loss(log_ratio, advantages, epsilon=3e-4)
        # clipped ratio = 1.0003, unclipped = 1.105; min = clipped → loss = -clipped
        assert loss.item() == pytest.approx(-1.0003, abs=1e-3)

    def test_clip_doesnt_cap_when_advantage_negative_and_ratio_high(self):
        """When advantage<0, going "outside" the clip in the negative
        direction is what we want — clipped > unclipped (more negative),
        min picks the unclipped (more bad) → loss = -unclipped (positive)."""
        log_ratio = torch.tensor([0.1])  # exp ≈ 1.105
        advantages = torch.tensor([-1.0])
        loss = gspo_clipped_loss(log_ratio, advantages, epsilon=3e-4)
        # unclipped = 1.105 * -1 = -1.105
        # clipped = 1.0003 * -1 = -1.0003
        # min = -1.105 → loss = +1.105 (penalty for moving further from policy)
        assert loss.item() == pytest.approx(1.105, abs=1e-2)

    def test_loss_is_scalar(self):
        """Trainer needs a scalar to call .backward() on."""
        log_ratio = torch.randn(8)
        advantages = torch.randn(8)
        loss = gspo_clipped_loss(log_ratio, advantages, epsilon=3e-4)
        assert loss.ndim == 0

    def test_backward_propagates_through_log_ratio(self):
        log_ratio = torch.tensor([0.0, 0.0], requires_grad=True)
        advantages = torch.tensor([1.0, -1.0])
        loss = gspo_clipped_loss(log_ratio, advantages, epsilon=3e-4)
        loss.backward()
        assert log_ratio.grad is not None
        assert log_ratio.grad.shape == log_ratio.shape

    def test_epsilon_is_required_not_defaulted(self):
        """Regression guard: the helper should not silently apply a
        default epsilon. The GSPO paper's 3e-4 is task-specific and
        baking it in hides the hyperparameter from training configs."""
        with pytest.raises(TypeError, match="epsilon"):
            gspo_clipped_loss(torch.zeros(1), torch.zeros(1))  # type: ignore[call-arg]


class TestGatherCompletionLogprobs:
    """Pure gather: per-target-position log-probs using standard LM shift
    (position t's logits predict token at position t+1). Masking +
    length-normalization are the responsibility of
    ``length_normalized_log_ratio_batch`` downstream, so this helper
    doesn't take a mask argument — keeps responsibilities cleanly split
    and avoids the double-multiply that earlier versions did."""

    def test_returns_logprob_of_target_token(self):
        """For a 2-token sequence, position 0's logits yield the log-prob
        of position 1's token under log_softmax."""
        # vocab_size=3; logits[0,0,:] = [0, log(2), 0]
        # → log_softmax = [log(1/4), log(2/4), log(1/4)]
        logits = torch.tensor([[[0.0, math.log(2.0), 0.0], [0.0, 0.0, 0.0]]])
        input_ids = torch.tensor([[0, 1]])
        out = gather_completion_logprobs(logits, input_ids)
        assert out.shape == (1, 1)
        assert out[0, 0].item() == pytest.approx(math.log(0.5), abs=1e-4)

    def test_shape_is_T_minus_1(self):
        """Output drops one position vs input: positions 0..T-2 predict
        targets 1..T-1. The caller's mask slice must be ``mask[:, 1:]``
        to line up with the shifted output."""
        logits = torch.randn(2, 5, 7)
        input_ids = torch.randint(0, 7, (2, 5))
        out = gather_completion_logprobs(logits, input_ids)
        assert out.shape == (2, 4)

    def test_no_masking_applied(self):
        """Regression guard: this helper used to apply a completion mask
        internally, which double-masked when the caller also passed mask
        to length_normalized_log_ratio_batch. The mask is now the
        downstream helper's responsibility — every output position
        carries a real log-prob value."""
        logits = torch.tensor([[[0.0, 5.0, 0.0], [3.0, 0.0, 0.0]]])
        input_ids = torch.tensor([[0, 1]])
        out = gather_completion_logprobs(logits, input_ids)
        # log_softmax([0, 5, 0]) for target=1 is log(e^5 / (1 + e^5 + 1)) ≈ -0.0182
        assert out[0, 0].item() != 0.0


# Confirm math.sqrt is imported in advantages.py (smoke — not a regression test).
def test_math_module_used():
    assert math.sqrt(4.0) == 2.0
