"""GSPO advantage computation.

Group Sequence Policy Optimization (Hu et al., DeepSeek 2025) extends GRPO
with two key features:

  1. **Group-relative advantage**: a sample's advantage is its reward
     z-scored against its group (K rollouts from the same checkpoint state,
     sharing ``group_id``). Pure RL — no value model.
  2. **Sequence-level importance ratio**: the policy ratio is computed at
     the sequence level (geometric mean of token-level ratios over the
     sequence length) before clipping, reducing variance vs token-level
     PPO for long completions.

This module owns (1) — the offline-computable math. (2) lives in
``train.py``'s gradient step where token log-probs are available; the
helper here is the ``length_normalized_log_ratio`` formula a trainer can
call once token log-probs are computed.

References:
  * GRPO: Shao et al. 2024 — group-relative baseline replaces a value
    model; standard PPO clipping per token.
  * GSPO: Hu et al. 2025 — same group-relative advantage but the policy
    ratio is sequence-level (length-normalized in log-space).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

import torch
import torch.nn.functional as F

from experiments.gspo.collate import GSPOSample


def compute_group_advantages(
    samples: Iterable[GSPOSample],
    *,
    epsilon: float = 1e-8,
) -> list[tuple[GSPOSample, float]]:
    """Z-score each sample's reward against its group's reward distribution.

    Returns ``(sample, advantage)`` pairs in input order. A group with all
    identical rewards has zero variance → advantage 0.0 for every member
    (the ``epsilon`` floor prevents NaN but produces a meaningless gradient
    signal — see ``zero_variance_group_ids``).

    The advantage is the standard GRPO/GSPO group-relative baseline:

        A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)
    """
    pairs = list(samples)
    by_group: dict[str, list[float]] = defaultdict(list)
    for s in pairs:
        by_group[s.group_id].append(s.reward)

    stats: dict[str, tuple[float, float]] = {}
    for gid, rewards in by_group.items():
        n = len(rewards)
        mean = sum(rewards) / n
        if n < 2:
            std = 0.0
        else:
            # Population std (divisor n, not n-1) — matches standard GRPO
            # practice where the group is the full population for the
            # update, not a sample of a larger distribution.
            var = sum((r - mean) ** 2 for r in rewards) / n
            std = math.sqrt(var)
        stats[gid] = (mean, std)

    out: list[tuple[GSPOSample, float]] = []
    for s in pairs:
        mean, std = stats[s.group_id]
        adv = (s.reward - mean) / (std + epsilon) if std > 0 else 0.0
        out.append((s, adv))
    return out


def zero_variance_group_ids(samples: Iterable[GSPOSample]) -> set[str]:
    """Group ids where every sample has identical reward — gradient
    signal is zero regardless of advantage formulation.

    Symptom of the current collation: ``group_id=run_id`` produces n=1
    groups, every group has zero variance. Until a re-roll launcher
    produces real K-rollout groups, training on this data is uninformative
    and ``train.py`` should refuse to start.
    """
    by_group: dict[str, set[float]] = defaultdict(set)
    for s in samples:
        by_group[s.group_id].add(s.reward)
    return {gid for gid, rewards in by_group.items() if len(rewards) <= 1}


def length_normalized_log_ratio(
    new_logp_tokens: list[float],
    old_logp_tokens: list[float],
) -> float:
    """GSPO's sequence-level importance ratio in log-space.

    Per-token ratio  ρ_t = π_new(t) / π_old(t).
    Sequence ratio   ρ̄  = (∏ ρ_t)^(1/L)   (geometric mean).

    In log-space:    log(ρ̄) = (1/L) · Σ (log π_new(t) − log π_old(t)).

    Empty sequence → 0.0 (no signal).
    """
    if not new_logp_tokens or not old_logp_tokens:
        return 0.0
    if len(new_logp_tokens) != len(old_logp_tokens):
        raise ValueError(f"length mismatch: new={len(new_logp_tokens)} old={len(old_logp_tokens)}")
    n = len(new_logp_tokens)
    return sum(new_logp_tokens[i] - old_logp_tokens[i] for i in range(n)) / n


def attach_advantage(sample: GSPOSample, advantage: float) -> GSPOSample:
    """Return ``sample`` with ``reward`` replaced by ``advantage``.

    Trainers consume ``(prompt, completion, advantage)`` triples. The
    ``GSPOSample`` dataclass is frozen + slots so callers use this helper
    to swap reward → advantage post-z-score.
    """
    return replace(sample, reward=advantage)


# ── tensor helpers for the training loop ──────────────────────────────
#
# These are the batched/tensor counterparts to the scalar math above,
# called inside the GSPO gradient step. Kept here (next to the scalar
# math + the dataclass) so the algorithmic surface lives in one file.


def length_normalized_log_ratio_batch(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched GSPO sequence-level log-ratio (length-normalized in log-space).

    Shapes:
      * ``new_logp``, ``old_logp``: ``[B, T]`` — per-token log-probs.
      * ``completion_mask``: ``[B, T]`` — 1.0 on completion tokens, 0.0 on
        prompt/padding. Only the masked tokens contribute to the ratio.

    Returns ``[B]`` of log(ρ̄) values, one per sample. Empty completions
    (mask all-zero) return 0.0 — clamp(length, min=1) guards the divisor;
    the numerator is already 0 since (∆logp)*0 = 0.
    """
    lengths = completion_mask.sum(dim=-1).clamp(min=1.0)
    log_diff = (new_logp - old_logp) * completion_mask
    return log_diff.sum(dim=-1) / lengths


def gspo_clipped_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """PPO-style clipped surrogate using the GSPO sequence-level ratio.

    Shapes:
      * ``log_ratio``: ``[B]`` from ``length_normalized_log_ratio_batch``.
      * ``advantages``: ``[B]`` group-relative advantage z-scores.

    ``epsilon`` is keyword-only and required — the GSPO paper's 3e-4 is
    task-specific (LLM math reasoning, big-vocab Qwen models) and not
    portable to other domains. Each trainer config must pick a value
    explicitly so the choice shows up in run metadata.

    Returns a scalar loss; ``.backward()`` flows through ``log_ratio``.
    """
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return -torch.min(unclipped, clipped).mean()


def gather_completion_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-target-position log-probs of the actually-generated tokens.

    Standard LM "shift": position ``t``'s logits predict the token at
    position ``t+1``. Returns the raw gathered log-probs (no masking) —
    masking + length normalization happen in
    ``length_normalized_log_ratio_batch`` which owns the mask anyway.

    Shapes:
      * ``logits``: ``[B, T, V]``
      * ``input_ids``: ``[B, T]``

    Returns ``[B, T-1]`` per-target-position log-probs. The output at
    position ``i`` is ``log π(input_ids[i+1] | input_ids[≤i])``.
    """
    shifted_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    return log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
