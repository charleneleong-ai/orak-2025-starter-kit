"""Episode-end retrospective credit assignment for MACLA procedural memory.

Companion to the per-step `RewardShaper` in `online_evaluator.py`. At the end
of an episode, MACLA's procedural memory walks the trace of procedures used
in the trajectory and applies a TD-lambda-weighted credit signal based on the
game outcome. The shaper says "this step looked productive"; this module says
"but this whole trajectory ended in defeat" — different time scales, both
contribute to the (alpha, beta) success_rate.

Design spec: docs/specs/2026-05-27-episode-credit-assignment-design.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeOutcome:
    """Game-agnostic summary of an episode for retrospective credit.

    Per-game `EpisodeSummarizer` adapters populate the continuous fields.
    All `*_norm` fields are normalised to [0, 1] so the credit math is
    scale-invariant across games.
    """

    is_victory: bool = False
    is_fatal_game_over: bool = False
    final_score_norm: float = 0.0
    time_alive_norm: float = 0.0
    progress_norm: float = 0.0
    n_steps: int = 0


@dataclass(frozen=True)
class EpisodeCreditConfig:
    """Knobs for `assign_retrospective_credit`. Hydra-overridable via the
    `episode_credit:` block in the agent yaml (mirrors `reward_shaping:`).
    """

    base_alpha_delta: float = 5.0
    base_beta_delta: float = 5.0
    td_lambda: float = 0.95


DEFAULT_EPISODE_CREDIT_CONFIG = EpisodeCreditConfig()


def _terminal_credit(outcome: EpisodeOutcome) -> float:
    """Map an EpisodeOutcome to a scalar credit in [-1, +1].

    - Clean victory → +1.0
    - Fatal defeat with 0 progress → -1.0
    - Fatal defeat with partial progress → linearly salvaged (max -0.5 at 100% progress)
    - Max-steps reached: linear blend of the three continuous signals (range ~[-0.3, +0.3] when inputs are in [0, 1])
    - is_victory takes precedence if both terminal flags are set
    """
    if outcome.is_victory:
        return 1.0
    if outcome.is_fatal_game_over:
        return -1.0 + 0.5 * outcome.progress_norm
    mean_progress = (outcome.final_score_norm + outcome.time_alive_norm + outcome.progress_norm) / 3
    return -0.3 + 0.6 * mean_progress
