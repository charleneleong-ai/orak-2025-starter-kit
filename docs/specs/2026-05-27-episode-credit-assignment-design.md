# Episode-end retrospective credit assignment — design

**Status:** draft for review
**Author:** charlene
**Date:** 2026-05-27
**Branch:** `feat/episode-credit-assignment`
**Builds on:** [`feat/sc2-reward-shaping` (PR #111)](../../pull/111) — that PR landed `StarCraftShaper` and unblocked per-step procedural-memory refinement. This spec adds the parallel episode-boundary update.

## Motivation

The fresh n=1 SC2 smoke with `StarCraftShaper` ([`sc2_reward_shaping_smoke_20260527T153806Z`](../../tree/feat/sc2-reward-shaping/game_logs/star_craft/sc2_reward_shaping_smoke_20260527T153806Z)) moved `successful_executions` from `0 → 2` (PR3 baseline → shaped), but `avg_procedure_success_rate` stayed at 0.50 — basically a coin flip. The shaper gives MACLA a per-step signal, but procedures are not yet meaningfully ranked by trajectory outcome: a procedure used in 10 losing episodes and 10 winning episodes should diverge from `0.5`, not stay there.

Frontier-model spike was ruled out for this benchmark (only open-weight models permitted). The next lever is **episode-end retrospective credit assignment**: at episode end, walk back through the procedures used in the trajectory and apply a credit signal proportional to the game outcome, weighted by recency (TD-lambda). This is the standard credit-assignment pattern from RL, applied to MACLA's Beta(alpha, beta) procedural-memory rather than to a value function.

## Goal

Add a framework-level retrospective credit-assignment pass that runs at episode boundary, distributing terminal credit across the trace of procedures used in the episode. Game-agnostic math + game-specific `EpisodeOutcome` adapter, mirroring the `RewardShaper` / `SHAPERS` pattern from the shaping PR.

**Success criteria:**
- After a fresh SC2 smoke with the same `gemma_26b` config, `avg_procedure_success_rate` drifts away from `0.50` (toward something measurably lower since this base model loses most episodes) — proves the retrospective signal is reaching the success_rate.
- A replay validation script over the existing 10-episode smoke trace produces non-trivial alpha/beta deltas per procedure (`avg |delta_alpha| + |delta_beta| > 0.1`) with signs that correlate with episode outcomes.
- No regression on pokemon / mario / 2048 (their summarizers stay unimplemented → `summarize_episode` returns `None` → no retrospective credit fires).

## Non-goals

- Pokemon / Mario / 2048 `EpisodeSummarizer` implementations — separate per-game PRs that follow this one.
- Tuning `base_alpha_delta`, `base_beta_delta`, or `td_lambda` empirically — first-pass defaults ship here; Hydra-driven sweeps later.
- Cross-episode meta-procedure learning consuming the trace (the existing `MetaProceduralLearner.extract_meta_procedure` could subscribe but stays out of scope).
- Wiring into `sweep_runner` (irrelevant — this is per-episode, not per-iteration).

## Existing context

**Per-procedure scoring is Beta(alpha, beta)**: [`Procedure.success_rate`](../../tree/feat/episode-credit-assignment/agents/macla/macla_lib.py) is `alpha / (alpha + beta)`. The update API at [`record_execution_outcome`](../../tree/feat/episode-credit-assignment/agents/macla/macla_lib.py) does `alpha += 1` on success, `beta += 1` (or `beta += 5` for fatal) on failure.

**Episode-end hook already exists**: [`_record_episode_end(episode, score)`](../../tree/feat/episode-credit-assignment/agents/macla/base.py) is called by `UnifiedMaclaAgent.record_episode_end` and currently logs wandb stats but does not touch any procedure's success_rate.

**Per-procedure trace is not currently recorded** — we'll add a `deque[str]` on `EnhancedHierarchicalMemorySystem` that captures every `proc_key` from `record_execution_outcome`.

## Design

### 1. Architecture & module split

| File | Action | Responsibility |
|---|---|---|
| `agents/macla/episode_credit.py` | NEW (~150 LOC) | `EpisodeOutcome` dataclass, `EpisodeCreditConfig` dataclass, `EpisodeSummarizer` base + `StarCraftEpisodeSummarizer`, `SUMMARIZERS` registry, `assign_retrospective_credit` pure function, `_terminal_credit` helper |
| `agents/macla/online_evaluator.py` | MODIFY (~25 LOC) | Add `summarize_episode(...)` to `OnlineAgentEvaluator`. Add `time_alive_norm_max_s` + `progress_norm_max_buildings` to `DEFAULT_SHAPING["star_craft"]` |
| `agents/macla/macla_lib.py` | MODIFY (~15 LOC) | Add `_episode_proc_trace: deque[str]` to `EnhancedHierarchicalMemorySystem`. Append on every `record_execution_outcome`. Add `drain_episode_trace()` method |
| `agents/macla/base.py` | MODIFY (~25 LOC) | Cache `_last_state_str` and `_last_is_fatal` from `_provide_feedback`. Prepend retrospective-credit block to `_record_episode_end`. Hold `_episode_credit_config` overridable via Hydra |
| `tests/test_episode_credit.py` | NEW (~120 LOC) | Framework math: `_terminal_credit` mapping, TD-lambda decay, frequently-used procedures accumulate, evicted proc_key skipped, empty trace no-op, config defaults |
| `tests/test_starcraft_episode_summarizer.py` | NEW (~80 LOC) | SC2 summarizer: populates `EpisodeOutcome` correctly from canonical final states, normalisation thresholds respected |
| `tests/test_macla_episode_trace.py` | NEW (~60 LOC) | `record_execution_outcome` appends; `drain_episode_trace` returns + clears; deque maxlen caps; missing-from-checkpoint defensive read |
| `experiments/sc2_episode_credit_replay.py` | NEW (~80 LOC) | Standalone replay over the existing smoke; prints alpha/beta deltas per procedure |
| `docs/specs/2026-05-27-episode-credit-assignment-design.md` | NEW | This doc |

No changes to: `StarCraftShaper`, detector stack, `run.py`, configs (except `DEFAULT_SHAPING` keys).

### 2. `EpisodeOutcome` — the framework's contract

Game-agnostic dataclass with normalised continuous signals so the credit math is scale-invariant:

```python
@dataclass(frozen=True)
class EpisodeOutcome:
    is_victory: bool = False
    is_fatal_game_over: bool = False
    final_score_norm: float = 0.0
    time_alive_norm: float = 0.0
    progress_norm: float = 0.0
    n_steps: int = 0
```

The terminal credit (the value that gets distributed across the trace) is computed by `_terminal_credit(outcome) -> float in [-1, +1]`:

- Clean victory → `+1.0`
- Fatal defeat, 0 progress → `-1.0`
- Fatal defeat, 30% progress → `-0.85` (partial salvage)
- Max-steps reached, 50% mean progress → `0.0` (neutral)
- Max-steps reached, 100% mean progress → `+0.3` (mild positive)

```python
def _terminal_credit(o: EpisodeOutcome) -> float:
    if o.is_victory:
        return 1.0
    if o.is_fatal_game_over:
        return -1.0 + 0.5 * o.progress_norm
    return -0.3 + 0.6 * (o.final_score_norm + o.time_alive_norm + o.progress_norm) / 3
```

### 3. Credit-assignment math

```python
@dataclass(frozen=True)
class EpisodeCreditConfig:
    base_alpha_delta: float = 5.0
    base_beta_delta: float = 5.0
    td_lambda: float = 0.95


DEFAULT_EPISODE_CREDIT_CONFIG = EpisodeCreditConfig()


def assign_retrospective_credit(
    memory: EnhancedHierarchicalMemorySystem,
    trace: list[str],
    outcome: EpisodeOutcome,
    config: EpisodeCreditConfig = DEFAULT_EPISODE_CREDIT_CONFIG,
) -> dict[str, tuple[float, float]]:
    if not trace:
        return {}

    credit = _terminal_credit(outcome)
    n = len(trace)
    deltas: dict[str, tuple[float, float]] = {}

    for i, proc_key in enumerate(trace):
        weight = config.td_lambda ** (n - 1 - i)
        if credit >= 0:
            delta_alpha = config.base_alpha_delta * credit * weight
            delta_beta = 0.0
        else:
            delta_alpha = 0.0
            delta_beta = config.base_beta_delta * abs(credit) * weight
        prev_alpha, prev_beta = deltas.get(proc_key, (0.0, 0.0))
        deltas[proc_key] = (prev_alpha + delta_alpha, prev_beta + delta_beta)

    for proc_key, (delta_alpha, delta_beta) in deltas.items():
        if proc_key in memory.procedural_memory:
            entry = memory.procedural_memory[proc_key]
            entry.procedure.alpha += delta_alpha
            entry.procedure.beta += delta_beta

    return deltas
```

**Worked example** (n=10, td_lambda=0.95, full victory):

| Step | Weight | delta_alpha |
|---|---|---|
| 0 (earliest) | 0.95^9 ≈ 0.630 | 5.0 × 1.0 × 0.630 = 3.15 |
| 4 (middle) | 0.95^5 ≈ 0.774 | 5.0 × 1.0 × 0.774 = 3.87 |
| 9 (terminal) | 0.95^0 = 1.000 | 5.0 × 1.0 × 1.000 = 5.00 |

For a 1000-step episode, the earliest procedure gets weight `0.95^999 ≈ 5e-23`. Decay naturally truncates the meaningful trace to the last ~60 steps (`0.95^60 ≈ 0.05`).

Frequently-used procedures accumulate weight (sum of K contributions if used K times in the trace) — the right behavior since a procedure used 30 times in a winning episode is more strongly endorsed than one used once.

### 4. Trace recording

In `EnhancedHierarchicalMemorySystem`:

```python
self._episode_proc_trace: deque[str] = deque(maxlen=2000)
```

`maxlen=2000` is generous (typical episodes are <500 steps); the TD-lambda weight at position -2000 is `0.95^2000 ≈ 10^-45`, so anything older is irrelevant anyway.

Capture in `record_execution_outcome` (one line):

```python
def record_execution_outcome(self, proc_key, success, context, is_fatal=False):
    if proc_key not in self.procedural_memory:
        return
    # ... existing alpha/beta update ...
    self._episode_proc_trace.append(proc_key)
```

Drain at episode end:

```python
def drain_episode_trace(self) -> list[str]:
    """Return trace in execution order (oldest -> newest) and clear it."""
    trace = list(self._episode_proc_trace)
    self._episode_proc_trace.clear()
    return trace
```

**Backwards-compat for older checkpoints**: defensive read `getattr(self, "_episode_proc_trace", None)` in `record_execution_outcome`. If missing, initialise on first touch.

### 5. Adapter interface — `EpisodeSummarizer`

Parallels `RewardShaper` / `SHAPERS`:

```python
class EpisodeSummarizer:
    def __init__(self, shaping: dict):
        self._shaping = shaping

    def summarize(
        self,
        *,
        final_state: str,
        final_score: float,
        is_fatal_game_over: bool,
        n_steps: int,
    ) -> EpisodeOutcome:
        raise NotImplementedError


SUMMARIZERS: dict[str, type[EpisodeSummarizer]] = {
    "star_craft": StarCraftEpisodeSummarizer,
    # Other games (pokemon_red, super_mario, twenty_fourty_eight) follow
    # in separate per-game PRs.
}
```

`StarCraftEpisodeSummarizer` reuses `StarCraftShaper`'s `extract_metrics` to avoid regex duplication — top-of-file import from `online_evaluator.py` (one-way; no circularity).

```python
from agents.macla.online_evaluator import SHAPERS


class StarCraftEpisodeSummarizer(EpisodeSummarizer):
    def __init__(self, shaping: dict):
        super().__init__(shaping)
        self._shaper = SHAPERS["star_craft"](shaping)

    def summarize(self, *, final_state, final_score, is_fatal_game_over, n_steps):
        metrics = self._shaper.extract_metrics(final_state)
        time_alive_s = metrics["game_time_sec"]
        building_count = metrics["building_count"]

        time_norm = min(1.0, time_alive_s / self._shaping["time_alive_norm_max_s"])
        progress_norm = min(1.0, building_count / self._shaping["progress_norm_max_buildings"])

        return EpisodeOutcome(
            is_victory=final_score > 0.5,
            is_fatal_game_over=is_fatal_game_over,
            final_score_norm=min(1.0, max(0.0, final_score)),
            time_alive_norm=time_norm,
            progress_norm=progress_norm,
            n_steps=n_steps,
        )
```

New keys in `DEFAULT_SHAPING["star_craft"]` (additive — no breaking change):

```python
"time_alive_norm_max_s": 600,           # 10 min = typical full game
"progress_norm_max_buildings": 20,      # solid Protoss tech tree
```

`OnlineAgentEvaluator` gets a sibling field:

```python
class OnlineAgentEvaluator:
    def __init__(self, game_name: str, ...):
        # ... existing _shaper setup ...
        summarizer_cls = SUMMARIZERS.get(game_name)
        self._summarizer = summarizer_cls(self._shaping) if summarizer_cls else None

    def summarize_episode(self, **kwargs) -> EpisodeOutcome | None:
        return self._summarizer.summarize(**kwargs) if self._summarizer else None
```

Games without a `SUMMARIZERS` entry return `None` → `_record_episode_end` skips retrospective credit → no regression.

### 6. Wiring

In `agents/macla/base.py:_record_episode_end`, prepend (the existing logic continues unchanged below):

```python
if self._evaluator and self._macla_agent and self._macla_agent.memory:
    outcome = self._evaluator.summarize_episode(
        final_state=self._last_state_str or "",
        final_score=score,
        is_fatal_game_over=self._last_is_fatal,
        n_steps=self._steps_in_current_episode,
    )
    if outcome is not None:
        trace = self._macla_agent.memory.drain_episode_trace()
        deltas = assign_retrospective_credit(
            self._macla_agent.memory, trace, outcome, config=self._episode_credit_config,
        )
        logger.info(
            f"[EpisodeCredit] episode={episode} credit={_terminal_credit(outcome):+.2f} "
            f"n_procs={len(deltas)} trace_len={len(trace)}"
        )
```

`base.py` `__init__` additions:
- `self._last_state_str: str = ""` — cache last `cur_state_str` from `_provide_feedback`
- `self._last_is_fatal: bool = False` — cache last `is_fatal_game_over`
- `self._episode_credit_config: EpisodeCreditConfig = DEFAULT_EPISODE_CREDIT_CONFIG` — Hydra-override via `episode_credit:` yaml block, mirroring the existing `reward_shaping:` block

## Testing strategy

| File | Tests | Outcome locked |
|---|---|---|
| `tests/test_episode_credit.py` | Parametrized `_terminal_credit` mapping (victory, fatal, partial, max-steps); TD-lambda decay (terminal position gets weight 1.0, earliest gets td_lambda^(n-1)); frequently-used procedures accumulate weight; empty trace returns empty dict; evicted proc_key is silently skipped; `EpisodeCreditConfig` defaults | Framework math, game-agnostic |
| `tests/test_starcraft_episode_summarizer.py` | Canonical final states (lifted from real smoke logs): victory, defeat with progress, defeat without progress, max-steps reached; normalisation thresholds clamp to `[0, 1]`; `n_steps` round-trips | SC2 adapter |
| `tests/test_macla_episode_trace.py` | `record_execution_outcome` appends to deque; `drain_episode_trace` returns + clears; deque maxlen=2000 caps growth; missing-attribute defensive read on older checkpoints | Memory-system integration |

~15-20 tests total. All synthetic data, no real SC2 dependency.

**Replay validation script** `experiments/sc2_episode_credit_replay.py`:

Walks the existing [`sc2_reward_shaping_smoke_20260527T153806Z/game_states.jsonl`](../../tree/feat/episode-credit-assignment/game_logs/star_craft/sc2_reward_shaping_smoke_20260527T153806Z) + per-iter log, reconstructs the trace from `select_procedure ... pk=proc_X` log lines, derives `EpisodeOutcome` per episode boundary from the final state + score, runs `assign_retrospective_credit` against a fresh memory system, prints per-procedure alpha/beta deltas.

**Decision gate** before running a fresh 2h smoke: `avg |delta_alpha| + |delta_beta| > 0.1` per procedure, signs correlate with episode outcomes (procedures used heavily in episodes with `is_victory=True` get net-positive `delta_alpha`).

## Scope guards

- **Pokemon / Mario / 2048 summarizers** — separate per-game PRs. Their `summarize_episode` returns `None` here.
- **Tuning magnitudes** — `base_alpha_delta=5.0, base_beta_delta=5.0, td_lambda=0.95` are first-pass; Hydra sweep over `episode_credit:` overrides follows.
- **Cross-episode meta-procedure learning consuming the trace** — `MetaProceduralLearner.extract_meta_procedure` already takes a `procedure_sequence`; possible future PR.
- **Wiring into `sweep_runner` triage** — irrelevant; this is per-episode, not per-iteration.

## Out-of-scope follow-ups

- Per-game `EpisodeSummarizer` implementations for pokemon, mario, 2048 (one PR each)
- Empirical tuning via Hydra sweep over `episode_credit:` overrides
- `autoresearch.episode_credit` package extraction once the orak implementation stabilises (follows the `autoresearch.janitor` migration pattern)
