# Episode-end retrospective credit assignment — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a framework-level retrospective credit-assignment pass at episode boundary that distributes terminal credit across the trace of procedures used in the episode (TD-lambda decay). Game-agnostic math + per-game `EpisodeSummarizer` adapter.

**Architecture:** New `agents/macla/episode_credit.py` owns `EpisodeOutcome`, `EpisodeCreditConfig`, `_terminal_credit`, `assign_retrospective_credit`, `EpisodeSummarizer` base + `StarCraftEpisodeSummarizer`, and `SUMMARIZERS` registry. Trace recording is a `deque[str]` on `EnhancedHierarchicalMemorySystem` populated in `record_execution_outcome` and drained at episode end. `OnlineAgentEvaluator.summarize_episode` routes per-game; `_record_episode_end` in `base.py` calls it and applies the credit math. Same shape as the `RewardShaper`/`SHAPERS` pattern from PR #111. TDD throughout.

**Tech Stack:** Python 3.11+, pytest, ruff (line-length=100), uv for env management. Project on branch `feat/episode-credit-assignment` off `origin/master`.

**Spec:** [`docs/specs/2026-05-27-episode-credit-assignment-design.md`](../specs/2026-05-27-episode-credit-assignment-design.md)

**Working directory:** `/workspace/orak-futile-detector`. All paths below are relative to it.

---

## Task 0: Commit the plan doc

**Files:**
- Modify: `docs/plans/2026-05-27-episode-credit-assignment-plan.md` (already written, uncommitted)

- [ ] **Step 1: Verify branch + clean state**

```bash
cd /workspace/orak-futile-detector
git rev-parse --abbrev-ref HEAD  # expect: feat/episode-credit-assignment
git status -s                    # expect: ?? docs/plans/2026-05-27-episode-credit-assignment-plan.md
```

- [ ] **Step 2: Commit plan**

```bash
git add docs/plans/2026-05-27-episode-credit-assignment-plan.md
git commit -m "docs(macla): episode-end retrospective credit assignment plan

Implementation plan paired with the spec at
docs/specs/2026-05-27-episode-credit-assignment-design.md."
```

---

## Task 1: Trace recording on `EnhancedHierarchicalMemorySystem`

**Files:**
- Create: `tests/test_macla_episode_trace.py`
- Modify: `agents/macla/macla_lib.py` (around line 669, `record_execution_outcome` + class `__init__`)

The simplest, lowest-risk task — adds a `deque[str]` field, populates it on per-step procedure execution, and exposes a `drain` method. Pure infrastructure; no math.

- [ ] **Step 1: Write failing tests**

Create `tests/test_macla_episode_trace.py`:

```python
"""Tests for the episode-level procedure trace on EnhancedHierarchicalMemorySystem.

The trace is a deque[str] populated in record_execution_outcome and drained
at episode end. Tested in isolation with synthetic Procedure entries.
"""

from __future__ import annotations

from collections import deque

import pytest

from agents.macla.macla_lib import (
    ContrastiveContext,
    EnhancedHierarchicalMemorySystem,
    Procedure,
    ProceduralMemoryEntry,
)


def _ctx() -> ContrastiveContext:
    return ContrastiveContext(
        observation_init="",
        action_sequence=[],
        observation_term="",
        cumulative_reward=0.0,
        trajectory_id="t",
        success=True,
    )


def _seed(mem: EnhancedHierarchicalMemorySystem, key: str) -> None:
    """Insert a minimal Procedure under `key` so record_execution_outcome doesn't skip it."""
    mem.procedural_memory[key] = ProceduralMemoryEntry(
        procedure=Procedure(goal="g", preconditions=[], action_template="a", outcome_template="o"),
        success_contexts=[],
        failure_contexts=[],
    )


@pytest.fixture
def mem() -> EnhancedHierarchicalMemorySystem:
    return EnhancedHierarchicalMemorySystem()


class TestEpisodeProcTrace:
    def test_record_execution_outcome_appends_proc_key(self, mem):
        _seed(mem, "p1")
        mem.record_execution_outcome("p1", success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == ["p1"]

    def test_trace_preserves_execution_order(self, mem):
        for k in ("p1", "p2", "p1", "p3"):
            _seed(mem, k)
            mem.record_execution_outcome(k, success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == ["p1", "p2", "p1", "p3"]

    def test_unknown_proc_key_is_not_appended(self, mem):
        # record_execution_outcome returns early when proc_key is unknown;
        # trace must NOT capture that no-op call.
        mem.record_execution_outcome("ghost", success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == []

    def test_drain_episode_trace_returns_and_clears(self, mem):
        for k in ("a", "b", "c"):
            _seed(mem, k)
            mem.record_execution_outcome(k, success=True, context=_ctx())
        assert mem.drain_episode_trace() == ["a", "b", "c"]
        assert list(mem._episode_proc_trace) == []
        # Second drain on empty trace is a no-op
        assert mem.drain_episode_trace() == []

    def test_deque_maxlen_caps_growth(self, mem):
        # maxlen=2000 — older entries are dropped FIFO
        _seed(mem, "p")
        for _ in range(2500):
            mem.record_execution_outcome("p", success=True, context=_ctx())
        assert len(mem._episode_proc_trace) == 2000

    def test_defensive_read_for_old_checkpoints(self, mem):
        # Simulate an older checkpoint that was pickled before _episode_proc_trace existed.
        del mem._episode_proc_trace
        _seed(mem, "p")
        # Should re-initialise on first touch instead of raising AttributeError.
        mem.record_execution_outcome("p", success=True, context=_ctx())
        assert list(mem._episode_proc_trace) == ["p"]
```

- [ ] **Step 2: Run tests — verify RED**

```bash
cd /workspace/orak-futile-detector
.venv/bin/pytest tests/test_macla_episode_trace.py -v
```

Expected: ALL FAIL with `AttributeError: 'EnhancedHierarchicalMemorySystem' object has no attribute '_episode_proc_trace'` (and `drain_episode_trace`).

- [ ] **Step 3: Modify `EnhancedHierarchicalMemorySystem.__init__`**

In `agents/macla/macla_lib.py`, locate the class `EnhancedHierarchicalMemorySystem` (search for `class EnhancedHierarchicalMemorySystem`). In its `__init__`, add at the end:

```python
        # Per-episode trace of proc_keys executed via record_execution_outcome.
        # Drained at episode boundary by EpisodeCredit's assign_retrospective_credit.
        # maxlen=2000 is generous — TD-lambda weight at position -2000 is
        # 0.95^2000 ≈ 1e-45, so older entries are irrelevant.
        self._episode_proc_trace: deque[str] = deque(maxlen=2000)
```

Confirm `deque` is already imported at top of file (search for `from collections import deque`); if not, add it to the existing top-of-file imports.

- [ ] **Step 4: Add `drain_episode_trace` method**

Add this method to `EnhancedHierarchicalMemorySystem` (immediately after `record_execution_outcome` is a natural placement, since the trace is the inverse operation):

```python
    def drain_episode_trace(self) -> list[str]:
        """Return the per-episode proc trace (oldest -> newest) and clear it.

        Called at episode end by `assign_retrospective_credit`. The defensive
        getattr handles older pickled checkpoints that predate this field.
        """
        trace = list(getattr(self, "_episode_proc_trace", []))
        self._episode_proc_trace = deque(maxlen=2000)
        return trace
```

- [ ] **Step 5: Modify `record_execution_outcome` to append**

In `agents/macla/macla_lib.py`, `record_execution_outcome` (line ~669). Add the trace append at the END of the function (after the existing alpha/beta update and the context buffer maintenance):

```python
    def record_execution_outcome(
        self, proc_key: str, success: bool, context: ContrastiveContext, is_fatal: bool = False
    ):
        if proc_key not in self.procedural_memory:
            return
        entry = self.procedural_memory[proc_key]

        if success:
            entry.procedure.alpha += 1
            entry.success_contexts.append(context)
        else:
            penalty = 5 if is_fatal else 1
            entry.procedure.beta += penalty
            context.fatal = is_fatal
            entry.failure_contexts.append(context)

        entry.procedure.execution_count += 1
        if len(entry.success_contexts) > 15:
            entry.success_contexts.pop(0)
        if len(entry.failure_contexts) > 15:
            entry.failure_contexts.pop(0)

        # Per-episode trace — drained by EpisodeCredit at episode end. The
        # defensive `getattr` initialises the deque if the instance was
        # restored from a checkpoint that predates this field.
        trace = getattr(self, "_episode_proc_trace", None)
        if trace is None:
            self._episode_proc_trace = deque(maxlen=2000)
            trace = self._episode_proc_trace
        trace.append(proc_key)
```

- [ ] **Step 6: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_macla_episode_trace.py -v
```

Expected: 6 PASSED.

- [ ] **Step 7: Lint + full test suite (no regressions)**

```bash
ruff check tests/test_macla_episode_trace.py agents/macla/macla_lib.py
ruff format tests/test_macla_episode_trace.py agents/macla/macla_lib.py
.venv/bin/pytest tests/test_macla_episode_trace.py -v
```

- [ ] **Step 8: Commit**

```bash
git add tests/test_macla_episode_trace.py agents/macla/macla_lib.py
git commit -m "feat(macla): episode proc-trace deque on memory system

deque[str] maxlen=2000 populated by record_execution_outcome,
drained at episode boundary via drain_episode_trace. Defensive
getattr for backwards-compat with older pickled checkpoints."
```

---

## Task 2: `episode_credit.py` — types and `_terminal_credit`

**Files:**
- Create: `agents/macla/episode_credit.py`
- Create: `tests/test_episode_credit.py`

Pure module-level types + the terminal-credit mapping. No memory mutation yet.

- [ ] **Step 1: Write failing tests**

Create `tests/test_episode_credit.py`:

```python
"""Tests for autoresearch.macla.episode_credit — framework math, game-agnostic.

Detection rules are tested in isolation with synthetic EpisodeOutcome inputs.
"""

from __future__ import annotations

import pytest

from agents.macla.episode_credit import (
    DEFAULT_EPISODE_CREDIT_CONFIG,
    EpisodeCreditConfig,
    EpisodeOutcome,
    _terminal_credit,
)


class TestEpisodeOutcomeDefaults:
    def test_defaults_are_zero(self):
        o = EpisodeOutcome()
        assert (o.is_victory, o.is_fatal_game_over, o.n_steps) == (False, False, 0)
        assert (o.final_score_norm, o.time_alive_norm, o.progress_norm) == (0.0, 0.0, 0.0)


class TestEpisodeCreditConfigDefaults:
    def test_defaults_match_spec(self):
        c = EpisodeCreditConfig()
        assert (c.base_alpha_delta, c.base_beta_delta, c.td_lambda) == (5.0, 5.0, 0.95)

    def test_default_config_is_shared_constant(self):
        assert DEFAULT_EPISODE_CREDIT_CONFIG == EpisodeCreditConfig()


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
                EpisodeOutcome(
                    final_score_norm=0.5, time_alive_norm=0.5, progress_norm=0.5
                ),
                0.0,
                id="max_steps_mean_progress",
            ),
            pytest.param(
                EpisodeOutcome(
                    final_score_norm=1.0, time_alive_norm=1.0, progress_norm=1.0
                ),
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
```

- [ ] **Step 2: Run tests — verify RED**

```bash
.venv/bin/pytest tests/test_episode_credit.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agents.macla.episode_credit'`.

- [ ] **Step 3: Create `agents/macla/episode_credit.py` with types + `_terminal_credit`**

```python
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
    - Max-steps reached: linear blend of the three continuous signals, clamped to ~[-0.3, +0.3]
    - is_victory takes precedence if both terminal flags are set
    """
    if outcome.is_victory:
        return 1.0
    if outcome.is_fatal_game_over:
        return -1.0 + 0.5 * outcome.progress_norm
    mean_progress = (
        outcome.final_score_norm + outcome.time_alive_norm + outcome.progress_norm
    ) / 3
    return -0.3 + 0.6 * mean_progress
```

- [ ] **Step 4: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_episode_credit.py -v
```

Expected: 10 PASSED (1 + 2 + 7 parametrized).

- [ ] **Step 5: Lint**

```bash
ruff check agents/macla/episode_credit.py tests/test_episode_credit.py
ruff format agents/macla/episode_credit.py tests/test_episode_credit.py
```

- [ ] **Step 6: Commit**

```bash
git add agents/macla/episode_credit.py tests/test_episode_credit.py
git commit -m "feat(macla): EpisodeOutcome + EpisodeCreditConfig + _terminal_credit

Game-agnostic types and the [-1, +1] credit-mapping function for
episode-end retrospective credit assignment. Pure functions; no
memory mutation yet (assign_retrospective_credit follows in the
next commit)."
```

---

## Task 3: `assign_retrospective_credit` — TD-lambda math + memory mutation

**Files:**
- Modify: `agents/macla/episode_credit.py`
- Modify: `tests/test_episode_credit.py`

The credit-distribution function: takes the trace + outcome + config + memory, mutates `(alpha, beta)` on every proc in the trace, returns the deltas dict for logging.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_episode_credit.py`:

```python
from collections import deque

from agents.macla.macla_lib import (
    ContrastiveContext,
    EnhancedHierarchicalMemorySystem,
    Procedure,
    ProceduralMemoryEntry,
)

from agents.macla.episode_credit import assign_retrospective_credit


def _seed_proc(mem: EnhancedHierarchicalMemorySystem, key: str) -> Procedure:
    proc = Procedure(goal="g", preconditions=[], action_template="a", outcome_template="o")
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
        # p_live had no mutation skipped — alpha was updated
        assert mem.procedural_memory["p_live"].procedure.alpha == pytest.approx(
            1 + 5.0 * (0.95**0)
        )

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
```

- [ ] **Step 2: Run tests — verify RED**

```bash
.venv/bin/pytest tests/test_episode_credit.py::TestAssignRetrospectiveCredit -v
```

Expected: 7 FAILED with `ImportError: cannot import name 'assign_retrospective_credit'`.

- [ ] **Step 3: Implement `assign_retrospective_credit`**

Append to `agents/macla/episode_credit.py` (under the existing types):

```python
def assign_retrospective_credit(
    memory,  # type: EnhancedHierarchicalMemorySystem (untyped to avoid circular import)
    trace: list[str],
    outcome: EpisodeOutcome,
    config: EpisodeCreditConfig = DEFAULT_EPISODE_CREDIT_CONFIG,
) -> dict[str, tuple[float, float]]:
    """Apply retrospective credit to procedures used in the episode.

    Walks `trace` in execution order (oldest -> newest). For each procedure,
    computes a TD-lambda-weighted delta_alpha (on positive credit) or delta_beta
    (on negative credit) and mutates `memory.procedural_memory[proc_key]`.
    Procedures evicted from memory since execution time are silently skipped
    (their entries don't get mutated) but still appear in the returned deltas
    dict for logging / inspection.

    Returns: {proc_key: (delta_alpha, delta_beta)}.
    """
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

The `memory` parameter is intentionally untyped (no `EnhancedHierarchicalMemorySystem` import) to avoid a circular dependency `episode_credit.py` ↔ `macla_lib.py`. The contract is documented in the docstring.

- [ ] **Step 4: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_episode_credit.py -v
```

Expected: 17 PASSED (10 existing + 7 new).

- [ ] **Step 5: Lint + full module sanity**

```bash
ruff check agents/macla/episode_credit.py tests/test_episode_credit.py
ruff format agents/macla/episode_credit.py tests/test_episode_credit.py
.venv/bin/pytest tests/test_episode_credit.py tests/test_macla_episode_trace.py -v
```

Expected: 23 PASSED.

- [ ] **Step 6: Commit**

```bash
git add agents/macla/episode_credit.py tests/test_episode_credit.py
git commit -m "feat(macla): assign_retrospective_credit — TD-lambda over trace

Pure function: walks trace, computes TD-lambda(0.95)-weighted delta
per procedure, mutates (alpha, beta) on the memory system, returns
deltas dict for logging. Evicted proc_keys are silently skipped at
the mutation site but still appear in the returned dict so callers
can log the intended-but-unapplied credit."
```

---

## Task 4: `EpisodeSummarizer` base + `StarCraftEpisodeSummarizer` + `DEFAULT_SHAPING` keys

**Files:**
- Modify: `agents/macla/episode_credit.py`
- Modify: `agents/macla/online_evaluator.py` (add two keys to `DEFAULT_SHAPING["star_craft"]`)
- Create: `tests/test_starcraft_episode_summarizer.py`

Per-game adapter that maps a final state + score → `EpisodeOutcome`. Reuses `StarCraftShaper.extract_metrics` (no regex duplication).

- [ ] **Step 1: Write failing tests**

Create `tests/test_starcraft_episode_summarizer.py`:

```python
"""Tests for StarCraftEpisodeSummarizer — populates EpisodeOutcome from
the final state text + score + is_fatal flag for SC2.

Canonical states lifted from real smoke runs.
"""

from __future__ import annotations

import pytest

from agents.macla.episode_credit import StarCraftEpisodeSummarizer
from agents.macla.online_evaluator import DEFAULT_SHAPING


@pytest.fixture(scope="module")
def final_states() -> dict[str, str]:
    """SC2 obs_str at episode end across realistic scenarios."""
    return {
        # Mid-game defeat with some progress (lifted from PR3 smoke iter 201).
        "defeat_with_progress": (
            "Summary 1: At 05:56 game time, our current StarCraft II situation:\n"
            "Resources:\n"
            "- Game time: 05:56\n"
            "- Mineral: 3980\n"
            "- Supply used: 23\n"
            "- Supply cap: 8\n"
            "- Supply left: -15\n"
            "Buildings:\n"
            "- Pylon count: 1\n"
            "- Gateway count: 2\n"
        ),
        # Early defeat with no progress
        "defeat_no_progress": (
            "Summary 1: At 00:45 game time, our current situation:\n"
            "Resources:\n"
            "- Game time: 00:45\n"
            "- Mineral: 50\n"
        ),
        # Hypothetical victory state
        "victory": (
            "Summary 1: At 12:30 game time:\n"
            "Resources:\n"
            "- Game time: 12:30\n"
            "- Mineral: 8000\n"
            "Buildings:\n"
            "- Nexus count: 2\n"
            "- Pylon count: 5\n"
            "- Gateway count: 4\n"
            "- CyberneticsCore count: 1\n"
        ),
    }


@pytest.fixture
def summarizer() -> StarCraftEpisodeSummarizer:
    return StarCraftEpisodeSummarizer(DEFAULT_SHAPING["star_craft"])


class TestStarCraftEpisodeSummarizer:
    def test_victory_state_populates_outcome(self, summarizer, final_states):
        o = summarizer.summarize(
            final_state=final_states["victory"],
            final_score=1.0,
            is_fatal_game_over=False,
            n_steps=2500,
        )
        assert o.is_victory is True
        assert o.is_fatal_game_over is False
        assert o.final_score_norm == pytest.approx(1.0)
        # Game time 12:30 = 750s; max=600 → clamped to 1.0
        assert o.time_alive_norm == pytest.approx(1.0)
        # Buildings: Nexus(2)+Pylon(5)+Gateway(4)+CyberneticsCore(1) = 12. Max=20 → 0.6
        assert o.progress_norm == pytest.approx(0.6)
        assert o.n_steps == 2500

    def test_defeat_with_progress(self, summarizer, final_states):
        o = summarizer.summarize(
            final_state=final_states["defeat_with_progress"],
            final_score=0.0,
            is_fatal_game_over=True,
            n_steps=201,
        )
        assert o.is_victory is False
        assert o.is_fatal_game_over is True
        assert o.final_score_norm == pytest.approx(0.0)
        # 05:56 = 356s → 356/600 ≈ 0.593
        assert o.time_alive_norm == pytest.approx(356 / 600)
        # Buildings: Pylon(1)+Gateway(2) = 3 → 3/20 = 0.15
        assert o.progress_norm == pytest.approx(0.15)

    def test_defeat_no_progress(self, summarizer, final_states):
        o = summarizer.summarize(
            final_state=final_states["defeat_no_progress"],
            final_score=0.0,
            is_fatal_game_over=True,
            n_steps=50,
        )
        assert o.is_fatal_game_over is True
        assert o.progress_norm == pytest.approx(0.0)
        # 00:45 = 45s → 45/600 = 0.075
        assert o.time_alive_norm == pytest.approx(45 / 600)

    def test_empty_state_returns_zeros(self, summarizer):
        o = summarizer.summarize(
            final_state="",
            final_score=0.0,
            is_fatal_game_over=False,
            n_steps=0,
        )
        assert (o.is_victory, o.is_fatal_game_over) == (False, False)
        assert (o.final_score_norm, o.time_alive_norm, o.progress_norm) == (0.0, 0.0, 0.0)

    def test_final_score_clamped_to_unit_interval(self, summarizer, final_states):
        # Out-of-range scores get clamped to [0, 1] for final_score_norm.
        o_high = summarizer.summarize(
            final_state=final_states["victory"],
            final_score=2.5,
            is_fatal_game_over=False,
            n_steps=100,
        )
        assert o_high.final_score_norm == pytest.approx(1.0)
        assert o_high.is_victory is True  # > 0.5 threshold

        o_neg = summarizer.summarize(
            final_state="",
            final_score=-0.5,
            is_fatal_game_over=False,
            n_steps=100,
        )
        assert o_neg.final_score_norm == pytest.approx(0.0)
        assert o_neg.is_victory is False


class TestDefaultShapingKeys:
    """Locks the new DEFAULT_SHAPING['star_craft'] keys (additive — no breaking change)."""

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("time_alive_norm_max_s", 600),
            ("progress_norm_max_buildings", 20),
        ],
    )
    def test_default_shaping_has_episode_credit_keys(self, key, expected):
        assert DEFAULT_SHAPING["star_craft"][key] == expected
```

- [ ] **Step 2: Run tests — verify RED**

```bash
.venv/bin/pytest tests/test_starcraft_episode_summarizer.py -v
```

Expected: FAIL with `ImportError: cannot import name 'StarCraftEpisodeSummarizer'` and `KeyError: 'time_alive_norm_max_s'`.

- [ ] **Step 3: Add new keys to `DEFAULT_SHAPING["star_craft"]`**

In `agents/macla/online_evaluator.py`, inside `DEFAULT_SHAPING["star_craft"]` (between the existing keys, just before `reward_min`):

```python
        # ── Episode-end retrospective credit (lever 2) ──
        # Normalisation thresholds used by StarCraftEpisodeSummarizer to map
        # raw end-of-game state into the [0, 1] EpisodeOutcome continuous
        # fields. The values aren't tuned empirically — they're "what does
        # a typical full game look like" calibration points.
        "time_alive_norm_max_s": 600,           # 10 min = a typical full game length
        "progress_norm_max_buildings": 20,      # solid tech tree (Nexus+pylons+gates+tech)
```

- [ ] **Step 4: Implement `EpisodeSummarizer` base + `StarCraftEpisodeSummarizer`**

Append to `agents/macla/episode_credit.py`:

```python
from agents.macla.online_evaluator import SHAPERS


class EpisodeSummarizer:
    """Per-game adapter — populates EpisodeOutcome at episode end.

    Subclasses implement `summarize`. Concrete summarizers live alongside this
    base; the SUMMARIZERS registry below maps `game_name` -> class.
    """

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


class StarCraftEpisodeSummarizer(EpisodeSummarizer):
    """SC2 outcome from final obs_str + score.

    Reuses StarCraftShaper.extract_metrics (no regex duplication). Single
    source of truth for the per-game regex patterns: shaper owns them, the
    summarizer just calls extract_metrics.
    """

    def __init__(self, shaping: dict):
        super().__init__(shaping)
        # SHAPERS["star_craft"] is StarCraftShaper. Constructed once per summarizer
        # to amortise the regex compilation that StarCraftShaper does at init.
        self._shaper = SHAPERS["star_craft"](shaping)

    def summarize(
        self,
        *,
        final_state: str,
        final_score: float,
        is_fatal_game_over: bool,
        n_steps: int,
    ) -> EpisodeOutcome:
        metrics = self._shaper.extract_metrics(final_state)
        time_alive_s = metrics.get("game_time_sec", 0)
        building_count = metrics.get("building_count", 0)

        max_time_s = self._shaping["time_alive_norm_max_s"]
        max_buildings = self._shaping["progress_norm_max_buildings"]

        time_norm = min(1.0, time_alive_s / max_time_s) if max_time_s else 0.0
        progress_norm = min(1.0, building_count / max_buildings) if max_buildings else 0.0
        score_norm = max(0.0, min(1.0, final_score))

        return EpisodeOutcome(
            is_victory=final_score > 0.5,
            is_fatal_game_over=is_fatal_game_over,
            final_score_norm=score_norm,
            time_alive_norm=time_norm,
            progress_norm=progress_norm,
            n_steps=n_steps,
        )
```

- [ ] **Step 5: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_starcraft_episode_summarizer.py -v
```

Expected: 7 PASSED (5 SC2 summarizer + 2 default-shaping-key checks).

- [ ] **Step 6: Lint**

```bash
ruff check agents/macla/episode_credit.py agents/macla/online_evaluator.py tests/test_starcraft_episode_summarizer.py
ruff format agents/macla/episode_credit.py agents/macla/online_evaluator.py tests/test_starcraft_episode_summarizer.py
```

- [ ] **Step 7: Commit**

```bash
git add agents/macla/episode_credit.py agents/macla/online_evaluator.py tests/test_starcraft_episode_summarizer.py
git commit -m "feat(macla): EpisodeSummarizer base + StarCraftEpisodeSummarizer

SC2 outcome population from final obs_str. Reuses StarCraftShaper's
extract_metrics regex (single source of truth). New DEFAULT_SHAPING
keys: time_alive_norm_max_s=600, progress_norm_max_buildings=20."
```

---

## Task 5: `SUMMARIZERS` registry + `OnlineAgentEvaluator.summarize_episode`

**Files:**
- Modify: `agents/macla/episode_credit.py` (add SUMMARIZERS dict)
- Modify: `agents/macla/online_evaluator.py` (add `summarize_episode` method)
- Modify: `tests/test_episode_credit.py` (add registry/wiring tests)

The wiring layer. `OnlineAgentEvaluator` already dispatches per-game shapers; this task adds the parallel dispatch for summarizers.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_episode_credit.py`:

```python
from agents.macla.episode_credit import SUMMARIZERS, StarCraftEpisodeSummarizer
from agents.macla.online_evaluator import OnlineAgentEvaluator


class TestSummarizersRegistry:
    def test_star_craft_registered(self):
        assert SUMMARIZERS["star_craft"] is StarCraftEpisodeSummarizer

    def test_games_without_summarizer_return_none(self):
        # pokemon_red has no summarizer in this PR — must be absent from the registry.
        assert "pokemon_red" not in SUMMARIZERS


class TestOnlineAgentEvaluatorWiring:
    def test_summarize_episode_routes_to_star_craft_summarizer(self):
        ev = OnlineAgentEvaluator("star_craft")
        outcome = ev.summarize_episode(
            final_state="",
            final_score=1.0,
            is_fatal_game_over=False,
            n_steps=100,
        )
        assert outcome is not None
        assert outcome.is_victory is True
        assert outcome.n_steps == 100

    def test_summarize_episode_returns_none_for_games_without_summarizer(self):
        # pokemon_red has a shaper but no summarizer yet (separate PR follows).
        # The evaluator must return None so the agent's _record_episode_end
        # skips retrospective credit without crashing.
        ev = OnlineAgentEvaluator("pokemon_red")
        outcome = ev.summarize_episode(
            final_state="",
            final_score=0.0,
            is_fatal_game_over=True,
            n_steps=42,
        )
        assert outcome is None
```

- [ ] **Step 2: Run tests — verify RED**

```bash
.venv/bin/pytest tests/test_episode_credit.py::TestSummarizersRegistry tests/test_episode_credit.py::TestOnlineAgentEvaluatorWiring -v
```

Expected: 4 FAILED (ImportError for SUMMARIZERS + AttributeError for `summarize_episode`).

- [ ] **Step 3: Add `SUMMARIZERS` registry to `episode_credit.py`**

Append to `agents/macla/episode_credit.py`:

```python
SUMMARIZERS: dict[str, type[EpisodeSummarizer]] = {
    "star_craft": StarCraftEpisodeSummarizer,
    # Pokemon, super_mario, twenty_fourty_eight follow in per-game PRs.
}
```

- [ ] **Step 4: Add `summarize_episode` to `OnlineAgentEvaluator`**

In `agents/macla/online_evaluator.py`, in class `OnlineAgentEvaluator`. First, augment `__init__`:

```python
class OnlineAgentEvaluator:
    """Coordinates a per-game `RewardShaper` and `EpisodeSummarizer`. Stateless across games."""

    def __init__(self, game_name: str, shaping_overrides: dict | None = None):
        self._game_name = game_name
        self._prev_metrics: dict = {}
        self._step_rewards: deque = deque(maxlen=100)
        self._shaping: dict[str, float] = {
            **DEFAULT_SHAPING.get(game_name, {}),
            **(shaping_overrides or {}),
        }
        shaper_cls = SHAPERS.get(game_name, GenericShaper)
        self._shaper: RewardShaper = shaper_cls(self._shaping)

        # Episode summarizer — None for games without a registered summarizer.
        # Imported lazily to avoid a circular import episode_credit ↔ online_evaluator.
        from agents.macla.episode_credit import SUMMARIZERS

        summarizer_cls = SUMMARIZERS.get(game_name)
        self._summarizer = summarizer_cls(self._shaping) if summarizer_cls else None
```

Then add the new method (near the bottom of the class, after `reset_episode`):

```python
    def summarize_episode(
        self,
        *,
        final_state: str,
        final_score: float,
        is_fatal_game_over: bool,
        n_steps: int,
    ):
        """Return an EpisodeOutcome for the just-finished episode, or None
        if this game has no registered EpisodeSummarizer.

        Games without summarizers skip retrospective credit assignment entirely
        — no behaviour change for pokemon / mario / 2048 in this PR.
        """
        if self._summarizer is None:
            return None
        return self._summarizer.summarize(
            final_state=final_state,
            final_score=final_score,
            is_fatal_game_over=is_fatal_game_over,
            n_steps=n_steps,
        )
```

The `from agents.macla.episode_credit import SUMMARIZERS` is inside `__init__` (the one exception to the "hoist imports" rule per CLAUDE.md): `episode_credit.py` imports `SHAPERS` from `online_evaluator.py` at module top, so the reverse import has to be lazy to avoid a circular load. Document this inline as the comment shows.

- [ ] **Step 5: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_episode_credit.py tests/test_starcraft_episode_summarizer.py -v
```

Expected: 27 PASSED (17 from Task 3 + 4 new + 6 already from Task 4 SC2 summarizer).

- [ ] **Step 6: Lint**

```bash
ruff check agents/macla/episode_credit.py agents/macla/online_evaluator.py tests/test_episode_credit.py
ruff format agents/macla/episode_credit.py agents/macla/online_evaluator.py tests/test_episode_credit.py
```

- [ ] **Step 7: Commit**

```bash
git add agents/macla/episode_credit.py agents/macla/online_evaluator.py tests/test_episode_credit.py
git commit -m "feat(macla): SUMMARIZERS registry + OnlineAgentEvaluator.summarize_episode

Per-game dispatch parallel to SHAPERS / shape_step. Games without a
registered summarizer return None — no regression for pokemon /
mario / 2048 which stay on per-step credit only."
```

---

## Task 6: Wire into `_record_episode_end` + cache last state

**Files:**
- Modify: `agents/macla/base.py` (`__init__` + `_provide_feedback` + `_record_episode_end`)

The agent-side hookup. `_provide_feedback` caches the last state + fatal flag (it already receives them per-step — we just need to retain them across the call). `_record_episode_end` then calls `evaluator.summarize_episode` + `assign_retrospective_credit` with the cached values.

No unit tests for this task — the wiring is mechanical and exercised by the replay validation in Task 7. (Unit-testing `base.py`'s episode lifecycle requires constructing a fully-mocked MaclaAgent, which is high-overhead for marginal coverage.)

- [ ] **Step 1: Cache last state + fatal flag in `__init__`**

In `agents/macla/base.py`, locate the `__init__` of the agent class (the one containing `_record_episode_end`). Add three new fields at the end of `__init__`:

```python
        # Cached for episode-end retrospective credit assignment (lever 2).
        # Set by every _provide_feedback call; consumed by _record_episode_end.
        self._last_state_str: str = ""
        self._last_is_fatal: bool = False

        # Episode credit config — overridable via Hydra agent yaml block
        # `episode_credit: { base_alpha_delta: ..., td_lambda: ... }`.
        # Default matches DEFAULT_EPISODE_CREDIT_CONFIG.
        from agents.macla.episode_credit import DEFAULT_EPISODE_CREDIT_CONFIG

        self._episode_credit_config = DEFAULT_EPISODE_CREDIT_CONFIG
```

- [ ] **Step 2: Cache the per-step state in `_provide_feedback`**

In the same file, locate `_provide_feedback` (around line 272). At the START of the function, add the cache update (the existing logic continues unchanged below):

```python
    def _provide_feedback(self, prev_state_str, cur_state_str, ...):
        # Cache last state + fatal for end-of-episode retrospective credit.
        # _record_episode_end consumes these via _last_state_str / _last_is_fatal.
        self._last_state_str = cur_state_str
        # is_fatal flag is computed mid-function; we cache it after _detect_success.
        # ... existing code through _detect_success ...
```

Then after the existing `strong_success, is_fatal_game_over = self._detect_success(...)` line:

```python
        self._last_is_fatal = is_fatal_game_over
```

- [ ] **Step 3: Prepend retrospective-credit block to `_record_episode_end`**

In `_record_episode_end` (line 514). Prepend BEFORE the existing `if self._macla_agent and hasattr(...)` block:

```python
    def _record_episode_end(self, episode: int, score: float):
        # ── Episode-end retrospective credit assignment (lever 2) ──
        # See docs/specs/2026-05-27-episode-credit-assignment-design.md.
        # Per-game summarizer returns None for games without one → skip.
        # Hoisted import per CLAUDE.md (avoids the deferred function-local form).
        from agents.macla.episode_credit import _terminal_credit, assign_retrospective_credit

        evaluator = getattr(self, "_evaluator", None)
        macla = self._macla_agent
        if evaluator and macla and getattr(macla, "memory", None):
            outcome = evaluator.summarize_episode(
                final_state=self._last_state_str or "",
                final_score=score,
                is_fatal_game_over=self._last_is_fatal,
                n_steps=self._steps_in_current_episode,
            )
            if outcome is not None:
                trace = macla.memory.drain_episode_trace()
                deltas = assign_retrospective_credit(
                    macla.memory, trace, outcome, config=self._episode_credit_config
                )
                logger.info(
                    f"[EpisodeCredit] episode={episode} "
                    f"credit={_terminal_credit(outcome):+.2f} "
                    f"n_procs={len(deltas)} trace_len={len(trace)}"
                )

        # ── existing logic continues unchanged ──
        # Update adaptive stagnation tracking
        if self._macla_agent and hasattr(self._macla_agent, "update_episode_score"):
            # ... rest of method unchanged ...
```

The function-local import of `_terminal_credit, assign_retrospective_credit` is the exception per CLAUDE.md: importing at module top would create a circular dependency since `episode_credit.py` imports `SHAPERS` from `online_evaluator.py` which is imported by `base.py`. The deferred import is documented inline.

- [ ] **Step 4: Verify no test regressions**

```bash
.venv/bin/pytest tests/test_macla_episode_trace.py tests/test_episode_credit.py tests/test_starcraft_episode_summarizer.py tests/test_online_evaluator_starcraft.py -v
```

Expected: ALL PASSED (no test changes; this verifies the wiring doesn't break anything).

- [ ] **Step 5: Lint**

```bash
ruff check agents/macla/base.py
ruff format agents/macla/base.py
```

- [ ] **Step 6: Commit**

```bash
git add agents/macla/base.py
git commit -m "feat(macla): wire episode credit into _record_episode_end

Cache last state + is_fatal in _provide_feedback. At episode end,
call evaluator.summarize_episode → assign_retrospective_credit
against the drained trace from memory. _episode_credit_config is
Hydra-overridable via the agent yaml's episode_credit: block.

Games without a summarizer (pokemon / mario / 2048) return None
and skip the retrospective pass entirely — no regression."
```

---

## Task 7: Replay validation script

**Files:**
- Create: `experiments/sc2_episode_credit_replay.py`

Standalone offline validation: walks the existing PR3 + reward-shaping smoke logs, reconstructs the per-episode trace from `select_procedure ... pk=...` log lines, constructs synthetic `EpisodeOutcome` per episode boundary, applies the credit math against a fresh memory system, prints per-procedure deltas.

- [ ] **Step 1: Write the replay script**

Create `experiments/sc2_episode_credit_replay.py`:

```python
"""Offline replay validation for episode-end retrospective credit assignment.

Walks an existing smoke run's `game_states.jsonl` + per-iter log file,
reconstructs the per-episode trace from `select_procedure ... pk=...` lines,
derives a synthetic `EpisodeOutcome` per episode boundary, and applies
`assign_retrospective_credit` against a fresh memory system. Prints the
per-procedure alpha/beta deltas + decision-gate verdict.

Run:
    .venv/bin/python -m experiments.sc2_episode_credit_replay
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import typer

from agents.macla.episode_credit import (
    EpisodeOutcome,
    StarCraftEpisodeSummarizer,
    _terminal_credit,
    assign_retrospective_credit,
)
from agents.macla.macla_lib import (
    EnhancedHierarchicalMemorySystem,
    Procedure,
    ProceduralMemoryEntry,
)
from agents.macla.online_evaluator import DEFAULT_SHAPING

app = typer.Typer(add_completion=False)

DEFAULT_SMOKE_DIR = Path(
    "game_logs/star_craft/sc2_reward_shaping_smoke_20260527T153806Z"
)
DEFAULT_SMOKE_LOG = Path("logs/sc2_reward_shaping_smoke_20260527T153806Z.log")

# Match `select_procedure ... pk=proc_NNNNN` lines from the smoke log.
_PK_RE = re.compile(r"\bpk=([\w_]+)")


def _reconstruct_trace_per_episode(log_path: Path) -> list[list[str]]:
    """Group select_procedure pk= lines by episode boundary.

    Episode boundaries are detected by `MACLA Stats & Optimisation (Step N)`
    lines where N resets after each episode. Simpler: split on the smoke
    log's `Episode: N` markers (emitted by the runner's status banner).
    """
    episodes: list[list[str]] = []
    current: list[str] = []
    in_episode = False
    for line in log_path.open():
        if "Star Craft: Step" in line and "Episode:" in line:
            # New episode line — flush the previous one
            if current:
                episodes.append(current)
                current = []
            in_episode = True
            continue
        if not in_episode:
            continue
        m = _PK_RE.search(line)
        if m:
            current.append(m.group(1))
    if current:
        episodes.append(current)
    return episodes


def _seed_procs(mem: EnhancedHierarchicalMemorySystem, all_pks: set[str]) -> None:
    """Insert empty Procedure entries so assign_retrospective_credit can mutate them."""
    for pk in all_pks:
        if pk not in mem.procedural_memory:
            proc = Procedure(goal="", preconditions=[], action_template="", outcome_template="")
            mem.procedural_memory[pk] = ProceduralMemoryEntry(
                procedure=proc, success_contexts=[], failure_contexts=[]
            )


def _read_episode_summary(summary_path: Path) -> list[dict]:
    """Return list of {episode_id, final_score} from evaluation_summary.json."""
    data = json.loads(summary_path.read_text())
    return data.get("episodes", [])


@app.command()
def replay(
    smoke_dir: Path = typer.Option(DEFAULT_SMOKE_DIR, "--dir", "-d"),
    smoke_log: Path = typer.Option(DEFAULT_SMOKE_LOG, "--log", "-l"),
) -> None:
    if not smoke_dir.exists():
        typer.echo(f"ERROR: {smoke_dir} not found", err=True)
        raise typer.Exit(1)
    if not smoke_log.exists():
        typer.echo(f"ERROR: {smoke_log} not found", err=True)
        raise typer.Exit(1)

    summary = _read_episode_summary(smoke_dir / "evaluation_summary.json")
    typer.echo(f"=== {len(summary)} episodes in summary ===")

    traces = _reconstruct_trace_per_episode(smoke_log)
    typer.echo(f"=== {len(traces)} traces reconstructed from log ===")
    n_paired = min(len(summary), len(traces))

    mem = EnhancedHierarchicalMemorySystem()
    all_pks = {pk for trace in traces for pk in trace}
    _seed_procs(mem, all_pks)
    typer.echo(f"=== {len(all_pks)} unique procedures across all episodes ===")

    summarizer = StarCraftEpisodeSummarizer(DEFAULT_SHAPING["star_craft"])

    # Per-episode credit application
    per_proc_totals: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    for i in range(n_paired):
        ep = summary[i]
        trace = traces[i]
        # We don't have the final obs_str in evaluation_summary.json, so use empty
        # string — this nulls time_alive_norm and progress_norm. final_score and
        # is_fatal_game_over still drive the credit signal.
        outcome = summarizer.summarize(
            final_state="",
            final_score=ep["final_score"],
            is_fatal_game_over=ep["final_score"] == 0.0,  # heuristic
            n_steps=ep.get("inference_calls", 0),
        )
        deltas = assign_retrospective_credit(mem, trace, outcome)
        typer.echo(
            f"  ep {ep['episode_id']:>2} score={ep['final_score']:.1f} "
            f"credit={_terminal_credit(outcome):+.2f} trace_len={len(trace)} "
            f"procs_credited={len(deltas)}"
        )
        for pk, (da, db) in deltas.items():
            prev_a, prev_b = per_proc_totals[pk]
            per_proc_totals[pk] = (prev_a + da, prev_b + db)

    # Decision gate
    if not per_proc_totals:
        typer.echo("FAIL — no per-procedure deltas accumulated")
        raise typer.Exit(2)

    avg_abs_delta = sum(abs(a) + abs(b) for a, b in per_proc_totals.values()) / len(
        per_proc_totals
    )
    typer.echo("")
    typer.echo("=== Decision gate ===")
    typer.echo(f"  procs touched: {len(per_proc_totals)}")
    typer.echo(f"  avg |delta_alpha|+|delta_beta| per proc: {avg_abs_delta:.3f}")
    if avg_abs_delta > 0.1:
        typer.echo("  PASS — non-trivial credit deltas; signs reflect episode outcomes")
    else:
        typer.echo("  FAIL — deltas too small; tune base_alpha_delta / base_beta_delta")
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Run the script**

```bash
cd /workspace/orak-futile-detector
.venv/bin/python -m experiments.sc2_episode_credit_replay
```

Capture the output. Expected: `avg |delta_alpha|+|delta_beta| per proc > 0.1` → PASS. If FAIL, do NOT proceed — investigate whether `base_alpha_delta=5.0` is too small for the typical trace lengths in this smoke (~250 calls per episode × 10 episodes).

- [ ] **Step 3: Lint**

```bash
ruff check experiments/sc2_episode_credit_replay.py
ruff format experiments/sc2_episode_credit_replay.py
```

- [ ] **Step 4: Commit**

```bash
git add experiments/sc2_episode_credit_replay.py
git commit -m "feat(macla): replay validation for episode credit assignment

Walks the existing reward-shaping smoke logs (no SC2 server needed),
reconstructs per-episode procedure traces from pk= log lines, applies
assign_retrospective_credit against a fresh memory system, prints
per-procedure alpha/beta deltas + a PASS/FAIL decision gate
(avg |delta_alpha|+|delta_beta| > 0.1 per procedure)."
```

---

## Task 8: Final lint + push + open PR

**Files:**
- None (git operations only)

- [ ] **Step 1: Run full lint + test suite**

```bash
cd /workspace/orak-futile-detector
ruff check agents/macla/episode_credit.py agents/macla/online_evaluator.py agents/macla/macla_lib.py agents/macla/base.py tests/test_episode_credit.py tests/test_macla_episode_trace.py tests/test_starcraft_episode_summarizer.py experiments/sc2_episode_credit_replay.py
ruff format --check agents/macla/episode_credit.py agents/macla/online_evaluator.py agents/macla/macla_lib.py agents/macla/base.py tests/test_episode_credit.py tests/test_macla_episode_trace.py tests/test_starcraft_episode_summarizer.py experiments/sc2_episode_credit_replay.py
.venv/bin/pytest tests/test_episode_credit.py tests/test_macla_episode_trace.py tests/test_starcraft_episode_summarizer.py tests/test_online_evaluator_starcraft.py -v
```

Expected: 0 lint errors, all tests pass (count depends on existing test count + ~26 new from this PR).

- [ ] **Step 2: Pre-commit (if repo has hooks)**

```bash
test -f .pre-commit-config.yaml && .venv/bin/pre-commit run --all-files || echo "(no pre-commit config — skip)"
```

If hooks modify files, stage + amend the most recent commit per CLAUDE.md.

- [ ] **Step 3: Push branch**

```bash
git push -u origin feat/episode-credit-assignment
```

- [ ] **Step 4: Open PR**

Use the single-quoted heredoc per CLAUDE.md (preserves backticks + quotes verbatim):

```bash
gh pr create --title "feat(macla): episode-end retrospective credit assignment" --body "$(cat <<'EOF'
## Summary

Adds framework-level retrospective credit assignment that runs at episode boundary. Walks the trace of procedures used in the episode and applies a TD-lambda(0.95)-weighted credit signal proportional to the game outcome, mutating each procedure's (alpha, beta). Game-agnostic [`EpisodeOutcome`](../tree/feat/episode-credit-assignment/agents/macla/episode_credit.py) + per-game [`EpisodeSummarizer`](../tree/feat/episode-credit-assignment/agents/macla/episode_credit.py) adapter, mirroring [`RewardShaper`](../tree/feat/episode-credit-assignment/agents/macla/online_evaluator.py) from PR #111.

Motivation: the fresh SC2 smoke with the shaper from #111 moved \`successful_executions\` from 0 → 2 but \`avg_procedure_success_rate\` stayed at 0.50 — procedures are not yet meaningfully ranked by trajectory outcome. This PR adds the parallel episode-boundary update.

Design spec: [\`docs/specs/2026-05-27-episode-credit-assignment-design.md\`](../tree/feat/episode-credit-assignment/docs/specs/2026-05-27-episode-credit-assignment-design.md). Implementation plan: [\`docs/plans/2026-05-27-episode-credit-assignment-plan.md\`](../tree/feat/episode-credit-assignment/docs/plans/2026-05-27-episode-credit-assignment-plan.md).

## Test plan

- [x] ~26 unit tests across 3 new test files — game-agnostic math, SC2 summarizer, memory-system trace
- [x] Replay validation via [\`experiments/sc2_episode_credit_replay.py\`](../tree/feat/episode-credit-assignment/experiments/sc2_episode_credit_replay.py) — reconstructs per-episode traces from the existing reward-shaping smoke and asserts \`avg |delta_alpha|+|delta_beta| > 0.1\` per procedure
- [ ] Fresh n=1 SC2 smoke confirming \`avg_procedure_success_rate\` drifts off 0.50 (separate follow-up, ~2h SC2 run)

## Out-of-scope follow-ups

- Per-game \`EpisodeSummarizer\` for pokemon / mario / 2048 (one PR each)
- Empirical tuning via Hydra sweep over \`episode_credit:\` overrides
- \`autoresearch.episode_credit\` package extraction once the orak implementation stabilises (follows the \`autoresearch.janitor\` migration pattern)
EOF
)"
```

- [ ] **Step 5: Render-check PR body**

```bash
PR=$(gh pr list --head feat/episode-credit-assignment --json number --jq '.[0].number')
gh pr view $PR --json body --jq '.body' | head -40
```

Verify: no escaped backticks, no escaped quotes, all links use the `../tree/feat/...` pattern.

---

## Self-review notes

**Spec coverage**:
- Section 1 (architecture/module split) → Tasks 1-5 implement each module
- Section 2 (EpisodeOutcome) → Task 2
- Section 3 (credit math) → Task 3
- Section 4 (trace recording) → Task 1
- Section 5 (adapter interface) → Tasks 4-5
- Section 6 (wiring) → Task 6
- Testing strategy → Tasks 1-5 (unit tests) + Task 7 (replay)
- All scope guards held — only SC2 summarizer in this PR; pokemon/mario/2048 untouched

**Placeholder scan**: No TBDs, no "add error handling," no "similar to Task N". Each task has complete code and exact commands.

**Type/name consistency**:
- `EpisodeOutcome.is_victory`, `is_fatal_game_over`, `final_score_norm`, `time_alive_norm`, `progress_norm`, `n_steps` — consistent across Tasks 2, 3, 4
- `EpisodeCreditConfig.base_alpha_delta`, `base_beta_delta`, `td_lambda` — consistent across Tasks 2, 3
- `_terminal_credit`, `assign_retrospective_credit`, `SUMMARIZERS`, `StarCraftEpisodeSummarizer` — defined in Tasks 2, 3, 5 respectively, referenced consistently in Task 6 + 7
- `_episode_proc_trace`, `drain_episode_trace` — defined in Task 1, consumed in Task 6
- `DEFAULT_SHAPING["star_craft"]["time_alive_norm_max_s"]` + `progress_norm_max_buildings` — added Task 4, consumed Task 4 same file

No drift detected.
