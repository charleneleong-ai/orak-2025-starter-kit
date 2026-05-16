"""Stage M (third signal): logprob_confidence via percentile rank.

Adds the third multiplicative signal to procedure quality:
``score = base_posterior × state_delta_confidence × logprob_confidence``

``Procedure.mean_logprob: float | None`` — mean per-token logprob from the
LLM call that generated this procedure's action sequence. Set at procedure
creation; ``None`` for procedures created before logprobs were plumbed
(backwards-compat).

``EnhancedHierarchicalMemorySystem._recent_logprobs: deque[float]`` —
rolling window of the last 50 mean_logprobs observed. Used as the
calibration distribution for percentile rank.

``BayesianProcedureSelector._logprob_confidence(entry)`` — returns:
  - 0.5 (neutral) if the entry's procedure.mean_logprob is None
  - 0.5 (neutral) if fewer than 10 samples in the rolling deque (bootstrap)
  - Else: percentile rank of entry.procedure.mean_logprob against the
    rolling deque, ∈ [0, 1].

``_compute_expected_utility`` multiplies by ``(0.5 + 0.5 * logprob_conf)``
— neutral 0.75 at 0.5, penalises to 0.5 at conf=0, leaves unchanged at 1.0.

Cross-model safe: each model's procedures calibrate against that model's
own logprob distribution. Distribution-free (no z-score assumption).
"""

from __future__ import annotations


def _mk_procedure(mean_logprob=None, map_name=None):
    from agents.macla.macla_lib import Procedure

    p = Procedure(goal="g", preconditions=[], steps=["s"])
    if map_name is not None:
        p.map_name = map_name
    if mean_logprob is not None:
        p.mean_logprob = mean_logprob
    return p


# ─── Procedure.mean_logprob ───────────────────────────────────────────────


def test_procedure_has_mean_logprob_field_default_none():
    from agents.macla.macla_lib import Procedure

    p = Procedure(goal="g", preconditions=[], steps=["s"])
    assert p.mean_logprob is None


def test_procedure_mean_logprob_settable():
    from agents.macla.macla_lib import Procedure

    p = Procedure(goal="g", preconditions=[], steps=["s"], mean_logprob=-1.2)
    assert p.mean_logprob == -1.2


# ─── _recent_logprobs deque ───────────────────────────────────────────────


def test_recent_logprobs_empty_on_init():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    assert list(mem._recent_logprobs) == []


def test_recent_logprobs_caps_at_50():
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    for i in range(70):
        mem._recent_logprobs.append(float(-i))
    assert len(mem._recent_logprobs) == 50
    assert mem._recent_logprobs[0] == -20.0  # oldest 20 evicted


def test_pending_logprob_default_none():
    """Slot used by the agent to hand off mean_logprob from the most-recent
    LLM call to the macla_lib procedure-creation site."""
    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    assert mem._pending_logprob is None


def test_recent_logprobs_survives_pickle_roundtrip():
    import pickle

    from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem

    mem = EnhancedHierarchicalMemorySystem()
    mem._recent_logprobs.append(-1.1)
    mem._recent_logprobs.append(-0.4)
    restored = pickle.loads(pickle.dumps(mem))
    assert list(restored._recent_logprobs) == [-1.1, -0.4]


# ─── logprob_confidence percentile rank ───────────────────────────────────


def test_logprob_confidence_neutral_when_entry_logprob_none():
    """Backwards-compat: pre-Stage-M procedures have mean_logprob=None and
    must score neutral 0.5."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    # populate the deque so the bootstrap path doesn't mask the test
    for lp in range(20):
        mem._recent_logprobs.append(-float(lp))
    entry = ProceduralMemoryEntry(procedure=_mk_procedure(mean_logprob=None))
    # Stage N: bootstrap returns 1.0 (neutral, no damping). See
    # tests/test_macla_stage_n_bootstrap_fix.py for the rationale.
    assert sel._logprob_confidence(entry) == 1.0


def test_logprob_confidence_neutral_when_fewer_than_10_samples():
    """Stage N bootstrap-neutral: with < 10 calibration samples the
    distribution isn't ranked yet — return 1.0 (no damping) so the new
    proc can fire while it earns its sample."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    for lp in range(5):
        mem._recent_logprobs.append(-float(lp))
    entry = ProceduralMemoryEntry(procedure=_mk_procedure(mean_logprob=-1.0))
    assert sel._logprob_confidence(entry) == 1.0


def test_logprob_confidence_high_when_entry_logprob_above_distribution():
    """An entry's mean_logprob better (closer to 0) than every sample in the
    rolling deque returns ~1.0."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    for lp in range(10):
        mem._recent_logprobs.append(-(lp + 1.0))  # -1.0 to -10.0
    entry = ProceduralMemoryEntry(procedure=_mk_procedure(mean_logprob=-0.1))
    assert sel._logprob_confidence(entry) == 1.0


def test_logprob_confidence_low_when_entry_logprob_below_distribution():
    """An entry's mean_logprob worse (more negative) than every sample
    returns 0.0."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    for lp in range(10):
        mem._recent_logprobs.append(-(lp + 1.0))  # -1.0 to -10.0
    entry = ProceduralMemoryEntry(procedure=_mk_procedure(mean_logprob=-100.0))
    assert sel._logprob_confidence(entry) == 0.0


def test_logprob_confidence_middle_when_entry_at_median():
    """An entry's mean_logprob equal to the median of the rolling deque
    returns ~0.5."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    # distribution: -0.0, -1.0, ..., -9.0; median ~ -4.5
    for lp in range(10):
        mem._recent_logprobs.append(-float(lp))
    entry = ProceduralMemoryEntry(procedure=_mk_procedure(mean_logprob=-4.5))
    conf = sel._logprob_confidence(entry)
    assert 0.4 <= conf <= 0.6, f"expected ~0.5 got {conf}"


# ─── _compute_expected_utility multiplies by logprob_confidence ───────────


def test_expected_utility_penalised_when_logprob_confidence_low():
    """A procedure with a poor mean_logprob (well below the rolling
    distribution) gets a lower EU than one with a strong mean_logprob,
    all else equal."""
    from agents.macla.macla_lib import (
        BayesianProcedureSelector,
        EnhancedHierarchicalMemorySystem,
        ProceduralMemoryEntry,
        Procedure,
    )

    mem = EnhancedHierarchicalMemorySystem()
    sel = BayesianProcedureSelector(mem)
    for lp in range(15):
        mem._recent_logprobs.append(-(lp + 1.0))

    obs = "Map Name: Route1\nScore: 0\n"

    def _mk_entry(mean_lp: float) -> ProceduralMemoryEntry:
        p = Procedure(
            goal="g",
            preconditions=[],
            steps=["s"],
            alpha=5,
            beta=1,
            map_name="Route1",
            mean_logprob=mean_lp,
        )
        e = ProceduralMemoryEntry(procedure=p, contexts={"general"}, goals={"g"})
        return e

    eu_high = sel._compute_expected_utility(_mk_entry(-0.1), obs, "g")
    eu_low = sel._compute_expected_utility(_mk_entry(-100.0), obs, "g")
    assert eu_low < eu_high, (
        f"Stage M logprob signal failed: high-logprob proc should outscore "
        f"low-logprob proc. got eu_high={eu_high:.4f} eu_low={eu_low:.4f}"
    )


# ─── _extract_mean_logprob from langchain response_metadata ───────────────


def test_extract_mean_logprob_returns_none_on_missing_logprobs():
    from agents.macla.structured_output import _extract_mean_logprob

    assert _extract_mean_logprob(None) is None
    assert _extract_mean_logprob({}) is None
    assert _extract_mean_logprob({"content": []}) is None


def test_extract_mean_logprob_averages_token_logprobs():
    """langchain-openai surfaces logprobs as:
    {"content": [{"token": "x", "logprob": -0.1, ...}, ...]}
    The mean over the `content` list is what we calibrate against."""
    from agents.macla.structured_output import _extract_mean_logprob

    logprobs_dict = {
        "content": [
            {"token": "Hello", "logprob": -0.1},
            {"token": " world", "logprob": -0.3},
            {"token": "!", "logprob": -0.2},
        ]
    }
    mean_lp = _extract_mean_logprob(logprobs_dict)
    assert mean_lp is not None
    assert abs(mean_lp - (-0.2)) < 1e-9


def test_extract_mean_logprob_handles_empty_content():
    from agents.macla.structured_output import _extract_mean_logprob

    assert _extract_mean_logprob({"content": []}) is None
