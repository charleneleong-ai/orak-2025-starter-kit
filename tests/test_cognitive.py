"""Unit tests for agents/_cognitive/. Mocked embeddings — no API access."""
from __future__ import annotations

import numpy as np
import pytest

from agents._cognitive import MemoryProvider, VectorMemoryProvider


def _det_embed(text: str) -> np.ndarray:
    """Deterministic embedding: hash text into a 4-dim vector. Stable across runs."""
    h = abs(hash(text))
    return np.array([
        float((h >> 0) % 13),
        float((h >> 4) % 17),
        float((h >> 8) % 19),
        float((h >> 12) % 23),
    ])


def _orthogonal_embed(text: str) -> np.ndarray:
    """Orthogonal vectors per text — every retrieval has cos sim = 0 unless self-match."""
    mapping = {
        "alpha": np.array([1.0, 0, 0, 0]),
        "beta":  np.array([0, 1.0, 0, 0]),
        "gamma": np.array([0, 0, 1.0, 0]),
        "delta": np.array([0, 0, 0, 1.0]),
    }
    return mapping.get(text, np.array([0.5, 0.5, 0.5, 0.5]))


# ── interface contract ─────────────────────────────────────────────────


def test_vector_provider_implements_memory_provider():
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    assert isinstance(p, MemoryProvider)
    assert p.name == "vector"
    assert p.is_available() is True


def test_initialize_no_op():
    """initialize should not raise even without backend."""
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    p.initialize("session_x", game_name="2048")  # should not raise


# ── add_event + retrieval ──────────────────────────────────────────────


def test_add_event_stores_memory():
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    p.add_event("merged 4+4=8", {"step": 1})
    assert p.stats()["memory_count"] == 1


def test_add_event_dedupes_by_similarity():
    """Two near-identical embeddings → second is dropped."""
    p = VectorMemoryProvider(embedding_fn=lambda t: np.array([1.0, 0, 0, 0]))
    p.add_event("first")
    p.add_event("second")  # same embedding → similarity 1.0 → deduped
    assert p.stats()["memory_count"] == 1


def test_add_event_keeps_dissimilar_memories():
    p = VectorMemoryProvider(embedding_fn=_orthogonal_embed)
    p.add_event("alpha")
    p.add_event("beta")
    assert p.stats()["memory_count"] == 2


def test_add_event_skips_empty_content():
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    p.add_event("")
    p.add_event("   ")
    assert p.stats()["memory_count"] == 0


def test_prefetch_returns_empty_for_empty_query():
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    p.add_event("something")
    assert p.prefetch("") == ""


def test_prefetch_returns_empty_when_below_threshold():
    p = VectorMemoryProvider(embedding_fn=_orthogonal_embed)
    p.add_event("alpha")
    # Query "beta" is orthogonal → similarity 0 < threshold 0.5 → no recall
    out = p.prefetch("beta", threshold=0.5)
    assert out == ""


def test_prefetch_returns_formatted_match():
    p = VectorMemoryProvider(embedding_fn=_orthogonal_embed)
    p.add_event("alpha", metadata={"step": 5})
    out = p.prefetch("alpha")
    assert "alpha" in out
    assert "step 5" in out


def test_max_memories_evicts_oldest():
    # Use add_memory (no dedup) + a varying embedding so each insert is kept.
    # add_event would dedup on similar embeddings; we want pure FIFO eviction here.
    counter = {"n": 0}
    def varying_embed(_text: str) -> np.ndarray:
        counter["n"] += 1
        v = np.zeros(4)
        v[counter["n"] % 4] = 1.0
        return v
    p = VectorMemoryProvider(embedding_fn=varying_embed, max_memories=3)
    p.add_memory("first", {"step": 1})
    p.add_memory("second", {"step": 2})
    p.add_memory("third", {"step": 3})
    p.add_memory("fourth", {"step": 4})
    assert p.stats()["memory_count"] == 3
    contents = [m["content"] for m in p._memories]
    assert "first" not in contents
    assert "fourth" in contents


def test_stats_track_retrievals_and_hits():
    p = VectorMemoryProvider(embedding_fn=_orthogonal_embed)
    p.add_event("alpha")
    p.add_event("beta")
    # 1 retrieval (during add_event dedup check) per add — so 2 retrievals
    # before any prefetch. Adjust expectation accordingly.
    base = p.stats()["retrievals"]
    p.prefetch("alpha")  # match → +1 retrieval, +1 hit
    p.prefetch("zzz")    # no match → +1 retrieval, +0 hits
    s = p.stats()
    assert s["retrievals"] == base + 2
    assert s["hits"] >= 1


# ── compat shims for pokemon's existing API ────────────────────────────


def test_pokemon_compat_add_memory_no_dedup():
    """Pokemon's existing API didn't dedup — preserve that."""
    p = VectorMemoryProvider(embedding_fn=lambda t: np.array([1.0, 0, 0, 0]))
    p.add_memory("first")
    p.add_memory("second")  # same embedding, but compat shim doesn't dedup
    assert p.stats()["memory_count"] == 2


def test_pokemon_compat_format_memories_for_prompt_handles_empty():
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    assert p.format_memories_for_prompt([]) == "N/A"


def test_pokemon_compat_retrieve_similar_signature():
    p = VectorMemoryProvider(embedding_fn=_orthogonal_embed)
    p.add_memory("alpha", {"step": 1})
    hits = p.retrieve_similar("alpha", top_k=1, threshold=0.4)
    assert len(hits) == 1
    assert hits[0]["content"] == "alpha"
    assert "similarity" in hits[0]


# ── lifecycle ──────────────────────────────────────────────────────────


def test_on_session_end_no_op():
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    p.add_event("x")
    p.on_session_end()  # should not raise, no-op for in-memory provider


def test_system_prompt_block_default_empty():
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    assert p.system_prompt_block() == ""


# ── backend selection ──────────────────────────────────────────────────


def test_stats_reports_backend_in_use():
    p = VectorMemoryProvider(embedding_fn=_det_embed)
    p.add_event("hello")
    s = p.stats()
    assert s["backend"] == "injected"


# ── SubtaskPlanner ─────────────────────────────────────────────────────


class _FakeMsg:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """Mock langchain-style LLM that returns a fixed response."""
    def __init__(self, response_text: str = ""):
        self._response_text = response_text
        self.invoke_count = 0

    def invoke(self, messages):
        self.invoke_count += 1
        return _FakeMsg(self._response_text)


def test_subtask_planner_parses_section():
    from agents._cognitive import LLMSubtaskPlanner
    response = (
        "### Subtask_reasoning\n"
        "Agent must leave the starting house to make any progress.\n"
        "### Subtask\n"
        "Exit the starting room through the south door.\n"
    )
    p = LLMSubtaskPlanner(_FakeLLM(response))
    out = p.plan(goal="champion", observation="RedsHouse interior", history="")
    assert out == "Exit the starting room through the south door."
    assert p.stats()["calls"] == 1


def test_subtask_planner_caches_when_replan_every_gt_1():
    from agents._cognitive import LLMSubtaskPlanner
    response = "### Subtask\nGo north\n"
    fake = _FakeLLM(response)
    p = LLMSubtaskPlanner(fake, replan_every=3)
    # First call hits LLM (step 1, 1%3=1 → not replan trigger; but cached is None so falls through)
    p.plan(goal="g", observation="o")
    # Subsequent calls should reuse cache, not invoke LLM
    p.plan(goal="g", observation="o")
    p.plan(goal="g", observation="o")
    # Step 4 wraps around (4%3=1, not replan; uses cache)
    assert fake.invoke_count == 1


def test_subtask_planner_falls_back_on_invoke_failure():
    from agents._cognitive import LLMSubtaskPlanner

    class _FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("network error")

    p = LLMSubtaskPlanner(_FailingLLM())
    out = p.plan(goal="g", observation="o")
    assert "Continue" in out  # generic fallback string
    assert p.stats()["parse_failures"] == 1


def test_subtask_planner_handles_missing_section_header():
    """LLM forgets the ### Subtask header — planner falls back to first short line."""
    from agents._cognitive import LLMSubtaskPlanner
    response = "Walk south to the door"  # bare answer, no section
    p = LLMSubtaskPlanner(_FakeLLM(response))
    out = p.plan(goal="g", observation="o")
    assert out == "Walk south to the door"


def test_default_planner_prompt_is_game_agnostic():
    """The default planner system prompt must teach inference of sub-goals
    via general exploration/progress heuristics — not bake any specific
    game's content. This is what lets it generalize across long-horizon
    games. Per-adapter overrides remain available for crisp domain
    knowledge (see test_subtask_planner_uses_custom_system_prompt_when_provided)."""
    from agents._cognitive.subtask_planner import DEFAULT_SYSTEM_PROMPT
    sp = DEFAULT_SYSTEM_PROMPT
    # Must reference the heuristic categories.
    for heuristic in ("loop", "exit", "score"):
        assert heuristic.lower() in sp.lower(), f"default planner prompt missing heuristic: {heuristic}"
    # Must keep the parser contract.
    assert "### Subtask" in sp
    # Must NOT bake any single game's specifics — the prompt is shared across
    # all games via LLMSubtaskPlanner's default. (Per-adapter overrides exist
    # for games that genuinely need them; pokemon does not.)
    for game_specific in ("RedsHouse", "PalletTown", "OaksLab", "Pokémon", "mario", "tile"):
        assert game_specific not in sp, (
            f"default planner prompt contains game-specific token: {game_specific!r}"
        )


def test_pokemon_adapter_does_not_export_planner_system_prompt():
    """Pokemon should not need a per-adapter override — the abstract default
    prompt covers it. If pokemon ever needs one again (e.g. domain-specific
    edge case), the override mechanism is still there; this test exists only
    to lock in the design choice that pokemon stays on the default."""
    from agents.pokemon_red import game_adapter
    assert not hasattr(game_adapter, "SUBTASK_PLANNER_SYSTEM"), (
        "pokemon adapter should not override the planner system prompt — the "
        "abstract default in subtask_planner.py is intended to handle it. If "
        "this test fails, decide deliberately whether pokemon needs an override."
    )


def test_subtask_planner_uses_custom_system_prompt_when_provided():
    """LLMSubtaskPlanner.__init__ accepts system_prompt=; the messages it
    sends to the LLM must include that custom prompt rather than the default.
    This is the per-adapter override mechanism — kept for future games that
    have crisp domain knowledge worth baking in."""
    from agents._cognitive import LLMSubtaskPlanner

    captured = {}

    class _CapturingLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return _FakeMsg("### Subtask\nGo south\n")

    custom = "ADAPTER_OVERRIDE_SENTINEL custom planner prompt for this game."
    p = LLMSubtaskPlanner(_CapturingLLM(), system_prompt=custom)
    p.plan(goal="champion", observation="some observation")
    system_msg = captured["messages"][0]
    assert "ADAPTER_OVERRIDE_SENTINEL" in system_msg.content
    # And the default heuristics text must NOT leak through.
    assert "Anti-loop" not in system_msg.content


def test_local_sentence_transformers_backend(monkeypatch):
    """When OPENAI_API_KEY is missing, provider falls through to the local
    sentence-transformers model. Skipped if sentence-transformers is not
    installed in this environment."""
    pytest.importorskip("sentence_transformers")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = VectorMemoryProvider()  # no embedding_fn — exercises the real chain
    p.add_event("the quick brown fox")
    p.add_event("a different memory entirely")
    out = p.prefetch("brown fox")
    assert "brown fox" in out  # semantic match should succeed
    assert p.stats()["backend"] == "local"
