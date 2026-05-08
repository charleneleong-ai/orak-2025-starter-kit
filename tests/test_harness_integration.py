"""Integration smoke test — agent end-to-end with mocked LLM.

Verifies the harness actually wires correctly into BaseOrakAgent:
* TrajectoryWriter is created on set_log_dir
* StepRecord lands per-step
* cached_tokens flow from usage object → log_extras → trajectory
* with_retries retries on transient errors, then succeeds
* _mark_fallback flips trajectory routing to failed_trajectories.jsonl

No real LLM call. No GPU. No game env. ~1 second runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents._harness import with_retries

# ── Minimal fake usage objects for the cache-stats path ──────────────────


class _FakeChatCompletionUsage:
    """Mimics vLLM / OpenAI ChatCompletions usage."""

    def __init__(self, prompt: int, completion: int, cached: int = 0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.prompt_tokens_details = type("D", (), {"cached_tokens": cached})()


# ── BaseOrakAgent end-to-end with mocked _get_action ─────────────────────


def _make_minimal_agent(tmp_path: Path):
    """Build a BaseOrakAgent subclass that doesn't need a real LLM or wandb."""
    from typing import ClassVar

    from agents.base import BaseOrakAgent
    from config.agent_config import LocalConfig
    from config.base import WandbConfig

    class _StubAgent(BaseOrakAgent):
        AGENT_TAGS: ClassVar[list[str]] = ["stub"]

        def __init__(self):
            cfg = LocalConfig(class_name="stub", model="stub-model", temperature=0.0)
            wandb_cfg = WandbConfig(mode="disabled")
            super().__init__(config=cfg, wandb_config=wandb_cfg)
            self._next_response = (
                "up",
                "fake reasoning",
                "fake goal",
                "Action: up",
                None,
                "fake prompt",
            )
            self._raise_next = False

        def _get_action(self, task_description, cur_state_str, obs_image=None):
            if self._raise_next:
                self._raise_next = False
                err = Exception("503 server error")
                err.status_code = 503
                raise err
            return self._next_response

    agent = _StubAgent()
    agent.set_log_dir(str(tmp_path))
    return agent


def test_trajectory_writer_attaches_on_set_log_dir(tmp_path: Path):
    agent = _make_minimal_agent(tmp_path)
    assert agent._trajectory_writer is not None
    assert agent._trajectory_writer.log_dir == tmp_path
    assert (tmp_path / "raw_requests.jsonl").parent.exists()


def test_act_records_step_in_trajectory(tmp_path: Path):
    agent = _make_minimal_agent(tmp_path)
    obs = {"obs_str": "Mario is at start", "game_info": {"score": 0}}

    # Set a usage object with cached_tokens=42 to verify the cache pathway
    agent._next_response = (
        "up",
        "reasoning",
        "goal",
        "Action: up",
        _FakeChatCompletionUsage(prompt=100, completion=10, cached=42),
        "user prompt",
    )

    result = agent.act(obs, step=1)
    assert result["action"] == "up"
    assert len(agent._trajectory_writer._buffer) == 1
    rec = agent._trajectory_writer._buffer[0]
    assert rec.step == 1
    assert rec.action == "up"
    assert rec.cached_tokens == 42
    assert rec.tokens_prompt == 100
    assert not rec.is_fallback


def test_episode_end_flushes_to_trajectory_samples(tmp_path: Path):
    agent = _make_minimal_agent(tmp_path)
    agent._next_response = (
        "up",
        "r",
        "g",
        "out",
        _FakeChatCompletionUsage(prompt=50, completion=5, cached=0),
        "p",
    )
    agent.act({"obs_str": "x", "game_info": {"score": 0}}, step=1)
    agent.act({"obs_str": "y", "game_info": {"score": 1}}, step=2)
    agent.record_episode_end(episode_id=0, game_name="mario", seed=0, final_score=10.0)

    out = tmp_path / "trajectory_samples.jsonl"
    assert out.exists(), "successful episode should land in trajectory_samples.jsonl"
    entry = json.loads(out.read_text().strip())
    assert entry["completed"] is True
    assert entry["n_steps"] == 2
    assert entry["n_fallbacks"] == 0
    assert entry["final_score"] == 10.0
    # Buffer cleared after flush
    assert len(agent._trajectory_writer._buffer) == 0


def test_mark_fallback_routes_to_failed_trajectories(tmp_path: Path):
    """Simulate the silent-fallback path: agent gets an error, marks fallback."""
    agent = _make_minimal_agent(tmp_path)
    agent._next_response = (
        "up",
        "r",
        "g",
        "out",
        _FakeChatCompletionUsage(prompt=50, completion=5),
        "p",
    )
    agent._mark_fallback("llm_error: ConnectionError")
    agent.act({"obs_str": "x", "game_info": {"score": 0}}, step=1)

    rec = agent._trajectory_writer._buffer[0]
    assert rec.is_fallback is True
    assert rec.fallback_reason == "llm_error: ConnectionError"

    agent.record_episode_end(episode_id=0, game_name="mario", seed=0, final_score=0.0)

    failed = tmp_path / "failed_trajectories.jsonl"
    assert failed.exists()
    entry = json.loads(failed.read_text().strip())
    assert entry["n_fallbacks"] == 1


def test_pending_fallback_consumed_per_step(tmp_path: Path):
    """_pending_fallback should reset after one step — not stick to subsequent ones."""
    agent = _make_minimal_agent(tmp_path)
    agent._next_response = (
        "up",
        "r",
        "g",
        "out",
        _FakeChatCompletionUsage(prompt=50, completion=5),
        "p",
    )
    agent._mark_fallback("llm_error")
    agent.act({"obs_str": "x", "game_info": {"score": 0}}, step=1)
    agent.act({"obs_str": "y", "game_info": {"score": 1}}, step=2)
    buf = agent._trajectory_writer._buffer
    assert buf[0].is_fallback is True
    assert buf[1].is_fallback is False, "second step should NOT inherit the fallback flag"


# ── retry_utils end-to-end via with_retries ──────────────────────────────


def test_with_retries_works_in_agent_context(tmp_path: Path, monkeypatch):
    """Realistic: simulate a transient 503 → retry → success on attempt 2."""
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_: None)

    calls = {"n": 0}

    def llm_call():
        calls["n"] += 1
        if calls["n"] < 2:
            err = Exception("503 server error")
            err.status_code = 503
            raise err
        return "ok"

    result = with_retries(llm_call, max_attempts=3, base_delay=0.01, label="test")
    assert result == "ok"
    assert calls["n"] == 2  # one retry, then success


# ── Variable-length _get_action tuple parser ────────────────────────────


def test_parser_handles_5_tuple_super_mario_shape(tmp_path: Path):
    """SuperMarioAgent returns (action, reasoning, output_text, usage, prompt) — 5 elements."""
    agent = _make_minimal_agent(tmp_path)
    agent._next_response = (
        "Jump Level: 0",
        "fake reasoning",
        "fake output",
        _FakeChatCompletionUsage(prompt=80, completion=8, cached=20),
        "fake prompt",
    )
    parsed = agent._parse_get_action_result(agent._next_response)
    assert parsed["action"] == "Jump Level: 0"
    assert parsed["reasoning"] == "fake reasoning"
    assert parsed["output_text"] == "fake output"
    assert parsed["usage"] is not None
    assert parsed.get("current_goal") is None  # absent from 5-tuple


def test_parser_handles_7_tuple_2048_shape(tmp_path: Path):
    """TwentyFourtyEightAgent: (action, reasoning, output_text, usage, prompt, game_phase, update_type)."""
    agent = _make_minimal_agent(tmp_path)
    tup = (
        "left",
        "r",
        "out",
        _FakeChatCompletionUsage(prompt=50, completion=5, cached=10),
        "p",
        "MID-CRITICAL",
        "atomic_entry",
    )
    parsed = agent._parse_get_action_result(tup)
    assert parsed["game_phase"] == "MID-CRITICAL"
    assert parsed["update_type"] == "atomic_entry"
    assert parsed["usage"] is not None


def test_parser_handles_8_tuple_macla_shape(tmp_path: Path):
    """UnifiedMaclaAgent: 4th slot is memory_stats, not usage."""
    agent = _make_minimal_agent(tmp_path)
    tup = (
        "up",
        "r",
        "out",
        {"method_counts": {"bayesian_procedure": 3}},
        "Goal: x",
        "EARLY",
        "macla_update",
        {"type": "atomic"},
    )
    parsed = agent._parse_get_action_result(tup)
    assert parsed["memory_stats"]["method_counts"]["bayesian_procedure"] == 3
    assert parsed["update_info"]["type"] == "atomic"
    # usage is absent — parser sets None so cache_stats noops gracefully
    assert parsed["usage"] is None


def test_parser_rejects_unknown_length(tmp_path: Path):
    agent = _make_minimal_agent(tmp_path)
    with pytest.raises(ValueError, match="Unknown _get_action tuple length"):
        agent._parse_get_action_result(("a", "b"))


def test_parser_passes_through_dict_form(tmp_path: Path):
    agent = _make_minimal_agent(tmp_path)
    out = agent._parse_get_action_result({"action": "x", "usage": None})
    assert out["action"] == "x"


def test_5_tuple_agent_flows_cache_stats_through_act(tmp_path: Path):
    """End-to-end: a 5-tuple agent (mario shape) now records cached_tokens."""
    agent = _make_minimal_agent(tmp_path)
    agent._next_response = (
        "Jump Level: 1",
        "r",
        "out",
        _FakeChatCompletionUsage(prompt=100, completion=10, cached=33),
        "p",
    )
    agent.act({"obs_str": "x", "game_info": {"score": 0}}, step=1)
    rec = agent._trajectory_writer._buffer[0]
    assert rec.cached_tokens == 33  # 5-tuple parser correctly extracted usage
    assert rec.action == "Jump Level: 1"


# ── Per-step legacy log + new trajectory log coexist ─────────────────────


def test_legacy_raw_requests_log_still_written(tmp_path: Path):
    agent = _make_minimal_agent(tmp_path)
    agent._next_response = (
        "up",
        "r",
        "g",
        "out",
        _FakeChatCompletionUsage(prompt=10, completion=2, cached=5),
        "p",
    )
    agent.act({"obs_str": "x", "game_info": {"score": 0}}, step=1)

    raw = tmp_path / "raw_requests.jsonl"
    assert raw.exists()
    rec = json.loads(raw.read_text().strip())
    assert rec["step"] == 1
    assert rec["tokens"]["cached"] == 5  # new field surfaced from usage
