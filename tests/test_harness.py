"""Unit tests for agents/_harness/. No LLM calls — all pure-python paths."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agents._harness import (
    ErrorClass,
    StepRecord,
    TrajectoryWriter,
    classify,
    convert_scratchpad_to_think,
    extract_cache_stats,
    format_recent_history,
    has_incomplete_scratchpad,
    jittered_backoff,
    with_retries,
)

# ── prompt_caching ──────────────────────────────────────────────────────


def test_extract_cache_stats_chatcompletions_object():
    """vLLM + OpenAI ChatCompletions both use this shape."""

    class Details:
        cached_tokens = 60

    class U:
        prompt_tokens_details = Details()
        prompt_tokens = 120
        completion_tokens = 15

    s = extract_cache_stats(U())
    assert s["cached_tokens"] == 60
    assert s["input_tokens"] == 120
    assert s["output_tokens"] == 15


def test_extract_cache_stats_chatcompletions_dict():
    s = extract_cache_stats(
        {
            "prompt_tokens_details": {"cached_tokens": 60},
            "prompt_tokens": 120,
            "completion_tokens": 15,
        }
    )
    assert s["cached_tokens"] == 60
    assert s["input_tokens"] == 120
    assert s["output_tokens"] == 15


def test_extract_cache_stats_openai_responses():
    """OpenAI Responses API (used by OpenAIPokemonVectorMemoryAgent)."""

    class Details:
        cached_tokens = 75

    class U:
        input_tokens_details = Details()
        input_tokens = 150
        output_tokens = 20

    s = extract_cache_stats(U())
    assert s["cached_tokens"] == 75
    assert s["input_tokens"] == 150
    assert s["output_tokens"] == 20


def test_extract_cache_stats_no_cache_details():
    """First call before cache warms — no prompt_tokens_details."""
    s = extract_cache_stats({"prompt_tokens": 100, "completion_tokens": 10})
    assert s == {"cached_tokens": 0, "input_tokens": 100, "output_tokens": 10}


def test_extract_cache_stats_handles_none():
    s = extract_cache_stats(None)
    assert s == {"cached_tokens": 0, "input_tokens": 0, "output_tokens": 0}


# ── retry_utils ─────────────────────────────────────────────────────────


def test_jittered_backoff_grows_with_attempt():
    # base=1, max=100. Attempt 1 ~ [1, 1.5], attempt 4 ~ [8, 12]
    d1 = jittered_backoff(1, base_delay=1.0, max_delay=100.0)
    d4 = jittered_backoff(4, base_delay=1.0, max_delay=100.0)
    assert 1.0 <= d1 <= 1.5
    assert 8.0 <= d4 <= 12.0


def test_jittered_backoff_caps_at_max():
    # 2^60 base would overflow — must cap
    d = jittered_backoff(60, base_delay=1.0, max_delay=10.0)
    assert d <= 15.0  # max + jitter_ratio * max


def test_classify_5xx_is_transient():
    err = Exception("oops")
    err.status_code = 502
    c = classify(err)
    assert c.cls == ErrorClass.TRANSIENT
    assert c.status == 502


def test_classify_401_is_terminal():
    err = Exception("unauthorized")
    err.status_code = 401
    c = classify(err)
    assert c.cls == ErrorClass.TERMINAL


def test_classify_rate_limit_message():
    c = classify(Exception("rate limit exceeded"))
    assert c.cls == ErrorClass.TRANSIENT


def test_classify_invalid_api_key_terminal():
    c = classify(Exception("Invalid API key provided"))
    assert c.cls == ErrorClass.TERMINAL


def test_with_retries_success_first_try():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert with_retries(fn, max_attempts=3) == "ok"
    assert len(calls) == 1


def test_with_retries_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)  # don't actually sleep
    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            err = Exception("server error")
            err.status_code = 503
            raise err
        return "ok"

    assert with_retries(fn, max_attempts=5, base_delay=0.01) == "ok"
    assert len(attempts) == 3


def test_with_retries_terminal_raises_immediately(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    attempts = []

    def fn():
        attempts.append(1)
        err = Exception("Invalid API key provided")
        err.status_code = 401
        raise err

    with pytest.raises(Exception) as info:
        with_retries(fn, max_attempts=3)
    assert len(attempts) == 1
    assert info.value.__classified__.cls == ErrorClass.TERMINAL


def test_with_retries_exhausts(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    attempts = []

    def fn():
        attempts.append(1)
        err = Exception("502 bad gateway")
        err.status_code = 502
        raise err

    with pytest.raises(Exception, match="502"):
        with_retries(fn, max_attempts=3, base_delay=0.01)
    assert len(attempts) == 3


# ── trajectory ──────────────────────────────────────────────────────────


def test_convert_scratchpad_to_think():
    src = "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD> answer"
    assert convert_scratchpad_to_think(src) == "<think>thinking</think> answer"


def test_has_incomplete_scratchpad():
    assert has_incomplete_scratchpad("<REASONING_SCRATCHPAD>partial")
    assert not has_incomplete_scratchpad("<REASONING_SCRATCHPAD>x</REASONING_SCRATCHPAD>")
    assert not has_incomplete_scratchpad("plain text")


def test_step_record_to_sharegpt():
    r = StepRecord(
        step=3,
        system_prompt="sys",
        user_prompt="u",
        assistant_output="a",
        action="left",
        reasoning="r",
        tokens_prompt=10,
        tokens_completion=5,
        tokens_total=15,
        cached_tokens=8,
    )
    sg = r.to_sharegpt()
    assert sg["step"] == 3
    assert sg["action"] == "left"
    assert sg["tokens"]["cached"] == 8
    assert sg["conversations"][0] == {"from": "system", "value": "sys"}
    assert sg["conversations"][1] == {"from": "human", "value": "u"}
    assert sg["conversations"][2]["from"] == "gpt"


def test_trajectory_writer_success_path(tmp_path: Path):
    w = TrajectoryWriter(tmp_path, model="test-model")
    w.add_step(
        StepRecord(step=1, system_prompt="s", user_prompt="u", assistant_output="a", action="up")
    )
    w.add_step(
        StepRecord(
            step=2, system_prompt="s", user_prompt="u2", assistant_output="a2", action="left"
        )
    )
    target = w.flush_episode(episode_id=0, completed=True, final_score=42.0, game_name="mario")
    assert target.name == "trajectory_samples.jsonl"
    line = json.loads(target.read_text().strip())
    assert line["completed"] is True
    assert line["n_steps"] == 2
    assert line["n_fallbacks"] == 0
    assert line["final_score"] == 42.0


def test_trajectory_writer_fallback_routes_to_failed(tmp_path: Path):
    w = TrajectoryWriter(tmp_path)
    w.add_step(
        StepRecord(
            step=1,
            system_prompt="s",
            user_prompt="u",
            assistant_output="a",
            action="up",
            is_fallback=True,
            fallback_reason="llm_error",
        )
    )
    target = w.flush_episode(episode_id=0, completed=True, final_score=0.0, game_name="mario")
    assert target.name == "failed_trajectories.jsonl"
    line = json.loads(target.read_text().strip())
    assert line["n_fallbacks"] == 1


def test_trajectory_writer_incomplete_routes_to_failed(tmp_path: Path):
    w = TrajectoryWriter(tmp_path)
    w.add_step(
        StepRecord(step=1, system_prompt="s", user_prompt="u", assistant_output="a", action="up")
    )
    target = w.flush_episode(episode_id=0, completed=False, final_score=0.0, game_name="mario")
    assert target.name == "failed_trajectories.jsonl"


def test_trajectory_writer_recent_returns_last_k(tmp_path: Path):
    w = TrajectoryWriter(tmp_path)
    for i in range(5):
        w.add_step(
            StepRecord(
                step=i,
                system_prompt="s",
                user_prompt=f"u{i}",
                assistant_output="a",
                action=f"act{i}",
            )
        )
    last3 = w.recent(3)
    assert [r.step for r in last3] == [2, 3, 4]
    # k larger than buffer returns the whole thing
    assert len(w.recent(99)) == 5
    # k=0 returns empty
    assert w.recent(0) == []


def test_format_recent_history_empty():
    assert format_recent_history([]) == ""


def test_format_recent_history_score_delta_and_state_change():
    records = [
        StepRecord(
            step=1,
            system_prompt=None,
            user_prompt="u",
            assistant_output="a",
            action="north",
            info_score=0.0,
            obs_digest="aaa",
        ),
        StepRecord(
            step=2,
            system_prompt=None,
            user_prompt="u",
            assistant_output="a",
            action="north",
            info_score=0.0,
            obs_digest="aaa",
        ),
        StepRecord(
            step=3,
            system_prompt=None,
            user_prompt="u",
            assistant_output="a",
            action="east",
            info_score=1.0,
            obs_digest="bbb",
        ),
    ]
    out = format_recent_history(records)
    lines = out.splitlines()
    assert "step 1: action=north" in lines[0]
    assert "score=0" in lines[0]
    assert "state=initial" in lines[0]
    assert "state=unchanged (loop?)" in lines[1]
    assert "(+0)" in lines[1]
    assert "state=changed" in lines[2]
    assert "(+1)" in lines[2]


def test_format_recent_history_handles_missing_score():
    records = [
        StepRecord(step=1, system_prompt=None, user_prompt="u", assistant_output="a", action="x"),
    ]
    out = format_recent_history(records)
    assert "score=?" in out
    assert "state=?" in out


def test_format_recent_history_action_truncation():
    long_action = "a" * 200
    records = [
        StepRecord(
            step=1,
            system_prompt=None,
            user_prompt="u",
            assistant_output="a",
            action=long_action,
            info_score=1.0,
            obs_digest="x",
        ),
    ]
    out = format_recent_history(records)
    # Action is clipped to 60 chars in the formatted line
    assert "a" * 60 in out
    assert "a" * 61 not in out


def test_step_record_to_sharegpt_with_outcome_fields():
    r = StepRecord(
        step=4,
        system_prompt="sys",
        user_prompt="u",
        assistant_output="a",
        action="left",
        info_score=2.0,
        obs_digest="deadbeef",
    )
    sg = r.to_sharegpt()
    assert sg["step"] == 4
    # to_sharegpt is the on-disk wire format — outcome fields don't have to
    # round-trip, but the dataclass should preserve them in memory.
    assert r.info_score == 2.0
    assert r.obs_digest == "deadbeef"


def test_trajectory_writer_clears_buffer_after_flush(tmp_path: Path):
    w = TrajectoryWriter(tmp_path)
    w.add_step(
        StepRecord(step=1, system_prompt="s", user_prompt="u", assistant_output="a", action="up")
    )
    w.flush_episode(episode_id=0, completed=True, final_score=1.0, game_name="m")
    # New episode starts fresh
    w.add_step(
        StepRecord(step=1, system_prompt="s", user_prompt="u", assistant_output="a", action="up")
    )
    target = w.flush_episode(episode_id=1, completed=True, final_score=1.0, game_name="m")
    lines = target.read_text().strip().splitlines()
    assert len(lines) == 2  # both episodes appended
    second = json.loads(lines[-1])
    assert second["episode_id"] == 1
    assert second["n_steps"] == 1  # didn't carry over
