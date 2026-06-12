"""Role-based thinking split: a separate fast (non-thinking) LLM for the
per-step action call, while the planner/reflector keep the thinking LLM.

The action call is the hot path (runs every game-step); planner/reflector are
occasional. On a Qwen3 reasoning model, routing the action call to an
``enable_thinking: false`` instance cuts the bulk of per-step latency while
strategy still reasons. Opt-in via ``config.fast_extra_body``; empty (default)
shares one LLM so behaviour is unchanged.
"""

from __future__ import annotations

import inspect

import pytest

import agents.macla.base as base_mod
import agents.macla.unified as unified_mod
from config.agent_config import LocalConfig


class TestConfigField:
    def test_fast_extra_body_defaults_empty(self):
        cfg = LocalConfig(class_name="UnifiedMaclaAgent", model="x")
        assert cfg.fast_extra_body == {}

    def test_fast_extra_body_in_to_dict(self):
        cfg = LocalConfig(
            class_name="UnifiedMaclaAgent",
            model="x",
            fast_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        assert cfg.to_dict()["fast_extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }


class TestWiring:
    """Source-grep the call routing: the per-step action call uses the fast
    LLM; the strategic planner/reflector keep the thinking LLM."""

    def test_action_call_uses_fast_llm(self):
        src = inspect.getsource(unified_mod)
        assert "safe_structured_invoke(self._fast_llm, messages, self._action_schema)" in src
        # and not the old thinking-LLM action call
        assert "safe_structured_invoke(self._llm, messages, self._action_schema)" not in src

    @pytest.mark.parametrize(
        "role_ctor", ["LLMSubtaskPlanner(llm=self._llm", "LLMSelfReflector(self._llm"]
    )
    def test_planner_and_reflector_keep_thinking_llm(self, role_ctor):
        assert role_ctor in inspect.getsource(unified_mod)

    def test_every_init_path_sets_fast_llm(self):
        src = inspect.getsource(base_mod)
        # one assignment per init path (Vertex, OpenAI, local) — share or split
        assert src.count("self._fast_llm") >= 3
        # local path opts into the split when config asks (extra_body swapped)
        assert '{**llm_kwargs, "extra_body": fast_extra_body}' in src
