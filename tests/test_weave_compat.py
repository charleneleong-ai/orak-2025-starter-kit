"""Pin the langchain auto-tracer neutralization (weave Option B).

``WEAVE_TRACE_LANGCHAIN`` gates the tracer's env registration but the
``inheritable=True`` ContextVar hook attaches even when it's false, so the
tracer's ``on_chat_model_start`` still fires, JSON-encodes a langchain
``ModelMetaclass``, raises ``TypeError``, leaks trace context, and leaked
4500+ threads in qwen35_n3 seed 1 (froze 2026-05-29T02:39Z). The patch
no-ops the callbacks; this file pins that it does.
"""

from __future__ import annotations

import inspect
import os

import pytest

from evaluation_utils.weave_compat import neutralize_weave_langchain_tracer


def test_sets_trace_langchain_env_default(monkeypatch):
    monkeypatch.delenv("WEAVE_TRACE_LANGCHAIN", raising=False)
    neutralize_weave_langchain_tracer()
    assert os.environ["WEAVE_TRACE_LANGCHAIN"] == "false"


def test_setdefault_does_not_override_explicit_env(monkeypatch):
    monkeypatch.setenv("WEAVE_TRACE_LANGCHAIN", "true")
    neutralize_weave_langchain_tracer()
    assert os.environ["WEAVE_TRACE_LANGCHAIN"] == "true"


def test_tracer_callbacks_are_noop():
    if not neutralize_weave_langchain_tracer():
        pytest.skip("weave langchain integration not importable")
    from weave.integrations.langchain.langchain import WeaveTracer

    # Pre-patch on_chat_model_start raised TypeError on a metaclass arg; the
    # no-op must swallow any args and return None without raising.
    assert WeaveTracer.on_chat_model_start(object(), {"name": "x"}, []) is None
    assert WeaveTracer._on_chat_model_start(object(), {"name": "x"}, []) is None


def test_sweep_planner_does_not_disable_weave():
    """Option B: the sweep no longer sets the weave-disabling env, so weave runs
    default-on and traces link to the W&B run. Pins the assignments (not the bare
    names, which the explanatory comment still references)."""
    from experiments.autoresearch import OrakPlanner

    src = inspect.getsource(OrakPlanner.plan_iters)
    assert '"WEAVE_DISABLED": "true"' not in src
    assert '"WEAVE_ENABLED": "false"' not in src
