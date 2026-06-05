"""Pin the per-step weave-client wrap short-circuit.

Pre-fix every step wrapped ``agent.act`` in ``with_weave_client(...)``.
``init_weave``'s short-circuit always missed (``ensure_project_exists``
mismatch with the agent's init), so each call ran ``finish()`` and
spawned a rich-progress-bar refresh thread. After ~hundreds of steps
the OS thread limit hit — see ``game_logs/pokemon_red/20260507_003616/``
(died step 199). The fix skips the wrap when the active weave client
already matches the agent's project.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evaluation_utils.runner import act_with_weave_context


@pytest.fixture
def weave(monkeypatch):
    """Mock the two weave entry points the helper hits. Tests set
    ``weave.get.return_value`` and read ``weave.wrap.call_count``."""
    from weave.trace.context import weave_client_context as wcc

    get = MagicMock(return_value=None)
    wrap = MagicMock(return_value=contextlib.nullcontext())
    monkeypatch.setattr(wcc, "get_weave_client", get)
    monkeypatch.setattr(wcc, "with_weave_client", wrap)
    return SimpleNamespace(get=get, wrap=wrap)


def _agent(project="chaleong/orak-pokemon-red"):
    return SimpleNamespace(
        _weave_project=project,
        act=lambda obs, step: {"action": "noop"},
    )


def test_short_circuit_when_project_matches(weave):
    weave.get.return_value = SimpleNamespace(entity="chaleong", project="orak-pokemon-red")
    assert act_with_weave_context(_agent(), {}, 1) == {"action": "noop"}
    weave.wrap.assert_not_called()


def test_thousand_steps_never_wrap(weave):
    """Crash repro: pre-fix this would hit the OS thread limit."""
    weave.get.return_value = SimpleNamespace(entity="chaleong", project="orak-pokemon-red")
    agent = _agent()
    for step in range(1000):
        act_with_weave_context(agent, {}, step)
    assert weave.wrap.call_count == 0


@pytest.mark.parametrize(
    "current",
    [
        SimpleNamespace(entity="chaleong", project="orak-super-mario"),  # different game
        None,  # cold start
        SimpleNamespace(project="orak-pokemon-red"),  # client missing entity
    ],
)
def test_wraps_when_active_client_does_not_match(weave, current):
    weave.get.return_value = current
    act_with_weave_context(_agent(), {}, 1)
    assert weave.wrap.call_count == 1


def test_short_circuit_on_disabled_stub(weave):
    """WEAVE_DISABLED makes weave.init return an entity=project="DISABLED"
    stub that never matches the real project — pre-guard it cache-missed
    every step and leaked a finish() thread per call (~12/min in qwen35_n3
    seed 1). The guard must skip the wrap."""
    weave.get.return_value = SimpleNamespace(entity="DISABLED", project="DISABLED")
    assert act_with_weave_context(_agent(), {}, 1) == {"action": "noop"}
    weave.wrap.assert_not_called()


@pytest.mark.parametrize("project", [None, ""])
def test_pass_through_when_no_weave_project(weave, project):
    agent = SimpleNamespace(act=lambda obs, step: {"ok": True})
    if project is not None:
        agent._weave_project = project
    assert act_with_weave_context(agent, {}, 1) == {"ok": True}
    weave.get.assert_not_called()
    weave.wrap.assert_not_called()
