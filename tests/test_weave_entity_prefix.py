"""Pin the contract that ``_weave_project`` always carries an entity prefix.

When ``WANDB_ENTITY`` is unset, ``WandbConfig.entity`` is ``None``. The
old code stored ``_weave_project = "orak-pokemon-red"`` (bare project),
and ``runner.py`` rpartitions on ``/`` to set the per-act() weave client
context — yielding ``entity=None``, which the trace.wandb.ai server
rejects with ``403 Forbidden / Project not found`` once per agent
step. A 300-step run produced 4,507 of these errors in the log.

We test the resolver directly to keep the test free of weave/wandb
network calls and pydantic-model boilerplate.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_REPO = Path(__file__).resolve().parent.parent

# Load only the helper symbol — importing agents.base whole pulls weave,
# pydantic models, harness modules, etc. The function is pure and the
# WandbConfig dependency is structural (it just reads .entity / .project).
_BASE_PATH = _REPO / "agents/base.py"


def _load_resolver():
    """Read agents/base.py and exec just enough of it to grab the helper."""
    src = _BASE_PATH.read_text()

    # Find the function definition and the closing of its body. We extract
    # the function as standalone source and exec it in a fresh namespace
    # so we don't pay the cost of importing weave/wandb/etc.
    start = src.index("def _resolve_weave_project")
    # Helper ends at the next top-level ``class`` or ``def`` keyword.
    rest = src[start:]
    # Find the next top-level def/class after ours.
    lines = rest.splitlines(keepends=True)
    end_line_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.startswith(("def ", "class ")):
            end_line_idx = i
            break
    fn_src = "".join(lines[:end_line_idx]) if end_line_idx else rest

    ns: dict = {}
    exec(fn_src, ns)
    return ns["_resolve_weave_project"]


_resolve = _load_resolver()


def _cfg(*, project="orak-pokemon-red", entity=None):
    return SimpleNamespace(project=project, entity=entity)


def test_explicit_entity_takes_precedence():
    """``WANDB_ENTITY=team`` wins over whatever the wandb run resolved."""
    run = SimpleNamespace(entity="from-netrc")
    assert _resolve(_cfg(entity="explicit-team"), run) == "explicit-team/orak-pokemon-red"


def test_falls_back_to_wandb_run_entity_when_config_entity_unset():
    """The regression: env var unset → config.entity is None.

    Pre-fix behaviour stored the bare project name and produced 4,500+
    403s per run. Post-fix we lift the entity off the wandb run that
    wandb just resolved from ``~/.netrc``.
    """
    run = SimpleNamespace(entity="chaleong")
    assert _resolve(_cfg(entity=None), run) == "chaleong/orak-pokemon-red"


def test_returns_bare_project_when_no_entity_anywhere():
    """If neither config nor run carry an entity, fall back gracefully.

    ``weave.init`` will then fail and ``_weave_client`` becomes None,
    which the runner already handles. We don't want to invent an entity.
    """
    run = SimpleNamespace(entity=None)
    assert _resolve(_cfg(entity=None), run) == "orak-pokemon-red"


def test_handles_missing_wandb_run():
    """Weave-disabled / wandb-disabled paths skip ``wandb.init`` and pass
    ``None`` for the run. The helper must not blow up."""
    assert _resolve(_cfg(entity=None), None) == "orak-pokemon-red"
    assert _resolve(_cfg(entity="team"), None) == "team/orak-pokemon-red"


def test_handles_run_without_entity_attribute():
    """Defensive: some test doubles don't expose ``.entity``.

    The helper must use ``getattr`` so it doesn't AttributeError on a
    sloppy mock.
    """
    bare_run = object()  # has no .entity
    assert _resolve(_cfg(entity=None), bare_run) == "orak-pokemon-red"
