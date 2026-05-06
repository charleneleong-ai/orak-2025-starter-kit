"""Pin the MACLA token-logging contract.

Pre-fix, ``raw_requests.jsonl`` showed ``"tokens": {"prompt": 0,
"completion": 0, "total": 0}`` for every step on every MACLA agent.
Two bugs combined:

1. ``safe_structured_invoke`` returned only the parsed Pydantic model
   from ``llm.with_structured_output(...)``, throwing away the AIMessage
   wrapper that carries ``usage_metadata``.
2. ``BaseOrakAgent._postprocess_log_extras`` only extracted cached
   tokens — even when usage WAS available, prompt/completion never
   reached ``log_extras``. The non-MACLA path surfaced them in
   ``BaseOrakAgent.get_action`` directly, but MACLA's override only
   called ``_postprocess_log_extras`` and skipped that block.

These tests pin both contracts so a future refactor doesn't silently
regress token telemetry again.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).resolve().parent.parent


# ── _extract_usage / safe_structured_invoke ────────────────────────────


def _load_structured_output():
    """Load structured_output without dragging the full agents package."""
    sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location(
        "agents.macla.structured_output_test_copy",
        _REPO / "agents/macla/structured_output.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_so = _load_structured_output()
_extract_usage = _so._extract_usage
safe_structured_invoke = _so.safe_structured_invoke


def test_extract_usage_lifts_langchain_usage_metadata():
    """The canonical case: AIMessage with usage_metadata dict."""
    raw = SimpleNamespace(
        usage_metadata={"input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290}
    )
    out = _extract_usage(raw)
    assert out == {
        "tokens_prompt": 1234,
        "tokens_completion": 56,
        "tokens_total": 1290,
        "raw_usage_metadata": raw.usage_metadata,
    }


def test_extract_usage_falls_back_to_response_metadata_token_usage():
    """Older provider integrations stash counts under response_metadata."""
    raw = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        },
    )
    out = _extract_usage(raw)
    assert out["tokens_prompt"] == 10
    assert out["tokens_completion"] == 5
    assert out["tokens_total"] == 15


def test_extract_usage_synthesises_total_when_missing():
    """If total_tokens isn't in the payload, sum prompt + completion."""
    raw = SimpleNamespace(usage_metadata={"input_tokens": 7, "output_tokens": 3})
    out = _extract_usage(raw)
    assert out["tokens_total"] == 10


def test_extract_usage_returns_none_when_neither_field_present():
    """No usage anywhere → None, downstream falls back to zero (no crash)."""
    raw = SimpleNamespace(usage_metadata=None, response_metadata={})
    assert _extract_usage(raw) is None
    assert _extract_usage(None) is None


# ── safe_structured_invoke contract ─────────────────────────────────────


class _Schema:
    """Minimal BaseModel-like stand-in. Pydantic isn't strictly needed
    here — we only check that the wrapper returns a 2-tuple."""


def test_safe_structured_invoke_returns_tuple_with_usage():
    """Pre-fix this returned bare parsed model; the new contract is
    always a (parsed, usage_or_None) tuple. Pinning it prevents an
    accidental revert that would break the MACLA caller."""
    parsed = MagicMock(name="ParsedSchema")
    raw = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    )
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = {"parsed": parsed, "raw": raw, "parsing_error": None}
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm

    result, usage = safe_structured_invoke(llm, [], _Schema)

    assert result is parsed
    assert usage["tokens_prompt"] == 100
    assert usage["tokens_completion"] == 20
    llm.with_structured_output.assert_called_once_with(_Schema, include_raw=True)


def test_safe_structured_invoke_handles_provider_without_include_raw():
    """Some langchain provider integrations silently drop include_raw and
    return the bare parsed model — we degrade gracefully (usage=None)
    instead of crashing."""
    from pydantic import BaseModel as PydanticModel

    class _Stub(PydanticModel):
        x: int = 1

    parsed = _Stub(x=42)
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = parsed  # no dict wrapper
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm

    result, usage = safe_structured_invoke(llm, [], _Stub)
    assert result is parsed
    assert usage is None


# ── _postprocess_log_extras token surfacing ────────────────────────────


def _load_postprocess():
    """Extract just the helper function from agents/base.py without
    importing the full module (which pulls weave + pydantic)."""
    src = (_REPO / "agents/base.py").read_text()
    start = src.index("    def _postprocess_log_extras")
    rest = src[start:]
    lines = rest.splitlines(keepends=True)
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        # Next sibling method (4-space indent + def)
        if line.startswith("    def "):
            end_idx = i
            break
    fn_src = "".join(lines[:end_idx]) if end_idx else rest

    # Wrap into a free function for isolation: drop the leading 4-space
    # method indent and rewrite ``self`` → an explicit object.
    import textwrap

    fn_src = textwrap.dedent(fn_src)

    # Provide stubs for the helpers it imports
    ns: dict = {
        "extract_cache_stats": lambda usage: {"cached_tokens": 0},
        "Any": object,
    }
    exec(
        "def _postprocess(self, log_extras, usage):\n"
        + textwrap.indent(fn_src.split("\n", 1)[1], "    ").rstrip()
        + "\n",
        ns,
    )
    return ns["_postprocess"]


_postprocess = _load_postprocess()


class _StubAgent:
    def __init__(self):
        self._cached_tokens_total = 0
        self._pending_fallback = None


def test_postprocess_extracts_dict_usage_into_token_keys():
    """The new path: safe_structured_invoke's normalised dict carries
    canonical keys; _postprocess copies them into log_extras."""
    extras: dict = {}
    usage = {"tokens_prompt": 500, "tokens_completion": 80, "tokens_total": 580}
    _postprocess(_StubAgent(), extras, usage)
    assert extras["tokens_prompt"] == 500
    assert extras["tokens_completion"] == 80
    assert extras["tokens_total"] == 580


def test_postprocess_extracts_openai_shaped_usage_object():
    """Backwards compat: when an OpenAI usage object is passed (e.g.
    from a non-MACLA agent's direct llm.invoke), surface tokens via
    attribute access."""
    extras: dict = {}
    usage = SimpleNamespace(prompt_tokens=42, completion_tokens=8, total_tokens=50)
    _postprocess(_StubAgent(), extras, usage)
    assert extras["tokens_prompt"] == 42
    assert extras["tokens_completion"] == 8
    assert extras["tokens_total"] == 50


def test_postprocess_does_not_overwrite_existing_token_total():
    """If the caller already populated tokens_total (e.g. the BaseOrak
    direct path), don't double-count."""
    extras = {"tokens_total": 999}
    usage = {"tokens_prompt": 1, "tokens_completion": 2, "tokens_total": 3}
    _postprocess(_StubAgent(), extras, usage)
    assert extras["tokens_total"] == 999  # untouched
    assert "tokens_prompt" not in extras


def test_postprocess_handles_none_usage_gracefully():
    """Procedure-cache hits don't invoke the LLM → usage is None."""
    extras: dict = {}
    _postprocess(_StubAgent(), extras, None)
    assert "tokens_prompt" not in extras
    assert "tokens_completion" not in extras
