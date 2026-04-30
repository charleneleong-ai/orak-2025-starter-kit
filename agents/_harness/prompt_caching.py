"""Prompt caching helpers — measure prefix cache hits across our backends.

Two backends are in scope:

* **vLLM** (primary) serving ``unsloth/gemma-4-E4B-it`` on the OpenAI-compatible
  chat-completions endpoint. Auto-caches by byte-identical prompt prefix;
  emits ``prompt_tokens_details.cached_tokens`` in usage.
* **OpenAI cloud** (baselines) — ``BaseOpenAIAgent`` via langchain ChatOpenAI,
  plus ``OpenAIPokemonVectorMemoryAgent`` calling ``client.responses.create``
  directly. ChatCompletions emits ``prompt_tokens_details.cached_tokens``;
  the Responses API emits ``input_tokens_details.cached_tokens``.
"""
from __future__ import annotations

from typing import Any


def extract_cache_stats(usage: Any) -> dict[str, int]:
    """Pull cache hit + token counts from a usage object.

    Returns ``{"cached_tokens": N, "input_tokens": N, "output_tokens": N}``;
    missing fields default to 0.

    Handles three usage shapes that appear in this codebase:
    * vLLM / OpenAI ChatCompletions ``CompletionUsage`` —
      ``prompt_tokens_details.cached_tokens`` + ``prompt_tokens`` /
      ``completion_tokens``.
    * OpenAI Responses ``ResponseUsage`` —
      ``input_tokens_details.cached_tokens`` + ``input_tokens`` /
      ``output_tokens``.
    * Plain dict (some custom adapters wrap usage as ``dict``).
    """
    if usage is None:
        return {"cached_tokens": 0, "input_tokens": 0, "output_tokens": 0}

    cached = 0

    prompt_details = _get(usage, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached += _get(prompt_details, "cached_tokens", 0)

    inp_details = _get(usage, "input_tokens_details", None)
    if inp_details is not None:
        cached += _get(inp_details, "cached_tokens", 0)

    return {
        "cached_tokens": cached,
        "input_tokens": _get(usage, "prompt_tokens", 0) or _get(usage, "input_tokens", 0),
        "output_tokens": _get(usage, "completion_tokens", 0) or _get(usage, "output_tokens", 0),
    }


def _get(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
