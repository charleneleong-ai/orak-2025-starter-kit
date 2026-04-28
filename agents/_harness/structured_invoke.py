"""LLM invocation helper that preserves usage metadata.

LangChain's ``llm.with_structured_output(schema).invoke(messages)`` returns
only the parsed Pydantic model — it discards the raw AIMessage and the
``usage_metadata``/``response_metadata`` fields with token counts. That means
the harness's ``extract_cache_stats`` never has anything to read.

The fix is simple: pass ``include_raw=True``, which returns
``{"raw": AIMessage, "parsed": Model, "parsing_error": Optional[Exception]}``.
This helper hides the boilerplate so every game-base call site can keep
the previous one-liner shape.
"""
from __future__ import annotations

from typing import Any


def structured_invoke_with_usage(
    llm: Any,
    messages: Any,
    output_schema: Any,
) -> tuple[Any, Any]:
    """Invoke ``llm`` with structured output, preserving raw usage metadata.

    Returns ``(parsed, usage)``:

    * ``parsed`` — the Pydantic model instance the caller expects.
    * ``usage`` — the raw AIMessage's ``usage_metadata`` dict (langchain
      0.3+ standardised shape: ``input_tokens``, ``output_tokens``, plus
      ``input_token_details.cache_read``). Falls back to
      ``response_metadata.token_usage`` (older shape) if usage_metadata is
      missing. ``None`` if the underlying message has neither.

    Raises ``ValueError`` if the structured-output parser couldn't reconstruct
    the schema — this is what previously surfaced as a ``pass`` action in
    pokemon's silent-fallback handler.
    """
    structured = llm.with_structured_output(output_schema, include_raw=True)
    result = structured.invoke(messages)
    parsed = result.get("parsed") if isinstance(result, dict) else None
    raw = result.get("raw") if isinstance(result, dict) else None
    err = result.get("parsing_error") if isinstance(result, dict) else None

    if parsed is None:
        msg = f"structured-output parsing failed for {output_schema.__name__}: {err}"
        raise ValueError(msg)

    usage = _extract_usage(raw)
    return parsed, usage


def _extract_usage(raw: Any) -> Any:
    """Pull a usage object out of a langchain AIMessage.

    Returns whatever shape is most useful for ``extract_cache_stats``:

    * langchain ``usage_metadata`` (preferred) — dict with ``input_tokens``,
      ``output_tokens``, ``input_token_details.cache_read``. We re-key it to
      match ``extract_cache_stats``' expectations
      (``prompt_tokens_details.cached_tokens``).
    * Older ``response_metadata.token_usage`` — passed through as-is.
    """
    if raw is None:
        return None

    um = getattr(raw, "usage_metadata", None)
    if um:
        # langchain's standardised dict has cache reads under
        # ``input_token_details.cache_read``. Re-shape to the OpenAI/vLLM
        # shape ``extract_cache_stats`` already understands.
        cached = 0
        details = um.get("input_token_details") if isinstance(um, dict) else None
        if isinstance(details, dict):
            cached = details.get("cache_read", 0)
        return {
            "prompt_tokens": um.get("input_tokens", 0) if isinstance(um, dict) else 0,
            "completion_tokens": um.get("output_tokens", 0) if isinstance(um, dict) else 0,
            "prompt_tokens_details": {"cached_tokens": cached},
        }

    rm = getattr(raw, "response_metadata", None)
    if rm and isinstance(rm, dict):
        return rm.get("token_usage")
    return None
