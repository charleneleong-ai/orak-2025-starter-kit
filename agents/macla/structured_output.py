"""
Structured output with JSON fallback for models that don't support native structured output.

Usage:
    result, usage = safe_structured_invoke(llm, messages, GameAction)
    # Works with Gemini, OpenAI (native), and Ollama/vLLM (JSON fallback)
"""

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel


def _extract_mean_logprob(logprobs: Any) -> float | None:
    """Compute mean per-token logprob from a langchain-openai logprobs dict.

    langchain-openai surfaces logprobs (when ``logprobs=True, top_logprobs=N``
    is set on the ChatOpenAI client) as ``response_metadata['logprobs']``
    with shape ``{"content": [{"token": str, "logprob": float, ...}, ...]}``.

    Stage M (third signal) calibrates procedure quality against the rolling
    distribution of these mean values, so cross-model safe. Returns None
    when the model didn't return logprobs (e.g. when the kwarg isn't
    supported or the API silently dropped it) — downstream signal then
    bootstraps to neutral 0.5.
    """
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    lps = [
        item.get("logprob")
        for item in content
        if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float))
    ]
    if not lps:
        return None
    return sum(lps) / len(lps)


def _extract_usage(raw: Any) -> dict | None:
    """Lift token counts off a LangChain AIMessage into a flat dict.

    Tries ``usage_metadata`` first (the canonical LangChain field on
    AIMessage; populated by every modern provider including the
    OpenAI-compatible vLLM endpoint), then falls back to legacy
    ``response_metadata['token_usage']`` for older provider integrations.
    Returns None if neither carries usage — in which case downstream
    token logging stays at zero, same as before this fix.

    Stage M: also surfaces ``mean_logprob`` extracted from
    ``response_metadata['logprobs']`` (None when the provider didn't
    return logprobs). The Bayesian selector calibrates procedure quality
    against the rolling distribution of these means.
    """
    if raw is None:
        return None
    response_meta = getattr(raw, "response_metadata", None) or {}
    mean_logprob = _extract_mean_logprob(
        response_meta.get("logprobs") if isinstance(response_meta, dict) else None
    )
    usage_meta = getattr(raw, "usage_metadata", None)
    if isinstance(usage_meta, dict) and usage_meta:
        # LangChain canonical: input_tokens / output_tokens / total_tokens.
        # Normalise to the names BaseOrakAgent.get_action expects so the
        # existing surfacing block in agents/base.py keeps working.
        return {
            "tokens_prompt": usage_meta.get("input_tokens", 0),
            "tokens_completion": usage_meta.get("output_tokens", 0),
            "tokens_total": usage_meta.get(
                "total_tokens",
                usage_meta.get("input_tokens", 0) + usage_meta.get("output_tokens", 0),
            ),
            "mean_logprob": mean_logprob,
            "raw_usage_metadata": usage_meta,
        }
    if isinstance(response_meta, dict):
        token_usage = response_meta.get("token_usage")
        if isinstance(token_usage, dict) and token_usage:
            return {
                "tokens_prompt": token_usage.get("prompt_tokens", 0),
                "tokens_completion": token_usage.get("completion_tokens", 0),
                "tokens_total": token_usage.get(
                    "total_tokens",
                    token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0),
                ),
                "mean_logprob": mean_logprob,
                "raw_usage_metadata": token_usage,
            }
    if mean_logprob is not None:
        return {
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "tokens_total": 0,
            "mean_logprob": mean_logprob,
        }
    return None


def safe_structured_invoke(
    llm,
    messages: list[BaseMessage],
    output_schema: type[BaseModel],
    fallback_to_json: bool = True,
) -> tuple[BaseModel, dict | None]:
    """
    Try native structured output first, fall back to JSON prompt + parsing.

    Args:
        llm: LangChain ChatModel instance
        messages: List of messages to send
        output_schema: Pydantic model class for the expected output
        fallback_to_json: If True, append JSON schema to prompt and parse response

    Returns:
        Tuple of ``(parsed_model, usage_dict_or_None)``. The usage dict carries
        ``tokens_prompt`` / ``tokens_completion`` / ``tokens_total`` keys that
        ``BaseOrakAgent.get_action`` already knows how to surface into
        ``raw_requests.jsonl``. Pre-fix, the whole AIMessage wrapper was
        thrown away — leaving ``"tokens": {"prompt": 0, "completion": 0}``
        in the per-step log, which broke W&B token cost tracking.
    """
    # Try native structured output. include_raw=True returns the wrapper
    # {"raw": AIMessage, "parsed": SchemaModel, "parsing_error": Exception?}
    # so we can read usage_metadata off the AIMessage.
    try:
        structured_llm = llm.with_structured_output(output_schema, include_raw=True)
        result = structured_llm.invoke(messages)
        if isinstance(result, dict) and "parsed" in result:
            parsed = result["parsed"]
            if parsed is not None:
                return parsed, _extract_usage(result.get("raw"))
            # Native structured-output returned no parse — fall through to
            # the JSON path so we still produce a result.
        elif isinstance(result, BaseModel):
            # Provider doesn't honour include_raw (some langchain integrations
            # silently ignore the kwarg) — we get just the parsed model and
            # lose usage on this path. Same as pre-fix behaviour for those
            # providers; not worse.
            return result, None
    except (NotImplementedError, AttributeError, TypeError):
        if not fallback_to_json:
            raise

    # Fallback: inject JSON schema into system prompt and parse response
    logger.debug(f"Falling back to JSON parsing for {output_schema.__name__}")
    schema_str = json.dumps(output_schema.model_json_schema(), indent=2)
    json_instruction = (
        f"\n\nYou MUST respond with valid JSON matching this schema:\n"
        f"```json\n{schema_str}\n```\n"
        f"Do not include any other text outside the JSON."
    )

    modified = list(messages)
    if modified and isinstance(modified[0], SystemMessage):
        modified[0] = SystemMessage(content=modified[0].content + json_instruction)
    else:
        modified.insert(0, SystemMessage(content=json_instruction))

    response = llm.invoke(modified)
    content = response.content if hasattr(response, "content") else str(response)
    parsed = _parse_json_response(content, output_schema)
    return parsed, _extract_usage(response)


def _parse_json_response(content: str, schema: type[BaseModel]) -> BaseModel:
    """Extract and parse JSON from LLM response text."""
    # Try direct parse
    try:
        return schema.model_validate_json(content)
    except Exception:
        pass

    # Try extracting from markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if code_block:
        try:
            return schema.model_validate_json(code_block.group(1).strip())
        except Exception:
            pass

    # Try extracting first {...} block
    brace_match = re.search(r"\{.*\}", content, re.DOTALL)
    if brace_match:
        try:
            return schema.model_validate_json(brace_match.group(0))
        except Exception:
            pass

    raise ValueError(f"Could not parse {schema.__name__} from LLM response:\n{content[:500]}")
