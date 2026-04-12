"""
Structured output with JSON fallback for models that don't support native structured output.

Usage:
    result = safe_structured_invoke(llm, messages, GameAction)
    # Works with Gemini, OpenAI (native), and Ollama/vLLM (JSON fallback)
"""
import json
import re

from langchain_core.messages import BaseMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel


def safe_structured_invoke(
    llm,
    messages: list[BaseMessage],
    output_schema: type[BaseModel],
    fallback_to_json: bool = True,
) -> BaseModel:
    """
    Try native structured output first, fall back to JSON prompt + parsing.

    Args:
        llm: LangChain ChatModel instance
        messages: List of messages to send
        output_schema: Pydantic model class for the expected output
        fallback_to_json: If True, append JSON schema to prompt and parse response
    """
    # Try native structured output
    try:
        structured_llm = llm.with_structured_output(output_schema)
        return structured_llm.invoke(messages)
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
    return _parse_json_response(content, output_schema)


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

    raise ValueError(
        f"Could not parse {schema.__name__} from LLM response:\n{content[:500]}"
    )
