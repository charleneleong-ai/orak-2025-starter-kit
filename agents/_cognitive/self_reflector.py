"""Self-reflection module — periodic critique injected back into the action prompt.

Mirrors ``LLMSubtaskPlanner``'s shape (abstract base + LLM-backed concrete
class with a cached/throttled invocation pattern). Reflects every N steps,
returns a short critique that ``UnifiedMaclaAgent`` prepends to the next
action prompt as ``[Recent critique]\\n...``.

When to use it:

* **Long-horizon games** where the agent loses context between steps —
  pokemon dialog chains, multi-stage puzzles. The critique surfaces
  patterns (looping, missed opportunities) the action LLM forgot.

When NOT to use it:

* **Short-horizon reactive games** (mario, 2048) where per-step
  decisions don't accumulate state. Reflection cost outweighs benefit.

Cost: 1 extra LLM call every ``reflect_every`` steps. Default 10 → ~10%
overhead vs vanilla action loop. Cached between calls.

The legacy pokemon ``OpenAIPokemonVectorMemoryAgent._module_self_reflection``
returned structured JSON (eval / critique / NewFacts). For cross-game
generality the default here returns a free-form critique — game adapters
can override ``system_prompt`` if they want a stricter schema.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

DEFAULT_SYSTEM_PROMPT = """You are a self-reflection module for a game agent.

Given the agent's recent action history and the current observation, produce a
concise critique to help the agent's next decision. Be specific, observable,
and actionable — reference concrete coordinates / object names / map names
from the observation rather than generic advice.

Output format (use exactly this header):

### Critique
- <2-4 short bullet points>
- Focus: is the agent looping? what worked? what should be tried differently?

Keep total output under 200 words. If the agent appears to be making
progress, say so plainly — a short positive critique is better than a long
generic one."""

DEFAULT_USER_PROMPT_TEMPLATE = """### Recent step history (last few)
{history}

### Last action taken
{last_action}

### Current observation (truncated)
{observation}

What's the right concise critique for the agent right now? Output the
`### Critique` section only."""


class SelfReflector(ABC):
    """Abstract base for self-reflection modules."""

    @abstractmethod
    def reflect(
        self,
        *,
        observation: str,
        last_action: str,
        history: str = "",
    ) -> str:
        """Return a short critique string (empty if no critique to add)."""

    def stats(self) -> dict[str, Any]:
        return {}


class LLMSelfReflector(SelfReflector):
    """LLM-backed reflector — calls the LLM every ``reflect_every`` steps.

    Compatible with langchain ``BaseChatModel`` (ChatOpenAI / vLLM via
    custom base_url) — only uses ``llm.invoke(messages)`` returning an
    AIMessage-like object with ``.content``.

    Adapter override: pass ``system_prompt=`` to bake game-specific
    critique guidance (e.g. pokemon's "track which NPCs you have/haven't
    interacted with"). Default is game-agnostic.

    Cache: between LLM calls, ``reflect()`` returns the previous critique
    so the action loop always has a non-empty hint after the first call.
    """

    def __init__(
        self,
        llm: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
        reflect_every: int = 10,
        observation_chars: int = 600,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._reflect_every = max(1, reflect_every)
        self._observation_chars = observation_chars
        self._cached: str = ""
        self._step_count: int = 0
        self._steps_since_reflect: int = self._reflect_every  # force first call
        self._call_count: int = 0
        self._parse_failures: int = 0

    @property
    def name(self) -> str:
        return "llm_self_reflector"

    def reflect(
        self,
        *,
        observation: str,
        last_action: str,
        history: str = "",
    ) -> str:
        self._step_count += 1
        self._steps_since_reflect += 1
        if self._steps_since_reflect < self._reflect_every and self._cached:
            return self._cached
        self._steps_since_reflect = 0

        user_text = self._user_prompt_template.format(
            history=history or "(no history yet)",
            last_action=last_action or "(no action yet)",
            observation=(observation or "")[: self._observation_chars],
        )
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=user_text),
        ]
        try:
            response = self._llm.invoke(messages)
            self._call_count += 1
            content = response.content if hasattr(response, "content") else str(response)
            critique = self._parse_critique(content)
            if critique:
                self._cached = critique
                return critique
            self._parse_failures += 1
        except Exception as e:
            logger.warning(f"[SelfReflector] reflect() failed: {e}")
            self._parse_failures += 1
        return self._cached  # may be "" on first failure

    def _parse_critique(self, text: str) -> str:
        if not text:
            return ""
        m = re.search(r"###\s*Critique\s*\n+(.+?)(?=\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback — accept any non-empty content within length limits
        stripped = text.strip()
        if 0 < len(stripped) < 2000:
            return stripped
        return ""

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._call_count,
            "steps": self._step_count,
            "parse_failures": self._parse_failures,
            "cached_chars": len(self._cached),
        }
