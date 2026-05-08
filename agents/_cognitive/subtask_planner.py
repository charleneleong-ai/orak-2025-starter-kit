"""Subtask planner — task decomposition for long-horizon games.

Lifted from pokemon's ``OpenAIPokemonVectorMemoryAgent._module_subtask_planning``
and reshaped as a reusable cognitive module. Same pattern as
``VectorMemoryProvider`` — abstract base + concrete LLM-backed implementation,
opt-in via game-agent config.

When to use it:

* **Long-horizon planning games** (pokemon: "become Pokemon Champion" → "leave
  starting house" → "walk to lab" → "talk to Oak"). The agent gets stuck
  without explicit decomposition.
* **Multi-stage tasks** where the action policy benefits from a current
  sub-goal hint (e.g. "find a switch" vs "exit through door").

When NOT to use it:

* **Reactive games with simple action spaces** (mario, 2048). The bottleneck
  is perception/strategy, not goal decomposition. Adding a planner doubles
  inference cost for no measurable lift.

Cost: 1 extra LLM call per step (or per N steps if cached). Roughly 2× the
per-step inference cost when used naively.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

# Generic prompts — game-specific overrides go via the game adapter (see
# ``LLMSubtaskPlanner.from_adapter`` below).
DEFAULT_SYSTEM_PROMPT = """You are a sub-goal planner for an agent playing a game.

Given the current observation, the agent's overall goal, and the recent step
history, produce a concrete near-term subtask the agent should focus on next.

Output format (use exactly these section headers):

### Subtask_reasoning
<2-3 sentences explaining why this subtask is the right immediate focus>

### Subtask
<a single concrete sub-goal in plain language, no more than one sentence>

The subtask should be achievable in 5-30 game steps. Avoid abstract subgoals
like "make progress" — instead pick something concrete like "exit the
starting room through the door" or "merge two 64 tiles in the bottom-left
corner". Reuse the prior subtask if it's still the right focus."""

DEFAULT_USER_PROMPT_TEMPLATE = """### Overall goal
{goal}

### Current observation
{observation}

### Recent step history (last few)
{history}

### Last subtask the agent was working on
{last_subtask}

What is the right immediate subtask now? Output the two sections only."""


class SubtaskPlanner(ABC):
    """Abstract base for sub-goal planners."""

    @abstractmethod
    def plan(
        self,
        *,
        goal: str,
        observation: str,
        history: str = "",
        last_subtask: str | None = None,
    ) -> str:
        """Return the next subtask description (one short sentence)."""

    def stats(self) -> dict[str, Any]:
        """Optional runtime stats (call count, etc.) for observability."""
        return {}


class LLMSubtaskPlanner(SubtaskPlanner):
    """Subtask planner that calls an LLM to produce the sub-goal.

    Compatible with langchain ``BaseChatModel`` (ChatOpenAI / ChatVertexAI /
    vLLM via ChatOpenAI with custom base_url) — the planner only uses
    ``llm.invoke(messages)`` returning an AIMessage with ``.content``.

    Game adapters can override the system or user prompt by passing
    ``system_prompt=`` / ``user_prompt_template=`` at construction. The
    template is formatted with ``goal``, ``observation``, ``history``,
    ``last_subtask``.

    Last subtask is cached on the planner — callers don't need to track it.
    Set ``replan_every=N`` to skip planning for N-1 steps and reuse the cached
    subtask. Default 1 (plan every step). Larger values cut inference cost
    proportionally.
    """

    def __init__(
        self,
        llm: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
        replan_every: int = 1,
        observation_chars: int = 600,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._replan_every = max(1, replan_every)
        self._observation_chars = observation_chars
        self._last_subtask: str | None = None
        self._step_count: int = 0
        self._steps_since_plan: int = self._replan_every  # force first call to plan
        self._call_count: int = 0
        self._parse_failures: int = 0

    @property
    def name(self) -> str:
        return "llm_subtask"

    def plan(
        self,
        *,
        goal: str,
        observation: str,
        history: str = "",
        last_subtask: str | None = None,
    ) -> str:
        self._step_count += 1
        self._steps_since_plan += 1
        cached = last_subtask if last_subtask is not None else self._last_subtask

        # Reuse cached subtask between replans (counter-based, not modulo —
        # avoids edge cases at step boundaries).
        if cached and self._steps_since_plan < self._replan_every:
            return cached
        self._steps_since_plan = 0

        user_text = self._user_prompt_template.format(
            goal=goal or "(no overall goal specified)",
            observation=(observation or "")[: self._observation_chars],
            history=history or "(no history yet)",
            last_subtask=cached or "(none yet — this is the first plan)",
        )
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=user_text),
        ]
        try:
            response = self._llm.invoke(messages)
            self._call_count += 1
            content = response.content if hasattr(response, "content") else str(response)
            subtask = self._parse_subtask(content)
            if subtask:
                self._last_subtask = subtask
                return subtask
            self._parse_failures += 1
        except Exception as e:
            logger.warning(f"[SubtaskPlanner] plan() failed: {e}")
            self._parse_failures += 1

        # Fall back to cached subtask, or a generic one
        return cached or "Continue making progress toward the goal."

    def _parse_subtask(self, text: str) -> str | None:
        """Extract the ``### Subtask`` section content."""
        if not text:
            return None
        m = re.search(r"###\s*Subtask\s*\n+(.+?)(?=\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip().split("\n")[0].strip()
        # Fallback — sometimes the LLM forgets section headers; take the first
        # short non-empty line
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) < 200 and not line.startswith("#"):
                return line
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._call_count,
            "steps": self._step_count,
            "parse_failures": self._parse_failures,
            "last_subtask": self._last_subtask or "",
        }
