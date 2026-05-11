"""Action validators — plan-do-check loop + tool gating.

After the action LLM proposes an action, two complementary validators run:

1. :class:`ToolGateValidator` — **rule check** against the current observation
   (cheap, deterministic). Each game adapter exports a
   ``validate_action(action, obs) -> (bool, reason)`` function; the validator
   delegates. Adapters that don't export it get pass-through.

2. :class:`LLMPlanValidator` — **subgoal-alignment check** (1 extra LLM call).
   Asks the LLM whether the proposed action is consistent with the current
   subgoal. Catches actions that pass tool gating but ignore the planner's
   subgoal (the diagnosed failure mode behind PR #31's Stage C / C′++ wedges
   where the agent walked to (3,7) but never used ``warp_with_warp_point``).

Both compose into :class:`CompositeValidator`, which short-circuits on the
first rejection so the LLM-backed check isn't paid when the cheap check has
already rejected.

The retry loop lives in ``UnifiedMaclaAgent._base_fallback``: when validation
fails, the critique is appended to the user prompt and the action LLM is
re-invoked. Bounded retries (default ``validation_max_retries=2``) cap cost.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

DEFAULT_PLAN_CHECK_SYSTEM_PROMPT = """You are an action validator for a game agent.

Given the agent's current subgoal and the action it proposed, decide whether
the action is *consistent with the subgoal*. The action LLM can hallucinate
or ignore the subgoal — your job is to catch that.

You are NOT checking whether the action is mechanically legal (a separate
tool-gate pass does that). You are checking whether it MAKES PROGRESS on
the subgoal.

Output format (use exactly these two sections):

### Verdict
<one word: ALIGNED, DIVERGENT, or UNSURE>

### Critique
<if DIVERGENT or UNSURE: one short sentence explaining why, suitable for
showing to the action LLM on retry. Reference the subgoal text and the
proposed action concretely. If ALIGNED: write "(none)".>"""


DEFAULT_PLAN_CHECK_USER_TEMPLATE = """### Current subgoal
{subgoal}

### Action the agent proposed
{action}

### Current observation (truncated)
{observation}

Is the action consistent with the subgoal? Output the two sections only."""


class ActionValidator(ABC):
    """Abstract base for action validators."""

    @abstractmethod
    def validate(
        self,
        *,
        action: str,
        observation: str = "",
        subgoal: str = "",
    ) -> tuple[bool, str]:
        """Return ``(valid, reason)``. ``reason`` is shown to the action LLM
        on retry; the empty string means no critique."""

    def stats(self) -> dict[str, Any]:
        return {}


# ── Rule-based gate ────────────────────────────────────────────────────


class ToolGateValidator(ActionValidator):
    """Delegates to an adapter-side ``validate_action(action, observation)``.

    If the adapter doesn't export one (or ``adapter_check=None``), every
    action passes — adapters that haven't been wired up don't break the
    pipeline.
    """

    def __init__(
        self,
        *,
        adapter_check: Callable[[str, str], tuple[bool, str]] | None,
    ) -> None:
        self._adapter_check = adapter_check
        self._calls = 0
        self._rejects = 0

    def validate(
        self,
        *,
        action: str,
        observation: str = "",
        subgoal: str = "",
    ) -> tuple[bool, str]:
        self._calls += 1
        if self._adapter_check is None:
            return True, ""
        try:
            valid, reason = self._adapter_check(action, observation)
        except Exception as e:
            logger.warning(f"[ToolGate] adapter check raised; treating as pass-through: {e}")
            return True, ""
        if not valid:
            self._rejects += 1
        return bool(valid), reason or ""

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "rejects": self._rejects,
            "reject_rate": (self._rejects / self._calls) if self._calls else 0.0,
        }


# ── LLM-backed subgoal-alignment check ─────────────────────────────────


class LLMPlanValidator(ActionValidator):
    """LLM-judged check that the action makes progress on the current
    subgoal. ``invoke()`` returns an AIMessage-like with ``.content``."""

    def __init__(
        self,
        llm: Any,
        *,
        system_prompt: str = DEFAULT_PLAN_CHECK_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_PLAN_CHECK_USER_TEMPLATE,
        observation_chars: int = 600,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._observation_chars = observation_chars
        self._calls = 0
        self._rejects = 0
        self._parse_failures = 0

    def validate(
        self,
        *,
        action: str,
        observation: str = "",
        subgoal: str = "",
    ) -> tuple[bool, str]:
        # No subgoal to validate against → pass through (no LLM call).
        if not subgoal or not subgoal.strip():
            return True, ""

        user_text = self._user_prompt_template.format(
            subgoal=subgoal,
            action=action,
            observation=(observation or "")[: self._observation_chars],
        )
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=user_text),
        ]
        try:
            response = self._llm.invoke(messages)
            self._calls += 1
            content = response.content if hasattr(response, "content") else str(response)
            verdict, critique = self._parse(content)
            if verdict == "ALIGNED":
                return True, ""
            if verdict in ("DIVERGENT", "UNSURE"):
                self._rejects += 1
                # On UNSURE we still pass the action through but log the critique;
                # otherwise an indecisive LLM blocks every step. Caller can
                # treat UNSURE as warning-only.
                if verdict == "UNSURE":
                    return True, ""
                return False, critique
            self._parse_failures += 1
        except Exception as e:
            logger.warning(f"[PlanCheck] validate() failed; passing through: {e}")
            self._parse_failures += 1
        # Fail-open: any parse/LLM trouble → don't block.
        return True, ""

    def _parse(self, text: str) -> tuple[str, str]:
        if not text:
            return "UNPARSED", ""
        v = re.search(r"###\s*Verdict\s*\n+\s*([A-Z]+)", text, re.IGNORECASE)
        c = re.search(r"###\s*Critique\s*\n+(.+?)(?=\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
        verdict = (v.group(1).upper() if v else "UNPARSED").strip()
        critique = c.group(1).strip() if c else ""
        if critique == "(none)":
            critique = ""
        return verdict, critique

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "rejects": self._rejects,
            "parse_failures": self._parse_failures,
        }


# ── Composition ────────────────────────────────────────────────────────


class CompositeValidator(ActionValidator):
    """Runs each underlying validator in order; short-circuits on the first
    rejection so the LLM-backed check isn't paid when the cheap rule check
    has already rejected the action."""

    def __init__(self, validators: Sequence[ActionValidator]) -> None:
        self._validators = list(validators)

    def validate(
        self,
        *,
        action: str,
        observation: str = "",
        subgoal: str = "",
    ) -> tuple[bool, str]:
        for v in self._validators:
            valid, reason = v.validate(action=action, observation=observation, subgoal=subgoal)
            if not valid:
                return False, reason
        return True, ""

    def stats(self) -> dict[str, Any]:
        return {type(v).__name__: v.stats() for v in self._validators}
