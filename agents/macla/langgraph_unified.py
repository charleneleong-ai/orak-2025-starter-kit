"""LangGraph variant of UnifiedMaclaAgent.

Reshapes the LLM-fallback path of ``UnifiedMaclaAgent._base_fallback`` as a
LangGraph state machine. Adds an optional self-verification (Reflexion-style) pass
(propose → verify-against-obs → commit) that the parent class can't
express without method-body branching.

Graph shape::

    START → ┬→ compute_critique ┐
            └→ retrieve_memory  ┴→ plan_subtask → compose_prompt → invoke_llm
                                                                    │
                                  ┌─────────────────────────────────┘
                                  ▼
                              verify_action (conditional)
                                  │
                                  ▼
                                 END

Without self-verification (default): straight-line replication of the parent's flow.
With self-verification (``use_verify_action: true``): two-pass LLM. First pass proposes
an action; second pass re-reads the obs + critique and either confirms
the proposed action or revises it. Catches hallucinations like the Stage
C'++ T=0.3 tool-selection failure (LLM walked to (3,7) but used
``move_to`` when ``warp_with_warp_point`` was needed).

Default off → identical behavior to ``UnifiedMaclaAgent``. Opt in per
agent config via ``use_verify_action: true``.
"""

from __future__ import annotations

import base64
import io
from typing import Any, TypedDict

import weave
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger

from agents._harness import with_retries
from agents.macla.structured_output import safe_structured_invoke
from agents.macla.unified import UnifiedMaclaAgent


class ActionGraphState(TypedDict, total=False):
    """Per-step state threaded through the action graph.

    Each node reads/writes a strict subset; LangGraph's default reducer
    (last-write-wins) is fine because no two nodes race on the same key.
    """

    # Inputs from _base_fallback
    goal: str
    observation: str
    obs_image: Any
    user_text_base: str

    # Cognitive module outputs (populated by parallel nodes)
    critique: str
    recalled: str
    subtask: str

    # Composed prompt + LLM result
    user_text: str
    proposed_action: str
    proposed_reasoning: str

    # Self-verification outputs (populated only when enabled)
    verified_action: str
    verified_reasoning: str
    was_revised: bool


def _build_action_graph() -> Any:
    """Build (but don't run) the action graph. Pure-structural helper —
    test-friendly and stateless across calls.

    Nodes are stubs at this layer; the agent instance binds real callables
    at construction time via ``_build_action_graph_bound``.
    """

    def _noop(state: ActionGraphState) -> dict:
        return {}

    sg: StateGraph = StateGraph(ActionGraphState)
    for name in (
        "compute_critique",
        "retrieve_memory",
        "plan_subtask",
        "compose_prompt",
        "invoke_llm",
        "verify_action",
    ):
        sg.add_node(name, _noop)

    sg.add_edge(START, "compute_critique")
    sg.add_edge(START, "retrieve_memory")
    sg.add_edge("compute_critique", "plan_subtask")
    sg.add_edge("retrieve_memory", "plan_subtask")
    sg.add_edge("plan_subtask", "compose_prompt")
    sg.add_edge("compose_prompt", "invoke_llm")
    sg.add_edge("invoke_llm", "verify_action")
    sg.add_edge("verify_action", END)
    return sg.compile()


class LangGraphMaclaAgent(UnifiedMaclaAgent):
    """UnifiedMaclaAgent with the LLM-fallback path as a LangGraph.

    Default behavior matches the parent. Opt into self-verification (Reflexion-style)
    by setting ``use_verify_action: true`` in the agent YAML.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = kwargs.get("config") or (args[0] if args else None)
        self._use_verify_action = bool(getattr(config, "use_verify_action", False))
        self._verify_max_iterations = int(getattr(config, "verify_max_iterations", 1))
        self._action_graph = self._build_action_graph_bound()
        if self._use_verify_action:
            logger.info(
                f"[LangGraph] self-verification (Reflexion-style) enabled "
                f"(max_iterations={self._verify_max_iterations})"
            )

    # ── Graph construction (instance-bound) ───────────────────────────

    def _build_action_graph_bound(self) -> Any:
        """Build the graph with instance method bindings on each node."""
        sg: StateGraph = StateGraph(ActionGraphState)
        sg.add_node("compute_critique", self._node_compute_critique)
        sg.add_node("retrieve_memory", self._node_retrieve_memory)
        sg.add_node("plan_subtask", self._node_plan_subtask)
        sg.add_node("compose_prompt", self._node_compose_prompt)
        sg.add_node("invoke_llm", self._node_invoke_llm)
        sg.add_node("verify_action", self._node_verify_action)

        sg.add_edge(START, "compute_critique")
        sg.add_edge(START, "retrieve_memory")
        sg.add_edge("compute_critique", "plan_subtask")
        sg.add_edge("retrieve_memory", "plan_subtask")
        sg.add_edge("plan_subtask", "compose_prompt")
        sg.add_edge("compose_prompt", "invoke_llm")
        sg.add_edge("invoke_llm", "verify_action")
        sg.add_edge("verify_action", END)
        return sg.compile()

    # ── Node implementations ──────────────────────────────────────────

    def _node_compute_critique(self, state: ActionGraphState) -> dict:
        # Critique is already refreshed by _get_action's reflection hook; this
        # node just reads the cached value off self. Kept as a separate node
        # so future variants can compute critique inline within the graph.
        return {"critique": getattr(self, "_last_critique", "") or ""}

    def _node_retrieve_memory(self, state: ActionGraphState) -> dict:
        if self._memory_provider is None:
            return {"recalled": ""}
        query = f"{state.get('goal', '')} | {state.get('observation', '')[:300]}"
        try:
            recalled = self._memory_provider.prefetch(query) or ""
        except Exception as e:
            logger.warning(f"[LangGraph] retrieve_memory failed: {e}")
            recalled = ""
        return {"recalled": recalled}

    def _node_plan_subtask(self, state: ActionGraphState) -> dict:
        if self._subtask_planner is None:
            return {"subtask": ""}
        try:
            history_str = self._build_subtask_history()
            recalled = state.get("recalled", "")
            if recalled:
                history_str = (
                    f"### Recalled prior memories\n{recalled}\n\n"
                    f"### Recent steps (this episode)\n{history_str}"
                )
            subtask = self._subtask_planner.plan(
                goal=state.get("goal", ""),
                observation=state.get("observation", ""),
                history=history_str,
            )
            return {"subtask": subtask or ""}
        except Exception as e:
            logger.warning(f"[LangGraph] plan_subtask failed: {e}")
            return {"subtask": ""}

    def _node_compose_prompt(self, state: ActionGraphState) -> dict:
        user_text = state.get("user_text_base", "")
        critique = state.get("critique", "")
        recalled = state.get("recalled", "")
        subtask = state.get("subtask", "")

        if critique:
            user_text = f"[Recent critique]\n{critique}\n\n{user_text}"
        if recalled:
            user_text = f"[Recalled memories from prior steps]\n{recalled}\n\n{user_text}"
        if subtask:
            user_text = (
                f"[Current sub-goal — focus on this for the next few steps]\n"
                f"{subtask}\n\n{user_text}"
            )
        return {"user_text": user_text}

    def _node_invoke_llm(self, state: ActionGraphState) -> dict:
        action, reasoning = self._invoke_action_llm(
            user_text=state.get("user_text", ""),
            obs_image=state.get("obs_image"),
        )
        return {"proposed_action": action, "proposed_reasoning": reasoning}

    def _node_verify_action(self, state: ActionGraphState) -> dict:
        """Second-pass LLM call that checks the proposed action against the
        observation + critique + subtask. Returns either the original action
        (if the verifier confirms) or a revised one.

        Skipped entirely when ``use_verify_action`` is False — returns the
        proposed action unchanged.
        """
        proposed = state.get("proposed_action", "")
        if not self._use_verify_action or not proposed:
            return {
                "verified_action": proposed,
                "verified_reasoning": state.get("proposed_reasoning", ""),
                "was_revised": False,
            }

        critique = state.get("critique", "")
        subtask = state.get("subtask", "")
        observation = state.get("observation", "")
        verify_user = (
            f"You proposed the action: {proposed}\n"
            f"Reasoning: {state.get('proposed_reasoning', '')}\n\n"
            f"Re-read the observation and verify the action will actually achieve the "
            f"sub-goal. Watch for tool-selection mistakes (e.g. emitting move_to(x,y) "
            f"when the tile is a WarpPoint and warp_with_warp_point is required).\n\n"
            f"### Observation\n{observation[:1500]}\n\n"
            f"### Sub-goal\n{subtask or '(none)'}\n\n"
            f"### Critique\n{critique or '(none)'}\n\n"
            f"If the proposed action is correct, restate it verbatim. If it's wrong, "
            f"propose the corrected action."
        )
        try:
            action, reasoning = self._invoke_action_llm(user_text=verify_user, obs_image=None)
            revised = action != proposed
            if revised:
                logger.debug(f"[LangGraph] self-verify revised action: {proposed!r} → {action!r}")
            return {
                "verified_action": action,
                "verified_reasoning": reasoning,
                "was_revised": revised,
            }
        except Exception as e:
            logger.warning(f"[LangGraph] verify_action failed; keeping proposed: {e}")
            return {
                "verified_action": proposed,
                "verified_reasoning": state.get("proposed_reasoning", ""),
                "was_revised": False,
            }

    # ── Shared LLM invocation ─────────────────────────────────────────

    def _invoke_action_llm(self, *, user_text: str, obs_image: Any | None) -> tuple[str, str]:
        """Call the action LLM via the same retries/structured-output stack
        the parent uses. Returns (action, reasoning) parsed from the result.
        """
        supports_vision = getattr(self, "_supports_vision", True)
        if supports_vision and obs_image:
            user_content: Any = [{"type": "text", "text": user_text}]
            buffered = io.BytesIO()
            obs_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                }
            )
            human_content = user_content
        else:
            human_content = user_text

        messages = [
            SystemMessage(content=self._adapter.SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]
        self._last_llm_user_text = user_text

        result, _usage = with_retries(
            lambda: safe_structured_invoke(self._llm, messages, self._action_schema),
            label="langgraph_macla.llm",
        )
        if result is None:
            return self._adapter.DEFAULT_ACTION, ""
        action = (
            self._adapter.extract_action(result)
            if hasattr(self._adapter, "extract_action")
            else getattr(result, "action", "")
        )
        reasoning = getattr(result, "reasoning", "") or ""
        return action, reasoning

    # ── Public override ───────────────────────────────────────────────

    @weave.op()
    def _base_fallback(self, goal: str, observation: str, **kwargs: Any) -> tuple[list[str], str]:
        """LLM fallback — invokes the LangGraph state machine."""

        # Build the SafeDict-formatted base user text (parent uses inline class)
        class SafeDict(dict):  # noqa: D401
            def __missing__(self, key: str) -> str:  # type: ignore[override]
                return ""

        user_text_base = self._adapter.USER_PROMPT_TEMPLATE.format_map(
            SafeDict(
                last_action=getattr(self, "_last_action", "none"),
                cur_state_str=observation,
                task_description=goal,
                prev_state_str=getattr(self, "_prev_state_str", ""),
            )
        )

        initial: ActionGraphState = {
            "goal": goal,
            "observation": observation,
            "obs_image": kwargs.get("obs_image"),
            "user_text_base": user_text_base,
        }
        final = self._action_graph.invoke(initial)
        action = final.get("verified_action") or final.get("proposed_action") or ""
        reasoning = final.get("verified_reasoning") or final.get("proposed_reasoning") or ""
        action = self._validate_action(action)
        return ([action], reasoning)
