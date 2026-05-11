"""LangGraph variant of UnifiedMaclaAgent.

Reshapes the LLM-fallback path of ``UnifiedMaclaAgent._base_fallback`` as a
LangGraph ``StateGraph``. Adds an optional **self-verification** pass —
propose → re-read obs → confirm or revise — that the parent class can't
express without method-body branching.

The self-verification pattern here is **Reflexion-style** (Shinn et al.,
2023, *Reflexion: Language Agents with Verbal Reinforcement Learning*,
arXiv:2303.11366). Reflexion's contribution is letting an agent verbally
critique its own output before committing — distinct from classic ReAct
(Yao et al., 2023, *ReAct: Synergizing Reasoning and Acting in Language
Models*, arXiv:2210.03629), which interleaves Thought-Act-Observation
with environment observations between cycles. Our single-action-per-step
env API can't surface an env observation between sub-thoughts in one
agent call, so Reflexion's shape is the natural fit.

State threading uses ``MessagesState`` so the verify pass sees the
proposal as an ``AIMessage`` in proper conversation form — a real
``[System, Human(prompt), AI(proposal), Human(verify_request)]`` chain
rather than re-stating "you proposed X" as a string blob.

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

Without self-verification (default): straight-line replication of the
parent's flow. With self-verification (``use_verify_action: true``): two
LLM calls per fallback step (proposal + verify). Catches hallucinations
like the Stage C'++ T=0.3 tool-selection failure (LLM walked to (3,7)
but used ``move_to`` when ``warp_with_warp_point`` was needed).

Default off → identical behavior to ``UnifiedMaclaAgent``. Opt in per
agent config via ``use_verify_action: true``.
"""

from __future__ import annotations

import base64
import io
from typing import Annotated, Any

import weave
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from agents._harness import with_retries
from agents.macla.structured_output import safe_structured_invoke
from agents.macla.unified import UnifiedMaclaAgent


class ActionGraphState(MessagesState, total=False):
    """Per-step state threaded through the action graph.

    Subclasses ``MessagesState`` so ``messages`` is reduced with langgraph's
    ``add_messages`` (append + dedupe by id). The proposal node appends an
    ``AIMessage`` carrying its proposed action; the verify node then sees
    the proposal in proper conversation form rather than as a string blob.

    Other fields use the default reducer (last-write-wins); no two nodes
    race on the same key, so simple replacement is correct.
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

    # Explicit re-declaration of MessagesState's messages field with the
    # add_messages reducer so subclassing semantics are obvious to readers.
    messages: Annotated[list, add_messages]


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
        # Build the conversation seed (system + first user turn) and persist
        # it on the state so the verify node can extend it with proper
        # AIMessage/HumanMessage continuation rather than a string blob.
        user_text = state.get("user_text", "")
        messages: list[Any] = [
            SystemMessage(content=self._adapter.SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]
        action, reasoning = self._invoke_action_llm(
            messages=messages,
            obs_image=state.get("obs_image"),
        )
        proposal_msg = AIMessage(
            content=f"Action: {action}\nReasoning: {reasoning}",
            additional_kwargs={"proposed_action": action},
        )
        return {
            "proposed_action": action,
            "proposed_reasoning": reasoning,
            "messages": [*messages, proposal_msg],
        }

    def _node_verify_action(self, state: ActionGraphState) -> dict:
        """Second-pass LLM call (Reflexion-style; Shinn et al. 2023) that
        checks the proposed action against the observation + critique +
        subtask. Returns either the original action (if the verifier confirms)
        or a revised one.

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
        verify_request = HumanMessage(
            content=(
                "Re-read the observation and verify the action you just proposed will "
                "actually achieve the sub-goal. Watch for tool-selection mistakes "
                "(e.g. emitting move_to(x,y) when the tile is a WarpPoint and "
                "warp_with_warp_point is required).\n\n"
                f"### Observation (truncated)\n{observation[:1500]}\n\n"
                f"### Sub-goal\n{subtask or '(none)'}\n\n"
                f"### Critique\n{critique or '(none)'}\n\n"
                "If the proposed action is correct, restate it verbatim. "
                "If it's wrong, propose the corrected action."
            )
        )
        # Continue the conversation from the proposal — proper message
        # chain rather than re-stating "You proposed X" in a string.
        prior_messages = state.get("messages") or []
        verify_messages = [*prior_messages, verify_request]
        try:
            action, reasoning = self._invoke_action_llm(messages=verify_messages, obs_image=None)
            revised = action != proposed
            if revised:
                logger.debug(f"[LangGraph] self-verify revised action: {proposed!r} → {action!r}")
            verify_response = AIMessage(
                content=f"Action: {action}\nReasoning: {reasoning}",
                additional_kwargs={"verified_action": action, "was_revised": revised},
            )
            return {
                "verified_action": action,
                "verified_reasoning": reasoning,
                "was_revised": revised,
                "messages": [verify_request, verify_response],
            }
        except Exception as e:
            logger.warning(f"[LangGraph] verify_action failed; keeping proposed: {e}")
            return {
                "verified_action": proposed,
                "verified_reasoning": state.get("proposed_reasoning", ""),
                "was_revised": False,
            }

    # ── Shared LLM invocation ─────────────────────────────────────────

    def _invoke_action_llm(self, *, messages: list[Any], obs_image: Any | None) -> tuple[str, str]:
        """Call the action LLM via the same retries/structured-output stack
        the parent uses. Returns (action, reasoning) parsed from the result.

        ``messages`` is the conversation seed — typically ``[SystemMessage,
        HumanMessage]`` for the proposal pass, extended with
        ``[..., AIMessage(proposed), HumanMessage(verify_request)]`` for
        the verify pass. ``obs_image`` is appended to the LAST HumanMessage
        as a vision content part when the model supports vision.
        """
        supports_vision = getattr(self, "_supports_vision", True)

        # Optionally splice the obs image into the last HumanMessage's content
        if supports_vision and obs_image:
            buffered = io.BytesIO()
            obs_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage) and isinstance(messages[i].content, str):
                    messages[i] = HumanMessage(
                        content=[
                            {"type": "text", "text": messages[i].content},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                            },
                        ]
                    )
                    break

        # Stash the last user text for telemetry parity with the parent agent
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is not None and isinstance(last_human.content, str):
            self._last_llm_user_text = last_human.content

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
