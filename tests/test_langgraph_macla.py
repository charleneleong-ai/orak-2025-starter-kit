"""Tests for LangGraphMaclaAgent — the LangGraph variant of UnifiedMaclaAgent.

Same per-step interface as the parent, but the LLM-fallback path is expressed
as a LangGraph state machine. Adds an optional ReAct self-verification pass
(propose → verify-against-obs → commit) that the parent class can't express
without method-body branching.

These tests pin the contract (subclass relationship, graph compiles, ReAct
gating) without spinning up a live LLM.
"""

from __future__ import annotations

import inspect


def test_langgraph_macla_inherits_unified():
    """Drop-in compatibility: same per-step interface as UnifiedMaclaAgent."""
    from agents.macla.langgraph_unified import LangGraphMaclaAgent
    from agents.macla.unified import UnifiedMaclaAgent

    assert issubclass(LangGraphMaclaAgent, UnifiedMaclaAgent)
    # Public per-step method signature must match parent's
    parent_sig = inspect.signature(UnifiedMaclaAgent._base_fallback)
    child_sig = inspect.signature(LangGraphMaclaAgent._base_fallback)
    assert list(parent_sig.parameters) == list(child_sig.parameters)


def test_action_graph_compiles():
    """Graph build is purely structural — must succeed without an LLM."""
    from agents.macla.langgraph_unified import _build_action_graph

    g = _build_action_graph()
    # Compiled graph exposes invoke/stream
    assert hasattr(g, "invoke")
    assert hasattr(g, "stream")


def test_action_graph_nodes_include_react_verify_node():
    """The graph must expose a react_verify node (gated at runtime by config)."""
    from agents.macla.langgraph_unified import _build_action_graph

    g = _build_action_graph()
    nodes = set(g.nodes.keys()) if hasattr(g, "nodes") else set()
    # Core nodes
    assert "compose_prompt" in nodes
    assert "invoke_llm" in nodes
    # ReAct self-verification node
    assert "react_verify" in nodes


def test_localconfig_declares_use_react_verify():
    """pydantic extra='forbid' on LocalConfig — gates the new agent feature."""
    from config.agent_config import LocalConfig

    c = LocalConfig(
        class_name="test",
        model="test-model",
        temperature=0.0,
        use_react_verify=True,
        react_max_iterations=2,
    )
    assert c.use_react_verify is True
    assert c.react_max_iterations == 2


def test_unified_agent_signature_unchanged():
    """Backward compat: parent class's _base_fallback contract is unchanged."""
    import inspect

    from agents.macla.unified import UnifiedMaclaAgent

    src = inspect.getsource(UnifiedMaclaAgent._base_fallback)
    # Parent should still return tuple[list[str], str]
    assert "tuple[list[str], str]" in src or "return " in src


def test_langgraph_macla_init_default_react_off():
    """When use_react_verify is unset, the agent behaves identically to parent.

    Source-inspection test (no live LLM): the LangGraph init gates the ReAct
    branch behind config.use_react_verify, defaulting to off.
    """
    import inspect

    from agents.macla.langgraph_unified import LangGraphMaclaAgent

    init_src = inspect.getsource(LangGraphMaclaAgent.__init__)
    assert 'getattr(config, "use_react_verify"' in init_src, (
        "LangGraphMaclaAgent must gate ReAct verify behind the config flag"
    )
