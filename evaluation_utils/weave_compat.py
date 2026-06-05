import os


def neutralize_weave_langchain_tracer() -> bool:
    """No-op weave's langchain auto-tracer to prevent a thread-leak hang.

    ``WEAVE_DISABLED`` gates ``weave.init``/ops but NOT the langchain
    ``BaseTracer`` hook — that's gated by ``WEAVE_TRACE_LANGCHAIN`` plus an
    ``inheritable=True`` ContextVar registration (see
    ``weave/integrations/langchain/langchain.py``). The tracer's
    ``on_chat_model_start`` JSON-encodes a langchain ``ModelMetaclass``,
    raises ``TypeError``, leaks trace context, and accumulated 4500+ stuck
    threads in qwen35_n3 seed 1 (froze 2026-05-29T02:39Z after 8h50m). The
    env var alone is insufficient because the inheritable hook attaches even
    when it's false, so we also patch the callbacks to no-ops.

    Call this once, before any agent runs ``act()``. Returns ``True`` if the
    tracer class was patched, ``False`` if weave's langchain integration
    isn't importable.
    """
    os.environ.setdefault("WEAVE_TRACE_LANGCHAIN", "false")
    try:
        from weave.integrations.langchain.langchain import WeaveTracer
    except ImportError:
        return False
    WeaveTracer.on_chat_model_start = lambda self, *a, **kw: None
    WeaveTracer._on_chat_model_start = lambda self, *a, **kw: None
    return True
