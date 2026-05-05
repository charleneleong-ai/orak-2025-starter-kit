"""Shared harness primitives.

The trajectory writer, retry/backoff helpers, and prompt-cache stat
extractor were lifted into the shared ``autoresearch`` package — re-exported
here so the agent-side import path stays stable. Update :mod:`autoresearch`
to evolve the shared API; ``structured_invoke_with_usage`` remains local
because it's langchain-coupled and not yet on autoresearch's surface.
"""

from autoresearch import (
    ClassifiedError,
    ErrorClass,
    StepRecord,
    TrajectoryWriter,
    classify,
    convert_scratchpad_to_think,
    extract_cache_stats,
    has_incomplete_scratchpad,
    jittered_backoff,
    with_retries,
)

from .structured_invoke import structured_invoke_with_usage

__all__ = [
    "ClassifiedError",
    "ErrorClass",
    "StepRecord",
    "TrajectoryWriter",
    "classify",
    "convert_scratchpad_to_think",
    "extract_cache_stats",
    "has_incomplete_scratchpad",
    "jittered_backoff",
    "structured_invoke_with_usage",
    "with_retries",
]
