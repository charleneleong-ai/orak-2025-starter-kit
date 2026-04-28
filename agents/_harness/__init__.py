"""Shared harness primitives.

* ``prompt_caching`` — cross-backend ``cached_tokens`` extraction (vLLM
  auto-caches by prefix; this module just measures).
* ``retry_utils`` — jittered backoff + classified-error retries.
* ``trajectory`` — ShareGPT-shaped per-episode trajectory writer.

Each module is independent; agents opt in à la carte.
"""
from .prompt_caching import extract_cache_stats
from .retry_utils import (
    ClassifiedError,
    ErrorClass,
    classify,
    jittered_backoff,
    with_retries,
)
from .structured_invoke import structured_invoke_with_usage
from .trajectory import (
    StepRecord,
    TrajectoryWriter,
    convert_scratchpad_to_think,
    has_incomplete_scratchpad,
)

__all__ = [
    "extract_cache_stats",
    "ClassifiedError",
    "ErrorClass",
    "classify",
    "jittered_backoff",
    "with_retries",
    "structured_invoke_with_usage",
    "StepRecord",
    "TrajectoryWriter",
    "convert_scratchpad_to_think",
    "has_incomplete_scratchpad",
]
