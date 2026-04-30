"""Retry helpers — jittered backoff + lightweight error classification.

Why jitter: under autoresearch parallelism multiple sweep workers can hit the
same vLLM server and retry in lock-step on a 5xx. Decorrelated jitter prevents
thundering herd.

Why a classifier: pokemon's existing agent does ``try/except → return "pass"``
which silently turns transient errors into agent decisions, polluting the
trajectory. We want transient → retry, terminal → raise.
"""
from __future__ import annotations

import enum
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from loguru import logger

_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Decorrelated exponential backoff. ``attempt`` is 1-based."""
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2**exponent), max_delay)

    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)
    return delay + jitter


class ErrorClass(enum.Enum):
    TRANSIENT = "transient"  # retry
    TERMINAL = "terminal"    # raise immediately
    UNKNOWN = "unknown"      # retry but log loudly


@dataclass
class ClassifiedError:
    cls: ErrorClass
    status: Optional[int]
    message: str
    original: BaseException


def classify(error: BaseException) -> ClassifiedError:
    """Classify by status code (if available) then exception message.

    Liberal on the transient side — we'd rather retry once and waste a
    second than silently swallow a recoverable error.
    """
    status = _extract_status(error)
    msg = str(error).lower()

    if status is not None:
        if status in (429,) or 500 <= status < 600:
            return ClassifiedError(ErrorClass.TRANSIENT, status, str(error), error)
        if status in (401, 403):
            return ClassifiedError(ErrorClass.TERMINAL, status, str(error), error)
        if 400 <= status < 500:
            # 400/404/etc — usually our fault (bad schema, bad URL) — terminal
            return ClassifiedError(ErrorClass.TERMINAL, status, str(error), error)

    transient_markers = (
        "rate limit", "rate_limit", "timed out", "timeout", "connection",
        "temporarily unavailable", "overloaded", "503", "502", "504",
        "broken pipe", "reset by peer",
    )
    if any(m in msg for m in transient_markers):
        return ClassifiedError(ErrorClass.TRANSIENT, status, str(error), error)

    terminal_markers = (
        "invalid api key", "unauthorized", "permission denied",
        "model not found", "context length",
    )
    if any(m in msg for m in terminal_markers):
        return ClassifiedError(ErrorClass.TERMINAL, status, str(error), error)

    return ClassifiedError(ErrorClass.UNKNOWN, status, str(error), error)


def _extract_status(error: BaseException) -> Optional[int]:
    for attr in ("status_code", "status", "http_status", "code"):
        v = getattr(error, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(error, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v
    return None


def with_retries(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    label: str = "llm_call",
) -> Any:
    """Run ``fn`` with classified retries + jittered backoff.

    Raises the original exception if all attempts exhaust or the error is
    terminal. Caller sees a ``ClassifiedError`` attached as ``__classified__``
    on the raised exception (cheap introspection without changing the type).
    """
    last: Optional[ClassifiedError] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except BaseException as e:
            ce = classify(e)
            last = ce
            if ce.cls is ErrorClass.TERMINAL:
                logger.error(f"[{label}] terminal error: {ce.status} {ce.message[:200]}")
                e.__classified__ = ce  # type: ignore[attr-defined]
                raise
            if attempt >= max_attempts:
                logger.error(f"[{label}] giving up after {attempt} attempts: {ce.message[:200]}")
                e.__classified__ = ce  # type: ignore[attr-defined]
                raise
            delay = jittered_backoff(attempt, base_delay=base_delay, max_delay=max_delay)
            logger.warning(
                f"[{label}] {ce.cls.value} (status={ce.status}) — "
                f"retry {attempt}/{max_attempts - 1} in {delay:.1f}s: {ce.message[:120]}"
            )
            time.sleep(delay)
    # Unreachable — raise above always exits
    assert last is not None
    raise last.original
