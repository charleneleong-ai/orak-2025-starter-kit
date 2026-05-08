"""
gRPC Server Utilities: Session Management and Concurrency Control

This module provides battle-tested utilities for managing game server sessions
and ensuring fail-fast concurrency control in a single-client gRPC environment.

Key Design Principles:
- Fail-fast: Lock contention immediately rejects requests with ABORTED status
- Session isolation: One active session at a time with timeout-based expiry
- Idempotency: Tracks last request_id to handle network retry scenarios
- Observable: Includes thread IDs and clear error messages for debugging
"""

import threading
import time
import uuid
from collections.abc import Callable
from functools import wraps

import grpc

# Module-level constants
SESSION_TIMEOUT_SECONDS = 120  # 2 minutes


class SessionManager:
    """
    Thread-safe session manager for single-client game server.

    Manages a single session token with timeout-based expiry. Only one client
    can hold a valid session at a time. Sessions expire after 2 minutes of
    inactivity to allow recovery from crashed clients.

    Thread Safety:
        All operations are atomic. Multiple threads can safely call methods
        without external synchronization.

    Example:
        >>> session = SessionManager()
        >>> token = session.register()
        >>> assert session.validate(token) == True
        >>> time.sleep(121)
        >>> assert session.validate(token) == False  # Expired
    """

    def __init__(self, timeout_seconds: int = SESSION_TIMEOUT_SECONDS):
        """
        Initialize session manager.

        Args:
            timeout_seconds: Session expiry timeout (default: 120 seconds)
        """
        self._token: str | None = None
        self._last_activity: float = 0.0
        self._timeout: int = timeout_seconds
        self._lock = threading.Lock()

    def register(self) -> str:
        """
        Register a new session and return a unique token.

        Generates a UUID v4 token, resets activity timer, and returns the token.
        This method does NOT check if an existing session is active - that
        validation should be done at the gRPC servicer level.

        Returns:
            str: UUID v4 session token (e.g., "a1b2c3d4-1234-5678-90ab-cdef12345678")

        Thread Safety:
            Atomic operation protected by internal lock.
        """
        with self._lock:
            self._token = str(uuid.uuid4())
            self._last_activity = time.time()
            return self._token

    def validate(self, token: str) -> bool:
        """
        Validate session token and check expiry.

        Returns True only if:
        1. A session has been registered (_token is not None)
        2. The provided token matches the registered token
        3. The session has not expired (activity within timeout window)

        Args:
            token: Session token to validate

        Returns:
            bool: True if token is valid and not expired, False otherwise

        Thread Safety:
            Atomic operation protected by internal lock.
        """
        with self._lock:
            if self._token is None:
                return False
            if token != self._token:
                return False
            if self.is_expired():
                return False
            return True

    def is_expired(self) -> bool:
        """
        Check if current session has expired.

        A session expires when no activity has occurred within the timeout
        window. Uses time.time() for monotonic wall-clock comparison.

        Returns:
            bool: True if expired or no session exists, False otherwise

        Thread Safety:
            Should be called within _lock for atomic checks, but safe to call
            independently for informational purposes.

        Note:
            This method does NOT acquire a lock - caller must hold _lock for
            atomic validate-and-operate patterns.
        """
        if self._token is None:
            return True
        return (time.time() - self._last_activity) > self._timeout

    def touch(self) -> None:
        """
        Update last activity timestamp to current time.

        Call this method to extend the session's lifetime. Should be called
        on every successful RPC that validates the session.

        Thread Safety:
            Atomic operation protected by internal lock.
        """
        with self._lock:
            self._last_activity = time.time()


class IdempotencyTracker:
    """
    Simple idempotency tracker for single-client game server.

    Tracks the most recent request_id to detect duplicate requests from
    network retries. In a single-client environment, we only need to remember
    the last request since clients process sequentially.

    Design Note:
        This is NOT a full idempotency cache (which would store results).
        It only detects duplicates. The caller is responsible for:
        1. Caching the previous response
        2. Returning the cached response for duplicate requests

    Thread Safety:
        All operations are atomic. Multiple threads can safely call methods
        without external synchronization.

    Example:
        >>> tracker = IdempotencyTracker()
        >>> request_id = "req-123"
        >>> assert tracker.is_duplicate(request_id) == False
        >>> tracker.record(request_id)
        >>> assert tracker.is_duplicate(request_id) == True
    """

    def __init__(self):
        """Initialize idempotency tracker with no stored request."""
        self._last_request_id: str | None = None
        self._lock = threading.Lock()

    def is_duplicate(self, request_id: str) -> bool:
        """
        Check if request_id matches the last recorded request.

        Args:
            request_id: Request identifier to check

        Returns:
            bool: True if request_id matches last recorded request, False otherwise

        Thread Safety:
            Atomic operation protected by internal lock.
        """
        with self._lock:
            return request_id == self._last_request_id

    def record(self, request_id: str) -> None:
        """
        Record request_id as the last processed request.

        Args:
            request_id: Request identifier to store

        Thread Safety:
            Atomic operation protected by internal lock.
        """
        with self._lock:
            self._last_request_id = request_id


def require_session(session_manager: SessionManager):
    """
    Decorator factory for validating session tokens on gRPC methods.

    Creates a decorator that extracts the session_token from the request,
    validates it using the provided SessionManager, and aborts with
    UNAUTHENTICATED status if validation fails.

    Args:
        session_manager: SessionManager instance to use for validation

    Returns:
        Callable: Decorator function

    Usage:
        >>> session_mgr = SessionManager()
        >>>
        >>> @require_session(session_mgr)
        >>> def GetObservation(self, request, context):
        >>>     # request.session_token already validated
        >>>     return perform_operation()

    Error Handling:
        Aborts with grpc.StatusCode.UNAUTHENTICATED and descriptive message:
        - "No active session" if no session registered
        - "Invalid session token" if token doesn't match
        - "Session expired" if session timed out

    Thread Safety:
        Session validation is atomic via SessionManager's internal locking.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, request, context, *args, **kwargs):
            # Extract token from request
            token = getattr(request, "session_token", None)
            if token is None:
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED, "Request missing session_token field"
                )

            # Validate session
            if not session_manager.validate(token):
                # Provide specific error message
                if session_manager._token is None:
                    context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "No active session. Call RegisterSession first.",
                    )
                elif session_manager.is_expired():
                    context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        f"Session expired (timeout: {session_manager._timeout}s)",
                    )
                else:
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid session token")

            # Touch session to extend lifetime
            session_manager.touch()

            # Proceed with operation
            return func(self, request, context, *args, **kwargs)

        return wrapper

    return decorator


def require_lock(lock: threading.Lock, operation_name: str = "operation"):
    """
    Decorator factory for fail-fast lock acquisition on gRPC methods.

    Creates a decorator that attempts to acquire the lock with blocking=False.
    If the lock is already held, immediately aborts with ABORTED status
    instead of queuing the request.

    This implements the critical "fail-fast" pattern for single-client game
    servers: if a Step is in progress, reject concurrent requests immediately
    rather than queueing them (which would break determinism).

    Args:
        lock: threading.Lock instance to acquire
        operation_name: Human-readable name for error messages (default: "operation")

    Returns:
        Callable: Decorator function

    Usage:
        >>> action_lock = threading.Lock()
        >>>
        >>> @require_lock(action_lock, "Step")
        >>> def Step(self, request, context):
        >>>     # Lock held for entire method execution
        >>>     return perform_step()

    Error Handling:
        Aborts with grpc.StatusCode.ABORTED if lock cannot be acquired:
        - Status: ABORTED (client should retry)
        - Message: "Busy: {operation_name} already in progress (thread_id: ...)"

    Thread Safety:
        Uses non-blocking lock acquisition (blocking=False). Always releases
        lock in finally block, even if method raises exception.

    Lock Ordering:
        If using multiple locks, always acquire in consistent global order
        to prevent deadlocks. Example: session_lock before action_lock.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, request, context, *args, **kwargs):
            # Try to acquire lock with fail-fast behavior
            acquired = lock.acquire(blocking=False)

            if not acquired:
                # Lock is held by another thread - reject immediately
                current_thread = threading.current_thread()
                context.abort(
                    grpc.StatusCode.ABORTED,
                    f"Busy: {operation_name} already in progress "
                    f"(attempted by thread: {current_thread.name})",
                )

            try:
                # Lock acquired - proceed with operation
                return func(self, request, context, *args, **kwargs)
            finally:
                # Always release lock, even if operation raised exception
                lock.release()

        return wrapper

    return decorator


def validate_session_and_acquire_lock(
    session_manager: SessionManager, lock: threading.Lock, operation_name: str = "operation"
):
    """
    Combined decorator for session validation + fail-fast lock acquisition.

    This is a convenience decorator that combines @require_session and
    @require_lock in the correct order. Use this for methods that need both
    session authentication and exclusive access to game state.

    Order of operations:
    1. Validate session token (abort if invalid/expired)
    2. Touch session to extend lifetime
    3. Attempt lock acquisition (abort if already held)
    4. Execute operation
    5. Release lock in finally block

    Args:
        session_manager: SessionManager instance for validation
        lock: threading.Lock instance to acquire
        operation_name: Human-readable name for error messages

    Returns:
        Callable: Decorator function

    Usage:
        >>> session_mgr = SessionManager()
        >>> action_lock = threading.Lock()
        >>>
        >>> @validate_session_and_acquire_lock(session_mgr, action_lock, "Step")
        >>> def Step(self, request, context):
        >>>     # Session validated AND lock held
        >>>     return perform_step()

    Error Handling:
        - Session errors: UNAUTHENTICATED status (see require_session)
        - Lock errors: ABORTED status (see require_lock)

    Thread Safety:
        Session validation is atomic. Lock acquisition is fail-fast.
        Lock is always released in finally block.
    """

    def decorator(func: Callable) -> Callable:
        # Apply decorators in order: session first, then lock
        @require_session(session_manager)
        @require_lock(lock, operation_name)
        @wraps(func)
        def wrapper(self, request, context, *args, **kwargs):
            return func(self, request, context, *args, **kwargs)

        return wrapper

    return decorator
