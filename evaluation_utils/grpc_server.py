"""
gRPC Server Implementation for Game Environment

This module implements the gRPC servicer for the Orak RL Evaluation Framework.
It provides a contract-first implementation using Protocol Buffers for type safety
and HTTP/2 for efficient communication.

Key Design Principles:
- Fail-fast concurrency: Lock contention immediately rejects requests with ABORTED status
- Session isolation: One active session at a time with timeout-based expiry
- Sequential processing: max_workers=1 enforces deterministic state transitions
- Binary efficiency: Images transmitted as raw JPEG bytes (not base64)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    gRPC Client (runner)                  │
    │  1. RegisterSession() → session_token                    │
    │  2. GetGameConfig(token) → game metadata                 │
    │  3. GetObservation(token) → obs + image bytes            │
    │  4. Step(token, action) → score + is_finished + new_obs  │
    └─────────────────────────────────────────────────────────┘
                            │
                      gRPC (HTTP/2)
                            │
    ┌─────────────────────────────────────────────────────────┐
    │              GameEnvServiceServicer (this file)          │
    │  - Session validation (2-minute timeout)                 │
    │  - Fail-fast locking (non-blocking acquire)              │
    │  - Idempotency tracking (last request_id)                │
    │  - Image encoding (PIL → JPEG bytes)                     │
    └─────────────────────────────────────────────────────────┘
                            │
    ┌─────────────────────────────────────────────────────────┐
    │                   Game Server (base_game_logic)          │
    │  - load_current_obs() → (str, bytes, dict)               │
    │  - dispatch_action_and_get_score(action) → (score, done) │
    │  - get_game_config() → dict                              │
    │  - reset_env() → obs                                     │
    └─────────────────────────────────────────────────────────┘
"""

import grpc
import threading
import time
import uuid
import logging
from concurrent import futures

from evaluation_utils.protos import game_service_pb2 as pb2
from evaluation_utils.protos import game_service_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)

# Session timeout: 2 minutes of inactivity
SESSION_TIMEOUT_SECONDS = 600


class GameEnvServiceServicer(pb2_grpc.GameEnvServiceServicer):
    """
    gRPC servicer for game environment.

    This servicer wraps a game_server instance and exposes it via gRPC RPCs.
    It enforces single-client access with session tokens, fail-fast concurrency
    control via non-blocking locks, and idempotency tracking for network retries.

    Thread Safety:
        - Session validation is atomic (uses internal checks)
        - Action execution uses fail-fast locking (non-blocking acquire)
        - Only one action can execute at a time (max_workers=1 + explicit lock)

    Session Management:
        - RegisterSession: Returns UUID token, rejects if another client connected
        - All other RPCs: Require valid session_token in request
        - Sessions expire after 2 minutes of inactivity

    Error Handling:
        - UNAUTHENTICATED: Missing/invalid/expired session token
        - PERMISSION_DENIED: Server occupied by another client
        - ABORTED: Lock contention (another operation in progress)
        - INTERNAL: Unexpected server error during step execution
    """

    def __init__(self, game_server):
        """
        Initialize the gRPC servicer.

        Args:
            game_server: Game server instance with the following interface:
                - load_current_obs() -> (obs_str: str, obs_image_bytes: bytes, game_info: dict)
                    Returns current observation as text, raw image bytes (JPEG), and game info dict
                - dispatch_action_and_get_score(action: str) -> (score: float, is_finished: bool, max_episodes_reached: bool)
                    Executes action and returns score, episode termination flag, and max episodes flag
                - get_game_config() -> dict
                    Returns game configuration dict with keys: game_id, max_steps, max_episodes, current_episode, current_step
                - reset_env() -> obs
                    Resets environment to initial state (called internally by game_server)

                Internal state accessed for avg_score calculation:
                - _total_score: Cumulative score across all episodes
                - _episodes: Number of completed episodes
        """
        self._game = game_server

        # Fail-fast lock for action execution
        # Acquired with blocking=False to immediately reject concurrent requests
        self._action_lock = threading.Lock()

        # Session management state
        self._session_token = None
        self._last_activity = time.time()

        # Idempotency tracking
        self._last_request_id = None

        # Progress tracking
        # Tracks how many Step() calls have been successfully applied in the *current* episode.
        # This is used to populate GameConfig.current_step for reconnect/resume flows.
        # Note: The underlying GameLogic tracks episodes, but does not track per-episode steps.
        self._current_step_in_episode = 0

        logger.info("GameEnvServiceServicer initialized")

    def _is_session_expired(self) -> bool:
        """
        Check if current session has exceeded timeout window.

        Returns:
            bool: True if session expired or no session exists
        """
        if self._session_token is None:
            return True
        return time.time() - self._last_activity > SESSION_TIMEOUT_SECONDS

    def _validate_session(self, token: str, context) -> bool:
        """
        Validate session token and check expiry.

        This method performs atomic validation and updates the last activity
        timestamp on success to extend the session lifetime.

        Args:
            token: Session token from client request
            context: gRPC context for aborting on validation failure

        Returns:
            bool: True if valid (never returns False - aborts on failure)

        Side Effects:
            - Aborts RPC with UNAUTHENTICATED status on validation failure
            - Updates _last_activity timestamp on success
        """
        # No session registered yet
        if self._session_token is None:
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "No active session. Call RegisterSession first."
            )
            return False

        # Token mismatch
        if token != self._session_token:
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "Invalid session token"
            )
            return False

        # Session expired
        if self._is_session_expired():
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                f"Session expired (timeout: {SESSION_TIMEOUT_SECONDS}s)"
            )
            return False

        # Valid session - extend lifetime
        self._last_activity = time.time()
        return True

    def RegisterSession(self, request, context):
        """
        Register a new session and return session token.

        This is the handshake RPC that must be called before any other operations.
        Only one client can hold an active session at a time. If a session already
        exists and has not expired, this RPC will abort with PERMISSION_DENIED.

        Args:
            request: Empty protobuf message
            context: gRPC context

        Returns:
            SessionResponse: Contains UUID v4 session token

        Error Codes:
            PERMISSION_DENIED: Another client has an active session
        """
        # Reject if session exists and not expired
        if self._session_token and not self._is_session_expired():
            logger.warning(
                "RegisterSession rejected: server occupied by another client"
            )
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Server occupied by another client. Wait for session to expire or "
                f"try again in {SESSION_TIMEOUT_SECONDS}s."
            )

        # Generate new session
        self._session_token = str(uuid.uuid4())
        self._last_activity = time.time()
        self._last_request_id = None  # Reset idempotency tracking

        logger.info(f"New session registered: {self._session_token}")
        return pb2.SessionResponse(session_token=self._session_token)

    def GetGameConfig(self, request, context):
        """
        Get game configuration and current progress.

        Returns metadata about the game including step/episode limits and
        current progress counters.

        Args:
            request: SessionRequest with session_token
            context: gRPC context

        Returns:
            GameConfig: Game metadata including game_id, max_steps, max_episodes,
                       current_episode, and current_step

        Error Codes:
            UNAUTHENTICATED: Invalid or expired session token
        """
        if not self._validate_session(request.session_token, context):
            return pb2.GameConfig()

        config = self._game.get_game_config()
        logger.debug(f"GetGameConfig: {config}")

        return pb2.GameConfig(
            game_id=config["game_id"],
            max_steps=config["max_steps"],
            max_episodes=config["max_episodes"],
            current_episode=config.get("current_episode", 0),
            # Prefer the game implementation's value if provided, otherwise use the
            # servicer-maintained per-episode step counter.
            current_step=config.get("current_step", self._current_step_in_episode),
        )

    def GetObservation(self, request, context):
        """
        Get current observation without executing an action.

        This RPC returns the current game state including text description,
        rendered image (raw JPEG bytes), and game-specific info dict.

        Uses fail-fast locking: if another operation is in progress, immediately
        aborts with ABORTED status rather than queueing the request.

        Args:
            request: SessionRequest with session_token
            context: gRPC context

        Returns:
            Observation: Contains obs_text (str), obs_image (bytes), and info (map<string, string>)

        Error Codes:
            UNAUTHENTICATED: Invalid or expired session token
            ABORTED: Another operation in progress (lock contention)
        """
        if not self._validate_session(request.session_token, context):
            return pb2.Observation()

        # Try-lock for fail-fast behavior
        acquired = self._action_lock.acquire(blocking=False)
        if not acquired:
            logger.warning("GetObservation rejected: lock held by another operation")
            context.abort(
                grpc.StatusCode.ABORTED,
                "Busy: Another operation in progress. Retry after brief delay."
            )

        try:
            obs_str, obs_image_bytes, game_info = self._game.load_current_obs()
            logger.debug(f"GetObservation: obs_str length={len(obs_str)}, "
                        f"image size={len(obs_image_bytes) if obs_image_bytes else 0} bytes")

            return pb2.Observation(
                obs_text=obs_str,
                obs_image=obs_image_bytes,  # Raw JPEG bytes, not base64
                info={k: str(v) for k, v in game_info.items()},
            )
        finally:
            self._action_lock.release()

    def Step(self, request, context):
        """
        Execute action and return result with new observation.

        This is the core game loop RPC. It:
        1. Validates session and checks idempotency
        2. Acquires action lock (fail-fast)
        3. Executes action via game_server.dispatch_action_and_get_score()
        4. Loads new observation after action execution
        5. Returns comprehensive result including score, termination flag, and new observation

        Idempotency:
            If request_id matches the last processed request, the action is NOT executed
            again (though current implementation just logs this case).

        Args:
            request: StepRequest with session_token, action (str), and optional request_id
            context: gRPC context

        Returns:
            StepResult: Contains score (float), is_finished (bool), avg_score (float),
                       and observation (new state after action)

        Error Codes:
            UNAUTHENTICATED: Invalid or expired session token
            ABORTED: Another action in progress (lock contention)
            INTERNAL: Exception during action execution
        """
        if not self._validate_session(request.session_token, context):
            return pb2.StepResult()

        # Idempotency check
        if request.request_id and request.request_id == self._last_request_id:
            logger.warning(
                f"Duplicate request detected: {request.request_id}. "
                "Should return cached result (not implemented)."
            )
            # TODO: Return cached result from previous execution
            # For now, we'll just continue and re-execute (safe for deterministic games)

        # Try-lock for fail-fast behavior
        acquired = self._action_lock.acquire(blocking=False)
        if not acquired:
            logger.warning("Step rejected: lock held by another action")
            context.abort(
                grpc.StatusCode.ABORTED,
                "Busy: Another action in progress. Retry after brief delay."
            )

        try:
            # Record request_id for idempotency tracking
            self._last_request_id = request.request_id

            logger.info(f"Executing action: {request.action}")

            # Execute action
            score, is_finished, _ = self._game.dispatch_action_and_get_score(
                request.action
            )

            logger.info(f"Action result: score={score}, is_finished={is_finished}")

            # Update progress counters *after* applying the step.
            # Semantics: current_step == number of completed steps in the current episode.
            # If the episode finished, GameLogic will have reset the env for the next episode,
            # so the "current step" for the next episode should be 0.
            if is_finished:
                self._current_step_in_episode = 0
            else:
                self._current_step_in_episode += 1

            # Get new observation after step
            obs_str, obs_image_bytes, game_info = self._game.load_current_obs()

            # Calculate average score
            # Division by max(1, episodes) prevents division by zero
            avg_score = self._game._total_score / max(self._game._episodes, 1)

            return pb2.StepResult(
                score=score,
                is_finished=is_finished,
                avg_score=avg_score,
                observation=pb2.Observation(
                    obs_text=obs_str,
                    obs_image=obs_image_bytes,
                    info={k: str(v) for k, v in game_info.items()},
                ),
            )
        except Exception as e:
            logger.error(f"Step failed: {e}", exc_info=True)
            context.set_details(f"Step failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise
        finally:
            self._action_lock.release()


def serve(game_server, port: int):
    """
    Start gRPC server with single-threaded executor.

    This function creates and starts a gRPC server with:
    - Single worker thread (max_workers=1) for sequential request processing
    - 50MB message size limits for large images
    - Insecure channel (no TLS) for internal tool usage

    The single-worker executor enforces that only one RPC can execute at a time,
    which combined with fail-fast locking ensures deterministic game state transitions.

    Args:
        game_server: Game server instance to wrap (see GameEnvServiceServicer.__init__)
        port: Port number to listen on (e.g., 33000)

    Blocks:
        This function blocks indefinitely, waiting for termination signal.

    Example:
        >>> from evaluation_utils.mcp_game_servers.base_game_logic import GameLogic
        >>> game = GameLogic("config.yaml")
        >>> serve(game, port=33000)  # Blocks until terminated
    """
    # Create server with single-threaded executor
    # max_workers=1 ensures sequential processing (one RPC at a time)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=[
            # 50MB message size limits for large game screenshots
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.max_send_message_length', 50 * 1024 * 1024),

            # Keep-alive settings to detect dead clients
            ('grpc.keepalive_time_ms', 30000),  # Ping every 30 seconds
            ('grpc.keepalive_timeout_ms', 10000),  # Wait 10s for pong
            ('grpc.keepalive_permit_without_calls', True),  # Ping even without RPCs
        ]
    )

    # Register servicer
    pb2_grpc.add_GameEnvServiceServicer_to_server(
        GameEnvServiceServicer(game_server), server
    )

    # Bind to port (all interfaces)
    server.add_insecure_port(f'[::]:{port}')

    # Start server
    server.start()
    logger.info(f"gRPC server started on port {port}")
    print(f"gRPC server started on port {port}")

    # Block until termination
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        server.stop(grace=5)
