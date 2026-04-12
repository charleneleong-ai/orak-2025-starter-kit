import grpc
import time
import logging
import io
import uuid
from PIL import Image

from evaluation_utils.protos import game_service_pb2 as pb2
from evaluation_utils.protos import game_service_pb2_grpc as pb2_grpc
import numpy as np

logger = logging.getLogger(__name__)

# Retry configuration for gRPC calls
MAX_RETRIES = 50
MAX_RETRY_TIME = 14000  # seconds
BACKOFF_BASE = 1.5
MAX_BACKOFF_INTERVAL = 10  # seconds
CALL_TIMEOUT = 300  # seconds

TRANSIENT_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
}


class GameEnv:
    """gRPC client for game environment."""

    def __init__(self, grpc_address: str):
        """
        Args:
            grpc_address: "host:port" (e.g., "localhost:33000")
        """
        self.address = grpc_address
        self.channel = grpc.insecure_channel(
            grpc_address,
            options=[
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ]
        )
        self.stub = pb2_grpc.GameEnvServiceStub(self.channel)
        self.session_token = None

    def _call_with_retry(self, method, request, timeout=CALL_TIMEOUT):
        """Execute gRPC call with exponential backoff retry."""
        start_time = time.time()
        attempt = 0
        last_exception = None

        # Get method name for logging (gRPC callables don't have __name__)
        method_name = getattr(method, '_method', 'unknown').decode('utf-8') if hasattr(method, '_method') else str(method)

        while attempt < MAX_RETRIES and (time.time() - start_time) < MAX_RETRY_TIME:
            attempt += 1
            try:
                return method(request, timeout=timeout)
            except grpc.RpcError as e:
                last_exception = e
                status_code = e.code()

                # ABORTED = lock contention, retry once after 100ms
                if status_code == grpc.StatusCode.ABORTED:
                    logger.warning(f"Lock contention on {method_name}, retrying once...")
                    time.sleep(0.1)
                    try:
                        return method(request, timeout=timeout)
                    except grpc.RpcError:
                        raise  # Give up after one retry

                # Transient errors: retry with backoff
                if status_code in TRANSIENT_CODES:
                    backoff = min(BACKOFF_BASE ** attempt, MAX_BACKOFF_INTERVAL)
                    logger.warning(
                        f"Transient error {status_code.name} on {method_name}, "
                        f"retry {attempt}/{MAX_RETRIES} in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue

                # Fatal errors: don't retry
                raise

        raise last_exception or RuntimeError("Max retries exceeded")

    def connect(self):
        """Establish session with server."""
        response = self._call_with_retry(
            self.stub.RegisterSession,
            pb2.Empty()
        )
        self.session_token = response.session_token
        logger.info(f"Connected with session: {self.session_token}")

    def get_game_config(self) -> dict:
        """Get game configuration."""
        response = self._call_with_retry(
            self.stub.GetGameConfig,
            pb2.SessionRequest(session_token=self.session_token)
        )
        return {
            "game_id": response.game_id,
            "max_steps": response.max_steps,
            "max_episodes": response.max_episodes,
            "current_episode": response.current_episode,
            "current_step": response.current_step,
        }

    def load_obs(self) -> dict:
        """Get current observation."""
        response = self._call_with_retry(
            self.stub.GetObservation,
            pb2.SessionRequest(session_token=self.session_token)
        )
        return self._parse_observation(response)

    def dispatch_final_action(self, action: str, request_id: str = None) -> dict:
        """Execute action and return result with new observation."""
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        if isinstance(action, (np.uint8, np.integer, np.floating)):
            action = str(convert_numpy(action))
        elif isinstance(action, np.ndarray):
            action = str(action.tolist())
        else:
            action = str(action)

        request = pb2.StepRequest(
            session_token=self.session_token,
            action=action,
            request_id=request_id or str(uuid.uuid4()),
        )
        response = self._call_with_retry(self.stub.Step, request)

        return {
            "score": response.score,
            "is_finished": response.is_finished,
            "avg_score": response.avg_score,
            "obs": self._parse_observation(response.observation),
        }

    def _parse_observation(self, obs_pb) -> dict:
        """Convert protobuf Observation to dict."""
        result = {
            "obs_str": obs_pb.obs_text,
            "game_info": dict(obs_pb.info),
        }

        # Decode image bytes to PIL Image
        if obs_pb.obs_image:
            result["obs_image"] = Image.open(io.BytesIO(obs_pb.obs_image))
            result["obs_image_str"] = ""  # Backwards compat (empty, use obs_image)
        else:
            result["obs_image"] = None
            result["obs_image_str"] = ""

        return result

    def close(self):
        """Close the channel."""
        self.channel.close()

    # Backwards compatibility methods (for existing runner.py)
    async def wait_for_ping(self):
        """Wait for server to be ready (compatibility wrapper)."""
        # Just call connect() which handles retry logic
        self.connect()
