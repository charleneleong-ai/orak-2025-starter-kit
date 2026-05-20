import logging
import os
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DATA_DIR = os.getenv("GAME_DATA_DIR", os.path.join(REPO_ROOT, "game_logs"))
GAME_RESULTS_PATH = os.path.join(GAME_DATA_DIR, "game_results.json")


def warn_if_tmp_data_dir(path: str, *, _warn=None) -> None:
    """Emit a loud warning when ``GAME_DATA_DIR`` is under ``/tmp/``.

    Stage R v5 iter5 was killed at launch by ENOSPC because the launcher
    set ``GAME_DATA_DIR=/tmp/orak-...`` and 4 iters × 600 steps of per-step
    checkpoint pickles saturated ``/tmp`` (much smaller than ``/workspace``
    on this box). Mitigations are twofold: rolling-window cleanup
    (``CheckpointManager(keep_last_n=...)``) plus this warning so anyone
    running an overnight sweep notices the misconfiguration before it
    bites them.

    ``_warn`` is injected for tests; production code uses loguru.
    """
    if not str(path).startswith("/tmp/"):
        return
    msg = (
        f"GAME_DATA_DIR={path!r} is under /tmp/. Long-running sweeps "
        "will accumulate per-step checkpoint pickles and may hit ENOSPC. "
        "Set GAME_DATA_DIR=/workspace/<sweep-name>/ instead, or rely on "
        "the in-repo game_logs/ default."
    )
    if _warn is not None:
        _warn(msg)
        return
    from loguru import logger as _loguru_logger  # local — keeps commons cheap

    _loguru_logger.warning(msg)


warn_if_tmp_data_dir(GAME_DATA_DIR)

# Get a logger for the module where the client is used
logger = logging.getLogger(__name__)

# Retry/backoff configuration for gRPC client interactions. Exposed as env vars so
# users can tweak them without touching the code.
GRPC_MAX_RETRY_TRIES = int(os.getenv("GRPC_MAX_RETRY_TRIES", "50"))
GRPC_MAX_RETRY_TIME = float(os.getenv("GRPC_MAX_RETRY_TIME", "14000"))
GRPC_BACKOFF_BASE = float(os.getenv("GRPC_BACKOFF_BASE", "1.5"))
GRPC_BACKOFF_MAX_INTERVAL = float(os.getenv("GRPC_BACKOFF_MAX_INTERVAL", "10"))
GRPC_CALL_TIMEOUT_SECONDS = float(os.getenv("GRPC_CALL_TIMEOUT_SECONDS", "300"))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    os.makedirs(GAME_DATA_DIR, exist_ok=True)
    log_file = os.path.join(GAME_DATA_DIR, "evaluation.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="[%X]",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


API_TOKEN = os.getenv("AICROWD_API_TOKEN", "")

BASE_URL = os.getenv("AICROWD_API_BASE_URL", "https://orak-game-api.aicrowd.com")

BASE_PORT = int(os.getenv("BASE_PORT", random.randint(10000, 50000)))
GAME_SERVER_PORTS = {
    "twenty_fourty_eight": BASE_PORT,
    "super_mario": BASE_PORT + 1,
    "pokemon_red": BASE_PORT + 2,
    "star_craft": BASE_PORT + 3,
}
MAX_EPISODES = 3
MAX_STEPS = {
    "twenty_fourty_eight": 1000,
    "super_mario": 100,
    "pokemon_red": 200,
    "star_craft": 1000,
}
