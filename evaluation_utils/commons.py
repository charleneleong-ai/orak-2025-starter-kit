import logging
import os
import random
from fastmcp.client.logging import LogMessage


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DATA_DIR = os.getenv("GAME_DATA_DIR", os.path.join(REPO_ROOT, "game_logs"))
GAME_RESULTS_PATH = os.path.join(GAME_DATA_DIR, "game_results.json")

# Get a logger for the module where the client is used
logger = logging.getLogger(__name__)

# This mapping is useful for converting MCP level strings to Python's levels
LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()

# Retry/backoff configuration for MCP client interactions. Exposed as env vars so
# users can tweak them without touching the code.
MCP_MAX_RETRY_TRIES = int(os.getenv("MCP_MAX_RETRY_TRIES", "50"))
MCP_MAX_RETRY_TIME = float(os.getenv("MCP_MAX_RETRY_TIME", "14000"))
MCP_BACKOFF_BASE = float(os.getenv("MCP_BACKOFF_BASE", "1.5"))
MCP_BACKOFF_MAX_INTERVAL = float(os.getenv("MCP_BACKOFF_MAX_INTERVAL", "10"))
MCP_CALL_TIMEOUT_SECONDS = float(os.getenv("MCP_CALL_TIMEOUT_SECONDS", "300"))

async def log_handler(message: LogMessage):
    """
    Handles incoming logs from the MCP server and forwards them
    to the standard Python logging system.
    """
    msg = message.data.get('msg')
    extra = message.data.get('extra')

    # Convert the MCP log level to a Python log level
    level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)

    # Log the message using the standard logging library
    logger.log(level, msg, extra=extra)


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


API_TOKEN = os.getenv("AICROWD_API_TOKEN")

BASE_URL = os.getenv("AICROWD_API_BASE_URL", "https://orak-game-api.aicrowd.com")

BASE_PORT = int(os.getenv("BASE_PORT", random.randint(10000, 50000)))
GAME_SERVER_PORTS = {
    "twenty_fourty_eight": BASE_PORT,
    "super_mario": BASE_PORT + 1,
    "pokemon_red": BASE_PORT + 2,
    "star_craft": BASE_PORT + 3,
    "street_fighter": BASE_PORT + 4,
}
MAX_EPISODES = 3
MAX_STEPS = {
    "twenty_fourty_eight": 1000,
    "street_fighter": 1000,
    "super_mario": 100,
    "pokemon_red": 200,
    "star_craft": 1000,
}