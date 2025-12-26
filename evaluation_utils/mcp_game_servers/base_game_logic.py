"""
GameLogic class - Pure game logic for the gRPC game server.

This class handles:
- Environment creation and configuration
- Observation loading and image processing
- Action dispatching and score tracking
- Episode management and reset logic
- Latched final observation for completed games

Key characteristics:
- No protocol dependencies (gRPC, MCP, etc.)
- Synchronous operations (no asyncio)
- Returns raw JPEG bytes for images
- Clean separation of game logic from server protocol
"""

import json
import os
import logging
from datetime import datetime
from typing import Tuple
from io import BytesIO
import omegaconf
from PIL import Image
import time

from mcp_game_servers.utils.module_creator import EnvCreator
from evaluation_utils.commons import GAME_DATA_DIR, MAX_STEPS, MAX_EPISODES, setup_logging, GAME_RESULTS_PATH

logger = logging.getLogger(__name__)

GAME_ID = os.getenv("GAME_ID", "")


def set_log_path(cfg, expand_log_path: bool = True):
    """Configure logging path based on environment and timestamp."""
    if expand_log_path:
        log_path = os.path.join(
            cfg.log_path,
            cfg.env_name,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        cfg.env.log_path = log_path
        os.makedirs(log_path, exist_ok=True)

    setup_logging()


class GameLogic:
    """
    Core game logic implementation - manages environment, observations, and scoring.

    This class is thread-safe when used from a single thread at a time. For multi-threaded
    access, the caller must provide external synchronization (e.g., locks in the gRPC server).
    """

    def __init__(self, config_path: str, expand_log_path: bool = True):
        """
        Initialize game logic with configuration.

        Args:
            config_path: Path to YAML configuration file
            expand_log_path: Whether to create timestamped log directories
        """
        logger.info(f"Initializing GameLogic with config: {config_path}")

        # Load configuration
        self.cfg = omegaconf.OmegaConf.load(config_path)
        set_log_path(self.cfg, expand_log_path)

        # Create environment
        self.env = EnvCreator(self.cfg).create()
        logger.info("Environment created successfully")

        # Observation state
        self.obs = None
        self.first_loading = True

        # Scoring and episode tracking
        self._score = 0
        self._total_score = 0
        self._episodes = 0
        self._max_steps = self.cfg.env.get("max_steps", MAX_STEPS.get(GAME_ID, 1000))
        self._current_step = 0
        self._start_time = time.time()
        self._end_time = None

        # Latched final observation logic
        # When all episodes are finished, we latch the final observation
        # so subsequent load_current_obs() calls return the same final state
        self._all_episodes_finished = False
        self._latched_final_obs = None  # Stores (obs_str, obs_image_bytes, game_info)
        self._latched_final_result = None  # Stores (score, is_finished, max_episodes_reached)

    def image_to_bytes(self, image: Image.Image) -> bytes:
        """
        Convert PIL Image to raw JPEG bytes.

        Args:
            image: PIL Image object (may be RGBA or RGB)

        Returns:
            Raw JPEG bytes
        """
        buffer = BytesIO()
        # Convert RGBA to RGB if necessary (JPEG doesn't support alpha channel)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG')
        return buffer.getvalue()

    def load_current_obs(self) -> Tuple[str, bytes, dict]:
        """
        Load the current observation from the environment.

        Returns:
            Tuple of (obs_str, obs_image_bytes, game_info)
            - obs_str: Text representation of observation
            - obs_image_bytes: Raw JPEG bytes (empty bytes if no image)
            - game_info: Dictionary with current game state information

        Note: If all episodes are finished, returns the latched final observation.
        """
        logger.info("Loading current observation")

        # Return latched final observation if game is complete
        if self._all_episodes_finished and self._latched_final_obs is not None:
            logger.info("Game finished. Returning latched final observation.")
            return self._latched_final_obs

        # Load initial observation on first call
        if self.first_loading:
            self.obs = self.env.initial_obs()
            self.first_loading = False

        # Extract observation components
        obs_str = self.obs.to_text()
        obs_image = getattr(self.obs, 'image', None)
        game_info = self.env.get_game_info()

        # Convert image to JPEG bytes
        if obs_image is not None:
            obs_image_bytes = self.image_to_bytes(obs_image)
        else:
            obs_image_bytes = b""

        return obs_str, obs_image_bytes, game_info

    def dispatch_action_and_get_score(self, action: str) -> Tuple[float, bool, bool]:
        """
        Execute an action on the environment and return the result.

        Args:
            action: String representation of the action to execute

        Returns:
            Tuple of (score, is_finished, max_episodes_reached)
            - score: Score achieved in this step
            - is_finished: Whether the current episode is finished
            - max_episodes_reached: Whether max episode limit has been reached

        Side effects:
            - Updates internal score tracking
            - Increments episode count on episode completion
            - Resets environment for new episode (if not at max episodes)
            - Latches final observation when max episodes reached
        """
        # Capture pre-step observation for potential latching
        pre_step_obs_str = ""
        pre_step_obs_image_bytes = b""
        if self.obs is not None:
            pre_step_obs_str = self.obs.to_text()
            obs_image = getattr(self.obs, 'image', None)
            if obs_image is not None:
                pre_step_obs_image_bytes = self.image_to_bytes(obs_image)
        pre_step_game_info = self.env.get_game_info()

        # Execute action
        action_obj = self.env.text2action(action)
        logger.info(f"Executing action: {action_obj}")

        self.obs, reward, terminated, truncated, info = self.env.step(action_obj)
        score, done = self.env.evaluate(self.obs)
        self._current_step += 1

        if self._current_step >= self._max_steps:
            terminated = True
            truncated = True
            done = True

        # Determine if episode is finished
        is_finished = terminated or truncated or done
        logger.info(f"Score: {score}, is_finished: {is_finished}")

        self._score = score

        # Check if we've hit max steps per episode
        # Note: Caller is responsible for tracking step count
        # This logic may need to be adjusted based on how steps are tracked

        # Track if we've reached max episodes
        max_episodes_reached = False

        # Handle episode completion
        if is_finished:
            self._episodes += 1
            self._total_score += score

            max_episodes = self.cfg.env.get("max_episodes", MAX_EPISODES)
            if self._episodes < max_episodes:
                # Start new episode
                logger.info(f"Episode {self._episodes} finished. Starting new episode...")
                self.obs = self.reset_env()
            else:
                # Max episodes reached - latch final observation
                max_episodes_reached = True
                self._all_episodes_finished = True
                self._latched_final_obs = (
                    pre_step_obs_str,
                    pre_step_obs_image_bytes,
                    pre_step_game_info
                )
                self._latched_final_result = (score, is_finished, max_episodes_reached)
                self.log_game_results()
                logger.info(f"Max episodes ({max_episodes}) reached. Latching final observation.")

        return score, is_finished, max_episodes_reached

    def log_game_results(self):
        self._end_time = time.time()
        with open(GAME_RESULTS_PATH, "w") as fp:
            json.dump({
                "score": self._score,
                "avg_score": (self._total_score / self._episodes) if self._episodes > 0 else 0,
                "start_time": self._start_time,
                "end_time": self._end_time,
                "game_info": self.env.get_game_info()
            }, fp)


    def get_game_config(self) -> dict:
        """
        Get current game configuration and state.

        Returns:
            Dictionary containing:
            - game_id: Identifier for the current game
            - max_steps: Maximum steps per episode
            - max_episodes: Maximum number of episodes
            - current_episode: Number of completed episodes
            - current_step: Number of steps in current episode (caller must track)
        """
        return {
            "game_id": GAME_ID,
            "max_steps": self._max_steps,
            "max_episodes": self.cfg.env.get("max_episodes", MAX_EPISODES),
            "current_episode": self._episodes,
            "current_step": self._current_step,
            # Note: current_step tracking removed as it was tied to _steps_times
            # which was used for timing metrics. Caller should track steps if needed.
        }

    def reset_env(self):
        """
        Reset the environment for a new episode.

        Returns:
            New observation after reset

        Side effects:
            - Attempts soft reset via env.reset()
            - Falls back to creating new environment instance if reset fails
            - Updates internal state for new episode
        """
        logger.info("Resetting environment for new episode...")

        try:
            new_obs = self.env.reset()
        except Exception as e:
            logger.warning(f"Soft reset failed: {e}. Creating new environment instance...")
            new_obs = None

        # Fall back to creating completely new environment if reset failed
        if new_obs is None:
            try:
                new_env = EnvCreator(self.cfg).create()
                self.env = new_env
                self.first_loading = True
                new_obs = self.env.initial_obs()
                logger.info("New environment instance created successfully")
            except Exception as e:
                logger.error(f"Failed to create new environment: {e}")
                raise

        self.obs = new_obs
        self._current_step = 0
        return new_obs

    def get_total_score(self) -> float:
        """Get cumulative score across all episodes."""
        return self._total_score

    def get_average_score(self) -> float:
        """Get average score per episode."""
        if self._episodes == 0:
            return 0.0
        return self._total_score / self._episodes

    def get_current_episode(self) -> int:
        """Get number of completed episodes."""
        return self._episodes

    def is_all_episodes_finished(self) -> bool:
        """Check if maximum episodes have been reached."""
        return self._all_episodes_finished
