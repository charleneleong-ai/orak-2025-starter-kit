import sys
import os
import json
from datetime import datetime
from typing import Tuple
import omegaconf
import logging
import time
import base64
from io import BytesIO
import traceback
import threading

from mcp.server.fastmcp import FastMCP
from mcp_game_servers.utils.module_creator import EnvCreator

from evaluation_utils.commons import GAME_DATA_DIR, GAME_RESULTS_PATH, setup_logging, MAX_STEPS, MAX_EPISODES

logger = logging.getLogger(__name__)

GAME_ID = os.getenv("GAME_ID", "")

if sys.platform == 'win32':
    import msvcrt
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

def set_log_path(cfg, expand_log_path: bool = True):
    if expand_log_path:
        log_path = os.path.join(
            cfg.log_path,
            cfg.env_name,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        cfg.env.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
    
    setup_logging()

class MCPGameServer:

    def __init__(self, mcp_server: FastMCP, config_path: str, expand_log_path: bool = True):
        self.cfg = self.create_config(config_path, expand_log_path)
        logger.info(f"config_path: {config_path}")
        self.mcp = mcp_server
        # Lock to serialize access to env (step/reset/cleanup) across tool calls
        # self._env_lock = threading.Lock()
        self.register_tools()

        # set env
        self.env = EnvCreator(self.cfg).create()

        # set temp var
        self.obs = None
        self.first_loading = True

        self._score = 0
        self._total_score = 0
        self._start_time = None
        self._end_time = None
        self._steps_times = []
        self._episodes = 0
        self._max_steps = MAX_STEPS[GAME_ID]
        self._all_episodes_finished = False
        self._latched_final_obs = None
        self._latched_final_result = None

    def create_config(self, config_path: str, expand_log_path: bool):
        cfg = omegaconf.OmegaConf.load(config_path)
        set_log_path(cfg, expand_log_path)
        return cfg

    def image2str(self, image):
        buffered = BytesIO()
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def load_current_obs(self) -> Tuple[str, dict]:
        logger.info("Loading current observation")
        # with self._env_lock:
        if self.first_loading:
            self.obs = self.env.initial_obs()
            self.first_loading = False
        obs_str, obs_image = self.obs.to_text(), getattr(self.obs, 'image', None)
        game_info = self.env.get_game_info()

        # encode image to base64
        if obs_image is not None:
            obs_image_str = self.image2str(obs_image)
        else:
            obs_image_str = ""
        return obs_str, obs_image_str, game_info
    
    def _snapshot_current_obs_unlocked(self):
        """Capture the current observation and game info. Caller must hold _env_lock."""
        if self.obs is None:
            obs_str = ""
            obs_image_str = ""
        else:
            obs_str = self.obs.to_text()
            obs_image = getattr(self.obs, 'image', None)
            obs_image_str = self.image2str(obs_image) if obs_image is not None else ""
        game_info = self.env.get_game_info()
        return {
            "obs_str": obs_str,
            "obs_image_str": obs_image_str,
            "game_info": game_info
        }
    
    def log_game_results(self):
        self._end_time = time.time()
        with open(GAME_RESULTS_PATH, "w") as fp:
            json.dump({
                "score": self._score,
                "avg_score": (self._total_score / self._episodes) if self._episodes > 0 else 0,
                "start_time": self._start_time,
                "end_time": self._end_time,
                "steps_times": self._steps_times,
                "game_info": self.env.get_game_info()
            }, fp)

    def reset_env(self):
        """Cleanup previous environment and create a new one."""
        logger.info("Resetting environment: cleaning up previous instance...")

        # with self._env_lock:
        old_env = self.env
        new_obs = None
        try:
            new_obs = self.env.reset()
        except Exception:
            # Fall back to creating a completely new env instance
            new_obs = None

        if new_obs is None:
            try:
                new_env = EnvCreator(self.cfg).create()
            except Exception as e:
                logger.error(f"Failed to create new environment: {e}")
                # Keep old (cleaned) env reference to avoid None; caller will see errors instead of hanging
                return

            self.env = new_env
            self.first_loading = True
            # Prime the first observation for the new env
            logger.info("New environment created successfully.")
            new_obs = self.env.initial_obs()

        # Reset per-episode step tracking so MAX_STEPS applies per episode, not globally
        self._steps_times = []
        self._start_time = None
        self._end_time = None

        return new_obs

    def dispatch_action_and_get_score(self, action_str: str) -> Tuple[int, bool, bool]:
        score = -1
        # with self._env_lock:
        action = self.env.text2action(action_str)
        logger.info(f"executing actions: {action}")
        pre_step_snapshot = self._snapshot_current_obs_unlocked()
        try:
            self.obs, reward, terminated, truncated, info = self.env.step(action)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise e
        score, done = self.env.evaluate(self.obs)
        # retries = 0
        # while score is None and retries < 30:
        #     self.obs, reward, terminated, truncated, info = self.env.step(action)
        #     score, done = self.env.evaluate(self.obs)
        #     retries += 1
        #     time.sleep(10)
        # if score is None:
        #     raise Exception("Failed to get score, is the game running?")
        is_finished = terminated or truncated or done
        logger.info(f"score: {score}, is_finished: {is_finished}")
        self._score = score
        if len(self._steps_times) >= self._max_steps:
            is_finished = True
        
        # Track if we've reached max episodes
        max_episodes_reached = False
        
        # Only reset the env when the episode actually finishes and we need a new one
        if is_finished:
            self._episodes += 1
            self._total_score += score
            if self._episodes < MAX_EPISODES:
                self.obs = self.reset_env()
            else:
                # Max episodes reached - latch final observation
                max_episodes_reached = True
                self._all_episodes_finished = True
                self._latched_final_obs = pre_step_snapshot
                logger.info(f"Max episodes ({MAX_EPISODES}) reached. Latching final observation.")
        
        return score, is_finished, max_episodes_reached
    
    def wait_for_env(self):
        while self.env is None:
            time.sleep(0.1)
            logger.info("Waiting for environment to be created...")
        logger.info("Environment created successfully.")

    def register_tools(self):
        @self.mcp.tool(name="load-obs", description="Load observation and game info from the server.")
        def load_obs() -> str:
            logger.info("Request for loading observation")
            if self._all_episodes_finished and self._latched_final_obs is not None:
                logger.info("Game finished. Returning latched final observation.")
                return json.dumps(self._latched_final_obs)
            self.wait_for_env()
            if not self._start_time:
                self._start_time = time.time()
            else:
                self._steps_times.append(time.time())
            
            try:
                obs_str, obs_image_str, game_info = self.load_current_obs()
            except Exception as e:
                logger.error(traceback.format_exc())
                raise e
            logger.info(f"load_obs result: {obs_str}, \n{game_info}")
            return json.dumps({
                "obs_str": obs_str,
                "obs_image_str": obs_image_str,
                "game_info": game_info
            })

        @self.mcp.tool(name="send-action-set", description="Send a action set to the server.")
        def send_action_set(action_set: list) -> str:
            self.wait_for_env()
            self.env.send_action_set(action_set)
            return "Action set sent"
        
        @self.mcp.tool(name='get-current-state', description='Get the current state of the game')
        def get_current_state() -> str:
            self.wait_for_env()
            current_state = self.env._receive_state()
            return json.dumps({
                "current_state": current_state
            })

        @self.mcp.tool(name="dispatch-final-action", description="Dispatch a client final action to the server and return score and termination flag")
        def dispatch_final_action(action_str: str) -> str:
            if self._all_episodes_finished and self._latched_final_result is not None:
                logger.info("Game finished. Returning latched final result.")
                return json.dumps(self._latched_final_result)
            self.wait_for_env()
            try:
                score, is_finished, max_episodes_reached = self.dispatch_action_and_get_score(action_str)
            except Exception as e:
                logger.error(traceback.format_exc())
                raise e
            logger.info(f"dispatch_final_action result: {score}, {is_finished}")
            
            response_dict = {
                "score": score,
                "avg_score": (self._total_score / self._episodes) if self._episodes > 0 else 0,
                "is_finished": is_finished
            }
            
            if max_episodes_reached:
                self._latched_final_result = response_dict
                self._all_episodes_finished = True
                self.log_game_results()
            
            return json.dumps(response_dict)
        
        @self.mcp.tool(name="get-game-config", description="Get the game config")
        def get_game_config() -> str:
            logger.info("Getting game config...")
            return json.dumps({
                "game_id": GAME_ID,
                "max_steps": self._max_steps,
                "max_episodes": MAX_EPISODES,
                "current_episode": self._episodes,
                "current_step": len(self._steps_times),
            })

    async def run(self):
        await self.mcp.run_streamable_http_async()
