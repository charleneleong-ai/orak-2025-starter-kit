import os
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, List, Optional
import multiprocessing
from PIL import Image
from collections import deque
import numpy as np
import re
from dacite import from_dict

from mcp_game_servers.base_env import BaseEnv
from mcp_game_servers.gameio.window_capture import WindowCapture
from mcp_game_servers.utils.types.game_io import Action, Obs

from .utils.bots import sc2_run_game
from .utils.actions import ActionDescriptions

LADDER_MAP_2023 = [
    # 'Altitude LE',
    # 'Ancient Cistern LE',
    # 'Babylon LE',
    # 'Dragon Scales LE',
    # 'Gresvan LE',
    # 'Neohumanity LE',
    # 'Royal Blood LE',
    # 'Abyssal Reef LE',
    "Ancient Cistern LIE",
    "LastFantasyAIE",
    'Flat64'
]

def wait_for_window(window_name: str, timeout: int = 30, interval: float = 1.0) -> WindowCapture:
    return None
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            wc = WindowCapture(window_name, adjust_dpi=True)
            return wc
        except Exception as e:
            print(f"[INFO] Waiting for window '{window_name}'... ({e})")
            time.sleep(interval)
    raise TimeoutError(f"Window '{window_name}' not found within {timeout} seconds.")

@dataclass
class StarCraftObs(Obs):
    observation: dict
    image: Optional[Image.Image] = None
    minimap_image: Optional[Image.Image] = None

    def to_text(self):

        def create_summary(category_data):
            # Gracefully handle missing or malformed category data
            if not isinstance(category_data, dict):
                return ""
            summary = ""
            for key, value in category_data.items():
                if isinstance(value, dict):
                    sub_summary = create_summary(value)
                    if sub_summary != "":
                        summary += f"\n{key.replace('_', ' ').capitalize()}:\n{sub_summary}"
                elif value != 0: 
                    summary += f"- {key.replace('_', ' ').capitalize()}: {value}\n"
            return summary
        
        # Normalize any JSON-serialized observation entries into dictionaries
        for key, value in list(self.observation.items()):
            if isinstance(value, str):
                try:
                    self.observation[key] = json.loads(value.replace("'", "\""))
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse value of '{key}' as JSON. Value: {value}")

        summary = ""

        for key, temp_obs in self.observation.items():

            # Skip entries that are not in the expected dict format
            if not isinstance(temp_obs, dict):
                continue

            resource_section = temp_obs.get('resource')
            # At episode boundaries or during initialization, we can see incomplete
            # observations where "resource" is missing or not yet populated. In that
            # case, skip this entry instead of crashing with a ValueError.
            if not isinstance(resource_section, dict):
                continue

            game_time = resource_section.get('game_time', "unknown time")

            temp_summary = f"{key}: At {game_time} game time, our current StarCraft II situation is as follows:\n\n"

            categories = [
                ("Resources", resource_section),
                ("Buildings", temp_obs.get("building", {})),
                ("Units", temp_obs.get("unit", {})),
                ("Research", temp_obs.get("research", {})),
                ("In Progress", temp_obs.get("in_progress", {})),
                ("Enemy", temp_obs.get("enemy", {})),
            ]

            for category, category_data in categories:
                category_summary = create_summary(category_data)
                if category_summary != "":
                    temp_summary += f"{category}:\n{category_summary}\n"
            
            summary += temp_summary

        return summary

@dataclass
class StarCraftAction(Action):
    actions: List[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        return iter(self.actions)

    def __getitem__(self, index: int) -> int:
        return self.actions[index]

    def __len__(self) -> int:
        return len(self.actions)

    def to_json(self) -> str:
        return json.dumps(self.actions)


class StarCraftEnv(BaseEnv):
    @dataclass
    class Config:
        task: str

        map_idx: int
        player_race: str

        bot_race: str
        bot_difficulty: int
        bot_build: int

        query_interval: int
        num_summaries: int
        num_actions: int
        
        log_path: str

        input_modality: str = "text"

    cfg: Config
    
    def configure(self):
        self.process_id = -1
        self.log_path = self.cfg.log_path

        self.map_pool = LADDER_MAP_2023
        self.map_idx = self.cfg.map_idx
        self.map_name = self.map_pool[self.map_idx]
        
        self.player_race = self.cfg.player_race

        self.bot_race = self.cfg.bot_race
        self.bot_difficulty = self.cfg.bot_difficulty
        self.bot_build = self.cfg.bot_build
        
        self.action_description = ActionDescriptions(self.player_race)
        self.action_dict_tmp = self.action_description.action_descriptions
        self.action_dict = {}
        for category in self.action_dict_tmp:
            for key, value in self.action_dict_tmp[category].items():
                self.action_dict[value.upper()] = key
        print("action_dict", self.action_dict)

        self.query_interval = self.cfg.query_interval
        self.num_summaries = self.cfg.num_summaries
        self.num_actions = self.cfg.num_actions

        self.summary = {}
        self.executed_actions = []

        # IMPORTANT:
        # This env runs inside a gRPC server which has background threads.
        # Forking a new process from inside a gRPC request handler can deadlock
        # (and produces logs like: "Other threads are currently calling into gRPC,
        # skipping fork() handlers"). To avoid the "stuck after reset" hang where
        # `action_step()` waits forever for `isReadyForNextStep`, we default to a
        # safer multiprocessing start method.
        #
        # Override if you know what you're doing:
        #   ORAK_STARCRAFT_MP_START_METHOD=spawn|forkserver|fork
        mp_start_method = os.getenv("ORAK_STARCRAFT_MP_START_METHOD", "spawn")
        try:
            ctx = multiprocessing.get_context(mp_start_method)
        except ValueError:
            ctx = multiprocessing.get_context()
        self._mp_ctx = ctx

        # Keep a reference to the Manager to prevent garbage collection.
        # Use the Manager only for the shared dict; use a normal Lock so it
        # does not depend on the Manager's IPC channel.
        self._manager = ctx.Manager()
        self.lock = ctx.Lock()
        self.transaction = self._manager.dict()
        self.transaction.update(
            {'information': {}, 'reward': 0, 'action': None,
             'done': False, 'result': None, 'iter': 0, 'command': None, "output_command_flag": False,
             'action_executed': [], 'action_failures': [], })
        self.isReadyForNextStep = ctx.Event()
        self.game_end_event = ctx.Event()
        self.game_over = ctx.Value('b', False)
        self.done_event = ctx.Event()
        self.p = None
        self.check_process(reset=True)
        self.check_process()

        self.use_image = self.cfg.input_modality in ["image", "text_image"]
        if self.use_image:
            self.window_capture = wait_for_window("StarCraft II", timeout=60)

    def check_process(self, reset=False):
        ctx = getattr(self, "_mp_ctx", multiprocessing.get_context())
        if self.p is not None:
            if self.p.is_alive():
                if not self.game_over.value:  # Check if the game is over
                    return  # If the game is not over, just return and do not restart the process
                self.p.terminate()
            self.p.join()
        if reset:
            # Clear any latched terminal/ready state from the previous episode so
            # the next worker process can run and signal steps correctly.
            try:
                self.done_event.clear()
            except Exception:
                pass
            try:
                self.isReadyForNextStep.clear()
            except Exception:
                pass
            try:
                self.game_end_event.clear()
            except Exception:
                pass
            try:
                self.game_over.value = False
            except Exception:
                pass

            self.transaction.update(
                {'information': {}, 'reward': 0, 'action': None,
                 'done': False, 'result': None, 'iter': 0, 'command': None, "output_command_flag": False,
                 'action_executed': [], 'action_failures': [], })
            if self.player_race == 'Protoss':
                # Use the same multiprocessing context as our sync primitives.
                self.p = ctx.Process(target=sc2_run_game, args=(
                    self.transaction, self.lock, self.isReadyForNextStep, self.game_end_event,
                    self.done_event, self.bot_race, self.bot_difficulty, self.bot_build, self.map_name, self.log_path))
            else:
                raise ValueError("Invalid race. Only 'Protoss' and 'Zerg' are supported.")
            self.p.start()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.check_process(reset=True)
        self.game_end_event.clear()

        image = None
        minimap_image = None
        if self.use_image:
            try:
                # image = self.window_capture.capture(log_path=self.cfg.log_path)
                image = self.transaction['map_image']
                minimap_image = self.transaction['minimap_image']
            except Exception as e:
                print(f"[WARNING] Failed to capture image: {e}")
                pass
            
        state = from_dict(StarCraftObs, {'observation': self.transaction['information'], 'image': image, 'minimap_image': minimap_image})

        return state

    def initial_obs(self) -> Obs:
        return self.reset()

    def obs2text(self, obs: Obs) -> str:
        if not obs.observation:
            return None
        else:
            return obs.to_text()

    def text2action(self, text: str) -> Action:

        individual_actions_pattern = r"\d+: <?([^>\n]+)>?"
        actions = re.findall(individual_actions_pattern, text)

        valid_actions = []

        for action in actions:
            action = action.upper()
            if action in self.action_dict:
                valid_actions.append(action)

        if len(valid_actions) > self.num_actions:
            valid_actions = valid_actions[:self.num_actions]
        elif len(valid_actions) < self.num_actions:
            for _ in range(self.num_actions - len(valid_actions)):
                valid_actions.append("EMPTY ACTION")
        
        final_actions = []

        for i in range(self.query_interval):
            if i % 2 == 0:
                final_actions.append(valid_actions.pop(0))
            else:
                final_actions.append("EMPTY ACTION")

        return StarCraftAction(actions=final_actions)
    
    def action_step(self, action) -> tuple[Obs, float, bool, bool, dict[str, Any]]:

        with self.lock:
            self.transaction['action'] = action

        while not (self.done_event.is_set() or self.isReadyForNextStep.is_set()):
            time.sleep(0.0001)

        if self.done_event.is_set():
            self.done_event.clear()
            self.isReadyForNextStep.clear()
            self.game_over.value = True
            if self.transaction['result'].name == 'Victory':
                self.transaction['reward'] += 50
        elif self.isReadyForNextStep.is_set():
            self.isReadyForNextStep.clear()
            self.check_process()
        state = self.transaction['information']

        return state, self.transaction['done']

    def step(self, action: Action) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        
        self.summary = {}
        self.executed_actions = []

        for i in range(self.query_interval):
            curr_action = self.action_dict[action.actions[i]]
            obs, done = self.action_step(curr_action)
            self.executed_actions.append(self.transaction['action_executed'])

            if done:
                break
            if i >= self.query_interval - self.num_summaries:
                summary_idx = i - self.query_interval + self.num_summaries + 1
                self.summary[f'Summary {summary_idx}'] = obs

        image = None
        minimap_image = None
        if self.use_image:
            # image = self.window_capture.capture(log_path=self.cfg.log_path)
            image = self.transaction['map_image']
            minimap_image = self.transaction['minimap_image']
            
        state = from_dict(StarCraftObs, {'observation': self.summary, 'image': image, 'minimap_image': minimap_image})

        return state, 0, done, False, None

    def evaluate(self, obs: Obs):
        result = self.transaction['result']
        if result is None:
            return 0, self.transaction['done']
        if result.name == "defeat":
            return 0, self.transaction['done']
        return 1, self.transaction['done']

    def get_game_info(self) -> dict:
        return {
            "player_race": self.player_race,
            "enemy_race": self.bot_race,
            "action_executed": self.executed_actions,
            "action_dict": self.action_dict,
            "num_actions": self.num_actions,
        }