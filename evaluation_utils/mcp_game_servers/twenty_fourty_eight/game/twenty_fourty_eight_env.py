import ast
import os
import json
import re
import logging
import pygame

from PIL import Image
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
from mcp_game_servers.twenty_fourty_eight.game.game import *

from rich import print

from mcp_game_servers.base_env import BaseEnv
from mcp_game_servers.utils.types.game_io import Action, Obs

logger = logging.getLogger(__name__)

THEME = "light"
TEXT_COL = (0, 0, 0)

DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500
SIZE = (DEFAULT_WIDTH, DEFAULT_HEIGHT)


@dataclass
class TwentyFourtyEightObs(Obs):
    observation: list
    terminated: bool
    score: int = 0
    max_tile: int = 0
    image: Image.Image = None

    def to_text(self):
        obs_text = f"Board of 2048 Games: \n {self.observation[0]} \n {self.observation[1]} \n {self.observation[2]} \n {self.observation[3]} \n Score: {self.score}"
        logger.info(f"{obs_text}")
        return obs_text


@dataclass
class TwentyFourtyEightAction(Action):
    actions: list[int] = field(default_factory=list)

    def __iter__(self) -> Iterator[int]:
        return iter(self.actions)

    def __getitem__(self, index: int) -> int:
        return self.actions[index]

    def __len__(self) -> int:
        return len(self.actions)

    def to_json(self) -> str:
        return json.dumps(self.actions)



class TwentyFourtyEightEnv(BaseEnv):
    @dataclass
    class Config:
        show_graphic: bool
        log_path: str
        target_tile: int
        task: str
        input_modality: str = "text_image"
        initial_board: Any = None
        initial_score: float = 0
        initial_step: int = 0
        max_episodes: int = 10
        max_steps: int = 1000

        def __post_init__(self):
            if self.initial_board:
                if isinstance(self.initial_board, str):
                    try:
                        self.initial_board = ast.literal_eval(self.initial_board)
                        logger.info(f"Parsed initial_board from string")
                    except Exception as e:
                        logger.error(f"Failed to parse initial_board: {e}, value: {self.initial_board}")
                        self.initial_board = None
                elif isinstance(self.initial_board, list):
                    if len(self.initial_board) != 4 or any(len(row) != 4 for row in self.initial_board):
                        logger.error(f"Invalid initial_board dimensions: {self.initial_board}")
                        self.initial_board = None
        
    cfg: Config

    def configure(self):
        self.observations = []
        self.previous_actions = []
        self.show_graphic = self.cfg.show_graphic
        init_game(self.show_graphic)
        self.target_tile = self.cfg.target_tile
        self.input_modality = self.cfg.input_modality
        self.score = getattr(self.cfg, "initial_score", 0)
        self.max_tile = 0

        self.consecutive_nochange_step = 0
        
        # Use initial board if provided, otherwise start new game
        initial_board = self.cfg.initial_board

        # Convert string to list if needed
        if initial_board and isinstance(initial_board, str):
            try:
                initial_board = ast.literal_eval(initial_board)
                logger.info(f"Converted initial_board from string to list")
            except Exception as e:
                logger.error(f"Failed to parse initial_board string: {e}, value: {initial_board}")
                initial_board = None

        if initial_board and isinstance(initial_board, list):
            # Validate it's a 4x4 board
            if len(initial_board) == 4 and all(isinstance(row, list) and len(row) == 4 for row in initial_board):
                self._env = initial_board
                try:
                    self.max_tile = max(max(row) for row in self._env)
                except Exception as e:
                    logger.error(f"Failed to compute max_tile: {e}, board: {self._env}")
                    self.max_tile = 0
                logger.info(f"Restoring game state from configuration: score={self.score} max_tile={self.max_tile}")
            else:
                logger.error(f"Invalid initial_board dimensions: {initial_board}")
                self._env = newGame(THEME, TEXT_COL, SIZE, self.show_graphic)
        else:
            if initial_board:
                logger.error(f"initial_board is invalid type: {type(initial_board)}")
            self._env = newGame(THEME, TEXT_COL, SIZE, self.show_graphic)
            
        self.use_image = self.input_modality in ["image", "text_image"]
        self.step_count = getattr(self.cfg, "initial_step", 0)
        
        self.log_path = self.cfg.log_path
        

    def pygame_surface_to_pil(self):
        # Get the pygame display surface
        surface = pygame.display.get_surface()
        data = pygame.image.tostring(surface, 'RGB')
        width, height = surface.get_size()
        image = Image.frombytes('RGB', (width, height), data)
        return image
    
    def _get_max_tile(self, board):
        """Calculate max tile from the board"""
        try:
            return max(max(row) for row in board)
        except Exception:
            return 0

    def initial_obs(self) -> Obs:
        observation = self._env
        terminated = False
        self.max_tile = self._get_max_tile(observation)

        image = None
        if self.use_image:
            image = self.pygame_surface_to_pil()

        obs = TwentyFourtyEightObs(
             observation=observation,
             terminated=terminated,
             score=self.score,
             max_tile=self.max_tile,
             image=image,
        )
        return obs

    def obs2text(self, obs: Obs) -> str:
        text = obs.to_text()
        return text

    def text2action(self, text: str) -> Action:
        matches = re.findall(r"\**([\w ]+)\**.*", text)
        # Convert matched actions to lowercase for case-insensitive comparison
        action = ["".join(match).lower() for match in matches]
        return TwentyFourtyEightAction(actions=action)

    def step(self, action: Action) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        if action.actions[0] in ['up', 'down', 'left', 'right']:
            # Game Environment settings
            new_board, merged_score = move(action.actions[0], deepcopy(self._env))
            self.score += merged_score 
        else:
            new_board = self._env

        # Only update board if there was a change
        if new_board != self._env:
            self._env = fillTwoOrFour(new_board)
            if self.show_graphic:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None, 0, True, False, None
                display(self._env, THEME, SIZE)
                pygame.display.flip()
            self.consecutive_nochange_step = 0
        else:
            self.consecutive_nochange_step += 1
        
        if self.consecutive_nochange_step >= 5: # five consecutive nochange step -> terminated
            terminated = True
        else:
            status = checkGameStatus(self._env, max_tile = self.target_tile)
            if status == "PLAY":
                terminated = False
            else:
                terminated = True

        observation = self._env
        self.max_tile = self._get_max_tile(observation)

        image = None
        if self.use_image:
            image = self.pygame_surface_to_pil()
            self.step_count += 1
            image_path = f"{self.log_path}/step_{self.step_count:04d}.png"
            image.save(image_path)
        
        obs = TwentyFourtyEightObs(
            observation=observation,
            terminated=terminated,
            score=self.score,
            max_tile=self.max_tile,
            image=image,
        )

        return obs, 0, obs.terminated, False, None

    def evaluate(self, obs: Obs):
        done = obs.terminated
        return min(1.0, obs.score/20_000), done

    def get_game_info(self) -> dict:
        return {
            "prev_state_str": None,
            "task_description": "Merge tiles to make a tile with the value of 2048",
            "score": self.score,
            "max_tile": self.max_tile,
            "board_state": self._env
        }