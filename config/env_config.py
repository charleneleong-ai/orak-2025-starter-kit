from dataclasses import dataclass
from typing import Literal

@dataclass
class TwentyFourtyEightEnvConfig:
    show_graphic: bool = True
    log_path: str = "./logs"
    target_tile: int = 2048
    task: str = "Merge Tiles to Reach the Target"
    input_modality: str = "text_image"
    env_name: str = "TwentyFourtyEight"
    max_episodes: int = 3
    max_steps: int = 1000

@dataclass
class PokemonRedEnvConfig:
    env_name: str = "PokemonRed"
    log_path: str = "./logs"
    task: str = "DefeatBrock"
    input_modality: str = "text"
    rom_path: str = "./executables/pokemon_red/pyboy/pokered.gbc"
    success_condition: str = "get_boulder_badge"
    max_episodes: int = 3
    max_steps: int = 200

@dataclass
class SuperMarioEnvConfig:
    env_name: str = "SuperMario"
    log_path: str = "./logs"
    task: str = "Complete stage 1-1"
    input_modality: str = "image+text"
    logging: bool = False
    max_episodes: int = 3
    max_steps: int = 100

@dataclass
class StarCraftEnvConfig:
    env_name: str = "StarCraft"
    log_path: str = "./logs"
    task: str = "1 vs 1 battle againt built-in ai"
    input_modality: str = "text_image"
    map_idx: int = 0
    player_race: str = "Protoss"
    bot_race: str = "Zerg"
    bot_difficulty: int = 4
    bot_build: int = 2
    query_interval: int = 10
    num_summaries: int = 1
    num_actions: int = 5
    max_episodes: int = 3
    max_steps: int = 1000

