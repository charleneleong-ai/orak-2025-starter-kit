from importlib import import_module
from typing import Any, Optional, Type

import hydra
from hydra import compose, initialize
from loguru import logger
from omegaconf import OmegaConf

from pathlib import Path

from config.base import Settings

ROOT_DIR = Path(__file__).parent.parent


def load_hydra_settings(config_name: str = "config") -> Settings:
    """Load Hydra settings from config name"""
    with initialize(version_base=hydra.__version__, config_path="../configs"):
        cfg = compose(config_name=config_name)
        ## Compose API does not Hydra resolver for hydra:runtime like @hydra.main(); need to manually override
        ## https://github.com/facebookresearch/hydra/issues/2017
        cfg["CWD"] = str(ROOT_DIR)

        cfg_dict: dict[str, Any] = dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        return Settings(**cfg_dict)


def get_module_by_class_path(class_path: str) -> Optional[Type]:  # type: ignore[type-arg]
    """
    Dynamically imports a class from a string class path.

    Args:
        class_path (str): Full class path in the format 'module.submodule.ClassName'.

    Returns:
        Type: The class referenced by class_path.

    Raises:
        ImportError: If the module or class cannot be imported.
    """
    try:
        logger.debug(f"Instantiating module by class path: {class_path}")
        module_name, class_name = class_path.rsplit(".", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        return cls  # type: ignore[no-any-return]
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import {class_path}: {e}") from e



def load_agent_map(settings: Settings, games: list[str] | None = None) -> dict[str, Any]:
    """Load agent map based on settings.
    
    Args:
        settings: Settings object containing game configurations
        games: Optional list of game names to load agents for. If None, loads all available agents.
    """

    agent_map = {}

    # If games list is provided, only load those agents
    should_load = lambda game: games is None or game in games

    def _create_agent(game_name: str, agent_config, wandb_config):
        """Instantiate agent, passing game_name for UnifiedMaclaAgent."""
        agent_cls = get_module_by_class_path(agent_config.class_name)
        if "unified" in agent_config.class_name.lower():
            return agent_cls(config=agent_config, wandb_config=wandb_config, game_name=game_name)
        return agent_cls(config=agent_config, wandb_config=wandb_config)

    for game_name in ["twenty_fourty_eight", "pokemon_red", "super_mario", "star_craft"]:
        game_settings = getattr(settings, game_name, None)
        if game_settings is None or not should_load(game_name):
            continue

        agent_config = game_settings.agent
        wandb_config = game_settings.wandb or settings.wandb

        if wandb_config and settings.wandb.run_id:
            wandb_config.run_id = settings.wandb.run_id

        agent_map[game_name] = _create_agent(game_name, agent_config, wandb_config)

    logger.info(f"Loaded agent map for games {games or 'all'}: {list(agent_map.keys())}")
    return agent_map
