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



def load_agent_map(settings: Settings) -> dict[str, Any]:
    """Load agent map based on settings."""

    agent_map = {}
    if settings.twenty_fourty_eight is not None:
        # Access agent config from the nested structure
        agent_config = settings.twenty_fourty_eight.agent
        agent_map["twenty_fourty_eight"] = get_module_by_class_path(
            agent_config.class_name
        )(config=agent_config, wandb_config=settings.wandb)

    if settings.pokemon_red is not None:
        agent_config = settings.pokemon_red.agent
        agent_map["pokemon_red"] = get_module_by_class_path(
            agent_config.class_name
        )(config=agent_config, wandb_config=settings.wandb)
        
    if settings.super_mario is not None:
        agent_config = settings.super_mario.agent
        agent_map["super_mario"] = get_module_by_class_path(
            agent_config.class_name
        )(config=agent_config, wandb_config=settings.wandb)
        
    if settings.star_craft is not None:
        agent_config = settings.star_craft.agent
        agent_map["star_craft"] = get_module_by_class_path(
            agent_config.class_name
        )(config=agent_config, wandb_config=settings.wandb)
    
    logger.info(f"Loaded agent map: {agent_map}")
    return agent_map
