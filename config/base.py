import os
from typing import Optional, Any
from pydantic import BaseModel
from config.agent_config import AgentConfig


class WandbConfig(BaseModel):
    """Weights & Biases configuration."""

    project: str = "orak-2048"
    entity: Optional[str] = None
    mode: str = "online"  # "online", "offline", or "disabled"
    tags: list = ["2048"]

    def model_post_init(self, __context):
        self.project = os.environ.get("WANDB_PROJECT", self.project)
        self.entity = os.environ.get("WANDB_ENTITY", self.entity)
        self.mode = os.environ.get("WANDB_MODE", self.mode)
        if self.tags is None:
            self.tags = ["2048"]

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"


class Settings(BaseModel):
    wandb: WandbConfig = WandbConfig()
    twenty_fourty_eight: AgentConfig = None
    pokemon_red: AgentConfig = None
    super_mario: AgentConfig = None
    star_craft: AgentConfig = None

