import os
from typing import Optional, Any
from pydantic import BaseModel
from config.agent_config import AgentConfig


class WandbConfig(BaseModel):
    """Weights & Biases configuration (includes Weave)."""

    project: str = "orak-2048"
    entity: Optional[str] = None
    mode: str = "online"  # "online", "offline", or "disabled"
    tags: list = ["2048"]
    
    # Weave-specific settings
    weave_enabled: bool = True

    def model_post_init(self, __context):
        self.project = os.environ.get("WANDB_PROJECT", self.project)
        self.entity = os.environ.get("WANDB_ENTITY", self.entity)
        self.mode = os.environ.get("WANDB_MODE", self.mode)
        
        # Check if Weave is explicitly disabled
        weave_enabled_env = os.environ.get("WEAVE_ENABLED", "true").lower()
        self.weave_enabled = weave_enabled_env in ["true", "1", "yes"]
        
        if self.tags is None:
            self.tags = ["2048"]

    @property
    def enabled(self) -> bool:
        """W&B logging enabled."""
        return self.mode != "disabled"
    
    @property
    def project_name(self) -> str:
        """Get the full project name for Weave initialization."""
        if self.entity:
            return f"{self.entity}/{self.project}"
        return self.project


class Settings(BaseModel):
    wandb: WandbConfig = WandbConfig()
    twenty_fourty_eight: AgentConfig = None
    pokemon_red: AgentConfig = None
    super_mario: AgentConfig = None
    star_craft: AgentConfig = None
