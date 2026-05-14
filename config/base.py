import os

from pydantic import BaseModel

from config.agent_config import AgentConfig
from config.env_config import (
    PokemonRedEnvConfig,
    StarCraftEnvConfig,
    SuperMarioEnvConfig,
    TwentyFourtyEightEnvConfig,
)


class WandbConfig(BaseModel):
    """Weights & Biases configuration (includes Weave)."""

    project: str = ""
    entity: str | None = None
    run_id: str | None = None
    mode: str = "online"  # "online", "offline", or "disabled"
    tags: list = []
    notes: str | None = None

    # Weave-specific settings
    weave_enabled: bool = True

    # Paired-rollout / agentic-RL collection metadata. Set by run.py from
    # --rollout-group-id / --rollout-idx / --adapter-name. Propagated into
    # wandb.init(config=...) and per-step raw_requests.jsonl so the trainer
    # side (offline DPO/GSPO or online LoRA loop) can filter rollouts by
    # group and join them with the adapter that produced them.
    rollout_group_id: str | None = None
    rollout_idx: int = 0
    adapter_name: str | None = None

    def model_post_init(self, __context):
        self.project = os.environ.get("WANDB_PROJECT", self.project)
        self.entity = os.environ.get("WANDB_ENTITY", self.entity)
        self.run_id = os.environ.get("WANDB_RUN_ID", self.run_id)
        self.mode = os.environ.get("WANDB_MODE", self.mode)

        # Check if Weave is explicitly disabled
        self.weave_enabled = os.environ.get("WEAVE_ENABLED", "true").lower() in ["true", "1", "yes"]

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


class TwentyFourtyEightConfig(BaseModel):
    agent: AgentConfig
    env: TwentyFourtyEightEnvConfig
    wandb: WandbConfig | None = None


class PokemonRedConfig(BaseModel):
    agent: AgentConfig
    env: PokemonRedEnvConfig
    wandb: WandbConfig | None = None


class SuperMarioConfig(BaseModel):
    agent: AgentConfig
    env: SuperMarioEnvConfig
    wandb: WandbConfig | None = None


class StarCraftConfig(BaseModel):
    agent: AgentConfig
    env: StarCraftEnvConfig
    wandb: WandbConfig | None = None


class Settings(BaseModel):
    wandb: WandbConfig = WandbConfig()
    twenty_fourty_eight: TwentyFourtyEightConfig = None
    pokemon_red: PokemonRedConfig = None
    super_mario: SuperMarioConfig = None
    star_craft: StarCraftConfig = None
