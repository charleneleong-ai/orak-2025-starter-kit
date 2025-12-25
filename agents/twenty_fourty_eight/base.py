from typing import ClassVar, Any, Optional
from pydantic import PrivateAttr, BaseModel, Field
import wandb
import weave
from loguru import logger
from config.agent_config import AgentConfig
from config.base import WandbConfig

class GameAction(BaseModel):
    """Structured output for 2048 game actions"""
    reasoning: str = Field(description="Detailed explanation of why this action was chosen")
    action: str = Field(description="The action to take: up, down, left, or right")

class TwentyFourtyEightAgent(weave.Model):
    TRACK: ClassVar[str] = "TRACK1"
    
    config: AgentConfig
    wandb_config: WandbConfig
    
    _prev_state_str: str = PrivateAttr(default="N/A")
    _last_action: str = PrivateAttr(default="No action yet")
    _step_count: int = PrivateAttr(default=0)
    _last_score: int = PrivateAttr(default=0)

    def __init__(self, config: AgentConfig = None, wandb_config: WandbConfig = None):
        super().__init__(config=config, wandb_config=wandb_config)
        
        if self.wandb_config and self.wandb_config.enabled:
            # Ensure tags is a list
            tags = list(self.wandb_config.tags) if self.wandb_config.tags else []
            
            # Add agent specific tags if available
            if hasattr(self, "AGENT_TAGS"):
                tags.extend(self.AGENT_TAGS)
                
            wandb.init(
                project=self.wandb_config.project, 
                entity=self.wandb_config.entity,
                config=self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                tags=tags,
                name=None,  # Auto-generate run name
            )

    @weave.op()
    def act(self, obs: dict[str, Any]) -> str:
        """Main action method tracked by Weave."""
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")
        obs_image = obs.get("obs_image", None)
        
        # Extract metrics directly from game_info
        current_game_score = int(game_info.get("score", 0))
        max_tile = int(game_info.get("max_tile", 0))
        
        # Calculate evaluation score as per 2048.md: min((score / 20000) * 100, 100)
        evaluation_score = min((current_game_score / 20000) * 100, 100)
        
        self._step_count += 1

        # Get action from LLM
        action, reasoning, output_text, usage = self._get_action(
            task_description=game_info.get("task_description", ""),
            cur_state_str=cur_state_str,
            obs_image=obs_image
        )

        if self.wandb_config and self.wandb_config.enabled:
            log_data = {
                "step": self._step_count,
                "game_score": current_game_score,
                "evaluation_score": evaluation_score,
                "max_tile": max_tile,
                "action": action,
                "score_delta": current_game_score - self._last_score,
            }
            
            # Log action distribution
            log_data[f"action/{action}"] = 1
            
            # Log reasoning length as a proxy for model complexity
            if reasoning:
                log_data["reasoning_length"] = len(reasoning)
            
            # Log usage if available
            if usage:
                # Handle different usage object structures (OpenAI vs others)
                if hasattr(usage, 'prompt_tokens'):
                    log_data["tokens_prompt"] = usage.prompt_tokens
                    log_data["tokens_completion"] = usage.completion_tokens
                    log_data["tokens_total"] = usage.total_tokens
                elif isinstance(usage, dict):
                    log_data.update(usage)

            # Log obs_str as text
            if cur_state_str:
                log_data["obs_str"] = wandb.Html(f"<pre>{cur_state_str}</pre>")
            
            # Log obs_image if available
            if obs_image is not None:
                try:
                    log_data["obs_image"] = wandb.Image(obs_image, caption=f"Step {self._step_count}")
                except Exception as e:
                    # If image logging fails, just continue
                    logger.error(f"Warning: Could not log image: {e}")
            
            wandb.log(log_data)

        self._prev_state_str = cur_state_str
        self._last_action = action
        self._last_score = current_game_score

        return action

    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, str, str, Any]:
        """
        Get action from LLM. This method should be implemented by subclasses.
        Returns: (action, reasoning, output_text, usage)
        """
        raise NotImplementedError

    def __del__(self):
        """Cleanup wandb on agent destruction."""
        if hasattr(self, "wandb_config") and self.wandb_config and self.wandb_config.enabled:
            try:
                wandb.finish()
            except:
                pass
