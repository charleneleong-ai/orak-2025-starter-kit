from typing import ClassVar, Any, Optional, Tuple, Dict
from pydantic import PrivateAttr, BaseModel, Field
import wandb
import weave
from loguru import logger
from config.agent_config import AgentConfig
from config.base import WandbConfig
from agents.base import OrakAgent

class GameAction(BaseModel):
    """Structured output for 2048 game actions"""
    reasoning: str = Field(description="Detailed explanation of why this action was chosen")
    action: str = Field(description="The action to take: up, down, left, or right")

class TwentyFourtyEightAgent(OrakAgent):
    
    def calculate_metrics(self, game_info: Dict[str, Any]) -> Dict[str, Any]:
        current_game_score = int(game_info.get("score", 0))
        return {
            "evaluation_score": min((current_game_score / 20000) * 100, 100),
            "max_tile": int(game_info.get("max_tile", 0))
        }

    def get_action(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")
        obs_image = obs.get("obs_image", None)
        
        action, reasoning, output_text, usage, prompt = self._get_action(
            task_description=game_info.get("task_description", ""),
            cur_state_str=cur_state_str,
            obs_image=obs_image
        )
        
        log_extras = {}
        if prompt:
            log_extras["prompt"] = prompt
        if output_text:
            log_extras["output_text"] = output_text
        if reasoning:
            log_extras["reasoning_length"] = len(reasoning)
        if usage:
             if hasattr(usage, 'prompt_tokens'):
                log_extras["tokens_prompt"] = usage.prompt_tokens
                log_extras["tokens_completion"] = usage.completion_tokens
                log_extras["tokens_total"] = usage.total_tokens
             elif isinstance(usage, dict):
                log_extras.update(usage)
                
        return action, log_extras

    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> Tuple[str, str, str, Any, str]:
        """
        Get action from LLM. This method should be implemented by subclasses.
        Returns: (action, reasoning, output_text, usage, prompt)
        """
        raise NotImplementedError
