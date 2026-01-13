import traceback
from typing import ClassVar, Any, Optional
from pydantic import PrivateAttr
import wandb
import weave
import os
import json
from loguru import logger
from config.agent_config import AgentConfig
from config.base import WandbConfig

class BaseOrakAgent(weave.Model):
    TRACK: ClassVar[str] = "TRACK1"
    
    config: AgentConfig
    wandb_config: WandbConfig
    
    _prev_state_str: str = PrivateAttr(default="N/A")
    _last_action: str = PrivateAttr(default="No action yet")
    _step_count: int = PrivateAttr(default=0)
    _last_score: int = PrivateAttr(default=0)
    
    # Stats tracking
    _stats: dict[str, int] = PrivateAttr(default_factory=lambda: {
        "total_inference_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0
    })
    _requests_log_path: Optional[str] = PrivateAttr(default=None)

    # Per-episode stats
    _episode_stats: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _current_episode_stats: dict[str, int] = PrivateAttr(default_factory=lambda: {
        "inference_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens": 0
    })

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
                id=self.wandb_config.run_id,
                resume="allow",
                config=self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                tags=tags,
                notes=self.wandb_config.notes,
                name=self.wandb_config.run_id
            )

    def set_log_dir(self, log_dir: str):
        """Set directory for logging raw requests."""
        os.makedirs(log_dir, exist_ok=True)
        self._requests_log_path = os.path.join(log_dir, "raw_requests.jsonl")

    def get_model_declaration(self) -> dict[str, Any]:
        """Return model declaration."""
        return {
            "name": self.config.model,
            "version": "unknown", 
            "provider": self.AGENT_TAGS[0] if hasattr(self, "AGENT_TAGS") and self.AGENT_TAGS else "unknown",
            "parameter_count": "unknown", 
        }

    def get_evaluation_summary(self, episodes: int) -> dict[str, Any]:
        """Return evaluation summary."""
        return {
            "total_inference_calls": self._stats["total_inference_calls"],
            "total_tokens": self._stats["total_tokens"],
            "evaluation_episodes": episodes,
            "mean_calls_per_episode": self._stats["total_inference_calls"] / episodes if episodes > 0 else 0,
            "mean_tokens_per_episode": self._stats["total_tokens"] / episodes if episodes > 0 else 0,
            "episodes": self._episode_stats,
        }

    def record_episode_end(self, episode_id: int, game_name: str, seed: Any, final_score: float):
        """Record stats for a completed episode."""
        self._episode_stats.append({
            "episode_id": episode_id,
            "game_name": game_name,
            "seed": seed,
            "inference_calls": self._current_episode_stats["inference_calls"],
            "tokens": self._current_episode_stats["tokens"],
            "final_score": final_score
        })
        # Reset current episode stats
        self._current_episode_stats = {
            "inference_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tokens": 0
        }

    @weave.op()
    def act(self, obs: dict[str, Any], step: int = None) -> str:
        """Main action method tracked by Weave."""
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")
        obs_image = obs.get("obs_image", None)
        
        current_score = int(float(game_info.get("score", 0)))
        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1

        # Get action from subclass
        action, log_extras = self.get_action(obs)
        
        # Update stats
        self._stats["total_inference_calls"] += 1
        self._current_episode_stats["inference_calls"] += 1
        
        if log_extras:
            tokens_prompt = log_extras.get("tokens_prompt", 0)
            tokens_completion = log_extras.get("tokens_completion", 0)
            tokens_total = log_extras.get("tokens_total", 0)
            
            # If total is not provided but parts are
            if tokens_total == 0 and (tokens_prompt > 0 or tokens_completion > 0):
                tokens_total = tokens_prompt + tokens_completion
                
            self._stats["total_input_tokens"] += tokens_prompt
            self._stats["total_output_tokens"] += tokens_completion
            self._stats["total_tokens"] += tokens_total
            
            self._current_episode_stats["input_tokens"] += tokens_prompt
            self._current_episode_stats["output_tokens"] += tokens_completion
            self._current_episode_stats["tokens"] += tokens_total

        # Log raw request if prompt is available
        if self._requests_log_path and log_extras and "prompt" in log_extras:
            try:
                with open(self._requests_log_path, "a", encoding="utf-8") as f:
                    record = {
                        "step": self._step_count,
                        "prompt": log_extras["prompt"],
                        "response": log_extras.get("output_text", ""),
                        "action": action,
                        "tokens": {
                            "prompt": log_extras.get("tokens_prompt", 0),
                            "completion": log_extras.get("tokens_completion", 0),
                            "total": log_extras.get("tokens_total", 0)
                        }
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to log raw request: {e}")
                raise ValueError(f"Failed to log raw request: {traceback.format_exc()}")

        if self.wandb_config and self.wandb_config.enabled:
            log_data = {
                "step": self._step_count,
                "score": current_score,
                "score_delta": current_score - self._last_score,
                "action": action,
            }
            
            # Add game specific metrics from game_info
            # We can log everything in game_info that is a number
            for k, v in game_info.items():
                if isinstance(v, (int, float)):
                    log_data[f"game_info/{k}"] = v
            
            # Add custom metrics from subclass
            custom_metrics = self.calculate_metrics(game_info)
            log_data.update(custom_metrics)
            
            # Add extras from get_action
            if log_extras:
                # Filter out prompt/output_text from wandb log to avoid clutter if they are huge
                # But keep tokens and reasoning length
                for k, v in log_extras.items():
                    if k not in ["prompt", "output_text"]:
                        log_data[k] = v

            # Log action distribution
            log_data[f"action/{action}"] = 1
            
            # Log obs_str as text
            if cur_state_str:
                log_data["obs_str"] = wandb.Html(f"<pre>{cur_state_str}</pre>")
            
            # Log obs_image if available
            if obs_image is not None:
                try:
                    log_data["obs_image"] = wandb.Image(obs_image, caption=f"Step {self._step_count}")
                except Exception as e:
                    # If image logging fails, just continue
                    logger.warning(f"Warning: Could not log image: {traceback.format_exc()}")
            
            wandb.log(log_data, step=self._step_count)

        self._prev_state_str = cur_state_str
        self._last_action = action
        self._last_score = current_score

        return action

    def get_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Get action from LLM.
        Common implementation that delegates to _get_action.
        """
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")
        obs_image = obs.get("obs_image", None)
        
        task_description = game_info.get("task_description", "")
        
        action, reasoning, output_text, usage, prompt = self._get_action(
            task_description=task_description,
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

    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, str, str, Any, str]:
        """
        Get action from LLM.
        This method should be overridden by subclasses and often decorated with @weave.op().
        Returns: (action, reasoning, output_text, usage, prompt)
        """
        raise NotImplementedError

    def calculate_metrics(self, game_info: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate custom metrics based on game info.
        Override this in subclasses.
        """
        return {}

    def __del__(self):
        """Cleanup wandb on agent destruction."""
        if hasattr(self, "wandb_config") and self.wandb_config and self.wandb_config.enabled:
            try:
                wandb.finish()
            except:
                pass

    # ===== Checkpoint Methods =====
    
    def get_state(self) -> dict[str, Any]:
        """
        Get agent state for checkpointing.
        
        Subclasses should override this to add their specific state,
        calling super().get_state() first to include base state.
        
        Returns:
            Dictionary containing agent state.
        """
        return {
            "stats": self._stats,
            "episode_stats": self._episode_stats,
            "current_episode_stats": self._current_episode_stats,
            "step_count": self._step_count,
            "last_score": self._last_score,
            "prev_state_str": self._prev_state_str,
            "last_action": self._last_action,
        }
    
    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load agent state from checkpoint.
        
        Subclasses should override this to restore their specific state,
        calling super().load_state(state) first to restore base state.
        
        Args:
            state: State dictionary from get_state()
        """
        self._stats = state.get("stats", self._stats)
        self._episode_stats = state.get("episode_stats", [])
        self._current_episode_stats = state.get(
            "current_episode_stats",
            {"inference_calls": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0}
        )
        self._step_count = state.get("step_count", 0)
        self._last_score = state.get("last_score", 0)
        self._prev_state_str = state.get("prev_state_str", "N/A")
        self._last_action = state.get("last_action", "No action yet")
        
        logger.info(f"Loaded agent state: {self._step_count} steps, {len(self._episode_stats)} episodes")
    
    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """
        Get metadata for checkpoint summary.
        
        Subclasses can override to add their own metadata,
        calling super().get_checkpoint_metadata() to include base metadata.
        
        Returns:
            Metadata dictionary with summary information.
        """
        return {
            "agent_class": self.__class__.__name__,
            "total_steps": self._step_count,
            "total_episodes": len(self._episode_stats),
            "total_inference_calls": self._stats.get("total_inference_calls", 0),
            "total_tokens": self._stats.get("total_tokens", 0),
        }

