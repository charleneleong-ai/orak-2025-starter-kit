import re
from typing import Any

import weave
from loguru import logger

from agents.macla.base import BaseMaclaAgent
from agents.twenty_fourty_eight.base import TwentyFourtyEightAgent
from config.agent_config import GeminiConfig, OpenAIConfig
from config.base import WandbConfig


class TwentyFourtyEightMaclaAgent(BaseMaclaAgent, TwentyFourtyEightAgent):
    config: GeminiConfig | OpenAIConfig

    def __init__(
        self,
        config: GeminiConfig | OpenAIConfig = None,
        wandb_config: WandbConfig = None,
    ):
        config = config or GeminiConfig()
        wandb_config = wandb_config or WandbConfig()

        TwentyFourtyEightAgent.__init__(self, config=config, wandb_config=wandb_config)

        self._init_macla_agent()

    @property
    def AGENT_TAGS(self):
        tags = ["2048", "macla"]
        if isinstance(self.config, GeminiConfig):
            tags.extend(["gemini", self.config.model, "vertex-ai"])
        elif isinstance(self.config, OpenAIConfig):
            tags.extend(["openai", self.config.model])
        return tags

    @weave.op()
    def _get_action(
        self, task_description: str, cur_state_str: str, obs_image: Any = None
    ) -> tuple[str, str, str, Any, str, str, str]:
        """Get action from LLM. Tracked by Weave for observability."""
        # 1. Provide feedback on previous execution
        update_info = self._provide_feedback(self._prev_state_str, cur_state_str)

        # 2. Execute task using MACLA
        action, execution_result, memory_stats = self._execute_task(
            task_description, cur_state_str, obs_image
        )

        # 3. Log MACLA stats
        self._log_stats(update_info, memory_stats)

        # 4. Validate action
        action = self._validate_action(action)

        # 5. Build output
        avg_exec_time = (
            sum(self._execution_times) / len(self._execution_times) if self._execution_times else 0
        )
        reasoning = f"MACLA Strategy: {execution_result.get('method', 'unknown')}. Confidence: {execution_result.get('confidence', 0.0)}. Avg Time (last 10): {avg_exec_time:.4f}s"
        output_text = f"Result: {execution_result}\nStats: {memory_stats}"
        goal = self._get_task_description({"task_description": task_description})

        game_phase, _ = self._determine_game_phase(cur_state_str)

        return (
            action,
            reasoning,
            output_text,
            memory_stats,
            f"Goal: {goal}\nObs: {cur_state_str}",
            game_phase,
            self._last_update_type,
        )

    def _base_fallback(self, goal: str, observation: str, **kwargs) -> list[str]:
        """
        2048-specific fallback that calls the base TwentyFourtyEightAgent LLM logic.
        """
        obs_image = kwargs.get("obs_image")
        try:
            # Call the parent TwentyFourtyEightAgent implementation
            action, _, _, _, _, _, _ = TwentyFourtyEightAgent._get_action(
                self, task_description=goal, cur_state_str=observation, obs_image=obs_image
            )
            return [action]
        except Exception as e:
            logger.error(f"Base fallback failed: {e}")
            return ["left"]

    def _validate_action(self, action: str) -> str:
        """Validate 2048 action."""
        if action not in ["left", "right", "up", "down"]:
            logger.warning(f"Invalid action '{action}', defaulting to 'left'")
            return "left"
        return action

    def _get_task_description(self, game_info: dict) -> str:
        """Get 2048 task description."""
        return game_info.get("task_description", "Merge tiles to reach 2048")

    def _get_default_goal(self) -> str:
        """Default goal for 2048."""
        return "Merge tiles to reach 2048"

    def _get_default_action(self) -> str:
        """Default action for 2048."""
        return "left"

    def _detect_success(
        self, execution_result: dict, prev_state_str: str, cur_state_str: str
    ) -> tuple[bool, bool]:
        """
        2048-specific success detection based on score improvement.
        """
        # Parse scores
        prev_score = 0
        cur_score = 0
        try:
            p_match = re.search(r"Score:\s*(\d+)", prev_state_str)
            c_match = re.search(r"Score:\s*(\d+)", cur_state_str)
            if p_match:
                prev_score = int(p_match.group(1))
            if c_match:
                cur_score = int(c_match.group(1))
        except:
            pass

        # Check if game ended (new episode started)
        is_fatal_game_over = self._steps_in_current_episode == 0

        # Success = score improved and didn't end game
        strong_success = (cur_score > prev_score) and not is_fatal_game_over

        return strong_success, is_fatal_game_over

    def _extract_context(self, observation: str | list) -> dict[str, Any]:
        """Extract task-relevant context from 2048 observation."""
        if isinstance(observation, list):
            observation = "\n".join(str(item) for item in observation)
        context = {}
        try:
            score_match = re.search(r"Score:\s*(\d+)", observation)
            if score_match:
                context["score"] = int(score_match.group(1))

            grid_match = re.search(r"Grid:\s*\n((?:.*\n){4})", observation)
            if grid_match:
                context["grid"] = grid_match.group(1).strip()
        except Exception as e:
            logger.warning(f"Failed to extract context: {e}")

        return context

    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        """Extract preconditions from 2048 observation."""
        preconditions = []
        if context_key:
            preconditions.append(f"board_state={context_key}")
        return preconditions

    def extract_postconditions(self, observation: str) -> dict[str, Any]:
        """Extract postconditions (outcomes) from 2048 observation."""
        # For 2048, postconditions are similar to context (current state)
        return self._extract_context(observation)

    def record_episode_end(self, episode, game_name, seed, score):
        """Called by runner at the end of an episode."""
        TwentyFourtyEightAgent.record_episode_end(self, episode, game_name, seed, score)
        self._record_episode_end(episode, score)
