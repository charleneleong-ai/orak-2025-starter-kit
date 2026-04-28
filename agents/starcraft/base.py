import traceback
import base64
import io
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from agents.base import BaseOrakAgent
from agents._harness import with_retries
import weave

GAME_RULES = """
### StarCraft II Game Rules ###

## Core Gameplay
- You are playing as **Protoss** against a **Zerg** opponent
- Win by managing economy, production, and combat to defeat the enemy
- Game progresses in real-time with resource accumulation and unit production

## Resource System
- **Minerals**: Primary resource for all actions, gathered by Probes
- **Vespene Gas**: Secondary resource for advanced units/upgrades, gathered by Probes from Assimilators
- **Supply System**:
  - Supply Cap: Maximum population (increased by building Pylons)
  - Supply Used: Current population from units
  - Supply Left: Available space (Cap - Used)
- **Worker Supply**: Number of Probe workers (gather resources)
- **Army Supply**: Population consumed by military units

## Action Execution
- Each game step requires **EXACTLY 5 actions**
- Actions execute **sequentially** in the order provided
- Actions have costs and prerequisites that must be satisfied:
  - Sufficient minerals/gas for costs
  - Required buildings must exist (e.g., Gateway needs Pylon first)
  - Available supply for unit production

## Action Categories
1. **TRAIN**: Produce units from buildings (TRAIN PROBE, TRAIN ZEALOT, etc.)
2. **BUILD**: Construct buildings (BUILD PYLON, BUILD GATEWAY, etc.)
3. **RESEARCH**: Unlock upgrades (RESEARCH CHARGE, RESEARCH WARPGATERESEARCH, etc.)
4. **SCOUTING**: Send units to explore (SCOUTING PROBE, SCOUTING OBSERVER, etc.)
5. **COMBAT**: Army commands (MULTI-ATTACK, MULTI-RETREAT)
6. **CHRONOBOOST**: Accelerate production/research at buildings
7. **EMPTY ACTION**: No operation (useful when waiting for resources)

## Strategic Considerations
- **Early Game**: Focus on economy (build Probes, Pylons, expand with Nexus)
- **Mid Game**: Balance production, tech, and army composition
- **Late Game**: Maintain production, upgrades, and army control
- **Build Order**: Pylons provide supply and power; always maintain supply buffer
- **Tech Tree**: Research units require specific buildings (e.g., Stalker needs Cybernetics Core)
"""

SYSTEM_PROMPT = f"""
You are an AI assistant playing StarCraft II as the Protoss race against a Zerg opponent.
Your goal is to defeat the enemy by managing your economy, production, and military forces effectively.

{GAME_RULES}

### Decision Output Format ###
Analyze the game state and determine the **5 most optimal actions** to take.

You must respond with a structure containing:
- "reasoning": Detailed strategic analysis explaining why these actions were chosen (analyze resources, build order, timing, and game plan)
- "current_goal": Your immediate strategic objective (e.g., "Expand economy", "Build army", "Tech up", "Attack")
- "actions": A list of exactly 5 action names from the ACTION_DICTIONARY

**IMPORTANT**: 
- Actions must be valid action names from ACTION_DICTIONARY
- Each action must be affordable with current/projected resources
- Actions execute in sequence, so consider resource costs accumulating
- Use "EMPTY ACTION" if waiting for resources or no good action available
"""

USER_PROMPT_TEMPLATE = """
### Game Info
- Player Race: {player_race}
- Enemy Race: {enemy_race}
- Available Actions: {num_actions} per turn

### Last Actions Taken
{last_action}

### Current Game State
{cur_state_str}

### ACTION_DICTIONARY
{action_dict}
"""


class StarCraftAction(BaseModel):
    """Structured output for StarCraft II game actions"""

    reasoning: str = Field(
        description="Detailed strategic explanation of why these actions were chosen"
    )
    current_goal: str = Field(
        description="Current strategic objective (e.g., 'Expand economy', 'Build army', 'Attack')"
    )
    actions: list[str] = Field(
        description="List of exactly 5 action names from ACTION_DICTIONARY",
        min_length=5,
        max_length=5,
    )


class StarCraftAgent(BaseOrakAgent):

    _llm: Optional[BaseChatModel] = PrivateAttr(default=None)
    _action_counts: dict[str, int] = PrivateAttr(default_factory=dict)

    def calculate_metrics(self, game_info: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate custom metrics based on game info.
        """
        metrics = {}

        # Extract relevant metrics from game state
        for key in [
            "minerals",
            "vespene",
            "supply_cap",
            "supply_used",
            "supply_left",
            "worker_supply",
            "army_supply",
            "game_time",
        ]:
            if key in game_info:
                metrics[key] = game_info[key]

        # Use server provided evaluation score if available
        if "evaluation_score" in game_info:
            metrics["evaluation_score"] = float(game_info["evaluation_score"])

        # Add action distribution counts (top 10 most used)
        sorted_actions = sorted(
            self._action_counts.items(), key=lambda x: x[1], reverse=True
        )
        for action, count in sorted_actions[:10]:
            # Sanitize action name for wandb (replace spaces with underscores)
            safe_action = action.replace(" ", "_").replace("-", "_")
            metrics[f"action_count/{safe_action}"] = count

        return metrics

    @weave.op()
    def _get_action(
        self, task_description: str, cur_state_str: str, obs_image: Any = None
    ) -> tuple[str, str, str, Any, str]:
        """Get action from LLM. This method is tracked by Weave for observability."""

        if not self._llm:
            raise ValueError("LLM not initialized")

        # Get game info from observation
        game_info = {}
        if hasattr(self, "_current_obs"):
            game_info = self._current_obs.get("game_info", {})

        action_dict = game_info.get("action_dict", {})
        player_race = game_info.get("player_race", "Protoss")
        enemy_race = game_info.get("enemy_race", "Zerg")
        num_actions = game_info.get("num_actions", 5)

        prompt_text = USER_PROMPT_TEMPLATE.format(
            player_race=player_race,
            enemy_race=enemy_race,
            num_actions=num_actions,
            last_action=self._last_action,
            cur_state_str=cur_state_str,
            action_dict=action_dict,
        )

        if task_description:
            prompt_text = f"### Task\n{task_description}\n\n" + prompt_text

        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        user_content = []
        user_content.append({"type": "text", "text": prompt_text})

        if obs_image:
            # Convert PIL to base64
            buffered = io.BytesIO()
            obs_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{img_str}"
            user_content.append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        messages.append(HumanMessage(content=user_content))

        structured_llm = self._llm.with_structured_output(StarCraftAction)

        usage = None
        output_text = ""

        try:
            # Invoke LLM with structured output
            response = with_retries(lambda: structured_llm.invoke(messages), label="starcraft.llm")

            reasoning = response.reasoning
            current_goal = response.current_goal
            actions_list = response.actions

            # Track action usage
            for action in actions_list:
                if action not in self._action_counts:
                    self._action_counts[action] = 0
                self._action_counts[action] += 1

            # Format actions as numbered list
            action_lines = [
                f"{i+1}: {action}" for i, action in enumerate(actions_list)
            ]
            action = "\n".join(action_lines)

            output_text = f"Goal: {current_goal}\n\nReasoning: {reasoning}\n\nActions:\n{action}"

        except Exception as e:
            logger.error(f"Error invoking LLM after retries: {traceback.format_exc()}")
            self._mark_fallback(f"llm_error: {type(e).__name__}: {str(e)[:200]}")
            # Fallback actions
            action = "\n".join(
                [
                    "1: TRAIN PROBE",
                    "2: BUILD PYLON",
                    "3: EMPTY ACTION",
                    "4: EMPTY ACTION",
                    "5: EMPTY ACTION",
                ]
            )
            reasoning = f"Error occurred: {e}"
            current_goal = "Recover from error"
            output_text = str(e)

        return action, reasoning, current_goal, output_text, usage, prompt_text

    def get_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Override to store current observation for use in _get_action.
        """
        # Store observation temporarily for _get_action to access
        self._current_obs = obs

        # Call parent implementation
        result = super().get_action(obs)

        # Clean up
        if hasattr(self, "_current_obs"):
            delattr(self, "_current_obs")

        return result
