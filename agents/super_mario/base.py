import base64
import io
import traceback
from typing import Any

import weave
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from agents._harness import structured_invoke_with_usage, with_retries
from agents.base import BaseOrakAgent

GAME_RULES = """
### Super Mario Bros Game Rules ###
- *Reach the Flag*: Navigate through the level and jump through the flagpole line before time runs out
- *Avoid Enemies*: Defeat or bypass enemies using jumps or power-ups
- *Collect Power-ups*: Gain abilities by collecting mushrooms and flowers. When powered-up, collisions with enemies reduce size instead of causing death
- *Preserve Lives*: Avoid hazards such as pits and enemies to stay alive

### Object Descriptions ###
- Bricks: Breakable blocks; may contain items or coins (Size: 16x16)
- Question Blocks: Reveal coins or power-ups when hit; deactivate after use (Size: 16x16)
- Pit: Falling in results in losing a life
- Warp Pipe: Raised above the ground, so Mario must jump over them when it appear in front (Size: 30xHeight(y))
- Monster Goomba: Basic enemy; can be defeated by jumping on it (Size: 16x16)
- Monster Koopa: Turtle enemy; retreats into shell when jumped on (Size: 20x24)
- Item Mushroom: Grows Mario larger, grants protection (Size: 16x16)
- Stairs: Used to ascend/descend terrain
- Flag: Touch to complete the level
- Ground: the ground level in the game is y=32

### Action Descriptions ###
- Mario (Size: 15x13) continuously moves to the right at a fixed speed
- You must choose an appropriate jump level to respond to upcoming obstacles
- Each jump level determines both:
    - How far Mario jumps horizontally (x distance)
    - How high Mario reaches at the peak of the jump (y height)
- Jump Levels *(values based on flat ground jumps)*:
    - Level 0: +0 in x, +0 in y (No jump, just walk)
    - Level 1: +42 in x, +35 in y
    - Level 2: +56 in x, +46 in y
    - Level 3: +63 in x, +53 in y
    - Level 4: +70 in x, +60 in y
    - Level 5: +77 in x, +65 in y
    - Level 6: +84 in x, +68 in y
    - *Note*: The values above assume Mario is jumping from flat ground. When jumping from elevated platforms or interacting with mid-air obstacles (e.g., bricks), the actual jump trajectory and landing position may vary.
- The key is choosing the *right jump level at the right moment*
- *Use higher levels* to jump over taller or farther obstacles
- Consider *the size* of Mario and objects
- While jumping, Mario follows a *parabolic arc*, moving upward and then downward in a smooth curve, so Mario can be *blocked by objects mid-air or be defeated by airborne enemies*
- Mario can step on top of bricks, blocks, warp pipes, and stairs
"""

SYSTEM_PROMPT = f"""
You are an AI assistant playing the Super Mario game. Your goal is to reach the flagpole at the end of each level without dying by avoiding obstacles, collecting power-ups, and defeating/avoiding enemies.

{GAME_RULES}

### Decision Output Format ###
Analyze the provided game state and determine the **single most optimal action** to take next.

You must respond with a structure containing:
- "reasoning": A detailed explanation of why this action was chosen (e.g. "Goomba is 40 units ahead on flat ground — a level 1 jump (+42 X) is sufficient to jump over or stomp it.")
- "jump_level": The jump level to take (integer from 0 to 6)
"""

USER_PROMPT_TEMPLATE = """
### Last Executed Action
{last_action}

### Current Game State
{cur_state_str}
"""


class GameAction(BaseModel):
    """Structured output for Super Mario game actions"""

    reasoning: str = Field(description="Detailed explanation of why this action was chosen")
    jump_level: int = Field(description="The jump level to take: 0 to 6", ge=0, le=6)


class SuperMarioAgent(BaseOrakAgent):
    _llm: BaseChatModel | None = PrivateAttr(default=None)
    _jump_level_counts: dict[int, int] = PrivateAttr(
        default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    )

    def calculate_metrics(self, game_info: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate custom metrics based on game info.
        """
        metrics = {}

        # Extract relevant metrics
        for key in ["coins", "lives", "time", "world", "stage", "x_pos"]:
            if key in game_info:
                metrics[key] = game_info[key]

        ## Calculate evaluation score (Normalsed Progress) to align with server
        # use server provided evaluation score if available to ensure alignment
        if "evaluation_score" in game_info:
            metrics["evaluation_score"] = float(game_info["evaluation_score"])
        else:
            x_pos = float(game_info.get("x_pos", 40))
            x_start = 40
            x_flag = 3161

            metrics["evaluation_score"] = (x_pos - x_start) / (x_flag - x_start) * 100.0

        metrics["score"] = float(game_info.get("score", 0))

        return metrics

    @weave.op()
    def _get_action(
        self, task_description: str, cur_state_str: str, obs_image: Any = None
    ) -> tuple[str, str, str, Any, str]:
        """Get action from LLM. This method is tracked by Weave for observability."""

        if not self._llm:
            raise ValueError("LLM not initialized")

        prompt_text = USER_PROMPT_TEMPLATE.format(
            last_action=self._last_action, cur_state_str=cur_state_str
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
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        messages.append(HumanMessage(content=user_content))

        usage = None
        output_text = ""

        try:
            response, usage = with_retries(
                lambda: structured_invoke_with_usage(self._llm, messages, GameAction),
                label="super_mario.llm",
            )

            jump_level = response.jump_level
            reasoning = response.reasoning

            # Track jump level usage
            self._jump_level_counts[jump_level] += 1

            action = f"Jump Level: {jump_level}"

            output_text = f"Action: {action}\nReasoning: {reasoning}"

        except Exception as e:
            logger.error(f"Error invoking LLM after retries: {traceback.format_exc()}")
            self._mark_fallback(f"llm_error: {type(e).__name__}: {str(e)[:200]}")
            # Default fallback action
            action = "Jump Level: 0"
            reasoning = f"Error: {e}"
            output_text = str(e)

        return action, reasoning, output_text, usage, prompt_text
