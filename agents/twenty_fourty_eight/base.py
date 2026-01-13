import traceback
import base64
import io
from typing import Any,  Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from agents.base import BaseOrakAgent


import weave



GAME_RULES = """
### 2048 Game Rules ### 
1. The game is played on a 4×4 grid. Tiles slide in one of four directions: 'up', 'down', 'left', or 'right'. 
2. Only two **consecutive tiles** with the SAME value can merge. Merges cannot occur across empty tiles. 
3. **Merging is directional**: 
   - Row-based merges occur on 'left' or 'right' actions. 
   - Column-based merges occur on 'up' or 'down' actions. 
4. **All tiles first slide in the chosen direction as far as possible**, then merges are applied. 
5. **A tile can merge only once per move**. When multiple same-value tiles are aligned (e.g., [2, 2, 2, 2]), merges proceed from the movement direction. For example: 
   - [2, 2, 2, 2] with 'left' results in [4, 4, 0, 0]. 
   - [2, 2, 2, 0] with 'left' results in [4, 2, 0, 0]. 
6. An action is only valid if it causes at least one tile to slide or merge. Otherwise, the action is ignored, and no new tile is spawned. 
7. After every valid action, a new tile (usually **90 percent chance of 2, 10 percent chance of 4**) appears in a random empty cell. 
8. The game ends when the board is full and no valid merges are possible. 
9. Score increases only when merges occur, and the increase equals the value of the new tile created from the merge. 
"""

SYSTEM_PROMPT = f"""
You are an expert AI agent specialized in playing the 2048 game with advanced strategic reasoning. 
Your primary goal is to achieve the highest possible tile value while maintaining long-term playability by preserving the flexibility of the board and avoiding premature game over. 

{GAME_RULES}

### Decision Output Format ### 
Analyze the provided game state and determine the **single most optimal action** to take next. 

You must respond with a structure containing:
- "reasoning": A detailed explanation of why this action was chosen
- "action": The action to take (must be one of: up, down, left, or right)
"""

USER_PROMPT_TEMPLATE = """
### Target task
{task_description}

### Previous state
{prev_state_str}

### Last executed action
{action}

### Current state
{cur_state_str}
"""

class GameAction(BaseModel):
    """Structured output for 2048 game actions"""
    reasoning: str = Field(description="Detailed explanation of why this action was chosen")
    action: str = Field(description="The action to take: up, down, left, or right")

class TwentyFourtyEightAgent(BaseOrakAgent):
    
    _llm: Optional[BaseChatModel] = PrivateAttr(default=None)

    def calculate_metrics(self, game_info: dict[str, Any]) -> dict[str, Any]:
        current_game_score = int(float(game_info.get("score", 0)))
        try:
            max_tile = int(game_info.get("max_tile", 0))
        except (ValueError, TypeError):
            max_tile = 0
        return {
            "evaluation_score": min((current_game_score / 20000) * 100, 100),
            "max_tile": max_tile
        }

    @weave.op()
    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, str, str, Any, str]:
        """Get action from LLM. This method is tracked by Weave for observability."""
        
        if not self._llm:
            raise ValueError("LLM not initialized")

        prompt_text = USER_PROMPT_TEMPLATE.format(
            task_description=task_description,
            prev_state_str=self._prev_state_str, 
            action=self._last_action, 
            cur_state_str=cur_state_str
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]

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
        
        # Invoke LLM
        structured_llm = self._llm.with_structured_output(GameAction)
        
        usage = None
        output_text = ""
        
        try:
            # Note: with_structured_output returns the data model directly
            response = structured_llm.invoke(messages)
            
            action = response.action.lower()
            reasoning = response.reasoning
            output_text = f"Action: {action}\nReasoning: {reasoning}"
            
        except Exception as e:
            logger.error(f"Error invoking LLM: {traceback.format_exc()}")
            action = "left"
            reasoning = f"Error: {e}"
            output_text = str(e)
            raise ValueError(f"LLM invocation failed: {traceback.format_exc()}")
        
        # Validate action
        if action not in ["left", "right", "up", "down"]:
            logger.warning(f"Invalid action '{action}', defaulting to 'left'")
            action = "left"
            
        return action, reasoning, output_text, usage, prompt_text
