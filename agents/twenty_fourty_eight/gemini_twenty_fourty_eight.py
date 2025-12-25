from typing import ClassVar, Any, Tuple
from pydantic import PrivateAttr
import re
import wandb
import weave
import io
import base64
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from config.agent_config import GeminiConfig
from config.base import WandbConfig
from agents.twenty_fourty_eight.base import TwentyFourtyEightAgent, GameAction

SYSTEM_PROMPT = """
You are an expert AI agent specialized in playing the 2048 game with advanced strategic reasoning. 
Your primary goal is to achieve the highest possible tile value while maintaining long-term playability by preserving the flexibility of the board and avoiding premature game over. 

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

### Decision Output Format ### 
Analyze the provided game state and determine the **single most optimal action** to take next. 
Return your decision in the following exact format: 
### Reasoning
<a detailed summary of why this action was chosen>
### Actions
<up, right, left, or down>

Ensure that: 
- The '### Reasoning' field provides a clear explanation of why the action is the best choice, including analysis of current tile positions, merge opportunities, and future flexibility. 
- The '### Actions' field contains only one of the four valid directions. 
"""

USER_PROMPT = """
### Target task
{task_description}

### Previous state
{prev_state_str}

### Last executed action
{action}

### Current state
{cur_state_str}

You should only respond in the format described below, and you should not output comments or other information.
Provide your response in the strict format: 
### Reasoning
<a detailed summary of why this action was chosen>
### Actions
<direction>
"""


class GeminiTwentyFourtyEightAgent(TwentyFourtyEightAgent):
    config: GeminiConfig
    _llm: Any = PrivateAttr()

    def __init__(
        self, 
        config: GeminiConfig = None, 
        wandb_config: WandbConfig = None,
    ):
        config = config or GeminiConfig()
        wandb_config = wandb_config or WandbConfig()

        super().__init__(
            config=config,
            wandb_config=wandb_config,
        )
        
        self._llm = ChatVertexAI(
            model_name=self.config.model,
            temperature=self.config.temperature,
            project=self.config.gcp_project,
            location=self.config.gcp_location,
        ).with_structured_output(GameAction)
        
    @property
    def AGENT_TAGS(self):
        return ["gemini", "vertex-ai"]

    @weave.op()
    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> Tuple[str, str, str, Any, str]:
        """Get action from LLM. This method is tracked by Weave for observability."""
        prompt = USER_PROMPT.format(
            task_description=task_description,
            prev_state_str=self._prev_state_str, 
            action=self._last_action, 
            cur_state_str=cur_state_str
        )

        content = [
            {"type": "text", "text": prompt}
        ]

        if obs_image:
             # Convert PIL to base64
             buffered = io.BytesIO()
             obs_image.save(buffered, format="JPEG")
             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
             image_url = f"data:image/jpeg;base64,{img_str}"
             content.append({"type": "image_url", "image_url": {"url": image_url}})

        # Create messages with system and user prompts
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content)
        ]
        
        # Invoke the model - Weave will automatically track this
        response = self._llm.invoke(messages)
        output_text = response.content if hasattr(response, 'content') else str(response)

        # Parse the reasoning
        reasoning = response.reasoning if hasattr(response, 'reasoning') else self._parse_reasoning(output_text)
        
        action = response.action if hasattr(response, 'action') else self._parse_actions(output_text.strip())
        if action not in ["left", "right", "up", "down"]:
            action = "left"  # Fall back to left if the action is not valid

        return action, reasoning, output_text, None, prompt # Usage not available in this implementation easily

    def _parse_reasoning(self, output):
        """Extract reasoning section from output."""
        reasoning_match = re.search(r"### Reasoning\s*\n(.+?)(?=### Actions|$)", output, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        return ""

    def _parse_actions(self, output):
        """Return the full string after ### Actions."""
        actions_match = re.search(r"### Actions\s*\n(.+)", output, re.IGNORECASE | re.DOTALL)
        if actions_match:
            actions_section = actions_match.group(1).strip()
            return actions_section
        return ""
