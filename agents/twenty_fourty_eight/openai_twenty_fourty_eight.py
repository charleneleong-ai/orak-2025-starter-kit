import openai
import re
import wandb
import weave
import io
import base64
import json
from typing import Any, ClassVar
from pydantic import PrivateAttr
from loguru import logger

from config.agent_config import OpenAIConfig
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

You must respond with a JSON object containing:
- "reasoning": A detailed explanation of why this action was chosen
- "action": The action to take (must be one of: up, down, left, or right)
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
"""


class OpenAITwentyFourtyEightAgent(TwentyFourtyEightAgent):
    config: OpenAIConfig
    
    _client: openai.OpenAI = PrivateAttr()
    _is_reasoning_model: bool = PrivateAttr(default=False)
    
    def __init__(
        self, 
        config: OpenAIConfig = None, 
        wandb_config: WandbConfig = None,
    ):
        # Load configurations
        config = config or OpenAIConfig()
        wandb_config = wandb_config or WandbConfig()
        
        # Initialize with Weave Model
        super().__init__(
            config=config,
            wandb_config=wandb_config
        )
        
        # Initialize OpenAI client
        self._client = openai.OpenAI(api_key=self.config.api_key)
        
        # Detect if this is a reasoning model (o1, o3, gpt-5, etc.)
        # These models use the responses API instead of chat completions
        model_lower = self.config.model.lower()
        self._is_reasoning_model = any(keyword in model_lower for keyword in ['o1', 'o3', 'gpt-5'])
        
        logger.info(f"Initialized OpenAI agent with model: {self.config.model}, using reasoning API: {self._is_reasoning_model}")

    @property
    def AGENT_TAGS(self):
        return ["openai"]

    def _parse_action_from_text(self, text: str) -> tuple[str, str]:
        """Parse action and reasoning from text response.
        
        Returns:
            tuple[str, str]: (action, reasoning)
        """
        # Try to parse as JSON first
        try:
            # Look for JSON object in the text
            json_match = re.search(r'\{[^}]*"action"[^}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                action = data.get("action", "").lower()
                reasoning = data.get("reasoning", text)
                if action in ["left", "right", "up", "down"]:
                    return action, reasoning
        except:
            pass
        
        # Fallback: look for action keywords in text
        text_lower = text.lower()
        for action in ["left", "right", "up", "down"]:
            if f'"{action}"' in text_lower or f"'{action}'" in text_lower or f"action: {action}" in text_lower:
                return action, text
        
        # Last resort: find first occurrence of an action word
        for action in ["left", "right", "up", "down"]:
            if action in text_lower:
                return action, text
        
        # Default fallback
        return "left", text

    @weave.op()
    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, str, str, Any]:
        """Get action from LLM. This method is tracked by Weave for observability."""
        
        prompt_text = USER_PROMPT.format(
            task_description=task_description,
            prev_state_str=self._prev_state_str, 
            action=self._last_action, 
            cur_state_str=cur_state_str
        )

        if self._is_reasoning_model:
            # Use responses API for reasoning models (o1, o3, gpt-5)
            # These models don't support system messages or structured outputs
            # Combine system and user prompts
            combined_prompt = f"{SYSTEM_PROMPT}\n\n{prompt_text}"
            
            # Build the input based on whether we have an image
            if obs_image:
                # Convert PIL to base64
                buffered = io.BytesIO()
                obs_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{img_str}"
                
                user_content = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": combined_prompt},
                            {"type": "input_image", "image_url": image_url}
                        ]
                    }
                ]
            else:
                user_content = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": combined_prompt}
                        ]
                    }
                ]
            
            # Call responses API
            api_params = {
                "model": self.config.model,
                "input": user_content,
            }
            
            # Add reasoning parameter if configured
            if hasattr(self.config, "reasoning_effort") and self.config.reasoning_effort:
                api_params["reasoning"] = {"effort": self.config.reasoning_effort}
            
            response = self._client.responses.create(**api_params)
            
            # Extract response text
            output_text = response.output_text
            usage = response.usage if hasattr(response, 'usage') else None
            
            # Parse action and reasoning from text
            action, reasoning = self._parse_action_from_text(output_text)
            
        else:
            # Use chat completions API for standard models (gpt-4o, etc.)
            user_content = [{"type": "text", "text": prompt_text}]

            if obs_image:
                # Convert PIL to base64
                buffered = io.BytesIO()
                obs_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{img_str}"
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})

            # Use Structured Outputs (parse)
            response = self._client.beta.chat.completions.parse(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format=GameAction,
            )
            
            parsed_response = response.choices[0].message.parsed
            usage = response.usage
            
            action = parsed_response.action.lower()
            reasoning = parsed_response.reasoning
            output_text = ""
        
        # Validate action
        if action not in ["left", "right", "up", "down"]:
            logger.warning(f"Invalid action '{action}', defaulting to 'left'")
            action = "left"
            
        return action, reasoning, output_text, usage
