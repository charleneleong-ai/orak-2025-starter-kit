import openai
import re
import wandb

from config.agent_config import OpenAIConfig
from config.base import WandbConfig

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


class OpenAITwentyFourtyEightAgent:
    TRACK = "TRACK1"
    
    def __init__(self, config: OpenAIConfig = None, wandb_config: WandbConfig = None):
        # Load configurations
        self.config = config or OpenAIConfig()
        self.wandb_config = wandb_config or WandbConfig()
        self.wandb_config.tags.extend(["openai"])
        
        # Initialize wandb
        if self.wandb_config.enabled:
            wandb.init(
                project=self.wandb_config.project,
                entity=self.wandb_config.entity,
                config=self.config.to_dict(),
                tags=self.wandb_config.tags,
                name=None, 
            )
        
        # Initialize OpenAI client
        self.client = openai.OpenAI()
        
        self.prev_state_str = "N/A"
        self.last_action = "No action yet"
        self.step_count = 0
        self.last_score = 0

    def act(self, obs):
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")
        
        # Extract metrics from observation
        info = obs.get("info", {})
        current_score = int(info.get("score", 0)) if info.get("score") else 0
        max_tile = int(info.get("max_tile", 0)) if info.get("max_tile") else 0
        
        self.step_count += 1

        response = self.client.responses.create(
            model=self.config.model,
            input=USER_PROMPT.format(
                task_description=game_info.get("task_description", ""),
                prev_state_str=self.prev_state_str, 
                action=self.last_action, 
                cur_state_str=cur_state_str
            ),
            instructions=SYSTEM_PROMPT,
            reasoning={
                "effort": self.config.reasoning_effort,
            }
        )
        
        output_text = response.output_text.strip()
        
        # Parse the reasoning
        reasoning = self._parse_reasoning(output_text)
        
        action = self._parse_actions(output_text)
        if action not in ["left", "right", "up", "down"]:
            action = "left"  # Fall back to left if the action is not valid

        # Log to wandb
        if self.wandb_config.enabled:
            log_data = {
                "step": self.step_count,
                "score": current_score,
                "max_tile": max_tile,
                "action": action,
                "score_delta": current_score - self.last_score,
            }
            
            # Log action distribution
            log_data[f"action/{action}"] = 1
            
            # Log reasoning length
            if reasoning:
                log_data["reasoning_length"] = len(reasoning)
            
            # Log API usage if available
            if hasattr(response, 'usage'):
                log_data["tokens_prompt"] = getattr(response.usage, 'prompt_tokens', 0)
                log_data["tokens_completion"] = getattr(response.usage, 'completion_tokens', 0)
                log_data["tokens_total"] = getattr(response.usage, 'total_tokens', 0)
            
            wandb.log(log_data)

        self.prev_state_str = cur_state_str
        self.last_action = action
        self.last_score = current_score

        return action

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
    
    def __del__(self):
        """Cleanup wandb on agent destruction."""
        if self.wandb_config.enabled:
            try:
                wandb.finish()
            except:
                pass
