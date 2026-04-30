import traceback
import base64
import io
import re
import ast
from typing import Any,  Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

from agents.base import BaseOrakAgent
from agents._harness import structured_invoke_with_usage, with_retries


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
You are an expert AI agent specialised in playing the 2048 game with advanced strategic reasoning. 
Your primary goal is to achieve the highest possible tile value while maintaining long-term playability by preserving the flexibility of the board and avoiding premature game over. 

{GAME_RULES}

### Strategic Optimisation Guide ###
To maximise your score and survival time (essential for learning):
1. **Corner Strategy**: Keep your highest value tile in a corner (e.g., top-left) and DO NOT move it unless forced.
2. **Monotonicity**: Ensure tiles decrease in value as they move away from the corner.
3. **Space Management**: Prioritise actions that merge tiles to free up grid space.

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

def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class TwentyFourtyEightAgent(BaseOrakAgent):
    
    _llm: Optional[BaseChatModel] = PrivateAttr(default=None)
    _game_phase: str = PrivateAttr(default="UNKNOWN")
    _last_valid_game_phase: str = PrivateAttr(default="EARLY")
    _last_update_type: str = PrivateAttr(default="atomic_entry")

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

    def extract_postconditions(self, success_contexts):
        """
        Extract domain-specific postconditions (state changes) from success contexts.
        To be used by learning agents (like MACLA) to refine procedures.
        
        Args:
            success_contexts: List of ContrastiveContext objects containing init/term observations.
            
        Returns:
            dict: Updates to be merged into discriminative_patterns (e.g. {'postconditions_added': [...]})
            or None to use default generic extraction.
        """
        # Default 2048 logic: 
        # Added = Emergent tiles (higher value)
        # Removed = Vanished tiles (merged)
        if not success_contexts:
            return None
            
        success_init_vocab = set()
        success_term_vocab = set()
        
        for ctx in success_contexts:
            if hasattr(ctx, "observation_init") and hasattr(ctx, "observation_term"):
                success_init_vocab.update(ctx.observation_init.lower().split())
                success_term_vocab.update(ctx.observation_term.lower().split())
            
        emergent = list(success_term_vocab - success_init_vocab)[:3]
        vanished = list(success_init_vocab - success_term_vocab)[:3]
        
        return {
            "postconditions_added": emergent,
            "postconditions_removed": vanished
        }

    @staticmethod
    def _extract_context(observation: str) -> str:
        """
        Extracts a rich unique context key from the game state.
        Format: {MaxTileLoc}_{Density}[_{LastAction}]
        """
        # 1. Parse Last Action if injected
        last_action_suffix = ""
        action_match = re.search(r"LastAction:\s*([a-zA-Z]+)", observation, re.IGNORECASE)
        if action_match:
            action_word = action_match.group(1).lower()
            # "No action yet" -> "No" -> "start"
            if action_word == "no":
                last_action_suffix = "_start"
            else:
                last_action_suffix = f"_{action_word}"

        ## 2. Parse Board State
        try:
            nums = []
            # Only parse lines that look like rows [x, x, x, x] to avoid Score or other numbers
            for line in observation.split('\n'):
                if line.strip().startswith('[') and ']' in line:
                    nums.extend([int(n) for n in re.findall(r'\d+', line)])
            
            if len(nums) != 16:
                # Fallback to loose searching if strict parsing fails
                all_nums = [int(n) for n in re.findall(r'\d+', observation)]
                # If we have 16+ numbers, assume the first 16 are the board or try heuristics
                # Note: Score is usually at the end.
                if len(all_nums) >= 16:
                     # Heuristic: usually board comes first. 
                     # But let's check for "Score:" label
                     if "Score:" in observation:
                         # Exclude the score from the numbers if possible 
                         pass
                     nums = all_nums[:16] # simplified fallback

            if len(nums) == 16:
                # Density
                empty_count = nums.count(0)
                if empty_count >= 8: density = "sparse"
                elif empty_count >= 4: density = "medium"
                else: density = "dense"

                # Max Tile Location
                max_val = max(nums)
                if max_val == 0: return "general"
                
                idx = nums.index(max_val)
                r, c = idx // 4, idx % 4
                
                # Geometric Classification
                loc_type = "inner"
                
                # Corners: (0,0), (0,3), (3,0), (3,3)
                is_corner = (r in [0, 3]) and (c in [0, 3])
                
                if is_corner:
                    v = "top" if r == 0 else "btm"
                    h = "left" if c == 0 else "right"
                    loc_type = f"corner_{v}_{h}"
                elif r in [0, 3]: # Edge Top/Bottom
                    v = "top" if r == 0 else "btm"
                    loc_type = f"edge_{v}"
                elif c in [0, 3]: # Edge Left/Right
                    h = "left" if c == 0 else "right"
                    loc_type = f"edge_{h}"
                else:
                    loc_type = "inner" # The "danger zone"
                
                return f"{loc_type}_{density}{last_action_suffix}"

        except Exception:
            pass
            
        return "general"

    # NOTE: get_action is inherited from BaseOrakAgent; the lenient parser
    # handles this game's 7-tuple (action, reasoning, output_text, usage,
    # prompt, game_phase, update_type) and feeds cache stats + fallback flag
    # automatically.

    def _get_phase_hint(self, phase_name: str) -> str:
        """Get the appropriate hint for a given phase name."""
        hints = {
            "EARLY": "\n[EARLY GAME - FOUNDATION]: Establish your corner anchor. Focus on reaching 64 or 128.",
            "EARLY-MID": "\n[EARLY-MID GAME - STABILISATION]: Build monotonic chains from your anchor. Keep flexibility.",
            "MID": "\n[MID GAME - CONSOLIDATION]: Maintain the monotonic chain from your corner anchor. Plan ahead.",
            "MID-CRITICAL": "\n[MID GAME - CRITICAL]: Board is full! Maintain monotonic structure strictly. One wrong move ends game.",
            "LATE": "\n[LATE GAME - ENDGAME]: Extreme precision required. Every move matters. Protect your anchor.",
        }
        return hints.get(phase_name, "")

    def _determine_game_phase(self, cur_state_str: str) -> tuple[str, str]:
        """
        Intelligently determine game phase based on multiple factors:
        - Max tile value
        - Board density (empty spaces)
        - Score progression
        - Strategic readiness (corner anchor established)
        
        Returns:
            Tuple of (phase_name, phase_hint)
            phase_name: Short identifier like "EARLY", "MID", "LATE", etc.
            phase_hint: Detailed strategic hint for the prompt
        """
        try:
            # Extract board values using ast.literal_eval
            # Board format: [val, val, val, val] on separate lines
            board = []
            for line in cur_state_str.split('\n'):
                stripped = line.strip()
                if stripped.startswith('[') and stripped.endswith(']'):
                    try:
                        row = ast.literal_eval(stripped)
                        if isinstance(row, list):
                            board.extend(row)
                    except (ValueError, SyntaxError):
                        continue
            
            if len(board) != 16:
                # Fallback to last valid phase
                return (self._last_valid_game_phase, self._get_phase_hint(self._last_valid_game_phase))
            
            current_max = max(board)
            empty_count = board.count(0)
            board_density = (16 - empty_count) / 16.0
            
            # Multi-factor phase determination
            phase_name = ""
            
            # EARLY: Focus on getting first high tile (target 64-128)
            if current_max < 64:
                phase_name = "EARLY"
            
            # EARLY-MID: Has decent progress, but board still flexible
            elif current_max < 128 and board_density < 0.75:
                phase_name = "EARLY-MID"
            
            # MID: Good progress, board getting full, need to be strategic
            elif current_max >= 128 and current_max < 512:
                if board_density > 0.85:
                    phase_name = "MID-CRITICAL"
                else:
                    phase_name = "MID"
            
            # LATE: Very high tiles, extreme caution needed
            elif current_max >= 512:
                phase_name = "LATE"
            
            # Store the valid phase for future fallback
            if phase_name:
                self._last_valid_game_phase = phase_name
                return (phase_name, self._get_phase_hint(phase_name))
            
        except Exception as e:
            logger.warning(f"Error determining game phase: {e}")
        
        # Fallback to last valid phase
        return (self._last_valid_game_phase, self._get_phase_hint(self._last_valid_game_phase))

    @weave.op()
    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, str, str, Any, str, str, str]:
        """Get action from LLM. This method is tracked by Weave for observability."""
        
        if not self._llm:
            raise ValueError("LLM not initialised")

        # Dynamic Game Phase Analysis for Prompt Optimisation
        phase_name, phase_hint = self._determine_game_phase(cur_state_str)
        self._game_phase = phase_name

        prompt_text = USER_PROMPT_TEMPLATE.format(
            task_description=task_description + phase_hint,
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
        
        # Invoke LLM (structured_invoke_with_usage preserves token usage)
        usage = None
        output_text = ""

        try:
            response, usage = with_retries(
                lambda: structured_invoke_with_usage(self._llm, messages, GameAction),
                label="twenty_fourty_eight.llm",
            )

            action = response.action.lower()
            reasoning = response.reasoning
            output_text = f"Action: {action}\nReasoning: {reasoning}"

        except Exception as e:
            logger.error(f"Error invoking LLM after retries: {traceback.format_exc()}")
            self._mark_fallback(f"llm_error: {type(e).__name__}: {str(e)[:200]}")
            action = "left"
            reasoning = f"Error: {e}"
            output_text = str(e)
            raise ValueError(f"LLM invocation failed: {traceback.format_exc()}")
        
        # Validate action
        if action not in ["left", "right", "up", "down"]:
            logger.warning(f"Invalid action '{action}', defaulting to 'left'")
            action = "left"
            
        return action, reasoning, output_text, usage, prompt_text, phase_name, "atomic_entry"
