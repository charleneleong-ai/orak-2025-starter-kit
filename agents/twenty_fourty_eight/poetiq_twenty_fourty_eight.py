import re
import ast
import traceback
import heapq
from typing import Any, Tuple, Dict, List
from pydantic import PrivateAttr
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from config.agent_config import PoetiqConfig
from config.base import WandbConfig
from agents.twenty_fourty_eight.base import TwentyFourtyEightAgent

SOLVER_PROMPT = """
You are an expert AI researcher and Python programmer specializing in game playing agents. Your goal is to write a Python function `get_move` that plays the game 2048 at a superhuman level.

**The Game:**
2048 is played on a 4x4 grid. The objective is to slide numbered tiles on a grid to combine them to create a tile with the number 2048.
0 represents an empty spot.

**Your Task:**
Write a function `get_move(grid)` that takes the current board state as input and returns the best move.

**Input:**
- `grid`: A 4x4 list of lists of integers representing the board (e.g., `[[0, 0, 0, 0], [2, 0, 0, 0], ...]`).

**Output:**
- Return a string, one of: `'up'`, `'down'`, `'left'`, `'right'`.

**Key Strategies to Implement:**
1.  **Corner Strategy:** Keep the highest value tile in one corner (e.g., bottom-right) and do not let it move.
2.  **Snake Chain:** Build a chain of decreasing values from the corner along the edge.
3.  **Monotonicity:** Keep rows and columns monotonic (values increasing or decreasing) to ensure tiles can merge easily.
4.  **Empty Space:** Prioritize moves that keep the board open (more zeros).
5.  **Lookahead (Optional):** If possible within constraints, simulate future moves to avoid game-over states.

**Coding Guidelines:**
- The code must be self-contained in a single function (helper functions can be defined inside or alongside).
- Use standard Python libraries. `numpy` is available if needed.
- Handle edge cases where a move might not change the board (invalid move).
- The function must be robust.

**Output Format:**
- Explanation of the strategy.
- A single markdown code block containing the Python code.

**Example Structure:**
```python
import random

def get_move(grid):
    \"""
    Implements a corner-hugging strategy.
    \"""
    # ... logic to calculate best move ...
    return 'right'
```
"""

FEEDBACK_PROMPT = """
**EXISTING PARTIAL/INCORRECT SOLUTIONS:**

Below are previous attempts to solve the game, along with their performance metrics (Score, Max Tile).
Analyze why they performed as they did. Did they get stuck? Did they move the corner tile?

### Best Strategies So Far
{best_strategies}

### Previous Attempt
Code:
{previous_code}

Performance:
- Score: {score}
- Max Tile: {max_tile}

**Your Goal:**
Synthesize a NEW, IMPROVED strategy that outperforms the previous ones.
- Fix bugs or logical flaws.
- Adjust heuristic weights.
- Implement a better lookahead or safety mechanism.

Output ONLY the new Python code within a markdown block.
"""

class PoetiqTwentyFourtyEightAgent(TwentyFourtyEightAgent):
    config: PoetiqConfig
    _llm: Any = PrivateAttr()
    _current_code: str = PrivateAttr(default="")
    _strategy_fn: Any = PrivateAttr(default=None)
    _history: list = PrivateAttr(default_factory=list)
    _best_strategies: list = PrivateAttr(default_factory=list) # Heap of top k strategies
    _last_max_tile: int = PrivateAttr(default=0)

    def __init__(
        self, 
        config: PoetiqConfig = None, 
        wandb_config: WandbConfig = None,
    ):
        config = config or PoetiqConfig(class_name="agents.twenty_fourty_eight.poetiq_twenty_fourty_eight.PoetiqTwentyFourtyEightAgent", model="gemini-pro-3-preview", temperature=0.2)
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
        )
        
        # Initialize with a first version
        self._evolve_code(initial=True)

    @property
    def AGENT_TAGS(self):
        return ["poetiq", "self-evolving", "gemini"]

    def _format_best_strategies(self, limit: int = 2) -> str:
        """Format the top strategies for the prompt."""
        if not self._best_strategies:
            return "No history yet."
        
        # _best_strategies is a heap of (score, max_tile, code)
        # We want to show the highest scores, so we sort them
        sorted_strategies = sorted(self._best_strategies, key=lambda x: x[0], reverse=True)[:limit]
        
        output = []
        for i, (score, max_tile, code) in enumerate(sorted_strategies):
            output.append(f"--- Strategy {i+1} (Score: {score}, Max Tile: {max_tile}) ---\n{code}\n")
        return "\n".join(output)

    def _evolve_code(self, initial: bool = False, last_score: int = 0, last_max_tile: int = 0):
        logger.info(f"Evolving code... (Initial: {initial}, Last Score: {last_score}, Max Tile: {last_max_tile})")
        
        if initial:
            prompt = SOLVER_PROMPT
        else:
            prompt = FEEDBACK_PROMPT.format(
                best_strategies=self._format_best_strategies(),
                previous_code=self._current_code,
                score=last_score,
                max_tile=last_max_tile
            )

        messages = [HumanMessage(content=prompt)]
        try:
            response = self._llm.invoke(messages)
            code = response.content
            
            # Clean up code (remove markdown if present)
            code = re.sub(r"```python\s*", "", code)
            code = re.sub(r"```", "", code)
            
            self._current_code = code
            self._compile_strategy()
            logger.info("Code evolved successfully.")
        except Exception as e:
            logger.error(f"Failed to evolve code: {e}")

    def _compile_strategy(self):
        try:
            local_scope = {}
            exec(self._current_code, {}, local_scope)
            if "get_move" in local_scope:
                self._strategy_fn = local_scope["get_move"]
            else:
                logger.error("Generated code does not contain 'get_move' function.")
                self._strategy_fn = None
        except Exception as e:
            logger.error(f"Failed to compile strategy: {e}")
            self._strategy_fn = None

    def _parse_grid(self, obs_str: str):
        lines = obs_str.strip().split('\n')
        grid = []
        for line in lines:
            line = line.strip()
            if line.startswith('['):
                try:
                    row = ast.literal_eval(line)
                    if isinstance(row, list) and len(row) == 4:
                        grid.append(row)
                except:
                    pass
        return grid

    def get_action(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # Track max tile for feedback
        game_info = obs.get("game_info", {})
        self._last_max_tile = int(game_info.get("max_tile", 0))

        if self._strategy_fn is None:
            return "left", {"error": "No strategy function"}
            
        try:
            obs_str = obs.get("obs_str", "")
            grid = self._parse_grid(obs_str)
            
            if not grid or len(grid) != 4:
                logger.warning("Failed to parse grid from obs_str")
                return "left", {"error": "Grid parse failed"}

            action = self._strategy_fn(grid)
            if action not in ["up", "down", "left", "right"]:
                return "left", {"error": f"Invalid action: {action}"}
            
            return action, {"code_version": len(self._history)}
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            return "left", {"error": str(e)}

    def record_episode_end(self, episode: int, game_name: str, seed: str, score: int):
        super().record_episode_end(episode, game_name, seed, score)
        
        # Store history
        self._history.append({"code": self._current_code, "score": score, "max_tile": self._last_max_tile})
        
        # Update best strategies heap (keep top 3)
        heapq.heappush(self._best_strategies, (score, self._last_max_tile, self._current_code))
        if len(self._best_strategies) > 3:
            heapq.heappop(self._best_strategies)

        # Evolve for next episode
        self._evolve_code(initial=False, last_score=score, last_max_tile=self._last_max_tile)