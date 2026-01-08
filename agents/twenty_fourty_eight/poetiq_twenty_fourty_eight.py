import heapq
import numpy as np
from typing import Any, Optional
from pydantic import PrivateAttr, BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from config.agent_config import PoetiqConfig
from config.base import WandbConfig
from agents.twenty_fourty_eight.base import TwentyFourtyEightAgent

# --- PROMPTS ---

OPTIMIZER_SYSTEM_PROMPT = """You are an expert at prompt engineering and game strategy for 2048.
Your goal is to evolve a textual 'Strategy Guide' or 'System Instructions' for another LLM (the Player LLM) that is playing the game 2048.

The Player LLM will receive:
1. The current board state (4x4 grid).
2. Your Strategy Guide.

And it must output the best move (up, down, left, right).

Your instructions should be:
- Clear and concise.
- Contain specific heuristics (e.g., 'Keep max tile in bottom-right corner', 'Build snake chain', 'Avoid moving up').
- Explain how to prioritize moves when the preferred move is impossible.
- Robust to gridlock scenarios.

DO NOT write Python code. Write INSTRUCTIONS for an AI Agent."""

OPTIMIZER_INITIAL_PROMPT = """Generate the initial set of instructions for the 2048 Player LLM.
Focus on the standard 'Corner Strategy' where the largest tile is kept in one corner."""

OPTIMIZER_FEEDBACK_PROMPT = """
**PERFORMANCE REPORT:**

The previous instructions were used to play a game.
- Score: {score}
- Max Tile: {max_tile}
- Steps Survived: {steps}
- Failure Mode: {failure_mode}

**FAILURE ANALYSIS:**
{failure_analysis}

**PREVIOUS INSTRUCTIONS:**
{previous_instructions}

**BEST INSTRUCTIONS HISTORY:**
{history}

**YOUR TASK:**
Rewrite the instructions to improve performance.
- Fix the weaknesses identified in the failure analysis.
- Reinforce successful behaviors from history.
- Be specific about what moves to avoid in critical situations.

Output ONLY the new instructions text.
"""

PLAYER_SYSTEM_PROMPT_TEMPLATE = """You are a superhuman 2048 playing agent.
Your goal is to reach the 2048 tile and maximize score.

**STRATEGIC INSTRUCTIONS (Follow these strictly):**
{instructions}

**GAME RULES:**
- 4x4 grid.
- Swipe to merge matching numbers.
- New tiles (2 or 4) spawn after every move.
- Game over if no moves are possible.
"""

PLAYER_USER_PROMPT_TEMPLATE = """
**CURRENT BOARD:**
{obs_str}

**DECISION:**
Based on the Strategic Instructions and the Current Board, what is the best move?
Respond with a valid JSON object:
{{
  "reasoning": "Explain your thought process here...",
  "action": "one of: up, down, left, right"
}}
"""

# --- STRUCTURED OUTPUT ---

class GameAction(BaseModel):
    reasoning: str = Field(description="The thought process behind the move.")
    action: str = Field(description="The move to take: up, down, left, or right.")

# --- AGENT CLASS ---

class PoetiqTwentyFourtyEightAgent(TwentyFourtyEightAgent):
    config: PoetiqConfig
    _llm_optimizer: Any = PrivateAttr()
    _llm_player: Any = PrivateAttr()
    _current_instructions: str = PrivateAttr(default="")
    _history: list = PrivateAttr(default_factory=list)  # All attempts
    _best_strategies: list = PrivateAttr(default_factory=list)  # Heap of top k strategies
    _last_max_tile: int = PrivateAttr(default=0)
    _episode_step_count: int = PrivateAttr(default=0)
    _failure_mode: Optional[str] = PrivateAttr(default=None)
    _rng: Any = PrivateAttr(default=None)

    def __init__(
        self, 
        config: PoetiqConfig = None, 
        wandb_config: WandbConfig = None,
    ):
        config = config or PoetiqConfig(
            class_name="agents.twenty_fourty_eight.poetiq_twenty_fourty_eight.PoetiqTwentyFourtyEightAgent", 
            model="gemini-pro-3-preview", 
            temperature=0.2
        )
        wandb_config = wandb_config or WandbConfig()

        super().__init__(
            config=config,
            wandb_config=wandb_config,
        )
        
        # Optimizer LLM (Generates Instructions)
        self._llm_optimizer = ChatVertexAI(
            model_name=self.config.model,
            temperature=0.7, # Higher temp for creative evolution
            project=self.config.gcp_project,
            location=self.config.gcp_location,
        )
        
        # Player LLM (Executes Instructions)
        # Using structured output for reliable actions
        self._llm_player = ChatVertexAI(
            model_name=self.config.model,
            temperature=0.0, # Low temp for precise execution
            project=self.config.gcp_project,
            location=self.config.gcp_location,
        ).with_structured_output(GameAction)
        
        # Initialize RNG for selection probability
        self._rng = np.random.default_rng(self.config.seed)
        
        # Initialize with a first version of instructions
        self._evolve_instructions(initial=True)

    @property
    def AGENT_TAGS(self):
        return ["poetiq", "prompt-optimization", "gemini"]

    def _format_history_for_prompt(self) -> str:
        """Format best strategies for the prompt."""
        if not self._best_strategies:
            return "No previous history yet."
        
        # Filter and sort
        selected = []
        for item in self._best_strategies:
            # item is (score, max_tile, instructions, failure_mode, steps)
            if self._rng.uniform() < self.config.selection_probability:
                selected.append(item)
        
        if not selected:
            # Fallback to just showing the best one if random selection picked nothing
             if self._best_strategies:
                selected = [max(self._best_strategies, key=lambda x: x[0])]
             else:
                return "No strategies selected."
        
        # Sort best to worst for the prompt context
        sorted_strategies = sorted(selected, key=lambda x: x[0], reverse=True)[:3]
        
        output = []
        for i, (score, max_tile, instructions, failure_mode, steps) in enumerate(sorted_strategies, 1):
            output.append(
                f"--- Strategy {i} (Score: {score}, Max Tile: {max_tile}) ---\n"
                f"{instructions}\n"
            )
        return "\n".join(output)

    def _evolve_instructions(self, initial: bool = False, last_score: int = 0, last_max_tile: int = 0, 
                             failure_mode: str = "unknown", steps: int = 0):
        logger.info(f"Evolving instructions... (Initial: {initial}, Last Score: {last_score}, Max Tile: {last_max_tile})")
        
        system_message = SystemMessage(content=OPTIMIZER_SYSTEM_PROMPT)
        
        if initial:
            prompt_content = OPTIMIZER_INITIAL_PROMPT
        else:
            failure_analysis = self._analyze_failure(last_score, last_max_tile, failure_mode, steps)
            
            prompt_content = OPTIMIZER_FEEDBACK_PROMPT.format(
                score=last_score,
                max_tile=last_max_tile,
                steps=steps,
                failure_mode=failure_mode,
                failure_analysis=failure_analysis,
                previous_instructions=self._current_instructions,
                history=self._format_history_for_prompt()
            )

        logger.info(f"Optimizer Prompt:\n{prompt_content}")

        messages = [system_message, HumanMessage(content=prompt_content)]
        
        try:
            response = self._llm_optimizer.invoke(messages)
            new_instructions = response.content.strip()
            
            self._current_instructions = new_instructions
            logger.info("Instructions evolved successfully.")
            logger.debug(f"New Instructions:\n{new_instructions}")
        except Exception as e:
            logger.error(f"Failed to evolve instructions: {e}")
            # Keep previous instructions on failure

    def _analyze_failure(self, score: int, max_tile: int, failure_mode: str, steps: int) -> str:
        """Analyze why the strategy failed."""
        analysis = []
        
        if failure_mode == "gridlock":
            analysis.append("- The board became gridlocked with no valid moves available.")
        elif failure_mode == "invalid_action_loop":
            analysis.append("- The agent kept trying invalid moves.")
        
        if max_tile < 128:
            analysis.append("- Very low max tile achieved - strategy is fundamentally flawed.")
        elif max_tile < 2048:
            analysis.append(f"- Reached {max_tile}, but failed to merge up to 2048.")
        
        if steps < 50:
            analysis.append("- Game ended extremely early.")
        
        return "\n".join(analysis) if analysis else "No specific failure patterns detected."

    def get_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # Track stats
        game_info = obs.get("game_info", {})
        self._last_max_tile = int(game_info.get("max_tile", 0))
        self._episode_step_count += 1
        obs_str = obs.get("obs_str", "")

        # Construct prompt for Player LLM
        system_prompt = PLAYER_SYSTEM_PROMPT_TEMPLATE.format(instructions=self._current_instructions)
        user_prompt = PLAYER_USER_PROMPT_TEMPLATE.format(obs_str=obs_str)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Call Player LLM
            # structured_output returns a GameAction object
            response = self._llm_player.invoke(messages)
            
            action = response.action.lower()
            reasoning = response.reasoning
            
            if action not in ["up", "down", "left", "right"]:
                # Fallback
                logger.warning(f"LLM returned invalid action: {action}")
                action = "left" # Default fallback
            
            return action, {
                "reasoning": reasoning,
                "instructions_version": len(self._history),
                "episode_step": self._episode_step_count
            }
            
        except Exception as e:
            logger.error(f"Player LLM execution failed: {e}")
            self._failure_mode = "runtime_error"
            return "left", {"error": str(e)}

    def record_episode_end(self, episode: int, game_name: str, seed: str, score: int):
        super().record_episode_end(episode, game_name, seed, score)
        
        # Determine failure mode if not set during execution
        if self._failure_mode is None:
            if self._episode_step_count >= 1000:
                self._failure_mode = "timeout"
            else:
                self._failure_mode = "gridlock"
        
        steps = self._episode_step_count
        
        # Store history
        self._history.append({
            "instructions": self._current_instructions, 
            "score": score, 
            "max_tile": self._last_max_tile,
            "failure_mode": self._failure_mode,
            "steps": steps
        })
        
        # Update best strategies heap (using min-heap for top-k)
        heapq.heappush(self._best_strategies, (
            score, 
            self._last_max_tile, 
            self._current_instructions,
            self._failure_mode,
            steps
        ))
        if len(self._best_strategies) > self.config.max_solutions:
            heapq.heappop(self._best_strategies)
        
        logger.info(f"Episode {episode} ended: Score={score}, MaxTile={self._last_max_tile}")

        # Evolve for next episode
        self._evolve_instructions(
            initial=False, 
            last_score=score, 
            last_max_tile=self._last_max_tile,
            failure_mode=self._failure_mode,
            steps=steps
        )
        
        # Reset
        self._episode_step_count = 0
        self._failure_mode = None
