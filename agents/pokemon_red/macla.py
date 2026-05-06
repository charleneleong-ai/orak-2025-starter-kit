import re
from typing import Any
from loguru import logger
from config.agent_config import GeminiConfig, OpenAIConfig
from config.base import WandbConfig
from agents.pokemon_red.base import PokemonRedAgent, USER_PROMPT_TEMPLATE
from agents.macla.base import BaseMaclaAgent
import weave

        
class PokemonRedMaclaAgent(BaseMaclaAgent, PokemonRedAgent):
    config: GeminiConfig | OpenAIConfig
        
    def __init__(
        self, 
        config: GeminiConfig | OpenAIConfig = None, 
        wandb_config: WandbConfig = None,
    ):
        config = config or GeminiConfig()
        wandb_config = wandb_config or WandbConfig()

        PokemonRedAgent.__init__(self, config=config, wandb_config=wandb_config)
        
        self._init_macla_agent()
    
    @property
    def AGENT_TAGS(self):
        tags = ["pokemon_red", "macla"]
        if isinstance(self.config, GeminiConfig):
            tags.extend(["gemini", self.config.model, "vertex-ai"])
        elif isinstance(self.config, OpenAIConfig):
            tags.extend(["openai", self.config.model])
        return tags

    # Compiled once so the per-step regex match in _extract_loop_state stays cheap.
    # The map name in pokemon obs is followed by a comma (``Map Name: OaksLab,``)
    # so the non-greedy capture stops there. ``Your position (x, y)`` is the
    # canonical line emitted by ``parse_game_state`` in pokemon_red_env.
    _LOOP_MAP_RE = re.compile(r"Map Name:\s*([^,\s]+)")
    _LOOP_POS_RE = re.compile(r"Your position \(x, y\):\s*\((\d+),\s*(\d+)\)")

    def _extract_loop_state(self, obs):
        """Lift ``(map, x, y)`` out of the pokemon obs for the loop detector.

        Both regexes are pinned to the format produced by
        ``PokemonRedEnv.parse_game_state``. Returns ``None`` (silencing
        the detector for this step) if either piece is missing — for
        instance during a battle screen where ``[Map Info]`` is replaced.
        """
        text = obs.get("obs_str", "")
        if not text:
            return None
        m_map = self._LOOP_MAP_RE.search(text)
        m_pos = self._LOOP_POS_RE.search(text)
        if not m_map or not m_pos:
            return None
        return (m_map.group(1), int(m_pos.group(1)), int(m_pos.group(2)))
    
    @weave.op()
    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, str, str, Any, str, str, str]:
        """Get action from LLM. Tracked by Weave for observability."""
        # 1. Preprocess observation FIRST - add full map, annotations, NPC/warp info
        cur_state_str = self._preprocess_observation(cur_state_str)
        
        # 2. Parse the enhanced state and update history
        parsed_state = self._parse_game_state(cur_state_str)
        self._update_history(parsed_state)
        
        # Inject persistent goal into state
        if self._current_goal:
            parsed_state["current_goal"] = self._current_goal
        
        # 3. Provide feedback on previous execution
        update_info = self._provide_feedback(self._prev_state_str, cur_state_str)
        
        # 4. Execute task using MACLA with preprocessed observation
        action, execution_result, memory_stats = self._execute_task(
            task_description, 
            cur_state_str, 
            obs_image
        )
        
        # 5. Log MACLA stats
        self._log_stats(update_info, memory_stats)
        
        # 6. Validate action
        action = self._validate_action(action)
        
        # 7. Build reasoning with MACLA metadata (use current_goal for more specific milestone)
        current_goal = self._current_goal if self._current_goal else self._get_task_description({"task_description": task_description})
        reasoning = self._build_reasoning_with_macla_metadata(execution_result, action=action, goal=current_goal)
        output_text = f"Result: {execution_result}\nStats: {memory_stats}"
        
        game_phase = self._current_goal
        
        # 8. Build Pokemon Red-style prompt for logging (preprocessed state already has full context)
        progress_context = self._infer_game_progress(parsed_state)
        step_history = self._build_step_history(last_n_steps=3)
        augmented_task = f"{task_description}\n\n[INTERNAL STATE KNOWLEDGE]\n{progress_context}\n\n[INSTRUCTIONS]\nBased on the observations and your internal state, determine the next logical step."
        
        prompt_text = USER_PROMPT_TEMPLATE.format(
            task_description=augmented_task,
            last_action=self._last_action,
            step_history=step_history,
            cur_state_str=cur_state_str,
        )
        
        return action, reasoning, output_text, memory_stats, prompt_text, game_phase, self._last_update_type

    def _base_fallback(self, goal: str, observation: str, **kwargs) -> list[str]:
        """
        Pokemon Red-specific fallback that calls the base PokemonRedAgent LLM logic.
        """
        obs_image = kwargs.get("obs_image")
        try:
            # Call the parent PokemonRedAgent implementation
            action, reasoning, _, _, _, _ = PokemonRedAgent._get_action(
                self,
                task_description=goal,
                cur_state_str=observation,
                obs_image=obs_image
            )
            # Store the LLM's reasoning so we can append MACLA metadata to it
            self._llm_reasoning = reasoning
            return [action]
        except Exception as e:
            logger.error(f"Base fallback failed: {e}")
            return ["a"]

    def _validate_action(self, action: str) -> str:
        """Validate Pokemon Red action."""
        valid_actions = ["up", "down", "left", "right", "a", "b", "start", "select"]
        # Also allow tool actions
        if action.startswith("use_tool("):
            return action
        if action not in valid_actions:
            logger.warning(f"Invalid action '{action}', defaulting to 'a'")
            return "a"
        return action
    
    def _get_task_description(self, game_info: dict) -> str:
        """Get Pokemon Red task description."""
        return game_info.get("task_description", "Become the Pokemon Champion")
    
    def _get_default_goal(self) -> str:
        """Default goal for Pokemon Red."""
        return "Become the Pokemon Champion"
    
    def _get_default_action(self) -> str:
        """Default action for Pokemon Red."""
        return "a"
    
    def _detect_success(self, execution_result: dict, prev_state_str: str, cur_state_str: str) -> tuple[bool, bool]:
        """
        Pokemon Red-specific success detection based on progress flags and events.
        """
        # Check for major game progress events
        strong_success = False
        is_fatal_game_over = False
        
        try:
            # Check for positive indicators of progress
            progress_indicators = [
                "Badge obtained",
                "defeated",
                "Level up",
                "evolved",
                "learned",
                "caught",
                "received",
                "Thank you"
            ]
            
            # Check if we made significant progress
            for indicator in progress_indicators:
                if indicator.lower() in cur_state_str.lower() and indicator.lower() not in prev_state_str.lower():
                    strong_success = True
                    break
            
            # Pokemon Red doesn't have traditional game overs - player respawns at Pokemon Center
            # So is_fatal_game_over remains False
            
        except Exception as e:
            logger.warning(f"Failed to detect success: {e}")
            
        return strong_success, is_fatal_game_over
    
    def _extract_context(self, observation: str | list) -> dict[str, Any]:
        """Extract task-relevant context from Pokemon Red observation."""
        if isinstance(observation, list):
            observation = "\n".join(str(item) for item in observation)
        context = {}
        try:
            # Extract map name
            map_match = re.search(r"Map Name:\s*([^\s,]+)", observation, re.IGNORECASE)
            if map_match:
                context["map_name"] = map_match.group(1)
            
            # Extract position
            pos_match = re.search(r"Your position \(x, y\): \((\d+), (\d+)\)", observation)
            if pos_match:
                context["position"] = (int(pos_match.group(1)), int(pos_match.group(2)))
            
            # Extract facing direction
            facing_match = re.search(r"Your facing direction:\s*(\w+)", observation)
            if facing_match:
                context["facing"] = facing_match.group(1)
            
            # Extract party info
            if "[Current Party]" in observation:
                party_section = observation.split("[Current Party]")[1].split("[" if "[" in observation.split("[Current Party]")[1] else "No more Pokemons")[0]
                party_count = len([l for l in party_section.split('\n') if l.strip() and "Name:" in l])
                context["party_size"] = party_count
            
            # Extract screen text (dialog, messages)
            if "Screen Text:" in observation:
                text_section = observation.split("Screen Text:")[1].split("[")[0].strip()
                context["screen_text"] = text_section
                
        except Exception as e:
            logger.warning(f"Failed to extract context: {e}")
            
        return context
    
    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        """
        Extract preconditions from Pokemon Red observation.
        
        Preconditions describe the state requirements before an action can be taken.
        For Pokemon Red, this includes location, facing direction, party state, and nearby objects.
        
        Args:
            context_key: The context key string (often derived from _extract_context)
            observation: The raw observation string
            
        Returns:
            List of precondition strings describing the current state
        """
        preconditions = []
        
        try:
            # Extract map name
            map_match = re.search(r"Map Name:\s*([^\s,]+)", observation, re.IGNORECASE)
            if map_match:
                preconditions.append(f"in_map:{map_match.group(1)}")
            
            # Extract position
            pos_match = re.search(r"Your position \(x, y\): \((\d+), (\d+)\)", observation)
            if pos_match:
                x, y = pos_match.group(1), pos_match.group(2)
                preconditions.append(f"at_position:({x},{y})")
            
            # Extract facing direction
            facing_match = re.search(r"Your facing direction:\s*(\w+)", observation)
            if facing_match:
                preconditions.append(f"facing:{facing_match.group(1)}")
            
            # Extract party size
            if "[Current Party]" in observation:
                party_section = observation.split("[Current Party]")[1].split("[" if "[" in observation.split("[Current Party]")[1] else "No more Pokemons")[0]
                party_count = len([l for l in party_section.split('\n') if l.strip() and "Name:" in l])
                preconditions.append(f"party_size:{party_count}")
                
                # Check if party has fainted Pokemon
                if "HP: 0" in party_section or "Status: FNT" in party_section:
                    preconditions.append("has_fainted_pokemon:true")
            
            # Check for important items in bag
            if "[Bag]" in observation:
                bag_section = observation.split("[Bag]")[1].split("[")[0]
                if "Oak's Parcel" in bag_section or "OAKS PARCEL" in bag_section.upper():
                    preconditions.append("has_oaks_parcel:true")
                if "Bike" in bag_section:
                    preconditions.append("has_bike:true")
                    
            # Check if in battle
            if "battle" in observation.lower() or "[Battle]" in observation:
                preconditions.append("in_battle:true")
            
            # Check if talking to NPC (dialog on screen)
            if "Screen Text:" in observation:
                text_section = observation.split("Screen Text:")[1].split("[")[0].strip()
                if text_section:
                    preconditions.append("has_dialog:true")
                    
            # Check for unexplored areas
            if "[Full Map]" in observation and "?" in observation.split("[Full Map]")[1].split("\n[")[0]:
                preconditions.append("has_unexplored_tiles:true")
                
        except Exception as e:
            logger.warning(f"Failed to extract preconditions: {e}")
        
        return preconditions if preconditions else ["no_preconditions"]

    def extract_postconditions(self, observation: str) -> dict[str, Any]:
        """Extract postconditions (outcomes) from Pokemon Red observation."""
        # For Pokemon Red, postconditions are similar to context (current state)
        return self._extract_context(observation)

    def record_episode_end(self, episode, game_name, seed, score):
        """Called by runner at the end of an episode."""
        PokemonRedAgent.record_episode_end(self, episode, game_name, seed, score)
        self._record_episode_end(episode, score)

