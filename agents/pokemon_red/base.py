import re
import traceback
import base64
import io
from typing import Any,  Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr
from agents.pokemon_red.openai_pokemon_memory_utils import (
    parse_game_state,
    get_map_memory_dict,
    replace_map_on_screen_with_full_map,
    replace_filtered_screen_text,
    get_warp_annotations,
    get_npc_annotations,
    detect_npc_interaction
)

from agents.base import BaseOrakAgent
from .pokemon_prompts import SYSTEM_PROMPT as ADVANCED_PROMPT
import weave

GAME_RULES = """
### Pokemon Red Game Rules ###
- *Goal*: The ultimate goal is to become the Pokemon Champion by defeating the Elite Four, but you need to complete various sub-tasks first (e.g., getting the package for Oak, beating Gym Leaders).
- *Movement*: You can move up, down, left, or right.
- *Interactions*: 'a' is used to interact with objects, talk to people, or confirm selections. 'b' is for cancelling or running (if you have running shoes, but not in Gen 1). 'start' opens the menu.
- *Battles*: When in battle, you need to choose moves or use items.
- *Pokemon*: You have a party of up to 6 Pokemon.

### Controls ###
- buttons: 'up', 'down', 'left', 'right', 'a', 'b', 'start', 'select'
"""


SYSTEM_PROMPT = f"""
{ADVANCED_PROMPT}

### Decision Output Format ###
You must respond with a structure containing:
- "reasoning": A detailed explanation of why this action was chosen (referencing the strategy, rules, and recent game state history below). Your reasoning MUST take into consideration the summary of recent game states (e.g., map names, party sizes, and any key transitions) as context for your decision, using the provided history.
- "current_goal": An inferred next milestone or sub-goal based on historical observations and current game state. This should reflect logical progression in the game.
- "action": The action to take. ALWAYS prefer high-level tool actions (e.g., use_tool(move_to, ...), use_tool(interact_with_object, ...), etc.) over low-level button presses ('up', 'down', 'left', 'right', 'a', 'b', 'start', 'select'), unless no tool is valid for the current state. Only use low-level actions if no tool applies or for precise menu/dialog choices/facing.
"""

USER_PROMPT_TEMPLATE = """
### Target task
{task_description}

### Last executed action
{last_action}

### Recent Step History (Last 3 Steps)
{step_history}

### Current state
{cur_state_str}
"""

class GameAction(BaseModel):
    """Structured output for Pokemon Red game actions"""
    reasoning: str = Field(description="Detailed explanation of why this action was chosen")
    action: str = Field(description="The action to take: up, down, left, right, a, b, start, select")
    current_goal: Optional[str] = Field(default=None, description="Inferred next milestone or sub-goal based on historical observations and current game state")

class PokemonRedAgent(BaseOrakAgent):
    
    _llm: Optional[BaseChatModel] = PrivateAttr(default=None)
    _history: list[dict] = PrivateAttr(default_factory=list)
    _text_history: list[str] = PrivateAttr(default_factory=list)
    _current_goal: str = PrivateAttr(default="")
    _map_memory: dict = PrivateAttr(default_factory=dict)
    _warp_memory: dict = PrivateAttr(default_factory=dict)

    def _preprocess_observation(self, obs_str: str) -> str:
        """Preprocess observation string to ensure '[Full Map]' is present using normalization utility."""
        try:
            # 1. Parse current state using utils to get map info
            parsed_state = parse_game_state(obs_str)
            
            # 2. Update map memory
            self._map_memory = get_map_memory_dict(parsed_state, self._map_memory)
            
            # 3. Retrieve current map grid
            map_current = []
            if parsed_state.get('map_info') and parsed_state['map_info'].get('map_name'):
                map_name = parsed_state['map_info']['map_name']
                if map_name in self._map_memory:
                    map_current = self._map_memory[map_name].get("explored_map", [])
            
            # 4. Inject full map into observation
            current_map_name = parsed_state.get('map_info', {}).get('map_name')
            
            # --- NPC Interaction Update (Stateful) ---
            target_obj = detect_npc_interaction(parsed_state, map_current)
            if target_obj and current_map_name:
                screen_text = parsed_state.get("screen_text")
                if "npcs" not in self._map_memory[current_map_name]:
                    self._map_memory[current_map_name]["npcs"] = {}
                
                # Get existing memory (handle legacy string format)
                npc_mem = self._map_memory[current_map_name]["npcs"].get(target_obj, {"lines": [], "summary": ""})
                if isinstance(npc_mem, str): 
                    npc_mem = {"lines": [npc_mem], "summary": npc_mem}
                    
                # Avoid duplicate contiguous lines (basic dedup for page turns)
                if not npc_mem["lines"] or npc_mem["lines"][-1] != screen_text:
                    npc_mem["lines"].append(screen_text)
                    
                    # Summarize using LLM
                    full_text = " ".join(npc_mem["lines"])
                    if len(full_text) < 1000 and self._llm:
                        try:
                            summary_prompt = f"Summarize this game dialogue from {target_obj} in 1 short sentence: '{full_text}'"
                            # We use direct invoke which might block slightly but ensures summary is ready
                            res = self._llm.invoke([HumanMessage(content=summary_prompt)])
                            npc_mem["summary"] = res.content.strip()
                        except Exception as summary_err:
                            logger.warning(f"Summary failed: {summary_err}")
                            npc_mem["summary"] = full_text[:50] + "..."
                    else:
                        npc_mem["summary"] = full_text[:50] + "..."
                        
                    self._map_memory[current_map_name]["npcs"][target_obj] = npc_mem

            # --- Annotation Generation (Warps + NPCs) ---
            warp_ann = get_warp_annotations(self._warp_memory, self._map_memory, current_map_name, map_current)
            npc_ann = get_npc_annotations(self._map_memory, current_map_name, map_current)
            
            # Merge annotations
            final_annotations = {**warp_ann, **npc_ann}

            obs_str = replace_map_on_screen_with_full_map(obs_str, map_current, final_annotations)
        except Exception as e:
            logger.warning(f"Failed to preprocess observation with map memory: {e}")
            
        return obs_str

    def _parse_game_state(self, state_str: str) -> dict:
        """Parses the text game state to extract key info for milestone tracking."""
        state = {
            "map_name": "",
            "current_goal": "",
            "party_size": 0,
            "has_parcel": False,
            "pos": None,
            "facing": None,
            "screen_type": "Unknown",
            "full_map_str": ""
        }
        
        # Extract Map Name
        map_match = re.search(r"Map Name:\s*([^\s,]+)", state_str, re.IGNORECASE)
        if map_match:
            state["map_name"] = map_match.group(1)

        # Extract Position & Facing (For warp tracking)
        pos_match = re.search(r"Your position \(x, y\): \((\d+), (\d+)\)", state_str)
        if pos_match:
            state["pos"] = (int(pos_match.group(1)), int(pos_match.group(2)))
        
        facing_match = re.search(r"Your facing direction:\s*(\w+)", state_str)
        if facing_match:
            state["facing"] = facing_match.group(1)
            
        # Extract Current Goal
        goal_match = re.search(r"Current Goal:\s*(.+)", state_str, re.IGNORECASE)
        if goal_match:
            state["current_goal"] = goal_match.group(1).strip()

        # Extract [Full Map] section (including '?' tiles)
        map_section = ""
        if "[Full Map]" in state_str:
            try:
                map_section = state_str.split("[Full Map]")[1]
                # Grab until next section or end
                next_header = re.search(r"\n\[", map_section)
                if next_header:
                    map_section = map_section[:next_header.start()]
                # Use strip('\n') to preserve horizontal indentation of the first line
                state["full_map_str"] = map_section.strip('\n')
            except Exception:
                state["full_map_str"] = ""
    
        # Check Party
        if "[Current Party]" in state_str:
            party_section = state_str.split("[Current Party]")[1].split("[")[0]
            if "No more Pokemons" not in party_section and "No more" not in party_section:
                # Count lines that look like pokemon entries to get size
                lines = [l for l in party_section.split('\n') if l.strip()]
                state["party_size"] = len(lines) if party_section.strip() else 0
                
        # Check Bag for Parcel
        if "[Bag]" in state_str:
            bag_section = state_str.split("[Bag]")[1].split("[")[0]
            if "Oak's Parcel" in bag_section or "OAKS PARCEL" in bag_section.upper():
                state["has_parcel"] = True

        # Extract Screen Text (Heuristic: Look for text not in brackets)
        # This is messy but we try to grab the last few lines of raw text that might be dialog
        # Assuming the state string puts raw text at the end or in a specific section.
        # For now, we will look for a "Screen Text" section if it exists, or just parse generic lines.
        if "Screen Text:" in state_str:
            text_section = state_str.split("Screen Text:")[1].split("[")[0].strip()
            state["screen_text"] = text_section
        
        return state

    def _update_history(self, state: dict):
        """Updates internal history with significant events."""
        text = state.get("screen_text", "").strip()
        if text and (not self._text_history or self._text_history[-1] != text):
            # Only add if it's different information
            if len(self._text_history) > 10:
                self._text_history.pop(0)
            self._text_history.append(text)
        
    def _build_step_history(self, last_n_steps: int = 3) -> str:
        """Builds a string summary of recent steps for prompt inclusion."""
        # Build history string for prompt
        history_lines = []
        if isinstance(self._history, list):
            for h in self._history[-last_n_steps:]:
                start = h.get('start_state', {})
                res = h.get('result_state', {})
                
                # Format start
                start_str = f"{start.get('map', '?')}"
                if start.get('pos'): start_str += f"{start.get('pos')}"
                if start.get('facing'): start_str += f"({start.get('facing')})"
                
                # Format result
                res_str = "???"
                if res:
                    res_str = f"{res.get('map', '?')}"
                    if res.get('pos'): res_str += f"{res.get('pos')}"
                    if res.get('facing'): res_str += f"({res.get('facing')})"

                history_lines.append(f"Step {h.get('step')}: {start_str} -> Action '{h.get('action')}' -> {res_str}")
                if h.get('reasoning'):
                    history_lines.append(f"   Reasoning: {h.get('reasoning')}")
        
        return"\n".join(history_lines) if history_lines else "No history yet."
        
    def _infer_game_progress(self, game_state: dict) -> str:
        """Infers the game state based on observations rather than hardcoded spoilers."""
        progress_indicators = []
        
        if game_state.get("current_goal"):
            progress_indicators.append(f"GOAL: Current goal is '{game_state['current_goal']}'.")
        
        # Fact-based observations
        
        if game_state["party_size"] == 0:
            progress_indicators.append("OBSERVATION: You have no Pokemon.")
        else:
            progress_indicators.append(f"OBSERVATION: Party size is {game_state['party_size']}. You are ready for battle.")

        map_name = game_state["map_name"]
        progress_indicators.append(f"LOCATION: Currently in {map_name}.")
        
        # Exploration Check
        if "?" in game_state.get("full_map_str", ""):
             progress_indicators.append("OBSERVATION: The current map contains unexplored areas ('?'). You have not seen the whole room yet. Please prioritise exploring the map fully.")
        else:
            progress_indicators.append("OBSERVATION: The current map has now been fully explored.")

        # NPC Summaries (Long-term Memory of this Map)
        if map_name and map_name in self._map_memory and "npcs" in self._map_memory[map_name]:
            npc_dict = self._map_memory[map_name]["npcs"]
            if npc_dict:
                progress_indicators.append("NPC KNOWLEDGE (What they said):")
                for sprite_id, data in npc_dict.items():
                    # Handle legacy string format vs new dict format
                    summary = data.get("summary", data) if isinstance(data, dict) else data
                    # Only show if not empty
                    if summary and summary != "...":
                        progress_indicators.append(f"- {sprite_id}: {summary}")

        # History-based context (Short-term memory)
        if self._text_history:
            history_str = " | ".join(self._text_history[-3:]) # Last 3 unique texts
            progress_indicators.append(f"RECENT DIALOG/TEXT: {history_str}")

        return "\n".join(progress_indicators)
    def calculate_metrics(self, game_info: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate custom metrics based on game info.
        """
        metrics = {}
        
        # Pass through common metrics if available
        for key in ["score", "evaluation_score"]:
            if key in game_info:
                metrics[key] = float(game_info[key])
                
        if "map_name" in game_info:
            metrics["map_name"] = game_info["map_name"]
            
        return metrics

    def get_state(self) -> dict[str, Any]:
        """Override to include PokemonRed-specific history."""
        state = super().get_state()
        state["pokemon_red_history"] = self._history
        state["pokemon_red_map_memory"] = self._map_memory
        state["pokemon_red_current_goal"] = self._current_goal
        state["pokemon_red_warp_memory"] = self._warp_memory
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Override to load PokemonRed-specific history."""
        super().load_state(state)
        # Restore history, defaulting to empty list if missing
        self._history = state.get("pokemon_red_history", [])
        if not isinstance(self._history, list):
            self._history = []
        
        # Restore map memory
        self._map_memory = state.get("pokemon_red_map_memory", {})
        if not isinstance(self._map_memory, dict):
            self._map_memory = {}
            
        # Restore current goal
        self._current_goal = state.get("pokemon_red_current_goal", "")
        
        # Restore warp memory
        self._warp_memory = state.get("pokemon_red_warp_memory", {})
        if not isinstance(self._warp_memory, dict):
            self._warp_memory = {}

            
    def _record_warp(self, start_state, action, end_state):
        start_pos = start_state.get("pos")
        start_map = start_state.get("map")
        end_pos = end_state.get("pos")
        end_map = end_state.get("map")
        
        if not start_pos or not start_map or not end_pos or not end_map:
            return
            
        x, y = start_pos
        
        # 1. Forward link: Step INTO the warp
        # If action was 'up', we moved 'up' to hit the warp tile.
        if action == "up": y -= 1
        elif action == "down": y += 1
        elif action == "left": x -= 1
        elif action == "right": x += 1
        
        if start_map not in self._warp_memory:
            self._warp_memory[start_map] = {}
        # Record destination map
        self._warp_memory[start_map][(x, y)] = end_map
        
        # 2. Backward link: Where we landed implies a return path
        if end_map not in self._warp_memory:
             self._warp_memory[end_map] = {}
        # Record return map at landing position
        self._warp_memory[end_map][end_pos] = start_map

    @weave.op()
    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, str, str, str,  Any, str]:
        """Get action from LLM. This method is tracked by Weave for observability."""
        
        if not self._llm:
            raise ValueError("LLM not initialized")
            
        # Early Warp Detection (to update memory before rendering)
        # Parse minimal state to check for warps so the new map render includes the return link immediately.
        try:
            raw_parsed = self._parse_game_state(cur_state_str)
            raw_map_name = raw_parsed.get("map_name")
            
            if self._history and self._history[-1].get("step") == self._step_count - 1:
                prev_step = self._history[-1]
                start_map = prev_step.get("start_state", {}).get("map")
                
                if start_map and raw_map_name and start_map != raw_map_name:
                    # Construct valid current info for recording
                    raw_current_info = {
                        "map": raw_map_name,
                        "pos": raw_parsed.get("pos"),
                        "facing": raw_parsed.get("facing"),
                        "screen_type": raw_parsed.get("screen_type")
                    }
                    self._record_warp(prev_step["start_state"], prev_step["action"], raw_current_info)
        except Exception as e:
            logger.warning(f"Early warp detection failed: {e}")

        # Preprocess observation
        cur_state_str = self._preprocess_observation(cur_state_str)

        # Dynamic Goal Injection
        parsed_state = self._parse_game_state(cur_state_str)
        
        # Inject persistent goal into parsed state so it appears in internal knowledge
        if self._current_goal:
            parsed_state["current_goal"] = self._current_goal
                
        # Capture Detailed State
        current_state_info = {
            "map": parsed_state.get("map_name", "Unknown"),
            "current_goal": self._current_goal, 
            "pos": parsed_state.get("pos"), # Tuple or None
            "facing": parsed_state.get("facing"),
            "type": parsed_state.get("screen_type")
        }
        
        # Update previous history entry with the resulting state
        if self._history and self._history[-1].get("step") == self._step_count - 1:
            self._history[-1]["result_state"] = current_state_info
            
        self._update_history(parsed_state)
        
        # Infer context instead of forcing a milestone
        progress_context = self._infer_game_progress(parsed_state)

        step_history = self._build_step_history(last_n_steps=3)
        
        # Augment task with internal reasoning context
        augmented_task = f"{task_description}\n\n[INTERNAL STATE KNOWLEDGE]\n{progress_context}\n\n[INSTRUCTIONS]\nBased on the observations and your internal state, determine the next logical step."
    
        # Override task description with the specific milestone context
        prompt_text = USER_PROMPT_TEMPLATE.format(
            task_description=augmented_task,
            last_action=self._last_action, 
            step_history=step_history,
            cur_state_str=cur_state_str,
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
        
        structured_llm = self._llm.with_structured_output(GameAction)
        
        action = "pass"
        reasoning = "None"
        current_goal = self._current_goal or "Unknown"
        usage = None
        output_text = ""
        
        try:
            # Note: with_structured_output returns the data model directly
            response = structured_llm.invoke(messages)
            
            action = response.action.lower()
            reasoning = response.reasoning
            current_goal = response.current_goal or "Unknown"
            
            # Update internal state with new goal
            if current_goal != "Unknown":
                self._current_goal = current_goal

            output_text = f"Action: {action}\nReasoning: {reasoning}\n Goal: {current_goal}"
            
                        
            # Update history
            if isinstance(self._history, list):
                self._history.append({
                    "step": self._step_count,
                    "action": action,
                    "reasoning": reasoning,
                    "start_state": current_state_info,
                    "result_state": {} # Will be updated next step
                })
                # Keep only last 10-20 to avoid memory leak, though we only use 3
                if len(self._history) > 20:
                    self._history.pop(0)

        except Exception as e:
            logger.error(f"Error invoking LLM: {traceback.format_exc()}")
            # Default fallback action
            action = "pass"
            reasoning = f"Error: {e}"
            output_text = str(e)
            
        return action, reasoning, current_goal, output_text, usage, prompt_text
