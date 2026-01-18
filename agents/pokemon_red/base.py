import re
import traceback
import base64
import io
from typing import Any,  Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr

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

_PROMPT_CONTENT = ADVANCED_PROMPT.split("# RESPONSE FORMAT")[0]

SYSTEM_PROMPT = f"""
{_PROMPT_CONTENT}

### Decision Output Format ###
You must respond with a structure containing:
- "reasoning": A detailed explanation of why this action was chosen (referencing the strategy and rules above).
- "action": The action button to press. Valid actions: 'up', 'down', 'left', 'right', 'a', 'b', 'start', 'select'.
"""

USER_PROMPT_TEMPLATE = """
### Target task
{task_description}

### Last executed action
{last_action}

### Current state
{cur_state_str}
"""

class GameAction(BaseModel):
    """Structured output for Pokemon Red game actions"""
    reasoning: str = Field(description="Detailed explanation of why this action was chosen")
    action: str = Field(description="The action to take: up, down, left, right, a, b, start, select")

class PokemonRedAgent(BaseOrakAgent):
    
    _llm: Optional[BaseChatModel] = PrivateAttr(default=None)

    def _parse_game_state(self, state_str: str) -> dict:
        """Parses the text game state to extract key info for milestone tracking."""
        state = {
            "map_name": "",
            "party_size": 0,
            "has_parcel": False
        }
        
        # Extract Map Name
        map_match = re.search(r"Map Name:\s*([^\s,]+)", state_str, re.IGNORECASE)
        if map_match:
            state["map_name"] = map_match.group(1)

        # Check Party
        if "[Current Party]" in state_str:
            party_section = state_str.split("[Current Party]")[1].split("[")[0]
            # If it's not "No more Pokemons", assume we have one
            # Typical text: "1. BULBASAUR L5" or just "No more Pokemons"
            if "No more Pokemons" not in party_section and "No more" not in party_section:
                state["party_size"] = 1
                
        # Check Bag for Parcel
        if "[Bag]" in state_str:
            bag_section = state_str.split("[Bag]")[1].split("[")[0]
            if "Oak's Parcel" in bag_section or "OAKS PARCEL" in bag_section.upper():
                state["has_parcel"] = True
                
        return state

    def _determine_current_milestone(self, game_state: dict) -> str:
        """Determines the current prologue milestone based on state."""
        map_name = game_state["map_name"].lower()
        has_pokemon = game_state["party_size"] > 0
        has_parcel = game_state["has_parcel"]
        
        # 1. Exit Red's House
        if "redshouse" in map_name:
            return "1. Exit Red's House (Find stairs in bedroom -> exit mat downstairs)."
            
        # 2. Encounter Professor Oak / 3. Choose Starter / 4. Rival Battle
        if "pallet" in map_name or "oakslab" in map_name:
            if has_parcel:
                return "7. Deliver Parcel to Oak in his Lab (Victory Condition)."
            
            if not has_pokemon:
                if "oakslab" in map_name:
                    return "3. Choose a Starter Pokémon (Interact with balls on table)."
                else:
                    return "2. Encounter Professor Oak (Try to walk North out of Pallet Town to trigger cutscene)."
            
            if "oakslab" in map_name:
                return "4. Defeat Rival (if in battle) OR Exit Lab to start journey."

            return "5. Travel North to Viridian City (via Route 1)."
            
        # 5. Route 1 Travel
        if "route1" in map_name:
            if has_parcel:
                return "7. Return to Pallet Town (Go South)."
            return "5. Travel North to Viridian City."
            
        # 6. Viridian City / Mart
        if "viridian" in map_name:
            if has_parcel:
                return "7. Return to Pallet Town (Go South via Route 1)."
            if "mart" in map_name:
                return "6. Receive 'Oak's Parcel' (Talk to clerk)."
            return "6. Find and Enter Viridian Mart (Look for 'SHOP' sign)."
            
        # Default fallback
        if has_parcel: return "7. Deliver Parcel to Oak."
        if has_pokemon: return "Explore and progress towards Viridian City."
        return "Explore and progress."

    def calculate_metrics(self, game_info: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate custom metrics based on game info.
        """
        metrics = {}
        
        # Pass through common metrics if available
        for key in ["score", "evaluation_score"]:
            if key in game_info:
                metrics[key] = float(game_info[key])
                
        # If specific pokemon red metrics are available in game_info (e.g. from wrapper)
        # We can add them here. For now, just a placeholder or extracting what we can.
        if "map_name" in game_info:
            metrics["map_name"] = game_info["map_name"]
            
        return metrics

    @weave.op()
    def _get_action(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, str, str, Any, str]:
        """Get action from LLM. This method is tracked by Weave for observability."""
        
        if not self._llm:
            raise ValueError("LLM not initialized")

        # Dynamic Goal Injection
        parsed_state = self._parse_game_state(cur_state_str)
        current_milestone = self._determine_current_milestone(parsed_state)
        
        # Override task description with the specific milestone context
        augmented_task = f"{task_description}\nCURRENT MILESTONE: {current_milestone}"

        prompt_text = USER_PROMPT_TEMPLATE.format(
            task_description=augmented_task,
            last_action=self._last_action, 
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
            # Default fallback action
            action = "pass"
            reasoning = f"Error: {e}"
            output_text = str(e)
            
        return action, reasoning, output_text, usage, prompt_text
