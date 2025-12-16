import openai
import re

SYSTEM_PROMPT = """
You are a helpful AI assistant trained to play StarCraft II.
Currently, you are playing as {player_race}. Enemy's race is {enemy_race}.
You will be given a status summary in a game.
Based on the given information, we want you to analyze the game progression, provide specific strategic suggestions, and suggest the most suitable actions for the current situation.

Analysis:    
1. Provide a brief overview of the current situation.
2. Describe our current status in terms of our resources, buildings, units, research, and actions in progess.
3. Infer our potential strategy based on our current situation.
4. Infer the enemy's potential strategy based on the available information.
5. Propose adjustments to our current strategy to counter the enemy's moves and capitalize our strengths.

Actions:
Based on the given information, we want you to make {num_actions} actionable and specific decisions to follow current strategy.
The action decisions should be extracted from the ACTION_DICTIONARY below.

Guidelines:
1. State current resource status after executing previous action.
2. Provide action decision that is immediately executable, based on current resource status.
3. State the cost of the decided action, and double check if it is indeed executable.
4. State the updated resource after execution of the action.
5. Repeat 1-4 {num_actions} times. Remember that these action decisions will be executed chronologically.

### ACTION_DICTIONARY
{action_dict}
"""

USER_PROMPT = """
### Current state
{cur_state_str}

You should only respond in the format described below:

### Analysis
1. ...
2. ...
3. ...
...

### Reasoning
1: [Current Resource] [ACTION] [Cost] [Availability] [Updated Resource]
2: ...
3: ...
...

### Actions
1: <ACTION1>
2: <ACTION2>
3: <ACTION3>
...

"""

MODEL = "gpt-5-nano"

class OpenAIStarCraftAgent:
    TRACK = "TRACK1"

    def __init__(self, num_actions=5):
        self.client = openai.OpenAI()
        
        self.num_actions = num_actions
    
    def act(self, obs):
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")

        action_dict = game_info.get("action_dict", {})

        formatted_system_prompt = SYSTEM_PROMPT.format(
            player_race=game_info.get("player_race"),
            enemy_race=game_info.get("enemy_race"),
            num_actions=self.num_actions,
            action_dict=action_dict
        )
        
        formatted_user_prompt = USER_PROMPT.format(
            cur_state_str=cur_state_str
        )
        
        response = self.client.responses.create(
            model=MODEL,
            input=formatted_user_prompt,
            instructions=formatted_system_prompt,
            reasoning={"effort": "low"}
        )
        
        output = response.output_text.strip()
        print(output)
        actions = self._parse_actions(output)
        
        return actions
    
    def _parse_actions(self, output):
        """
        Return the full string after ### Actions.
        """
        actions_match = re.search(r"### Actions\s*\n(.+)", output, re.IGNORECASE | re.DOTALL)
        if actions_match:
            actions_section = actions_match.group(1).strip()
            return actions_section
        return ""