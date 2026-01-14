from agents.twenty_fourty_eight.base import TwentyFourtyEightAgent
from agents.openai_agent import BaseOpenAIAgent
from pydantic import Field

class OpenAITwentyFourtyEightAgent(BaseOpenAIAgent, TwentyFourtyEightAgent):
    """
    OpenAI-based agent for 2048.
    Inherits OpenAI initialization from BaseOpenAIAgent and game logic from TwentyFourtyEightAgent.
    """
    pass
