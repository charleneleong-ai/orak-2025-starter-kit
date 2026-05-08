from agents.openai_agent import BaseOpenAIAgent
from agents.starcraft.base import StarCraftAgent


class OpenAIStarCraftAgent(BaseOpenAIAgent, StarCraftAgent):
    """
    OpenAI-based agent for StarCraft II.
    Inherits OpenAI initialization from BaseOpenAIAgent and game logic from StarCraftAgent.
    """

    pass
