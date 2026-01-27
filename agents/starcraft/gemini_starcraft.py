from agents.gemini_agent import BaseGeminiAgent
from agents.starcraft.base import StarCraftAgent


class GeminiStarCraftAgent(BaseGeminiAgent, StarCraftAgent):
    """
    Gemini-based agent for StarCraft II.
    Inherits Gemini initialization from BaseGeminiAgent and game logic from StarCraftAgent.
    """

    pass
