from agents.gemini_agent import BaseGeminiAgent
from agents.super_mario.base import SuperMarioAgent

class GeminiMarioAgent(BaseGeminiAgent, SuperMarioAgent):
    """
    Gemini-based agent for Super Mario.
    Inherits OpenAI initialization from BaseOpenAIAgent and game logic from SuperMarioAgent.
    """
    pass