from agents.openai_agent import BaseOpenAIAgent
from agents.pokemon_red.base import PokemonRedAgent


class OpenAIPokemonRedAgent(BaseOpenAIAgent, PokemonRedAgent):
    """
    OpenAI-based agent for Pokemon Red.
    Inherits OpenAI initialization from BaseOpenAIAgent and game logic from PokemonRedAgent.
    """

    pass
