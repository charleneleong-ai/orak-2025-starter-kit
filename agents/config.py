from agents.random_mario import RandomMarioAgent
from agents.super_mario.base import SuperMarioAgent as BaseSuperMarioAgent

from agents.random_pokemon import RandomPokemonAgent

from agents.random_twenty_fourty_eight import RandomTwentyFourtyEightAgent
from agents.twenty_fourty_eight.base import TwentyFourtyEightAgent as BaseTwentyFourtyEightAgent


from agents.random_starcraft import RandomStarCraftAgent
# from agents.openai_starcraft import OpenAIStarCraftAgent

PokemonAgent = RandomPokemonAgent

# TwentyFourtyEightAgent = OpenAITwentyFourtyEightAgent
TwentyFourtyEightAgent = RandomTwentyFourtyEightAgent

# SuperMarioAgent = OpenAIMarioAgent
SuperMarioAgent = RandomMarioAgent

# StarCraftAgent = OpenAIStarCraftAgent
StarCraftAgent = RandomStarCraftAgent

# Default Generic Agents
GenericSuperMarioAgent = BaseSuperMarioAgent
GenericTwentyFourtyEightAgent = BaseTwentyFourtyEightAgent
