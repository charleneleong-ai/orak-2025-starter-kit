from langchain_google_vertexai import ChatVertexAI

from config.agent_config import GeminiConfig
from config.base import WandbConfig
from agents.twenty_fourty_eight.base import TwentyFourtyEightAgent

class GeminiTwentyFourtyEightAgent(TwentyFourtyEightAgent):
    config: GeminiConfig
    
    def __init__(
        self, 
        config: GeminiConfig = None, 
        wandb_config: WandbConfig = None,
    ):
        config = config or GeminiConfig()
        wandb_config = wandb_config or WandbConfig()

        super().__init__(
            config=config,
            wandb_config=wandb_config,
        )
        
        self._llm = ChatVertexAI(
            model_name=self.config.model,
            temperature=self.config.temperature,
            project=self.config.gcp_project,
            location=self.config.gcp_location,
        )
        
    @property
    def AGENT_TAGS(self):
        return ["2048", "gemini", self.config.model, "vertex-ai"]
