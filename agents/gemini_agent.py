from typing import Optional
from langchain_google_vertexai import ChatVertexAI
from loguru import logger
from pydantic import Field, PrivateAttr
from config.agent_config import GeminiConfig
from config.base import WandbConfig
from agents.base import BaseOrakAgent

class BaseGeminiAgent(BaseOrakAgent):
    """
    Base agent for Gemini models using LangChain via Vertex AI.
    Handles initialization of the ChatVertexAI client.
    """
    model_name: str = Field(default="gemini-pro")
    
    _llm: Optional[ChatVertexAI] = PrivateAttr(default=None)

    def __init__(
        self, 
        config: GeminiConfig = None, 
        wandb_config: WandbConfig = None,
        **kwargs
    ):
        # Load configurations
        config = config or GeminiConfig()
        wandb_config = wandb_config or WandbConfig()
        
        super().__init__(
            config=config,
            wandb_config=wandb_config,
            **kwargs
        )  
        
        self._initialize_llm()

    def _initialize_llm(self):
        self._llm = ChatVertexAI(
            model_name=self.config.model,
            temperature=self.config.temperature,
            project=self.config.gcp_project,
            location=self.config.gcp_location,
        )
        
        logger.info(f"Initialized Gemini agent with model: {self.config.model}")

    @property
    def AGENT_TAGS(self):
        return ["gemini", "vertex-ai", self.config.model]
