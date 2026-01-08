from typing import Any, Optional
from langchain_openai import ChatOpenAI
from loguru import logger

from config.agent_config import OpenAIConfig
from config.base import WandbConfig
from langchain_openai import ChatOpenAI
from agents.twenty_fourty_eight.base import TwentyFourtyEightAgent
from pydantic import Field

class OpenAITwentyFourtyEightAgent(TwentyFourtyEightAgent):
    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0)
    api_key: Optional[str] = Field(default=None)

    
    def __init__(
        self, 
        config: OpenAIConfig = None, 
        wandb_config: WandbConfig = None,
    ):
        # Load configurations
        config = config or OpenAIConfig()
        wandb_config = wandb_config or WandbConfig()
        
        # Initialize with Weave Model
        super().__init__(
            config=config,
            wandb_config=wandb_config
        )  
        # Detect if this is a reasoning model (o1, o3, gpt-5, etc.)
        model_lower = self.config.model.lower()
        is_reasoning_model = any(keyword in model_lower for keyword in ['o1', 'o3', 'gpt-5'])
        
        model_kwargs = {}
        if is_reasoning_model:
            # Pass reasoning_effort if configured
            if hasattr(self.config, "reasoning_effort") and self.config.reasoning_effort:
                model_kwargs["reasoning_effort"] = self.config.reasoning_effort
        
        # Initialize OpenAI client via LangChain
        # For o1 models, temperature is often not supported or restricted
        temperature = self.config.temperature
        if is_reasoning_model:
             # Usually o1-preview / o1-mini don't support temperature
             # We set it to 1.0 (default) or None to let the library handle it
             temperature = 1.0
             
        self._llm = ChatOpenAI(
            model=self.config.model,
            api_key=self.config.api_key,
            temperature=temperature,
            model_kwargs=model_kwargs
        )
        
        logger.info(f"Initialized OpenAI agent with model: {self.config.model}, reasoning: {is_reasoning_model}")

    @property
    def AGENT_TAGS(self):
        return ["openai", self.config.model]
