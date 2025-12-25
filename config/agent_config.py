"""
Centralised configuration for agents.
All agent configurations are defined here and can be logged to wandb.
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional, Any
from pydantic import ConfigDict


@dataclass
class AgentConfig:
    class_name: str
    model: str
    temperature: float
    track: Literal["TRACK1", "TRACK2"] = "TRACK1"


@dataclass
class GeminiConfig(AgentConfig):
    __pydantic_config__ = ConfigDict(extra="forbid")
    """Configuration for Gemini (Vertex AI) agent."""

    model: str = "gemini-pro-3"
    temperature: float = 0.1
    gcp_project: Optional[str] = None
    gcp_location: str = "us-central1"
    thinking_level: str = "high"  # low,, high
    track: str = "TRACK1"

    def __post_init__(self):
        # Load from environment
        self.gcp_project = os.environ.get("GCP_PROJECT", self.gcp_project)
        self.gcp_location = os.environ.get("GCP_LOCATION", self.gcp_location)

        if not self.gcp_project:
            raise ValueError("GCP_PROJECT environment variable not set")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for wandb logging."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "gcp_project": self.gcp_project,
            "gcp_location": self.gcp_location,
            "thinking_level": self.thinking_level,
            "track": self.track,
        }


@dataclass
class OpenAIConfig(AgentConfig):
    __pydantic_config__ = ConfigDict(extra="forbid")
    """Configuration for OpenAI agent."""

    model: str = "gpt-5-nano"
    temperature: float = 0.1
    reasoning_effort: str = "high"  # low, medium, high
    max_tokens: Optional[int] = None
    track: str = "TRACK1"
    api_key: str = os.environ.get("OPENAI_API_KEY")

    def __post_init__(self):
        # Validate OpenAI API key exists
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for wandb logging."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "track": self.track,
        }


AgentConfig = OpenAIConfig | GeminiConfig
