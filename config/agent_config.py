"""
Centralised configuration for agents.
All agent configurations are defined here and can be logged to wandb.
"""

import os
from dataclasses import dataclass, field
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

    model: str = "gemini-pro-3-preview"
    refinement_model: str = ""  # optional separate model for MACLA refinement (e.g. gemini-2.5-pro)
    temperature: float = 0.1
    gcp_project: Optional[str] = None
    gcp_location: str = "us-central1"
    thinking_level: str = "high"  # low, high
    game_config_path: str = ""  # path to game config YAML for UnifiedMaclaAgent
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
            "refinement_model": self.refinement_model,
            "temperature": self.temperature,
            "gcp_project": self.gcp_project,
            "gcp_location": self.gcp_location,
            "thinking_level": self.thinking_level,
            "game_config_path": self.game_config_path,
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


@dataclass
class LocalConfig(AgentConfig):
    __pydantic_config__ = ConfigDict(extra="forbid")
    """Configuration for local model inference via OpenAI-compatible API.

    Supported backends (server_type):
      vllm   — Production GPU serving (A100/H100). Continuous batching,
               PagedAttention, tensor parallelism, FP8 quantization.
               Best throughput for multi-request workloads.
               Setup: pip install vllm && python -m vllm.entrypoints.openai.api_server
      ollama — Easiest local setup (Mac/Linux). Uses MLX on Apple Silicon
               for near-native performance. One-command install + model pull.
               Setup: brew install ollama && ollama pull qwen3:8b
      mlx    — Apple-native ML framework via mlx-lm. Best raw inference
               speed on Apple Silicon (~20-30% faster than llama.cpp).
               More control over quantization than Ollama.
               Setup: pip install mlx-lm && mlx_lm.server --model <hf-repo>

    All backends expose an OpenAI-compatible chat/completions API,
    so the agent code (ChatOpenAI) is identical — only the URL changes.

    Env var overrides (no YAML change needed):
      LOCAL_BASE_URL  — override base_url  (e.g. http://remote-gpu:8000/v1)
      LOCAL_MODEL     — override model     (e.g. qwen3:8b)
    """

    model: str = "Qwen/Qwen3-32B"
    temperature: float = 0.7
    base_url: str = "http://localhost:8000/v1"  # vLLM default
    api_key: str = "local"  # dummy — required by ChatOpenAI but ignored by local servers
    max_tokens: int = 2048
    track: str = "TRACK1"
    game_config_path: str = ""
    server_type: str = "vllm"  # vllm | ollama | mlx
    # Most local models are text-only. Set True only for vision-capable models
    # (e.g. Qwen2.5-VL, Llama-4-Scout, Pixtral). Otherwise the agent will
    # send images as base64 data URLs which most servers reject.
    supports_vision: bool = False
    # Vendor-specific extras forwarded to the inference server via OpenAI
    # `extra_body`. Examples:
    #   vLLM guided decoding: {"guided_json": {...}}
    #   Sampling overrides:   {"top_k": 40, "repetition_penalty": 1.05}
    #   Qwen3 chat template:  {"chat_template_kwargs": {...}}
    extra_body: dict = field(default_factory=dict)
    # vLLM-specific (ignored by Ollama/MLX)
    tensor_parallel_size: int = 1  # num GPUs for tensor parallelism
    gpu_memory_utilization: float = 0.90
    quantization: str = ""  # awq, gptq, fp8, or empty for none
    max_model_len: int = 8192
    # MACLA tuning (per-game overrides via YAML)
    macla_max_theta: float = 0.40
    macla_min_theta: float = 0.05
    macla_theta_base: float = 0.30
    macla_theta_decay: float = 0.002
    macla_warmup_steps: int = 0
    macla_n_min_s: int = 3
    macla_n_min_f: int = 3

    def __post_init__(self):
        # Env var overrides for easy CLI switching
        self.base_url = os.environ.get("LOCAL_BASE_URL", self.base_url)
        self.model = os.environ.get("LOCAL_MODEL", self.model)

        # Resolve server_type to default base_url
        url_defaults = {
            "vllm": "http://localhost:8000/v1",
            "ollama": "http://localhost:11434/v1",
            "mlx": "http://localhost:8081/v1",  # 8080 commonly taken (SSH, llama-server)
        }
        if self.base_url == url_defaults.get("vllm") and self.server_type != "vllm":
            self.base_url = url_defaults.get(self.server_type, self.base_url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "server_type": self.server_type,
            "supports_vision": self.supports_vision,
            "extra_body": self.extra_body,
            "max_tokens": self.max_tokens,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization,
            "max_model_len": self.max_model_len,
            "track": self.track,
            "macla_max_theta": self.macla_max_theta,
            "macla_min_theta": self.macla_min_theta,
            "macla_theta_base": self.macla_theta_base,
            "macla_theta_decay": self.macla_theta_decay,
            "macla_warmup_steps": self.macla_warmup_steps,
            "macla_n_min_s": self.macla_n_min_s,
            "macla_n_min_f": self.macla_n_min_f,
        }


@dataclass
class PoetiqConfig(GeminiConfig):
    """Configuration for Poetiq self-evolving agent."""
    # Evolution parameters
    max_iterations: int = 10  # Max evolution cycles per episode
    max_solutions: int = 5  # Number of solutions to keep in history
    selection_probability: float = 1.0  # Probability of showing history in prompt
    improving_order: bool = True  # Show solutions worst->best
    return_best_result: bool = True  # Return best across all iterations

    # LLM parameters
    request_timeout: int = 60 * 5  # 5 minutes per LLM call
    per_iteration_retries: int = 2  # Retries per evolution

    # Random seed
    seed: int = 0


AgentConfig = OpenAIConfig | GeminiConfig | LocalConfig | PoetiqConfig
