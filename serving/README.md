# Local Model Serving for MACLA

Serve local LLMs via OpenAI-compatible APIs for use with MACLA agents.

Three supported backends:

| Backend | Best for | API endpoint | Setup |
|---------|----------|--------------|-------|
| **vLLM** | A100/H100 GPU servers | `localhost:8000/v1` | `pip install vllm` |
| **Ollama** | Easy local Mac/Linux | `localhost:11434/v1` | `brew install ollama` |
| **MLX** | Max perf Apple Silicon | `localhost:8081/v1` | `pip install mlx-lm` |

All expose the same OpenAI-compatible chat/completions API — the MACLA agent code is identical across backends.

## Thinking models and `max_tokens`

Game agents need chain-of-thought to learn — strategic decisions, reading
game state, planning ahead. Use thinking models (Qwen3, DeepSeek-R1) and
size `max_tokens` to fit both internal reasoning AND the structured JSON
action: **16384 is a safe default**.

If a step exits with `Could not parse response content as the length limit
was reached`, increase `max_tokens` further. Don't disable thinking —
it's how the agent learns from each game state.

**Hardware caveat**: Qwen3 thinking on Apple Silicon is slow (~3 min/step
for 8B-4bit). Use Mac for plumbing tests; run real game evaluations on
A100/H100 via vLLM where thinking + 16k tokens completes in seconds.

## A100/H100 (vLLM)

Production GPU serving with continuous batching, PagedAttention, tensor parallelism, and FP8 quantization.

### Recommended models

| Model | Params (active) | A100 GPUs | Quantization | Notes |
|-------|-----------------|-----------|--------------|-------|
| `Qwen/Qwen3-32B` | 32B | 1x 80GB | none | Strong reasoning, best single-GPU option |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 17B | 1x 40GB | none | Good structured output |
| `deepseek-ai/DeepSeek-R1-0528` | 37B active | 2x 80GB | none | Reasoning-focused |
| `moonshotai/Kimi-K2-Instruct` | 32B active (1T total) | 4x 80GB | fp8 | Top agentic performance |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B | 1x 80GB | none | Fast inference |

### Bare metal

```bash
pip install vllm

# Qwen3-32B on 1x A100-80GB
./serving/vllm_serve.sh Qwen/Qwen3-32B 1

# Kimi K2 on 4x A100 with FP8
./serving/vllm_serve.sh moonshotai/Kimi-K2-Instruct 4 fp8
```

### Docker

```bash
# Default: Qwen3-32B
docker compose -f serving/docker-compose.vllm.yaml up

# Kimi K2 on 4 GPUs
VLLM_MODEL=moonshotai/Kimi-K2-Instruct VLLM_TP=4 VLLM_QUANT=fp8 \
  docker compose -f serving/docker-compose.vllm.yaml up
```

### Verify

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-32B", "messages": [{"role": "user", "content": "Say hello"}]}'
```

## Mac — Ollama

Easiest setup. Uses MLX under the hood on Apple Silicon for near-native performance.

```bash
brew install ollama
ollama serve  # starts on port 11434

# Pull a model
ollama pull qwen3:8b     # ~5GB, runs on 16GB Mac
ollama pull qwen3:32b    # ~20GB, needs 36GB+ RAM
```

API available at `http://localhost:11434/v1`.

## Mac — MLX (mlx-lm)

Best raw inference speed on Apple Silicon (~20-30% faster than Ollama for prompt eval). More control over quantization.

```bash
# Install via project optional extras
uv pip install -e ".[local-mac]"

# Serve a model (auto-downloads from HuggingFace ~5GB)
mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8081
```

API available at `http://localhost:8081/v1`.

## Running MACLA with local model

```bash
# A100 with vLLM (default config)
python run.py -c local --local --games twenty_fourty_eight

# Mac with Ollama (override via env vars)
LOCAL_BASE_URL=http://localhost:11434/v1 LOCAL_MODEL=qwen3:8b \
  python run.py -c local --local --games twenty_fourty_eight

# Mac with MLX (port 8081)
LOCAL_BASE_URL=http://localhost:8081/v1 LOCAL_MODEL=mlx-community/Qwen3-8B-4bit \
  python run.py -c local --local --games twenty_fourty_eight
```

## Environment variables

### vLLM server

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `Qwen/Qwen3-32B` | Model to serve |
| `VLLM_TP` | `1` | Tensor parallel GPUs |
| `VLLM_QUANT` | (none) | Quantization: `awq`, `gptq`, `fp8` |
| `VLLM_PORT` | `8000` | API port |
| `VLLM_GPU_UTIL` | `0.90` | GPU memory utilization |
| `VLLM_MAX_MODEL_LEN` | `8192` | Max context length |
| `HF_TOKEN` | (none) | HuggingFace token for gated models |

### MACLA agent overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_BASE_URL` | (from YAML) | Override base URL |
| `LOCAL_MODEL` | (from YAML) | Override model name |
