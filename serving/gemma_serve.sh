#!/usr/bin/env bash
# Launch vLLM serving Gemma-3n-E4B-it (multimodal) for A100-40GB.
#
# Defaults are sized for a single A100-40GB. Weights ~8GB bf16,
# leaving plenty of room for KV cache at max_model_len=16384.
#
# Local cache (no download needed if already present):
#   /workspace/.hf_home/hub/models--unsloth--gemma-3n-E4B-it
#
# Usage:
#   ./serving/gemma_serve.sh                            # bf16 (default)
#   ./serving/gemma_serve.sh unsloth/gemma-3n-E4B-it     # explicit model
#
# Override via env:
#   GEMMA_MODEL, GEMMA_PORT, GEMMA_GPU_UTIL, GEMMA_MAX_MODEL_LEN
#
# To serve with the local LoRA adapter trained at /workspace/gemma4_rl/gemma_4_lora,
# pass --enable-lora and --lora-modules (vLLM mounts adapters at request time).
set -euo pipefail

MODEL="${1:-${GEMMA_MODEL:-unsloth/gemma-3n-E4B-it}}"
PORT="${GEMMA_PORT:-8000}"
GPU_UTIL="${GEMMA_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-16384}"

# Use the project HF cache so we don't re-download
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

echo "============================================"
echo "  vLLM / Gemma-3n-E4B-it"
echo "  Model:       ${MODEL}"
echo "  Port:        ${PORT}"
echo "  GPU Util:    ${GPU_UTIL}"
echo "  Max Ctx:     ${MAX_MODEL_LEN}"
echo "  HF_HOME:     ${HF_HOME}"
echo "============================================"

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser pythonic \
    --dtype bfloat16
