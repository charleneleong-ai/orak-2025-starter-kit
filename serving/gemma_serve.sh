#!/usr/bin/env bash
# Launch vLLM serving Gemma-4-E4B-it (multimodal) for A100-40GB.
#
# Defaults are sized for a single A100-40GB. Weights ~8GB bf16,
# leaving plenty of room for KV cache at max_model_len=16384.
#
# IMPORTANT: gemma-4 ships with model_type=gemma4 which only transformers
# >= 5.5.1 recognizes. The project's own venv pins transformers==4.57.1
# (required for the agent client side). To avoid breaking the project,
# we keep a *separate* serving venv at /workspace/vllm-serve/.venv with
# vllm==0.19.1 + transformers>=5.5.1. The agent talks to vLLM over the
# OpenAI-compatible HTTP API, so the version split is safe.
#
# Bootstrap the serving venv once:
#   uv venv --python 3.11 /workspace/vllm-serve/.venv
#   VIRTUAL_ENV=/workspace/vllm-serve/.venv uv pip install \
#       "vllm==0.19.1" "transformers>=5.5.1" timm
#
# Local cache (no download needed if already present):
#   /workspace/.hf_home/hub/models--unsloth--gemma-4-E4B-it
#
# Usage:
#   ./serving/gemma_serve.sh                            # bf16 (default)
#   ./serving/gemma_serve.sh unsloth/gemma-4-E4B-it     # explicit model
#
# Override via env:
#   GEMMA_MODEL, GEMMA_PORT, GEMMA_GPU_UTIL, GEMMA_MAX_MODEL_LEN, GEMMA_VENV
#
# To serve with the local LoRA adapter trained at /workspace/gemma4_rl/gemma_4_lora,
# pass --enable-lora and --lora-modules (vLLM mounts adapters at request time).
set -euo pipefail

MODEL="${1:-${GEMMA_MODEL:-unsloth/gemma-4-E4B-it}}"
PORT="${GEMMA_PORT:-8000}"
GPU_UTIL="${GEMMA_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-16384}"
VENV="${GEMMA_VENV:-/workspace/vllm-serve/.venv}"

# Use the project HF cache so we don't re-download
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

echo "============================================"
echo "  vLLM / Gemma-4-E4B-it"
echo "  Model:       ${MODEL}"
echo "  Port:        ${PORT}"
echo "  GPU Util:    ${GPU_UTIL}"
echo "  Max Ctx:     ${MAX_MODEL_LEN}"
echo "  HF_HOME:     ${HF_HOME}"
echo "  VENV:        ${VENV}"
echo "============================================"

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "ERROR: serving venv not found at ${VENV}"
    echo "Bootstrap once with:"
    echo "  uv venv --python 3.11 ${VENV}"
    echo "  VIRTUAL_ENV=${VENV} uv pip install \"vllm==0.19.1\" \"transformers>=5.5.1\" timm"
    exit 1
fi

exec "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser pythonic \
    --dtype bfloat16
