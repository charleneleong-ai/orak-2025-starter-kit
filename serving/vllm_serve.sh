#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server on A100/H100.
#
# Usage:
#   ./serving/vllm_serve.sh                          # defaults: Qwen3-32B, 1 GPU
#   ./serving/vllm_serve.sh Qwen/Qwen3-32B 1         # explicit model + TP
#   ./serving/vllm_serve.sh moonshotai/Kimi-K2-Instruct 4 fp8  # Kimi K2, 4-GPU TP, FP8
#
# Env vars (override defaults):
#   VLLM_MODEL, VLLM_TP, VLLM_QUANT, VLLM_PORT, VLLM_GPU_UTIL, VLLM_MAX_MODEL_LEN
set -euo pipefail

MODEL="${1:-${VLLM_MODEL:-Qwen/Qwen3-32B}}"
TP="${2:-${VLLM_TP:-1}}"
QUANT="${3:-${VLLM_QUANT:-}}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "============================================"
echo "  vLLM Server for MACLA"
echo "  Model:       ${MODEL}"
echo "  TP:          ${TP}"
echo "  Quant:       ${QUANT:-none}"
echo "  Port:        ${PORT}"
echo "  GPU Util:    ${GPU_UTIL}"
echo "  Max Ctx:     ${MAX_MODEL_LEN}"
echo "============================================"

ARGS=(
    --model "${MODEL}"
    --tensor-parallel-size "${TP}"
    --port "${PORT}"
    --gpu-memory-utilization "${GPU_UTIL}"
    --max-model-len "${MAX_MODEL_LEN}"
    --trust-remote-code
    --enable-auto-tool-choice
    --dtype auto
)

if [[ -n "${QUANT}" ]]; then
    ARGS+=(--quantization "${QUANT}")
fi

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
