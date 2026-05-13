#!/usr/bin/env bash
# Launch vLLM serving Qwen 3.6 27B FP8 — A100-40GB ceiling-check experiment.
#
# Companion to serving/gemma_serve.sh. Uses the same separate serving venv
# (/workspace/vllm-serve/.venv). The Gemma vLLM process MUST be stopped
# before launching this — only one model fits on a single A100 40GB.
#
# Background: Stage A→G experiments (PRs #31, #62, #66, #67, #68, #69, #70)
# established the pokemon milestone-4 ceiling lives in LLM reasoning at the
# milestone boundary, not action/procedure/reflection layers. Stage H is the
# ceiling check: does a newer dense model (Qwen 3.6 27B, Apr 2026) break
# past the 57.14% plateau where Gemma 4-26B-A4B saturates?
#
# Sizing on A100-40GB:
#   FP8 27B weights:  ~27 GB
#   KV cache budget:  ~7 GB at gpu-memory-utilization=0.90
#   max_model_len:    8192 (down from 16384 for gemma) keeps KV cache fit
#
# Usage:
#   ./serving/qwen_serve.sh                          # Qwen3.6-27B-FP8 (default)
#   ./serving/qwen_serve.sh Qwen/Qwen3.5-27B-FP8     # older sibling
#   ./serving/qwen_serve.sh Qwen/Qwen3.5-35B-A3B-GPTQ-Int4   # MoE-A3B alternative
#
# Env overrides:
#   QWEN_MODEL, QWEN_PORT, QWEN_GPU_UTIL, QWEN_MAX_MODEL_LEN, QWEN_VENV
set -euo pipefail

MODEL="${1:-${QWEN_MODEL:-Qwen/Qwen3.6-27B-FP8}}"
PORT="${QWEN_PORT:-8000}"
GPU_UTIL="${QWEN_GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
VENV="${QWEN_VENV:-/workspace/vllm-serve/.venv}"

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

echo "============================================"
echo "  vLLM / Qwen ceiling-check (Stage H)"
echo "  Model:       ${MODEL}"
echo "  Port:        ${PORT}"
echo "  GPU Util:    ${GPU_UTIL}"
echo "  Max Ctx:     ${MAX_MODEL_LEN}"
echo "  HF_HOME:     ${HF_HOME}"
echo "  VENV:        ${VENV}"
echo "============================================"

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "ERROR: serving venv not found at ${VENV}"
    echo "Bootstrap once:  ./serving/bootstrap_venv.sh"
    exit 1
fi

# Pre-flight: any GPU in use?
if command -v nvidia-smi >/dev/null 2>&1; then
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ "${used_mib:-0}" -gt 5000 ]]; then
        echo "WARNING: GPU already showing ${used_mib} MiB used. Stop the gemma vLLM first:"
        echo "    pkill -f 'vllm.entrypoints.openai.api_server'"
        echo "Otherwise this launch will fail or OOM."
        echo "Continuing in 5s..." && sleep 5
    fi
fi

# Qwen 3.x uses hermes-style tool calling in vLLM
exec "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype auto
