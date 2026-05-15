#!/usr/bin/env bash
# Launch vLLM serving Qwen 3.5 35B-A3B-GPTQ-Int4 — Stage H ceiling check on A100-40GB.
#
# Companion to serving/gemma_serve.sh. Uses the same separate serving venv
# (/workspace/vllm-serve/.venv). The Gemma vLLM process MUST be stopped
# before launching this — only one model fits on a single A100 40GB.
#
# Background: Stage A→G experiments (PRs #31, #62, #66, #67, #68, #69, #70)
# established the pokemon milestone-4 ceiling lives in LLM reasoning at the
# milestone boundary, not action/procedure/reflection layers. Stage H is the
# ceiling check.
#
# Default model history:
#   Initial plan:  Qwen/Qwen3.6-27B-FP8 (newest dense, Apr 2026)
#   Why pivoted:   A100 has no native FP8 compute. vLLM falls back to
#                  Marlin INT4 emulation — inference was ~1 min/step in
#                  trial runs, 15+ hours for n=3. Not viable.
#   Current:       Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 (Feb 2026 MoE, 3B
#                  active params, native INT4 on A100). Expected
#                  ~5-10x faster than FP8 27B.
#
# Sizing on A100-40GB:
#   GPTQ-Int4 35B-A3B weights: ~17 GB
#   KV cache budget:           ~15 GB at gpu-memory-utilization=0.90, len=16384
#   Headroom for activations + CUDA graph: ~7 GB
#
# Usage:
#   ./serving/qwen_serve.sh                                   # 35B-A3B Int4 (default)
#   ./serving/qwen_serve.sh Qwen/Qwen3.6-27B-FP8              # original FP8 plan (slow on A100)
#   ./serving/qwen_serve.sh Qwen/Qwen3.6-35B-A3B-FP8          # newer MoE FP8 (would also be slow)
#
# Env overrides:
#   QWEN_MODEL, QWEN_PORT, QWEN_GPU_UTIL, QWEN_MAX_MODEL_LEN, QWEN_VENV
set -euo pipefail

MODEL="${1:-${QWEN_MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}}"
PORT="${QWEN_PORT:-8000}"
GPU_UTIL="${QWEN_GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-16384}"
VENV="${QWEN_VENV:-/workspace/vllm-serve/.venv}"

# Auto-enable Qwen reasoning parser for Thinking-mode variants so vLLM strips
# <think>...</think> blocks server-side and returns them in `reasoning_content`
# (the agent harness sees only the post-think tool-call output). Override
# with QWEN_REASONING_PARSER=foo to force, or QWEN_REASONING_PARSER=' ' to disable.
if [[ -z "${QWEN_REASONING_PARSER+x}" ]]; then
    if [[ "${MODEL,,}" == *"thinking"* ]]; then
        QWEN_REASONING_PARSER="qwen3"
    else
        QWEN_REASONING_PARSER=""
    fi
fi

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
        echo "WARNING: GPU already showing ${used_mib} MiB used. Stop any other vLLM first:"
        echo "    pkill -f 'vllm.entrypoints.openai.api_server'"
        echo "Otherwise this launch will fail or OOM."
        echo "Continuing in 5s..." && sleep 5
    fi
fi

# Qwen 3.x uses hermes-style tool calling in vLLM.
# For GPTQ-Int4 MoE on A100, CUDA graph capture is fine (native INT4
# compute). For FP8 fallback path (no native FP8 on A100), add
# --enforce-eager to save GPU memory at the cost of throughput.
#
# Thinking-mode variants (model name contains "thinking") auto-enable
# --reasoning-parser qwen3 so vLLM separates <think> blocks into
# `reasoning_content` and returns clean tool-call output in `content`.
reasoning_args=()
if [[ -n "${QWEN_REASONING_PARSER// }" ]]; then
    reasoning_args=(--reasoning-parser "${QWEN_REASONING_PARSER}")
    echo "[serve] reasoning parser: ${QWEN_REASONING_PARSER}"
fi

exec "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype auto \
    "${reasoning_args[@]}"
