#!/usr/bin/env bash
# Launch SGLang serving the same Qwen model qwen_serve.sh uses — the SGLang arm
# of the vLLM-vs-SGLang canary A/B (see experiments/sglang_vs_vllm_ab/README.md).
#
# Why SGLang: its RadixAttention auto-reuses KV cache across requests sharing a
# prefix. The orak agentic-eval workload is its best case — every step re-sends
# a long, near-identical prompt (system + game rules + history), and the many
# parallel rollouts all share the same system prefix. RadixAttention is ON by
# default (disable with --disable-radix-cache for the no-cache control arm).
#
# OpenAI-compatible server on the SAME port (8000) and the SAME tool-call /
# reasoning-parser contract as qwen_serve.sh, so the agent harness and the
# canary's pre-flight (`curl :8000/v1/models`) are unchanged — only the backend
# differs. Only one model fits on a single A100-40GB: stop any vLLM first
# (`pkill -f vllm.entrypoints.openai.api_server`).
#
# Separate serving venv at /workspace/sglang-serve/.venv (mirrors the
# vllm-serve venv). Bootstrap once:
#   uv venv /workspace/sglang-serve/.venv --python 3.11
#   /workspace/sglang-serve/.venv/bin/pip install "sglang[all]"
#
# Usage:
#   ./serving/sglang_serve.sh                                  # default model
#   ./serving/sglang_serve.sh Qwen/Qwen3.5-35B-A3B-GPTQ-Int4   # explicit
#   SGLANG_RADIX=0 ./serving/sglang_serve.sh                   # no-cache control arm
#
# Env overrides:
#   SGLANG_MODEL, SGLANG_PORT, SGLANG_MEM_FRAC, SGLANG_CTX_LEN, SGLANG_VENV,
#   SGLANG_TP, SGLANG_RADIX (0 disables RadixAttention), SGLANG_REASONING_PARSER
set -euo pipefail

MODEL="${1:-${SGLANG_MODEL:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}}"
[[ $# -gt 0 ]] && shift
EXTRA_ARGS=("$@")
PORT="${SGLANG_PORT:-8000}"
VENV="${SGLANG_VENV:-/workspace/sglang-serve/.venv}"
TP="${SGLANG_TP:-1}"

# Match qwen_serve.sh's sizing split: Qwen3.6 weights are larger, so trim the
# static memory fraction + context to keep A100-40GB headroom.
if [[ "${MODEL,,}" == *"qwen3.6"* ]]; then
    MEM_FRAC="${SGLANG_MEM_FRAC:-0.85}"
    CTX_LEN="${SGLANG_CTX_LEN:-12288}"
else
    MEM_FRAC="${SGLANG_MEM_FRAC:-0.90}"
    CTX_LEN="${SGLANG_CTX_LEN:-16384}"
fi

# Auto-enable the qwen3 reasoning parser for Thinking / Qwen3.6 / Reasoning
# variants so SGLang separates <think> blocks into reasoning_content and
# returns clean tool-call output (same contract as the vLLM arm). Override with
# SGLANG_REASONING_PARSER=foo, or ' ' to disable.
if [[ -z "${SGLANG_REASONING_PARSER+x}" ]]; then
    if [[ "${MODEL,,}" == *"thinking"* ]] || \
       [[ "${MODEL,,}" == *"qwen3.6"* ]] || \
       [[ "${MODEL,,}" == *"reasoning"* ]]; then
        SGLANG_REASONING_PARSER="qwen3"
    else
        SGLANG_REASONING_PARSER=""
    fi
fi

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

radix_args=()
[[ "${SGLANG_RADIX:-1}" == "0" ]] && radix_args=(--disable-radix-cache)

reasoning_args=()
if [[ -n "${SGLANG_REASONING_PARSER// }" ]]; then
    reasoning_args=(--reasoning-parser "${SGLANG_REASONING_PARSER}")
fi

echo "============================================"
echo "  SGLang / Qwen — canary A/B (SGLang arm)"
echo "  Model:       ${MODEL}"
echo "  Port:        ${PORT}"
echo "  Mem frac:    ${MEM_FRAC}"
echo "  Max Ctx:     ${CTX_LEN}"
echo "  TP:          ${TP}"
echo "  RadixCache:  ${SGLANG_RADIX:-1} (1=on)"
echo "  Reasoning:   ${SGLANG_REASONING_PARSER:-<none>}"
echo "  VENV:        ${VENV}"
echo "============================================"

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "ERROR: SGLang serving venv not found at ${VENV}"
    echo "Bootstrap once:"
    echo "  uv venv ${VENV} --python 3.11"
    echo "  ${VENV}/bin/pip install 'sglang[all]'"
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ "${used_mib:-0}" -gt 5000 ]]; then
        echo "WARNING: GPU already showing ${used_mib} MiB used. Stop the vLLM arm first:"
        echo "    pkill -f 'vllm.entrypoints.openai.api_server'"
        echo "Continuing in 5s..." && sleep 5
    fi
fi

# Qwen tool calling in SGLang uses the qwen25 parser. RadixAttention (prefix
# cache) is the default; the no-cache control arm passes --disable-radix-cache.
exec "${VENV}/bin/python" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --tp "${TP}" \
    --mem-fraction-static "${MEM_FRAC}" \
    --context-length "${CTX_LEN}" \
    --tool-call-parser qwen25 \
    --trust-remote-code \
    "${radix_args[@]}" \
    "${reasoning_args[@]}" \
    "${EXTRA_ARGS[@]}"
