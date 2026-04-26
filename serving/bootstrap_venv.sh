#!/usr/bin/env bash
# Bootstrap the side venv used by gemma_serve.sh.
#
# Why a side venv: gemma-4 ships with model_type=gemma4 which only
# transformers >= 5.5.1 recognises, but the project pins transformers==4.57.1
# for the agent client. Keeping serving deps in a separate venv avoids
# breaking the agent. The agent talks to vLLM over HTTP — version split is safe.
#
# Idempotent: re-running upgrades pins to the latest matching specifiers.
#
# Usage:
#   ./serving/bootstrap_venv.sh                    # uses /workspace/vllm-serve/.venv
#   GEMMA_VENV=/path/to/venv ./serving/bootstrap_venv.sh
set -euo pipefail

VENV="${GEMMA_VENV:-/workspace/vllm-serve/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not on PATH. Install: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "Creating venv at ${VENV} (python ${PYTHON_VERSION})"
    uv venv --python "${PYTHON_VERSION}" "${VENV}"
fi

echo "Installing/upgrading vllm + transformers 5.x + timm into ${VENV}"
VIRTUAL_ENV="${VENV}" uv pip install \
    "vllm==0.19.1" \
    "transformers>=5.5.1" \
    timm

echo
echo "Versions:"
"${VENV}/bin/python" -c "
import vllm, transformers, timm
print(f'  vllm:         {vllm.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  timm:         {timm.__version__}')
"

echo
echo "Done. Launch the server with:  ./serving/gemma_serve.sh"
