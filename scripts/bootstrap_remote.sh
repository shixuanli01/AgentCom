#!/usr/bin/env bash
set -euo pipefail

CUDA_CHANNEL="${1:-cu128}"
VENV_PATH="${VENV_PATH:-.venv-agentcom}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$CUDA_CHANNEL" != "cu128" && "$CUDA_CHANNEL" != "cu124" ]]; then
  echo "Usage: $0 [cu128|cu124]" >&2
  exit 2
fi

"$PYTHON_BIN" -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ "$CUDA_CHANNEL" == "cu128" ]]; then
  "$VENV_PATH/bin/python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1+cu128
else
  "$VENV_PATH/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 torch
fi

"$VENV_PATH/bin/python" -m pip install \
  transformers==4.51.3 \
  datasets==3.6.0 \
  accelerate==1.7.0 \
  numpy==2.2.6 \
  tqdm==4.67.1 \
  pytest

echo
echo "Environment ready. Activate it with:"
echo "  source $VENV_PATH/bin/activate"
