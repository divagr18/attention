#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a CUDA-enabled Runpod template.
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-.venv-runpod}"
CACHE_PATH="${CACHE_PATH:-.cache/runpod}"

command -v nvidia-smi >/dev/null
nvidia-smi

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
"${VENV_PATH}/bin/python" -m pip install -r requirements-runpod.txt

mkdir -p "${CACHE_PATH}/triton"
export TRITON_CACHE_DIR="$(pwd)/${CACHE_PATH}/triton"
"${VENV_PATH}/bin/python" - <<'PY'
import torch
import triton
assert torch.cuda.is_available(), "CUDA is unavailable inside the pod"
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("gpu:", torch.cuda.get_device_name(0))
print("memory_gb:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
PY

echo "Setup complete. Activate with: source ${VENV_PATH}/bin/activate"
