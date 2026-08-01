#!/usr/bin/env bash
set -euo pipefail

# Kept on D: because the host C: drive has insufficient free capacity for the
# CUDA-enabled PyTorch and Triton wheels.  Override these paths after moving the
# WSL worktree to an ext4-backed Linux location for final performance work.
VENV_PATH="${VENV_PATH:-/mnt/d/Attention/.venv-wsl}"
CACHE_PATH="${CACHE_PATH:-/mnt/d/Attention/.wsl-cache}"

mkdir -p "${CACHE_PATH}"
export PIP_CACHE_DIR="${CACHE_PATH}/pip"
export TRITON_CACHE_DIR="${CACHE_PATH}/triton"

python3 -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
"${VENV_PATH}/bin/python" -m pip install triton
"${VENV_PATH}/bin/python" -c "import torch, triton; print('torch', torch.__version__); print('triton', triton.__version__); print('cuda', torch.cuda.is_available())"
