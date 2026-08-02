#!/usr/bin/env bash
set -euo pipefail

# E29: extend the validated 32K two-layer page-fine model to 64K. The exact
# windowed local-attention implementation removes the previous L^2 workspace.
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
BASE="results/transformer_learned_pagefine1_2layer_32k_2400.pt"

if [[ ! -f "${BASE}" ]]; then
  echo "Missing required converged 32K checkpoint: ${BASE}" >&2
  exit 1
fi

"${PYTHON_BIN}" experiments/train_tiny_transformer.py \
  --variant learned --retrieval-unit page_fine --retrieval-width 4 \
  --task-family multirecord \
  --context 65536 --local-window 64 --block-size 64 \
  --top-blocks 1 --top-tokens 1 --batch-size 1 \
  --steps 2400 --eval-batches 256 \
  --d-model 64 --layers 2 --heads 4 \
  --learning-rate 0.0003 --router-loss-weight 1.0 \
  --gradient-checkpointing \
  --init-checkpoint "${BASE}" \
  --checkpoint results/transformer_learned_pagefine1_2layer_64k.pt \
  --output results/transformer_learned_pagefine1_2layer_64k.json

"${PYTHON_BIN}" experiments/triton_kvcache_transformer_decode.py \
  --checkpoint results/transformer_learned_pagefine1_2layer_64k.pt \
  --output results/triton_kvcache_pagefine1_2layer_decode_64k.json
