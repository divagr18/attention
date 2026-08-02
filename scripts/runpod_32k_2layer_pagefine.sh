#!/usr/bin/env bash
set -euo pipefail

# E27: scale the validated 16K two-layer, one-center page-fine model to 32K.
# This training path still materializes full attention scores, so use a pod
# with at least 48 GB VRAM (80 GB gives more evaluation headroom).
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
BASE="results/transformer_learned_pagefine1_2layer_16k.pt"

if [[ ! -f "${BASE}" ]]; then
  echo "Missing required 16K checkpoint: ${BASE}" >&2
  exit 1
fi

"${PYTHON_BIN}" experiments/train_tiny_transformer.py \
  --variant learned --retrieval-unit page_fine --retrieval-width 4 \
  --task-family multirecord \
  --context 32768 --local-window 64 --block-size 64 \
  --top-blocks 1 --top-tokens 1 --batch-size 1 \
  --steps 1200 --eval-batches 256 \
  --d-model 64 --layers 2 --heads 4 \
  --learning-rate 0.0003 --router-loss-weight 1.0 \
  --gradient-checkpointing \
  --init-checkpoint "${BASE}" \
  --checkpoint results/transformer_learned_pagefine1_2layer_32k.pt \
  --output results/transformer_learned_pagefine1_2layer_32k.json

"${PYTHON_BIN}" experiments/triton_kvcache_transformer_decode.py \
  --checkpoint results/transformer_learned_pagefine1_2layer_32k.pt \
  --output results/triton_kvcache_pagefine1_2layer_decode_32k.json
