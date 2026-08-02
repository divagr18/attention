#!/usr/bin/env bash
set -euo pipefail

# E28: distinguish insufficient 32K adaptation from a failed scale transfer.
# Continue the first 32K checkpoint for another 1,200 updates without changing
# context length, model depth, router budget, or retrieval granularity.
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
BASE="results/transformer_learned_pagefine1_2layer_32k.pt"

if [[ ! -f "${BASE}" ]]; then
  echo "Missing required first-pass 32K checkpoint: ${BASE}" >&2
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
  --checkpoint results/transformer_learned_pagefine1_2layer_32k_2400.pt \
  --output results/transformer_learned_pagefine1_2layer_32k_2400.json

"${PYTHON_BIN}" experiments/triton_kvcache_transformer_decode.py \
  --checkpoint results/transformer_learned_pagefine1_2layer_32k_2400.pt \
  --output results/triton_kvcache_pagefine1_2layer_decode_32k_2400.json
