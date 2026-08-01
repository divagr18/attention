#!/usr/bin/env bash
set -euo pipefail

# Train each fine-selector budget from the same 32-token curriculum checkpoint
# and measure it with the matching Triton span-gather decode path.
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
BASE="results/transformer_learned_span32_16k.pt"
COMMON=(--variant learned --retrieval-unit page_fine --retrieval-width 4 --task-family multirecord --context 16384 --local-window 64 --block-size 64 --top-blocks 1 --batch-size 1 --steps 1200 --eval-batches 64 --d-model 64 --layers 1 --heads 4 --learning-rate 0.0003 --router-loss-weight 1.0 --init-checkpoint "${BASE}")

for budget in 1 2 4 8 16; do
  checkpoint="results/transformer_learned_pagefine${budget}_16k.pt"
  "${PYTHON_BIN}" experiments/train_tiny_transformer.py "${COMMON[@]}" --top-tokens "${budget}" --checkpoint "${checkpoint}" --output "results/transformer_learned_pagefine${budget}_16k.json"
  "${PYTHON_BIN}" experiments/triton_kvcache_transformer_decode.py --checkpoint "${checkpoint}" --output "results/triton_kvcache_pagefine${budget}_decode_16k.json"
done
