#!/usr/bin/env bash
set -euo pipefail

# Matched-compute control for progressive widening.  The staged path starts at
# span3_16k and runs 1,200 updates at widths 8, 16, 32, then page: 4,800 total.
# This control starts at the identical checkpoint and trains page attention for
# the same 4,800 updates.
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"

"${PYTHON_BIN}" experiments/train_tiny_transformer.py \
  --variant learned --retrieval-unit page --task-family multirecord \
  --context 16384 --local-window 64 --block-size 64 --top-blocks 1 --top-tokens 1 \
  --batch-size 1 --steps 4800 --eval-batches 256 --d-model 64 --layers 1 --heads 4 \
  --learning-rate 0.0005 --router-loss-weight 1.0 \
  --init-checkpoint results/transformer_learned_span3_16k.pt \
  --checkpoint results/transformer_learned_page_direct_matched_16k.pt \
  --output results/transformer_learned_page_direct_matched_16k.json

"${PYTHON_BIN}" experiments/triton_kvcache_transformer_decode.py \
  --checkpoint results/transformer_learned_page_direct_matched_16k.pt \
  --output results/triton_kvcache_page_direct_matched_decode_16k.json
