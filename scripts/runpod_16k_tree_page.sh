#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
BASE="results/transformer_learned_pagefine1_2layer_16k.pt"
mkdir -p results

if [[ ! -f "${BASE}" ]]; then
  echo "Missing converged page-fine base checkpoint: ${BASE}" >&2
  exit 1
fi

"${PYTHON_BIN}" experiments/test_sparse_attention.py
"${PYTHON_BIN}" experiments/test_hierarchical_page_tree.py

"${PYTHON_BIN}" experiments/train_tiny_transformer.py \
  --variant tree --task-family multirecord \
  --context 16384 --local-window 64 --block-size 64 \
  --top-blocks 1 --retrieval-pages 1 --tree-fanout 16 --tree-beam 1 \
  --tree-summary structural_slots --tree-leaf-slots 1 --tree-slots 4 \
  --retrieval-unit page_fine --retrieval-width 4 --top-tokens 1 --historical-store bf16 \
  --batch-size 1 --steps 0 --eval-batches 256 \
  --d-model 64 --layers 2 --heads 4 --learning-rate 0.0003 \
  --router-loss-weight 1.0 --gradient-checkpointing --freeze-base-model \
  --init-checkpoint "${BASE}" \
  --checkpoint results/transformer_tree_pagefine1_2layer_16k.pt \
  --output results/transformer_tree_pagefine1_2layer_16k.json

"${PYTHON_BIN}" experiments/benchmark_page_tree.py \
  --context 131072 --hot-window 8192 --page-size 256 \
  --tree-fanout 16 --tree-beam 4 --retrieval-pages 4 \
  --heads 4 --head-dim 16 \
  --output results/page_tree_128k.json
