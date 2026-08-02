#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
mkdir -p results

"${PYTHON_BIN}" experiments/test_sparse_attention.py
"${PYTHON_BIN}" experiments/test_hierarchical_page_tree.py

"${PYTHON_BIN}" experiments/train_tiny_transformer.py \
  --variant tree --task-family multirecord \
  --context 16384 --local-window 256 --block-size 256 \
  --top-blocks 4 --retrieval-pages 4 --tree-fanout 16 --tree-beam 4 \
  --retrieval-unit page --top-tokens 1 --historical-store bf16 \
  --batch-size 1 --steps 1200 --eval-batches 256 \
  --d-model 64 --layers 2 --heads 4 --learning-rate 0.0005 \
  --router-loss-weight 1.0 \
  --checkpoint results/transformer_tree_page_16k.pt \
  --output results/transformer_tree_page_16k.json

"${PYTHON_BIN}" experiments/benchmark_page_tree.py \
  --context 16384 --hot-window 8192 --page-size 256 \
  --tree-fanout 16 --tree-beam 4 --retrieval-pages 4 \
  --heads 4 --head-dim 16 \
  --output results/page_tree_16k.json
