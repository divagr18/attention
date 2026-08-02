#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
mkdir -p results

# The fused three-level kernel supports up to 4,096 pages with 16-way
# branching. At 1M tokens it compares 144 fixed tree slots to 4,064 flat
# pages while retaining one 8K exact hot window and one retrieved page.
"${PYTHON_BIN}" experiments/benchmark_page_tree.py \
  --context 1048576 --hot-window 8192 --page-size 256 \
  --tree-fanout 16 --tree-beam 1 --tree-leaf-slots 1 --tree-slots 4 \
  --retrieval-pages 1 --heads 4 --head-dim 16 \
  --output results/page_tree_1m_triton_tree.json
