#!/usr/bin/env bash
set -euo pipefail

# Phase 0.1: evaluate the E27 32K checkpoint on stratified (sampled) record
# distances and report accuracy by distance decile. Steps=0 means no training;
# this only re-evaluates the existing checkpoint. If accuracy is flat across
# deciles, the "medium-distance" anomaly was a single fixed-position artifact.
# Point --init-checkpoint at the E28 ..._2400 checkpoint to compare the
# continuation as well.
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
BASE="results/transformer_learned_pagefine1_2layer_32k.pt"

if [[ ! -f "${BASE}" ]]; then
  echo "Missing required 32K checkpoint: ${BASE}" >&2
  exit 1
fi

"${PYTHON_BIN}" experiments/train_tiny_transformer.py \
  --variant learned --retrieval-unit page_fine --retrieval-width 4 \
  --task-family multirecord \
  --context 32768 --local-window 64 --block-size 64 \
  --top-blocks 1 --top-tokens 1 --batch-size 1 \
  --steps 0 --eval-batches 256 \
  --d-model 64 --layers 2 --heads 4 \
  --distance-sampling \
  --init-checkpoint "${BASE}" \
  --output results/phase0_distance_decile_32k.json
