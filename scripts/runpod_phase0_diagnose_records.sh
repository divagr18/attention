#!/usr/bin/env bash
set -euo pipefail

# Phase 0.2: per-record diagnostic on the E27 32K checkpoint. For each of the
# four fixed record positions, force the query to that record and report
# accuracy plus the router's selected span center vs the true record start.
# Isolates whether record index 2 (the "medium" position) is specifically hard
# and whether its router center is off-by-one onto the separator/value token.
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
BASE="results/transformer_learned_pagefine1_2layer_32k.pt"

if [[ ! -f "${BASE}" ]]; then
  echo "Missing required 32K checkpoint: ${BASE}" >&2
  exit 1
fi

"${PYTHON_BIN}" experiments/diagnose_records.py \
  --checkpoint "${BASE}" \
  --batches 128 \
  --output results/phase0_record_diagnostic_32k.json
