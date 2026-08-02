#!/usr/bin/env bash
set -euo pipefail

# Phase 0.4: disambiguate the held-out (82.42%) vs decode-loop (68.75%) gap.
# The decode loop reports exact-in-decode and Triton accuracy per query slot on
# identical samples. Run once with the irregular schedule and once with uniform
# slots. If per-slot exact < held-out, the gap is data weighting/shift; if
# per-slot exact >= held-out but Triton < exact, the fused kernel is the cause.
# Requires the Triton environment (the decode path imports triton_paged_attention).
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
BASE="results/transformer_learned_pagefine1_2layer_32k.pt"

if [[ ! -f "${BASE}" ]]; then
  echo "Missing required 32K checkpoint: ${BASE}" >&2
  exit 1
fi

"${PYTHON_BIN}" experiments/triton_kvcache_transformer_decode.py \
  --checkpoint "${BASE}" \
  --output results/phase0_decode_perslot_irregular_32k.json

"${PYTHON_BIN}" experiments/triton_kvcache_transformer_decode.py \
  --checkpoint "${BASE}" \
  --uniform-slots \
  --output results/phase0_decode_perslot_uniform_32k.json
