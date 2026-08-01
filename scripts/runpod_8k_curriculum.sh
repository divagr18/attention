#!/usr/bin/env bash
set -euo pipefail

# Requires scripts/setup_runpod.sh.  This keeps the full attention training
# path unchanged; do not use the laptop-only final-query shortcut.
PYTHON_BIN="${PYTHON_BIN:-.venv-runpod/bin/python}"
COMMON=(--variant learned --task-family multirecord --context 8192 --local-window 64 --block-size 64 --top-blocks 1 --top-tokens 1 --batch-size 2 --steps 1200 --eval-batches 64 --d-model 64 --layers 1 --heads 4 --learning-rate 0.0005 --router-loss-weight 1.0)

${PYTHON_BIN} experiments/train_tiny_transformer.py "${COMMON[@]}" --retrieval-unit span --retrieval-width 3 --checkpoint results/transformer_learned_span3_8k.pt --output results/transformer_learned_span3_8k.json
${PYTHON_BIN} experiments/train_tiny_transformer.py "${COMMON[@]}" --retrieval-unit span --retrieval-width 8 --init-checkpoint results/transformer_learned_span3_8k.pt --checkpoint results/transformer_learned_span8_8k.pt --output results/transformer_learned_span8_8k.json
${PYTHON_BIN} experiments/train_tiny_transformer.py "${COMMON[@]}" --retrieval-unit span --retrieval-width 16 --init-checkpoint results/transformer_learned_span8_8k.pt --checkpoint results/transformer_learned_span16_8k.pt --output results/transformer_learned_span16_8k.json
${PYTHON_BIN} experiments/train_tiny_transformer.py "${COMMON[@]}" --retrieval-unit span --retrieval-width 32 --init-checkpoint results/transformer_learned_span16_8k.pt --checkpoint results/transformer_learned_span32_8k.pt --output results/transformer_learned_span32_8k.json
${PYTHON_BIN} experiments/train_tiny_transformer.py "${COMMON[@]}" --retrieval-unit page --init-checkpoint results/transformer_learned_span32_8k.pt --checkpoint results/transformer_learned_page_staged_8k.pt --output results/transformer_learned_page_staged_8k.json

${PYTHON_BIN} experiments/triton_kvcache_transformer_decode.py --checkpoint results/transformer_learned_page_staged_8k.pt --output results/triton_kvcache_page_staged_decode_8k.json
