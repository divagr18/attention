# Initial retrieval experiments

`run_selector_benchmark.py` is the first runnable part of the experiment
plan. It generates deterministic synthetic long-context retrieval cases and
evaluates lightweight selector baselines using known evidence spans.

It is intentionally model-independent. The current lexical scoring proxy is
an integration harness, not a substitute for dense-transformer attention.
Replace it with dense teacher attention or learned router scores in the next
stage without changing the case format or metrics.

Run from the repository root:

```powershell
python experiments/run_selector_benchmark.py --cases 200 --seed 7 --output results/selector_smoke.json
```

The output reports block recall, token recall, and score-count proxies for:

- random block selection,
- recency block selection,
- flat token selection,
- hierarchical block-to-token selection,
- oracle selection.

## Actual Transformer test

`train_tiny_transformer.py` is a native GPT-style causal Transformer for the
first architecture comparison. It is not pretrained: the key/value bindings
are random in every example, so successful prediction requires recovering
distant evidence rather than memorizing a mapping.

Run comparable dense, local-only, and oracle-retrieval conditions:

```powershell
python experiments/train_tiny_transformer.py --variant dense --output results/transformer_dense.json
python experiments/train_tiny_transformer.py --variant sliding --output results/transformer_sliding.json
python experiments/train_tiny_transformer.py --variant oracle --output results/transformer_oracle.json
```

The oracle condition keeps local attention and additionally exposes the three
labelled evidence tokens to the final query. If it performs close to dense
attention and well above local-only attention, the candidate-set design is
ready for a learned router.

Run the first learned-router condition:

```powershell
python experiments/train_tiny_transformer.py --variant learned --context 256 --local-window 32 --block-size 32 --top-blocks 1 --top-tokens 1 --steps 2000 --output results/transformer_learned_256.json
```

This condition is trained with explicit synthetic evidence labels, but its
answer evaluation uses only the router's hard block/token selection. It is a
retrieval-mechanism proof, not yet teacher-free long-context language-model
training.

## Causal page-tree routing (subquadratic historical lookup)

`hierarchical_page_tree.py` is the one-vector baseline. The active tiny-task
index is `multi_vector_page_tree.py`: completed pages retain up to four
structural routing slots, so rare records are not averaged into page noise.
With fixed fan-out, beam, and slots, search is
`O(beam * fanout * slots * log(pages))`; exact attention remains one unified
softmax over the fixed hot window plus selected candidates.

Run the numerical and causal-index regressions:

```powershell
python experiments/test_sparse_attention.py
python experiments/test_hierarchical_page_tree.py
```

The tiny model now has four comparable controls: `dense`, `sliding` (local),
`flat`, and `tree`.  This is a fast mechanism harness; `tree` uses full
router scores during training supervision but only scores tree-selected page
tokens at inference.

```powershell
python experiments/train_tiny_transformer.py --variant tree --task-family multirecord --context 16384 --local-window 64 --block-size 64 --top-blocks 1 --retrieval-pages 1 --tree-fanout 16 --tree-beam 1 --tree-summary structural_slots --tree-leaf-slots 1 --tree-slots 4 --retrieval-unit page_fine --retrieval-width 4 --top-tokens 1 --batch-size 1 --steps 0 --eval-batches 256 --d-model 64 --layers 2 --heads 4 --learning-rate 0.0003 --router-loss-weight 1 --historical-store bf16 --gradient-checkpointing --freeze-base-model --init-checkpoint results/transformer_learned_pagefine1_2layer_16k.pt --output results/transformer_tree_slots_pagefine1_2layer_16k.json
```

Benchmark router-inclusive controls.  `tree_initial_build_ms` is reported
separately because a real causal prefill appends each page when it leaves the
hot window; decode pays `tree_search_only` plus gather and exact attention.

```bash
.venv-runpod/bin/python experiments/benchmark_page_tree.py --context 131072 --hot-window 8192 --page-size 256 --tree-fanout 16 --tree-beam 1 --tree-leaf-slots 1 --tree-slots 4 --retrieval-pages 1 --heads 4 --head-dim 64 --output results/page_tree_128k.json
```

After the 128K fused-tree correctness check, run the 1M crossover measurement:

```bash
bash scripts/runpod_1m_tree_crossover.sh
```

`cascading_kv_attention.py` is the model-neutral Q/K/V integration core.  It
is intentionally BF16-only: quantization and latent compression are separate
follow-on experiments.  On the 96GB pod, install the optional model stack and
generate a version-locked Qwen integration manifest before binding the core
to that release's projection/cache interface:

```bash
.venv-runpod/bin/pip install -r requirements-qwen.txt
.venv-runpod/bin/python experiments/qwen35_cascading_adapter.py --output results/qwen35_adapter_manifest.json
```

## 4K decode benchmark

`benchmark_decode_attention.py` measures real CUDA decode operations over an
existing full-KV cache. It compares dense exact attention, flat retrieval, and
hierarchical block-to-token retrieval followed by exact candidate attention.

```powershell
python experiments/benchmark_decode_attention.py --context 4096 --local-window 256 --block-size 256 --top-blocks 4 --top-tokens 1 --output results/decode_benchmark_4k.json
```

It measures decode-time attention and routing work only. Since this stage
retains full historical KV by design, KV-cache capacity is the same across
variants; cache-memory reduction belongs to the later compact-latent stage.

## Linux Triton environment

The custom-kernel path uses Ubuntu WSL2 with Docker, not the Windows Python
environment. Build and verify the pinned container from WSL or PowerShell:

```powershell
docker build -f docker/Dockerfile.triton -t cascading-triton:dev .
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace cascading-triton:dev python3 experiments/triton_smoke.py
```

For repeated kernel compilation, place the repository in the Linux WSL
filesystem (for example `~/src/attention`) before mounting it into the
container; avoid `/mnt/d/...` for the compiler cache and source tree.

`triton_paged_attention.py` is the first fused kernel. It receives a contiguous
candidate buffer (local window plus selected historical pages) and fuses exact
attention's score, softmax, and value reduction. Run it from Ubuntu WSL after
the Triton bootstrap:

```bash
/root/.venv-triton/bin/python /mnt/d/Attention/experiments/triton_paged_attention.py
```

To benchmark the fused kernel inside the full 32K paged retrieval path:

```bash
/root/.venv-triton/bin/python /mnt/d/Attention/experiments/triton_paged_decode_benchmark.py
```

`triton_local_prefill.py` separately benchmarks full-sequence causal prefill
against fused exact local-window prefill. This must be interpreted separately
from decode measurements:

```bash
/root/.venv-triton/bin/python /mnt/d/Attention/experiments/triton_local_prefill.py --context 4096 --window 256
```

`triton_prefill_router_overhead.py` measures causal block-summary/top-block
updates at a fixed interval during prefill, alongside the fused local-window
kernel:

```bash
/root/.venv-triton/bin/python /mnt/d/Attention/experiments/triton_prefill_router_overhead.py --context 131072
```

`triton_promotion_cache_benchmark.py` compares router-on-every-token decode
with a promoted page-ID cache reused by the fused page-gather kernel:

```bash
/root/.venv-triton/bin/python /mnt/d/Attention/experiments/triton_promotion_cache_benchmark.py --context 131072
```

`evaluate_promotion_ttl.py` evaluates how stale promoted evidence affects a
trained router when the active query changes during a multi-step episode:

```powershell
python experiments/evaluate_promotion_ttl.py --checkpoint results/transformer_learned_multirecord_256.pt --output results/promotion_ttl.json
```

To compare fixed and event-driven refreshes at the same routing budget, use an
irregular query-change schedule:

```powershell
python experiments/evaluate_promotion_ttl.py --checkpoint results/transformer_learned_multirecord_256.pt --ttls 8 16 --change-steps 5,13,21,29,37,45,53 --output results/promotion_adaptive_refresh.json
```

The report also includes an embedding-drift gate. Its refresh threshold is
configurable without rerunning model training:

```powershell
python experiments/evaluate_promotion_ttl.py --checkpoint results/transformer_learned_multirecord_256.pt --ttls 8 16 --change-steps 5,13,21,29,37,45,53 --drift-threshold 0.05 --output results/promotion_embedding_drift.json
```

The corresponding model is trained with four addressable records per context:

```powershell
python experiments/train_tiny_transformer.py --variant learned --task-family multirecord --context 256 --local-window 32 --block-size 32 --top-blocks 1 --top-tokens 1 --steps 2000 --checkpoint results/transformer_learned_multirecord_256.pt --output results/transformer_learned_multirecord_256.json
```

`train_refresh_head.py` trains a small decoder-state-only classifier to decide
whether promoted pages should be replaced, then evaluates it against fresh
routing on the irregular schedule:

```powershell
python experiments/train_refresh_head.py --checkpoint results/transformer_learned_multirecord_256.pt --steps 800 --noise-std 0.05 --output results/learned_refresh_head.json
```

For a held-out quality/routing sweep, train once and evaluate multiple noise
levels and thresholds:

```powershell
python experiments/train_refresh_head.py --checkpoint results/transformer_learned_multirecord_256.pt --steps 800 --noise-std 0.05 --eval-noise-stds 0.05 0.20 0.50 --thresholds 0.30 0.50 0.70 --output results/learned_refresh_head_sweep.json
```

## Integrated 4K model loop

Train the 4K learned-router condition and run its multi-step decoder-query
integration benchmark with an independently trained dense control:

```powershell
python experiments/train_tiny_transformer.py --variant learned --task-family multirecord --context 4096 --local-window 64 --block-size 64 --top-blocks 1 --top-tokens 1 --batch-size 2 --steps 1600 --d-model 64 --layers 1 --heads 4 --checkpoint results/transformer_learned_multirecord_4k.pt --output results/transformer_learned_multirecord_4k.json
python experiments/benchmark_integrated_4k_decode.py --learned-checkpoint results/transformer_learned_multirecord_4k.pt --dense-checkpoint results/transformer_dense_multirecord_4k.pt --output results/integrated_decode_4k.json
```

## Triton K/V-cache model integration

From the prepared WSL Triton environment, prefill the trained model's first
layer K/V cache and decode against a promoted historical page:

```bash
/root/.venv-triton/bin/python /mnt/d/Attention/experiments/triton_kvcache_transformer_decode.py --checkpoint /mnt/d/Attention/results/transformer_learned_multirecord_4k.pt --output /mnt/d/Attention/results/triton_kvcache_transformer_decode_4k.json
```

The current model is trained on three-token record spans, but this kernel
gathers 64-token pages. Treat its latency as validated and its 0% page-level
quality as the next training/kernel integration problem.
