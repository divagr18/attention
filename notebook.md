# Cascading Attention Experiment Notebook

**Project:** Cascading Attention for Long-Context Language Models  
**Started:** 1 August 2026

This log records every completed experiment, its configuration, result, and
what can (and cannot) be inferred from it.

## E0 — Deterministic synthetic selector harness

### Purpose

Establish a reproducible benchmark for historical exact retrieval before
introducing a neural model.

### Implementation

`experiments/run_selector_benchmark.py` generates 8,192-token synthetic
contexts with labelled evidence spans. It reports evidence block/token recall
and the number of entries scored by random, recency, flat, hierarchical, and
oracle selector controls.

### Validation

- A repeat of the same B=256 run with seed 7 produced an identical SHA-256
  output file.
- Random and recency controls fail on distant evidence.
- Oracle selection has perfect labelled-evidence recall.

### Offline selector configurations (1,000 cases each)

| Block size | Top blocks | Top tokens | Hierarchical token recall | Block recall | Scores/query |
|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 128 | 0.512 | 1.000 | 576 |
| 256 | 4 | 128 | 0.427 | 1.000 | 1,056 |
| 256 | 8 | 256 | 0.429 | 1.000 | 2,080 |
| 512 | 4 | 128 | 0.382 | 1.000 | 2,064 |

For comparison, the flat lexical control scored all 8,192 historical tokens.

### Interpretation

This confirms the harness and the bookkeeping advantage of coarse-to-fine
selection. It does **not** evaluate a learned neural router: scores are a
lexical proxy, so this is not an architectural-quality result.

## E1 — Actual dense Transformer viability

### Setup

A native, 149,248-parameter two-layer GPT-style causal Transformer was
trained for 2,000 updates on random delayed key/value bindings.

- Context: 256 tokens
- Local window: 32 tokens
- 64 possible answer values (chance accuracy: 1.56%)
- Evidence always outside the local window
- Evaluation: 2,048 held-out examples

### Result

| Attention mode | Answer accuracy |
|---|---:|
| Dense causal | 92.3% |
| Sliding local window | 1.5% |
| Sliding + oracle evidence | 100.0% |

Dense accuracy remained 92.3% in the far-distance bucket. The local-only
condition stayed at chance, confirming that the task requires distant access.

### Interpretation

The oracle result proves that a local attention window plus a tiny exact
historical candidate set is sufficient for this task. It does not show whether
the candidate set can be found without labels.

## E2 — Learned hierarchical router on single-hop retrieval

### Setup

The same model gained an 8,192-parameter router. It partitions the 224-token
historical region into seven 32-token blocks, selects one block, selects one
token within it, and promotes that token plus two following record tokens into
the final query's exact attention set.

Training uses direct synthetic block and token labels. Evaluation uses the
router's hard selection only.

### Result

| Metric | Result |
|---|---:|
| Answer accuracy | 100.0% |
| Router block recall | 100.0% |
| Router token recall | 100.0% |

### Interpretation

The complete coarse-to-fine retrieval and exact-reintegration path works in a
real Transformer for unambiguous single-hop matching. The router currently
uses input-token embeddings and direct labels, so this is a mechanism proof,
not semantic long-context retrieval.

## E3 — Overwrite and distractor retrieval

### Hypothesis

The learned router can use query-specified structural markers—not merely a
matching key—to select the current/canonical record among competing historical
records.

### Planned setup

Extend the data generator with:

- **Overwrite:** an old and a latest value for the same key; the query requests
  the latest record.
- **Distractor:** several same-key decoy records; the query requests the
  canonical record.
- A mixed evaluation set containing ordinary, overwrite, and distractor
  examples.

The router will receive the same direct synthetic labels as E2 so this
experiment isolates representational and routing difficulty. A subsequent
experiment will replace these labels with dense-teacher distributions.

### Result

The same 157,440-parameter model was trained for 2,000 updates on a balanced
mixture of ordinary, overwrite, and distractor cases. Context, local window,
router budget, and evaluation size matched E2.

| Metric | Result |
|---|---:|
| Answer accuracy | 99.95% |
| Near-distance accuracy | 100.0% |
| Medium-distance accuracy | 99.87% |
| Far-distance accuracy | 100.0% |
| Router block recall | 100.0% |
| Router token recall | 100.0% |

### Interpretation

With direct labels, the router can distinguish structured records carrying the
same key by using the query's marker and the record's following marker. This
is stronger than E2 because key matching alone cannot solve the overwrite or
distractor cases. It remains supervised synthetic retrieval.

## E4 — Dense-teacher router supervision

### Question

Can the router learn useful block/token priorities from a dense Transformer's
attention distributions instead of direct evidence-position labels?

### Setup

A dense teacher was trained for 2,500 updates on the E3 mixed suite, reaching
91.31% held-out answer accuracy. A fresh learned-retrieval student was then
trained for 2,500 updates with:

- ordinary answer cross-entropy, and
- KL losses matching the teacher's final-layer, final-query attention mass,
  aggregated over heads, at both token and block resolution.

The router received no evidence-position, block, or marker labels. At
evaluation it used only hard learned retrieval.

### Result

| Model / metric | Result |
|---|---:|
| Dense teacher answer accuracy | 91.31% |
| Teacher-supervised student answer accuracy | 80.71% |
| Student near accuracy | 83.37% |
| Student medium accuracy | 79.25% |
| Student far accuracy | 80.61% |
| Student block recall against labelled evidence | 76.66% |
| Student record-start token recall | 0.15% |

### Interpretation

This is a successful removal of direct router labels, but not yet a quality
match to the teacher. The very low record-start recall alongside moderate
answer accuracy indicates that raw final-layer teacher attention is often
placed on the value or marker token rather than the labelled key-token start.
Our current three-token promotion policy assumes a record-start selection, so
attention imitation and the retrieval representation are misaligned.

### Next experiment

Distil **record-span** importance rather than raw token attention: aggregate
teacher attention over each three-token record, supervise a score for its
start position, and compare against the current raw-attention target. This
keeps teacher-only routing supervision while aligning the target with the
promoted exact-memory unit.

## E5 — Record-span teacher distillation

### Question

Does teacher supervision become effective when its target is expressed in the
same three-token record unit that sparse attention will promote?

### Setup

The teacher and student configurations match E4. Instead of assigning the
teacher's attention mass to individual historical tokens, the target score for
each candidate record start is the summed attention mass of that start and its
next two tokens. The student uses the same hard one-block/one-start retrieval
budget at evaluation, with no direct evidence labels.

### Result

| Metric | Raw-token teacher target (E4) | Record-span target (E5) |
|---|---:|---:|
| Answer accuracy | 80.71% | 99.66% |
| Block recall | 76.66% | 97.22% |
| Labelled record-start recall | 0.15% | 11.23% |
| Far-distance answer accuracy | 80.61% | 99.63% |

### Interpretation

This is the first teacher-only router-supervision result that matches the
direct-label mechanism test in answer quality. The lower exact-start metric is
not contradictory: neighbouring start positions can still promote an exact
span containing the key, marker, and value. Block recall and answer accuracy
are the more meaningful measurements for this promotion policy.

The work is aligned with the original plan's central claim: hierarchical
selection can approximate dense historical access while exact attention is
restricted to a local window plus a small retrieved set. The next plan-aligned
gate is scaling this validated mechanism to a substantially longer context and
measuring actual GPU time and memory, before adding recurrent memory or
compression.

## E6 — 4K decode-time hardware benchmark

### Purpose

Measure actual CUDA decode operations at the first scaled context, rather than
infer speed from sparse FLOP counts. This test holds full historical KV fixed,
as specified for the pre-compression stage.

### Setup

- Context: 4,096 tokens
- Batch: 8; heads: 4; head dimension: 16; FP16 KV
- Local exact window: 256 tokens
- Hierarchical router: 256-token blocks, top 4 blocks, top 1 promoted record
- 300 timed GPU iterations after warm-up

### Result

| Decode path | Mean latency |
|---|---:|
| Dense exact attention | 246 µs |
| Flat token retrieval + candidate attention | 567 µs |
| Hierarchical retrieval + candidate attention | 872 µs |

The hierarchy scores 1,039 router entries/query (`15` block summaries plus
`4 × 256` token entries), compared with 3,840 for flat routing. It attends
exactly to 259 candidates (256 local + a three-token record), rather than
4,096. The score-workspace estimate falls from 0.250 MiB to 0.016 MiB. Full
KV-cache capacity remains 8 MiB in all conditions because historical KV has
not been compressed.

### Interpretation

At 4K, the un-fused gather implementation is slower despite doing less routing
and exact-attention work. This is not a failure of the retrieval-quality
result; it confirms the proposal's hardware warning that irregular gathers can
erase theoretical sparsity gains. The next systems experiment should use
block-contiguous/paged storage and fused candidate gathering, then repeat this
benchmark at 4K and larger contexts. No recurrent-memory or compression
component should be introduced before that comparison.

## E7 — Block-paged retrieval and 32K scaling

### Purpose

Test whether retrieving contiguous historical pages narrows the hardware gap
created by fine-grained dynamic KV gathering, then repeat the decode benchmark
at a context length where hierarchical routing has a much larger asymptotic
advantage.

### Result

| Context / decode path | Mean latency |
|---|---:|
| 4K dense | 248 µs |
| 4K fine-token hierarchy | 955 µs |
| 4K paged-block hierarchy | 822 µs |
| 32K dense | 620 µs |
| 32K flat retrieval | 696 µs |
| 32K fine-token hierarchy | 849 µs |
| 32K paged-block hierarchy | 670 µs |

At 32K, the paged path scores 1,151 routing entries/query versus 32,512 for
flat routing and attends to 1,280 exact candidates (256 local + four 256-token
pages). It is only 8% slower than dense decode despite using unfused PyTorch
operations. The fine-token gather path remains substantially slower.

### Interpretation

Custom kernels should help materially. Dense decode currently benefits from
one regular, optimized operation, whereas the paged sparse path launches
separate block routing, top-k, page gather/reshape, concatenation, and exact
attention operations. A fused paged kernel can eliminate intermediate buffers
and launch overhead while retaining contiguous page reads. The 32K near-parity
result is the correct threshold for investing in that implementation.

The immediate systems task is a CUDA/Triton-style fused kernel for fixed
retrieval buckets. Triton is not installed in this workspace, so that requires
adding a kernel toolchain or implementing a CUDA extension before a true
custom-kernel benchmark can be run.

## E8 — Linux Triton kernel environment and fused paged attention

### Environment decision

Native Windows Triton was not used as the primary environment. Ubuntu 24.04
WSL2 was installed and verified to expose the RTX 3050. Docker Desktop builds
were abandoned for this machine because `C:` had only about 4 GB free and the
CUDA/PyTorch image build repeatedly dropped the Docker daemon connection.

Installing Linux packages directly under `/mnt/d` also failed with a bus error
during PyTorch unpacking. A second Ubuntu WSL instance was therefore imported
with its virtual disk stored on `D:` (`Ubuntu-24.04-D`), giving the kernel
toolchain a native ext4 filesystem with sufficient capacity. It uses PyTorch
2.11.0+cu128 and Triton 3.6.0.

### Kernel

`experiments/triton_paged_attention.py` implements the first fused GPU kernel.
The input is an already contiguous candidate buffer containing the local
window plus selected historical pages. One Triton program per batch/head fuses
the QK score calculation, numerically stable streaming softmax, and V
reduction without materializing attention scores or probabilities.

### Result

| Candidate buffer | Triton fused kernel | PyTorch reference | Speedup |
|---|---:|---:|---:|
| 1,280 tokens; batch 8, 4 heads, head dim 16 | 91.23 µs | 267.38 µs | 2.93× |

The output passed numerical comparison against the PyTorch reference at FP16
tolerances.

### Interpretation

This validates the systems hypothesis from E7: the dense-versus-paged gap was
largely kernel/launch/intermediate-buffer overhead, not an inherent cost of
candidate attention. The next kernel experiment is to benchmark this fused
candidate-attention kernel inside the full 32K paged retrieval path, then
consider fusing page gather and candidate assembly at fixed retrieval buckets.

## E9 — Fused Triton attention in the full 32K paged path

### Purpose

Replace only the exact-attention stage of the complete paged retrieval path
with the E8 Triton kernel, while retaining GPU block scoring, top-4 page
selection, and page assembly in PyTorch. This measures the actual end-to-end
decode trade-off rather than the isolated candidate-attention primitive.

### Setup

- Context: 32,768 tokens
- Batch: 8; heads: 4; head dimension: 16; FP16
- Local window: 256 tokens
- Historical pages: four 256-token blocks
- Exact candidate buffer: 1,280 tokens
- Routing work: 1,151 entries/query (127 block summaries + 1,024 page tokens)
- 300 timed GPU iterations

### Result

| Full decode path | Mean latency |
|---|---:|
| Dense PyTorch exact attention | 596.77 Âµs |
| Paged retrieval, PyTorch candidate attention | 590.38 Âµs |
| Paged retrieval, fused Triton candidate attention | **494.88 Âµs** |

The fused path is **1.21Ã— faster** than dense decode while retaining the full
routing and page-assembly cost. Its numerical candidate-attention output was
validated against the FP16 PyTorch reference before timing.

### Interpretation

This is the first measured end-to-end decode speedup for the architecture,
not merely a sparse-FLOP estimate. The remaining bottleneck is page selection
and candidate-buffer assembly, which still uses several PyTorch operations.
The next kernel step is to fuse fixed-bucket page gather/assembly with the
candidate-attention launch, then retest at 32K and a larger context length.

## E10 — Fused page-gather and exact-attention kernel

### Purpose

Eliminate the remaining PyTorch candidate-buffer assembly by passing selected
historical page IDs directly to the attention kernel. The kernel derives each
candidate's KV location from its local-window/page position and streams QK,
softmax, and V directly from the full historical KV cache.

### Result

| Full 32K decode path | Mean latency |
|---|---:|
| Dense PyTorch exact attention | 594.89 Âµs |
| Fused Triton page gather + exact attention | **229.76 Âµs** |

This is a **2.59Ã— speedup** at the same batch, head, window, block, and
retrieval settings as E9. The fused kernel passed numerical comparison against
the FP16 PyTorch attention result for the selected pages.

### Interpretation

The E9 bottleneck was indeed candidate construction. With selected-page IDs
read directly inside the kernel, the architecture has a material end-to-end
decode advantage at 32K. The next measurement should scale this exact kernel
to 128K context and vary page/retrieval budgets; after that, profile prefill
separately before introducing cache compression or recurrent memory.

## E11 â€” 128K decode scaling and retrieval-budget sweep

### Setup

The E10 fused page-gather kernel was run at 131,072-token context with the
same FP16 batch/head dimensions. The local exact window remained 256 tokens;
each selected page contains 256 tokens. Every condition includes block routing
and fused direct reads from the full KV cache.

### Result

| Retrieved pages | Dense decode | Fused paged decode | Speedup |
|---:|---:|---:|---:|
| 2 | 2,057.33 Âµs | 224.00 Âµs | 9.18Ã— |
| 4 | 2,024.22 Âµs | 229.33 Âµs | 8.83Ã— |
| 8 | 2,074.13 Âµs | 243.09 Âµs | 8.53Ã— |

### Interpretation

The decode advantage grows substantially with context length because dense
attention must read the entire 128K KV history while the fused path reads the
local window plus a fixed number of selected pages. Increasing the retrieved
budget from two to eight pages adds only 19 Âµs in this configuration. This is
the expected adaptive-compute behavior and provides a practical budget range
for later quality experiments.

The next distinct experiment is **prefill**. The current fused kernel targets
one decode query, so prefill needs a separate local-window/kernel strategy and
must be benchmarked independently; decode gains must not be extrapolated to
prefill.

## E12 â€” Fused local-causal prefill

### Purpose

Measure full-sequence prefill separately from decode. A Triton kernel computes
one exact causal local-window attention result per batch/head/query position,
using a streaming softmax over the previous 256 tokens. It was checked against
an explicit causal-window masked reference on a smaller sequence before timing.

### Result

| Context | Dense causal flash attention | Fused 256-token local prefill | Speedup |
|---:|---:|---:|---:|
| 4K | 0.65 ms | 1.17 ms | 0.55Ã— |
| 16K | 5.41 ms | 3.53 ms | 1.53Ã— |
| 32K | 20.97 ms | 6.68 ms | 3.14Ã— |

### Interpretation

At short context, PyTorch's dense flash-attention implementation is more
efficient than the first local Triton kernel. The expected crossover appears
between 4K and 16K; at 32K, fixed-window prefill is over three times faster.

This validates the hot-window portion of the cascading architecture for
prefill. It does **not** yet include historical retrieval at every prefill
position. The next systems task is to define the periodic block-summary/router
update schedule during prefill and measure its incremental overhead, then join
it with the validated decode path.

## E13 â€” 128K and 256K local-prefill scaling

### Result

| Context | Dense causal prefill | Fused 256-token local prefill | Speedup |
|---:|---:|---:|---:|
| 128K | 373.35 ms | 23.46 ms | 15.91Ã— |
| 256K | 1,963.03 ms | 52.06 ms | 37.71Ã— |

### Interpretation

The local path scales approximately with context length, whereas dense causal
prefill continues to exhibit the expected quadratic growth. The 256K result
is the strongest prefill scaling evidence in this prototype, although it uses
a batch-1, four-head, head-dimension-16 microbenchmark and should not be read
as an end-to-end language-model throughput figure.

## E14 â€” Periodic causal block routing during prefill

### Setup

At 128K context, block summaries were constructed over 256-token blocks and a
causal top-4 block router was invoked every 256 tokens (512 routing updates).
The routing operation masks future blocks, so selected blocks are causal even
though the benchmark vectorizes summary construction for measurement.

### Result

| Component | Mean time |
|---|---:|
| Fused local prefill | 23.77 ms |
| Periodic block-summary/router work | 0.78 ms |
| Combined prefill + routing | 24.43 ms |
| Incremental combined overhead | 2.79% |

### Interpretation

Periodic hierarchical routing can be added to long-context local prefill at a
small measured cost in this configuration. This completes the first prefill
cost decomposition: local exact computation dominates, while block routing is
cheap enough to run at a 256-token cadence. The next integration task is to
use these routed blocks to prepare decode-time page eligibility/promotion state
without re-running full routing for every generated token.

## E15 â€” Promoted page-ID cache during decode

### Setup

At 128K context, the top four historical page IDs selected by the router are
treated as a temporary promoted cache. The fused gather-attention kernel reads
those IDs directly. The comparison measures one decode query with either a new
block-router pass or reuse of the promoted IDs.

### Result

| Decode component | Mean latency |
|---|---:|
| Router only | 144.42 Âµs |
| Reroute every query + fused attention | 234.54 Âµs |
| Promoted page IDs + fused attention | 93.29 Âµs |

Reusing promoted page IDs is **2.51Ã— faster** than rerouting every query. The
router accounts for 61.57% of the rerouted decode path in this configuration.

### Interpretation

This validates the temporary-promotion mechanism's systems rationale: block
selection should be refreshed at an interval or task boundary, not necessarily
at every decode token. This benchmark assumes promoted pages remain relevant;
the next model-quality experiment must measure retrieval stability and answer
accuracy as promotion TTL changes.

## Artifact index

- Architecture proposal: `cascading_attention_architecture.md`
- Research plan: `plan.md`
- Selector harness: `experiments/run_selector_benchmark.py`
- Transformer experiment: `experiments/train_tiny_transformer.py`
- Experiment reports: `results/*.json`

## E16 — Promotion TTL quality under changing retrieval targets

### Setup

The earlier learned router was trained on a single historical record, so it
could not validly evaluate whether a cache stays correct while selecting among
several records. A new 157,440-parameter learned-retrieval Transformer was
therefore trained for 2,000 updates on four independently addressable records
per 256-token context. Each query selects one record; the router promotes one
32-token block and one three-token record span.

For the TTL test, the same context is queried for 64 decode steps, with the
target key changing every eight steps. Fresh routing is the control. Cached
promotions are refreshed every 1, 4, 16, or 64 steps; eight episodes of 32
examples were evaluated per setting.

### Result

| Promotion refresh TTL | Fresh-router accuracy | Cached accuracy | Cached − fresh | Cache equals fresh router |
|---:|---:|---:|---:|---:|
| 1 step | 100.0% | 100.0% | 0.0 pp | 100.0% |
| 4 steps | 100.0% | 100.0% | 0.0 pp | 100.0% |
| 16 steps | 100.0% | 51.0% | -49.0 pp | 50.0% |
| 64 steps | 100.0% | 38.1% | -61.9 pp | 37.5% |

### Interpretation

Promotion reuse is safe while its refresh cadence is at least as frequent as
the retrieval-target changes in this controlled task. Once the cache spans a
target change, the stale page is usually wrong and answer quality collapses.
The systems win in E15 is therefore conditional: caching must be coupled to a
refresh policy based on query/task change, uncertainty, or a bounded TTL.
This is a 256-token synthetic quality test, not evidence that a 128K language
model can use a fixed TTL unchanged.

## E17 — Adaptive promotion refresh at equal routing budget

### Setup

The TTL schedule was made deliberately irregular: the retrieval target changes
at steps 5, 13, 21, 29, 37, 45, and 53 of a 64-step episode. This creates eight
total refresh opportunities including step zero. Fixed TTL-8 and an adaptive
query-signature gate therefore use the same average budget: eight router runs
per episode (12.5% of decode steps). The signature gate refreshes only when
the three-token query identity changes.

### Result

| Strategy | Router refreshes / 64 steps | Cached accuracy | Cache equals fresh router |
|---|---:|---:|---:|
| Fixed TTL 8 | 8 | 67.8% | 67.2% |
| Fixed TTL 16 | 4 | 32.2% | 31.2% |
| Adaptive query-signature gate | 8 | 100.0% | 100.0% |

### Interpretation

At equal router frequency, refresh timing matters: an event-driven gate avoids
the 32.2-point accuracy loss of fixed TTL-8. This is proof of principle only;
the gate reads an explicit synthetic query identity. The next experiment must
replace that oracle-like feature with signals available in a real decoder,
such as a cheap query-state distance threshold, router-score margin, or a
learned refresh head.

## E18 — Decoder-observable embedding-drift refresh gate

### Setup

The explicit query-signature gate in E17 was replaced by a cheap state signal:
the mean raw token embedding of the final three query tokens. The cache is
refreshed when cosine distance from the state at the last refresh exceeds
0.05. This calculation reads only the current query representation and does
not score any historical block, so it is available before paying router cost.

### Result

Using the same irregular schedule and eight refreshes per 64 steps as E17:

| Strategy | Refreshes / 64 steps | Cached accuracy | Cache equals fresh router |
|---|---:|---:|---:|
| Fixed TTL 8 | 8 | 67.9% | 67.2% |
| Embedding-drift gate (cosine distance > 0.05) | 8 | 100.0% | 100.0% |

### Interpretation

The adaptive result survives removal of the explicit token-equality check.
The signal is still unusually clean because this synthetic query changes one
key token at a time. It establishes feasibility, not a production threshold.
The next quality experiment should train a small refresh head on decoder query
states and evaluate a precision/recall versus routing-budget curve under
noisy, semantically similar query changes.

## E19 — Learned decoder-state refresh head

### Setup

A 2-layer, 32-hidden-unit binary head was trained for 800 updates on frozen
router-query states. Its input is the current state, the state saved at the
last refresh, their absolute difference, and their elementwise product. Its
label is whether the frozen hierarchical router's promoted record has changed.
The gate receives Gaussian state-observation noise (standard deviation 0.05)
in both training and the irregular-change evaluation; it never scores
historical blocks itself.

### Result

| Metric | Learned refresh head |
|---|---:|
| Refreshes / 64 steps | 8 |
| Routing fraction | 12.5% |
| Cached accuracy | 100.0% |
| Cache equals fresh router | 100.0% |
| Refresh precision | 100.0% |
| Refresh recall | 100.0% |

### Interpretation

The refresh decision can be learned from decoder-observable state pairs,
without calculating a new historical routing score. This is the strongest
promotion-cache quality result so far, but it is still deliberately simple:
the key-level retrieval target changes cleanly and labels are derived from the
frozen router. The next credible step is a held-out sweep over higher noise,
nearby/ambiguous query states, and thresholds to produce a refresh-budget
versus quality curve.

## E20 — Learned refresh-head noise and threshold frontier

### Setup

The E19 head was evaluated on held-out contexts across three state-observation
noise levels and thresholds 0.30, 0.50, and 0.70. Each condition uses the same
irregular target changes and fresh routing only as a measurement control.

### Result

| State noise | Threshold range | Cached accuracy | Refreshes / 64 steps |
|---:|---:|---:|---:|
| 0.05 | 0.30–0.70 | 100.0% | 8.0 (12.5%) |
| 0.20 | 0.30–0.70 | 100.0% | 8.0 (12.5%) |
| 0.50 | 0.30 | 100.0% | 64.0 (100.0%) |
| 0.50 | 0.50 | 100.0% | 57.6 (90.0%) |
| 0.50 | 0.70 | 100.0% | 18.3 (28.5%) |

At 0.50 noise the gate maintained 100% refresh recall but introduced false
positives: its precision was 12.5%, 13.9%, and 43.8% at thresholds 0.30,
0.50, and 0.70 respectively.

### Interpretation

The learned policy is robust at the moderate noise range tested. Under severe
noise it fails safely by rerouting too often rather than losing answer
quality; threshold 0.70 recovers much of the routing saving. The limitation is
clear: this still tests discrete synthetic retrieval targets, not naturally
evolving language-model hidden states. The next meaningful architecture step
is to integrate the gate into a genuine autoregressive Transformer decoding
loop and measure end-to-end latency and answer retention together.

## E21 — Integrated 4K decoder-query loop

### Setup

A one-layer, 353,216-parameter learned-retrieval Transformer was trained at
4,096 tokens on the four-record task (1,600 updates, direct router labels).
It reached 96.9% held-out accuracy and 100.0% router token/block recall. The
integration benchmark then issued 16 sequential retrieval queries against a
fixed 4K context, changing the target at steps 3, 7, and 12. It measures full
model CUDA wall time, answer accuracy, and router calls. Adaptive promotion
uses the decoder-state cosine-drift gate.

### Result

| Mode | Answer accuracy | Mean query time | Router calls / 16 steps |
|---|---:|---:|---:|
| Dense (separately trained) | 2.3% | 23.66 ms | 0 |
| Sliding local only | 0.0% | 25.20 ms | 0 |
| Routed, reroute each query | 97.7% | 26.54 ms | 16 |
| Routed, adaptive promotion | 97.7% | 25.16 ms | 4 |

Adaptive promotion reduced full-model query time by 5.2% versus rerouting on
every query while eliminating 75% of router invocations.

### Interpretation

This is the first end-to-end 4K model integration result: the routed and
cached mechanisms work together inside actual Transformer inference, rather
than only in a selector harness or Triton attention microbenchmark. The dense
baseline is **not** a valid quality comparison yet—it did not converge in the
same small-model/batch training budget (2.3% accuracy)—while local-only is the
expected no-history control. The PyTorch model still constructs a full masked
attention matrix, so the 5.2% end-to-end gain understates what the fused paged
kernel can provide. This loop replaces query tokens in a fixed context; it is
not yet a KV-cached natural-language generation loop with appended tokens.

## E23 — Page-promotion curriculum and corrected Triton integration

Direct page training achieved 1.6% accuracy. Span-to-page curriculum training
reached 71.9% held-out accuracy. In the 16-step integration loop, the full
PyTorch page model and the corrected prefetched-KV Triton page path both
achieved 68.8% accuracy. Triton decode was 1.04 ms versus 22.56 ms for full
PyTorch reroute (21.7x); adaptive promotion used four router calls versus 16.

This is the first quality-preserving trained-model K/V-cache/Triton page
integration. The remaining bottleneck is page-model quality, not the systems
contract.

## E22 — 4K prefetched-KV Triton page decode

### Setup

The one-layer 4K trained Transformer was run in WSL/Triton. Its first-layer
K/V cache is prefetched once from the fixed 4K context. Each decode query
updates only the final three query-token K/V entries, then fused Triton
attention reads the 64-token local tail plus one selected 64-token page.
Kernel compilation is excluded from timings.

### Result

| Metric | Result |
|---|---:|
| K/V-cache prefill | 0.23 ms |
| Full PyTorch routed query | 22.77 ms |
| Triton prefetched full-page decode | 1.45 ms |
| Decode speedup | 15.7x |
| Adaptive router calls / 16 steps | 4 vs. 16 rerouted |
| Exact three-token routed model accuracy | 100.0% |
| Triton full-page model accuracy | 0.0% |

### Interpretation

The systems half works: trained-model K/V tensors can be prefetched and read
through the fused page kernel at a large latency reduction. But this exposed a
critical model/kernel contract mismatch. The Transformer was trained to
promote a three-token record span, while the kernel exposes an entire
64-token page; the extra candidates change attention normalization enough to
destroy answer quality. The next change must train full-page promotion
directly, or implement a fused selected-record-span gather kernel.

## E24 — 16K matched-compute full-page control

### Purpose

Test whether progressive span-to-page widening fixes the quality failure of
direct full-page promotion at 16K. Both conditions begin from the same weak
16K three-token-span checkpoint and receive 4,800 further updates.

### Result

| Condition | Answer accuracy | Near | Medium | Far | Triton decode | PyTorch reroute |
|---|---:|---:|---:|---:|---:|---:|
| Direct 64-token page, 4,800 updates | 76.56% | 98.31% | 3.28% | 100.0% | 0.290 ms | 57.11 ms |
| Span 8 → 16 → 32 → 64-token page, 1,200 updates/stage | 70.70% | 100.0% | 1.32% | 100.0% | 0.311 ms | 56.83 ms |

In both decode-loop integrations, the Triton result exactly matched the
corresponding full PyTorch routed model at 68.75% accuracy. Router block and
token recall were 100% in every training evaluation.

### Interpretation

The full-page failure is not a routing failure and is not repaired by the
matched-compute widening curriculum at 16K. The strong near/far results and
near-zero medium accuracy expose a positional/candidate-normalization
failure: the correct page is found, but adding irrelevant page tokens changes
how the model uses that page. The fused page kernel remains correct and fast;
candidate granularity is the quality bottleneck.

## E25 — 16K page-fine selector budget frontier

### Purpose

Retain page-level routing while promoting only short exact spans from inside
the selected 64-token page. Each condition starts from the perfect 32-token
span curriculum checkpoint, uses four-token promoted spans, and varies the
number of within-page centers.

### Result

| Fine centers/page | Answer accuracy | Near | Medium | Far | Triton decode | Adaptive Triton decode |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.0% | 100.0% | 100.0% | 100.0% | 0.278 ms | 0.273 ms |
| 2 | 100.0% | 100.0% | 100.0% | 100.0% | 0.300 ms | 0.288 ms |
| 4 | 100.0% | 100.0% | 100.0% | 100.0% | 0.288 ms | 0.282 ms |
| 8 | 100.0% | 100.0% | 100.0% | 100.0% | 0.285 ms | 0.279 ms |
| 16 | 100.0% | 100.0% | 100.0% | 100.0% | 0.333 ms | 0.292 ms |

The Triton path matched the full routed model at 100% for every budget.

### Interpretation

One four-token span from the routed page is sufficient on this task. It is
also the smallest and fastest measured configuration, so later experiments
use `page_fine`, one within-page center, and retrieval width four. This
isolates the full-page collapse to unnecessary page candidates rather than
the coarse page-routing representation.

## E26 — Two-layer 16K page-fine K/V-cache integration

### Purpose

Verify that the selected page-fine configuration remains accurate in a deeper
Transformer, then extend the fused K/V-cache decode path from one layer to
two layers.

### Setup

A two-layer 16K model was initialized from the one-layer page-fine,
one-center checkpoint and trained for 1,200 updates with activation
checkpointing. The decoder prefilled layer-one and layer-two K/V caches.
During decode, it recomputed the three-token query tail, used local attention
for the first two tail tokens, and used fused fine-span attention for the
final query in both layers.

### Result

| Metric | Result |
|---|---:|
| Held-out answer accuracy | 100.0% |
| Router block/token recall | 100.0% |
| Two-layer K/V prefill | 56.44 ms |
| Full PyTorch reroute | 113.05 ms |
| Fused Triton page-fine decode | 0.585 ms |
| Adaptive fused decode | 0.554 ms |
| Exact model / Triton decode accuracy | 100.0% / 100.0% |

### Interpretation

The page-fine solution survives depth expansion, and the two-layer fused
decoder preserves model quality while reducing routed decode latency by about
193× relative to full PyTorch rerouting. The 24 GB RTX 4090 was sufficient
for this 16K inference benchmark after disabling autograd in the benchmark's
full-PyTorch control. Scaling two-layer *training* to 32K will require more
memory because the current training implementation still materializes full
attention matrices.
