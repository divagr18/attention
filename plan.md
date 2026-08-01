# Cascading Attention: Initial Experiment Plan

## Objective

Validate the central architectural claim in stages:

> A hierarchical block-to-token router can recover the useful historical
> evidence of dense attention while keeping exact attention limited to a
> recent working window plus a small retrieved set.

The first milestone deliberately excludes recurrent memory, adaptive compute,
anchors, and compact historical latents. Those features should be added only
after the explicit-retrieval path is measurable and reliable.

## Experimental Platform

Begin with a small decoder-only transformer (approximately 125M–350M
parameters) once the benchmark harness is validated. Initial experiments use
8K contexts and then scale to 16K and 32K.

Default architecture settings:

| Parameter | Initial value |
|---|---:|
| Hot local window (`W`) | 1,024 tokens |
| Historical block size (`B`) | 256 tokens |
| Retrieved blocks (`m`) | 4 |
| Retrieved historical tokens (`k`) | 128 |
| Sparse-exact layer cadence | Every fourth layer |
| Historical representation | Full KV initially |
| Exact normalization | One softmax over local + retrieved tokens |

The document's 1M-token configuration is a scale target, not an appropriate
first experiment. At short contexts we can establish causal evidence about
the router, integration, and runtime before spending on systems work.

## Experiment 0 — Dense Teacher and Benchmark Harness

### Hypothesis

The proposed evaluation can unambiguously distinguish retrieval quality from
language-model quality.

### Work

Build deterministic synthetic tasks with explicit evidence labels:

- exact passkeys (UUIDs, values, names, identifiers),
- overwritten facts,
- semantically similar distractors,
- uniformly varied evidence distances,
- two-block rule/data composition,
- multi-token answers requiring retained historical evidence.

Run a dense baseline and record task accuracy, per-query historical attention
mass, block-level teacher targets, top-token teacher targets, latency, and
memory. Explicit evidence locations are the primary labels; teacher attention
is only auxiliary supervision because attention mass is not always causal
importance.

### Deliverables

- Seeded benchmark generator with JSONL cases and evidence spans.
- Selector evaluation runner and JSON summary.
- Baselines: random, recency, flat token scoring, hierarchical retrieval, and
  oracle selection.

### Success criteria

- Re-running with the same seed produces byte-identical cases and metrics.
- Every generated case validates its evidence span and correct answer.
- Metrics are reported by task family and evidence-distance bucket.

## Experiment 1 — Offline Hierarchical-Router Feasibility

### Hypothesis

A two-stage selector has high evidence recall at a much smaller candidate
budget than flat historical scoring.

### Design

Keep the model frozen. Train or evaluate routers from dense-model hidden
states. Compare:

1. Random historical blocks.
2. Recency-only selection.
3. Flat token router over all historical tokens.
4. Block-only retrieval.
5. Hierarchical block-to-token retrieval.
6. Oracle evidence selection.

Run a deliberately small ablation grid:

| Variable | Values |
|---|---|
| Block size | 128, 256, 512 |
| Routing vectors per block | 1, 4 |
| Retrieved blocks | 2, 4, 8 |
| Retrieved tokens | 64, 128, 256 |

### Metrics

- Evidence-containing block recall at `m`.
- Evidence-token recall at `k`.
- Historical teacher-attention mass captured.
- Recall by evidence-distance bucket.
- Distractor and overwrite robustness.
- Router latency and number of historical representations scored.

### Promotion gate

Proceed only if the selected configuration reaches approximately 95% block
recall and 90% token recall on single-hop retrieval, remains stable with
distance, and has a measurable cost advantage over flat scoring at 32K.

## Experiment 2 — Sparse Exact Attention Integration

### Hypothesis

Unified exact attention over `local window ∪ retrieved history` preserves most
of dense-model quality at substantially lower attention cost.

### Design

Replace selected attention layers with local-window plus retrieved-token
attention. Compare dense, sliding-window only, flat retrieval, hierarchical
retrieval, and oracle retrieval. Freeze most base weights first, training the
block summaries, routers, and any reconstruction projections, then perform a
short end-to-end adaptation.

### Metrics and gate

Measure task accuracy, loss, dense-to-oracle gap closed, router recall,
prefill/decode latency, and peak KV memory. The oracle condition must approach
dense quality; otherwise integration is at fault rather than selection.

The hierarchical variant should close 80–90% of the gap between sliding-window
and dense attention without router overhead erasing the theoretical savings.

## Experiment 3 — Prevent Local-Path Collapse

Compare ordinary next-token training, router supervision, local-window
dropout, variable hot-window size, distance-balanced evidence, and the
combined curriculum. Track historical attention mass, router entropy,
recency bias, accuracy with historical access disabled, and multi-block
composition accuracy.

## Experiment 4 — Normalization Ablation

Compare a unified softmax over local and retrieved tokens against separately
normalized local/historical branches and a learned branch gate. Use essential,
irrelevant, and contradictory historical evidence cases. Evaluate accuracy,
calibration, and unwanted historical contribution.

## Follow-on Order

1. Compact historical latents versus full historical KV.
2. Promoted cache and fixed TTLs.
3. Structural and learned anchors.
4. Recurrent memory as a routing aid.
5. Recurrent-to-exact layer ratios.
6. Adaptive retrieval budgets.
7. Tree/ANN block indexing.
8. 128K through 1M-token scaling and hardware-aware kernels.

## First Run Set

The first milestone comprises twelve substantive runs:

- one dense baseline,
- one sliding-window baseline,
- four offline router configurations,
- one flat-retrieval integration run,
- one hierarchical-retrieval integration run,
- one oracle-retrieval integration run,
- three training-curriculum variants.

The currently implemented work begins with Experiment 0 and the selector
portion of Experiment 1. It provides a reproducible benchmark before a
model-specific training stack is selected.

## Initial Actual-Transformer Result (1 August 2026)

The first model-level comparison uses a native 149,248-parameter, two-layer
causal Transformer trained for 2,000 updates on random delayed key/value
bindings. Context length is 256 and the local window is 32 tokens; evidence is
always placed outside that local window.

| Attention mode | Held-out answer accuracy |
|---|---:|
| Dense causal attention | 92.3% |
| Sliding local window | 1.5% |
| Sliding window + oracle evidence tokens | 100.0% |

The local-only result is approximately chance (1/64), showing that the task
requires historical access. The oracle result establishes that a very small
exact historical candidate set is sufficient for this task. It does **not**
yet establish that a learned hierarchical router can find that set.

### Immediate next step

Implement block scoring and token scoring over the Transformer's hidden
states, first training the router with the synthetic evidence spans and then
removing those labels at evaluation time. The decisive comparison is learned
hierarchical retrieval versus the dense, local-only, and oracle conditions
above.

## Learned-Router Result (1 August 2026)

The next condition is complete. A 157,440-parameter version of the same
Transformer uses a learned hierarchical router with seven 32-token historical
blocks. The router selects one block, then one token, and promotes the
three-token record beginning at that token into the final query's exact
attention set. Training includes explicit synthetic block and token labels;
evaluation uses the router's hard selections only.

| Attention mode | Answer accuracy | Block recall | Token recall |
|---|---:|---:|---:|
| Learned hierarchical retrieval | 100.0% | 100.0% | 100.0% |

This confirms that the proposed coarse-to-fine candidate-selection path can
be learned for exact single-hop retrieval, and that the selected evidence can
be reintegrated into a real Transformer attention operation. It is not yet a
long-context language-model result because routing supervision comes from the
synthetic evidence generator rather than a dense teacher or natural task
outcomes.

### Next experiment

Expand the task suite with overwrite and semantic-distractor cases, then
replace direct evidence labels with dense-teacher block/token distributions.
Compare direct-label, dense-teacher, and end-task-only router objectives by
router recall, answer accuracy, and score budget. After that, increase context
from 256 to 4K and benchmark actual GPU time and memory.

## Promotion-cache TTL result (1 August 2026)

The promotion-cache design has now been tested for quality in a real tiny
Transformer, not just latency. A multi-record learned router was trained on
four separately addressable historical records, then evaluated while its query
target changed every eight decode steps.

| Refresh TTL | Cached answer accuracy | Fresh-router accuracy |
|---:|---:|---:|
| 1 | 100.0% | 100.0% |
| 4 | 100.0% | 100.0% |
| 16 | 51.0% | 100.0% |
| 64 | 38.1% | 100.0% |

This validates the direction of temporary page promotion, but rejects an
unconditional long fixed TTL. The next design experiment should test adaptive
refresh signals—router-score margin, query-state change, or a learned
refresh gate—against fixed TTL at equal average routing budget.

## Equal-budget adaptive-refresh result (1 August 2026)

With irregular target changes, fixed TTL-8 and an event-driven query-signature
gate each used eight router calls per 64 decode steps. Fixed TTL-8 achieved
67.8% cached answer accuracy, while the adaptive gate retained 100.0%.
The architectural direction is therefore correct: promotion state needs an
event-driven refresh mechanism, not just a periodically scheduled reroute.
The signature is explicit only in this synthetic task; production work now
needs a decoder-observable refresh signal.

## Decoder-state refresh result (1 August 2026)

A cosine-distance gate on the current query's raw token-embedding state
replaced the explicit synthetic query signature. At the same eight router
calls per 64 steps as fixed TTL-8, it retained 100.0% cached accuracy versus
67.9% for fixed cadence. Next, train a small refresh classifier on query
states so the policy can be evaluated under noisier, more realistic changes.

## Learned refresh-head result (1 August 2026)

A small 2-layer head trained on frozen decoder/router query-state pairs
predicted whether the promoted record should be replaced. Under state noise
(standard deviation 0.05) and irregular changes it made eight refreshes per
64 steps, retained 100.0% cached accuracy, and achieved 100.0% refresh
precision and recall. The next experiment is a held-out noise/ambiguity and
threshold sweep to establish the quality-versus-routing-budget frontier.

## Refresh-head robustness sweep (1 August 2026)

Across held-out state noise of 0.05 and 0.20, every tested threshold retained
100.0% cached accuracy with eight refreshes per 64 steps (12.5% routing). At
noise 0.50, quality remained 100.0% but the gate over-refreshed: threshold
0.70 used 18.3 refreshes (28.5% routing), while lower thresholds approached
full rerouting. This is a safe failure mode, but shows calibration is now the
main issue. Next integrate the policy with actual autoregressive generation
and report end-to-end cost alongside answer quality.

## 4K integrated decoder result (1 August 2026)

The first trained 4K routed Transformer reached 96.9% held-out retrieval
accuracy with 100.0% router recall. In a 16-step integrated decoder-query
loop with irregular changes, rerouting achieved 97.7% accuracy at 26.54 ms per
query and 16 router calls; adaptive promotion kept 97.7% accuracy at 25.16 ms
and four router calls. This verifies integration, but it is not yet a fair
dense-quality comparison (the tiny dense baseline did not converge) and it is
not a KV-cached appended-token generation loop. The next work is to implement
that true decode loop and use the fused paged kernel rather than PyTorch's
full masked attention implementation.

## Successful full-page KV-cache integration (1 August 2026)

Span-to-page curriculum raised page quality from 1.6% to 71.9%. The corrected
Triton K/V-cache path matched the page model at 68.8% in the integrated loop
and reduced decode time from 22.56 ms to 1.04 ms (21.7x), with four rather
than 16 router calls. Next improve page-model quality before scaling further.

## 4K KV-cache Triton integration result (1 August 2026)

The fused Triton page-gather kernel now consumes the trained 4K Transformer's
prefetched first-layer K/V cache. It achieved 1.45 ms decode versus 22.77 ms
for the full PyTorch routed query (15.7x), with K/V prefill at 0.23 ms.
However, quality was 0.0% because the model was trained on promoted
three-token records while the kernel promotes whole 64-token pages. Candidate
granularity is therefore a learned-model contract. Next either train full-page
promotion directly or implement a span-gather kernel, then repeat this
integration measurement.
