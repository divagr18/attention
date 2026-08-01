# Cascading Attention for Long-Context Language Models

## A Refined Architecture for Dense Recent Context, Sparse Historical Retrieval, and Recurrent Memory

**Date:** 1 August 2026  
**Status:** Research concept / architecture proposal  
**Scope:** Long-context transformer architecture, efficient attention, hierarchical retrieval, recurrent memory, and hardware-aware implementation

---

## 1. Executive Summary

The central proposal is to treat an LLM context window as a hierarchy of memory resolutions rather than as a uniform sequence in which every token receives the same attention budget.

The initial intuition was:

> Use fully quadratic attention for the latest portion of the context and a subquadratic mechanism for the older portion.

This intuition is strong because recent context usually contains the model's active working state: the current reasoning chain, the latest user request, recently introduced variables, nearby code, unresolved references, and current dialogue state. Older context behaves more like episodic or long-term memory: it is often needed selectively rather than continuously.

However, a fixed percentage of the context should not remain dense. If the dense region is always a constant fraction of the total context, the architecture remains asymptotically quadratic. The stronger design is therefore:

> Maintain a fixed or slowly growing exact working-memory window, progressively demote older information into cheaper representations, and selectively promote relevant historical information back into exact token-level attention.

The refined system combines three complementary capabilities:

1. **Dense recent attention** for exact local computation.
2. **Hierarchical sparse retrieval** for recovering specific old tokens or blocks.
3. **Recurrent or delta-rule memory** for preserving diffuse, continuously accumulated global state.

The final architecture should not necessarily be rebuilt around several parallel attention streams. Multiple streams are a useful design and ablation axis, but the cleaner default is a unified exact attention operation whose candidate set is assembled from several memory mechanisms.

The architecture can be summarized as:

> **Age-based representation demotion, hierarchical content-based promotion, and periodic exact reintegration.**

---

## 2. Original Intuition

The original proposal was to split attention by recency:

- The latest \(x\%\) of the context receives full quadratic attention.
- The remaining context is processed through a subquadratic mechanism.

The intuition behind this split is compelling:

- Recent tokens are disproportionately important to next-token prediction.
- Active reasoning tends to be local in time.
- Exact copying, syntax, variable binding, and short-range reference resolution require high-resolution token interactions.
- Older information is often accessed episodically rather than continuously.
- Uniform full attention wastes compute on old tokens that are irrelevant to most queries.

This mirrors a natural memory hierarchy:

- **Recent context:** working memory.
- **Older context:** long-term or episodic memory.

The key insight is therefore not simply sparsity. It is that **different parts of the context should receive different representational fidelity and computational budgets**.

---

## 3. Why a Fixed Percentage Is Not Enough

Let the total sequence length be \(N\), and let the dense recent region contain \(r = xN\) tokens.

If dense attention occurs only within the recent region, its cost is:

\[
O(r^2) = O(x^2N^2)
\]

If recent queries attend densely over the full historical context, the cost is:

\[
O(rN) = O(xN^2)
\]

In either case, a constant percentage leaves the architecture asymptotically quadratic. It reduces the coefficient but does not change the scaling law.

A stronger choice is a dense window \(W\) that is:

- fixed,
- logarithmic in sequence length,
- or sublinear in sequence length.

For example:

\[
W = O(1), \quad O(\log N), \quad \text{or} \quad O(N^\beta), \; \beta < 1
\]

A practical formulation is:

\[
W(N) = \min(W_{\max}, \rho N)
\]

At short contexts, the model can behave almost like a dense transformer. At long contexts, the working-memory region stops growing.

With a fixed sliding window, local attention across the sequence costs approximately:

\[
O(NW)
\]

This is much more attractive for contexts in the hundreds of thousands or millions of tokens.

---

## 4. Core Design Principle: Cascading Exactness

The architecture should be understood as a cascade of representational fidelity.

Every token begins in a high-resolution state when it is recent. As it ages, it is progressively moved into cheaper storage and cheaper access paths. Importantly, old information is not permanently discarded merely because it is old.

The desired behavior is:

1. A token enters the context in full-resolution working memory.
2. It receives exact dense attention while it remains recent.
3. When it leaves the hot window, it is stored in a compressed but individually addressable representation.
4. It also contributes to block-level and recurrent summaries.
5. A future query can retrieve its block or token representation.
6. The selected historical information is promoted back into exact attention.

Thus, age controls the **default cost** allocated to a token, not whether the token remains accessible.

A concise statement of the principle is:

> Information decays in resolution, not in recoverability.

---

## 5. Lessons from Kimi Delta Attention

Kimi Delta Attention contributes a useful model of long-term recurrent memory.

The important conceptual lessons are:

- A fixed-size recurrent state can preserve long-range information at constant per-token cost.
- Delta-rule updates can be substantially more expressive than naive linear attention.
- Channel-wise forgetting is more flexible than applying one decay value to an entire head.
- Different state dimensions can naturally learn different effective memory timescales.
- A recurrent mechanism is effective for broad, continuously accumulated state.
- Pure recurrent memory is not sufficient for every kind of long-context task, so periodic exact or global attention remains useful.

Kimi's architecture uses a layerwise hybrid rather than simply placing every mechanism in parallel inside every layer. The design alternates several KDA layers with a global MLA layer. This suggests a useful prior:

> Recurrent memory can carry cheap continuous state through most layers, while periodic attention layers recover exact historical information and refresh global interactions.

KDA is especially well suited to information such as:

- overall topic,
- document-level state,
- running sentiment,
- discourse progression,
- accumulated entities,
- task status,
- broad semantic context.

It is less naturally suited to preserving arbitrarily many exact details such as:

- identifiers,
- numbers,
- quotations,
- code symbols,
- rare facts,
- exact wording.

Therefore, recurrent memory should not become the sole representation of old context.

---

## 6. Lessons from DeepSeek Sparse Attention

DeepSeek Sparse Attention contributes a complementary mechanism.

Rather than compressing all historical information into a fixed-size state, it uses a lightweight indexer to identify the most relevant historical entries and then performs normal attention over the selected subset.

The useful conceptual lessons are:

- Historical access should be query-dependent.
- A cheap router can approximate the important mass of dense attention.
- Once relevant tokens are selected, exact attention can operate over them.
- The router can be trained initially to imitate dense attention distributions.
- Sparse attention works best when the selection mechanism and storage layout are designed with hardware efficiency in mind.
- Shared selection across heads can be more practical than independent token gathering per head.

DSA is especially well suited to questions such as:

- What exact fact was stated earlier?
- Which earlier code block defines this function?
- What number, name, or constraint was introduced far back in the sequence?
- Which historical passage is relevant to the current query?

Its limitation is that a flat indexer may still compare each query against the entire past, even if the expensive attention operation itself is sparse. The indexing path can therefore remain quadratic in sequence length, albeit with a much smaller constant.

This motivates a hierarchical retrieval mechanism.

---

## 7. Complementarity of Dense, Sparse, and Recurrent Mechanisms

The three mechanisms solve different problems.

### Dense local attention

Best for:

- active computation,
- recent syntax,
- exact copying,
- variable binding,
- current chain-of-thought state,
- local code semantics,
- nearby references.

### Sparse historical retrieval

Best for:

- exact old facts,
- specific quotations,
- historical identifiers,
- distant instructions,
- old code definitions,
- rare but important events.

### Recurrent or delta memory

Best for:

- diffuse global state,
- long-term topic continuity,
- accumulated semantic context,
- broad narrative progression,
- document-level summaries,
- information not localized to one particular historical block.

These mechanisms should be treated as complementary, but that does not imply that they must be implemented as independently normalized parallel attention streams.

---

## 8. Refined Architecture

## 8.1 Hot Working-Memory Window

Maintain a dense causal attention window over the latest \(W\) tokens.

A plausible range is:

- 8K tokens for lower-cost models,
- 16K tokens as a strong default,
- 32K tokens for reasoning- or code-heavy models.

The hot window stores full-resolution KV or the model's standard high-fidelity attention representation.

Every query attends densely to this region.

This window protects:

- exact local reasoning,
- syntax and grammar,
- token copying,
- nearby references,
- code and mathematical dependencies,
- the active user instruction,
- recent tool outputs.

---

## 8.2 Warm Per-Token Historical Memory

When a token leaves the hot window, it should not necessarily retain conventional full KV. Instead, it can be stored as a compact per-token latent representation.

Let \(c_s\) be the compact latent for historical token \(s\), where:

\[
\dim(c_s) \ll \dim(K_s) + \dim(V_s)
\]

When selected, the latent can be expanded or projected into the form needed by exact attention.

This provides two benefits:

1. Historical tokens remain individually recoverable.
2. KV-cache memory grows much more slowly.

The important distinction is between:

- **compression with addressability**, and
- **compression by irreversible aggregation**.

The architecture should strongly prefer the former.

---

## 8.3 Block-Level Historical Memory

Divide the historical sequence into blocks of \(B\) tokens.

Example block sizes:

- 256 tokens,
- 512 tokens,
- 1,024 tokens.

Each block stores one or more routing representations:

\[
r_b = f(c_{bB:(b+1)B})
\]

A single summary may be too lossy, so a block can maintain:

- several routing vectors,
- semantic centroids,
- structural markers,
- recurrent states,
- high-salience token keys,
- metadata about headings, code boundaries, speakers, or tools.

The block summary is primarily a **routing object**, not a complete replacement for the block's original information.

---

## 8.4 Hierarchical Historical Retrieval

The historical selector should itself be cascading.

### Stage A: Retrieve blocks

For query \(q_t\), score block summaries and select the top \(m\) blocks:

\[
\mathcal{B}_t = \operatorname{TopM}_b \; S_{\text{block}}(q_t, r_b)
\]

### Stage B: Retrieve tokens inside selected blocks

Within the chosen blocks, score individual token latents and select the top \(k\) historical entries:

\[
\mathcal{R}_t = \operatorname{TopK}_{s \in \mathcal{B}_t} S_{\text{token}}(q_t, c_s)
\]

### Stage C: Exact reintegration

The retrieved tokens are projected into attention keys and values and added to the exact candidate set.

The cascade is:

\[
\text{entire context}
\rightarrow
\text{relevant blocks}
\rightarrow
\text{relevant tokens}
\rightarrow
\text{exact attention}
\]

This avoids running a token-level router over the entire historical sequence for every query.

---

## 8.5 Persistent Anchor Memory

Some tokens should remain permanently eligible for exact access regardless of age.

Possible anchors include:

- system instructions,
- developer instructions,
- user constraints,
- section headings,
- function signatures,
- definitions,
- variable assignments,
- tool outputs,
- explicit decisions,
- task goals,
- tokens repeatedly retrieved in earlier steps,
- high-attention or high-surprisal tokens,
- user-marked memory.

Let \(\mathcal{A}_t\) be the anchor set.

The exact attention candidate set becomes:

\[
\mathcal{C}_t =
\underbrace{\{t-W, \ldots, t-1\}}_{\text{hot local window}}
\cup
\underbrace{\mathcal{R}_t}_{\text{retrieved historical tokens}}
\cup
\underbrace{\mathcal{A}_t}_{\text{persistent anchors}}
\]

---

## 8.6 Unified Exact Attention

The local, retrieved, and anchor tokens should preferably be merged before normalization.

Compute:

\[
o_t =
\operatorname{softmax}
\left(
\frac{q_t K_{\mathcal{C}_t}^{\top}}{\sqrt{d}}
\right)
V_{\mathcal{C}_t}
\]

This is preferable to separately computing:

- local attention,
- historical attention,
- anchor attention,

and then summing them.

Separate softmax operations normalize each branch independently. This can artificially force every branch to contribute substantial probability mass even when one branch is irrelevant.

A unified softmax preserves direct competition among:

- recent tokens,
- retrieved old tokens,
- persistent anchors.

Thus, the multiple mechanisms act primarily as **candidate producers**, not necessarily as independently normalized attention streams.

---

## 8.7 Recurrent Global Memory

A KDA-like recurrent state can be maintained alongside the explicit token memory.

Let \(M_t\) be the recurrent state:

\[
M_t = \operatorname{Update}(M_{t-1}, k_t, v_t, g_t)
\]

where the update includes:

- delta-rule correction,
- channel-wise forgetting,
- learned write strength,
- possibly separate erase and write controls.

The recurrent state should preserve diffuse information that may not map cleanly to one historical block.

There are two strong ways to integrate it.

### Option A: Layerwise alternation

Use several recurrent layers followed by one sparse-exact attention layer:

\[
\text{KDA}
\rightarrow
\text{KDA}
\rightarrow
\text{KDA}
\rightarrow
\text{hierarchical sparse exact attention}
\]

A 3:1 ratio is a reasonable starting point, but should be treated as an initialization for ablations rather than a fixed rule.

### Option B: Recurrent block construction

Use recurrent or delta-rule memory to produce richer block routing summaries.

In this interpretation:

- recurrent memory answers, "Which historical region might matter?"
- sparse exact attention answers, "What exactly was stored there?"

This is particularly attractive because it uses recurrent memory where compression is acceptable—routing—while retaining token-level representations for exact recovery.

---

## 9. Multiple Attention Streams: Considered, Not Assumed

One refinement discussed was the possibility of multiple attention streams. This is worth evaluating, but it should not become the architecture's premise without evidence.

Potential variants include:

### 9.1 Parallel branches

A layer might contain:

- dense local attention,
- sparse historical attention,
- recurrent global memory,

with a learned gate combining their outputs.

Potential benefit:

- each mechanism can specialize,
- all memory horizons are accessible in every layer.

Potential costs:

- separate normalization across branches,
- branch-scale mismatch,
- gate collapse,
- duplicated projections,
- irregular kernels,
- more difficult credit assignment,
- increased implementation complexity.

### 9.2 Head-group specialization

Different groups of heads could specialize in:

- local context,
- medium-range retrieval,
- long-range retrieval,
- anchors,
- recurrent summaries.

Potential benefit:

- fine-grained specialization inside each layer.

Potential costs:

- independent token gathering per head is hardware-unfriendly,
- head specialization may collapse or become redundant,
- selection and batching become complicated.

### 9.3 Layerwise specialization

Different layers use different operators:

- recurrent/delta layers,
- local dense layers,
- sparse global layers.

This is likely the strongest default because:

- kernels remain more regular,
- each layer has one clear operator,
- training may be more stable,
- infrastructure is simpler,
- periodic exact layers can refresh global state.

### 9.4 Multiple recurrent timescales

Instead of several attention streams, the recurrent state can contain fast-, medium-, and slow-decay channel groups.

This may capture much of the desired memory specialization without requiring multiple attention branches.

### Current prior

The preferred initial design is:

> Layerwise operator specialization plus multiple recurrent decay timescales, with a unified exact attention candidate set.

Parallel attention streams should be tested as an ablation rather than adopted as the default.

---

## 10. Age, Salience, Structure, and Query Relevance

A purely age-based cascade is not enough.

An instruction from 200K tokens ago can be more important than an incidental sentence from 100 tokens ago.

The routing priority should depend on multiple factors:

\[
P_{t,s} =
f(
\text{query relevance},
\text{recency},
\text{salience},
\text{structure},
\text{historical use}
)
\]

A practical token or block score might be:

\[
I_{t,s} =
I_{\text{content}}(q_t, c_s)
+
 b_{\text{age}}(t-s)
+
 b_{\text{structure}}(s)
+
 b_{\text{importance}}(s)
+
 b_{\text{reuse}}(s)
\]

Interpretation:

- **Content relevance:** Does the current query match the historical item?
- **Age bias:** Recent tokens receive a prior advantage.
- **Structural bias:** Headings, definitions, tool calls, and code boundaries receive boosts.
- **Importance bias:** High-salience tokens remain easier to recover.
- **Reuse bias:** Tokens repeatedly retrieved in the current reasoning episode remain warm.

The desired policy is therefore:

> Age-stratified, salience-overridable attention.

---

## 11. Temporary Promotion and Secondary Working Memory

When an old block is retrieved, it may remain relevant for several consecutive decoding steps.

Re-running the router for the same content at every token wastes compute and can cause retrieval instability.

A useful mechanism is temporary promotion:

1. Retrieve an old block or token set.
2. Add it to a secondary working-memory cache.
3. Retain it for \(H\) decoding steps or until a learned eviction gate removes it.
4. Allow exact attention to access it without re-running full retrieval.

This creates:

- a primary hot window for recent tokens,
- a secondary episodic cache for currently active historical evidence.

The model can therefore sustain a reasoning episode about an old passage without repeatedly rediscovering it.

Promotion criteria could include:

- retrieval score,
- attention mass after reintegration,
- repeated use across layers,
- gradient-based importance during training,
- explicit structural status.

---

## 12. Complexity Analysis

Let:

- \(N\): total context length,
- \(W\): dense recent window,
- \(B\): historical block size,
- \(m\): number of selected blocks,
- \(k\): number of selected historical tokens,
- \(A\): number of anchors.

### Exact attention cost per query

The exact candidate set has approximately:

\[
W + k + A
\]

tokens.

Exact attention therefore costs:

\[
O(W + k + A)
\]

per decoding query, ignoring projection dimension constants.

### Hierarchical routing cost

A naive block scan costs:

\[
O(N/B)
\]

per query.

Token routing inside selected blocks costs:

\[
O(mB)
\]

A more advanced tree, clustering, or approximate nearest-neighbor index could reduce block routing toward:

\[
O\left(m \log \frac{N}{B}\right)
\]

The overall target is therefore approximately:

\[
O\left(
W + k + A + mB + m\log\frac{N}{B}
\right)
\]

per query.

### Memory cost

If hot tokens store full KV and historical tokens store compact latent vectors:

\[
O(W d_{KV})
+
O(N d_c)
+
O\left(\frac{N}{B} d_r\right)
\]

where:

- \(d_{KV}\) is full key-value storage,
- \(d_c\) is compact per-token latent size,
- \(d_r\) is block-routing representation size,
- \(d_c \ll d_{KV}\).

This preserves token addressability while reducing historical cache memory.

---

## 13. Illustrative 1M-Token Configuration

A plausible first implementation for a one-million-token model:

| Component | Example configuration |
|---|---:|
| Dense hot window | 16K tokens |
| Historical block size | 512 tokens |
| Top retrieved blocks | 8 |
| Historical tokens selected | 1,024 |
| Persistent anchors | 256 tokens |
| Temporary promoted cache | 2K–8K tokens |
| Recurrent-to-exact layer ratio | 3:1 |
| Historical token representation | Compact MLA-style latent |
| Block routing representation | 2–8 vectors per block |
| Exact attention normalization | Unified softmax |

For each decoding query:

1. Read the latest 16K tokens directly.
2. Read the temporary promoted historical cache.
3. Score block summaries.
4. Select the top historical blocks.
5. Score token latents inside those blocks.
6. Select the most relevant old tokens.
7. Merge local tokens, promoted tokens, retrieved tokens, and anchors.
8. Run one exact softmax over the merged set.
9. Update recurrent memory.
10. Optionally promote heavily used historical tokens for subsequent steps.

---

## 14. Training Strategy

A sparse or hierarchical model is unlikely to learn reliable long-range retrieval from ordinary next-token prediction alone. Local context solves too many examples, so the historical mechanism may remain undertrained.

A staged training recipe is preferable.

## 14.1 Phase 1: Dense Teacher or Dense Warm-Up

Train with dense attention at shorter contexts, or temporarily enable a dense teacher model.

Collect:

- dense attention distributions,
- attention mass per block,
- top historical tokens,
- persistent high-importance tokens,
- cross-layer retrieval patterns,
- examples where old context changes the prediction.

The dense model provides supervision for the router.

---

## 14.2 Phase 2: Block Router Training

Freeze most of the main model and train the block router to reproduce dense block-level attention mass.

Let \(p_{\text{block}}\) be dense teacher attention aggregated by block and \(\hat p_{\text{block}}\) be router scores.

Use:

\[
\mathcal{L}_{\text{block}}
=
D_{\mathrm{KL}}
\left(
 p_{\text{block}}
\Vert
 \hat p_{\text{block}}
\right)
\]

Alternative or additional losses:

- ranking loss,
- top-\(m\) recall loss,
- contrastive block retrieval,
- importance-weighted binary classification.

---

## 14.3 Phase 3: Token Router Training

Inside teacher-selected or model-selected blocks, train the token router to reproduce dense token-level importance.

\[
\mathcal{L}_{\text{token}}
=
D_{\mathrm{KL}}
\left(
 p_{\text{token}}
\Vert
 \hat p_{\text{token}}
\right)
\]

The objective should prioritize recall of important tokens over perfect probability calibration.

Useful metrics:

- recall of top dense-attention tokens,
- fraction of dense attention mass captured,
- answer accuracy after sparsification,
- retrieval stability across adjacent decoding steps.

---

## 14.4 Phase 4: Sparse End-to-End Adaptation

Enable actual top-\(m\) block selection and top-\(k\) token selection.

Train the whole model end-to-end.

Important interventions:

### Local-window dropout

Randomly hide parts of the recent window so the model cannot solve every task locally.

### Variable hot-window size

Train with different values of \(W\) so the model does not memorize a fixed boundary.

### Historical evidence placement

Place critical evidence at many distances, not only at the beginning of the sequence.

### Multi-block composition

Create tasks requiring evidence from two or more distant blocks.

### Exact-detail retrieval

Include numbers, names, code symbols, quotations, and identifiers that cannot be reconstructed from a semantic summary.

### Router budget penalty

Penalize excessive retrieval:

\[
\mathcal{L}_{\text{budget}}
= \lambda_b m + \lambda_t k
\]

or use a differentiable expected-compute penalty.

### Recall penalty

Explicitly penalize missing teacher-important tokens or blocks.

### Promotion consistency

Reward stable reuse of a retrieved block over a short reasoning episode rather than forcing rediscovery at every step.

---

## 15. Preventing Local-Branch Collapse

The greatest training risk is that the model learns to ignore historical memory because next-token prediction is predominantly local.

Symptoms include:

- low router entropy,
- repetitive selection of recent blocks,
- sparse attention receiving negligible mass,
- recurrent state becoming unused,
- poor retrieval despite good language-model loss.

Countermeasures:

- mask or corrupt local context on selected examples,
- make some tasks impossible without old evidence,
- use explicit retrieval supervision,
- train on delayed-reference tasks,
- add synthetic and natural long-range dependencies,
- monitor historical attention mass,
- introduce curriculum from easy single-block retrieval to multi-block reasoning,
- penalize the model when local-only predictions conflict with old evidence.

The sparse and recurrent pathways should be trained as first-class capabilities, not treated as passive efficiency modules.

---

## 16. Cross-Branch Normalization

If local and historical attention are computed independently:

\[
o_{\text{local}}
=
\operatorname{softmax}(qK_L^\top)V_L
\]

\[
o_{\text{historical}}
=
\operatorname{softmax}(qK_H^\top)V_H
\]

then each branch is normalized to total probability mass one.

Simply adding the outputs is not equivalent to attention over the union of the tokens. It can over-weight a weak historical branch or force a contribution from an irrelevant stream.

The preferred solution for exact token memories is:

\[
o =
\operatorname{softmax}
\left(
q[K_L;K_H;K_A]^\top
\right)
[V_L;V_H;V_A]
\]

A recurrent global state may still be fused separately because it is not naturally represented as a set of exact tokens. It should use a learned gate, normalization, or residual scale.

---

## 17. Historical Composition Problem

Retrieving one old block at a time is not enough for many reasoning tasks.

The model may need to:

- compare two distant sections,
- combine a rule from one block with data from another,
- identify contradictions across the history,
- reason about a sequence of old events,
- integrate several code definitions.

Possible solutions:

1. Retrieve multiple blocks per query.
2. Preserve exact local attention inside each block during initial encoding.
3. Use periodic global sparse layers.
4. Allow promoted historical blocks to interact over several layers.
5. Use recursive block summaries that exchange information.
6. Add multi-hop router passes.
7. Maintain a secondary working-memory cache for retrieved evidence.

A useful pattern is:

- first sparse layer retrieves evidence,
- following recurrent or local layers process it,
- later sparse layer retrieves additional evidence conditioned on the updated state.

This naturally creates multi-hop historical reasoning.

---

## 18. Hardware-Aware Design

A theoretically sparse model can still be slower than dense attention if the implementation performs irregular token gathers and fragmented memory reads.

Key design constraints:

### Block-level retrieval

Retrieve contiguous blocks before selecting tokens. This improves memory locality.

### Paged historical cache

Store historical latents in fixed-size pages aligned with the routing blocks.

### Shared selection

Share selected blocks or token sets across groups of heads where possible.

### Fixed retrieval buckets

Use a small number of allowed values for \(m\), \(k\), and promoted-cache size so kernels can be batched efficiently.

### Batched expansion

If historical token latents must be projected into KV, expand all selected entries in one fused operation.

### Unified sparse kernel

Merge local, promoted, retrieved, and anchor entries into a compact contiguous candidate buffer before attention.

### Avoid per-token dynamic control flow

The router may be dynamic, but the execution path should remain regular at the block and batch level.

### Quantized routing

Block and token routing can potentially use lower precision than exact attention.

The model architecture and the memory layout must be co-designed. Sparse attention should not be treated as an abstract masking scheme layered on top of a dense implementation.

---

## 19. Storage Cascade

A practical storage hierarchy:

| Tier | Content | Representation | Default access |
|---|---|---|---|
| Hot | Latest tokens | Full KV | Always dense |
| Promoted | Currently relevant old tokens | Full or expanded KV | Temporarily dense |
| Warm | Recently aged-out history | Compact per-token latents | Token retrieval |
| Cold | Distant history | Compact token latents + block summaries | Hierarchical retrieval |
| Recurrent | Global accumulated state | Fixed-size matrix/state | Always available |
| Anchors | Critical tokens anywhere | Full or compact exact entries | Always eligible |

Demotion can proceed as:

\[
\text{Hot}
\rightarrow
\text{Warm}
\rightarrow
\text{Cold}
\]

Promotion can proceed as:

\[
\text{Cold/Warm}
\rightarrow
\text{Retrieved}
\rightarrow
\text{Promoted working memory}
\]

The architecture is therefore a dynamic memory manager rather than merely a sparse mask.

---

## 20. Potential Failure Modes

## 20.1 Router misses

The correct historical block may never enter the candidate set.

Mitigations:

- high-recall routing,
- multiple routing vectors per block,
- anchor pool,
- recurrent hints,
- multi-hop retrieval,
- uncertainty-triggered expansion of \(m\) or \(k\).

## 20.2 Irreversible compression

A fixed-size summary may merge several unrelated facts and lose exact details.

Mitigation:

- use summaries for routing,
- preserve individually addressable token latents,
- avoid replacing all old tokens with one recurrent state.

## 20.3 Recency over-bias

The router may over-select recent history even when distant context matters.

Mitigation:

- train with variable evidence distances,
- use distance-balanced negatives,
- regularize age priors,
- evaluate retrieval by age bucket.

## 20.4 Local-path collapse

The model ignores long-term mechanisms.

Mitigation:

- retrieval-dependent training data,
- local dropout,
- explicit router supervision,
- historical counterfactuals.

## 20.5 Recurrent-memory saturation

The recurrent state becomes overloaded or forgets rare information.

Mitigation:

- channel-wise timescales,
- separate erase/write controls,
- periodic exact refresh,
- use recurrent state for diffuse information rather than exact storage.

## 20.6 Retrieval thrashing

Adjacent queries retrieve different blocks despite working on the same historical evidence.

Mitigation:

- temporary promotion,
- hysteresis in eviction,
- reuse bias,
- cache-aware routing.

## 20.7 Hardware underutilization

Sparse theoretical FLOPs do not translate into wall-clock gains.

Mitigation:

- block-contiguous layouts,
- shared selection,
- fused kernels,
- fixed retrieval buckets,
- hardware-aware benchmarking.

## 20.8 Exact attention overload

The union of local, promoted, anchor, and retrieved tokens becomes too large.

Mitigation:

- explicit compute budget,
- dynamic top-\(k\),
- anchor eviction or compression,
- per-layer retrieval rather than every layer,
- promotion TTL.

---

## 21. Dynamic Compute Allocation

A fixed retrieval budget may be wasteful for easy queries and insufficient for difficult ones.

The model can estimate uncertainty and adapt its historical budget.

For example:

\[
k_t = k_{\min} + g(u_t)(k_{\max}-k_{\min})
\]

where \(u_t\) is an uncertainty or retrieval-need score.

Possible signals:

- entropy of the block router,
- low maximum block score,
- disagreement between local and recurrent predictions,
- low confidence in next-token logits,
- repeated failed retrieval,
- explicit long-range reference markers.

The model may respond by:

- selecting more blocks,
- selecting more tokens,
- performing a second retrieval hop,
- promoting blocks for longer,
- invoking a broader global attention layer.

This converts the architecture from fixed sparse attention into **adaptive-compute memory access**.

---

## 22. Research Questions

The architecture raises several substantive research questions.

### 22.1 What is the optimal hot-window schedule?

Compare:

- fixed \(W\),
- capped percentage,
- logarithmic growth,
- learned dynamic window,
- task-conditioned window.

### 22.2 Should recurrent memory be parallel or layerwise?

Compare:

- recurrent and exact branches in every layer,
- alternating recurrent and exact layers,
- recurrent summaries used only for routing,
- recurrent state only in selected layers.

### 22.3 How many routing vectors should each block store?

One vector may be too coarse. Compare:

- one centroid,
- multiple semantic centroids,
- learned memory slots,
- structure-aware vectors,
- high-salience token representatives.

### 22.4 What should be promoted?

Compare promotion of:

- individual tokens,
- contiguous blocks,
- expanded KV,
- compact latents,
- summaries plus exact excerpts.

### 22.5 How long should promoted memory persist?

Compare:

- fixed TTL,
- attention-mass-based eviction,
- learned gating,
- LRU-style eviction,
- task-boundary eviction.

### 22.6 How should anchors be selected?

Compare:

- rule-based structural anchors,
- learned salience,
- cumulative historical attention,
- surprisal,
- user or system annotations.

### 22.7 Can the router be made sublinear?

Explore:

- tree indexes,
- learned clustering,
- product quantization,
- approximate nearest neighbor search,
- recurrent routing states,
- coarse-to-fine search.

### 22.8 Can multiple timescales emerge inside one recurrent state?

Investigate whether channel-wise forgetting naturally produces:

- fast working state,
- medium discourse state,
- slow global state.

If successful, this may reduce the need for explicitly separate memory streams.

---

## 23. Recommended Ablation Matrix

A serious evaluation should isolate each architectural choice.

### Attention layout

- dense transformer,
- sliding-window only,
- flat sparse retrieval,
- hierarchical sparse retrieval,
- recurrent only,
- recurrent + flat sparse,
- recurrent + hierarchical sparse.

### Operator arrangement

- all mechanisms parallel,
- headwise specialization,
- layerwise alternation,
- recurrent summaries only for routing.

### Layer ratios

- 1 recurrent : 1 exact,
- 3 recurrent : 1 exact,
- 7 recurrent : 1 exact,
- 15 recurrent : 1 exact.

### Hot-window size

- 4K,
- 8K,
- 16K,
- 32K.

### Block size

- 128,
- 256,
- 512,
- 1,024 tokens.

### Retrieval budget

- top 2, 4, 8, 16 blocks,
- top 256, 512, 1,024, 2,048 historical tokens.

### Promotion

- disabled,
- fixed TTL,
- learned TTL,
- attention-based eviction.

### Anchors

- none,
- structural only,
- learned only,
- hybrid structural + learned.

### Router supervision

- no teacher,
- dense-attention KL,
- ranking loss,
- end-task supervision only,
- combined losses.

---

## 24. Evaluation Suite

The architecture should not be evaluated only on needle-in-a-haystack tests.

A useful benchmark suite should include:

### Exact retrieval

- names,
- numbers,
- UUIDs,
- quotations,
- code identifiers,
- key-value pairs.

### Multi-hop retrieval

- combine facts from multiple distant blocks,
- retrieve a rule and apply it to distant data,
- compare old events.

### Long-document reasoning

- legal or technical documents,
- research papers,
- books,
- long reports,
- multi-file codebases.

### Dialogue continuity

- old user preferences,
- constraints introduced many turns earlier,
- corrections and decisions,
- conflicting instructions.

### Code tasks

- distant symbol definitions,
- cross-file dependencies,
- long execution traces,
- bug localization,
- repository-level reasoning.

### Streaming state

- cumulative counters,
- changing world state,
- event sequences,
- hidden state shifts,
- long-running agents.

### Adversarial distractors

- semantically similar but incorrect blocks,
- repeated names with different values,
- overwritten variables,
- stale instructions,
- near-duplicate passages.

### Efficiency

Measure:

- prefill latency,
- decode latency,
- KV-cache memory,
- router overhead,
- actual GPU utilization,
- end-to-end throughput,
- energy or FLOPs per generated token.

The key metric is not merely sparse FLOPs. It is the quality-efficiency frontier under real hardware constraints.

---

## 25. Minimal Prototype

A minimal research prototype could avoid building a full new pretraining stack.

### Stage 1

Take an existing transformer and replace selected attention layers with:

- local sliding-window attention,
- block-level routing,
- token selection inside blocks,
- exact attention over the merged candidate set.

### Stage 2

Train only:

- routing projections,
- block summaries,
- token selector,
- historical latent expansion layers.

Keep most base-model weights frozen.

### Stage 3

Add recurrent delta layers or recurrent routing summaries.

### Stage 4

Fine-tune the full model on synthetic and natural long-context tasks.

This staged approach isolates whether hierarchical sparse retrieval works before introducing recurrent memory and dynamic promotion.

---

## 26. Conceptual Pseudocode

```python
class CascadingAttentionState:
    hot_kv                 # latest W tokens
    historical_latents     # compact per-token representations
    block_summaries        # routing vectors for historical blocks
    anchors                # permanently eligible critical entries
    promoted_cache         # temporarily active historical evidence
    recurrent_state        # KDA-like diffuse memory


def decode_step(query, state, budget):
    # 1. Always-visible exact candidates
    local = state.hot_kv
    promoted = state.promoted_cache
    anchors = state.anchors

    # 2. Coarse historical retrieval
    block_scores = score_blocks(query, state.block_summaries)
    selected_blocks = top_m(block_scores, budget.num_blocks)

    # 3. Fine token retrieval inside selected blocks
    candidate_latents = gather_latents(
        state.historical_latents,
        selected_blocks,
    )
    token_scores = score_tokens(query, candidate_latents)
    selected_latents = top_k(token_scores, budget.num_tokens)

    # 4. Expand selected historical entries into exact KV
    retrieved_kv = expand_to_kv(selected_latents)

    # 5. Build one exact candidate set
    exact_candidates = concat(
        local,
        promoted,
        anchors,
        retrieved_kv,
    )

    # 6. Unified exact softmax
    exact_output = attention(query, exact_candidates)

    # 7. Read recurrent diffuse memory
    recurrent_output = read_recurrent(query, state.recurrent_state)

    # 8. Fuse recurrent memory with exact attention
    gate = recurrent_gate(query, exact_output, recurrent_output)
    output = exact_output + gate * recurrent_output

    # 9. Update memory state
    state.recurrent_state = update_recurrent(
        state.recurrent_state,
        query,
        output,
    )

    state.promoted_cache = update_promoted_cache(
        state.promoted_cache,
        retrieved_kv,
        exact_attention_mass=exact_output.attention_weights,
    )

    return output, state
```

This pseudocode preserves the core principle:

- multiple mechanisms discover or retain memory,
- but exact token memories compete in one attention normalization.

---

## 27. Strongest Version of the Research Thesis

The weaker claim is:

> Recent tokens deserve dense attention; old tokens deserve sparse attention.

The stronger claim is:

> Long-context transformers should manage memory through dynamic fidelity. Tokens begin as exact working memory, are gradually compressed and indexed as they age, and are selectively reconstructed or promoted when future computation requires them.

The architecture is not merely an attention mask. It is a learned memory hierarchy with:

- representation demotion,
- hierarchical routing,
- exact recall,
- recurrent global state,
- salience-aware persistence,
- temporary promotion,
- adaptive compute allocation.

This makes it closer to a memory system than a conventional transformer attention variant.

---

## 28. Current Recommended Design

The architecture most worth building first is:

1. A fixed 8K–16K dense hot window.
2. Compact per-token historical latents rather than full historical KV.
3. Historical blocks of roughly 256–1,024 tokens.
4. Multiple routing vectors per block.
5. Coarse block retrieval followed by fine token retrieval.
6. A persistent anchor pool for critical information.
7. A temporary promoted cache for repeatedly used historical evidence.
8. One unified exact softmax over local, promoted, retrieved, and anchor tokens.
9. KDA-like recurrent layers or recurrent routing summaries for diffuse global memory.
10. Layerwise alternation as the default operator arrangement.
11. Parallel multiple-attention streams only as an ablation.
12. Dense-teacher router warm-up followed by sparse end-to-end adaptation.
13. Training data designed to force exact and compositional use of old context.
14. Hardware-aware block layouts and fused candidate gathering.

---

## 29. Final View

The initial intuition is correct: the latest context deserves more exact attention than the distant past.

The architecture becomes substantially stronger when recency is treated as one component of a general memory hierarchy rather than as a hard binary split.

The final model should behave as follows:

- recent tokens are exact by default,
- old tokens are compressed but addressable,
- summaries route rather than replace,
- recurrent memory preserves diffuse state,
- important information can override age,
- retrieved history is promoted into exact computation,
- repeated historical evidence remains temporarily active,
- expensive attention is spent only where the current computation benefits from it.

The defining idea is therefore:

> **Cascading attention should be a reversible hierarchy of memory fidelity, not merely a recent-window mask attached to a sparse backend.**

