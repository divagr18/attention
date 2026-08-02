#!/usr/bin/env python3
"""Regression tests for causal page-tree traversal and gathered attention."""

from __future__ import annotations

import torch

from hierarchical_page_tree import HierarchicalPageTree
from cascading_kv_attention import CascadingAttentionConfig, CascadingKVAttention
from multi_vector_page_tree import MultiVectorPageTree
from train_tiny_transformer import CausalAttention, HierarchicalRouter


def main() -> None:
    torch.manual_seed(7)
    # A single tree level exposes every leaf as a child of the root, making
    # selection deterministic and independently checking score ordering.
    pages = torch.eye(16).unsqueeze(0)
    tree = HierarchicalPageTree.from_page_keys(pages, fanout=16)
    found = tree.search(torch.nn.functional.one_hot(torch.tensor([11]), 16).float(), beam=4, retrieval_pages=2)
    assert found.page_indices.tolist()[0][0] == 11
    assert found.score_count == 16 and found.depth == 1

    causal = HierarchicalPageTree(fanout=4, max_pages=16)
    for page in pages[:, :9].unbind(dim=1):
        causal.append(page)
    assert causal.page_count == 9
    result = causal.search(torch.randn(1, 16), beam=4, retrieval_pages=4)
    assert result.page_indices.shape == (1, 4)
    assert result.score_count <= 4 * 4 * max(1, result.depth)

    slots = torch.zeros(1, 32, 4, 32)
    record_pages = torch.tensor([2, 10, 23, 31])
    slots[0, record_pages, 0, record_pages] = 1
    multi = MultiVectorPageTree.from_page_slots(slots, fanout=4)
    multi_found = multi.search(torch.nn.functional.one_hot(torch.tensor([23]), 32).float(), beam=4, retrieval_pages=1)
    assert multi_found.page_indices.item() == 23
    assert multi_found.score_count < 32 * 4
    causal_multi = MultiVectorPageTree(fanout=4, slots=4, max_pages=32)
    for page in slots.unbind(dim=1):
        causal_multi.append(page)
    causal_found = causal_multi.search(torch.nn.functional.one_hot(torch.tensor([23]), 32).float(), beam=4, retrieval_pages=1)
    assert causal_found.page_indices.item() == 23

    # The gathered tree candidates must use the same unified softmax as the
    # masked reference over precisely that candidate set.
    attention = CausalAttention(16, 4).eval()
    hidden = torch.randn(2, 64, 16)
    retrieved = torch.tensor([[7, 8, 9], [19, 20, 21]])
    tree_output, _ = attention(hidden, variant="tree", local_window=8, evidence_positions=None, retrieved_indices=retrieved)
    reference, _ = attention(hidden, variant="tree", local_window=8, evidence_positions=None, retrieved_indices=retrieved, capture_attention=True)
    error = (tree_output - reference).abs().max().item()
    assert error < 1e-5, f"tree candidate attention mismatch: {error}"

    router = HierarchicalRouter(16).eval()
    embeddings = torch.randn(2, 320, 16)
    output = router(
        embeddings,
        historical_length=256,
        block_size=16,
        top_blocks=4,
        top_tokens=1,
        retrieval_unit="page",
        router_index="tree",
        tree_fanout=4,
        tree_beam=4,
        retrieval_pages=4,
    )
    # In inference the tree scores only selected pages' token keys; all other
    # token-score slots remain -inf and cannot hide an exhaustive old-token scan.
    assert int(torch.isfinite(output.token_scores).sum(dim=1).max()) == 64
    assert output.tree_score_count < 16 * 16

    core = CascadingKVAttention(4, CascadingAttentionConfig(hot_window=8, page_size=4, tree_fanout=4, tree_beam=2, retrieval_pages=2))
    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 2, 16, 4)
    v = torch.randn_like(k)
    router_pages = k[:, :, :8].view(1, 2, 2, 4, 4).mean(dim=(1, 3))
    tree = core.make_tree(router_pages)
    combined, search = core(q, k, v, tree)
    batch = torch.arange(1).unsqueeze(1)
    page_k = k[:, :, :8].view(1, 2, 2, 4, 4).permute(0, 2, 1, 3, 4)[batch, search.page_indices].permute(0, 2, 1, 3, 4).flatten(2, 3)
    page_v = v[:, :, :8].view(1, 2, 2, 4, 4).permute(0, 2, 1, 3, 4)[batch, search.page_indices].permute(0, 2, 1, 3, 4).flatten(2, 3)
    candidates_k, candidates_v = torch.cat((k[:, :, 8:], page_k), dim=2), torch.cat((v[:, :, 8:], page_v), dim=2)
    manual_scores = torch.einsum("bhd,bhld->bhl", q, candidates_k) / 2
    manual = torch.einsum("bhl,bhld->bhd", manual_scores.softmax(-1), candidates_v)
    assert torch.allclose(combined, manual, atol=1e-6)
    print(f"tree: depth={output.tree_depth} scored={output.tree_score_count} selected_pages=4")
    print(f"tree candidate attention: max error {error:.2e}")


if __name__ == "__main__":
    main()
