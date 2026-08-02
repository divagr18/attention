#!/usr/bin/env python3
"""Parity gates for the model-neutral cascading K/V attention core.

Gate 1: with zero completed pages the core must equal dense attention over the
full sequence (validates the unified softmax and fp32 accumulation).
Gate 2: with every page retrieved the gathered candidate set equals the full
sequence, so the core must again equal dense (validates the page gather path).
"""

from __future__ import annotations

import torch

from cascading_kv_attention import CascadingAttentionConfig, CascadingKVAttention


def dense_reference(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bhd,bhld->bhl", query.float(), keys.float()) / (query.size(-1) ** 0.5)
    return torch.einsum("bhl,bhld->bhd", scores.softmax(dim=-1), values.float())


def main() -> None:
    torch.manual_seed(7)
    batch, heads, head_dim, length = 2, 4, 32, 24
    config = CascadingAttentionConfig(hot_window=8, page_size=4, tree_fanout=4, tree_beam=4, retrieval_pages=4)
    core = CascadingKVAttention(head_dim, config).eval()
    keys = torch.randn(batch, heads, length, head_dim)
    values = torch.randn(batch, heads, length, head_dim)
    query = torch.randn(batch, heads, head_dim)

    output, search = core(query, keys, values, tree=None)
    assert search is None, "page_count=0 must skip the tree"
    error = (output.float() - dense_reference(query, keys, values)).abs().max().item()
    assert error < 1e-4, f"gate1 (local-only) mismatch: {error}"
    print(f"gate1 local-only parity: max error {error:.2e}")

    page_count = (length - config.hot_window) // config.page_size
    assert page_count == 4
    tree = core.make_tree(torch.randn(batch, page_count, head_dim))
    output, search = core(query, keys, values, tree=tree)
    assert search is not None and search.page_indices.size(1) == page_count, "gate2 must retrieve every page"
    error = (output.float() - dense_reference(query, keys, values)).abs().max().item()
    assert error < 1e-4, f"gate2 (full-retrieval) mismatch: {error}"
    print(f"gate2 full-retrieval parity: max error {error:.2e}")

    forced_page = 2
    output, search = core(query, keys, values, tree=tree, force_page_indices=torch.tensor([[forced_page], [forced_page]]))
    assert search is None, "force_page_indices must bypass the tree search"
    paged = page_count * config.page_size
    page_start = forced_page * config.page_size
    ref_keys = torch.cat((keys[:, :, paged:], keys[:, :, page_start : page_start + config.page_size]), dim=2)
    ref_values = torch.cat((values[:, :, paged:], values[:, :, page_start : page_start + config.page_size]), dim=2)
    error = (output.float() - dense_reference(query, ref_keys, ref_values)).abs().max().item()
    assert error < 1e-4, f"gate3 (oracle) mismatch: {error}"
    print(f"gate3 oracle-force parity: max error {error:.2e}")


if __name__ == "__main__":
    main()
