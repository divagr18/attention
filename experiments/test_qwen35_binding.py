#!/usr/bin/env python3
"""CPU parity for the Qwen cascading attention function (no model required).

Gate 1: prefix below the hot window -> page_count=0 -> dense over the full cache.
Gate 2: prefix sized so every page is retrieved -> candidates == full cache -> dense.
"""

from __future__ import annotations

import torch

from cascading_kv_attention import CascadingAttentionConfig
from qwen35_cascading_binding import make_cascading_attention

HEADS, KV_HEADS, HEAD_DIM = 16, 4, 32


def dense_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if query.size(1) != key.size(1):
        groups = query.size(1) // key.size(1)
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhd,bhld->bhl", query[:, :, 0, :].float(), key.float()) / (query.size(-1) ** 0.5)
    return torch.einsum("bhl,bhld->bhd", scores.softmax(dim=-1), value.float()).unsqueeze(2).transpose(1, 2)


def run(config: CascadingAttentionConfig, length: int) -> float:
    fn = make_cascading_attention(config)
    query = torch.randn(1, HEADS, 1, HEAD_DIM)
    key = torch.randn(1, KV_HEADS, length, HEAD_DIM)
    value = torch.randn(1, KV_HEADS, length, HEAD_DIM)
    output, _ = fn(None, query, key, value)
    return (output.float() - dense_reference(query, key, value)).abs().max().item()


def main() -> None:
    torch.manual_seed(7)
    config = CascadingAttentionConfig(hot_window=8, page_size=4, tree_fanout=4, tree_beam=4, retrieval_pages=4)
    gate1 = run(config, length=6)
    assert gate1 < 1e-4, f"gate1 (page_count=0) mismatch: {gate1}"
    print(f"gate1 page_count=0 parity: max error {gate1:.2e}")
    gate2 = run(config, length=config.hot_window + config.retrieval_pages * config.page_size)
    assert gate2 < 1e-4, f"gate2 (full retrieval) mismatch: {gate2}"
    print(f"gate2 full-retrieval parity: max error {gate2:.2e}")


if __name__ == "__main__":
    main()
