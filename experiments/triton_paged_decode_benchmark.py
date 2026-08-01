"""End-to-end 32K paged retrieval decode benchmark using fused Triton attention."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from triton_paged_attention import paged_decode_attention, reference_attention


def paged_candidates(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_summaries: torch.Tensor,
    *,
    local_window: int,
    block_size: int,
    top_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coarse route and assemble local + contiguous selected page candidates."""
    batch_size, heads, context, head_dim = keys.shape
    historical_length = context - local_window
    scores = torch.einsum("bhd,bhgd->bhg", query, block_summaries).mean(dim=1)
    selected_blocks = scores.topk(top_blocks, dim=-1).indices
    pages = keys[:, :, :historical_length].view(batch_size, heads, historical_length // block_size, block_size, head_dim).permute(0, 2, 1, 3, 4)
    value_pages = values[:, :, :historical_length].view(batch_size, heads, historical_length // block_size, block_size, head_dim).permute(0, 2, 1, 3, 4)
    batch_indices = torch.arange(batch_size, device=query.device).unsqueeze(1)
    selected_keys = pages[batch_indices, selected_blocks].permute(0, 2, 1, 3, 4).flatten(start_dim=2, end_dim=3).contiguous()
    selected_values = value_pages[batch_indices, selected_blocks].permute(0, 2, 1, 3, 4).flatten(start_dim=2, end_dim=3).contiguous()
    return torch.cat((keys[:, :, historical_length:], selected_keys), dim=2).contiguous(), torch.cat((values[:, :, historical_length:], selected_values), dim=2).contiguous()


def benchmark(operation, iterations: int) -> float:
    for _ in range(30):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.mean(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--top-blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.context - args.local_window) % args.block_size:
        parser.error("historical context must divide evenly into block size")
    torch.manual_seed(7)
    query = torch.randn(args.batch_size, args.heads, args.head_dim, device="cuda", dtype=torch.float16)
    keys = torch.randn(args.batch_size, args.heads, args.context, args.head_dim, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    historical_length = args.context - args.local_window
    summaries = keys[:, :, :historical_length].view(args.batch_size, args.heads, historical_length // args.block_size, args.block_size, args.head_dim).mean(dim=3).contiguous()
    candidate_keys, candidate_values = paged_candidates(query, keys, values, summaries, local_window=args.local_window, block_size=args.block_size, top_blocks=args.top_blocks)
    torch.testing.assert_close(paged_decode_attention(query, candidate_keys, candidate_values), reference_attention(query, candidate_keys, candidate_values), rtol=2e-2, atol=2e-2)
    dense_us = benchmark(lambda: reference_attention(query, keys, values), args.iterations)
    paged_torch_us = benchmark(lambda: reference_attention(query, *paged_candidates(query, keys, values, summaries, local_window=args.local_window, block_size=args.block_size, top_blocks=args.top_blocks)), args.iterations)
    paged_triton_us = benchmark(lambda: paged_decode_attention(query, *paged_candidates(query, keys, values, summaries, local_window=args.local_window, block_size=args.block_size, top_blocks=args.top_blocks)), args.iterations)
    candidates = args.local_window + args.top_blocks * args.block_size
    report = {
        "context": args.context,
        "candidates": candidates,
        "routed_entries": historical_length // args.block_size + args.top_blocks * args.block_size,
        "dense_torch_mean_us": dense_us,
        "paged_torch_mean_us": paged_torch_us,
        "paged_triton_mean_us": paged_triton_us,
        "triton_vs_dense_speedup": dense_us / paged_triton_us,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
