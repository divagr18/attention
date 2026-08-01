"""Measure decode-time savings from reusing promoted historical page IDs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from triton_paged_attention import paged_gather_decode_attention


def select_blocks(query: torch.Tensor, summaries: torch.Tensor, top_blocks: int) -> torch.Tensor:
    return torch.einsum("bhd,bhgd->bhg", query, summaries).mean(dim=1).topk(top_blocks, dim=-1).indices.contiguous()


def timed(operation, iterations: int) -> float:
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
    parser.add_argument("--context", type=int, default=131072)
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
    historical = args.context - args.local_window
    summaries = keys[:, :, :historical].view(args.batch_size, args.heads, historical // args.block_size, args.block_size, args.head_dim).mean(dim=3).contiguous()
    promoted_blocks = select_blocks(query, summaries, args.top_blocks)
    torch.testing.assert_close(
        paged_gather_decode_attention(query, keys, values, promoted_blocks, local_window=args.local_window, block_size=args.block_size),
        paged_gather_decode_attention(query, keys, values, select_blocks(query, summaries, args.top_blocks), local_window=args.local_window, block_size=args.block_size),
    )
    router_us = timed(lambda: select_blocks(query, summaries, args.top_blocks), args.iterations)
    rerouted_us = timed(lambda: paged_gather_decode_attention(query, keys, values, select_blocks(query, summaries, args.top_blocks), local_window=args.local_window, block_size=args.block_size), args.iterations)
    promoted_us = timed(lambda: paged_gather_decode_attention(query, keys, values, promoted_blocks, local_window=args.local_window, block_size=args.block_size), args.iterations)
    report = {
        "context": args.context,
        "local_window": args.local_window,
        "block_size": args.block_size,
        "top_blocks": args.top_blocks,
        "router_only_mean_us": router_us,
        "reroute_each_query_mean_us": rerouted_us,
        "promoted_cache_mean_us": promoted_us,
        "promotion_speedup": rerouted_us / promoted_us,
        "per_query_router_share_percent": router_us / rerouted_us * 100,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
