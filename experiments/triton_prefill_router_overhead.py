"""Measure periodic causal block-routing overhead during fused local prefill."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from triton_local_prefill import local_causal_prefill


def periodic_router(query: torch.Tensor, keys: torch.Tensor, *, block_size: int, router_stride: int, top_blocks: int) -> torch.Tensor:
    """Vectorized causal schedule: route only at every router_stride tokens."""
    batch, heads, context, dim = query.shape
    summaries = keys.view(batch, heads, context // block_size, block_size, dim).mean(dim=3)
    update_positions = torch.arange(router_stride - 1, context, router_stride, device=query.device)
    update_queries = query[:, :, update_positions]
    scores = torch.einsum("bhud,bhgd->bhug", update_queries, summaries)
    block_end_positions = torch.arange(block_size - 1, context, block_size, device=query.device)
    causal = block_end_positions.unsqueeze(0) <= update_positions.unsqueeze(1)
    scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    # Early updates have fewer causal blocks than the requested budget.  Pad
    # unused slots with -1 instead of returning masked future block IDs.
    selected = torch.full((*scores.shape[:-1], top_blocks), -1, dtype=torch.long, device=query.device)
    for update in range(update_positions.numel()):
        available = int(causal[update].sum())
        count = min(top_blocks, available)
        if count:
            selected[:, :, update, :count] = scores[:, :, update].topk(count, dim=-1).indices
    return selected


def timed(operation, iterations: int) -> float:
    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.mean(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--router-stride", type=int, default=256)
    parser.add_argument("--top-blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.context % args.block_size or args.context % args.router_stride:
        parser.error("context must divide evenly into block-size and router-stride")
    torch.manual_seed(7)
    shape = (args.batch_size, args.heads, args.context, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.float16)
    keys = torch.randn_like(query)
    values = torch.randn_like(query)
    local_ms = timed(lambda: local_causal_prefill(query, keys, values, args.window), args.iterations)
    router_ms = timed(lambda: periodic_router(query, keys, block_size=args.block_size, router_stride=args.router_stride, top_blocks=args.top_blocks), args.iterations)
    combined_ms = timed(lambda: (local_causal_prefill(query, keys, values, args.window), periodic_router(query, keys, block_size=args.block_size, router_stride=args.router_stride, top_blocks=args.top_blocks)), args.iterations)
    report = {
        "context": args.context,
        "window": args.window,
        "block_size": args.block_size,
        "router_stride": args.router_stride,
        "top_blocks": args.top_blocks,
        "updates_per_prefill": args.context // args.router_stride,
        "local_prefill_mean_ms": local_ms,
        "periodic_router_mean_ms": router_ms,
        "combined_mean_ms": combined_ms,
        "combined_overhead_percent": (combined_ms / local_ms - 1) * 100,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
