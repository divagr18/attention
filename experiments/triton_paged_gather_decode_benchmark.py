"""Benchmark fused selected-page gather plus exact attention at 32K decode."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from triton_paged_attention import paged_gather_decode_attention, reference_attention


def select_blocks(query, summaries, top_blocks):
    return torch.einsum("bhd,bhgd->bhg", query, summaries).mean(dim=1).topk(top_blocks, dim=-1).indices.contiguous()


def make_candidates(keys, values, selected, local_window, block_size):
    batch, heads, context, dim = keys.shape
    historical = context - local_window
    pages = keys[:, :, :historical].view(batch, heads, historical // block_size, block_size, dim).permute(0, 2, 1, 3, 4)
    value_pages = values[:, :, :historical].view(batch, heads, historical // block_size, block_size, dim).permute(0, 2, 1, 3, 4)
    batch_indices = torch.arange(batch, device=keys.device).unsqueeze(1)
    selected_keys = pages[batch_indices, selected].permute(0, 2, 1, 3, 4).flatten(2, 3).contiguous()
    selected_values = value_pages[batch_indices, selected].permute(0, 2, 1, 3, 4).flatten(2, 3).contiguous()
    return torch.cat((keys[:, :, historical:], selected_keys), dim=2).contiguous(), torch.cat((values[:, :, historical:], selected_values), dim=2).contiguous()


def benchmark(operation, iterations):
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


def main():
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
    torch.manual_seed(7)
    query = torch.randn(args.batch_size, args.heads, args.head_dim, device="cuda", dtype=torch.float16)
    keys = torch.randn(args.batch_size, args.heads, args.context, args.head_dim, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    historical = args.context - args.local_window
    summaries = keys[:, :, :historical].view(args.batch_size, args.heads, historical // args.block_size, args.block_size, args.head_dim).mean(3).contiguous()
    selected = select_blocks(query, summaries, args.top_blocks)
    candidate_keys, candidate_values = make_candidates(keys, values, selected, args.local_window, args.block_size)
    torch.testing.assert_close(
        paged_gather_decode_attention(query, keys, values, selected, local_window=args.local_window, block_size=args.block_size),
        reference_attention(query, candidate_keys, candidate_values),
        rtol=2e-2,
        atol=2e-2,
    )
    dense_us = benchmark(lambda: reference_attention(query, keys, values), args.iterations)
    gather_us = benchmark(lambda: paged_gather_decode_attention(query, keys, values, select_blocks(query, summaries, args.top_blocks), local_window=args.local_window, block_size=args.block_size), args.iterations)
    report = {
        "context": args.context,
        "local_window": args.local_window,
        "block_size": args.block_size,
        "top_blocks": args.top_blocks,
        "batch_size": args.batch_size,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dense_torch_mean_us": dense_us,
        "fused_gather_attention_mean_us": gather_us,
        "speedup": dense_us / gather_us,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
