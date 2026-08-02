"""Fused local-causal attention kernel and prefill benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def local_causal_prefill_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    scale: tl.constexpr,
    context: tl.constexpr,
    window: tl.constexpr,
    head_dim: tl.constexpr,
    block_candidates: tl.constexpr,
    block_dim: tl.constexpr,
):
    program = tl.program_id(axis=0)
    sequence_index = program % context
    batch_head = program // context
    dim_offsets = tl.arange(0, block_dim)
    query_base = (batch_head * context + sequence_index) * head_dim
    query = tl.load(query_ptr + query_base + dim_offsets, mask=dim_offsets < head_dim, other=0.0).to(tl.float32)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_dim,), tl.float32)
    for start in range(0, window, block_candidates):
        candidate_offsets = start + tl.arange(0, block_candidates)
        key_indices = sequence_index - window + 1 + candidate_offsets
        valid = (key_indices >= 0) & (key_indices <= sequence_index)
        mask = valid[:, None] & (dim_offsets[None, :] < head_dim)
        base_offsets = (batch_head * context + key_indices[:, None]) * head_dim + dim_offsets[None, :]
        keys = tl.load(key_ptr + base_offsets, mask=mask, other=0.0).to(tl.float32)
        values = tl.load(value_ptr + base_offsets, mask=mask, other=0.0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        # An early query can have a whole candidate chunk before position zero.
        # Keep that chunk numerically finite, then explicitly zero its mass.
        scores = tl.where(valid, scores, -1.0e9)
        block_maximum = tl.max(scores, axis=0)
        new_maximum = tl.maximum(running_max, block_maximum)
        probabilities = tl.where(valid, tl.exp(scores - new_maximum), 0.0)
        rescale = tl.exp(running_max - new_maximum)
        accumulator = accumulator * rescale + tl.sum(values * probabilities[:, None], axis=0)
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=0)
        running_max = new_maximum
    tl.store(output_ptr + query_base + dim_offsets, accumulator / running_sum, mask=dim_offsets < head_dim)


def local_causal_prefill(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, window: int) -> torch.Tensor:
    batch, heads, context, head_dim = query.shape
    if not (query.is_cuda and query.is_contiguous() and keys.is_contiguous() and values.is_contiguous()):
        raise ValueError("contiguous CUDA tensors are required")
    if keys.shape != query.shape or values.shape != query.shape or head_dim > 256:
        raise ValueError("unsupported Q/K/V shape")
    output = torch.empty_like(query)
    # Smaller candidate blocks keep register pressure manageable at large head_dim.
    block_candidates = 128 if head_dim <= 64 else 32
    local_causal_prefill_kernel[(batch * heads * context,)](
        query,
        keys,
        values,
        output,
        scale=head_dim**-0.5,
        context=context,
        window=window,
        head_dim=head_dim,
        block_candidates=block_candidates,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


def local_reference(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, window: int) -> torch.Tensor:
    context = query.size(2)
    positions = torch.arange(context, device=query.device)
    allowed = (positions[:, None] >= positions[None, :]) & (positions[:, None] - positions[None, :] < window)
    scores = torch.einsum("bhid,bhjd->bhij", query, keys) * query.size(-1) ** -0.5
    return torch.einsum("bhij,bhjd->bhid", scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1), values)


def benchmark(operation, iterations: int) -> float:
    for _ in range(20):
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
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.window % 128:
        parser.error("window must be divisible by 128")
    torch.manual_seed(7)
    shape = (args.batch_size, args.heads, args.context, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.float16)
    keys = torch.randn_like(query)
    values = torch.randn_like(query)
    # Small explicit-mask reference validates causal-window semantics without
    # allocating an impractical dense score matrix at the benchmark context.
    small = min(args.context, 256)
    torch.testing.assert_close(
        local_causal_prefill(query[:, :, :small].contiguous(), keys[:, :, :small].contiguous(), values[:, :, :small].contiguous(), min(args.window, small)),
        local_reference(query[:, :, :small], keys[:, :, :small], values[:, :, :small], min(args.window, small)),
        rtol=2e-2,
        atol=2e-2,
    )
    local_ms = benchmark(lambda: local_causal_prefill(query, keys, values, args.window), args.iterations)
    dense_ms = benchmark(lambda: F.scaled_dot_product_attention(query, keys, values, is_causal=True), args.iterations)
    report = {
        "context": args.context,
        "window": args.window,
        "batch_size": args.batch_size,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dense_prefill_mean_ms": dense_ms,
        "local_triton_prefill_mean_ms": local_ms,
        "speedup": dense_ms / local_ms,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
