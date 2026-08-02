"""First fused Triton kernel for exact attention over paged candidates.

The caller supplies a contiguous candidate buffer consisting of the local
window followed by selected historical pages.  One program computes one
batch/head query: QK^T, numerically stable streaming softmax, and AV are fused
without materializing a score or probability tensor.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def paged_decode_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    scale: tl.constexpr,
    candidates: tl.constexpr,
    head_dim: tl.constexpr,
    block_candidates: tl.constexpr,
    block_dim: tl.constexpr,
):
    program = tl.program_id(axis=0)
    dim_offsets = tl.arange(0, block_dim)
    query = tl.load(query_ptr + program * head_dim + dim_offsets, mask=dim_offsets < head_dim, other=0.0).to(tl.float32)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_dim,), tl.float32)
    for start in range(0, candidates, block_candidates):
        candidate_offsets = start + tl.arange(0, block_candidates)
        mask = (candidate_offsets[:, None] < candidates) & (dim_offsets[None, :] < head_dim)
        base_offsets = (program * candidates + candidate_offsets[:, None]) * head_dim + dim_offsets[None, :]
        keys = tl.load(key_ptr + base_offsets, mask=mask, other=0.0).to(tl.float32)
        values = tl.load(value_ptr + base_offsets, mask=mask, other=0.0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(candidate_offsets < candidates, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        new_maximum = tl.maximum(running_max, block_maximum)
        probabilities = tl.exp(scores - new_maximum)
        rescale = tl.exp(running_max - new_maximum)
        accumulator = accumulator * rescale + tl.sum(values * probabilities[:, None], axis=0)
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=0)
        running_max = new_maximum
    tl.store(output_ptr + program * head_dim + dim_offsets, accumulator / running_sum, mask=dim_offsets < head_dim)


def paged_decode_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Fused attention for q=[B,H,D], K/V=[B,H,L,D] contiguous CUDA tensors."""
    if not (query.is_cuda and keys.is_cuda and values.is_cuda):
        raise ValueError("CUDA tensors are required")
    if not (query.is_contiguous() and keys.is_contiguous() and values.is_contiguous()):
        raise ValueError("contiguous candidate pages are required")
    batch, heads, head_dim = query.shape
    if keys.shape != (batch, heads, keys.size(2), head_dim) or values.shape != keys.shape:
        raise ValueError("incompatible Q/K/V shapes")
    if head_dim > 64:
        raise ValueError("first kernel supports head_dim <= 64")
    output = torch.empty_like(query)
    block_dim = triton.next_power_of_2(head_dim)
    paged_decode_attention_kernel[(batch * heads,)](
        query,
        keys,
        values,
        output,
        scale=head_dim**-0.5,
        candidates=keys.size(2),
        head_dim=head_dim,
        block_candidates=128,
        block_dim=block_dim,
        num_warps=4,
    )
    return output


@triton.jit
def paged_gather_decode_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    selected_blocks_ptr,
    output_ptr,
    scale: tl.constexpr,
    context: tl.constexpr,
    local_window: tl.constexpr,
    block_size: tl.constexpr,
    top_blocks: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_candidates: tl.constexpr,
    block_dim: tl.constexpr,
):
    """Attention directly from full KV plus selected contiguous page IDs."""
    program = tl.program_id(axis=0)
    batch = program // heads
    dim_offsets = tl.arange(0, block_dim)
    query = tl.load(query_ptr + program * head_dim + dim_offsets, mask=dim_offsets < head_dim, other=0.0).to(tl.float32)
    candidates = local_window + top_blocks * block_size
    historical_length = context - local_window
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_dim,), tl.float32)
    for start in range(0, candidates, block_candidates):
        candidate_offsets = start + tl.arange(0, block_candidates)
        local_mask = candidate_offsets < local_window
        page_indices = (candidate_offsets - local_window) // block_size
        safe_page_indices = tl.maximum(page_indices, 0)
        block_ids = tl.load(
            selected_blocks_ptr + batch * top_blocks + safe_page_indices,
            mask=(~local_mask) & (candidate_offsets < candidates),
            other=0,
        )
        historical_indices = block_ids * block_size + (candidate_offsets - local_window) % block_size
        token_indices = tl.where(local_mask, historical_length + candidate_offsets, historical_indices)
        mask = (candidate_offsets[:, None] < candidates) & (dim_offsets[None, :] < head_dim)
        base_offsets = (program * context + token_indices[:, None]) * head_dim + dim_offsets[None, :]
        keys = tl.load(key_ptr + base_offsets, mask=mask, other=0.0).to(tl.float32)
        values = tl.load(value_ptr + base_offsets, mask=mask, other=0.0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(candidate_offsets < candidates, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        new_maximum = tl.maximum(running_max, block_maximum)
        probabilities = tl.exp(scores - new_maximum)
        rescale = tl.exp(running_max - new_maximum)
        accumulator = accumulator * rescale + tl.sum(values * probabilities[:, None], axis=0)
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=0)
        running_max = new_maximum
    tl.store(output_ptr + program * head_dim + dim_offsets, accumulator / running_sum, mask=dim_offsets < head_dim)


def paged_gather_decode_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, selected_blocks: torch.Tensor, *, local_window: int, block_size: int) -> torch.Tensor:
    """Fused page gather and attention from full KV; selected_blocks is [B,m]."""
    batch, heads, head_dim = query.shape
    if selected_blocks.shape[0] != batch or not selected_blocks.is_contiguous():
        raise ValueError("selected_blocks must be contiguous [batch, top_blocks]")
    if not (query.is_contiguous() and keys.is_contiguous() and values.is_contiguous()):
        raise ValueError("contiguous CUDA tensors are required")
    if keys.shape != (batch, heads, keys.size(2), head_dim) or values.shape != keys.shape:
        raise ValueError("incompatible Q/K/V shapes")
    if head_dim > 64 or (keys.size(2) - local_window) % block_size:
        raise ValueError("unsupported head dimension or page layout")
    output = torch.empty_like(query)
    paged_gather_decode_attention_kernel[(batch * heads,)](
        query,
        keys,
        values,
        selected_blocks,
        output,
        scale=head_dim**-0.5,
        context=keys.size(2),
        local_window=local_window,
        block_size=block_size,
        top_blocks=selected_blocks.size(1),
        heads=heads,
        head_dim=head_dim,
        block_candidates=128,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


@triton.jit
def span_gather_decode_attention_kernel(
    query_ptr, key_ptr, value_ptr, centers_ptr, output_ptr,
    scale: tl.constexpr, context: tl.constexpr, local_window: tl.constexpr,
    top_tokens: tl.constexpr, span_width: tl.constexpr, heads: tl.constexpr,
    head_dim: tl.constexpr, block_candidates: tl.constexpr, block_dim: tl.constexpr,
):
    program = tl.program_id(axis=0)
    batch = program // heads
    dims = tl.arange(0, block_dim)
    query = tl.load(query_ptr + program * head_dim + dims, mask=dims < head_dim, other=0.0).to(tl.float32)
    candidates = local_window + top_tokens * span_width
    historical = context - local_window
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_dim,), tl.float32)
    for start in range(0, candidates, block_candidates):
        offsets = start + tl.arange(0, block_candidates)
        local = offsets < local_window
        span_offsets = offsets - local_window
        center_ids = span_offsets // span_width
        centers = tl.load(centers_ptr + batch * top_tokens + center_ids, mask=(~local) & (offsets < candidates), other=0)
        # Clamp span tails to the last historical token to match the training-time
        # router contract; unclamped centers near the boundary would read query-tail K/V.
        span_indices = tl.minimum(centers + span_offsets % span_width, historical - 1)
        token_indices = tl.where(local, historical + offsets, span_indices)
        mask = (offsets[:, None] < candidates) & (dims[None, :] < head_dim)
        base = (program * context + token_indices[:, None]) * head_dim + dims[None, :]
        keys = tl.load(key_ptr + base, mask=mask, other=0.0).to(tl.float32)
        values = tl.load(value_ptr + base, mask=mask, other=0.0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(offsets < candidates, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        probabilities = tl.exp(scores - new_max)
        rescale = tl.exp(running_max - new_max)
        accumulator = accumulator * rescale + tl.sum(values * probabilities[:, None], axis=0)
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=0)
        running_max = new_max
    tl.store(output_ptr + program * head_dim + dims, accumulator / running_sum, mask=dims < head_dim)


def span_gather_decode_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, centers: torch.Tensor, *, local_window: int, span_width: int) -> torch.Tensor:
    """Fused local attention plus short spans beginning at selected centers."""
    batch, heads, head_dim = query.shape
    if not (query.is_contiguous() and keys.is_contiguous() and values.is_contiguous() and centers.is_contiguous()):
        raise ValueError("contiguous CUDA tensors are required")
    if centers.shape[0] != batch or head_dim > 64:
        raise ValueError("invalid centers or head dimension")
    output = torch.empty_like(query)
    span_gather_decode_attention_kernel[(batch * heads,)](
        query, keys, values, centers, output, scale=head_dim**-0.5,
        context=keys.size(2), local_window=local_window, top_tokens=centers.size(1),
        span_width=span_width, heads=heads, head_dim=head_dim,
        block_candidates=128, block_dim=triton.next_power_of_2(head_dim), num_warps=4,
    )
    return output


def reference_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # Keep the reference in the same FP16 tensor-core regime as the decode
    # baselines.  The fused kernel still accumulates internally in FP32.
    scores = torch.einsum("bhd,bhld->bhl", query, keys) * query.size(-1) ** -0.5
    return torch.einsum("bhl,bhld->bhd", scores.softmax(dim=-1), values)


def benchmark(operation, iterations: int) -> float:
    for _ in range(25):
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=1280)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    torch.manual_seed(7)
    query = torch.randn(args.batch_size, args.heads, args.head_dim, device="cuda", dtype=torch.float16)
    keys = torch.randn(args.batch_size, args.heads, args.candidates, args.head_dim, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    fused = paged_decode_attention(query, keys, values)
    reference = reference_attention(query, keys, values)
    torch.testing.assert_close(fused, reference, rtol=2e-2, atol=2e-2)
    fused_us = benchmark(lambda: paged_decode_attention(query, keys, values), args.iterations)
    reference_us = benchmark(lambda: reference_attention(query, keys, values), args.iterations)
    print(f"correctness=pass candidates={args.candidates} head_dim={args.head_dim}")
    print(f"triton_mean_us={fused_us:.2f}")
    print(f"torch_mean_us={reference_us:.2f}")
    print(f"speedup={reference_us / fused_us:.2f}x")


if __name__ == "__main__":
    main()
