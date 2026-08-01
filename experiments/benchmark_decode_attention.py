#!/usr/bin/env python3
"""GPU decode microbenchmark for dense, flat, and hierarchical attention.

This benchmark models one decode query against an existing full-KV cache.  The
hierarchical path performs actual GPU block scoring, token scoring only inside
selected blocks, contiguous-record promotion, KV gathering, and one exact
softmax over local plus promoted candidates.  It therefore measures decode
compute, not prefill kernels or compressed-cache memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    context: int
    local_window: int
    block_size: int
    top_blocks: int
    top_tokens: int
    batch_size: int
    heads: int
    head_dim: int
    iterations: int
    warmup: int
    seed: int
    device: str


def exact_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bhd,bhld->bhl", query, keys) / query.size(-1) ** 0.5
    weights = scores.softmax(dim=-1)
    return torch.einsum("bhl,bhld->bhd", weights, values)


def candidate_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = indices[:, None, :, None].expand(-1, keys.size(1), -1, keys.size(-1))
    selected_keys = keys.gather(2, expanded)
    selected_values = values.gather(2, expanded)
    return exact_attention(query, selected_keys, selected_values)


def flat_retrieval(query: torch.Tensor, router_keys: torch.Tensor, config: Config) -> torch.Tensor:
    scores = torch.einsum("bhd,bhld->bhl", query, router_keys)
    centers = scores.mean(dim=1).topk(config.top_tokens, dim=-1).indices
    return build_candidate_indices(centers, config, query.device)


def hierarchical_retrieval(query: torch.Tensor, router_keys: torch.Tensor, block_summaries: torch.Tensor, config: Config) -> torch.Tensor:
    block_scores = torch.einsum("bhd,bhgd->bhg", query, block_summaries).mean(dim=1)
    selected_blocks = block_scores.topk(config.top_blocks, dim=-1).indices
    offsets = torch.arange(config.block_size, device=query.device)
    token_indices = (selected_blocks.unsqueeze(-1) * config.block_size + offsets).flatten(start_dim=1)
    expanded = token_indices[:, None, :, None].expand(-1, router_keys.size(1), -1, router_keys.size(-1))
    selected_router_keys = router_keys.gather(2, expanded)
    token_scores = torch.einsum("bhd,bhld->bhl", query, selected_router_keys).mean(dim=1)
    centers = token_indices.gather(1, token_scores.topk(config.top_tokens, dim=-1).indices)
    return build_candidate_indices(centers, config, query.device)


def paged_block_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, block_summaries: torch.Tensor, config: Config) -> torch.Tensor:
    """Retrieve whole contiguous blocks, avoiding fine-grained KV gathering."""
    block_scores = torch.einsum("bhd,bhgd->bhg", query, block_summaries).mean(dim=1)
    selected_blocks = block_scores.topk(config.top_blocks, dim=-1).indices
    historical_length = config.context - config.local_window
    pages_shape = (config.batch_size, config.heads, historical_length // config.block_size, config.block_size, config.head_dim)
    key_pages = keys[:, :, :historical_length].view(pages_shape).permute(0, 2, 1, 3, 4)
    value_pages = values[:, :, :historical_length].view(pages_shape).permute(0, 2, 1, 3, 4)
    batch_indices = torch.arange(config.batch_size, device=query.device).unsqueeze(1)
    selected_keys = key_pages[batch_indices, selected_blocks].permute(0, 2, 1, 3, 4).flatten(start_dim=2, end_dim=3)
    selected_values = value_pages[batch_indices, selected_blocks].permute(0, 2, 1, 3, 4).flatten(start_dim=2, end_dim=3)
    local_keys = keys[:, :, historical_length:]
    local_values = values[:, :, historical_length:]
    return exact_attention(query, torch.cat((local_keys, selected_keys), dim=2), torch.cat((local_values, selected_values), dim=2))


def build_candidate_indices(centers: torch.Tensor, config: Config, device: torch.device) -> torch.Tensor:
    record_offsets = torch.arange(3, device=device)
    promoted = (centers.unsqueeze(-1) + record_offsets).clamp_max(config.context - config.local_window - 1).flatten(start_dim=1)
    local = torch.arange(config.context - config.local_window, config.context, device=device).expand(centers.size(0), -1)
    return torch.cat((local, promoted), dim=1)


def timed(operation, config: Config) -> dict[str, float]:
    for _ in range(config.warmup):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(config.iterations):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return {"mean_us": statistics.mean(samples), "p50_us": statistics.median(samples), "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--top-blocks", type=int, default=4)
    parser.add_argument("--top-tokens", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    if config.device != "cuda" or not torch.cuda.is_available():
        parser.error("this benchmark requires CUDA")
    if (config.context - config.local_window) % config.block_size:
        parser.error("historical context must divide evenly into block-size")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    shape = (config.batch_size, config.heads, config.context, config.head_dim)
    query = torch.randn(config.batch_size, config.heads, config.head_dim, device=device, dtype=torch.float16)
    keys = torch.randn(shape, device=device, dtype=torch.float16)
    values = torch.randn(shape, device=device, dtype=torch.float16)
    historical_length = config.context - config.local_window
    router_keys = keys[:, :, :historical_length]
    block_summaries = router_keys.view(config.batch_size, config.heads, historical_length // config.block_size, config.block_size, config.head_dim).mean(dim=3).contiguous()
    dense = lambda: exact_attention(query, keys, values)
    flat = lambda: candidate_attention(query, keys, values, flat_retrieval(query, router_keys, config))
    hierarchical = lambda: candidate_attention(query, keys, values, hierarchical_retrieval(query, router_keys, block_summaries, config))
    paged = lambda: paged_block_attention(query, keys, values, block_summaries, config)
    torch.cuda.reset_peak_memory_stats(device)
    results = {"dense": timed(dense, config), "flat_retrieval": timed(flat, config), "hierarchical_retrieval": timed(hierarchical, config), "hierarchical_paged_blocks": timed(paged, config)}
    element_bytes = torch.tensor([], dtype=torch.float16).element_size()
    candidate_count = config.local_window + 3 * config.top_tokens
    report = {
        "config": asdict(config),
        "results": results,
        "dense_to_hierarchical_speedup": {
            "fine_token_path": results["dense"]["mean_us"] / results["hierarchical_retrieval"]["mean_us"],
            "paged_block_path": results["dense"]["mean_us"] / results["hierarchical_paged_blocks"]["mean_us"],
        },
        "kv_cache_mib": 2 * config.batch_size * config.heads * config.context * config.head_dim * element_bytes / 2**20,
        "attention_score_workspace_mib": {
            "dense": config.batch_size * config.heads * config.context * element_bytes / 2**20,
            "candidate": config.batch_size * config.heads * candidate_count * element_bytes / 2**20,
        },
        "routing_scores_per_query": {"flat": historical_length, "hierarchical": historical_length // config.block_size + config.top_blocks * config.block_size},
        "candidate_tokens": {"fine_token_path": candidate_count, "paged_block_path": config.local_window + config.top_blocks * config.block_size},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
