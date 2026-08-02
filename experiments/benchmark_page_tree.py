#!/usr/bin/env python3
"""Router-inclusive dense/local/flat/tree decode benchmark.

Tree build is reported separately from search.  In a causal prefill runtime
the build/update work happens once as pages leave the hot window; decode pays
only the fixed-budget search, gather, and unified exact attention below.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from hierarchical_page_tree import HierarchicalPageTree
from multi_vector_page_tree import MultiVectorPageTree


@dataclass
class Config:
    context: int
    hot_window: int
    page_size: int
    tree_fanout: int
    tree_beam: int
    tree_slots: int
    tree_leaf_slots: int
    retrieval_pages: int
    batch_size: int
    heads: int
    head_dim: int
    iterations: int
    warmup: int
    device: str
    seed: int


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(operation, config: Config, device: torch.device) -> dict[str, float]:
    for _ in range(config.warmup):
        operation()
    synchronize(device)
    samples: list[float] = []
    for _ in range(config.iterations):
        start = time.perf_counter()
        operation()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return {"mean_ms": statistics.mean(samples), "p50_ms": statistics.median(samples)}


def exact_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bhd,bhld->bhl", query, keys) / query.size(-1) ** 0.5
    return torch.einsum("bhl,bhld->bhd", scores.softmax(dim=-1), values)


def gather_pages(keys: torch.Tensor, values: torch.Tensor, pages: torch.Tensor, hot_window: int, page_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    historical = keys.size(2) - hot_window
    page_count = historical // page_size
    key_pages = keys[:, :, :historical].view(keys.size(0), keys.size(1), page_count, page_size, keys.size(-1)).permute(0, 2, 1, 3, 4)
    value_pages = values[:, :, :historical].view(values.size(0), values.size(1), page_count, page_size, values.size(-1)).permute(0, 2, 1, 3, 4)
    batch = torch.arange(keys.size(0), device=keys.device).unsqueeze(1)
    selected_keys = key_pages[batch, pages].permute(0, 2, 1, 3, 4).flatten(start_dim=2, end_dim=3)
    selected_values = value_pages[batch, pages].permute(0, 2, 1, 3, 4).flatten(start_dim=2, end_dim=3)
    return torch.cat((keys[:, :, historical:], selected_keys), dim=2), torch.cat((values[:, :, historical:], selected_values), dim=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--hot-window", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--tree-fanout", type=int, default=16)
    parser.add_argument("--tree-beam", type=int, default=4)
    parser.add_argument("--tree-slots", type=int, default=4)
    parser.add_argument("--tree-leaf-slots", type=int, default=1)
    parser.add_argument("--retrieval-pages", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    if config.retrieval_pages > config.tree_beam:
        parser.error("--retrieval-pages must be no greater than --tree-beam")
    historical = config.context - config.hot_window
    if historical <= 0 or historical % config.page_size:
        parser.error("context - hot-window must be a positive multiple of page-size")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    query = torch.randn(config.batch_size, config.heads, config.head_dim, device=device)
    keys = torch.randn(config.batch_size, config.heads, config.context, config.head_dim, device=device)
    values = torch.randn_like(keys)
    page_count = historical // config.page_size
    router_query = query.mean(dim=1)
    page_keys = keys[:, :, :historical].view(config.batch_size, config.heads, page_count, config.page_size, config.head_dim).mean(dim=(1, 3))
    build_start = time.perf_counter()
    page_slots = page_keys.new_zeros(config.batch_size, page_count, config.tree_leaf_slots, config.head_dim)
    page_slots[:, :, 0] = page_keys
    slot_valid = torch.zeros(config.batch_size, page_count, config.tree_leaf_slots, dtype=torch.bool, device=device)
    slot_valid[:, :, 0] = True
    tree = MultiVectorPageTree.from_page_slots(page_slots, fanout=config.tree_fanout, slots=config.tree_slots, slot_valid=slot_valid)
    synchronize(device)
    tree_build_ms = (time.perf_counter() - build_start) * 1e3
    causal_build_start = time.perf_counter()
    causal_tree = MultiVectorPageTree(fanout=config.tree_fanout, slots=config.tree_slots, max_pages=page_count)
    for page in range(page_count):
        causal_tree.append(page_slots[:, page], slot_valid[:, page])
    synchronize(device)
    causal_build_ms = (time.perf_counter() - causal_build_start) * 1e3

    def local() -> torch.Tensor:
        return exact_attention(query, keys[:, :, historical:], values[:, :, historical:])

    def dense() -> torch.Tensor:
        return exact_attention(query, keys, values)

    def flat() -> torch.Tensor:
        scores = torch.einsum("bd,bpd->bp", router_query, page_keys)
        pages = scores.topk(config.retrieval_pages, dim=-1).indices
        selected_keys, selected_values = gather_pages(keys, values, pages, config.hot_window, config.page_size)
        return exact_attention(query, selected_keys, selected_values)

    def tree_path() -> torch.Tensor:
        pages = tree.search(router_query, beam=config.tree_beam, retrieval_pages=config.retrieval_pages).page_indices
        selected_keys, selected_values = gather_pages(keys, values, pages, config.hot_window, config.page_size)
        return exact_attention(query, selected_keys, selected_values)

    search = lambda: tree.search(router_query, beam=config.tree_beam, retrieval_pages=config.retrieval_pages)
    report = {
        "config": asdict(config),
        "timings": {
            "dense_end_to_end": timed(dense, config, device),
            "local_end_to_end": timed(local, config, device),
            "flat_router_gather_attention": timed(flat, config, device),
            "tree_search_only": timed(search, config, device),
            "tree_router_gather_attention": timed(tree_path, config, device),
            "tree_initial_build_ms": tree_build_ms,
            "tree_causal_append_build_ms": causal_build_ms,
        },
        "candidate_tokens": {
            "local": config.hot_window,
            "flat_and_tree": config.hot_window + config.retrieval_pages * config.page_size,
        },
        "router_scores_per_query": {
            "flat_pages": page_count,
            "tree_max": tree.search(router_query, beam=config.tree_beam, retrieval_pages=config.retrieval_pages).score_count,
        },
        "tree_depth": tree.depth,
        "historical_store": "bf16",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
