"""4K one-layer Transformer decode using a prefetched KV cache and Triton pages.

The model's first-layer K/V tensors are computed once for a fixed context.
For each decoder query only its three query tokens update the tail of that
cache; fused Triton attention reads the 64-token local tail plus one promoted
64-token historical page.  This is a real model-layer integration, although
the toy router was trained to promote a three-token record rather than a full
page, so full-page quality is reported explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evaluate_promotion_ttl import build_context, query_tokens
from train_tiny_transformer import Config, TinyRetrievalTransformer, VALUE_START
from triton_paged_attention import paged_gather_decode_attention


def load_model(path: Path) -> tuple[TinyRetrievalTransformer, Config]:
    checkpoint = torch.load(path, map_location="cuda", weights_only=True)
    config = Config(**checkpoint["config"])
    config.device = "cuda"
    model = TinyRetrievalTransformer(config).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])
    return model, config


@torch.no_grad()
def qkv(model: TinyRetrievalTransformer, tokens: torch.Tensor, position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.arange(position_offset, position_offset + tokens.size(1), device=tokens.device)
    x = model.token_embedding(tokens) + model.position_embedding(positions)
    normalized = model.blocks[0].norm1(x)
    packed = model.blocks[0].attention.qkv(normalized)
    batch, length, _ = packed.shape
    heads, dim = model.blocks[0].attention.heads, model.blocks[0].attention.head_dim
    packed = packed.view(batch, length, 3, heads, dim).permute(2, 0, 3, 1, 4)
    return packed[0], packed[1], packed[2], x


@torch.no_grad()
def page_decode(model: TinyRetrievalTransformer, config: Config, tokens: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, selected_blocks: torch.Tensor) -> torch.Tensor:
    # Query-dependent tail K/V replaces the cached query region.  The earlier
    # 4,093 tokens remain a real reusable K/V cache across decode steps.
    tail_q, tail_k, tail_v, tail_x = qkv(model, tokens[:, -3:], config.context - 3)
    key_cache = key_cache.clone()
    value_cache = value_cache.clone()
    key_cache[:, :, -3:] = tail_k
    value_cache[:, :, -3:] = tail_v
    query = tail_q[:, :, -1].contiguous()
    attended = paged_gather_decode_attention(
        query, key_cache.contiguous(), value_cache.contiguous(), selected_blocks.contiguous(),
        local_window=config.local_window, block_size=config.block_size,
    ).float()
    # `attended` is [batch, heads, head_dim]; flatten in the same head-major
    # order as CausalAttention's [batch, length=1, channels] output.
    attention = model.blocks[0].attention.output(attended.reshape(tokens.size(0), -1))
    x = tail_x[:, -1] + attention
    x = x + model.blocks[0].mlp(model.blocks[0].norm2(x))
    logits = model.output(model.norm(x))
    return logits[:, VALUE_START : VALUE_START + 64].argmax(dim=-1) + VALUE_START


def elapsed(operation, iterations: int) -> float:
    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, config = load_model(args.checkpoint)
    if len(model.blocks) != 1:
        raise ValueError("this first integration supports the one-layer 4K checkpoint")
    base, keys, values = build_context(1, config.context, torch.device("cuda"), config.context - config.local_window)
    prefill_ms = elapsed(lambda: qkv(model, base), 20)
    _, key_cache, value_cache, _ = qkv(model, base)
    slots, current = [], 0
    for step in range(args.steps):
        if step in {3, 7, 12}:
            current = (current + 1) % 4
        slots.append(current)
    # Compile and warm the fused kernel before recording decode timings.
    warm_tokens, _ = query_tokens(base, keys, values, slots[0])
    _, warm_routing, _ = model(warm_tokens, variant="learned", local_window=config.local_window, evidence_positions=None, block_size=config.block_size, top_blocks=config.top_blocks, top_tokens=config.top_tokens)
    warm_blocks = warm_routing.block_scores.topk(config.top_blocks, dim=-1).indices
    for _ in range(10):
        page_decode(model, config, warm_tokens, key_cache, value_cache, warm_blocks)
    torch.cuda.synchronize()
    exact_correct = page_correct = total = reroute_calls = adaptive_calls = 0
    page_times, reroute_times, adaptive_times = [], [], []
    for _ in range(args.episodes):
        promoted = promoted_state = None
        for slot in slots:
            tokens, targets = query_tokens(base, keys, values, slot)
            start = time.perf_counter()
            logits, routing, _ = model(tokens, variant="learned", local_window=config.local_window, evidence_positions=None, block_size=config.block_size, top_blocks=config.top_blocks, top_tokens=config.top_tokens)
            torch.cuda.synchronize()
            reroute_times.append((time.perf_counter() - start) * 1000)
            exact_prediction = logits[:, -1, VALUE_START : VALUE_START + 64].argmax(dim=-1) + VALUE_START
            exact_correct += int(exact_prediction.eq(targets).sum())
            selected = routing.block_scores.topk(config.top_blocks, dim=-1).indices
            start = time.perf_counter()
            page_prediction = page_decode(model, config, tokens, key_cache, value_cache, selected)
            torch.cuda.synchronize()
            page_times.append((time.perf_counter() - start) * 1000)
            page_correct += int(page_prediction.eq(targets).sum())
            reroute_calls += 1
            state = model.router.query(model.token_embedding(tokens[:, -3:]).mean(dim=1))
            if promoted is None or bool(1.0 - torch.nn.functional.cosine_similarity(state, promoted_state).mean() > 0.05):
                promoted = selected
                promoted_state = state
                adaptive_calls += 1
            start = time.perf_counter()
            page_decode(model, config, tokens, key_cache, value_cache, promoted)
            torch.cuda.synchronize()
            adaptive_times.append((time.perf_counter() - start) * 1000)
            total += 1
    report = {
        "context": config.context,
        "prefill_kv_cache_ms": prefill_ms,
        "steps_per_episode": args.steps,
        "exact_router_model_accuracy": exact_correct / total,
        "triton_full_page_model_accuracy": page_correct / total,
        "triton_page_decode_ms": sum(page_times) / len(page_times),
        "full_pytorch_reroute_ms": sum(reroute_times) / len(reroute_times),
        "triton_adaptive_page_decode_ms": sum(adaptive_times) / len(adaptive_times),
        "reroute_calls_per_episode": reroute_calls / args.episodes,
        "adaptive_router_calls_per_episode": adaptive_calls / args.episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
