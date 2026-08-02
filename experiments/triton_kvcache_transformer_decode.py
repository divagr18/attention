"""One- and two-layer Transformer decode using prefetched KV caches and Triton.

The model's K/V tensors are computed once for a fixed context. For each
decoder query only its three query tokens update the cache tail; fused Triton
attention reads the local tail plus the promoted page or fine span. Two-layer
models additionally prefill the second-layer cache from the first layer's
local-only historical states.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evaluate_promotion_ttl import build_context, query_tokens
from train_tiny_transformer import Config, TinyRetrievalTransformer, VALUE_START
from triton_paged_attention import paged_gather_decode_attention, span_gather_decode_attention


def load_model(path: Path) -> tuple[TinyRetrievalTransformer, Config]:
    checkpoint = torch.load(path, map_location="cuda", weights_only=True)
    config = Config(**checkpoint["config"])
    config.device = "cuda"
    model = TinyRetrievalTransformer(config).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])
    return model, config


@torch.no_grad()
def qkv(
    model: TinyRetrievalTransformer,
    tokens: torch.Tensor | None = None,
    position_offset: int = 0,
    *,
    layer_index: int = 0,
    hidden: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a layer's Q/K/V projection and its input hidden states."""
    if hidden is None:
        if tokens is None:
            raise ValueError("tokens are required when hidden states are not supplied")
        positions = torch.arange(position_offset, position_offset + tokens.size(1), device=tokens.device)
        hidden = model.token_embedding(tokens) + model.position_embedding(positions)
    block = model.blocks[layer_index]
    normalized = block.norm1(hidden)
    packed = block.attention.qkv(normalized)
    batch, length, _ = packed.shape
    heads, dim = block.attention.heads, block.attention.head_dim
    packed = packed.view(batch, length, 3, heads, dim).permute(2, 0, 3, 1, 4)
    return packed[0], packed[1], packed[2], hidden


def selected_attention(
    config: Config,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    if config.retrieval_unit == "page_fine":
        return span_gather_decode_attention(
            query.contiguous(), key_cache.contiguous(), value_cache.contiguous(), candidates.contiguous(),
            local_window=config.local_window, span_width=config.retrieval_width,
        ).float()
    return paged_gather_decode_attention(
        query.contiguous(), key_cache.contiguous(), value_cache.contiguous(), candidates.contiguous(),
        local_window=config.local_window, block_size=config.block_size,
    ).float()


def local_tail_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    tail_start: int,
    local_window: int,
) -> torch.Tensor:
    """Exact local attention for the non-final query tokens in a short tail."""
    outputs = []
    for offset in range(query.size(2)):
        position = tail_start + offset
        start = max(0, position - local_window + 1)
        keys = key_cache[:, :, start : position + 1]
        values = value_cache[:, :, start : position + 1]
        scores = (query[:, :, offset : offset + 1] @ keys.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        outputs.append((scores.softmax(dim=-1) @ values).squeeze(2))
    return torch.stack(outputs, dim=2)


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


@torch.no_grad()
def fine_decode(model: TinyRetrievalTransformer, config: Config, tokens: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    tail_q, tail_k, tail_v, tail_x = qkv(model, tokens[:, -3:], config.context - 3)
    key_cache = key_cache.clone()
    value_cache = value_cache.clone()
    key_cache[:, :, -3:] = tail_k
    value_cache[:, :, -3:] = tail_v
    attended = span_gather_decode_attention(
        tail_q[:, :, -1].contiguous(), key_cache.contiguous(), value_cache.contiguous(), centers.contiguous(),
        local_window=config.local_window, span_width=config.retrieval_width,
    )
    attention = model.blocks[0].attention.output(attended.reshape(tokens.size(0), -1))
    x = tail_x[:, -1] + attention
    x = x + model.blocks[0].mlp(model.blocks[0].norm2(x))
    logits = model.output(model.norm(x))
    return logits[:, VALUE_START : VALUE_START + 64].argmax(dim=-1) + VALUE_START


@torch.no_grad()
def prefill_two_layer_cache(
    model: TinyRetrievalTransformer,
    config: Config,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build reusable caches for both layers of a fixed context.

    Sparse retrieval only affects the final query position.  The final three
    positions are replaced on every decode step, so a local-only prefill gives
    exact first-layer states for every reusable historical position.
    """
    _, key_one, value_one, embeddings = qkv(model, tokens, layer_index=0)
    layer_one_hidden, _ = model.blocks[0](
        embeddings,
        variant="sliding",
        local_window=config.local_window,
        evidence_positions=None,
        retrieved_indices=None,
    )
    _, key_two, value_two, _ = qkv(model, layer_index=1, hidden=layer_one_hidden)
    return key_one, value_one, key_two, value_two


@torch.no_grad()
def two_layer_decode(
    model: TinyRetrievalTransformer,
    config: Config,
    tokens: torch.Tensor,
    key_one_cache: torch.Tensor,
    value_one_cache: torch.Tensor,
    key_two_cache: torch.Tensor,
    value_two_cache: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Decode a query tail through two sparse-attention layers.

    The first two tail tokens only require local attention.  The final token
    uses the routed page/span candidates in both layers, matching the model's
    learned-attention mask exactly.
    """
    tail_start = config.context - 3
    query_one, tail_key_one, tail_value_one, tail_embeddings = qkv(
        model, tokens[:, -3:], tail_start, layer_index=0
    )
    key_one = key_one_cache.clone()
    value_one = value_one_cache.clone()
    key_one[:, :, -3:] = tail_key_one
    value_one[:, :, -3:] = tail_value_one

    local_one = local_tail_attention(
        query_one[:, :, :2], key_one, value_one,
        tail_start=tail_start, local_window=config.local_window,
    )
    routed_one = selected_attention(config, query_one[:, :, -1], key_one, value_one, candidates)
    attended_one = torch.cat((local_one, routed_one.unsqueeze(2)), dim=2)
    block_one = model.blocks[0]
    tail_hidden_one = tail_embeddings + block_one.attention.output(
        attended_one.transpose(1, 2).contiguous().view(tokens.size(0), 3, -1)
    )
    tail_hidden_one = tail_hidden_one + block_one.mlp(block_one.norm2(tail_hidden_one))

    query_two, tail_key_two, tail_value_two, _ = qkv(model, layer_index=1, hidden=tail_hidden_one)
    key_two = key_two_cache.clone()
    value_two = value_two_cache.clone()
    key_two[:, :, -3:] = tail_key_two
    value_two[:, :, -3:] = tail_value_two
    routed_two = selected_attention(config, query_two[:, :, -1], key_two, value_two, candidates)
    block_two = model.blocks[1]
    final_hidden = tail_hidden_one[:, -1] + block_two.attention.output(
        routed_two.reshape(tokens.size(0), -1)
    )
    final_hidden = final_hidden + block_two.mlp(block_two.norm2(final_hidden))
    logits = model.output(model.norm(final_hidden))
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
    if len(model.blocks) not in (1, 2):
        raise ValueError("the KV-cache integration supports one or two Transformer layers")
    base, keys, values = build_context(1, config.context, torch.device("cuda"), config.context - config.local_window)
    if len(model.blocks) == 1:
        prefill_ms = elapsed(lambda: qkv(model, base), 20)
        _, key_cache, value_cache, _ = qkv(model, base)
        layer_caches: tuple[torch.Tensor, ...] = (key_cache, value_cache)
    else:
        prefill_ms = elapsed(lambda: prefill_two_layer_cache(model, config, base), 20)
        layer_caches = prefill_two_layer_cache(model, config, base)
    slots, current = [], 0
    for step in range(args.steps):
        if step in {3, 7, 12}:
            current = (current + 1) % 4
        slots.append(current)
    # Compile and warm the fused kernel before recording decode timings.
    warm_tokens, _ = query_tokens(base, keys, values, slots[0])
    _, warm_routing, _ = model(warm_tokens, variant="learned", local_window=config.local_window, evidence_positions=None, block_size=config.block_size, top_blocks=config.top_blocks, top_tokens=config.top_tokens)
    warm_blocks = warm_routing.block_scores.topk(config.top_blocks, dim=-1).indices
    warm_centers = warm_routing.retrieved_indices[:, ::config.retrieval_width]
    for _ in range(10):
        candidates = warm_centers if config.retrieval_unit == "page_fine" else warm_blocks
        if len(model.blocks) == 2:
            two_layer_decode(model, config, warm_tokens, *layer_caches, candidates)
        elif config.retrieval_unit == "page_fine":
            fine_decode(model, config, warm_tokens, *layer_caches, warm_centers)
        else:
            page_decode(model, config, warm_tokens, *layer_caches, warm_blocks)
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
            centers = routing.retrieved_indices[:, ::config.retrieval_width]
            start = time.perf_counter()
            candidates = centers if config.retrieval_unit == "page_fine" else selected
            if len(model.blocks) == 2:
                page_prediction = two_layer_decode(model, config, tokens, *layer_caches, candidates)
            else:
                decode = fine_decode if config.retrieval_unit == "page_fine" else page_decode
                page_prediction = decode(model, config, tokens, *layer_caches, candidates)
            torch.cuda.synchronize()
            page_times.append((time.perf_counter() - start) * 1000)
            page_correct += int(page_prediction.eq(targets).sum())
            reroute_calls += 1
            state = model.router.query(model.token_embedding(tokens[:, -3:]).mean(dim=1))
            if promoted is None or bool(1.0 - torch.nn.functional.cosine_similarity(state, promoted_state).mean() > 0.05):
                promoted = candidates
                promoted_state = state
                adaptive_calls += 1
            start = time.perf_counter()
            if len(model.blocks) == 2:
                two_layer_decode(model, config, tokens, *layer_caches, promoted)
            else:
                decode(model, config, tokens, *layer_caches, promoted)
            torch.cuda.synchronize()
            adaptive_times.append((time.perf_counter() - start) * 1000)
            total += 1
    report = {
        "context": config.context,
        "layers": len(model.blocks),
        "retrieval_unit": config.retrieval_unit,
        "top_tokens": config.top_tokens,
        "retrieval_width": config.retrieval_width,
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
