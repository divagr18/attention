"""Evaluate answer quality and retrieval stability under promoted-record TTLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_tiny_transformer import (
    KEY_START,
    NOISE_START,
    QUERY_TOKEN,
    SEPARATOR_TOKEN,
    VALUE_START,
    VOCAB_SIZE,
    Config,
    TinyRetrievalTransformer,
)


def build_context(batch_size: int, context: int, device: torch.device, historical_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four separately addressable historical records per context."""
    tokens = torch.randint(NOISE_START, VOCAB_SIZE, (batch_size, context), device=device)
    keys = torch.arange(KEY_START, KEY_START + 4, device=device)
    values = torch.randint(VALUE_START, VALUE_START + 64, (batch_size, 4), device=device)
    if historical_length is None:
        historical_length = context - 32
    positions = torch.linspace(4, historical_length - 4, 4, device=device).round().long().tolist()
    for slot, position in enumerate(positions):
        tokens[:, position] = keys[slot]
        tokens[:, position + 1] = SEPARATOR_TOKEN
        tokens[:, position + 2] = values[:, slot]
    return tokens, keys, values


def query_tokens(base: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = base.clone()
    tokens[:, -3] = QUERY_TOKEN
    tokens[:, -2] = SEPARATOR_TOKEN
    tokens[:, -1] = keys[slot]
    return tokens, values[:, slot]


@torch.no_grad()
def route(model: TinyRetrievalTransformer, tokens: torch.Tensor, config: Config) -> torch.Tensor:
    embeddings = model.token_embedding(tokens)
    routing = model.router(
        embeddings,
        historical_length=config.context - config.local_window,
        block_size=config.block_size,
        top_blocks=config.top_blocks,
        top_tokens=config.top_tokens,
        retrieval_unit=model.retrieval_unit,
        retrieval_width=model.retrieval_width,
    )
    return routing.retrieved_indices


@torch.no_grad()
def answer(model: TinyRetrievalTransformer, tokens: torch.Tensor, targets: torch.Tensor, retrieved: torch.Tensor, config: Config) -> torch.Tensor:
    logits, _, _ = model(
        tokens,
        variant="learned",
        local_window=config.local_window,
        evidence_positions=None,
        block_size=config.block_size,
        top_blocks=config.top_blocks,
        top_tokens=config.top_tokens,
        retrieved_indices_override=retrieved,
    )
    prediction = logits[:, -1, VALUE_START : VALUE_START + 64].argmax(dim=-1) + VALUE_START
    return prediction.eq(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ttls", type=int, nargs="+", default=(1, 4, 16, 64))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--change-every", type=int, default=8)
    parser.add_argument("--change-steps", type=str, default="", help="Comma-separated irregular target-change steps (overrides --change-every).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--drift-threshold", type=float, default=0.05, help="Cosine-distance threshold for the embedding-drift refresh gate.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    config = Config(**checkpoint["config"])
    config.device = "cuda"
    device = torch.device("cuda")
    model = TinyRetrievalTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    change_steps = (
        {int(value) for value in args.change_steps.split(",") if value.strip()}
        if args.change_steps
        else {step for step in range(1, args.steps) if step % args.change_every == 0}
    )
    # One globally shared query schedule makes strategy comparisons identical.
    slots = []
    current = 0
    for step in range(args.steps):
        if step in change_steps:
            current = (current + int(torch.randint(1, 4, (1,), generator=generator, device=device))) % 4
        slots.append(current)
    results = []
    strategies = [(f"fixed_ttl_{ttl}", ttl, "fixed") for ttl in args.ttls]
    strategies.extend([
        ("adaptive_query_signature", None, "signature"),
        ("adaptive_embedding_drift", None, "drift"),
    ])
    for strategy, ttl, refresh_mode in strategies:
        fresh_correct = cached_correct = cache_matches = total = refreshes = 0
        for _ in range(args.episodes):
            base, keys, values = build_context(args.batch_size, config.context, device, config.context - config.local_window)
            promoted = None
            promoted_signature = None
            promoted_query_state = None
            for step, slot in enumerate(slots):
                tokens, targets = query_tokens(base, keys, values, slot)
                fresh = route(model, tokens, config)
                signature = tokens[:, -3:]
                query_state = model.token_embedding(signature).mean(dim=1)
                should_refresh = promoted is None
                if not should_refresh and refresh_mode == "signature":
                    should_refresh = should_refresh or not torch.equal(signature, promoted_signature)
                elif not should_refresh and refresh_mode == "drift":
                    cosine_distance = 1.0 - torch.nn.functional.cosine_similarity(query_state, promoted_query_state, dim=-1)
                    should_refresh = should_refresh or bool(cosine_distance.mean() > args.drift_threshold)
                elif not should_refresh:
                    should_refresh = should_refresh or step % ttl == 0
                if should_refresh:
                    promoted = fresh
                    promoted_signature = signature.clone()
                    promoted_query_state = query_state.clone()
                    refreshes += 1
                fresh_correct += int(answer(model, tokens, targets, fresh, config).sum())
                cached_correct += int(answer(model, tokens, targets, promoted, config).sum())
                cache_matches += int(promoted.eq(fresh).all(dim=1).sum())
                total += args.batch_size
        results.append({
            "strategy": strategy,
            "ttl": ttl,
            "refreshes_per_episode": refreshes / args.episodes,
            "routing_fraction": refreshes / (args.episodes * args.steps),
            "fresh_accuracy": fresh_correct / total,
            "cached_accuracy": cached_correct / total,
            "accuracy_delta": cached_correct / total - fresh_correct / total,
            "cache_matches_fresh_router": cache_matches / total,
        })
    report = {
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "query_change_interval": args.change_every,
        "query_change_steps": sorted(change_steps),
        "embedding_drift_threshold": args.drift_threshold,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
