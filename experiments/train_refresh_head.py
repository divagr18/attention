"""Train and evaluate a cheap learned promotion-refresh gate.

The gate consumes only the current and cached decoder/router query states.  It
does not score historical blocks.  Its label is whether the frozen learned
router would select a different promoted record for the current query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from evaluate_promotion_ttl import build_context, query_tokens, route, answer
from train_tiny_transformer import (
    QUERY_TOKEN, SEPARATOR_TOKEN, Config, TinyRetrievalTransformer,
)


class RefreshHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width * 4, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, current: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
        features = torch.cat((current, cached, (current - cached).abs(), current * cached), dim=-1)
        return self.network(features).squeeze(-1)


def make_queries(base: torch.Tensor, keys: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    tokens = base.clone()
    tokens[:, -3] = QUERY_TOKEN
    tokens[:, -2] = SEPARATOR_TOKEN
    tokens[:, -1] = keys[slots]
    return tokens


@torch.no_grad()
def query_state(model: TinyRetrievalTransformer, tokens: torch.Tensor, noise_std: float) -> torch.Tensor:
    state = model.router.query(model.token_embedding(tokens[:, -3:]).mean(dim=1))
    if noise_std:
        state = state + torch.randn_like(state) * noise_std
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-noise-stds", type=float, nargs="+", help="Held-out state-noise levels; defaults to --noise-std.")
    parser.add_argument("--thresholds", type=float, nargs="+", help="Held-out refresh thresholds; defaults to --threshold.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = Config(**checkpoint["config"])
    config.device = "cuda"
    model = TinyRetrievalTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = RefreshHead(config.d_model).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device).manual_seed(19)

    # Train on independent contexts and both unchanged and changed queries.
    head.train()
    for step in range(args.steps):
        base, keys, _ = build_context(args.batch_size, config.context, device, config.context - config.local_window)
        current_slots = torch.randint(0, 4, (args.batch_size,), generator=generator, device=device)
        unchanged = torch.rand(args.batch_size, generator=generator, device=device) < 0.5
        shifts = torch.randint(1, 4, (args.batch_size,), generator=generator, device=device)
        cached_slots = torch.where(unchanged, current_slots, (current_slots + shifts) % 4)
        current_tokens = make_queries(base, keys, current_slots)
        cached_tokens = make_queries(base, keys, cached_slots)
        with torch.no_grad():
            current_route = route(model, current_tokens, config)
            cached_route = route(model, cached_tokens, config)
            labels = (~current_route.eq(cached_route).all(dim=1)).float()
            current_state = query_state(model, current_tokens, args.noise_std)
            cached_state = query_state(model, cached_tokens, args.noise_std)
        loss = F.binary_cross_entropy_with_logits(head(current_state, cached_state), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Test with irregular changes: eight intended refreshes in 64 steps.
    head.eval()
    def evaluate(noise_std: float, threshold: float) -> dict[str, float]:
        eval_generator = torch.Generator(device=device).manual_seed(23)
        change_steps = {5, 13, 21, 29, 37, 45, 53}
        slots, current = [], 0
        for step in range(64):
            if step in change_steps:
                current = (current + int(torch.randint(1, 4, (1,), generator=eval_generator, device=device))) % 4
            slots.append(current)
        fresh_correct = cached_correct = cache_matches = total = refreshes = 0
        gate_tp = gate_fp = gate_fn = 0
        with torch.no_grad():
            for _ in range(args.episodes):
                base, keys, values = build_context(args.batch_size, config.context, device, config.context - config.local_window)
                promoted = cached_state = cached_fresh = None
                for slot in slots:
                    tokens, targets = query_tokens(base, keys, values, slot)
                    fresh = route(model, tokens, config)
                    state = query_state(model, tokens, noise_std)
                    if promoted is None:
                        refresh = True
                        router_changed = True
                    else:
                        probability = torch.sigmoid(head(state, cached_state))
                        refresh = bool(probability.mean() >= threshold)
                        router_changed = bool((~fresh.eq(cached_fresh).all(dim=1)).any())
                    if refresh:
                        promoted, cached_state, cached_fresh = fresh, state, fresh
                        refreshes += 1
                    if refresh and router_changed:
                        gate_tp += 1
                    elif refresh:
                        gate_fp += 1
                    elif router_changed:
                        gate_fn += 1
                    fresh_correct += int(answer(model, tokens, targets, fresh, config).sum())
                    cached_correct += int(answer(model, tokens, targets, promoted, config).sum())
                    cache_matches += int(promoted.eq(fresh).all(dim=1).sum())
                    total += args.batch_size
        return {
            "noise_std": noise_std,
            "threshold": threshold,
            "refreshes_per_episode": refreshes / args.episodes,
            "routing_fraction": refreshes / (args.episodes * len(slots)),
            "fresh_accuracy": fresh_correct / total,
            "cached_accuracy": cached_correct / total,
            "cache_matches_fresh_router": cache_matches / total,
            "gate_refresh_precision": gate_tp / max(1, gate_tp + gate_fp),
            "gate_refresh_recall": gate_tp / max(1, gate_tp + gate_fn),
        }
    noise_levels = args.eval_noise_stds or [args.noise_std]
    thresholds = args.thresholds or [args.threshold]
    sweep = [evaluate(noise_std, threshold) for noise_std in noise_levels for threshold in thresholds]
    report = {
        "training_steps": args.steps,
        "training_noise_std": args.noise_std,
        "sweep": sweep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
