"""Compare 4K dense, local, routed, and adaptive-promoted decoder queries.

This is an integration benchmark over a trained Transformer rather than a
kernel microbenchmark.  Each step changes the active retrieval query inside a
fixed 4K context, measures answer accuracy, CUDA wall time, and router calls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evaluate_promotion_ttl import build_context, query_tokens, route
from train_tiny_transformer import Config, TinyRetrievalTransformer, VALUE_START


def load_model(path: Path) -> tuple[TinyRetrievalTransformer, Config]:
    device = torch.device("cuda")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = Config(**checkpoint["config"])
    config.device = "cuda"
    model = TinyRetrievalTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


@torch.no_grad()
def predicted(model: TinyRetrievalTransformer, config: Config, tokens: torch.Tensor, variant: str, retrieved: torch.Tensor | None = None) -> torch.Tensor:
    logits, _, _ = model(
        tokens, variant=variant, local_window=config.local_window,
        evidence_positions=None, block_size=config.block_size,
        top_blocks=config.top_blocks, top_tokens=config.top_tokens,
        retrieved_indices_override=retrieved,
    )
    return logits[:, -1, VALUE_START : VALUE_START + 64].argmax(dim=-1) + VALUE_START


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--drift-threshold", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    learned, config = load_model(args.learned_checkpoint)
    dense, dense_config = load_model(args.dense_checkpoint)
    if dense_config.context != config.context:
        raise ValueError("checkpoints need the same context length")
    device = torch.device("cuda")
    # Query changes deliberately do not align with a simple periodic TTL.
    change_steps = {3, 7, 12}
    slots, slot = [], 0
    for step in range(args.steps):
        if step in change_steps:
            slot = (slot + 1) % 4
        slots.append(slot)
    summaries = {name: {"correct": 0, "total": 0, "router_calls": 0, "elapsed": 0.0} for name in ("dense", "sliding", "routed_reroute", "routed_adaptive_promotion")}
    with torch.no_grad():
        for _ in range(args.episodes):
            base, keys, values = build_context(1, config.context, device, config.context - config.local_window)
            promoted = promoted_state = None
            for slot in slots:
                tokens, targets = query_tokens(base, keys, values, slot)
                for name, model, variant, retrieved in (
                    ("dense", dense, "dense", None),
                    ("sliding", learned, "sliding", None),
                ):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    prediction = predicted(model, config, tokens, variant, retrieved)
                    torch.cuda.synchronize()
                    summaries[name]["elapsed"] += time.perf_counter() - start
                    summaries[name]["correct"] += int(prediction.eq(targets).sum())
                    summaries[name]["total"] += 1
                # Fresh routing and exact candidate attention.
                torch.cuda.synchronize()
                start = time.perf_counter()
                fresh = route(learned, tokens, config)
                prediction = predicted(learned, config, tokens, "learned", fresh)
                torch.cuda.synchronize()
                summaries["routed_reroute"]["elapsed"] += time.perf_counter() - start
                summaries["routed_reroute"]["router_calls"] += 1
                summaries["routed_reroute"]["correct"] += int(prediction.eq(targets).sum())
                summaries["routed_reroute"]["total"] += 1
                state = learned.router.query(learned.token_embedding(tokens[:, -3:]).mean(dim=1))
                if promoted is None or 1.0 - torch.nn.functional.cosine_similarity(state, promoted_state).mean() > args.drift_threshold:
                    promoted = fresh
                    promoted_state = state
                    summaries["routed_adaptive_promotion"]["router_calls"] += 1
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = predicted(learned, config, tokens, "learned", promoted)
                torch.cuda.synchronize()
                summaries["routed_adaptive_promotion"]["elapsed"] += time.perf_counter() - start
                summaries["routed_adaptive_promotion"]["correct"] += int(prediction.eq(targets).sum())
                summaries["routed_adaptive_promotion"]["total"] += 1
    report = {"context": config.context, "steps_per_episode": args.steps, "episodes": args.episodes, "change_steps": sorted(change_steps), "results": {}}
    for name, summary in summaries.items():
        report["results"][name] = {
            "answer_accuracy": summary["correct"] / summary["total"],
            "mean_query_ms": 1000 * summary["elapsed"] / summary["total"],
            "router_calls_per_episode": summary["router_calls"] / args.episodes,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
