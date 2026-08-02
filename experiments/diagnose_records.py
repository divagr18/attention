"""Per-record retrieval diagnostic for the multirecord task.

Loads a trained checkpoint and, for each of the four fixed record positions,
forces the query to target that record and reports answer accuracy together
with the router's selected span center relative to the true record start.
This isolates whether a distance bucket (for example "medium") is genuinely
hard or whether a single fixed record position drives the bucket average.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_tiny_transformer import (
    KEY_COUNT,
    VALUE_START,
    Config,
    TinyRetrievalTransformer,
    make_batch,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=128, help="Evaluation batches per record index.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = Config(**checkpoint["config"])
    config.device = str(device)
    model = TinyRetrievalTransformer(config).to(device)
    # Pre-tree checkpoints lack the tree-only summary weights; flat routing never
    # uses them, so tolerate exactly those and reject any other mismatch.
    load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
    tree_only = {"router.page_summary.weight", "router.internal_summary.weight"}
    real_missing = set(load_result.missing_keys) - tree_only
    if real_missing or load_result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={sorted(real_missing)} unexpected={sorted(load_result.unexpected_keys)}")
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    historical_length = config.context - config.local_window
    record_positions = torch.linspace(4, historical_length - 4, 4, device=device).round().long().tolist()
    width = config.retrieval_width

    per_record = []
    for chosen in range(4):
        correct = total = block_hits = exact_start = within_span = offset_sum = 0
        offset_histogram: dict[int, int] = {}
        for _ in range(args.batches):
            tokens, targets, evidence_positions = make_batch(config, device, generator, force_chosen=chosen)
            logits, routing, _ = model(
                tokens,
                variant="learned",
                local_window=config.local_window,
                evidence_positions=evidence_positions,
                block_size=config.block_size,
                top_blocks=config.top_blocks,
                top_tokens=config.top_tokens,
            )
            if routing is None:
                raise RuntimeError("diagnostic requires a learned/flat/tree router")
            predictions = logits[:, -1, VALUE_START : VALUE_START + KEY_COUNT].argmax(dim=-1) + VALUE_START
            correct += int(predictions.eq(targets).sum())
            total += config.batch_size
            centers = routing.retrieved_indices[:, ::width]
            true_start = evidence_positions
            for offset in (centers[:, 0] - true_start).tolist():
                offset_sum += offset
                offset_histogram[offset] = offset_histogram.get(offset, 0) + 1
                if offset == 0:
                    exact_start += 1
                if -1 <= offset <= width - 2:
                    within_span += 1
            retrieved_blocks = routing.retrieved_indices // config.block_size
            true_block = true_start // config.block_size
            block_hits += int(retrieved_blocks.eq(true_block.unsqueeze(1)).any(dim=1).sum())
        per_record.append({
            "record_index": chosen,
            "record_position": record_positions[chosen],
            "distance": config.context - record_positions[chosen],
            "accuracy": correct / total if total else 0.0,
            "block_recall": block_hits / total if total else 0.0,
            "exact_start_fraction": exact_start / total if total else 0.0,
            "value_in_span_fraction": within_span / total if total else 0.0,
            "mean_center_offset": offset_sum / total if total else 0.0,
            "offset_histogram": {str(key): value for key, value in sorted(offset_histogram.items())},
            "count": total,
        })

    report = {
        "context": config.context,
        "local_window": config.local_window,
        "block_size": config.block_size,
        "retrieval_unit": config.retrieval_unit,
        "retrieval_width": width,
        "batches_per_record": args.batches,
        "records": per_record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
