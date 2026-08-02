#!/usr/bin/env python3
"""RULER-style long-context eval for the cascading binding.

Two task families beyond single-needle NIAH:
- multi_niah: several (key, value) needles in distinct pages; query one key's
  value. The router must discriminate the target needle's page from distractor
  needles and filler, testing content-based multi-page retrieval.
- aggregation: a target word appears N times across the context; query the count.
  Fixed-budget page retrieval cannot gather every occurrence, so this exposes
  the global-aggregation limit that motivates the recurrent-memory pillar.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from cascading_kv_attention import CascadingAttentionConfig
from qwen35_cascading_binding import register_cascading_attention, set_router_weights
from qwen35_niah_eval import generate_answer

FILLER = "This is generic filler text that does not contain any useful information. "
AGG_WORD = "zebra"


def _bos(tokenizer) -> list[int]:
    return [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []


def build_multi_niah(tokenizer, context_tokens: int, num_needles: int, target_idx: int):
    keys = random.sample(range(100, 999), num_needles)
    values = random.sample(range(1000, 9999), num_needles)
    query_ids = tokenizer(f"\nQuestion: What is the magic number for key {keys[target_idx]}? Answer:", add_special_tokens=False).input_ids
    needle_ids = [tokenizer(f"The magic number for key {keys[i]} is {values[i]}.", add_special_tokens=False).input_ids for i in range(num_needles)]
    filler_ids = tokenizer(FILLER, add_special_tokens=False).input_ids
    bos = _bos(tokenizer)
    total_needle = sum(len(n) for n in needle_ids)
    filler_budget = max(0, context_tokens - len(query_ids) - len(bos) - total_needle)
    gap = filler_budget // (num_needles + 1)
    filler_seg = (filler_ids * (gap // len(filler_ids) + 1))[:gap]
    seq = list(bos)
    needle_positions = []
    for i in range(num_needles):
        seq += filler_seg
        needle_positions.append(len(seq))
        seq += needle_ids[i]
    remainder = filler_budget - gap * num_needles
    seq += (filler_ids * (remainder // len(filler_ids) + 1))[:remainder]
    seq += query_ids
    input_ids = torch.tensor([seq[:context_tokens]], device="cuda")
    return input_ids, str(values[target_idx]), needle_positions[target_idx], len(needle_ids[target_idx])


def build_aggregation(tokenizer, context_tokens: int, count: int):
    query_ids = tokenizer(f"\nQuestion: How many times does the word '{AGG_WORD}' appear in the text above? Answer with a number:", add_special_tokens=False).input_ids
    filler_ids = tokenizer(FILLER, add_special_tokens=False).input_ids
    word_ids = tokenizer(AGG_WORD + ". ", add_special_tokens=False).input_ids
    bos = _bos(tokenizer)
    filler_budget = max(0, context_tokens - len(query_ids) - len(bos) - count * len(word_ids))
    chunk = filler_budget // max(1, count)
    filler_seg = (filler_ids * (chunk // len(filler_ids) + 1))[:chunk]
    seq = list(bos)
    for _ in range(count):
        seq += filler_seg
        seq += word_ids
    seq += query_ids
    input_ids = torch.tensor([seq[:context_tokens]], device="cuda")
    return input_ids, str(count)


def first_number(text: str) -> str | None:
    match = re.search(r"\d+", text)
    return match.group(0) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--hot-window", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--retrieval-pages", type=int, default=4)
    parser.add_argument("--routing-rotary-dim", type=int, default=64)
    parser.add_argument("--router-weights", type=Path, default=None)
    parser.add_argument("--task", choices=("multi_niah", "aggregation"), default="multi_niah")
    parser.add_argument("--num-needles", type=int, default=8)
    parser.add_argument("--agg-count", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    core_config = CascadingAttentionConfig(
        hot_window=args.hot_window,
        page_size=args.page_size,
        retrieval_pages=args.retrieval_pages,
        tree_beam=args.retrieval_pages,
        routing_rotary_dim=args.routing_rotary_dim,
    )
    register_cascading_attention(core_config)
    if args.router_weights is not None:
        set_router_weights(torch.load(args.router_weights, weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager").cuda().eval()

    modes = ["dense", "oracle", "routed"] if args.task == "multi_niah" else ["dense", "routed"]
    results = {mode: {"correct": 0, "total": 0} for mode in modes}
    for sample in range(args.num_samples):
        if args.task == "multi_niah":
            target_idx = random.randrange(args.num_needles)
            input_ids, answer, target_pos, needle_len = build_multi_niah(tokenizer, args.context_tokens, args.num_needles, target_idx)
            start_page = target_pos // args.page_size
            end_page = (target_pos + needle_len - 1) // args.page_size
            oracle_pages = list(range(start_page, end_page + 1))
        else:
            input_ids, answer = build_aggregation(tokenizer, args.context_tokens, args.agg_count)
            oracle_pages = []
        for mode in modes:
            text = generate_answer(model, tokenizer, input_ids, args.max_new_tokens, mode, oracle_pages)
            results[mode]["correct"] += int(first_number(text) == answer)
            results[mode]["total"] += 1
        if (sample + 1) % max(1, args.num_samples // 10) == 0:
            print(f"sample {sample + 1}/{args.num_samples}", flush=True)

    summary = {mode: results[mode]["correct"] / results[mode]["total"] if results[mode]["total"] else 0.0 for mode in modes}
    report = {
        "task": args.task,
        "context_tokens": args.context_tokens,
        "num_needles": args.num_needles,
        "agg_count": args.agg_count,
        "retrieval_pages": args.retrieval_pages,
        "num_samples": args.num_samples,
        "results": results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
