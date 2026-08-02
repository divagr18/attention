#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval eval for the Qwen3.5 cascading binding.

Compares three decode conditions on a number-retrieval task: dense (eager
throughout), oracle (cascading decode that force-promotes the needle's page),
and routed (cascading decode with the untrained content router).  Prefill is
eager for all conditions, so this isolates decode-time retrieval quality.  The
decisive gate is oracle ~= dense: does the model answer correctly when given the
right page?
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from cascading_kv_attention import CascadingAttentionConfig
from qwen35_cascading_binding import register_cascading_attention, set_oracle_pages, set_router_weights

FILLER = "This is generic filler text that does not contain any useful information. "
NEEDLE_TEMPLATE = "The magic number is {number}."
QUERY = "\nQuestion: What is the magic number? Answer:"
MODES = ("dense", "oracle", "routed")


def build_niah(tokenizer, context_tokens: int, depth_fraction: float, number: int) -> tuple[torch.Tensor, int]:
    needle_ids = tokenizer(NEEDLE_TEMPLATE.format(number=number), add_special_tokens=False).input_ids
    query_ids = tokenizer(QUERY, add_special_tokens=False).input_ids
    filler_ids = tokenizer(FILLER, add_special_tokens=False).input_ids
    haystack_len = context_tokens - len(query_ids)
    haystack = (filler_ids * (haystack_len // len(filler_ids) + 1))[:haystack_len]
    insert_pos = int(depth_fraction * max(0, haystack_len - len(needle_ids)))
    context = (haystack[:insert_pos] + needle_ids + haystack[insert_pos:])[:haystack_len]
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    full = bos + context + query_ids
    return torch.tensor([full], device="cuda"), len(bos) + insert_pos, len(needle_ids)


def first_number(text: str) -> str | None:
    match = re.search(r"\d+", text)
    return match.group(0) if match else None


@torch.no_grad()
def generate_answer(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int, mode: str, oracle_pages: list[int]) -> str:
    # SDPA prefill (memory-efficient dense, O(L) memory) populates the cache for
    # every condition, so the eval fits long context; only the decode differs.
    model.config._attn_implementation = "sdpa"
    out = model(input_ids, use_cache=True)
    cache = out.past_key_values
    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [next_token.item()]
    model.config._attn_implementation = "sdpa" if mode == "dense" else "cascading"
    set_oracle_pages(oracle_pages if mode == "oracle" else None)
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens - 1):
        out = model(next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token.item())
        if next_token.item() == eos:
            break
    set_oracle_pages(None)
    model.config._attn_implementation = "eager"
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--hot-window", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--retrieval-pages", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--router-weights", type=Path, default=None, help="Trained router projections from qwen35_router_train.py.")
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
    )
    register_cascading_attention(core_config)
    if args.router_weights is not None:
        set_router_weights(torch.load(args.router_weights, weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager").cuda().eval()

    results = {mode: {str(depth): {"correct": 0, "total": 0} for depth in args.depths} for mode in MODES}
    for depth in args.depths:
        for _ in range(args.num_samples):
            number = random.randint(100, 999)
            input_ids, needle_position, needle_len = build_niah(tokenizer, args.context_tokens, depth, number)
            page_count = max(0, (input_ids.size(1) - args.hot_window) // args.page_size)
            start_page = needle_position // args.page_size
            end_page = (needle_position + needle_len - 1) // args.page_size
            oracle_pages = list(range(start_page, end_page + 1))
            while len(oracle_pages) < args.retrieval_pages and oracle_pages[-1] + 1 < page_count:
                oracle_pages.append(oracle_pages[-1] + 1)
            for mode in MODES:
                text = generate_answer(model, tokenizer, input_ids, args.max_new_tokens, mode, oracle_pages)
                results[mode][str(depth)]["correct"] += int(first_number(text) == str(number))
                results[mode][str(depth)]["total"] += 1

    summary = {}
    for mode in MODES:
        total_correct = total_count = 0
        per_depth = {}
        for depth, counts in results[mode].items():
            per_depth[depth] = counts["correct"] / counts["total"] if counts["total"] else 0.0
            total_correct += counts["correct"]
            total_count += counts["total"]
        summary[mode] = {"overall": total_correct / total_count if total_count else 0.0, "per_depth": per_depth}

    report = {
        "model": args.model,
        "dtype": args.dtype,
        "context_tokens": args.context_tokens,
        "hot_window": args.hot_window,
        "page_size": args.page_size,
        "retrieval_pages": args.retrieval_pages,
        "num_samples": args.num_samples,
        "depths": args.depths,
        "results": results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
