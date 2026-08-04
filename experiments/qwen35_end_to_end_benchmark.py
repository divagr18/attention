#!/usr/bin/env python3
"""End-to-end dense-vs-cascading benchmark on NIAH.

Measures prefill and decode latency plus answer accuracy for:
- dense: SDPA prefill + SDPA decode (full attention throughout).
- cascading: fused local-window prefill + routed decode (local window plus
  flat top-k pages).
The prefill speedup is the subquadratic claim; the decode speedup is the
retrieval claim; accuracy confirms the cascading path preserves quality.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from cascading_kv_attention import CascadingAttentionConfig
from qwen35_cascading_binding import register_cascading_attention, set_flat_routing, set_oracle_pages, set_router_weights
from qwen35_niah_eval import build_niah, first_number


@torch.no_grad()
def time_forward(model, input_ids, impl, decode_steps, decode_impl=None):
    model.config._attn_implementation = impl
    # Time prefill through the backbone only; the full-sequence lm_head would
    # materialize [seq, vocab] logits (tens of GiB at long context) and is not
    # part of the attention/DeltaNet prefill we are measuring.
    backbone = model.model
    backbone(input_ids, use_cache=True)  # warmup
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = backbone(input_ids, use_cache=True)
    cache = out.past_key_values
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - start) * 1000
    logits = model.lm_head(out.last_hidden_state[:, -1:, :].to(model.lm_head.weight.dtype))
    next_token = logits.argmax(dim=-1)
    generated = [next_token.item()]
    if decode_impl is not None:
        model.config._attn_implementation = decode_impl
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(decode_steps):
        out = model(next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_token.item())
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - start) * 1000
    return prefill_ms, decode_ms / decode_steps, generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 65536])
    parser.add_argument("--hot-window", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--retrieval-pages", type=int, default=4)
    parser.add_argument("--routing-rotary-dim", type=int, default=64)
    parser.add_argument("--router-weights", type=Path, default=None)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--oracle-all-pages", action="store_true", help="Diagnostic: oracle retrieves all pages, testing whether the model needs more than the needle page.")
    parser.add_argument("--oracle-window", type=int, default=0, help="Extra pages retrieved on each side of the needle pages (0 = needle pages only).")
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
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa").cuda().eval()

    results = []
    with torch.no_grad():
        for context in args.contexts:
            number = random.randint(100, 999)
            input_ids, needle_position, needle_len = build_niah(tokenizer, context, args.depth, number)
            answer = str(number)
            start_page = needle_position // args.page_size
            end_page = (needle_position + needle_len - 1) // args.page_size
            page_count = (input_ids.size(1) - args.hot_window) // args.page_size
            if args.oracle_all_pages:
                oracle_pages = list(range(page_count))
            else:
                oracle_pages = list(
                    range(
                        max(0, start_page - args.oracle_window),
                        min(end_page + args.oracle_window + 1, page_count),
                    )
                )
            # Diagnostic: decode the retrieved span and confirm the needle text
            # is actually inside it. If false, correct=0 is a page-mapping bug,
            # not a model limitation.
            retrieved_text = tokenizer.decode(
                input_ids[0, oracle_pages[0] * args.page_size : (oracle_pages[-1] + 1) * args.page_size].tolist(),
                skip_special_tokens=True,
            )
            row = {
                "context": context,
                "answer": answer,
                "needle_position": needle_position,
                "needle_pages": [start_page, end_page],
                "oracle_pages": oracle_pages,
                "needle_in_retrieved_pages": int(str(number) in retrieved_text),
            }

            prefill_ms, decode_ms, generated = time_forward(model, input_ids, "sdpa", args.decode_steps)
            row["dense_prefill_ms"] = prefill_ms
            row["dense_decode_ms_per_step"] = decode_ms
            row["dense_correct"] = int(first_number(tokenizer.decode(generated, skip_special_tokens=True)) == answer)

            # First demonstration uses the oracle (force the needle's page): the
            # untrained Llama router would answer incorrectly. Oracle and routed
            # have the same retrieval latency, so the speedup measurement is valid.
            set_oracle_pages(oracle_pages)
            set_flat_routing(False)
            prefill_ms, decode_ms, generated = time_forward(model, input_ids, "cascading", args.decode_steps)
            set_oracle_pages(None)
            row["cascading_prefill_ms"] = prefill_ms
            row["cascading_decode_ms_per_step"] = decode_ms
            row["cascading_correct"] = int(first_number(tokenizer.decode(generated, skip_special_tokens=True)) == answer)

            # Diagnostic: SDPA prefill + cascading decode + oracle. Isolates
            # whether correct=0 is the local-window prefill or the selective
            # decode/retrieval path.
            set_oracle_pages(oracle_pages)
            set_flat_routing(False)
            _, _, generated = time_forward(model, input_ids, "sdpa", args.decode_steps, decode_impl="cascading")
            set_oracle_pages(None)
            row["sdpa_prefill_cascading_decode_correct"] = int(first_number(tokenizer.decode(generated, skip_special_tokens=True)) == answer)

            row["prefill_speedup"] = row["dense_prefill_ms"] / row["cascading_prefill_ms"]
            row["decode_speedup"] = row["dense_decode_ms_per_step"] / row["cascading_decode_ms_per_step"]
            results.append(row)
            print(json.dumps(row, indent=2), flush=True)

    report = {
        "hot_window": args.hot_window,
        "page_size": args.page_size,
        "retrieval_pages": args.retrieval_pages,
        "decode_steps": args.decode_steps,
        "depth": args.depth,
        "oracle_window": args.oracle_window,
        "oracle_all_pages": args.oracle_all_pages,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
