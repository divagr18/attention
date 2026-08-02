#!/usr/bin/env python3
"""Generate offline router-distillation targets from Qwen3.5's dense attention.

For each NIAH example, prefills the prompt eagerly, then decodes the final query
token through a teacher-capture attention function that records each
full-attention layer's query, post-RoPE key, and attention weights.  From these
it derives the renormalized page-mass teacher target and the router inputs
(mean/max page keys and head-averaged query), averaged over the full-attention
layers, and saves them so the tiny router trains without the large model.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from qwen35_niah_eval import build_niah

FULL_ATTENTION_INTERVAL = 4  # full attention every 4th layer: indices 3,7,...,31
CAPTURED: dict[int, dict[str, torch.Tensor]] = {}


def install_teacher_capture() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    eager_fn = ALL_ATTENTION_FUNCTIONS["eager"]

    def teacher_capture_forward(module, query, key, value, attention_mask=None, **kwargs):
        attn_output, attn_weights = eager_fn(module, query, key, value, attention_mask=attention_mask, **kwargs)
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is not None:
            CAPTURED[layer_idx] = {
                "query": query.detach().float().cpu(),
                "key": key.detach().float().cpu(),
                "attn_weights": attn_weights.detach().float().cpu() if attn_weights is not None else None,
            }
        return attn_output, attn_weights

    ALL_ATTENTION_FUNCTIONS["teacher_capture"] = teacher_capture_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--hot-window", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--max-depth", type=float, default=0.8, help="Keep needles in the paged region, outside the local window.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    install_teacher_capture()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="teacher_capture").cuda().eval()
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", None) or config.text_config.num_hidden_layers
    full_attention_layers = [i for i in range(num_layers) if (i + 1) % FULL_ATTENTION_INTERVAL == 0]

    targets, page_keys_list, queries = [], [], []
    skipped = 0
    with torch.no_grad():
        for sample in range(args.num_samples):
            number = random.randint(100, 999)
            depth = random.uniform(0.0, args.max_depth)
            input_ids, _, _ = build_niah(tokenizer, args.context_tokens, depth, number)
            length = input_ids.size(1)
            page_count = (length - args.hot_window) // args.page_size
            paged = page_count * args.page_size

            CAPTURED.clear()
            cache = model(input_ids[:, :-1], use_cache=True).past_key_values
            CAPTURED.clear()
            model(input_ids[:, -1:], past_key_values=cache, use_cache=False)

            layer_targets, layer_page_keys, layer_queries = [], [], []
            usable = True
            for layer_idx in full_attention_layers:
                cap = CAPTURED.get(layer_idx)
                if cap is None or cap["attn_weights"] is None:
                    usable = False
                    break
                attn = cap["attn_weights"][0, :, 0, :].mean(dim=0)  # [L], head-averaged
                page_mass = attn[:paged].view(page_count, args.page_size).sum(dim=1)
                total = page_mass.sum()
                if total < 1e-3:
                    usable = False
                    break
                layer_targets.append(page_mass / total)
                paged_key = cap["key"][0, :, :paged, :].view(cap["key"].size(1), page_count, args.page_size, -1)
                layer_page_keys.append(torch.cat((paged_key.mean(dim=(0, 2)), paged_key.amax(dim=(0, 2))), dim=-1))
                layer_queries.append(cap["query"][0, :, 0, :].mean(dim=0))
            if not usable:
                skipped += 1
                continue
            targets.append(torch.stack(layer_targets).mean(dim=0))
            page_keys_list.append(torch.stack(layer_page_keys).mean(dim=0))
            queries.append(torch.stack(layer_queries).mean(dim=0))
            if (sample + 1) % 100 == 0:
                print(f"sample {sample + 1}/{args.num_samples} kept={len(targets)} skipped={skipped}", flush=True)

    bundle = {
        "targets": torch.stack(targets),
        "page_keys": torch.stack(page_keys_list),
        "queries": torch.stack(queries),
        "page_count": page_count if targets else 0,
        "page_size": args.page_size,
        "hot_window": args.hot_window,
        "context_tokens": args.context_tokens,
        "num_samples": len(targets),
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"saved {len(targets)} examples (skipped {skipped}) to {args.output}")


if __name__ == "__main__":
    main()
