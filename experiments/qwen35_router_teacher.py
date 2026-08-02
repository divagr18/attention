#!/usr/bin/env python3
"""Generate offline router-distillation targets from Qwen3.5's dense attention.

For each NIAH example, prefills the prompt, then decodes the final query token
through a teacher-capture attention function that records each full-attention
layer's attention.  The capture aggregates to tiny per-page summaries on-GPU
(renormalized page-mass target, mean/max page keys, head-averaged query),
averaged over the full-attention layers, so the tiny router trains offline
without the large model and without large host transfers.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from qwen35_niah_eval import build_niah

FULL_ATTENTION_INTERVAL = 4  # full attention every 4th layer: indices 3,7,...,31
CAPTURED: dict[int, dict[str, torch.Tensor]] = {}


def install_teacher_capture(page_count: int, page_size: int, routing_rotary_dim: int = 0) -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    paged = page_count * page_size

    def teacher_capture_forward(module, query, key, value, attention_mask=None, **kwargs):
        # Standard eager attention (GQA repeat, scaled dot-product, additive mask,
        # fp32 softmax). The module applies the output gate after this returns.
        num_heads = query.size(1)
        num_kv_heads = key.size(1)
        if num_heads != num_kv_heads:
            groups = num_heads // num_kv_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        scaling = kwargs.get("scaling", query.size(-1) ** -0.5)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        layer_idx = getattr(module, "layer_idx", None)
        # Only the single-query decode step is captured; skip the prefill call and
        # aggregate to per-page summaries on-GPU to avoid large host transfers.
        if layer_idx is not None and query.size(2) == 1:
            with torch.no_grad():
                attn = attn_weights[0, :, 0, :].mean(dim=0)
                page_mass = attn[:paged].view(page_count, page_size).sum(dim=1)
                paged_key = key[0, :, :paged, :].view(key.size(1), page_count, page_size, -1)
                if routing_rotary_dim:
                    # Drop leading RoPE dims to match the binding's content-based routing.
                    paged_key = paged_key.clone()
                    paged_key[..., :routing_rotary_dim] = 0.0
                page_keys_mm = torch.cat((paged_key.mean(dim=(0, 2)), paged_key.amax(dim=(0, 2))), dim=-1)
                query_avg = query[0, :, 0, :].mean(dim=0)
                if routing_rotary_dim:
                    query_avg = query_avg.clone()
                    query_avg[..., :routing_rotary_dim] = 0.0
            CAPTURED[layer_idx] = {
                "page_mass": page_mass.detach().float().cpu(),
                "page_keys": page_keys_mm.detach().float().cpu(),
                "query": query_avg.detach().float().cpu(),
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
    parser.add_argument("--routing-rotary-dim", type=int, default=0, help="Zero this many leading RoPE dims in routing inputs for length-invariant routing.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    bos_len = 1 if tokenizer.bos_token_id is not None else 0
    page_count = (args.context_tokens + bos_len - args.hot_window) // args.page_size
    install_teacher_capture(page_count, args.page_size, args.routing_rotary_dim)
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

            CAPTURED.clear()
            cache = model(input_ids[:, :-1], use_cache=True).past_key_values
            CAPTURED.clear()
            model(input_ids[:, -1:], past_key_values=cache, use_cache=False)

            layer_targets, layer_page_keys, layer_queries = [], [], []
            usable = True
            for layer_idx in full_attention_layers:
                cap = CAPTURED.get(layer_idx)
                if cap is None:
                    usable = False
                    break
                page_mass = cap["page_mass"]
                total = page_mass.sum()
                if total < 1e-3:
                    usable = False
                    break
                layer_targets.append(page_mass / total)
                layer_page_keys.append(cap["page_keys"])
                layer_queries.append(cap["query"])
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
        "page_count": page_count,
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
