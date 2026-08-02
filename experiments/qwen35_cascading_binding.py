#!/usr/bin/env python3
"""Bind CascadingKVAttention into Qwen3.5's full-attention layers and parity-test it.

The binding registers a custom attention function in transformers'
ALL_ATTENTION_FUNCTIONS registry, so the host forward (q_proj, query/gate split,
q_norm/k_norm, partial RoPE, cache.update, sigmoid gate, o_proj) is reused
verbatim and only the softmax kernel is swapped.  The decode parity gate runs a
prefix with eager attention, then decodes one token with eager vs cascading and
asserts the logits match in fp32.  Retrieval (page_count > 0) is Phase 2.2; here
the core runs its page_count==0 dense fast path, which is exactly eager decode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from cascading_kv_attention import CascadingAttentionConfig, CascadingKVAttention

_ORACLE_PAGES: list[int] | None = None


def set_oracle_pages(pages: list[int] | None) -> None:
    """Force every layer to promote these pages (oracle retrieval); None restores routing."""
    global _ORACLE_PAGES
    _ORACLE_PAGES = pages


_ROUTER_WEIGHTS: dict[str, torch.Tensor] | None = None


def set_router_weights(weights: dict[str, torch.Tensor] | None) -> None:
    """Load trained router projections (page_summary, query_proj) into the core on first use."""
    global _ROUTER_WEIGHTS
    _ROUTER_WEIGHTS = weights


_FLAT_ROUTING: bool = False


def set_flat_routing(enabled: bool) -> None:
    """Score all pages flatly (top-k) instead of the beam tree; robust at moderate page counts."""
    global _FLAT_ROUTING
    _FLAT_ROUTING = enabled


def make_cascading_attention(config: CascadingAttentionConfig):
    """Build a transformers-compatible attention fn backed by the cascading core."""
    core_holder: dict[str, CascadingKVAttention] = {}

    def cascading_attention_forward(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
        if query.size(2) != 1:
            raise NotImplementedError("cascading binding is decode-only (S==1) in v1; prefill needs causal masking")
        if "core" not in core_holder:
            # Match the host model's dtype so the tree's summary projections agree with the K/V.
            core = CascadingKVAttention(query.size(-1), config).to(device=query.device, dtype=query.dtype).eval()
            if _ROUTER_WEIGHTS is not None:
                with torch.no_grad():
                    core.page_summary.weight.copy_(_ROUTER_WEIGHTS["page_summary"].to(query.dtype))
                    core.query_proj.weight.copy_(_ROUTER_WEIGHTS["query_proj"].to(query.dtype))
            core_holder["core"] = core
        core = core_holder["core"]
        heads = query.size(1)
        kv_heads = key.size(1)
        if heads != kv_heads:
            groups = heads // kv_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        length = key.size(2)
        page_count = max(0, (length - config.hot_window) // config.page_size)
        tree = None
        force_page_indices = None
        if page_count:
            # Mean/max page summaries drive routing; the core gathers exact K/V.
            paged = page_count * config.page_size
            page_kv = key[:, :, :paged].view(key.size(0), heads, page_count, config.page_size, key.size(-1))
            page_keys = core.page_summary(torch.cat((page_kv.mean(dim=(1, 3)), page_kv.amax(dim=(1, 3))), dim=-1))
            tree = core.make_tree(page_keys)
            if _ORACLE_PAGES:
                valid_pages = [page for page in _ORACLE_PAGES if 0 <= page < page_count]
                if valid_pages:
                    force_page_indices = torch.tensor(valid_pages, device=key.device, dtype=torch.long).unsqueeze(0).expand(key.size(0), len(valid_pages))
            elif _FLAT_ROUTING:
                # Flat routing: score every page and take top-k. Robust at moderate
                # page counts, where the untrained beam tree prunes the target page.
                search_query = core.query_proj(query[:, :, 0, :].mean(dim=1))
                flat_scores = torch.einsum("bd,bpd->bp", search_query, page_keys) / (query.size(-1) ** 0.5)
                force_page_indices = flat_scores.topk(min(config.retrieval_pages, page_count), dim=-1).indices
        output, _ = core(query[:, :, 0, :].contiguous(), key.contiguous(), value.contiguous(), tree=tree, force_page_indices=force_page_indices)
        return output.unsqueeze(2).transpose(1, 2).contiguous(), None

    return cascading_attention_forward


def register_cascading_attention(config: CascadingAttentionConfig) -> None:
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError as error:  # pragma: no cover - version-dependent location
        raise SystemExit("Could not import ALL_ATTENTION_FUNCTIONS from this transformers version.") from error
    ALL_ATTENTION_FUNCTIONS["cascading"] = make_cascading_attention(config)


def _load(model: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, attn_implementation="eager").cuda().eval()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--prefix-tokens", type=int, default=8, help="Gate 1 prefix length (below hot-window so page_count=0).")
    parser.add_argument("--hot-window", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--retrieval-pages", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    core_config = CascadingAttentionConfig(
        hot_window=args.hot_window,
        page_size=args.page_size,
        retrieval_pages=args.retrieval_pages,
        tree_beam=args.retrieval_pages,
    )
    register_cascading_attention(core_config)
    model = _load(args.model)
    vocab = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size

    def decode_parity(prefix_len: int) -> float:
        input_ids = torch.randint(0, vocab, (1, prefix_len), device="cuda")
        next_id = torch.randint(0, vocab, (1, 1), device="cuda")
        model.config._attn_implementation = "eager"
        eager_cache = model(input_ids, use_cache=True).past_key_values
        reference = model(next_id, past_key_values=eager_cache).logits[:, -1, :]
        cascading_cache = model(input_ids, use_cache=True).past_key_values
        model.config._attn_implementation = "cascading"
        test = model(next_id, past_key_values=cascading_cache).logits[:, -1, :]
        model.config._attn_implementation = "eager"
        return (reference - test).abs().max().item()

    gate1_error = decode_parity(args.prefix_tokens)
    gate2_prefix = args.hot_window + args.retrieval_pages * args.page_size
    gate2_error = decode_parity(gate2_prefix)
    report = {
        "model": args.model,
        "hot_window": args.hot_window,
        "page_size": args.page_size,
        "retrieval_pages": args.retrieval_pages,
        "gate1_prefix_tokens": args.prefix_tokens,
        "gate1_max_abs_error": gate1_error,
        "gate2_prefix_tokens": gate2_prefix,
        "gate2_max_abs_error": gate2_error,
        "parity": gate1_error < 1e-3 and gate2_error < 1e-3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["parity"]:
        raise SystemExit(f"decode parity FAILED: gate1={gate1_error:.3e} gate2={gate2_error:.3e}")


if __name__ == "__main__":
    main()
