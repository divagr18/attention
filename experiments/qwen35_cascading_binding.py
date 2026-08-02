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

_CORES: dict[tuple[int, str], CascadingKVAttention] = {}


def _core(head_dim: int, device: torch.device, config: CascadingAttentionConfig) -> CascadingKVAttention:
    key = (head_dim, str(device))
    if key not in _CORES:
        _CORES[key] = CascadingKVAttention(head_dim, config).to(device).eval()
    return _CORES[key]


def make_cascading_attention(config: CascadingAttentionConfig):
    """Build a transformers-compatible attention fn backed by the cascading core."""

    def cascading_attention_forward(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
        if query.size(2) != 1:
            raise NotImplementedError("cascading binding is decode-only (S==1) in v1; prefill needs causal masking")
        heads = query.size(1)
        kv_heads = key.size(1)
        if heads != kv_heads:
            groups = heads // kv_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        core = _core(query.size(-1), query.device, config)
        output, _ = core(query[:, :, 0, :].contiguous(), key.contiguous(), value.contiguous(), tree=None)
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
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    core_config = CascadingAttentionConfig()
    register_cascading_attention(core_config)
    model = _load(args.model)
    vocab = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
    input_ids = torch.randint(0, vocab, (1, args.prefix_tokens), device="cuda")
    next_id = torch.randint(0, vocab, (1, 1), device="cuda")

    model.config._attn_implementation = "eager"
    eager_cache = model(input_ids, use_cache=True).past_key_values
    reference = model(next_id, past_key_values=eager_cache).logits[:, -1, :]

    cascading_cache = model(input_ids, use_cache=True).past_key_values
    model.config._attn_implementation = "cascading"
    test = model(next_id, past_key_values=cascading_cache).logits[:, -1, :]
    model.config._attn_implementation = "eager"

    max_error = (reference - test).abs().max().item()
    report = {
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "decode_parity_max_abs_error": max_error,
        "parity": max_error < 1e-3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["parity"]:
        raise SystemExit(f"decode parity FAILED: max abs error {max_error:.3e}")


if __name__ == "__main__":
    main()
