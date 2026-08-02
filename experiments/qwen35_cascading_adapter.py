#!/usr/bin/env python3
"""Probe a Qwen3.5 checkpoint and emit the cascading-attention integration manifest.

Loads the installed Transformers implementation (no monkey-patching), captures
the hybrid layer layout, the full-attention forward contract, the GQA / RoPE /
output-gating parameters, and the runtime KV-cache class, then writes a
machine-readable manifest.  The model-specific binding is written against this
manifest rather than against an assumed remote class, so a transformers version
bump surfaces here instead of silently changing a checkpoint's semantics.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


def _cfg(config, key, default=None):
    """Read a field from a possibly-multimodal config (top-level or text_config)."""
    value = getattr(config, key, None)
    if value is not None:
        return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return getattr(text_config, key, default)
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager", help="eager avoids a flash-att dependency for probing.")
    parser.add_argument("--probe-tokens", type=int, default=8, help="Tiny prompt length for the runtime cache probe (0 skips it).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise SystemExit("Install transformers>=5 and accelerate (requirements-qwen.txt) before running the Qwen adapter probe.") from error

    config = AutoConfig.from_pretrained(args.model)

    layer_types = _cfg(config, "layer_types")
    head_dim = _cfg(config, "head_dim")
    num_attention_heads = _cfg(config, "num_attention_heads")
    num_key_value_heads = _cfg(config, "num_key_value_heads")
    rope_parameters = _cfg(config, "rope_parameters")
    rope_dict = rope_parameters if isinstance(rope_parameters, dict) else {}
    partial_rotary_factor = _cfg(config, "partial_rotary_factor")
    if partial_rotary_factor is None:
        partial_rotary_factor = rope_dict.get("partial_rotary_factor")
    rotary_dim = int(head_dim * partial_rotary_factor) if head_dim and partial_rotary_factor else None

    report = {
        "model": args.model,
        "architecture": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
        "is_multimodal": hasattr(config, "vision_config"),
        "hidden_size": _cfg(config, "hidden_size"),
        "num_hidden_layers": _cfg(config, "num_hidden_layers"),
        "intermediate_size": _cfg(config, "intermediate_size"),
        "vocab_size": _cfg(config, "vocab_size"),
        "max_position_embeddings": _cfg(config, "max_position_embeddings"),
        "layer_types": layer_types,
        "full_attention_indices": [index for index, kind in enumerate(layer_types) if kind == "full_attention"] if layer_types else None,
        "full_attention_interval": _cfg(config, "full_attention_interval"),
        "attention": {
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "num_key_value_groups": (num_attention_heads // num_key_value_heads) if num_attention_heads and num_key_value_heads else None,
            "head_dim": head_dim,
            "attn_output_gate": _cfg(config, "attn_output_gate"),
            "attention_bias": _cfg(config, "attention_bias"),
            "attention_dropout": _cfg(config, "attention_dropout"),
            "rms_norm_eps": _cfg(config, "rms_norm_eps"),
        },
        "rope": {
            "rope_type": rope_dict.get("rope_type", _cfg(config, "rope_type")),
            "rope_theta": rope_dict.get("rope_theta", _cfg(config, "rope_theta")),
            "partial_rotary_factor": partial_rotary_factor,
            "rotary_dim": rotary_dim,
            "mrope_section": rope_dict.get("mrope_section"),
        },
        "linear_attention": {
            "linear_num_value_heads": _cfg(config, "linear_num_value_heads"),
            "linear_num_key_heads": _cfg(config, "linear_num_key_heads"),
            "linear_value_head_dim": _cfg(config, "linear_value_head_dim"),
            "linear_key_head_dim": _cfg(config, "linear_key_head_dim"),
            "linear_conv_kernel_dim": _cfg(config, "linear_conv_kernel_dim"),
        },
        "adapter_core": "experiments/cascading_kv_attention.py:CascadingKVAttention",
    }

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation=args.attn_implementation)
    attention_modules = []
    for name, module in model.named_modules():
        type_name = type(module).__name__
        if "attention" in type_name.lower() and "delta" not in type_name.lower():
            attention_modules.append({"name": name, "class": type_name, "forward": str(inspect.signature(module.forward))})
    report["candidate_periodic_attention_modules"] = attention_modules

    if args.probe_tokens and torch.cuda.is_available():
        try:
            input_ids = torch.randint(0, report["vocab_size"] or 1000, (1, args.probe_tokens), device="cuda")
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
            cache = outputs.past_key_values
            report["runtime_cache"] = {
                "class": type(cache).__name__,
                "num_layers": len(getattr(cache, "layers", [])),
                "layer_classes": [type(layer).__name__ for layer in getattr(cache, "layers", [])],
            }
        except Exception as error:  # noqa: BLE001 - a failed forward must not drop the static manifest
            report["runtime_cache"] = {"error": repr(error)}

    report["next_action"] = (
        "Bind CascadingKVAttention to the full_attention layers only: preserve the q_proj query/gate split, "
        "q_norm/k_norm, partial RoPE, and past_key_values.update; replace the dense softmax with the routed "
        "local-window + tree-page core over the post-RoPE cached K/V (GQA: expand KV heads to query groups); "
        "then apply sigmoid(gate) and o_proj. Validate dense<->routed parity before training router summaries."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
