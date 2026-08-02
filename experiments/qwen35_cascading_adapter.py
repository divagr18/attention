#!/usr/bin/env python3
"""Validate a Qwen3.5 checkpoint for the cascading-attention adapter.

This deliberately does not monkey-patch an unverified remote model class.
It loads the exact installed Transformers implementation, identifies periodic
attention modules, and emits a machine-readable integration manifest.  The
model-specific replacement hook can then be written against the reported
forward signature rather than silently changing a checkpoint's semantics.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise SystemExit("Install transformers>=5 and accelerate in the RunPod environment before running the Qwen adapter probe.") from error
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="flash_attention_2")
    candidates = []
    for name, module in model.named_modules():
        type_name = type(module).__name__
        if "attention" in type_name.lower() and "delta" not in type_name.lower():
            candidates.append({"name": name, "class": type_name, "forward": str(inspect.signature(module.forward))})
    report = {
        "model": args.model,
        "architecture": getattr(model.config, "architectures", None),
        "hidden_size": getattr(model.config, "hidden_size", None),
        "candidate_periodic_attention_modules": candidates,
        "adapter_core": "experiments/cascading_kv_attention.py:CascadingKVAttention",
        "next_action": "Bind the adapter core to the reported Q/K/V projection and cache interface, then run dense/local/flat/tree parity before LoRA training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
