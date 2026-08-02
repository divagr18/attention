#!/usr/bin/env python3
"""Verify windowed local attention matches the full masked reference path."""

from __future__ import annotations

import torch

from train_tiny_transformer import CausalAttention


def main() -> None:
    torch.manual_seed(7)
    attention = CausalAttention(16, 4).eval()
    hidden = torch.randn(2, 64, 16)
    evidence = torch.tensor([7, 19])
    retrieved = torch.tensor([[7, 8, 9], [19, 20, 21]])
    cases = (
        ("sliding", None, None),
        ("oracle", evidence, None),
        ("learned", None, retrieved),
        ("tree", None, retrieved),
    )
    for variant, positions, indices in cases:
        windowed, _ = attention(
            hidden,
            variant=variant,
            local_window=8,
            evidence_positions=positions,
            retrieved_indices=indices,
        )
        reference, _ = attention(
            hidden,
            variant=variant,
            local_window=8,
            evidence_positions=positions,
            retrieved_indices=indices,
            capture_attention=True,
        )
        error = (windowed - reference).abs().max().item()
        assert error < 1e-5, f"{variant} mismatch: {error}"
        print(f"{variant}: max error {error:.2e}")


if __name__ == "__main__":
    main()
