#!/usr/bin/env python3
"""Train the cascading router (page_summary + query_proj) by dense-teacher distillation.

Loads offline teacher targets and router inputs from qwen35_router_teacher.py and
trains a flat router (score every page) with forward KL against the renormalized
page-mass target, plus temperature and label smoothing.  The trained projections
are saved for qwen35_cascading_binding.set_router_weights().  The flat router is
trained here; the hierarchical tree is used only at inference (equivalent at the
eval's page counts, where the beam covers every internal node).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Offline bundle from qwen35_router_teacher.py.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--label-smooth", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    bundle = torch.load(args.data, weights_only=True)
    targets = bundle["targets"]       # [N, page_count]
    page_keys = bundle["page_keys"]   # [N, page_count, 2D]
    queries = bundle["queries"]       # [N, D]
    page_count = targets.size(1)
    channels = queries.size(1)
    num = targets.size(0)

    page_summary = nn.Linear(2 * channels, channels, bias=False)
    query_proj = nn.Linear(channels, channels, bias=False)
    optimizer = torch.optim.AdamW([page_summary.weight, query_proj.weight], lr=args.lr)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, num, (args.batch_size,))
        page_keys_proj = page_summary(page_keys[idx])     # [B, page_count, D]
        query_proj_out = query_proj(queries[idx])         # [B, D]
        scores = torch.einsum("bd,bpd->bp", query_proj_out, page_keys_proj) / math.sqrt(channels)
        student_log = F.log_softmax(scores / args.temperature, dim=-1)
        teacher = (1 - args.label_smooth) * targets[idx] + args.label_smooth / page_count
        loss = F.kl_div(student_log, teacher, reduction="batchmean") * (args.temperature ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 20) == 0:
            with torch.no_grad():
                top_teacher = targets[idx].argmax(dim=-1)
                topk = scores.topk(min(4, page_count), dim=-1).indices
                recall = topk.eq(top_teacher.unsqueeze(-1)).any(dim=-1).float().mean().item()
            print(f"step={step} loss={loss.item():.4f} recall@4={recall:.3f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"page_summary": page_summary.weight.detach(), "query_proj": query_proj.weight.detach()}, args.output)
    print(f"saved router weights to {args.output}")


if __name__ == "__main__":
    main()
