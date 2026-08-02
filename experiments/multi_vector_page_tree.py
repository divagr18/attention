"""Fixed-slot causal page tree for preserving multiple historical records."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MultiVectorTreeSearch:
    page_indices: Tensor
    score_count: int
    depth: int


class MultiVectorPageTree:
    """Tree whose every page/node holds a fixed number of routing slots.

    A node preserves its highest-norm child slots.  For the structured tiny
    task, unused slots are exactly zero and each nonzero slot is the existing
    flat router's record-key vector, so this operation retains distinct records
    rather than averaging them into one lossy page vector.  Query work remains
    O(beam * fanout * slots * log(pages)).
    """

    def __init__(self, *, fanout: int, slots: int) -> None:
        if fanout < 2 or slots < 1:
            raise ValueError("fanout must be at least two and slots must be positive")
        self.fanout = fanout
        self.slots = slots
        self.levels: list[Tensor] = []  # [batch, nodes, slots, channels], leaves first
        self.valid_levels: list[Tensor] = []  # [batch, nodes, slots]

    @classmethod
    def from_page_slots(cls, page_slots: Tensor, *, fanout: int, slots: int | None = None, slot_valid: Tensor | None = None) -> "MultiVectorPageTree":
        if page_slots.ndim != 4:
            raise ValueError("page_slots must have shape [batch, pages, slots, channels]")
        if not page_slots.size(1):
            raise ValueError("at least one page is required")
        tree = cls(fanout=fanout, slots=slots or page_slots.size(2))
        tree.levels = [page_slots]
        tree.valid_levels = [slot_valid if slot_valid is not None else page_slots.square().sum(dim=-1).ne(0)]
        tree._rebuild_parents()
        return tree

    @property
    def page_count(self) -> int:
        return self.levels[0].size(1)

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    def _rebuild_parents(self) -> None:
        source = self.levels[0]
        source_valid = self.valid_levels[0]
        levels = [source]
        valid_levels = [source_valid]
        while source.size(1) > 1:
            groups = math.ceil(source.size(1) / self.fanout)
            pad = groups * self.fanout - source.size(1)
            if pad:
                source = torch.cat((source, source.new_zeros(source.size(0), pad, source.size(2), source.size(-1))), dim=1)
                source_valid = torch.cat((source_valid, torch.zeros(source_valid.size(0), pad, source_valid.size(2), device=source.device, dtype=torch.bool)), dim=1)
            source_slots = source.size(2)
            candidates = source.view(source.size(0), groups, self.fanout * source_slots, source.size(-1))
            candidate_valid = source_valid.view(source_valid.size(0), groups, self.fanout * source_slots)
            norms = candidates.square().sum(dim=-1).masked_fill(~candidate_valid, float("-inf"))
            target_slots = min(self.slots, candidates.size(2))
            keep = norms.topk(target_slots, dim=-1).indices
            source = candidates.gather(2, keep.unsqueeze(-1).expand(-1, -1, -1, candidates.size(-1)))
            source_valid = candidate_valid.gather(2, keep)
            levels.append(source)
            valid_levels.append(source_valid)
        self.levels = levels
        self.valid_levels = valid_levels

    def node_scores(self, query: Tensor) -> tuple[Tensor, ...]:
        """All-node max-slot scores; used only for supervised warm-up."""
        return tuple(
            torch.einsum("bd,bnsd->bns", query, level).masked_fill(~valid, float("-inf")).amax(dim=-1) / math.sqrt(query.size(-1))
            for level, valid in zip(self.levels, self.valid_levels)
        )

    def search(self, query: Tensor, *, beam: int, retrieval_pages: int) -> MultiVectorTreeSearch:
        if query.ndim != 2:
            raise ValueError("query must have shape [batch, channels]")
        if retrieval_pages < 1 or retrieval_pages > beam or retrieval_pages > self.page_count:
            raise ValueError("require 1 <= retrieval_pages <= min(beam, page_count)")
        batch = query.size(0)
        current = torch.zeros(batch, 1, device=query.device, dtype=torch.long)
        score_count = 0
        for level in range(len(self.levels) - 2, -1, -1):
            children = self.levels[level]
            children_valid = self.valid_levels[level]
            offsets = torch.arange(self.fanout, device=query.device)
            child_ids = current.unsqueeze(-1) * self.fanout + offsets
            valid = child_ids < children.size(1)
            safe_ids = child_ids.clamp_max(children.size(1) - 1)
            gathered = children.gather(
                1,
                safe_ids.reshape(batch, -1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, children.size(2), children.size(-1)),
            ).view(batch, current.size(1), self.fanout, children.size(2), children.size(-1))
            gathered_valid = children_valid.gather(
                1, safe_ids.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, children.size(2))
            ).view(batch, current.size(1), self.fanout, children.size(2))
            scores = torch.einsum("bd,bnfsd->bnfs", query, gathered).masked_fill(~gathered_valid, float("-inf")).amax(dim=-1) / math.sqrt(query.size(-1))
            valid = valid & gathered_valid.any(dim=-1)
            scores = scores.masked_fill(~valid, float("-inf"))
            flat_scores, flat_ids = scores.flatten(1), child_ids.flatten(1)
            width = min(beam, int(valid.sum(dim=(1, 2)).min().item()))
            current = flat_ids.gather(1, flat_scores.topk(width, dim=-1).indices)
            score_count += int(valid.sum(dim=(1, 2)).max().item()) * children.size(2)
        return MultiVectorTreeSearch(current[:, :retrieval_pages], score_count, self.depth)
