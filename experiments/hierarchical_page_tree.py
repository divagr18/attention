"""Causal, fixed-budget hierarchical routing over completed KV pages.

The tree stores one learned-compatible vector per page and only exposes a
fixed number of children at each level.  It is deliberately independent of a
model: callers own page-summary projections and retain the exact K/V pages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class TreeSearch:
    """Leaves selected by a fixed-beam traversal and its accounting."""

    page_indices: Tensor
    score_count: int
    depth: int


class HierarchicalPageTree:
    """A batched causal page tree with O(beam * fanout * log(pages)) search.

    Pages are appended only after they are complete.  A caller should retain
    the raw K/V pages separately; this index holds only summary vectors.
    """

    def __init__(
        self, *, fanout: int = 16, aggregate: Callable[[Tensor], Tensor] | None = None, max_pages: int | None = None
    ) -> None:
        if fanout < 2:
            raise ValueError("fanout must be at least two")
        self.fanout = fanout
        self.aggregate = aggregate
        self.max_pages = max_pages
        self.levels: list[Tensor] = []  # leaves first, root last
        self.level_counts: list[int] = []

    @classmethod
    def from_page_keys(
        cls, page_keys: Tensor, *, fanout: int = 16, aggregate: Callable[[Tensor], Tensor] | None = None
    ) -> "HierarchicalPageTree":
        if page_keys.ndim != 3:
            raise ValueError("page_keys must have shape [batch, pages, channels]")
        if not page_keys.size(1):
            raise ValueError("at least one completed page is required")
        tree = cls(fanout=fanout, aggregate=aggregate, max_pages=page_keys.size(1))
        tree.levels = [page_keys]
        tree._rebuild_parents()
        tree.level_counts = [level.size(1) for level in tree.levels]
        return tree

    @property
    def page_count(self) -> int:
        return self.level_counts[0] if self.level_counts else 0

    @property
    def depth(self) -> int:
        return max(0, len(self.levels) - 1)

    def append(self, page_key: Tensor) -> None:
        """Append one completed page and refresh only its ancestor path.

        Give the constructor ``max_pages`` for a preallocated causal index;
        that path performs O(log(max_pages)) summary updates without copying
        earlier leaves.  Dynamic construction is retained only as a small
        convenience fallback for tests and offline setup.
        """
        if page_key.ndim == 2:
            page_key = page_key.unsqueeze(1)
        if page_key.ndim != 3 or page_key.size(1) != 1:
            raise ValueError("page_key must have shape [batch, channels] or [batch, 1, channels]")
        if not self.levels and self.max_pages is not None:
            self._allocate(page_key)
        if not self.levels:
            self.levels = [page_key]
            self._rebuild_parents()
            self.level_counts = [level.size(1) for level in self.levels]
            return
        if page_key.shape[0] != self.levels[0].shape[0] or page_key.shape[2] != self.levels[0].shape[2]:
            raise ValueError("page key batch/channels must match the tree")
        if self.max_pages is not None:
            self._append_preallocated(page_key)
            return
        self.levels[0] = torch.cat((self.levels[0], page_key), dim=1)
        self._rebuild_parents()
        self.level_counts = [level.size(1) for level in self.levels]

    def _allocate(self, page_key: Tensor) -> None:
        assert self.max_pages is not None
        if self.max_pages < 1:
            raise ValueError("max_pages must be positive")
        count = self.max_pages
        self.levels = []
        while True:
            self.levels.append(page_key.new_zeros(page_key.size(0), count, page_key.size(2)))
            if count == 1:
                break
            count = math.ceil(count / self.fanout)
        self.level_counts = [0 for _ in self.levels]

    def _aggregate_group(self, group: Tensor) -> Tensor:
        mean = group.mean(dim=1, keepdim=True)
        maximum = group.amax(dim=1, keepdim=True)
        aggregate_input = torch.cat((mean, maximum), dim=-1)
        return self.aggregate(aggregate_input) if self.aggregate is not None else 0.5 * (mean + maximum)

    @torch.no_grad()
    def _append_preallocated(self, page_key: Tensor) -> None:
        assert self.max_pages is not None
        index = self.level_counts[0]
        if index >= self.max_pages:
            raise ValueError("causal page-tree capacity exhausted")
        self.levels[0][:, index : index + 1].copy_(page_key)
        self.level_counts[0] += 1
        child_index = index
        for level in range(1, len(self.levels)):
            parent = child_index // self.fanout
            start = parent * self.fanout
            end = min(start + self.fanout, self.level_counts[level - 1])
            summary = self._aggregate_group(self.levels[level - 1][:, start:end])
            self.levels[level][:, parent : parent + 1].copy_(summary)
            self.level_counts[level] = math.ceil(self.level_counts[level - 1] / self.fanout)
            child_index = parent

    def _rebuild_parents(self) -> None:
        """Build learned-compatible mean/max aggregate inputs bottom-up."""
        source = self.levels[0]
        levels = [source]
        while source.size(1) > 1:
            groups = math.ceil(source.size(1) / self.fanout)
            pad = groups * self.fanout - source.size(1)
            if pad:
                source = torch.cat((source, source.new_zeros(source.size(0), pad, source.size(2))), dim=1)
            grouped = source.view(source.size(0), groups, self.fanout, source.size(2))
            # Mean-plus-max is an inexpensive, stable aggregate.  The model's
            # learned internal projection turns this into the stored node key.
            valid = torch.ones(groups * self.fanout - pad, device=source.device, dtype=torch.bool)
            valid = torch.cat((valid, torch.zeros(pad, device=source.device, dtype=torch.bool))) if pad else valid
            valid = valid.view(groups, self.fanout).unsqueeze(0).unsqueeze(-1)
            count = valid.sum(dim=2).clamp_min(1)
            mean = (grouped * valid).sum(dim=2) / count
            maximum = grouped.masked_fill(~valid, float("-inf")).amax(dim=2)
            aggregate_input = torch.cat((mean, maximum), dim=-1)
            source = self.aggregate(aggregate_input) if self.aggregate is not None else 0.5 * (mean + maximum)
            levels.append(source)
        self.levels = levels

    def search(self, query: Tensor, *, beam: int, retrieval_pages: int) -> TreeSearch:
        """Return leaf pages after a causal fixed-budget top-down traversal."""
        if query.ndim != 2:
            raise ValueError("query must have shape [batch, channels]")
        if not self.levels:
            raise ValueError("cannot search an empty tree")
        if beam < 1 or retrieval_pages < 1 or retrieval_pages > beam or retrieval_pages > self.page_count:
            raise ValueError("require 1 <= retrieval_pages <= min(beam, page_count)")
        if query.shape[0] != self.levels[0].shape[0] or query.shape[1] != self.levels[0].shape[2]:
            raise ValueError("query batch/channels must match tree keys")

        batch = query.size(0)
        current = torch.zeros(batch, 1, device=query.device, dtype=torch.long)  # root index
        score_count = 0
        for level in range(len(self.levels) - 2, -1, -1):
            children = self.levels[level][:, : self.level_counts[level]]
            offsets = torch.arange(self.fanout, device=query.device)
            child_ids = current.unsqueeze(-1) * self.fanout + offsets
            valid = child_ids < children.size(1)
            safe_ids = child_ids.clamp_max(children.size(1) - 1)
            gathered = children.gather(
                1,
                safe_ids.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, children.size(-1)),
            ).view(batch, current.size(1), self.fanout, children.size(-1))
            scores = torch.einsum("bd,bnfd->bnf", query, gathered) / math.sqrt(query.size(-1))
            scores = scores.masked_fill(~valid, float("-inf"))
            flat_scores = scores.flatten(start_dim=1)
            flat_ids = child_ids.flatten(start_dim=1)
            width = min(beam, int(valid.sum(dim=(1, 2)).min().item()))
            selected = flat_scores.topk(width, dim=-1).indices
            current = flat_ids.gather(1, selected)
            score_count += int(valid.sum(dim=(1, 2)).max().item())
        # Sort selected leaves by their final traversal score only indirectly;
        # the traversal's fixed beam is the intentional approximate index.
        return TreeSearch(page_indices=current[:, :retrieval_pages], score_count=score_count, depth=self.depth)
