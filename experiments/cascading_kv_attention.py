"""Model-neutral exact K/V attention core for a hierarchical page index.

This module is the integration boundary for Qwen's periodic Gated Attention
layers: the model supplies projected Q/K/V tensors and learned page-summary
keys, while this core owns causal tree search, contiguous page gathering, and
the one normalized attention operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hierarchical_page_tree import HierarchicalPageTree, TreeSearch


@dataclass(frozen=True)
class CascadingAttentionConfig:
    hot_window: int = 8192
    page_size: int = 256
    tree_fanout: int = 16
    tree_beam: int = 4
    retrieval_pages: int = 4
    historical_store: str = "bf16"
    routing_rotary_dim: int = 0  # zero this many leading RoPE dims in routing for length-invariant content routing


class CascadingKVAttention(nn.Module):
    """Shared-page, unified-softmax attention over local plus retrieved K/V."""

    def __init__(self, channels: int, config: CascadingAttentionConfig) -> None:
        super().__init__()
        if config.historical_store != "bf16":
            raise ValueError("v1 retains exact BF16 historical K/V; compression is intentionally out of scope")
        if config.retrieval_pages > config.tree_beam:
            raise ValueError("retrieval_pages must be no greater than tree_beam")
        self.config = config
        self.page_summary = nn.Linear(2 * channels, channels, bias=False)
        self.internal_summary = nn.Linear(2 * channels, channels, bias=False)
        self.query_proj = nn.Linear(channels, channels, bias=False)

    def make_tree(self, router_keys: Tensor) -> HierarchicalPageTree:
        """Create an offline/static tree from [batch, pages, channels] keys.

        Autoregressive runtimes should instead allocate a
        ``HierarchicalPageTree(max_pages=...)`` and append completed pages as
        they cross the hot-window boundary.
        """
        return HierarchicalPageTree.from_page_keys(router_keys, fanout=self.config.tree_fanout, aggregate=self.internal_summary)

    def forward(self, query: Tensor, keys: Tensor, values: Tensor, tree: HierarchicalPageTree | None = None, force_page_indices: Tensor | None = None, page_count: int | None = None) -> tuple[Tensor, TreeSearch | None]:
        """Attend exactly over local K/V plus selected historical pages.

        Shapes are ``query=[B,H,D]`` and ``keys/values=[B,H,L,D]``.  The paged
        prefix length is ``page_count * page_size`` (explicit ``page_count``, or
        ``tree.page_count`` when omitted); everything after it is
        the local region (at least ``hot_window`` tokens once the caller appends
        pages as they leave the window).  With no completed pages the core is a
        single dense softmax over the full sequence.  ``force_page_indices``
        ``[B,n]`` overrides the tree search and gathers those pages directly
        (oracle retrieval).  Router keys share D and come from the host layer.
        """
        if keys.shape != values.shape or query.ndim != 3 or keys.ndim != 4:
            raise ValueError("expected query [B,H,D] and equal keys/values [B,H,L,D]")
        length = keys.size(2)
        if page_count is None:
            page_count = 0 if tree is None else tree.page_count
        paged = page_count * self.config.page_size
        if paged > length:
            raise ValueError("tree pages exceed available K/V length")
        local_k = keys[:, :, paged:]
        local_v = values[:, :, paged:]
        search = None
        page_indices = force_page_indices
        if page_indices is None and page_count:
            search = tree.search(self.query_proj(query.mean(dim=1)), beam=self.config.tree_beam, retrieval_pages=self.config.retrieval_pages)
            page_indices = search.page_indices
        if page_indices is not None:
            if paged == 0:
                raise ValueError("force_page_indices requires a non-empty paged prefix")
            key_pages = keys[:, :, :paged].view(keys.size(0), keys.size(1), page_count, self.config.page_size, keys.size(-1)).permute(0, 2, 1, 3, 4)
            value_pages = values[:, :, :paged].view(values.size(0), values.size(1), page_count, self.config.page_size, values.size(-1)).permute(0, 2, 1, 3, 4)
            batch = torch.arange(keys.size(0), device=keys.device).unsqueeze(1)
            retrieved_keys = key_pages[batch, page_indices].permute(0, 2, 1, 3, 4).flatten(start_dim=2, end_dim=3)
            retrieved_values = value_pages[batch, page_indices].permute(0, 2, 1, 3, 4).flatten(start_dim=2, end_dim=3)
            candidates_k = torch.cat((local_k, retrieved_keys), dim=2)
            candidates_v = torch.cat((local_v, retrieved_values), dim=2)
        else:
            candidates_k, candidates_v = local_k, local_v
        # Unified softmax in fp32 to match dense eager numerics; cast back after.
        scale = math.sqrt(query.size(-1))
        if query.size(1) == candidates_k.size(1):
            scores = torch.einsum("bhd,bhld->bhl", query.float(), candidates_k.float()) / scale
            output = torch.einsum("bhl,bhld->bhd", scores.softmax(dim=-1), candidates_v.float())
        else:
            # GQA-native: group query heads by KV head so decode never expands
            # the full cache (repeat_interleave per step was the decode bottleneck).
            grouped = query.view(candidates_k.size(0), candidates_k.size(1), query.size(1) // candidates_k.size(1), query.size(-1))
            scores = torch.einsum("bhgd,bhld->bhgl", grouped.float(), candidates_k.float()) / scale
            output = torch.einsum("bhgl,bhld->bhgd", scores.softmax(dim=-1), candidates_v.float())
            output = output.flatten(1, 2)
        return output.to(query.dtype), search
