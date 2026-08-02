"""Fused beam-1 traversal for the fixed-slot causal page tree.

The first kernel targets the 128K benchmark layout: leaf pages, 16-way parent
nodes, 16-way grandparent nodes, then one root.  It performs all three routing
levels inside one Triton program per batch item, avoiding Python dispatches
and GPU synchronizations between levels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def tree_route_3level_kernel(
    query_ptr, leaf_ptr, parent_ptr, grand_ptr,
    leaf_valid_ptr, parent_valid_ptr, grand_valid_ptr, output_ptr,
    page_count: tl.constexpr, parent_count: tl.constexpr, grand_count: tl.constexpr,
    inner_slots: tl.constexpr, head_dim: tl.constexpr, block_dim: tl.constexpr,
):
    batch = tl.program_id(axis=0)
    dims = tl.arange(0, block_dim)
    query = tl.load(query_ptr + batch * head_dim + dims, mask=dims < head_dim, other=0.0).to(tl.float32)

    # Root -> grandparent.  There are at most 16 children, each with four
    # retained slots, so a single 64-element score vector is sufficient.
    offsets = tl.arange(0, 64)
    grand_nodes = offsets // inner_slots
    grand_slots = offsets % inner_slots
    grand_valid = (grand_nodes < grand_count) & tl.load(
        grand_valid_ptr + (batch * grand_count + grand_nodes) * inner_slots + grand_slots,
        mask=grand_nodes < grand_count,
        other=0,
    ).to(tl.int1)
    grand_base = ((batch * grand_count + grand_nodes) * inner_slots + grand_slots) * head_dim
    grand_keys = tl.load(grand_ptr + grand_base[:, None] + dims[None, :], mask=grand_valid[:, None] & (dims[None, :] < head_dim), other=0.0).to(tl.float32)
    grand_scores = tl.sum(grand_keys * query[None, :], axis=1)
    grand_scores = tl.where(grand_valid, grand_scores, -float("inf"))
    best_grand = tl.argmax(grand_scores, axis=0)
    best_grand_node = best_grand // inner_slots

    # Grandparent -> parent.
    parent_nodes = best_grand_node * 16 + offsets // inner_slots
    parent_slots = offsets % inner_slots
    parent_valid = (parent_nodes < parent_count) & tl.load(
        parent_valid_ptr + (batch * parent_count + parent_nodes) * inner_slots + parent_slots,
        mask=parent_nodes < parent_count,
        other=0,
    ).to(tl.int1)
    parent_base = ((batch * parent_count + parent_nodes) * inner_slots + parent_slots) * head_dim
    parent_keys = tl.load(parent_ptr + parent_base[:, None] + dims[None, :], mask=parent_valid[:, None] & (dims[None, :] < head_dim), other=0.0).to(tl.float32)
    parent_scores = tl.sum(parent_keys * query[None, :], axis=1)
    parent_scores = tl.where(parent_valid, parent_scores, -float("inf"))
    best_parent = tl.argmax(parent_scores, axis=0)
    best_parent_node = best_grand_node * 16 + best_parent // inner_slots

    # Parent -> leaf page.  Leaves use one slot, so only the first 16 lanes
    # are active; the remaining lanes keep the block shape regular.
    leaf_nodes = best_parent_node * 16 + offsets
    leaf_valid = (offsets < 16) & (leaf_nodes < page_count) & tl.load(
        leaf_valid_ptr + batch * page_count + leaf_nodes,
        mask=(offsets < 16) & (leaf_nodes < page_count),
        other=0,
    ).to(tl.int1)
    leaf_base = (batch * page_count + leaf_nodes) * head_dim
    leaf_keys = tl.load(leaf_ptr + leaf_base[:, None] + dims[None, :], mask=leaf_valid[:, None] & (dims[None, :] < head_dim), other=0.0).to(tl.float32)
    leaf_scores = tl.sum(leaf_keys * query[None, :], axis=1)
    leaf_scores = tl.where(leaf_valid, leaf_scores, -float("inf"))
    best_leaf = tl.argmax(leaf_scores, axis=0)
    tl.store(output_ptr + batch, best_parent_node * 16 + best_leaf)


def tree_route_3level(
    query: torch.Tensor,
    leaves: torch.Tensor,
    parents: torch.Tensor,
    grandparents: torch.Tensor,
    leaf_valid: torch.Tensor,
    parent_valid: torch.Tensor,
    grand_valid: torch.Tensor,
) -> torch.Tensor:
    """Return one page ID per batch from tree levels [leaf,parent,grand]."""
    if not all(t.is_cuda and t.is_contiguous() for t in (query, leaves, parents, grandparents, leaf_valid, parent_valid, grand_valid)):
        raise ValueError("contiguous CUDA tensors are required")
    batch, page_count, leaf_slots, head_dim = leaves.shape
    if query.shape != (batch, head_dim) or leaf_slots != 1 or parents.size(2) != grandparents.size(2):
        raise ValueError("unsupported tree layout")
    parent_count, grand_count, inner_slots = parents.size(1), grandparents.size(1), parents.size(2)
    if inner_slots > 4 or grand_count > 16 or parent_count > 256 or head_dim > 64:
        raise ValueError("tree exceeds first fused-kernel limits")
    output = torch.empty(batch, device=query.device, dtype=torch.long)
    tree_route_3level_kernel[(batch,)](
        query, leaves, parents, grandparents,
        leaf_valid, parent_valid, grand_valid, output,
        page_count=page_count, parent_count=parent_count, grand_count=grand_count,
        inner_slots=inner_slots, head_dim=head_dim, block_dim=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output
