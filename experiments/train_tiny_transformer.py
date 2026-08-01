#!/usr/bin/env python3
"""Train a tiny causal Transformer on a delayed exact-retrieval task.

The three modes intentionally share all model code and training data:
  dense   : full causal attention
  sliding : fixed causal local window
  oracle  : local window plus the labelled historical evidence tokens

This is the first architecture test.  It answers whether exact attention over
local context plus a small, correct historical candidate set is sufficient
before a learned hierarchical router is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


KEY_COUNT = 64
KEY_START = 1
VALUE_START = KEY_START + KEY_COUNT
QUERY_TOKEN = VALUE_START + KEY_COUNT
SEPARATOR_TOKEN = QUERY_TOKEN + 1
LATEST_TOKEN = SEPARATOR_TOKEN + 1
CANONICAL_TOKEN = LATEST_TOKEN + 1
DECOY_TOKEN = CANONICAL_TOKEN + 1
NOISE_START = DECOY_TOKEN + 1
VOCAB_SIZE = 256


@dataclass
class Config:
    variant: str
    context: int
    local_window: int
    batch_size: int
    steps: int
    eval_batches: int
    d_model: int
    layers: int
    heads: int
    learning_rate: float
    block_size: int
    top_blocks: int
    top_tokens: int
    router_loss_weight: float
    task_family: str
    seed: int
    device: str
    retrieval_unit: str = "span"
    retrieval_width: int = 3


def make_batch(config: Config, device: torch.device, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """Return input tokens, answer targets, and labelled evidence positions."""
    if config.context <= config.local_window + 8:
        raise ValueError("context must exceed local_window by at least eight tokens")
    tokens = torch.randint(
        NOISE_START,
        VOCAB_SIZE,
        (config.batch_size, config.context),
        generator=generator,
        device=device,
    )
    keys = torch.randint(KEY_START, KEY_START + KEY_COUNT, (config.batch_size,), generator=generator, device=device)
    values = torch.randint(VALUE_START, VALUE_START + KEY_COUNT, (config.batch_size,), generator=generator, device=device)
    historical_length = config.context - config.local_window
    # Keep every evidence span outside the local window of the final query.
    positions = torch.randint(0, historical_length - 3, (config.batch_size,), generator=generator, device=device)
    for row in range(config.batch_size):
        position = int(positions[row])
        key = int(keys[row])
        value = int(values[row])
        family = config.task_family
        if family == "mixed":
            family = ("single", "overwrite", "distractor")[row % 3]
        if family == "multirecord":
            # Four independent records emulate a persistent context with
            # multiple possible promoted pages.  The query chooses one.
            record_positions = torch.linspace(4, historical_length - 4, 4, device=device).round().long()
            record_keys = torch.randperm(KEY_COUNT, generator=generator, device=device)[:4] + KEY_START
            record_values = torch.randint(VALUE_START, VALUE_START + KEY_COUNT, (4,), generator=generator, device=device)
            chosen = int(torch.randint(0, 4, (1,), generator=generator, device=device))
            for record_position, record_key, record_value in zip(record_positions.tolist(), record_keys.tolist(), record_values.tolist()):
                tokens[row, record_position : record_position + 3] = torch.tensor([record_key, SEPARATOR_TOKEN, record_value], device=device)
            positions[row] = record_positions[chosen]
            tokens[row, -3:] = torch.tensor([QUERY_TOKEN, SEPARATOR_TOKEN, record_keys[chosen]], device=device)
            values[row] = record_values[chosen]
            continue
        marker = SEPARATOR_TOKEN
        if family == "overwrite":
            marker = LATEST_TOKEN
            old_position = (position + historical_length // 2) % (historical_length - 3)
            tokens[row, old_position : old_position + 3] = torch.tensor([key, SEPARATOR_TOKEN, torch.randint(VALUE_START, VALUE_START + KEY_COUNT, (1,), generator=generator, device=device).item()], device=device)
        elif family == "distractor":
            marker = CANONICAL_TOKEN
            for offset in (historical_length // 3, 2 * historical_length // 3):
                decoy_position = (position + offset) % (historical_length - 3)
                tokens[row, decoy_position : decoy_position + 3] = torch.tensor([key, DECOY_TOKEN, torch.randint(VALUE_START, VALUE_START + KEY_COUNT, (1,), generator=generator, device=device).item()], device=device)
        tokens[row, position : position + 3] = torch.tensor([key, marker, value], device=device)
        tokens[row, -3:] = torch.tensor([QUERY_TOKEN, marker, key], device=device)
    return tokens, values, positions


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, *, variant: str, local_window: int, evidence_positions: Tensor | None, retrieved_indices: Tensor | None, capture_attention: bool = False) -> tuple[Tensor, Tensor | None]:
        batch_size, length, channels = x.shape
        qkv = self.qkv(x).view(batch_size, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(length, device=x.device)
        causal = positions[:, None] >= positions[None, :]
        if variant == "dense":
            allowed = causal.unsqueeze(0).expand(batch_size, -1, -1).clone()
        else:
            local = positions[:, None] - positions[None, :] < local_window
            allowed = (causal & local).unsqueeze(0).expand(batch_size, -1, -1).clone()
            if variant == "oracle":
                if evidence_positions is None:
                    raise ValueError("oracle attention needs evidence positions")
                batch_indices = torch.arange(batch_size, device=x.device)
                # The final query can recover the exact key, separator, and value tokens.
                for offset in range(3):
                    allowed[batch_indices, length - 1, evidence_positions + offset] = True
            if variant == "learned":
                if retrieved_indices is None:
                    raise ValueError("learned attention needs retrieved indices")
                batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
                allowed[batch_indices, length - 1, retrieved_indices] = True
        scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
        attention = self.dropout(scores.softmax(dim=-1))
        output = attention @ value
        return self.output(output.transpose(1, 2).contiguous().view(batch_size, length, channels)), attention if capture_attention else None


class Block(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalAttention(d_model, heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor, *, variant: str, local_window: int, evidence_positions: Tensor | None, retrieved_indices: Tensor | None, capture_attention: bool = False) -> tuple[Tensor, Tensor | None]:
        attention_output, attention = self.attention(self.norm1(x), variant=variant, local_window=local_window, evidence_positions=evidence_positions, retrieved_indices=retrieved_indices, capture_attention=capture_attention)
        x = x + attention_output
        return x + self.mlp(self.norm2(x)), attention


@dataclass
class RouterOutput:
    block_scores: Tensor
    token_scores: Tensor
    retrieved_indices: Tensor


class HierarchicalRouter(nn.Module):
    """Coarse block scoring followed by fine token scoring within top blocks."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)

    def forward(self, embeddings: Tensor, *, historical_length: int, block_size: int, top_blocks: int, top_tokens: int, retrieval_unit: str = "span", retrieval_width: int = 3) -> RouterOutput:
        if historical_length % block_size:
            raise ValueError("historical length must divide evenly into block_size")
        # Score record starts: include each key's following structural marker.
        record_embeddings = embeddings[:, :historical_length].clone()
        record_embeddings[:, :-1] = record_embeddings[:, :-1] + embeddings[:, 1:historical_length]
        query = self.query(embeddings[:, -3:].mean(dim=1))
        keys = self.key(record_embeddings)
        token_scores = torch.einsum("bd,btd->bt", query, keys) / math.sqrt(query.size(-1))
        block_scores = token_scores.view(token_scores.size(0), historical_length // block_size, block_size).amax(dim=-1)
        selected_blocks = block_scores.topk(top_blocks, dim=-1).indices
        offsets = torch.arange(block_size, device=embeddings.device)
        candidates = (selected_blocks.unsqueeze(-1) * block_size + offsets).flatten(start_dim=1)
        candidate_scores = token_scores.gather(1, candidates)
        centers = candidates.gather(1, candidate_scores.topk(top_tokens, dim=-1).indices)
        if retrieval_unit == "span":
            # A retrieved key token brings its short exact record with it.
            record_offsets = torch.arange(retrieval_width, device=embeddings.device)
            retrieved = (centers.unsqueeze(-1) + record_offsets).clamp_max(historical_length - 1).flatten(start_dim=1)
        elif retrieval_unit == "page":
            # The full selected page matches the paged Triton gather kernel.
            page_offsets = torch.arange(block_size, device=embeddings.device)
            retrieved = (selected_blocks.unsqueeze(-1) * block_size + page_offsets).flatten(start_dim=1)
        elif retrieval_unit == "page_fine":
            # Coarse page selection followed by short exact spans within the
            # selected page.  This retains page-level routing while avoiding
            # full-page attention normalization over irrelevant tokens.
            span_offsets = torch.arange(retrieval_width, device=embeddings.device)
            retrieved = (centers.unsqueeze(-1) + span_offsets).clamp_max(historical_length - 1).flatten(start_dim=1)
        else:
            raise ValueError(f"unknown retrieval unit: {retrieval_unit}")
        return RouterOutput(block_scores=block_scores, token_scores=token_scores, retrieved_indices=retrieved)


class TinyRetrievalTransformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        self.blocks = nn.ModuleList(Block(config.d_model, config.heads) for _ in range(config.layers))
        self.router = HierarchicalRouter(config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, VOCAB_SIZE, bias=False)
        self.context = config.context
        self.retrieval_unit = config.retrieval_unit
        self.retrieval_width = config.retrieval_width

    def forward(self, tokens: Tensor, *, variant: str, local_window: int, evidence_positions: Tensor | None, block_size: int | None = None, top_blocks: int | None = None, top_tokens: int | None = None, capture_attention: bool = False, retrieved_indices_override: Tensor | None = None) -> tuple[Tensor, RouterOutput | None, Tensor | None]:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        token_embeddings = self.token_embedding(tokens)
        routing = None
        retrieved_indices = None
        if variant == "learned":
            if block_size is None or top_blocks is None or top_tokens is None:
                raise ValueError("learned attention needs router configuration")
            if retrieved_indices_override is None:
                routing = self.router(token_embeddings, historical_length=tokens.size(1) - local_window, block_size=block_size, top_blocks=top_blocks, top_tokens=top_tokens, retrieval_unit=self.retrieval_unit, retrieval_width=self.retrieval_width)
                retrieved_indices = routing.retrieved_indices
            else:
                retrieved_indices = retrieved_indices_override
        x = token_embeddings + self.position_embedding(positions)
        captured_attention = None
        for index, block in enumerate(self.blocks):
            x, attention = block(x, variant=variant, local_window=local_window, evidence_positions=evidence_positions, retrieved_indices=retrieved_indices, capture_attention=capture_attention and index == len(self.blocks) - 1)
            if attention is not None:
                captured_attention = attention
        return self.output(self.norm(x)), routing, captured_attention


@torch.no_grad()
def evaluate(model: TinyRetrievalTransformer, config: Config, device: torch.device, generator: torch.Generator) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    router_token_hits = 0
    router_block_hits = 0
    by_bucket: dict[str, list[int]] = {"near": [0, 0], "medium": [0, 0], "far": [0, 0]}
    for _ in range(config.eval_batches):
        tokens, targets, evidence_positions = make_batch(config, device, generator)
        logits, routing, _ = model(tokens, variant=config.variant, local_window=config.local_window, evidence_positions=evidence_positions, block_size=config.block_size, top_blocks=config.top_blocks, top_tokens=config.top_tokens)
        predictions = logits[:, -1, VALUE_START : VALUE_START + KEY_COUNT].argmax(dim=-1) + VALUE_START
        matches = predictions.eq(targets)
        if routing is not None:
            router_token_hits += int(routing.retrieved_indices.eq(evidence_positions.unsqueeze(1)).any(dim=1).sum())
            retrieved_blocks = routing.retrieved_indices // config.block_size
            router_block_hits += int(retrieved_blocks.eq((evidence_positions // config.block_size).unsqueeze(1)).any(dim=1).sum())
        distances = config.context - evidence_positions
        for name, mask in (("near", distances < config.context // 3), ("medium", (distances >= config.context // 3) & (distances < 2 * config.context // 3)), ("far", distances >= 2 * config.context // 3)):
            by_bucket[name][0] += int(matches[mask].sum())
            by_bucket[name][1] += int(mask.sum())
        correct += int(matches.sum())
        total += config.batch_size
    metrics = {"accuracy": correct / total}
    metrics.update({f"accuracy_{name}": successes / count if count else 0.0 for name, (successes, count) in by_bucket.items()})
    if config.variant == "learned":
        metrics["router_token_recall"] = router_token_hits / total
        metrics["router_block_recall"] = router_block_hits / total
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("dense", "sliding", "oracle", "learned"), required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--top-blocks", type=int, default=1)
    parser.add_argument("--top-tokens", type=int, default=1)
    parser.add_argument("--router-loss-weight", type=float, default=1.0)
    parser.add_argument("--retrieval-unit", choices=("span", "page", "page_fine"), default="span")
    parser.add_argument("--retrieval-width", type=int, default=3, help="Promoted token count for span retrieval.")
    parser.add_argument("--task-family", choices=("single", "overwrite", "distractor", "mixed", "multirecord"), default="single")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize model weights before training (for retrieval-unit curricula).")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-router-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-target", choices=("raw_token", "record_span"), default="raw_token")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    if config.variant == "learned" and (config.context - config.local_window) % config.block_size:
        parser.error("context - local-window must divide evenly into block-size")
    device = torch.device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    model = TinyRetrievalTransformer(config).to(device)
    if args.init_checkpoint is not None:
        initialization = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        initial_state = initialization["model_state"]
        position_key = "position_embedding.weight"
        if initial_state[position_key].shape != model.state_dict()[position_key].shape:
            # Preserve the learned positional trend when extending a context
            # curriculum (for example 8K -> 16K), then let fine-tuning adapt.
            source = initial_state[position_key].T.unsqueeze(0)
            initial_state[position_key] = F.interpolate(
                source,
                size=config.context,
                mode="linear",
                align_corners=True,
            ).squeeze(0).T
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    teacher = None
    if args.teacher_checkpoint is not None:
        checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=True)
        teacher = TinyRetrievalTransformer(config).to(device)
        teacher.load_state_dict(checkpoint["model_state"])
        teacher.eval()
    start = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, config.steps + 1):
        tokens, targets, evidence_positions = make_batch(config, device, generator)
        logits, routing, _ = model(tokens, variant=config.variant, local_window=config.local_window, evidence_positions=evidence_positions, block_size=config.block_size, top_blocks=config.top_blocks, top_tokens=config.top_tokens)
        answer_logits = logits[:, -1, VALUE_START : VALUE_START + KEY_COUNT]
        loss = F.cross_entropy(answer_logits, targets - VALUE_START)
        if routing is not None:
            if teacher is None:
                block_targets = evidence_positions // config.block_size
                router_loss = F.cross_entropy(routing.block_scores, block_targets) + F.cross_entropy(routing.token_scores, evidence_positions)
                loss = loss + config.router_loss_weight * router_loss
            else:
                with torch.no_grad():
                    _, _, teacher_attention = teacher(tokens, variant="dense", local_window=config.local_window, evidence_positions=None, capture_attention=True)
                    historical_attention = teacher_attention.mean(dim=1)[:, -1, : config.context - config.local_window]
                    if args.teacher_target == "record_span":
                        # Each router candidate denotes a record start and
                        # promotes that start plus its next two exact tokens.
                        # Distil the teacher's importance for that same unit.
                        span_attention = historical_attention.clone()
                        span_attention[:, :-1] = span_attention[:, :-1] + historical_attention[:, 1:]
                        span_attention[:, :-2] = span_attention[:, :-2] + historical_attention[:, 2:]
                        historical_attention = span_attention
                    token_target = historical_attention / historical_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    block_target = token_target.view(config.batch_size, -1, config.block_size).sum(dim=-1)
                router_loss = F.kl_div(F.log_softmax(routing.token_scores, dim=-1), token_target, reduction="batchmean") + F.kl_div(F.log_softmax(routing.block_scores, dim=-1), block_target, reduction="batchmean")
                loss = loss + args.teacher_router_loss_weight * router_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, config.steps // 10) == 0:
            history.append({"step": step, "loss": float(loss.detach())})
            print(f"step={step} loss={loss.detach().item():.4f}", flush=True)
    elapsed_seconds = time.perf_counter() - start
    metrics = evaluate(model, config, device, generator)
    report = {"config": asdict(config), "parameters": sum(parameter.numel() for parameter in model.parameters()), "elapsed_seconds": elapsed_seconds, "history": history, "evaluation": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": asdict(config)}, args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
