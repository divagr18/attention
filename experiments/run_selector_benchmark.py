#!/usr/bin/env python3
"""Deterministic synthetic benchmark for cascading-attention selectors."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Case:
    family: str
    query_terms: tuple[str, ...]
    tokens: tuple[str, ...]
    evidence_indices: tuple[int, ...]
    evidence_block: int
    distance: int


def distance_bucket(distance: int) -> str:
    if distance < 512:
        return "near"
    if distance < 2_048:
        return "medium"
    return "far"


def make_case(rng: random.Random, *, context_tokens: int, block_size: int) -> Case:
    families = ("passkey", "overwrite", "distractor", "composition")
    family = rng.choice(families)
    block_count = context_tokens // block_size
    evidence_block = rng.randrange(0, block_count - 2)
    evidence_start = evidence_block * block_size + rng.randrange(8, block_size - 8)
    tokens = [f"noise_{rng.randrange(10_000)}" for _ in range(context_tokens)]
    label = f"item_{rng.randrange(1_000_000)}"
    value = uuid.UUID(int=rng.getrandbits(128)).hex[:16]
    query_terms = (label,)

    if family == "passkey":
        evidence = (label, "value", value)
        answer_terms = evidence
    elif family == "overwrite":
        old_value = uuid.UUID(int=rng.getrandbits(128)).hex[:16]
        tokens[evidence_start : evidence_start + 3] = [label, "value", old_value]
        evidence_start = min(context_tokens - 12, evidence_start + block_size // 2)
        evidence = (label, "updated_value", value)
        answer_terms = evidence
    elif family == "distractor":
        for _ in range(3):
            start = rng.randrange(context_tokens - 4)
            if abs(start - evidence_start) > 8:
                tokens[start : start + 3] = [label, "value", uuid.UUID(int=rng.getrandbits(128)).hex[:16]]
        evidence = (label, "canonical_value", value)
        answer_terms = evidence
    else:
        rule = f"rule_{rng.randrange(10_000)}"
        evidence = (rule, "maps", label, "to", value)
        query_terms = (rule, label)
        answer_terms = evidence

    tokens[evidence_start : evidence_start + len(evidence)] = evidence
    evidence_indices = tuple(range(evidence_start, evidence_start + len(answer_terms)))
    distance = context_tokens - evidence_start
    return Case(
        family=family,
        query_terms=query_terms,
        tokens=tuple(tokens),
        evidence_indices=evidence_indices,
        evidence_block=evidence_start // block_size,
        distance=distance,
    )


def blocks(token_count: int, block_size: int) -> Iterable[range]:
    for start in range(0, token_count, block_size):
        yield range(start, min(start + block_size, token_count))


def lexical_score(query_terms: tuple[str, ...], items: Iterable[str]) -> int:
    joined = set(items)
    return sum(term in joined for term in query_terms)


def top_indices(scores: list[int], count: int, rng: random.Random | None = None) -> list[int]:
    order = list(range(len(scores)))
    if rng is not None:
        rng.shuffle(order)
    return sorted(order, key=lambda index: scores[index], reverse=True)[:count]


def select(case: Case, *, strategy: str, block_size: int, top_blocks: int, top_tokens: int, rng: random.Random) -> tuple[set[int], int]:
    all_blocks = list(blocks(len(case.tokens), block_size))
    if strategy == "oracle":
        return set(case.evidence_indices), 0
    if strategy == "random":
        selected_blocks = rng.sample(range(len(all_blocks)), min(top_blocks, len(all_blocks)))
        candidates = [index for block in selected_blocks for index in all_blocks[block]]
        return set(candidates[:top_tokens]), len(all_blocks)
    if strategy == "recency":
        selected_blocks = list(range(max(0, len(all_blocks) - top_blocks), len(all_blocks)))
        candidates = [index for block in selected_blocks for index in all_blocks[block]]
        return set(candidates[-top_tokens:]), len(all_blocks)
    if strategy == "flat":
        scores = [lexical_score(case.query_terms, (token,)) for token in case.tokens]
        return set(top_indices(scores, top_tokens, rng)), len(case.tokens)
    if strategy != "hierarchical":
        raise ValueError(f"unknown strategy: {strategy}")
    block_scores = [lexical_score(case.query_terms, (case.tokens[index] for index in block)) for block in all_blocks]
    selected_blocks = top_indices(block_scores, top_blocks, rng)
    candidates = [index for block in selected_blocks for index in all_blocks[block]]
    token_scores = [lexical_score(case.query_terms, (case.tokens[index],)) for index in candidates]
    selected = [candidates[index] for index in top_indices(token_scores, top_tokens, rng)]
    return set(selected), len(all_blocks) + len(candidates)


def evaluate(cases: list[Case], *, strategy: str, block_size: int, top_blocks: int, top_tokens: int, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for case in cases:
        selected, score_count = select(case, strategy=strategy, block_size=block_size, top_blocks=top_blocks, top_tokens=top_tokens, rng=rng)
        evidence = set(case.evidence_indices)
        rows.append({
            "family": case.family,
            "distance_bucket": distance_bucket(case.distance),
            "token_recall": len(selected & evidence) / len(evidence),
            "block_recall": float(any(index // block_size == case.evidence_block for index in selected)),
            "scores_evaluated": score_count,
        })
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped["overall"].append(row)
        grouped[f"family:{row['family']}"].append(row)
        grouped[f"distance:{row['distance_bucket']}"].append(row)
    return {
        "strategy": strategy,
        "overall": {
            "cases": len(rows),
            "token_recall": statistics.mean(float(row["token_recall"]) for row in rows),
            "block_recall": statistics.mean(float(row["block_recall"]) for row in rows),
            "mean_scores_evaluated": statistics.mean(int(row["scores_evaluated"]) for row in rows),
        },
        "slices": {
            name: {
                "cases": len(items),
                "token_recall": statistics.mean(float(item["token_recall"]) for item in items),
                "block_recall": statistics.mean(float(item["block_recall"]) for item in items),
            }
            for name, items in sorted(grouped.items()) if name != "overall"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--top-blocks", type=int, default=4)
    parser.add_argument("--top-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/selector_smoke.json"))
    args = parser.parse_args()
    if args.context_tokens < args.block_size * 3:
        parser.error("context-tokens must contain at least three blocks")
    rng = random.Random(args.seed)
    cases = [make_case(rng, context_tokens=args.context_tokens, block_size=args.block_size) for _ in range(args.cases)]
    report = {
        "config": {
            "cases": args.cases,
            "context_tokens": args.context_tokens,
            "block_size": args.block_size,
            "top_blocks": args.top_blocks,
            "top_tokens": args.top_tokens,
            "seed": args.seed,
        },
        "results": [
            evaluate(cases, strategy=strategy, block_size=args.block_size, top_blocks=args.top_blocks, top_tokens=args.top_tokens, seed=args.seed)
            for strategy in ("random", "recency", "flat", "hierarchical", "oracle")
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
