"""
Dataset construction pipeline for tensor-instruct.

Responsibility
--------------
Given one or more ChatML data sources (LocalSource / HubSource) and a
tokenizer, produce a HuggingFace ``datasets.Dataset`` with ``input_ids``
and ``labels`` columns ready for causal language model instruct fine-tuning.

Loss masking
------------
Labels are set to -100 (ignored by cross-entropy) on every token that is
NOT part of an assistant turn. The model only learns to produce assistant
responses — it never receives a gradient signal from system or user tokens.

ChatML format expected per example
-----------------------------------
    {"messages": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is gradient descent?"},
        {"role": "assistant", "content": "Gradient descent is …"}
    ]}

System turns are optional. Multi-turn conversations are fully supported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

# Tokens that delimit ChatML turns.
_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"

# Roles that receive gradient signal (loss computed on their output).
_TRAIN_ROLES: frozenset[str] = frozenset({"assistant"})

_IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_instruct_dataset(
    sources: dict,
    tokenizer,
    context_length: int,
    seed: int = 42,
):
    """
    Build and return a ``datasets.Dataset`` of tokenised, loss-masked examples.

    Parameters
    ----------
    sources:
        A ``dict[source, weight]`` mapping (already normalised by Mix).
        Weights influence how many examples are sampled from each source
        relative to the others.
    tokenizer:
        A HuggingFace tokeniser.
    context_length:
        Maximum tokens per example.  Longer conversations are truncated.
    seed:
        Shuffle seed.

    Returns
    -------
    dataset:
        HuggingFace ``Dataset`` with ``input_ids`` and ``labels`` columns.
    stats:
        Dict with ``total``, ``kept``, and ``truncated`` example counts.
    """
    import datasets as hf_datasets
    import random

    print("[tensor-instruct] Building tokenised dataset …")

    all_input_ids: list[list[int]] = []
    all_labels:    list[list[int]] = []

    stats = {"total": 0, "kept": 0, "truncated": 0, "skipped": 0}

    # Collect all examples across sources, weighted by sample count.
    # For instruct fine-tuning we work example-by-example rather than
    # streaming a flat token array.
    source_batches: list[list[dict]] = []
    weights: list[float] = []

    for source, weight in sources.items():
        examples = list(_iter_source_examples(source))
        source_batches.append(examples)
        weights.append(weight)
        print(f"  → {source!r}  ({len(examples):,} examples, weight {weight:.2f})")

    # Interleave sources according to their weights.
    merged = _weighted_merge(source_batches, weights, seed=seed)

    for example in tqdm(merged, desc="  Tokenising", unit="ex", leave=False):
        stats["total"] += 1
        messages = example.get("messages")

        if not messages or not isinstance(messages, list):
            stats["skipped"] += 1
            continue

        result = _tokenize_conversation(messages, tokenizer, context_length)
        if result is None:
            stats["skipped"] += 1
            continue

        if result["truncated"]:
            stats["truncated"] += 1

        all_input_ids.append(result["input_ids"])
        all_labels.append(result["labels"])
        stats["kept"] += 1

    if not all_input_ids:
        raise RuntimeError(
            "No valid training examples were produced from the provided sources.\n"
            "Check that your JSONL files contain the expected "
            '{"messages": [{...}]} schema.'
        )

    print(
        f"  ✓ {stats['kept']:,} examples kept  "
        f"({stats['truncated']:,} truncated, {stats['skipped']:,} skipped)"
    )

    dataset = hf_datasets.Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels":    all_labels,
    })

    return dataset, stats


def estimate_example_count(sources: dict) -> int:
    """
    Fast example count across all LocalSource entries.
    HubSource entries return 0 (streaming — count unknown without loading).
    """
    from tensor.instruct.data._sources import LocalSource

    total = 0
    for source in sources:
        if isinstance(source, LocalSource):
            total += sum(1 for _ in source.iter_examples())
    return total


# ---------------------------------------------------------------------------
# ChatML tokenisation + loss masking
# ---------------------------------------------------------------------------

def _tokenize_conversation(
    messages: list[dict],
    tokenizer,
    context_length: int,
) -> dict | None:
    """
    Tokenise a single conversation with ChatML formatting and loss masking.

    Returns a dict with ``input_ids``, ``labels``, and ``truncated`` flag,
    or ``None`` if the conversation contains no trainable (assistant) tokens.
    """
    input_ids: list[int] = []
    labels:    list[int] = []

    for msg in messages:
        role    = msg.get("role", "").strip()
        content = msg.get("content", "")

        if not role or content is None:
            continue

        # ── encode each segment separately so we can mask precisely ──────
        header_ids  = _encode(tokenizer, f"{_IM_START}{role}\n")
        content_ids = _encode(tokenizer, content)
        footer_ids  = _encode(tokenizer, f"{_IM_END}\n")

        if role in _TRAIN_ROLES:
            # Compute loss on the content and the closing <|im_end|> token.
            # The header is context the model already sees — no gradient there.
            input_ids.extend(header_ids + content_ids + footer_ids)
            labels.extend(
                [_IGNORE_INDEX] * len(header_ids)
                + content_ids
                + footer_ids
            )
        else:
            # System / user turns: pure context, no gradient.
            segment = header_ids + content_ids + footer_ids
            input_ids.extend(segment)
            labels.extend([_IGNORE_INDEX] * len(segment))

    if not input_ids:
        return None

    # Check we have at least one trainable token before truncating.
    has_trainable = any(l != _IGNORE_INDEX for l in labels)
    if not has_trainable:
        return None

    truncated = len(input_ids) > context_length
    input_ids = input_ids[:context_length]
    labels    = labels[:context_length]

    # After truncation it's possible the trainable tokens were all cut off.
    if all(l == _IGNORE_INDEX for l in labels):
        return None

    return {"input_ids": input_ids, "labels": labels, "truncated": truncated}


def _encode(tokenizer, text: str) -> list[int]:
    """Encode *text* to token IDs, no special tokens prepended."""
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Source iteration helpers
# ---------------------------------------------------------------------------

def _iter_source_examples(source) -> Iterator[dict]:
    """Dispatch to the right iterator based on source type."""
    from tensor.instruct.data._sources import LocalSource, HubSource

    if isinstance(source, LocalSource):
        yield from source.iter_examples()
    elif isinstance(source, HubSource):
        yield from _iter_hub_examples(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")


def _iter_hub_examples(source) -> Iterator[dict]:
    import datasets as hf_datasets

    print(f"    Streaming from Hub: {source.repo_id} …")
    ds = hf_datasets.load_dataset(
        source.repo_id,
        name=source.subset,
        split=source.split,
        streaming=True,
        trust_remote_code=False,
    )

    col = source.messages_column
    for row in ds:
        messages = row.get(col)
        if messages:
            # Normalise role names if a mapping is provided.
            if source.role_mapping:
                messages = [
                    {
                        "role": source.role_mapping.get(m["role"], m["role"]),
                        "content": m.get("content", ""),
                    }
                    for m in messages
                ]
            yield {"messages": messages}


# ---------------------------------------------------------------------------
# Weighted merge
# ---------------------------------------------------------------------------

def _weighted_merge(
    batches: list[list[dict]],
    weights: list[float],
    seed: int,
) -> list[dict]:
    """
    Interleave examples from multiple sources proportionally to their weights.

    For simplicity we materialise all examples and then interleave by
    sampling without replacement according to the weight ratios.
    """
    import random

    if len(batches) == 1:
        result = list(batches[0])
        random.Random(seed).shuffle(result)
        return result

    # Find the target total size (sum of all examples).
    total = sum(len(b) for b in batches)

    # Build the merged list by round-robin weighted selection.
    rng = random.Random(seed)
    iters = [iter(b) for b in batches]
    exhausted = [False] * len(batches)
    buffers: list[list[dict]] = [list(b) for b in batches]
    for buf in buffers:
        rng.shuffle(buf)

    result: list[dict] = []
    indices = list(range(len(batches)))

    while not all(exhausted):
        live   = [i for i in indices if not exhausted[i]]
        w      = [weights[i] for i in live]
        total_w = sum(w)
        probs  = [wi / total_w for wi in w]

        # Pick a source proportionally.
        r = rng.random()
        cumulative = 0.0
        chosen = live[0]
        for idx, p in zip(live, probs):
            cumulative += p
            if r <= cumulative:
                chosen = idx
                break

        if buffers[chosen]:
            result.append(buffers[chosen].pop())
        else:
            exhausted[chosen] = True

    return result