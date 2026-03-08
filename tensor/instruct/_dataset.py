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

Memory management
-----------------
Examples are streamed one at a time through ``Dataset.from_generator`` —
nothing is materialised into Python heap. A safe ``max_examples`` cap is
auto-derived from available system RAM unless explicitly overridden in
``InstructConfig``.

    Available RAM   Auto cap
    ─────────────   ────────
    < 8 GB          10,000
    < 16 GB         25,000
    < 32 GB         50,000
    < 64 GB         150,000
    ≥ 64 GB         unlimited

ChatML format expected per example
------------------------------------
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
# Memory-aware cap
# ---------------------------------------------------------------------------

def _auto_max_examples() -> int | None:
    """
    Derive a safe example cap from available system RAM.

    Returns ``None`` (unlimited) when RAM is large enough that capping
    would be unnecessarily restrictive. The thresholds are conservative —
    each tokenised example is roughly 4–8 KB in Arrow format, so even
    150k examples sits comfortably under 2 GB of dataset memory.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        # psutil unavailable — apply a safe default.
        return 25_000

    if available_gb < 8:
        cap = 10_000
    elif available_gb < 16:
        cap = 25_000
    elif available_gb < 32:
        cap = 50_000
    elif available_gb < 64:
        cap = 150_000
    else:
        return None  # unlimited

    return cap


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_instruct_dataset(
    sources: dict,
    tokenizer,
    context_length: int,
    max_examples: int | None = "auto",
    seed: int = 42,
):
    """
    Build and return a ``datasets.Dataset`` of tokenised, loss-masked examples.

    Parameters
    ----------
    sources:
        A ``dict[source, weight]`` mapping (already normalised by Mix).
    tokenizer:
        A HuggingFace tokeniser.
    context_length:
        Maximum tokens per example. Longer conversations are truncated.
    max_examples:
        ``"auto"``  — derive cap from available RAM (default).
        ``None``    — no cap, use all examples (only safe on high-RAM machines).
        ``int``     — explicit cap, e.g. ``10_000``.
    seed:
        Shuffle seed.

    Returns
    -------
    dataset:
        HuggingFace ``Dataset`` with ``input_ids`` and ``labels`` columns.
    stats:
        Dict with ``total``, ``kept``, ``truncated``, and ``skipped`` counts.
    """
    import datasets as hf_datasets

    # Resolve max_examples.
    if max_examples == "auto":
        resolved_cap = _auto_max_examples()
    else:
        resolved_cap = max_examples  # None or explicit int

    if resolved_cap is not None:
        print(
            f"[tensor-instruct] Building tokenised dataset …  "
            f"(cap: {resolved_cap:,} examples)"
        )
    else:
        print("[tensor-instruct] Building tokenised dataset …  (cap: unlimited)")

    stats = {"total": 0, "kept": 0, "truncated": 0, "skipped": 0}

    # Log sources before entering the generator so the output appears
    # immediately rather than only when the first example is consumed.
    for source, weight in sources.items():
        print(f"  → {source!r}  (weight {weight:.2f})")

    def generate_examples():
        for source, weight in sources.items():
            for example in _iter_source_examples(source):
                if resolved_cap is not None and stats["kept"] >= resolved_cap:
                    return

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

                stats["kept"] += 1
                yield {
                    "input_ids": result["input_ids"],
                    "labels":    result["labels"],
                }

    dataset = hf_datasets.Dataset.from_generator(generate_examples)

    if not len(dataset):
        raise RuntimeError(
            "No valid training examples were produced from the provided sources.\n"
            "Check that your JSONL files contain the expected "
            '{"messages": [{"role": ..., "content": ...}, ...]} schema.'
        )

    print(
        f"  ✓ {stats['kept']:,} examples kept  "
        f"({stats['truncated']:,} truncated, {stats['skipped']:,} skipped)"
    )

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
            total += source.example_count()
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

        # Encode each segment separately so we can mask precisely.
        header_ids  = _encode(tokenizer, f"{_IM_START}{role}\n")
        content_ids = _encode(tokenizer, content)
        footer_ids  = _encode(tokenizer, f"{_IM_END}\n")

        if role in _TRAIN_ROLES:
            # Compute loss on content and closing <|im_end|>.
            # Header is context — no gradient.
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

    # Must have at least one trainable token before truncation.
    if not any(l != _IGNORE_INDEX for l in labels):
        return None

    truncated = len(input_ids) > context_length
    input_ids = input_ids[:context_length]
    labels    = labels[:context_length]

    # After truncation all trainable tokens may have been cut off.
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
            if source.role_mapping:
                messages = [
                    {
                        "role":    source.role_mapping.get(m["role"], m["role"]),
                        "content": m.get("content", ""),
                    }
                    for m in messages
                ]
            yield {"messages": messages}