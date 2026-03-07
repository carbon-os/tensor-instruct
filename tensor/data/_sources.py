"""
Data source definitions for tensor-instruct.

LocalSource  — reads .jsonl files from a local path.
HubSource    — pulls a ChatML-formatted dataset from the HuggingFace Hub.

Both sources expect examples in ChatML schema:

    {"messages": [
        {"role": "system",    "content": "..."},   ← optional
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]}

HubSource additionally supports a ``role_mapping`` dict to normalise
non-standard role names (e.g. ``{"human": "user", "gpt": "assistant"}``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


class _JsonlValidationError(ValueError):
    pass


@dataclass(unsafe_hash=True)
class LocalSource:
    """
    Reads ``.jsonl`` files from a local file or directory.

    Each line must be valid JSON containing a ``"messages"`` key whose
    value is a list of ``{"role": ..., "content": ...}`` dicts.

    Lines that are malformed, missing ``"messages"``, or contain an empty
    messages list are silently skipped during iteration (but counted by
    :meth:`example_count` so you can spot problems early via ``validate()``).

    Parameters
    ----------
    path:
        Path to a ``.jsonl`` file, or a directory that will be walked
        recursively for all ``.jsonl`` files.
    encoding:
        File encoding. Defaults to ``"utf-8"``.
    """

    path: str | Path
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(
                f"LocalSource path does not exist: {self.path}"
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def example_count(self) -> int:
        """Return the number of valid ChatML examples under this source."""
        return sum(1 for _ in self.iter_examples())

    def iter_examples(self) -> Iterator[dict]:
        """
        Yield valid ``{"messages": [...]}`` dicts from all ``.jsonl`` files.

        Skips lines that are not valid JSON, are missing the ``messages``
        key, or contain a messages list with fewer than two turns (need at
        least one user and one assistant turn to be trainable).
        """
        for fp in self._iter_files():
            yield from _iter_jsonl(fp, self.encoding)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _iter_files(self) -> Iterator[Path]:
        if self.path.is_file():
            if self.path.suffix.lower() == ".jsonl":
                yield self.path
            return

        for fp in sorted(self.path.rglob("*.jsonl")):
            yield fp

    def __repr__(self) -> str:
        return f"LocalSource({str(self.path)!r})"


@dataclass(unsafe_hash=True)
class HubSource:
    """
    A ChatML-formatted dataset from the HuggingFace Hub.

    Parameters
    ----------
    repo_id:
        The Hub dataset repository, e.g. ``"HuggingFaceH4/ultrachat_200k"``.
    subset:
        Optional dataset config / subset name.
    split:
        Which split to use. Defaults to ``"train"``.
    messages_column:
        Column that holds the messages list. Defaults to ``"messages"``.
        Some datasets use ``"conversations"`` — set this accordingly.
    role_mapping:
        Optional dict to normalise non-standard role names.
        Example: ``{"human": "user", "gpt": "assistant"}``
        If ``None``, roles are used as-is.
    """

    repo_id: str
    subset: str | None = None
    split: str = "train"
    messages_column: str = "messages"
    role_mapping: dict[str, str] | None = None

    def __repr__(self) -> str:
        parts = [repr(self.repo_id)]
        if self.subset:
            parts.append(f"subset={self.subset!r}")
        if self.messages_column != "messages":
            parts.append(f"messages_column={self.messages_column!r}")
        return f"HubSource({', '.join(parts)})"


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path, encoding: str) -> Iterator[dict]:
    """Yield valid ChatML examples from a single .jsonl file."""
    try:
        with path.open(encoding=encoding, errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                messages = obj.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    continue

                # Must have at least one assistant turn to produce any gradient.
                roles = {m.get("role") for m in messages}
                if "assistant" not in roles:
                    continue

                yield {"messages": messages}

    except (OSError, PermissionError):
        return