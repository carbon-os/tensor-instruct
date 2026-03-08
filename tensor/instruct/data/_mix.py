"""
Mix — weighted combination of instruct data sources.
"""

from __future__ import annotations

from typing import Union
from tensor.instruct.data._sources import LocalSource, HubSource

AnySource = Union[LocalSource, HubSource]


class Mix:
    """
    Weighted combination of instruct data sources.

    Weights do not need to sum to 1.0 — they are normalised internally.

    Example
    -------
    .. code-block:: python

        Mix({
            LocalSource("./my-chats.jsonl"):                               0.8,
            HubSource("HuggingFaceH4/ultrachat_200k", split="train_sft"): 0.2,
        })
    """

    def __init__(self, sources: dict[AnySource, float]) -> None:
        if not sources:
            raise ValueError("Mix requires at least one source.")

        total = sum(sources.values())
        if total <= 0:
            raise ValueError("Mix weights must sum to a positive number.")

        self._sources: dict[AnySource, float] = {
            src: w / total for src, w in sources.items()
        }

    @property
    def sources(self) -> dict[AnySource, float]:
        """Normalised ``{source: weight}`` mapping."""
        return dict(self._sources)

    def __repr__(self) -> str:
        parts = [f"  {src!r}: {w:.2f}" for src, w in self._sources.items()]
        return "Mix({\n" + ",\n".join(parts) + "\n})"