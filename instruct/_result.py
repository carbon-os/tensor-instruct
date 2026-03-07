from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstructResult:
    """
    Immutable result stored on the Instruct instance after train() completes.

    Attributes
    ----------
    checkpoint:
        Absolute path to the .safetensors output directory.
    examples_trained:
        Number of training examples consumed (across all epochs).
    epochs:
        Number of epochs completed.
    elapsed:
        Human-readable wall-clock time, e.g. "1h 42m 08s".
    base_model_id:
        The resolved base model path or HuggingFace ID used as the starting point.
    """

    checkpoint: Path
    examples_trained: int
    epochs: int
    elapsed: str
    base_model_id: str

    def __str__(self) -> str:
        return (
            f"InstructResult(\n"
            f"  checkpoint       = {self.checkpoint}\n"
            f"  examples_trained = {self.examples_trained:,}\n"
            f"  epochs           = {self.epochs}\n"
            f"  elapsed          = {self.elapsed}\n"
            f"  base             = {self.base_model_id}\n"
            f")"
        )


def format_elapsed(seconds: float) -> str:
    """Convert a raw second count into a human-readable string."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"