"""
InstructConfig — optional training hyperparameters for an Instruct run.

Defaults are tuned for supervised fine-tuning on top of a pretrained base.
Learning rate is intentionally lower than pretraining — we are shaping
behaviour, not injecting knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class InstructConfig:
    """
    Optional training configuration for instruct fine-tuning.

    When omitted from :class:`Instruct`, all values are auto-inferred from
    the model size and available hardware.

    Parameters
    ----------
    epochs:
        Number of full passes over the training dataset.
    max_examples:
        Maximum number of training examples to use.
        ``"auto"`` — derive a safe cap from available system RAM (default).
        ``None``   — no cap, use all examples (safe only on high-RAM machines).
        ``int``    — explicit cap, e.g. ``50_000``.
    devices:
        Number of GPUs to use.  ``None`` = all visible GPUs, or CPU if none.
    dtype:
        Model and training dtype.  ``"bfloat16"`` is recommended for A100+.
    context_length:
        Maximum token length per training example.  Conversations longer than
        this are truncated.  ``None`` = inferred from model config, capped at
        8192 for memory safety.
    batch_size_per_device:
        Per-device micro-batch size.  ``None`` = auto.
    gradient_accumulation_steps:
        Gradient accumulation steps.  Effective batch =
        ``batch_size_per_device × devices × gradient_accumulation_steps``.
    learning_rate:
        Peak learning rate.  2e-5 is a safe default for instruct fine-tuning.
        Pretraining uses 1e-4; instruct tuning should be lower to avoid
        catastrophic forgetting.
    warmup_ratio:
        Fraction of total steps used for linear LR warm-up.
    save_steps:
        Save a checkpoint every N steps.  ``0`` = only save at the end.
    logging_steps:
        Log training metrics every N steps.
    seed:
        Random seed for reproducibility.
    """

    epochs: int = 3
    max_examples: int | str | None = "auto"
    devices: int | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    context_length: int | None = None
    batch_size_per_device: int | None = None
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    save_steps: int = 500
    logging_steps: int = 10
    seed: int = 42

    def resolve_devices(self) -> int:
        if self.devices is not None:
            return self.devices
        try:
            import torch
            return max(torch.cuda.device_count(), 1)
        except ImportError:
            return 1

    def resolve_dtype_torch(self):
        import torch
        return {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }[self.dtype]