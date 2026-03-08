"""
Custom training progress callback for tensor-instruct.

Replaces the default HuggingFace ProgressCallback with a cleaner display
that shows human-readable estimated time remaining instead of raw H:M:S,
and suppresses the per-step log dict spam.
"""

from __future__ import annotations

import time
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class TensorProgressCallback(TrainerCallback):
    """
    Clean progress callback that shows:

        Step 188/77892  epoch 0.01  loss 1.392  lr 1.45e-06  est 84hr 10mins remaining

    Replaces the default ProgressCallback which prints a raw H:MM:SS tqdm
    bar and also spams a log dict line every logging_steps.
    """

    def __init__(self) -> None:
        self._t_start:    float | None = None
        self._last_loss:  float | None = None
        self._last_lr:    float | None = None
        self._last_print: float        = 0.0
        # Minimum seconds between printed lines — avoids flooding the terminal
        # while still giving a live feel.
        self._print_interval: float    = 10.0

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._t_start = time.time()
        self._last_print = 0.0
        if state.is_world_process_zero:
            print(f"[tensor-instruct] Training started  ({state.max_steps:,} total steps)\n")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        """Capture loss and lr from the log dict — don't print it."""
        if not state.is_world_process_zero or not logs:
            return
        if "loss" in logs:
            self._last_loss = float(logs["loss"])
        if "learning_rate" in logs:
            self._last_lr = float(logs["learning_rate"])

        # Throttle output so we print at most once every _print_interval seconds.
        now = time.time()
        if now - self._last_print < self._print_interval:
            return
        self._last_print = now
        self._print_status(state)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.is_world_process_zero:
            epoch = int(state.epoch or 0)
            print(f"\n[tensor-instruct] Epoch {epoch} complete  (step {state.global_step:,})\n")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.is_world_process_zero:
            elapsed = _fmt_duration(time.time() - (self._t_start or time.time()))
            print(f"\n[tensor-instruct] Training complete  —  {elapsed} total\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _print_status(self, state: TrainerState) -> None:
        step     = state.global_step
        total    = state.max_steps
        epoch    = state.epoch or 0.0
        elapsed  = time.time() - (self._t_start or time.time())

        # ETA derived from actual step throughput so far.
        eta_str = "calculating…"
        if step > 0 and elapsed > 0:
            secs_per_step = elapsed / step
            remaining     = secs_per_step * (total - step)
            eta_str       = f"est {_fmt_duration(remaining)} remaining"

        pct  = f"{100 * step / total:.1f}%" if total else "?"
        loss = f"{self._last_loss:.3f}" if self._last_loss is not None else "…"
        lr   = f"{self._last_lr:.3e}"   if self._last_lr   is not None else "…"

        print(
            f"  step {step:,}/{total:,} ({pct})  "
            f"epoch {epoch:.2f}  "
            f"loss {loss}  "
            f"lr {lr}  "
            f"{eta_str}"
        )


def _fmt_duration(seconds: float) -> str:
    """Convert seconds into a human-readable string like '2hr 14mins' or '45mins'."""
    seconds = max(0, int(seconds))
    h, rem  = divmod(seconds, 3600)
    m, _    = divmod(rem, 60)

    if h > 0 and m > 0:
        return f"{h}hr {m}mins"
    if h > 0:
        return f"{h}hr"
    if m > 0:
        return f"{m}mins"
    return "less than 1 min"