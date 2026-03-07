"""
Instruct — the main entry point for tensor-instruct.

Usage
-----
.. code-block:: python

    from tensor.instruct import Instruct, InstructConfig

    run = Instruct(
        base="./output/my-pretrained-base",
        data="./my-chats.jsonl",
        output="./output/my-instruct-model",
        config=InstructConfig(epochs=3, devices=4),
    )

    run.validate()
    run.estimate()
    run.train()

    print(run.result)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch

from tensor.instruct._config import InstructConfig
from tensor.instruct._result import InstructResult, format_elapsed
from tensor.instruct.data._sources import LocalSource, HubSource
from tensor.instruct.data._mix import Mix

_DataArg = Union[str, Path, "LocalSource", "HubSource", "Mix"]

_RUN_STATE_FILE = "tensor_instruct_state.json"

# Suffixes that indicate an already instruction-tuned model.
_INSTRUCT_SUFFIXES = (
    "-instruct", "-it", "-chat", "-dpo", "-rlhf", "-sft",
    "_instruct", "_it", "_chat",
)

# ChatML special tokens that must be present in the tokenizer vocabulary.
_CHATML_TOKENS = ["<|im_start|>", "<|im_end|>"]


class BaseModelError(ValueError):
    """Raised when an instruction-tuned model is passed instead of a raw base."""


class Instruct:
    """
    Instruct fine-tuning on a raw pretrained base model.

    Accepts a base model produced by ``tensor-pretrain`` or any compatible
    HuggingFace base model ID.  Instruction-tuned variants are rejected
    immediately at construction time.

    Parameters
    ----------
    base:
        Local path to a ``tensor-pretrain`` checkpoint, or a HuggingFace
        base model ID (e.g. ``"Qwen/Qwen3-8B-Base"``).
        Instruct/chat variants raise :class:`BaseModelError`.
    data:
        Training data.  Accepts:
        - A path string or ``Path`` to a ``.jsonl`` file or directory of
          ``.jsonl`` files → converted to a ``LocalSource`` automatically.
        - A ``LocalSource`` or ``HubSource`` instance.
        - A ``Mix`` of weighted sources.
    output:
        Directory where the trained ``.safetensors`` checkpoint will be written.
    config:
        Optional :class:`InstructConfig`.  When omitted, sensible defaults
        are inferred from the model size and available hardware.
    """

    def __init__(
        self,
        base: str | Path,
        data: _DataArg,
        output: str | Path,
        config: InstructConfig | None = None,
    ) -> None:
        self._base_input   = str(base)
        self._base_model_id = _resolve_base(base)   # raises BaseModelError early
        self._data         = _normalise_data(data)
        self._output       = Path(output).expanduser().resolve()
        self._config       = config or InstructConfig()
        self._result: InstructResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def result(self) -> InstructResult:
        """The :class:`InstructResult` from the most recent ``train()`` call."""
        if self._result is None:
            raise RuntimeError("train() has not been called yet.")
        return self._result

    def validate(self) -> None:
        """
        Run pre-flight checks on the base model, data, and hardware.

        Raises an informative error as early as possible rather than letting
        the training loop fail partway through.
        """
        print("[tensor-instruct] Validating …")

        # 1. Base model.
        _check_model_accessible(self._base_model_id)
        print(f"  ✓ base model   {self._base_model_id}")

        # 2. Data sources.
        for source in self._data.keys():
            if isinstance(source, LocalSource):
                n = source.example_count()
                if n == 0:
                    raise RuntimeError(
                        f"No valid ChatML examples found under: {source.path}\n"
                        "Each JSONL line must be: "
                        '{"messages": [{"role": ..., "content": ...}, ...]}'
                    )
                print(f"  ✓ data source  {source!r}  ({n:,} examples)")
            elif isinstance(source, HubSource):
                print(f"  ✓ data source  {source!r}  (hub — validated at stream time)")

        # 3. Output directory.
        self._output.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ output       {self._output}")

        # 4. Hardware.
        devices = self._config.resolve_devices()
        _check_hardware(devices, self._config.dtype)
        print(f"  ✓ hardware     {devices}× device(s), dtype={self._config.dtype}")

        print("[tensor-instruct] Validation passed.\n")

    def estimate(self) -> None:
        """
        Print a time estimate without running training.

        Counts examples across LocalSource entries and projects wall time
        based on model size, device count, and configured epochs.
        """
        from tensor.instruct._dataset import estimate_example_count

        print("[tensor-instruct] Estimating …")

        tokenizer = _load_tokenizer(self._base_model_id)
        context_length = _resolve_context_length(self._config, tokenizer)
        devices = self._config.resolve_devices()

        n_examples = estimate_example_count(self._data)
        est_time = _estimate_wall_time(
            n_examples, self._config.epochs, devices, self._base_model_id
        )

        print()
        print(f"  base          {self._base_model_id}")
        print(f"  data          {_fmt_sources(self._data)}  →  ~{n_examples:,} examples")
        print(f"  context       {context_length} tokens max per example")
        print(f"  epochs        {self._config.epochs}")
        print(f"  devices       {devices}× device(s), dtype={self._config.dtype}")
        print(f"  est. time     {est_time}")
        print(f"  output        {self._output}")
        print()

    def train(self) -> None:
        """
        Run the full instruct fine-tuning pipeline and write a
        ``.safetensors`` checkpoint to the configured output directory.

        When complete, ``run.result`` is available with the checkpoint path,
        example count, and elapsed time.
        """
        from transformers import (
            AutoModelForCausalLM,
            TrainingArguments,
            Trainer,
        )
        from tensor.instruct._dataset import build_instruct_dataset

        t_start = time.time()

        print("[tensor-instruct] Starting instruct fine-tuning run")
        print(f"  base   → {self._base_model_id}")
        print(f"  output → {self._output}\n")

        # ── 1. Load tokeniser ──────────────────────────────────────────────
        print("[tensor-instruct] Loading tokenizer …")
        tokenizer = _load_tokenizer(self._base_model_id)
        context_length = _resolve_context_length(self._config, tokenizer)
        print(f"  context length: {context_length} tokens")

        # ── 2. Build dataset ───────────────────────────────────────────────
        dataset, stats = build_instruct_dataset(
            sources=self._data,
            tokenizer=tokenizer,
            context_length=context_length,
            seed=self._config.seed,
        )

        # ── 3. Load model ──────────────────────────────────────────────────
        print(f"\n[tensor-instruct] Loading base model ({self._base_model_id}) …")
        devices    = self._config.resolve_devices()
        torch_dtype = self._config.resolve_dtype_torch()
        attn_impl  = _pick_attention_impl()

        model = AutoModelForCausalLM.from_pretrained(
            self._base_model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=False,
        )

        # Ensure ChatML special tokens exist in the vocab.
        # This is a no-op for Qwen3 (they ship with them); it's a safety net
        # for other base models.
        tokenizer, model = _ensure_chatml_tokens(tokenizer, model)

        # ── 4. Training arguments ──────────────────────────────────────────
        batch_size = self._config.batch_size_per_device or _auto_batch_size(
            context_length, devices
        )

        steps_per_epoch = max(
            1,
            len(dataset) // (
                batch_size * devices * self._config.gradient_accumulation_steps
            ),
        )
        total_steps  = steps_per_epoch * self._config.epochs
        warmup_steps = max(1, int(total_steps * self._config.warmup_ratio))

        self._output.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(self._output),
            num_train_epochs=self._config.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            learning_rate=self._config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            bf16=(self._config.dtype == "bfloat16"),
            fp16=(self._config.dtype == "float16"),
            gradient_checkpointing=True,
            logging_steps=self._config.logging_steps,
            save_steps=(
                self._config.save_steps
                if self._config.save_steps > 0
                else total_steps + 1
            ),
            save_total_limit=2,
            dataloader_num_workers=min(4, os.cpu_count() or 1),
            report_to="none",
            seed=self._config.seed,
            remove_unused_columns=False,
        )

        collator = _ChatMLCollator(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )

        # ── 5. Persist run state (for resume) ─────────────────────────────
        self._save_run_state()

        # ── 6. Train ───────────────────────────────────────────────────────
        print(
            f"\n[tensor-instruct] Training …  "
            f"({stats['kept']:,} examples × {self._config.epochs} epoch(s))\n"
        )
        trainer.train()

        # ── 7. Save final checkpoint as .safetensors ───────────────────────
        final_path = self._output / "model"
        final_path.mkdir(exist_ok=True)

        print(f"\n[tensor-instruct] Saving checkpoint → {final_path}")
        model.save_pretrained(final_path, safe_serialization=True)
        tokenizer.save_pretrained(final_path)

        elapsed = format_elapsed(time.time() - t_start)

        self._result = InstructResult(
            checkpoint=final_path,
            examples_trained=stats["kept"] * self._config.epochs,
            epochs=self._config.epochs,
            elapsed=elapsed,
            base_model_id=self._base_model_id,
        )

        print(f"\n[tensor-instruct] ✓ Done in {elapsed}")
        print(self._result)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    @classmethod
    def resume(cls, output_dir: str | Path) -> "Instruct":
        """
        Reconstruct an :class:`Instruct` instance from a previously saved run.

        Reads ``tensor_instruct_state.json`` from *output_dir* and restores
        all original arguments so training can continue from the last checkpoint.
        """
        output_dir = Path(output_dir).expanduser().resolve()
        state_file = output_dir / _RUN_STATE_FILE

        if not state_file.exists():
            raise FileNotFoundError(
                f"No run state found at {state_file}. "
                "Make sure this is a valid tensor-instruct output directory."
            )

        state  = json.loads(state_file.read_text())
        config = InstructConfig(**state["config"])
        data   = _deserialise_data(state["data"])

        instance = cls.__new__(cls)
        instance._base_input    = state["base_input"]
        instance._base_model_id = state["base_model_id"]
        instance._data          = data
        instance._output        = output_dir
        instance._config        = config
        instance._result        = None

        print(f"[tensor-instruct] Resumed run from {output_dir}")
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_run_state(self) -> None:
        state = {
            "base_input":    self._base_input,
            "base_model_id": self._base_model_id,
            "config": {
                "epochs":                    self._config.epochs,
                "devices":                   self._config.devices,
                "dtype":                     self._config.dtype,
                "context_length":            self._config.context_length,
                "batch_size_per_device":     self._config.batch_size_per_device,
                "gradient_accumulation_steps": self._config.gradient_accumulation_steps,
                "learning_rate":             self._config.learning_rate,
                "warmup_ratio":              self._config.warmup_ratio,
                "save_steps":                self._config.save_steps,
                "logging_steps":             self._config.logging_steps,
                "seed":                      self._config.seed,
            },
            "data": _serialise_data(self._data),
        }
        state_file = self._output / _RUN_STATE_FILE
        self._output.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Data collator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _ChatMLCollator:
    """
    Pads ``input_ids``, ``labels``, and ``attention_mask`` to the longest
    sequence in the batch.  Label padding uses -100 so padded positions are
    never included in the loss.
    """
    tokenizer: object  # PreTrainedTokenizerBase

    def __call__(self, features: list[dict]) -> dict:
        input_ids = [f["input_ids"] for f in features]
        labels    = [f["labels"]    for f in features]

        max_len = max(len(x) for x in input_ids)
        pad_id  = self.tokenizer.pad_token_id

        p_input_ids = []
        p_labels    = []
        p_attn_mask = []

        for ids, lbls in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            p_input_ids.append(ids + [pad_id] * pad_len)
            p_labels.append(lbls    + [-100]  * pad_len)
            p_attn_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids":      torch.tensor(p_input_ids, dtype=torch.long),
            "labels":         torch.tensor(p_labels,    dtype=torch.long),
            "attention_mask": torch.tensor(p_attn_mask, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_base(base: str | Path) -> str:
    """
    Validate and normalise the base model identifier.

    Accepts a local path (tensor-pretrain checkpoint) or a HuggingFace ID.
    Raises :class:`BaseModelError` if an instruct variant is detected.
    """
    raw = str(base).strip()
    lower = raw.lower()

    for suffix in _INSTRUCT_SUFFIXES:
        if lower.endswith(suffix) or f"{suffix}-" in lower or f"{suffix}_" in lower:
            raise BaseModelError(
                f"'{raw}' appears to be an instruction-tuned model.\n"
                "tensor-instruct only accepts raw base models.\n"
                "Use the base variant, or a checkpoint produced by tensor-pretrain."
            )

    # Local path — return as-is (tensor-pretrain outputs are always raw bases).
    if Path(raw).exists():
        return raw

    # HuggingFace ID — return as-is.
    return raw


def _normalise_data(data: _DataArg) -> dict:
    """Coerce every accepted data= form into a ``{source: weight}`` dict."""
    if isinstance(data, Mix):
        return data.sources
    if isinstance(data, (str, Path)):
        return {LocalSource(data): 1.0}
    if isinstance(data, (LocalSource, HubSource)):
        return {data: 1.0}
    raise TypeError(
        f"data= must be a path, LocalSource, HubSource, or Mix. Got: {type(data)}"
    )


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _ensure_chatml_tokens(tokenizer, model):
    """
    Add <|im_start|> and <|im_end|> to the tokenizer and resize the model
    embedding table if the tokens are not already present.

    For Qwen3-based models this is a no-op — they ship with these tokens.
    For other bases, this guarantees the ChatML delimiter tokens have
    dedicated embeddings rather than being split into subwords.
    """
    vocab = tokenizer.get_vocab()
    missing = [t for t in _CHATML_TOKENS if t not in vocab]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
        print(f"  + Added ChatML special tokens: {missing}")
    return tokenizer, model


def _resolve_context_length(config: InstructConfig, tokenizer) -> int:
    if config.context_length:
        return config.context_length
    model_max = getattr(tokenizer, "model_max_length", 4096)
    if model_max > 131072:
        model_max = 4096
    vram_gb = _get_free_vram_gb()
    if vram_gb < 20:
        safe_ctx = 2048
    elif vram_gb < 40:
        safe_ctx = 4096
    else:
        safe_ctx = 8192
    return min(model_max, safe_ctx)


def _get_free_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free / (1024 ** 3)
    except Exception:
        pass
    return 999.0


def _pick_attention_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


def _auto_batch_size(context_length: int, devices: int) -> int:
    vram_gb = _get_free_vram_gb()
    if context_length >= 4096:
        if vram_gb < 20:   return 1
        elif vram_gb < 40: return 2
        else:              return 4
    else:
        if vram_gb < 20:   return 2
        elif vram_gb < 40: return 4
        else:              return 8


def _check_model_accessible(model_id: str) -> None:
    # Local path — just check it exists.
    p = Path(model_id)
    if p.exists():
        if not (p / "config.json").exists():
            raise FileNotFoundError(
                f"Local base path exists but looks incomplete "
                f"(no config.json): {p}\n"
                "Make sure this is a valid tensor-pretrain output."
            )
        return

    # HuggingFace ID.
    try:
        from huggingface_hub import model_info
        model_info(model_id)
    except Exception as e:
        raise RuntimeError(
            f"Could not access model '{model_id}' on HuggingFace Hub.\n"
            f"Check your internet connection or HF_TOKEN. Details: {e}"
        )


def _check_hardware(devices: int, dtype: str) -> None:
    try:
        import torch
        if devices > 1 and torch.cuda.device_count() < devices:
            raise RuntimeError(
                f"Requested {devices} GPU(s) but only "
                f"{torch.cuda.device_count()} are visible. "
                "Set CUDA_VISIBLE_DEVICES or reduce InstructConfig.devices."
            )
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bfloat16 is not supported on this hardware. "
                "Use dtype='float16' or dtype='float32'."
            )
    except ImportError:
        pass


def _estimate_wall_time(
    n_examples: int,
    epochs: int,
    devices: int,
    model_id: str,
) -> str:
    """
    Rough wall-time estimate for instruct fine-tuning.
    Baseline: A100 at ~2,000 examples/hr for a 7B model at context 2048.
    """
    if n_examples == 0:
        return "unknown (could not count examples — Hub sources not sampled)"

    # Try to infer param count for scaling.
    param_b = 7.0
    try:
        from transformers import AutoConfig as HFAutoConfig
        hf_cfg = HFAutoConfig.from_pretrained(model_id)
        p = getattr(hf_cfg, "num_parameters", None)
        if p:
            param_b = p / 1e9
    except Exception:
        pass

    base_examples_per_hr = max(500, 2_000 * (7 / max(param_b, 0.1)))
    total_examples_per_hr = base_examples_per_hr * devices
    hours = (n_examples * epochs) / total_examples_per_hr

    if hours < 1 / 12:
        return f"~{max(1, int(hours * 60))} min"
    if hours < 1:
        return f"~{int(hours * 60)} min"
    if hours < 48:
        return f"~{hours:.1f} hrs"
    return f"~{hours / 24:.1f} days"


def _fmt_sources(sources: dict) -> str:
    parts = [repr(src) for src in sources]
    return " + ".join(parts) if len(parts) > 1 else parts[0]


def _serialise_data(sources: dict) -> list[dict]:
    out = []
    for src, w in sources.items():
        if isinstance(src, LocalSource):
            out.append({"type": "local", "path": str(src.path), "weight": w})
        elif isinstance(src, HubSource):
            out.append({
                "type":            "hub",
                "repo_id":         src.repo_id,
                "subset":          src.subset,
                "split":           src.split,
                "messages_column": src.messages_column,
                "role_mapping":    src.role_mapping,
                "weight":          w,
            })
    return out


def _deserialise_data(records: list[dict]) -> dict:
    result = {}
    for r in records:
        if r["type"] == "local":
            result[LocalSource(r["path"])] = r["weight"]
        elif r["type"] == "hub":
            result[HubSource(
                repo_id=r["repo_id"],
                subset=r.get("subset"),
                split=r.get("split", "train"),
                messages_column=r.get("messages_column", "messages"),
                role_mapping=r.get("role_mapping"),
            )] = r["weight"]
    return result