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

_INSTRUCT_SUFFIXES = (
    "-instruct", "-it", "-chat", "-dpo", "-rlhf", "-sft",
    "_instruct", "_it", "_chat",
)

_CHATML_TOKENS = ["<|im_start|>", "<|im_end|>"]


class BaseModelError(ValueError):
    """Raised when an instruction-tuned model is passed instead of a raw base."""


# ──────────────────────────────────────────────────────────────────────────────
# GPU detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_gpu() -> dict:
    """
    Return a capability dict for the primary CUDA device.

    Keys
    ----
    name          : str   — GPU name as reported by CUDA
    vram_gb       : float — total VRAM in GB
    free_vram_gb  : float — free VRAM in GB
    is_blackwell  : bool  — Blackwell architecture (compute cap 10.x)
    is_hopper     : bool  — Hopper architecture (sm_90)
    is_ampere     : bool  — Ampere architecture (sm_80 / sm_86)
    is_ada        : bool  — Ada Lovelace architecture (sm_89)
    sm_count      : int   — number of streaming multiprocessors
    has_fp8       : bool  — native FP8 Tensor Core support (Hopper+)
    """
    info = {
        "name":         "cpu",
        "vram_gb":      0.0,
        "free_vram_gb": 0.0,
        "is_blackwell": False,
        "is_hopper":    False,
        "is_ampere":    False,
        "is_ada":       False,
        "sm_count":     0,
        "has_fp8":      False,
    }

    try:
        if not torch.cuda.is_available():
            return info

        props         = torch.cuda.get_device_properties(0)
        major, minor  = props.major, props.minor
        total_bytes   = props.total_memory
        free_bytes, _ = torch.cuda.mem_get_info(0)

        info["name"]         = props.name
        info["vram_gb"]      = total_bytes / (1024 ** 3)
        info["free_vram_gb"] = free_bytes  / (1024 ** 3)
        info["sm_count"]     = props.multi_processor_count
        info["is_blackwell"] = major >= 10
        info["is_hopper"]    = major == 9
        info["is_ampere"]    = major == 8 and minor in (0, 6)
        info["is_ada"]       = major == 8 and minor == 9
        info["has_fp8"]      = major >= 9

    except Exception:
        pass

    return info


def _print_gpu_banner(gpu: dict) -> None:
    """Print a short GPU summary with any unlocked optimisations."""
    if gpu["name"] == "cpu":
        print("  device         CPU (no CUDA GPU detected)")
        return

    print(f"  device         {gpu['name']}")
    print(f"  vram           {gpu['vram_gb']:.0f} GB total  /  {gpu['free_vram_gb']:.0f} GB free")
    print(f"  SMs            {gpu['sm_count']}")

    unlocked = []
    if gpu["has_fp8"]:
        unlocked.append("FP8 training")
    if gpu["is_blackwell"]:
        unlocked.append("FlashAttention-3 (if installed)")
    elif gpu["is_hopper"]:
        unlocked.append("max-autotune compile")
    elif gpu["is_ampere"] or gpu["is_ada"]:
        # A100 SXM: 108 SMs, stable Triton/Inductor, max-autotune is the right mode
        unlocked.append("max-autotune compile")
        unlocked.append("TF32 Tensor Cores")

    if unlocked:
        print(f"  optimisations  {', '.join(unlocked)}")


# ──────────────────────────────────────────────────────────────────────────────
# Instruct
# ──────────────────────────────────────────────────────────────────────────────

class Instruct:
    """
    Instruct fine-tuning on a raw pretrained base model.

    Parameters
    ----------
    base:
        Local path to a ``tensor-pretrain`` checkpoint, or a HuggingFace
        base model ID (e.g. ``"Qwen/Qwen3-0.6B-Base"``).
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
        self._base_input    = str(base)
        self._base_model_id = _resolve_base(base)
        self._data          = _normalise_data(data)
        self._output        = Path(output).expanduser().resolve()
        self._config        = config or InstructConfig()
        self._result: InstructResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def result(self) -> InstructResult:
        if self._result is None:
            raise RuntimeError("train() has not been called yet.")
        return self._result

    def validate(self) -> None:
        print("[tensor-instruct] Validating …")

        _check_model_accessible(self._base_model_id)
        print(f"  ✓ base model   {self._base_model_id}")

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

        self._output.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ output       {self._output}")

        devices = self._config.resolve_devices()
        _check_hardware(devices, self._config.dtype)
        print(f"  ✓ hardware     {devices}× device(s), dtype={self._config.dtype}")

        print("[tensor-instruct] Validation passed.\n")

    def estimate(self) -> None:
        from tensor.instruct._dataset import estimate_example_count

        print("[tensor-instruct] Estimating …")

        tokenizer      = _load_tokenizer(self._base_model_id)
        gpu            = _detect_gpu()
        context_length = _resolve_context_length(self._config, tokenizer, gpu)
        devices        = self._config.resolve_devices()

        n_examples = estimate_example_count(self._data)
        est_time   = _estimate_wall_time(
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
        """
        from transformers import (
            AutoModelForCausalLM,
            TrainingArguments,
            Trainer,
            ProgressCallback,
        )
        from tensor.instruct._dataset import build_instruct_dataset
        from tensor.instruct._callback import TensorProgressCallback

        t_start = time.time()

        # ── 1. Detect GPU and print capabilities ───────────────────────────
        gpu = _detect_gpu()

        print("[tensor-instruct] Starting instruct fine-tuning run")
        print(f"  base   → {self._base_model_id}")
        print(f"  output → {self._output}")
        _print_gpu_banner(gpu)
        print()

        # ── 2. Enable TF32 for faster matmuls on Ampere / Ada / Hopper / Blackwell
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ── 3. Load tokeniser ──────────────────────────────────────────────
        print("[tensor-instruct] Loading tokenizer …")
        tokenizer      = _load_tokenizer(self._base_model_id)
        context_length = _resolve_context_length(self._config, tokenizer, gpu)
        print(f"  context length: {context_length} tokens")

        # ── 4. Build dataset ───────────────────────────────────────────────
        dataset, stats = build_instruct_dataset(
            sources=self._data,
            tokenizer=tokenizer,
            context_length=context_length,
            max_examples=self._config.max_examples,
            seed=self._config.seed,
        )

        # ── 5. Load model ──────────────────────────────────────────────────
        print(f"\n[tensor-instruct] Loading base model ({self._base_model_id}) …")
        devices     = self._config.resolve_devices()
        torch_dtype = _resolve_dtype(self._config, gpu)
        attn_impl   = _pick_attention_impl(gpu)

        model = AutoModelForCausalLM.from_pretrained(
            self._base_model_id,
            dtype=torch_dtype,
            attn_implementation=attn_impl,
            trust_remote_code=False,
        )

        tokenizer, model = _ensure_chatml_tokens(tokenizer, model)

        # ── 6. FP8 training via Transformer Engine (Hopper / Blackwell) ───
        fp8_active = False
        if gpu["has_fp8"] and _transformer_engine_available():
            print("[tensor-instruct] Enabling FP8 training via Transformer Engine …")
            model = _wrap_fp8(model)
            fp8_active = True

        # ── 7. Compile model (only on GPUs with enough VRAM) ──────────────
        if _should_compile(gpu):
            compile_mode = "max-autotune" if gpu["sm_count"] >= 100 else "reduce-overhead"
            print(f"[tensor-instruct] Compiling model (mode={compile_mode}) …")
            model = torch.compile(model, mode=compile_mode)

        # ── 8. Training arguments ──────────────────────────────────────────
        batch_size = self._config.batch_size_per_device or _auto_batch_size(
            context_length, gpu
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

        # Gradient checkpointing only needed when VRAM is tight.
        use_grad_ckpt = True

        training_args = TrainingArguments(
            output_dir=str(self._output),
            num_train_epochs=self._config.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            learning_rate=self._config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            bf16=(torch_dtype == torch.bfloat16 and not fp8_active),
            fp16=(torch_dtype == torch.float16),
            gradient_checkpointing=use_grad_ckpt,
            optim="adamw_torch_fused",
            dataloader_pin_memory=True,
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
            disable_tqdm=True,
        )

        collator = _ChatMLCollator(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )

        # Swap out the default ProgressCallback for our clean one.
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(TensorProgressCallback())

        # ── 9. Persist run state ───────────────────────────────────────────
        self._save_run_state()

        # ── 10. Train ──────────────────────────────────────────────────────
        print(
            f"\n[tensor-instruct] Training …  "
            f"({len(dataset):,} examples × {self._config.epochs} epoch(s))\n"
        )
        trainer.train()

        # ── 11. Save checkpoint ────────────────────────────────────────────
        final_path = self._output / "model"
        final_path.mkdir(exist_ok=True)

        print(f"\n[tensor-instruct] Saving checkpoint → {final_path}")
        model.save_pretrained(final_path, safe_serialization=True)
        tokenizer.save_pretrained(final_path)

        elapsed = format_elapsed(time.time() - t_start)

        self._result = InstructResult(
            checkpoint=final_path,
            examples_trained=len(dataset) * self._config.epochs,
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
                "epochs":                      self._config.epochs,
                "max_examples":                self._config.max_examples,
                "devices":                     self._config.devices,
                "dtype":                       self._config.dtype,
                "context_length":              self._config.context_length,
                "batch_size_per_device":       self._config.batch_size_per_device,
                "gradient_accumulation_steps": self._config.gradient_accumulation_steps,
                "learning_rate":               self._config.learning_rate,
                "warmup_ratio":                self._config.warmup_ratio,
                "save_steps":                  self._config.save_steps,
                "logging_steps":               self._config.logging_steps,
                "seed":                        self._config.seed,
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
    tokenizer: object

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
    raw   = str(base).strip()
    lower = raw.lower()

    for suffix in _INSTRUCT_SUFFIXES:
        if lower.endswith(suffix) or f"{suffix}-" in lower or f"{suffix}_" in lower:
            raise BaseModelError(
                f"'{raw}' appears to be an instruction-tuned model.\n"
                "tensor-instruct only accepts raw base models.\n"
                "Use the base variant, or a checkpoint produced by tensor-pretrain."
            )

    if Path(raw).exists():
        return raw
    return raw


def _normalise_data(data: _DataArg) -> dict:
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

    p = Path(model_id)
    if p.exists():
        tokenizer_file = p / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"No tokenizer.json found in local checkpoint: {p}\n"
                "This usually means the tensor-pretrain run did not complete "
                "cleanly and the tokenizer was never saved.\n"
                "Re-run tensor-pretrain, or copy the tokenizer files from the "
                "original upstream base model into this directory."
            )

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _ensure_chatml_tokens(tokenizer, model):
    vocab   = tokenizer.get_vocab()
    missing = [t for t in _CHATML_TOKENS if t not in vocab]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))
        print(f"  + Added ChatML special tokens: {missing}")
    return tokenizer, model


def _resolve_dtype(config: InstructConfig, gpu: dict):
    return {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[config.dtype]


def _resolve_context_length(config: InstructConfig, tokenizer, gpu: dict) -> int:
    if config.context_length:
        return config.context_length

    model_max = getattr(tokenizer, "model_max_length", 4096)
    if model_max > 131072:
        model_max = 4096

    vram_gb = gpu["free_vram_gb"]

    if gpu["is_blackwell"] or vram_gb >= 64:
        safe_ctx = 8192
    elif vram_gb >= 40:
        safe_ctx = 4096
    elif vram_gb >= 20:
        safe_ctx = 4096
    else:
        safe_ctx = 2048

    return min(model_max, safe_ctx)


def _pick_attention_impl(gpu: dict) -> str:
    if gpu["is_blackwell"]:
        try:
            import flash_attn_3  # noqa: F401
            return "flash_attention_3"
        except ImportError:
            pass

    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass

    return "eager"


def _transformer_engine_available() -> bool:
    try:
        import transformer_engine  # noqa: F401
        return True
    except ImportError:
        return False


def _wrap_fp8(model):
    try:
        import transformer_engine.pytorch as te

        def replace_linear(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    has_bias = child.bias is not None
                    te_linear = te.Linear(
                        child.in_features,
                        child.out_features,
                        bias=has_bias,
                    )
                    te_linear.weight.data.copy_(child.weight.data)
                    if has_bias:
                        te_linear.bias.data.copy_(child.bias.data)

                    # ── cast to match the source layer dtype ──────────────
                    te_linear = te_linear.to(child.weight.dtype)

                    setattr(module, name, te_linear)
                else:
                    replace_linear(child)

        replace_linear(model)
        print("  + FP8 Linear layers active")

    except Exception as e:
        print(f"  ! FP8 wrap failed ({e}) — falling back to BF16")

    return model


def _should_compile(gpu: dict) -> bool:
    # sm_120 (Blackwell consumer) has unstable Triton support — skip.
    if gpu["is_blackwell"]:
        return False
    # A100 SXM (sm_80, 108 SMs) and L40S (sm_89, 142 SMs) both have
    # stable Triton/Inductor support. max-autotune is safe and recommended.
    # Compile is also enabled for Hopper and any other GPU with enough VRAM.
    try:
        major = int(torch.__version__.split(".")[0])
        return (
            major >= 2
            and torch.cuda.is_available()
            and gpu["free_vram_gb"] >= 20
        )
    except Exception:
        return False


def _auto_batch_size(context_length: int, gpu: dict) -> int:
    vram_gb = gpu["free_vram_gb"]

    if context_length >= 8192:
        if vram_gb >= 64:  return 4
        return 1
    elif context_length >= 4096:
        if vram_gb >= 64:  return 16
        if vram_gb >= 40:  return 8
        if vram_gb >= 20:  return 4
        return 1
    else:
        if vram_gb >= 64:  return 32
        if vram_gb >= 40:  return 16
        if vram_gb >= 20:  return 8
        return 2      # was 4 — too aggressive for 16 GB


def _check_model_accessible(model_id: str) -> None:
    p = Path(model_id)
    if p.exists():
        if not (p / "config.json").exists():
            raise FileNotFoundError(
                f"Local base path exists but looks incomplete "
                f"(no config.json): {p}\n"
                "Make sure this is a valid tensor-pretrain output."
            )
        return
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
    if n_examples == 0:
        return "unknown (could not count examples — Hub sources not sampled)"

    param_b = 7.0
    try:
        from transformers import AutoConfig as HFAutoConfig
        hf_cfg = HFAutoConfig.from_pretrained(model_id)
        p = getattr(hf_cfg, "num_parameters", None)
        if p:
            param_b = p / 1e9
    except Exception:
        pass

    base_examples_per_hr  = max(500, 2_000 * (7 / max(param_b, 0.1)))
    total_examples_per_hr = base_examples_per_hr * devices
    hours = (n_examples * epochs) / total_examples_per_hr

    total_minutes = int(hours * 60)

    if total_minutes < 1:
        return "est less than 1 min"
    if total_minutes < 60:
        return f"est {total_minutes} mins"

    h = total_minutes // 60
    m = total_minutes % 60
    if m == 0:
        return f"est {h}hr"
    return f"est {h}hr {m}mins"


def _get_free_vram_gb() -> float:
    """Legacy helper — kept for any external callers. Use _detect_gpu() internally."""
    try:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free / (1024 ** 3)
    except Exception:
        pass
    return 999.0


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