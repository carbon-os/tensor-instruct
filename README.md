# tensor-instruct

**Instruct fine-tuning on raw base models. Point it at your data, get a production instruct model back.**

`tensor-instruct` is part of the **Tensor Framework** by Netangular. It takes a raw pretrained base — either a `tensor-pretrain` checkpoint or any compatible HuggingFace base model — and applies supervised instruct fine-tuning on top, using your own ChatML-formatted data.

No preprocessing pipelines. No data wrangling. Loss-masked on assistant turns only. A production-ready instruct model on the other side.

---

## The Tensor Framework
```
tensor-pretrain    Continuous pretraining on Tensor or custom base models
tensor-adapt       Teach behavior via low-rank adapters (instruct / chat)
tensor-datagen     Generate instruct fine-tune datasets from your files
tensor-instruct    Instruct fine-tuning on raw base models               ← you are here
tensor-inference   Run .safetensors output at high velocity (C++)
```

---

## How It Works

1. **Point it at your base** — a `tensor-pretrain` checkpoint or any compatible HuggingFace base model ID. Instruct-tuned variants are rejected immediately.
2. **Point it at your data** — a local `.jsonl` file, a directory of `.jsonl` files, a HuggingFace Hub dataset, or a weighted mix of any combination. All data must be in ChatML format.
3. **Get a production instruct model** — output is a clean `.safetensors` checkpoint, loss-masked on assistant turns only, ready for `tensor-inference`.

---

## ChatML Format

`tensor-instruct` exclusively uses the ChatML conversation format. Each line
of your `.jsonl` is one conversation:
```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is gradient descent?"}, {"role": "assistant", "content": "Gradient descent is an optimisation algorithm that minimises a loss function by iteratively stepping in the direction of steepest descent."}]}
{"messages": [{"role": "user", "content": "Write a SQL query that counts rows by date."}, {"role": "assistant", "content": "SELECT DATE(created_at), COUNT(*) FROM your_table GROUP BY DATE(created_at) ORDER BY DATE(created_at);"}]}
```

- `system` turns are optional. If omitted, the conversation starts from the first `user` turn.
- Multi-turn conversations are fully supported — alternate `user` and `assistant` turns as many times as needed.
- Loss is computed **only on assistant turns**. System and user tokens are masked out and receive no gradient signal.
- Lines that are malformed, missing a `messages` key, or containing no `assistant` turn are silently skipped.

---

## Quick Start

### Install
```bash
pip install tensor-instruct
```

### From a tensor-pretrain checkpoint
```python
from tensor.instruct import Instruct

run = Instruct(
    base="./output/my-pretrained-base",
    data="./my-chats.jsonl",
    output="./output/my-instruct-model",
)

run.train()
```

### From a raw HuggingFace base model
```python
from tensor.instruct import Instruct

run = Instruct(
    base="Qwen/Qwen3-8B-Base",
    data="./my-chats.jsonl",
    output="./output/my-instruct-model",
)

run.train()
```

### From a HuggingFace Hub dataset
```python
from tensor.instruct import Instruct
from tensor.instruct.data import HubSource

run = Instruct(
    base="./output/my-pretrained-base",
    data=HubSource("HuggingFaceH4/ultrachat_200k", split="train_sft"),
    output="./output/my-instruct-model",
)

run.train()
```

### Mixed sources
```python
from tensor.instruct import Instruct
from tensor.instruct.data import Mix, LocalSource, HubSource

run = Instruct(
    base="./output/my-pretrained-base",
    data=Mix({
        LocalSource("./my-chats.jsonl"):                               0.8,
        HubSource("HuggingFaceH4/ultrachat_200k", split="train_sft"): 0.2,
    }),
    output="./output/my-instruct-model",
)

run.train()
```

---

## Configuration

`InstructConfig` is optional. When omitted, `tensor-instruct` infers sensible
defaults from the model size and available hardware.
```python
from tensor.instruct import Instruct, InstructConfig

run = Instruct(
    base="./output/my-pretrained-base",
    data="./my-chats.jsonl",
    output="./output/my-instruct-model",
    config=InstructConfig(
        epochs=3,
        devices=4,
        dtype="bfloat16",
        learning_rate=2e-5,
    ),
)

run.train()
```

| Parameter | Default | Notes |
|---|---|---|
| `epochs` | `3` | Full passes over the training dataset |
| `devices` | all visible GPUs | Number of GPUs to use |
| `dtype` | `"bfloat16"` | Recommended for A100+. Use `"float16"` on older hardware |
| `context_length` | auto | Max tokens per example. Inferred from model config, capped at 8192 |
| `batch_size_per_device` | auto | Inferred from VRAM and context length |
| `gradient_accumulation_steps` | `4` | Effective batch = `batch_size × devices × accumulation_steps` |
| `learning_rate` | `2e-5` | Lower than pretraining by design — instruct tuning shapes behaviour, not knowledge |
| `warmup_ratio` | `0.03` | Fraction of total steps used for linear LR warm-up |
| `save_steps` | `500` | Checkpoint every N steps. `0` = only save at end |
| `logging_steps` | `10` | Log training metrics every N steps |
| `seed` | `42` | Random seed for reproducibility |

---

## Data Sources

`data=` accepts a path string, a single source, or a weighted `Mix`. All are valid:
```python
data="./my-chats.jsonl"              # path string → auto-promoted to LocalSource
data=LocalSource("./my-chats.jsonl") # explicit single file
data=LocalSource("./my-data-dir/")   # directory — walked recursively for .jsonl files
data=HubSource("...", split="...")   # HuggingFace Hub dataset
data=Mix({ LocalSource(...): 0.8, HubSource(...): 0.2 })  # weighted mix
```

### HubSource options
```python
HubSource(
    repo_id="cognitivecomputations/dolphin",
    subset=None,                          # dataset config / subset name
    split="train",                        # dataset split
    messages_column="conversations",      # column holding the messages list
    role_mapping={                        # normalise non-standard role names
        "human": "user",
        "gpt": "assistant",
    },
)
```

`role_mapping` handles datasets that use alternative role naming conventions.
Common mappings:
- `{"human": "user", "gpt": "assistant"}` — Alpaca / Dolphin style
- `{"Human": "user", "Assistant": "assistant"}` — OpenAssistant style

---

## Validate and Estimate

Before committing to a full training run, use `validate()` and `estimate()` to
catch problems early and review projected time.
```python
run = Instruct(
    base="./output/my-pretrained-base",
    data="./my-chats.jsonl",
    output="./output/my-instruct-model",
)

run.validate()   # checks base model, data, and hardware — raises early if anything is wrong
run.estimate()   # prints example count, epoch plan, and wall time estimate
run.train()
```

`run.estimate()` output:
```
  base          ./output/my-pretrained-base
  data          LocalSource('./my-chats.jsonl')  →  ~12,400 examples
  context       4096 tokens max per example
  epochs        3
  devices       4× device(s), dtype=bfloat16
  est. time     ~2.3 hrs
  output        ./output/my-instruct-model
```

---

## Resuming
```python
from tensor.instruct import Instruct

run = Instruct.resume("./output/my-instruct-model")
run.train()
```

`Instruct.resume()` reads the run config and checkpoint state from the output
directory. No need to reconstruct the original arguments.

---

## Results

After `run.train()` completes, the result is available on the instance:
```python
run.train()

print(run.result.checkpoint)         # path to .safetensors output
print(run.result.examples_trained)   # examples × epochs
print(run.result.elapsed)            # wall time
```

The `.safetensors` checkpoint is your fully trained instruct model, ready to
take directly into `tensor-inference` for serving.

---

## Base Model Enforcement

`tensor-instruct` enforces raw base models at construction time. Any model
whose name or path contains an instruct/chat suffix raises `BaseModelError`
immediately — before any weights are downloaded or loaded.
```python
from tensor.instruct import Instruct

run = Instruct(
    base="Qwen/Qwen3-8B-Instruct",   # ← raises BaseModelError immediately
    ...
)
```
```
BaseModelError: 'Qwen/Qwen3-8B-Instruct' appears to be an instruction-tuned model.
tensor-instruct only accepts raw base models.
Use the base variant, or a checkpoint produced by tensor-pretrain.
```

This is intentional. Fine-tuning on top of an existing instruct model means
inheriting someone else's refusal patterns, tone decisions, and formatting
choices — all of which will bleed through and conflict with your own training
data. Start from a raw base every time.

---

## Loss Masking

Training loss is computed **only on assistant response tokens**. System and
user tokens are set to `-100` and are completely excluded from the gradient.

Within each assistant turn, the opening header (`<|im_start|>assistant\n`) is
also masked — loss is computed on the response content and the closing
`<|im_end|>` delimiter only. The model learns to produce responses and to
close them correctly. It does not receive a gradient for re-reading its own
context.

---

## ChatML Token Handling

For Qwen3-based models, `<|im_start|>` and `<|im_end|>` are already part of
the vocabulary — no action is taken. For other base models, `tensor-instruct`
automatically adds these tokens to the tokenizer and resizes the model
embedding table accordingly. This is transparent and requires no configuration.

---

## Building Production Agents

The Tensor Framework is designed specifically for agentic workflows, and
`tensor-instruct` is where you define exactly how your agent behaves.

Because you're starting from a raw base — either a `tensor-pretrain`
checkpoint with your domain knowledge already baked in, or a clean upstream
base — there is no prior fine-tuning to fight against. Your training data
defines the agent's output format, tone, tool-call structure, and refusal
behaviour entirely from scratch.
```
your domain data
    ↓
tensor-pretrain  →  domain-trained base
                         ↓
              your ChatML behavior data
                         ↓
              tensor-instruct  →  production instruct model
                                        ↓
                               tensor-inference
```

If your agent needs to output raw JSON, reject out-of-scope requests with a
specific error format, or follow a strict function-call schema — define that
in your JSONL. The model learns exactly what you show it, with nothing
inherited from a prior instruct tuning stage.

---

## Architecture

| Component | Role |
|---|---|
| `tensor.instruct.Instruct` | Main entry point — configuration, validation, estimation, and training |
| `tensor.instruct.Instruct.resume` | Reconstructs a run from an existing output directory checkpoint |
| `tensor.instruct.InstructConfig` | Optional training parameters — epochs, devices, dtype, learning rate |
| `tensor.instruct.data.LocalSource` | Reads `.jsonl` files from a local file or directory |
| `tensor.instruct.data.HubSource` | HuggingFace Hub dataset source with optional role normalisation |
| `tensor.instruct.data.Mix` | Multi-source weighted example mixing |

---

## Roadmap

- **v1.0 (Current):** Local JSONL and Hub data sources, ChatML-only, full-parameter fine-tuning, loss masking on assistant turns, resume support
- **v1.1:** Packed sequence training for improved GPU utilisation · per-epoch validation split with loss reporting
- **v1.2:** Direct export to `tensor-inference` optimised graph format

---

## License

Apache 2.0 — free for commercial and private use within the Tensor Framework ecosystem.

---

*Part of the **Tensor Framework** by Netangular.*