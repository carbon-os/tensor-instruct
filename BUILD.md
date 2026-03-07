# BUILD.md — tensor-instruct

Complete guide to cloning, installing, and running your first instruct fine-tuning job.

---

## Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| CUDA | 11.8+ (for GPU training) |
| GPU | NVIDIA GPU with 16GB+ VRAM (A100 recommended) |
| Disk | 50GB+ free for model weights + dataset cache |

CPU-only works for smoke-testing but is not viable for real training runs.

---

## 1. Clone
```bash
git clone https://github.com/carbon-os/tensor-instruct.git
cd tensor-instruct
```

---

## 2. Set Up Environment

Create and activate a virtual environment first — this keeps the heavy ML
dependencies isolated.
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

---

## 3. Install

### Standard install (recommended)
```bash
pip install --upgrade pip setuptools
pip install -e .
```

The `-e` flag installs in editable mode — changes to the source files take
effect immediately without reinstalling.

### With FlashAttention-2 (A100 / H100, faster training)
```bash
pip install -e ".[flash-attn]"
```

FlashAttention-2 is auto-detected at runtime. If the package is present it
is used automatically — no config change needed.

### Verify the install
```bash
python -c "from tensor.instruct import Instruct, InstructConfig; print('OK')"
```

---

## 4. HuggingFace Authentication

Some base models on HuggingFace Hub are gated. If your base is a local
`tensor-pretrain` checkpoint you can skip this step entirely.
```bash
pip install huggingface_hub
huggingface-cli login
```

Paste your token when prompted. Tokens can be created at
https://huggingface.co/settings/tokens — use a `read` token.

Alternatively set the environment variable directly:
```bash
export HF_TOKEN=hf_...
```

---

## 5. File Structure

After cloning and installing the structure should look like this:
```
tensor-instruct/
├── tensor/
│   ├── __init__.py
│   └── instruct/
│       ├── __init__.py
│       ├── _config.py
│       ├── _dataset.py
│       ├── _instruct.py
│       ├── _result.py
│       └── data/
│           ├── __init__.py
│           ├── _mix.py
│           └── _sources.py
├── docs/
├── pyproject.toml
├── README.md
└── BUILD.md
```

---

## 6. Prepare Your Data

`tensor-instruct` expects data in **ChatML format**. Each line of your
`.jsonl` file is one conversation:
```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is gradient descent?"}, {"role": "assistant", "content": "Gradient descent is an optimisation algorithm that minimises a loss function by iteratively moving in the direction of steepest descent."}]}
{"messages": [{"role": "user", "content": "Write a SQL query that counts rows grouped by date."}, {"role": "assistant", "content": "SELECT DATE(created_at), COUNT(*) FROM your_table GROUP BY DATE(created_at) ORDER BY DATE(created_at);"}]}
```

Rules:
- Every line must be valid JSON with a top-level `messages` key.
- Each message must have a `role` (`system`, `user`, or `assistant`) and a `content` string.
- `system` turns are optional. If omitted, the conversation starts from the first `user` turn.
- Multi-turn conversations are fully supported — include as many alternating `user`/`assistant` turns as you need.
- Lines that are malformed, missing `messages`, or contain no `assistant` turn are silently skipped.

---

## 7. Your First Training Run
```python
from tensor.instruct import Instruct

run = Instruct(
    base="./output/my-pretrained-base",   # tensor-pretrain output, or any HF base ID
    data="./my-chats.jsonl",              # your ChatML JSONL file or directory
    output="./output/my-instruct-model",
)

run.validate()    # catches problems before wasting GPU time
run.estimate()    # prints example count + wall time estimate
run.train()       # runs the full pipeline

print(run.result)
```

Save this as `train.py` and run it:
```bash
python train.py
```

---

## 8. Common Configurations

### Local JSONL, single file
```python
from tensor.instruct import Instruct

run = Instruct(
    base="./output/my-pretrained-base",
    data="./my-chats.jsonl",
    output="./output/my-instruct-model",
)

run.train()
```

### Local JSONL, directory of files
```python
from tensor.instruct import Instruct

run = Instruct(
    base="./output/my-pretrained-base",
    data="./my-chatml-data/",     # walks recursively for all .jsonl files
    output="./output/my-instruct-model",
)

run.train()
```

### HuggingFace Hub dataset
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

### Hub dataset with non-standard role names
Some Hub datasets use `"human"` / `"gpt"` instead of `"user"` / `"assistant"`.
Use `role_mapping` to normalise them:
```python
from tensor.instruct.data import HubSource

HubSource(
    "cognitivecomputations/dolphin",
    messages_column="conversations",
    role_mapping={"human": "user", "gpt": "assistant"},
)
```

### Mixed sources (local + Hub)
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

### With explicit config
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

### Using a raw HuggingFace base instead of a tensor-pretrain checkpoint
```python
from tensor.instruct import Instruct

run = Instruct(
    base="Qwen/Qwen3-8B-Base",
    data="./my-chats.jsonl",
    output="./output/my-instruct-model",
)

run.train()
```

### Resume an interrupted run
```python
from tensor.instruct import Instruct

run = Instruct.resume("./output/my-instruct-model")
run.train()
```

---

## 9. Multi-GPU with torchrun

For 2+ GPU runs, `torchrun` handles process spawning automatically.
`tensor-instruct` picks up the distributed environment through the
HuggingFace `Trainer` / `accelerate` stack — no code changes needed.
```bash
torchrun --nproc_per_node=4 train.py
```

Set `devices=` in `InstructConfig` to match `--nproc_per_node`.

---

## 10. Output

After `train()` completes the output directory contains:
```
output/my-instruct-model/
├── model/
│   ├── model.safetensors          # your instruct fine-tuned model
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── tensor_instruct_state.json     # run metadata (used by resume)
```

The `model/` directory is a self-contained HuggingFace-compatible checkpoint.
Load it anywhere `from_pretrained` is accepted, or pass it directly to
`tensor-inference` for high-velocity serving.

---

## 11. Troubleshooting

**`BaseModelError` at startup**
You passed an instruct or chat model. Switch to the base variant — remove
the `-Instruct`, `-it`, `-Chat`, or `-sft` suffix. If you want to fine-tune
on top of a `tensor-pretrain` output, pass the local path to the checkpoint
directory instead.

**`RuntimeError: No valid training examples were produced`**
Your JSONL data either doesn't parse or doesn't match the expected schema.
Run `run.validate()` first — it will print the example count per source.
Make sure every line has a top-level `messages` key and at least one
`assistant` turn.

**`CUDA out of memory`**
Reduce `batch_size_per_device` or `context_length` in `InstructConfig`.
For a 16GB GPU try `batch_size_per_device=1, context_length=1024`.
Gradient checkpointing is always enabled — this is already the main
memory-saving lever.

**`huggingface_hub.errors.RepositoryNotFoundError`**
The model is gated. Run `huggingface-cli login` and accept the model
licence on huggingface.co.

**`bfloat16 not supported`**
Your GPU does not support bf16 (requires Ampere or newer). Switch to
`dtype="float16"` in `InstructConfig`.

**All examples skipped, 0 kept**
Every line in your JSONL either has no `assistant` turn, has fewer than
two messages, or is malformed JSON. Check that your data follows the
schema in section 6.

---

## License

Apache 2.0 — free for commercial and private use within the Tensor Framework ecosystem.