# BUILD.md — tensor-instruct

Complete guide to cloning, installing, and running your first instruct fine-tuning job.

---

## Recommended GPU Rental

| Use case | GPU | VRAM | Price |
|---|---|---|---|
| Serious training runs | A100 SXM | 80GB | ~$1.49/hr |
| Smoke tests / iteration | L40S | 48GB | ~$0.86/hr |

The A100 SXM is the primary target for this guide. FlashAttention-2 runs natively
on it, all tooling is mature and stable, and it hits ~70% of theoretical peak FLOPs
out of the box. The L40S is a solid iteration card — use it to validate your data
pipeline and config before committing to a full A100 run.

---

## Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| CUDA | 11.8+ |
| GPU | A100 SXM (80GB) for training · L40S (48GB) for smoke tests |
| Disk | 50GB+ free for model weights + dataset cache |

CPU-only works for smoke-testing but is not viable for real training runs.

---

## 1. Clone
```bash
git clone https://github.com/carbon-os/tensor-instruct.git
cd tensor-instruct
```

---

## 2. Install Conda (Miniconda)
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init bash
source ~/.bashrc
```

Verify:
```bash
conda --version
```

---

## 3. Set Up Environment

Always create a dedicated conda environment — never install into `base` as it
is reserved for conda's own tooling and will cause conflicts.
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n tensor python=3.12 -y
conda activate tensor
```

---

## 4. Install

### Standard install
```bash
pip install --upgrade pip setuptools
pip install -e .
```

### With FlashAttention-2 (A100 SXM / L40S — recommended)
```bash
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

FlashAttention-2 is fully supported and stable on both the A100 SXM and L40S.
It is auto-detected at runtime — if the package is present it is used
automatically with no config change needed. FA2 brings the A100 SXM to ~70%
of theoretical peak FLOPs, which is the best you will get on this architecture.

### Verify the install
```bash
python -c "from tensor.instruct import Instruct, InstructConfig; print('OK')"
python -c "import flash_attn; print('FA2 OK:', flash_attn.__version__)"
```

---

## 5. HuggingFace Authentication

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

## 6. File Structure

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

## 7. Prepare Your Data

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

## 8. Your First Training Run
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

## 9. Common Configurations

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

### With explicit config (A100 SXM — full run)
```python
from tensor.instruct import Instruct, InstructConfig

run = Instruct(
    base="./output/my-pretrained-base",
    data="./my-chats.jsonl",
    output="./output/my-instruct-model",
    config=InstructConfig(
        epochs=3,
        devices=4,          # 4x A100 SXM via torchrun
        dtype="bfloat16",   # A100 SXM supports bf16 natively
        learning_rate=2e-5,
    ),
)

run.train()
```

### With explicit config (L40S — smoke test / iteration)
```python
from tensor.instruct import Instruct, InstructConfig

run = Instruct(
    base="./output/my-pretrained-base",
    data="./my-chats.jsonl",
    output="./output/my-instruct-model",
    config=InstructConfig(
        epochs=1,
        devices=1,
        dtype="bfloat16",
        batch_size_per_device=2,
        context_length=2048,    # reduce if hitting OOM on 48GB
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

## 10. Multi-GPU with torchrun (A100 SXM)

The A100 SXM nodes listed in the GPU rental table support up to 8x GPUs.
`torchrun` handles process spawning automatically — no code changes needed.
```bash
torchrun --nproc_per_node=8 train.py
```

Set `devices=` in `InstructConfig` to match `--nproc_per_node`.

---

## 11. Output

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

## 12. Troubleshooting

**`BaseModelError` at startup**
You passed an instruct or chat model. Switch to the base variant — remove
the `-Instruct`, `-it`, `-Chat`, or `-sft` suffix.

**`RuntimeError: No valid training examples were produced`**
Your JSONL data either doesn't parse or doesn't match the expected schema.
Run `run.validate()` first. Make sure every line has a top-level `messages`
key and at least one `assistant` turn.

**`CUDA out of memory` on L40S**
The L40S has 48GB VRAM. Reduce `batch_size_per_device` or `context_length`
in `InstructConfig`. Try `batch_size_per_device=1, context_length=2048` as
a safe starting point. If you need longer context or larger batch sizes,
move to an A100 SXM node.

**`CUDA out of memory` on A100 SXM**
Reduce `batch_size_per_device` or `context_length` in `InstructConfig`.
At 8192 context start with `batch_size_per_device=4`. Gradient checkpointing
is always enabled.

**`bfloat16 not supported`**
Both the A100 SXM and L40S support bf16 natively (Ampere and Ada architectures).
If you see this error your rental node may have provisioned a different GPU
than expected — verify with `python -c "import torch; print(torch.cuda.get_device_name(0))"`.

**`FP8 wrap failed — undefined symbol`**
Transformer Engine was installed from conda and mismatches your pip torch ABI.
Reinstall TE from source:
```bash
pip uninstall transformer-engine transformer-engine-torch -y
MAX_JOBS=$(nproc) pip install --no-build-isolation "transformer-engine[pytorch]"
```

**`RemoveError: 'setuptools' is a dependency of conda`**
You are trying to install into the `base` conda environment. Always use a
dedicated environment:
```bash
conda create -n tensor python=3.12 -y
conda activate tensor
```

**`huggingface_hub.errors.RepositoryNotFoundError`**
The model is gated. Run `huggingface-cli login` and accept the model
licence on huggingface.co.

**All examples skipped, 0 kept**
Every line in your JSONL either has no `assistant` turn, has fewer than
two messages, or is malformed JSON. Check the schema in section 7.