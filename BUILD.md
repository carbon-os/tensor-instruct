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

### RTX Pro 6000 / Blackwell (full stack)

**Step 1 — base install:**
```bash
pip install -e ".[blackwell]"
```

**Step 2 — install PyTorch 2.10 with CUDA 12.8:**
```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

**Step 3 — set CUDA paths (required for Triton and TE):**
```bash
export CUDA_HOME=/usr/local/cuda
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo 'export PYTORCH_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
```

**Step 4 — build Transformer Engine from source against your torch:**
```bash
pip install ninja
MAX_JOBS=$(nproc) pip install --no-build-isolation "transformer-engine[pytorch]"
```
This takes 5–30 minutes depending on CPU core count. It must be built from
source to match the torch ABI — conda prebuilts are tied to older torch versions
and will cause undefined symbol errors at runtime.

**Step 5 — install FlashAttention-2 (prebuilt, no compilation):**
```bash
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

> To build flash-attn from source instead (10–45 min):
> ```bash
> MAX_JOBS=$(nproc) pip install flash-attn --no-build-isolation
> ```

**Step 6 — verify the full stack:**
```bash
python3 -c "
import torch
import transformer_engine.pytorch as te
print('torch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('TE OK')
"
```

```bash
python3 -c "import flash_attn; print('FA2 OK:', flash_attn.__version__)"
```

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

## 5. Known Issues and Fixes (RTX Pro 6000 / Blackwell sm_120)

These patches are required in `tensor/instruct/_instruct.py` for stable
training on Blackwell. Apply them once after cloning.

**Fix 1 — disable torch.compile on Blackwell (Triton sm_120 support is unstable):**
```python
def _should_compile(gpu: dict) -> bool:
    if gpu["is_blackwell"]:
        return False
    try:
        major = int(torch.__version__.split(".")[0])
        return (
            major >= 2
            and torch.cuda.is_available()
            and gpu["free_vram_gb"] >= 20
        )
    except Exception:
        return False
```

**Fix 2 — correct compile mode string (hyphen not underscore):**
```python
compile_mode = "max-autotune" if gpu["sm_count"] >= 100 else "reduce-overhead"
```

**Fix 3 — always use gradient checkpointing (required at 8192 context):**
```python
use_grad_ckpt = True
```

**Fix 4 — reduce batch size for 8192 context on high-VRAM cards:**
```python
if context_length >= 8192:
    if vram_gb >= 64:  return 4
```

**Fix 5 — cast FP8 Linear layers to match model dtype:**

In `_wrap_fp8`, after the bias copy add:
```python
te_linear = te_linear.to(child.weight.dtype)
```

**Fix 6 — remove max_autotune from Blackwell optimisations banner:**
```python
    unlocked = []
    if gpu["has_fp8"]:
        unlocked.append("FP8 training")
    if gpu["is_blackwell"]:
        unlocked.append("FlashAttention-3 (if installed)")
    elif gpu["is_hopper"]:
        unlocked.append("max-autotune compile")
```

---

## 6. HuggingFace Authentication

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

## 7. File Structure

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

## 8. Prepare Your Data

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

## 9. Your First Training Run
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

## 10. Common Configurations

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

## 11. Multi-GPU with torchrun

For 2+ GPU runs, `torchrun` handles process spawning automatically.
`tensor-instruct` picks up the distributed environment through the
HuggingFace `Trainer` / `accelerate` stack — no code changes needed.
```bash
torchrun --nproc_per_node=4 train.py
```

Set `devices=` in `InstructConfig` to match `--nproc_per_node`.

---

## 12. Output

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

## 13. Troubleshooting

**`BaseModelError` at startup**
You passed an instruct or chat model. Switch to the base variant — remove
the `-Instruct`, `-it`, `-Chat`, or `-sft` suffix.

**`RuntimeError: No valid training examples were produced`**
Your JSONL data either doesn't parse or doesn't match the expected schema.
Run `run.validate()` first. Make sure every line has a top-level `messages`
key and at least one `assistant` turn.

**`CUDA out of memory`**
Reduce `batch_size_per_device` or `context_length` in `InstructConfig`.
For a 16GB GPU try `batch_size_per_device=1, context_length=1024`.
Gradient checkpointing is always enabled.

**`FP8 wrap failed — undefined symbol`**
Transformer Engine was installed from conda and mismatches your pip torch ABI.
Reinstall TE from source:
```bash
pip uninstall transformer-engine transformer-engine-torch -y
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
MAX_JOBS=$(nproc) pip install --no-build-isolation "transformer-engine[pytorch]"
```

**`RemoveError: 'setuptools' is a dependency of conda`**
You are trying to install into the `base` conda environment. Always use a
dedicated environment:
```bash
conda create -n tensor python=3.12 -y
conda activate tensor
```

**`RuntimeError: Unrecognized mode=max_autotune`**
The compile mode string uses a hyphen not an underscore. See Fix 2 in
section 5.

**`CUDA error: an illegal memory access` during autotuning**
Triton autotuning is unstable on Blackwell sm_120. Apply Fix 1 in section 5
to disable `torch.compile` on Blackwell entirely.

**`huggingface_hub.errors.RepositoryNotFoundError`**
The model is gated. Run `huggingface-cli login` and accept the model
licence on huggingface.co.

**`bfloat16 not supported`**
Your GPU does not support bf16 (requires Ampere or newer). Switch to
`dtype="float16"` in `InstructConfig`.

**All examples skipped, 0 kept**
Every line in your JSONL either has no `assistant` turn, has fewer than
two messages, or is malformed JSON. Check the schema in section 8.