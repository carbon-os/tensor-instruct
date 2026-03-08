"""
Quick inference test for a tensor-instruct output.
Not part of the Tensor Framework — just a sanity-check script.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "/root/testing/output/my-instruct-model/model"

SYSTEM_PROMPT = "You are a helpful assistant."

MESSAGES = [
    {"role": "user", "content": "Explain what a transformer neural network is in simple terms."},
]

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.7
TOP_P          = 0.9

# ── Load ──────────────────────────────────────────────────────────────────────

print(f"Loading model from {MODEL_PATH} …")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False,
)
model.eval()
print("Model loaded.\n")

# ── Format prompt ─────────────────────────────────────────────────────────────

# Build the full messages list with system prompt prepended.
full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + MESSAGES

# Use the tokenizer's built-in chat template if available,
# otherwise fall back to manual ChatML formatting.
if hasattr(tokenizer, "apply_chat_template"):
    prompt = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
else:
    parts = []
    for msg in full_messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    prompt = "\n".join(parts)

# ── Inference ─────────────────────────────────────────────────────────────────

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("=== Prompt ===")
for msg in MESSAGES:
    print(f"[{msg['role']}] {msg['content']}")
print()
print("=== Response ===")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode only the newly generated tokens, not the prompt.
new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(response)
print()

# ── Interactive loop (optional) ───────────────────────────────────────────────

print("=== Interactive mode (Ctrl+C to exit) ===\n")

history = list(full_messages)

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        parts = []
        for msg in history:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

    history.append({"role": "assistant", "content": response})

    print(f"\nAssistant: {response}\n")