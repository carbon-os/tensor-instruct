# LLM Training Strategies: The Mental Model

There is often confusion between "learning new things" and "learning how to behave." This guide separates the three distinct stages of training, using the **"Brain vs. Glasses"** analogy to clarify when to use which method.

---

## 1. Continued Pre-training (CPT)

**"Reading the Textbook"**

This is not about chatting; this is about raw information absorption. You are taking a base model (which knows English, code, and general history) and forcing it to read thousands of documents from a new, specific domain (e.g., Law, Medicine, Internal Company Docs).

* **Goal:** Inject **new knowledge** or language capabilities.
* **Mechanism:** The model predicts the next token on raw text. No "User/Assistant" format.
* **Analogy:** A student sitting in a library reading a stack of medical textbooks. They aren't talking to anyone; they are just absorbing facts.
* **Result:** A smarter **Base Model**. It still can't chat; if you ask "What is aspirin?", it might autocomplete with "...and its molecular weight is..." rather than answering you.

---

## 2. Full Instruct Fine-Tuning (FFT)

**"Brain Surgery"**

This is the process of alignment—teaching the model *how* to use the facts it already has. You update **100% of the model's weights**.

* **Goal:** Deep behavioral alignment. Teaching the model to format answers, follow complex logic, and adopt a specific persona.
* **Mechanism:** Supervised Fine-Tuning (SFT) on Q&A pairs (User/Assistant).
* **Analogy:** **Brain Surgery.** You are physically rewiring the neural pathways of the brain. You are permanently altering how the model thinks and processes information to make "being helpful" the path of least resistance.
* **Best For:**
* **Small Models (0.5B – 14B):** Small brains need every neuron they have to understand complex instructions. Restricting them with adapters (LoRA) can make them "dumb."
* **Abundant Compute:** When you have an A100/H100 and the model fits in VRAM easily.


* **Risk:** **Catastrophic Forgetting.** If you cut too many wires, it might forget the facts it learned during pre-training.

---

## 3. LoRA Instruct Fine-Tuning

**"Putting on Glasses"**

Low-Rank Adaptation (LoRA) freezes the main model and trains tiny "adapter" layers that sit on top of the attention blocks. You only update ~1% of the parameters.

* **Goal:** Efficient behavioral alignment. Getting the model to chat without spending millions on compute.
* **Mechanism:** SFT on Q&A pairs, but gradients only update the adapter layers.
* **Analogy:** **Specialized Glasses.** The brain (base model) stays exactly the same—you don't touch it. You just put a pair of glasses on it that distorts the input and output, forcing the model to "see" the world as a helpful assistant.
* **Best For:**
* **Massive Models (70B+):** You physically cannot fit the gradients of a 70B model on a single GPU. LoRA is the only option.
* **Safety:** You cannot "break" the base model because you aren't touching it. If the LoRA is bad, just take the glasses off.


* **Trade-off:** Lower capacity. You are asking the model to learn a new behavior using only 1% of its brain power.

---

## Summary Comparison

| Feature | Continued Pre-training | Full Fine-Tuning (FFT) | LoRA Fine-Tuning |
| --- | --- | --- | --- |
| **Primary Goal** | **Knowledge** (Facts) | **Behavior** (Deep Habits) | **Behavior** (Surface Bias) |
| **Analogy** | Reading a Textbook | Brain Surgery | Specialized Glasses |
| **Weights Updated** | 100% | 100% | ~1% |
| **VRAM Usage** | Massive (High) | Massive (High) | Low |
| **Knowledge Retention** | High (Adds facts) | Risk of Forgetting | Perfect (Base is frozen) |
| **Ideal For** | New domains (Law, Code) | **Small Models (0.6B - 14B)** | **Huge Models (70B+)** |

---

## The Decision Matrix: What should I use?

### Scenario A: "I want the model to know about my company's internal API."

* **Strategy:** **Continued Pre-training** (on API docs) $\rightarrow$ **Full Instruct Fine-Tuning** (on Q&A).
* *Reason:* It needs new facts (CPT) and then needs to learn how to answer questions about them (FFT).

### Scenario B: "I want to make Qwen-0.6B speak like a pirate."

* **Strategy:** **Full Instruct Fine-Tuning.**
* *Reason:* The model is small. It needs its full capacity to master the "Pirate" dialect. LoRA might make it sound like a pirate who barely speaks English.

### Scenario C: "I want to make Llama-3-70B summarize news articles."

* **Strategy:** **LoRA.**
* *Reason:* A 70B model is brilliant but huge. You can't fine-tune it fully without 8x H100s. LoRA allows you to do it on a smaller setup, and the base model is smart enough that "glasses" are all it needs.