# 🚀 Module 6: Recent Innovations in Language Models
> **Ch. 16 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [🌍 The Three Generations of Language Models (The Big Picture)](#big-picture)
2. [🎭 Generation 2: ELMo (Contextual LSTMs)](#elmo)
3. [🏗️ Generation 3: The Foundation Model Paradigm](#foundation)
4. [🔍 The BERT Family (Understanding)](#bert)
5. [🗣️ The GPT Family (Generation)](#gpt)
6. [🛠️ Adapting Foundation Models to Tasks (Fine-Tuning, LoRA, RAG)](#adapting)
7. [🤖 Alignment: Making Models Helpful and Safe (RLHF & DPO)](#alignment)
8. [🚀 The Frontier: 2024 and Beyond (MoE, Agents, Reasoning)](#frontier)
9. [🌳 Complete Family Tree](#family-tree)
10. [❌ Common Beginner Mistakes](#mistakes)
11. [🎤 Interview Q&A (Top 6 Modern Questions)](#interview)
12. [⚡ One-Page Flash Card Cheat Sheet](#revision)

---

## 🌍 The Three Generations of Language Models (The Big Picture) {#big-picture}

> **TL;DR:** NLP evolved from static vectors (1 word = 1 vector) to contextual vectors (1 word = dynamic vector) to massive foundation models trained on the entire internet.

### Generation 1: Static Embeddings
**Examples:** Word2Vec, GloVe, FastText
**The Problem:** `Word2Vec("bank")` always returns the SAME vector, whether it's a "river bank" or a "financial bank". It just averages the meanings together.

### Generation 2: Contextual Embeddings
**Examples:** ELMo, ULMFiT
**The Solution:** The entire sentence is passed through an RNN (like a Bidirectional LSTM). The vector for "bank" is computed dynamically based on the surrounding words.

### Generation 3: Foundation Models
**Examples:** GPT, BERT, T5, Llama, Gemini
**The Paradigm Shift:** Instead of "train one model for one task", modern AI trains **one huge Transformer** on unlabelled text from the entire internet. This base model is then adapted for any downstream task.

---

## 🎭 Generation 2: ELMo (Contextual LSTMs) {#elmo}

**Paper:** *"Deep contextualized word representations"* — Peters et al., Allen AI, 2018

ELMo computes dynamic word vectors using a deep **Bidirectional LSTM**.

```
Input: "I went to the river bank to fish"

Left-to-right LSTM (forward):
  h→_5 = LSTM_fwd(embed("bank"),  h→_4)  → captures "I went to the river" context

Right-to-left LSTM (backward):
  h←_5 = LSTM_bwd(embed("bank"),  h←_6)  → captures "to fish" context

ELMo embedding for "bank" = concat([h→_5, h←_5]) from ALL layers
```

**Numerical Example — "bank" disambiguation:**
```
Context A: "river bank"
  ELMo("bank") → [0.71, -0.23, 0.55, ...]  ← "geographical feature" region

Context B: "bank account"  
  ELMo("bank") → [-0.31, 0.82, -0.12, ...]  ← "financial institution" region

cosine_similarity(Context A, Context B) ≈ 0.11   ← Very different! ✅
```

---

## 🏗️ Generation 3: The Foundation Model Paradigm {#foundation}

Today's Foundation Models split into three architectural branches based on the Transformer:

| Architecture | Famous Examples | Best For |
|--------------|-----------------|----------|
| **Encoder-only** | BERT, RoBERTa, DeBERTa | Understanding (Classification, NER, Search) |
| **Decoder-only** | GPT, Llama, Mistral, Gemma, Qwen | Generation (Chatbots, Coding, Reasoning) |
| **Encoder-Decoder** | T5, FLAN-T5, BART | Translation, Summarization |

### T5 (Text-to-Text Transfer Transformer)
Google's T5 proved that *everything* can be framed as a text-to-text problem.
- **Translation:** `translate English to French: Hello` → `Bonjour`
- **Classification:** `sentiment: Amazing movie` → `positive`
- **Question Answering:** `question: Capital of Japan?` → `Tokyo`

---

## 🔍 The BERT Family (Understanding) {#bert}

**Paper:** *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* (2018)
**Architecture:** Stacked Transformer **Encoder** blocks.

### Masked Language Modeling (MLM)
BERT is trained to fill in the blanks using bidirectional context:
`"The cat [MASK] on the mat"` → Predicts "sat".

Of every token selected for replacement:
- **80%** → `[MASK]`
- **10%** → Random word
- **10%** → Keep original word

### The BERT Descendants
- **RoBERTa (Meta):** Removed Next Sentence Prediction, trained longer on more data, used dynamic masking. Outperformed original BERT.
- **ALBERT (Google):** Parameter sharing across layers + factorized embeddings. Smaller memory footprint, faster training.
- **DeBERTa:** Separated content embeddings from position embeddings (disentangled attention). Massive boost in understanding benchmarks.

---

## 🗣️ The GPT Family (Generation) {#gpt}

**Paper:** *"Improving Language Understanding by Generative Pre-Training"* (2018)
**Architecture:** Stacked Transformer **Decoder** blocks.

**Pre-Training Objective:** Next Token Prediction.
*(Self-supervised, causal/left-to-right attention only)*

### The Scaling Evolution
- **GPT-1 (2018):** 117M parameters. Required fine-tuning for tasks.
- **GPT-2 (2019):** 1.5B parameters. Exhibited **zero-shot generation** (answering questions without any fine-tuning).
- **GPT-3 (2020):** 175B parameters. Proved **Scaling Laws** (performance scales predictably with compute, data, and parameters). Mastered few-shot learning.

---

## 🛠️ Adapting Foundation Models to Tasks (Fine-Tuning, LoRA, RAG) {#adapting}

Once a massive model is pre-trained, it must be adapted to a specific use case.

### 1. Full Fine-Tuning (e.g., with BERT)
Update all weights of the model. Here is a classic Keras example of fine-tuning BERT:

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

encoded = tokenizer(["This movie was brilliant!"], return_tensors="tf")
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)  # Very small LR!
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
# model.fit(...)
```

### 2. Parameter-Efficient Fine-Tuning (PEFT)
Full fine-tuning is impossible for a 70B parameter LLM on consumer hardware.
- **LoRA (Low-Rank Adaptation):** Freeze the massive original model. Train only tiny, low-rank adapter matrices. Inexpensive, low VRAM usage.
- **QLoRA:** Combines 4-bit quantization with LoRA. Allows fine-tuning massive models on a single GPU.

### 3. Quantization
Store weights using fewer bits to save VRAM and speed up inference.
- FP32 (100% memory) → FP16 (50%) → INT8 (25%) → INT4 (12.5%).

### 4. RAG (Retrieval-Augmented Generation)
**Problem:** LLMs hallucinate and their knowledge is frozen in time.
**Solution:** `Question → Retriever → Finds relevant documents → Feeds to LLM → LLM answers`. 
Provides up-to-date facts without retraining.

---

## 🤖 Alignment: Making Models Helpful and Safe (RLHF & DPO) {#alignment}

A raw pretrained GPT model just continues text. If you ask it a question, it might reply with another question. We must **align** it.

### 1. Instruction Tuning
Fine-tune the model on `Instruction → Desired Response` pairs (e.g., "Summarize this...", "Translate...").

### 2. RLHF (Reinforcement Learning from Human Feedback)
The pipeline that created ChatGPT:
`Pretraining → Supervised Fine-Tuning → Reward Model (predicts human preference) → PPO (Reinforcement Learning)`

### 3. Constitutional AI (Anthropic)
The model learns from an explicit set of principles (a "constitution"). It generates an answer, performs a **self-critique**, and revises it. Reduces reliance on expensive human raters.

### 4. DPO (Direct Preference Optimization)
Modern replacement for PPO. Optimize the LLM directly on `Preferred Answer vs Rejected Answer`. Simpler, stable, and requires no reward model. (Used by Llama 3).

---

## 🚀 The Frontier: 2024 and Beyond (MoE, Agents, Reasoning) {#frontier}

### Mixture of Experts (MoE)
Instead of activating every neuron, activate only a few expert networks per token.
`Router → Sends token to Expert 5 and Expert 19`
**Advantage:** Massive parameter count with low computation cost per token (e.g., Mixtral, DeepSeek-MoE).

### Multimodal Models
Everything (Text, Images, Audio, Video) is converted into embeddings and processed by the same Transformer backbone. (e.g., GPT-4o, Gemini 1.5, Claude 3.5).

### Agentic AI & Function Calling
Models don't just generate text; they interact with tools.
`Question → Reason → Call Tool (Search/Calculator) → Read Result → Final Answer`
**Function Calling:** The LLM generates a structured JSON object (`{"city": "Tokyo"}`) which the application executes.

### Reasoning Models (e.g., OpenAI o1)
A new class of models focused on structured reasoning, multi-step planning, and deliberate tool use rather than just fluent text generation.

### Long Context Windows
Original GPT-1 had a 512-token context. Modern models (Gemini 1.5 Pro) support **1M to 2M+ tokens**, enabled by RoPE, FlashAttention, and KV-cache optimizations.

---

## 🌳 Complete Family Tree {#family-tree}

```text
Word2Vec
      │
      ▼
ELMo (Contextual LSTMs)
      │
      ▼
Transformer (2017)
      │
      ├────────────────────────┐
      │                        │
      ▼                        ▼
BERT (Encoder)           GPT-1 (Decoder)
      │                        │
      ▼                        ▼
RoBERTa                  GPT-2
ALBERT                   GPT-3
DeBERTa                  InstructGPT / ChatGPT
      │                        │
      └─────────┬──────────────┘
                ▼
      Modern Foundation Models
      (Llama, Gemini, Claude, Mistral, Qwen, T5)
```

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. Using BERT to generate text** ❌
> Reality: BERT is NOT generative. It was trained to fill in blanks (MLM). Generating text requires iterative masking and has an exponential search space. Use GPT for generation.

**2. Fine-tuning BERT with a huge learning rate** ❌
> Reality: Adam with `lr=0.001` will destroy BERT's pretrained weights (catastrophic forgetting). Always use a tiny learning rate (e.g., `2e-5`) and a warmup schedule.

**3. Confusing Instruction Tuning with RLHF** ❌
> Reality: Instruction tuning is just supervised learning on (prompt, response) pairs. RLHF/DPO optimizes the model based on human *preferences* (choosing response A over response B).

---

## 🎤 Interview Q&A (Top 6 Modern Questions) {#interview}

**Q1: Why did decoder-only models become dominant for LLMs?**
> **A:** Decoder-only Transformers are naturally autoregressive, making them perfectly suited for scalable pretraining on next-token prediction and text generation. They also support in-context learning seamlessly.

**Q2: What is the difference between instruction tuning and RLHF/DPO?**
> **A:** Instruction tuning teaches a model to follow instructions using supervised examples. RLHF (or DPO) further aligns the model with human preferences by explicitly training the model to maximize a reward or choose a preferred response over a rejected one.

**Q3: What is RAG?**
> **A:** Retrieval-Augmented Generation (RAG) retrieves relevant external documents at inference time and places them in the prompt. This gives the model up-to-date, grounded information without requiring expensive retraining, significantly reducing hallucinations.

**Q4: Why is LoRA widely used instead of full fine-tuning?**
> **A:** Full fine-tuning requires updating billions of weights, consuming massive VRAM. LoRA keeps the pretrained model frozen and trains only tiny, low-rank adapter matrices. This reduces memory requirements drastically while maintaining high accuracy.

**Q5: What is the difference between BERT and T5?**
> **A:** BERT is an encoder-only model optimized purely for language understanding (classification, NER). T5 uses an encoder-decoder architecture and formulates *every* task (even classification) as a text-to-text generation problem.

**Q6: What is a Mixture of Experts (MoE) network?**
> **A:** Instead of routing every token through one massive Feed-Forward Network, MoE uses a router to send each token to a small subset of "expert" networks. This gives the model massive capacity while keeping active computation per token low.

---

## ⚡ One-Page Flash Card Cheat Sheet {#revision}

```
╔══════════════════════════════════════════════════════════════════════╗
║          MODULE 6 CHEAT SHEET: MODERN LANGUAGE MODELS                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  THE THREE GENERATIONS:                                                ║
║  1. Static Embeddings (Word2Vec): Same vector for "bank" always.       ║
║  2. Contextual (ELMo): BiLSTM. Different vectors based on context.     ║
║  3. Foundation Models: Massive Transformers, self-supervised on Web.   ║
║                                                                        ║
║  FOUNDATION ARCHITECTURES:                                             ║
║  • Encoder-Only (BERT): Bidirectional. Best for Understanding.         ║
║    ↳ Trained on Masked LM (MLM). RoBERTa, ALBERT, DeBERTa.             ║
║  • Decoder-Only (GPT): Causal (Left-to-Right). Best for Generation.    ║
║    ↳ Trained on Next Token Prediction. Llama, Mistral, Qwen.           ║
║  • Encoder-Decoder (T5): Text-to-Text format for everything.           ║
║                                                                        ║
║  ALIGNMENT (Making models chatty & safe):                              ║
║  • Instruction Tuning: Train on Instruction → Response pairs.          ║
║  • RLHF: Uses a Reward Model and PPO to match human preferences.       ║
║  • DPO: Simpler modern alternative to RLHF (No reward model needed).   ║
║                                                                        ║
║  ADAPTATION & EFFICIENCY:                                              ║
║  • LoRA: Freeze base model, train tiny low-rank adapters (saves VRAM). ║
║  • Quantization: Shrink weights from FP32 to INT8/INT4 (saves memory). ║
║  • RAG: Retrieve documents from a database at runtime to stop          ║
║         hallucinations and provide fresh facts.                        ║
║                                                                        ║
║  THE FRONTIER:                                                         ║
║  • MoE: Router + Experts (huge capacity, cheap inference).             ║
║  • Agents: Models that use tools (Search, Code Exec) via JSON.         ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

---

**🔗 Previous Module →** [05_The_Transformer_Architecture.md](05_The_Transformer_Architecture.md)  
**🔗 Chapter Complete! →** [Back to Chapter Index](../notes.md)
