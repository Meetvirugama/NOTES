# 🎯 Module 4: Attention Mechanisms — Deep Dive
> **Ch. 16 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [🌍 Start Here: The Big Picture](#big-picture)
2. [🔍 The Core Concept: Query, Key, Value (Q, K, V)](#qkv)
3. [🔄 Types of Attention Mechanisms](#types)
4. [🧠 Classic Seq2Seq Math (Bahdanau & Luong)](#classic)
5. [💻 Implementing Attention in Keras](#implementation)
6. [❌ Common Beginner Mistakes](#mistakes)
7. [🎤 Interview Q&A (Top 5)](#interview)
8. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** RNNs cram entire sentences into one fixed-size vector, creating a bottleneck and forgetting early words. Attention fixes this by giving the model a dynamic "memory" of all past words. 

### The Bottleneck Problem (Why standard RNNs fail)

**Standard Encoder-Decoder:**
```
"Romeo loves Juliet and they both tragically die for love"
        ↓ (10 words compressed into one 256D vector)
        c = [0.21, -0.43, ...]   ← 256 numbers must describe ALL of this!
        ↓
"Roméo aime Juliette et ils meurent tous les deux tragiquement pour l'amour"
```
The Decoder reads 12 French words from a single summary. Information about "Romeo" is diluted by the 9 subsequent words.

For a 50-word sentence, the gradient flowing from word 50 to word 1 passes through 49 multiplications of numbers $\le 1$, resulting in severe **vanishing gradients** (e.g., $0.9^{49} \approx 0.005$).

### The Attention Solution

Instead of one Context Vector for the ENTIRE translation, compute a **unique context vector for each decoder step**, by dynamically attending to different encoder positions.

```
Generating "Roméo"   → Focus mostly on "Romeo"
Generating "aime"    → Focus mostly on "loves"  
Generating "meurent" → Focus mostly on "die", "tragically"
```
> 💡 **Intuition:** Think of how YOU translate. You don't memorize the whole sentence. Your eyes dart back to the source text for each word. Attention gives the model eyes.

---

## 🔍 The Core Concept: Query, Key, Value (Q, K, V) {#qkv}

> **TL;DR:** Attention is built on the Q, K, V paradigm. It works exactly like a database or search engine.

Every token produces a **Query (Q)**, **Key (K)**, and **Value (V)**.

- 🔎 **Query (Q):** What am I looking for? (e.g., "I need a subject noun")
- 🏷️ **Key (K):** What information do I contain? (e.g., "I am a singular noun")
- 📦 **Value (V):** What should I send if selected? (e.g., the actual semantic vector for the word)

### Scaled Dot-Product Attention

The attention score is computed by taking the dot product of the Query and the Key, then applying softmax to get weights, and finally multiplying by the Value.

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Why Divide by $\sqrt{d_k}$?**
If the embedding dimension $d=1024$, dot products grow massive.
A softmax of $[999, 1001, 1005]$ becomes almost one-hot $\rightarrow$ **Vanishing gradients.**
Scaling by $\frac{1}{\sqrt{d_k}}$ keeps values stable and gradients healthy.

---

## 🔄 Types of Attention Mechanisms {#types}

> **TL;DR:** Attention is just a routing mechanism. How you arrange the Q, K, and V determines what the attention does.

| Type | How it Works | Where it's Used |
|------|-------------|-----------------|
| **Cross-Attention** | Q comes from Decoder. K, V come from Encoder. | Translation, Seq2Seq |
| **Self-Attention** | Q, K, V all come from the SAME sequence. Every token attends to every other token. | Transformers, GPT, BERT |
| **Multi-Head Attention** | Run multiple attention operations in parallel (e.g., Head 1: Grammar, Head 2: Verb). Concatenate at the end. | Almost everywhere |
| **Causal (Masked) Attention** | Mask future tokens with $-\infty$ before softmax. Token $t$ cannot look at token $t+1$. | Autoregressive LLMs (GPT) |

---

## 🧠 Classic Seq2Seq Math (Bahdanau & Luong) {#classic}

> **TL;DR:** The two foundational attention algorithms from 2014/2015. They map an RNN Encoder to an RNN Decoder.

### Bahdanau (Additive) Attention
- **Scores using:** A small feed-forward network with a `tanh` activation.
- **Uses Decoder State:** Previous state $s_{t-1}$.
- **Context Vector applied:** BEFORE the decoder LSTM runs.

**The Math:**
$$e_{t,i} = \mathbf{v}_a^T \tanh(\mathbf{W}_a s_{t-1} + \mathbf{U}_a h_i)$$
Softmax the scores $e$ to get weights $\alpha$. Context $c_t = \sum \alpha_{t,i} h_i$.

### Luong (Multiplicative) Attention
- **Scores using:** Dot product (much faster!).
- **Uses Decoder State:** Current state $s_t$.
- **Context Vector applied:** AFTER the decoder LSTM runs.

**The Math (Dot Scoring):**
$$e_{t,i} = s_t^T h_i$$
Because it's a simple dot product, it takes $O(T \cdot d)$ instead of $O(T \cdot d^2)$. This is the standard used in Keras (`keras.layers.Attention`).

### Reading the Alignment Matrix

If we plot the softmax weights $\alpha_{t,i}$:
- **Rows:** Target words
- **Columns:** Source words
- **Explainability:** Off-diagonal bright spots reveal where the model learned grammatical inversions (e.g., adjective-noun swaps between English and French).

> ⚠️ **"Attention Is Not Explanation":** While heatmaps look great, modern research shows high attention weight $\neq$ true causal importance. It's a useful diagnostic, not absolute ground truth.

---

## 💻 Implementing Attention in Keras {#implementation}

```python
from tensorflow import keras

# 1. ENCODER (Must return sequences!)
encoder_inputs = keras.layers.Input(shape=[None])
enc_emb = keras.layers.Embedding(vocab_size, embed_dim)(encoder_inputs)
# return_sequences=True gives ALL states for attention to query
encoder_outputs, state_h, state_c = keras.layers.LSTM(
    256, return_sequences=True, return_state=True
)(enc_emb)
encoder_state = [state_h, state_c]

# 2. DECODER
decoder_inputs = keras.layers.Input(shape=[None])
dec_emb = keras.layers.Embedding(vocab_size, embed_dim)(decoder_inputs)
decoder_lstm = keras.layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_state)

# 3. ATTENTION (Luong-style dot-product)
attention_layer = keras.layers.Attention()
# Query = decoder outputs, Key/Value = encoder outputs
context_vector = attention_layer([decoder_outputs, encoder_outputs])

# 4. COMBINE AND OUTPUT
decoder_combined = keras.layers.Concatenate(axis=-1)([decoder_outputs, context_vector])
output = keras.layers.Dense(vocab_size, activation="softmax")(decoder_combined)
```

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. Forgetting `return_sequences=True` on the Encoder** ❌
> Reality: Without it, the LSTM only returns the final timestep. Attention needs the full sequence of states to look back at!

**2. Treating Bahdanau and Luong as modern LLM architecture** ❌
> Reality: They are Seq2Seq cross-attention mechanisms designed for RNNs. Modern LLMs use Self-Attention (where Q, K, V come from the same sentence) inside Transformers.

**3. Ignoring the $\sqrt{d_k}$ scaling factor in custom implementations** ❌
> Reality: Without scaling, the dot products explode, pushing the softmax into regions with near-zero gradients. The model will fail to train.

---

## 🎤 Interview Q&A (Top 5) {#interview}

**Q1: What problem does Attention solve in Encoder-Decoder networks?**
> **A:** It solves the bottleneck problem. Standard Seq2Seq compresses the entire source sentence into one fixed-size context vector, leading to information loss for long sentences. Attention allows the decoder to dynamically "look back" at specific encoder states at every decoding step.

**Q2: What is the difference between Self-Attention and Cross-Attention?**
> **A:** In **Self-Attention**, Query, Key, and Value all come from the same sequence to learn internal relationships. In **Cross-Attention** (e.g., translation encoder-decoder), Queries come from the Decoder, while Keys and Values come from the Encoder.

**Q3: Why is attention called "scaled" dot-product attention?**
> **A:** The raw dot product $QK^T$ is divided by $\sqrt{d_k}$ to prevent the variance from growing too large with high dimensions. Without it, the softmax distribution becomes overly peaked (one-hot), leading to vanishing gradients.

**Q4: How do you implement masking for autoregressive models?**
> **A:** By applying a causal mask before the softmax step. All positions corresponding to "future" tokens are set to $-\infty$, which softmax turns into a $0$ weight.

**Q5: In Bahdanau Attention, what is the alignment score computing?**
> **A:** It measures the "compatibility" between the current decoder state and each encoder state. A high score means "encoder position $i$ contains information highly relevant to what the decoder is trying to generate next."

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║                    MODULE 4 — ATTENTION FLASH CARD               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  THE PROBLEM WITH RNNs:                                          ║
║  1. Fixed-size bottleneck (forgets long sequences)               ║
║  2. Vanishing gradients across long time steps                   ║
║                                                                  ║
║  THE CORE ENGINE (Q, K, V):                                      ║
║  Query (What I want) * Key (What I have) = Score                 ║
║  Attention = Softmax( Q·K^T / √d_k ) * V                         ║
║                                                                  ║
║  ATTENTION TYPES:                                                ║
║  • Self-Attention:  Q, K, V from same sequence (Transformers)    ║
║  • Cross-Attention: Q from Decoder, K/V from Encoder (Seq2Seq)   ║
║  • Causal Masked:   Mask future tokens with -inf (GPT)           ║
║                                                                  ║
║  CLASSIC SEQ2SEQ MATH:                                           ║
║  Bahdanau = additive (tanh), uses s_{t-1}                        ║
║  Luong    = multiplicative (dot), uses s_t (Default in Keras)    ║
║                                                                  ║
║  ⚠️ IMPLEMENTATION TRAP:                                         ║
║  Must use return_sequences=True on Encoder LSTM so the Decoder   ║
║  has a sequence of Key/Values to attend to!                      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

---

**🔗 Previous Module →** [03_Encoder_Decoder_and_Translation.md](03_Encoder_Decoder_and_Translation.md)  
**🔗 Next Module →** [05_The_Transformer_Architecture.md](05_The_Transformer_Architecture.md)
