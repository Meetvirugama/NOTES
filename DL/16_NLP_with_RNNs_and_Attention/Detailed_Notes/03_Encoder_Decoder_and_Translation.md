# 🌐 Module 3: Encoder-Decoder Networks and Neural Machine Translation
> **Ch. 16 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [Step 0: Special Tokens](#tokens)
3. [Why a Standard RNN Can't Do Translation](#standard-fails)
4. [The Encoder-Decoder Architecture: Step by Step](#architecture)
5. [Teacher Forcing: Stable Training](#teacher-forcing)
6. [Inference: Generating the Translation](#inference)
7. [Beam Search: Smarter Decoding with Numbers](#beam-search)
8. [Full Keras Implementation](#keras-impl)
9. [Limitations of Classical Encoder-Decoder](#limitations)
10. [Key Terms Dictionary](#terms)
11. [Common Beginner Mistakes](#mistakes)
12. [Interview Q&A (Top 8)](#interview)
13. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** An Encoder-Decoder network maps a variable-length input sequence to a variable-length output sequence, translating between languages with different lengths and word orders.

An Encoder-Decoder network maps a **variable-length input sequence** to a **variable-length output sequence**. The input and output can differ in length, word order, and vocabulary.

**Applications:**
| Task | Input | Output |
|------|-------|--------|
| Machine Translation | "I love you" (EN, 3 words) | "Je t'aime" (FR, 2 words) |
| Text Summarization | 500-word article | 3-sentence summary |
| Question Answering | Passage + Question | 1-sentence answer |
| Speech Recognition | Audio waveform (2000 ms) | "Hello World" (2 words) |

**The Key Challenge:**
Input length ≠ Output length, and word order can be **completely different** between languages.

```
English: "The cat sat on the mat"     (6 words, Subject-Verb pattern)
French:  "Le chat s'est assis sur le tapis"  (7 words)
Spanish: "El gato se sentó en la alfombra"   (7 words, but adjective order flips)
German:  "Die Katze saß auf der Matte"       (6 words, verb at end in complex sentences)
```

---

## 🏷️ Step 0: Special Tokens (<SOS>, <EOS>, <PAD>, <UNK>) {#tokens}

> **TL;DR:** Sequence models must know when a sentence starts and ends, especially since the output length is dynamic. We use special tokens to guide the decoder.

One of the most important topics that beginners often miss is why sequence models need special tokens.

Why can't we just train on normal sentences? Suppose the target sentence is `Je t'aime`. The decoder needs to know:
1. When to start
2. When to stop

Without special tokens, it has no idea. During inference, without an `<EOS>` token, the decoder would continue predicting words forever.

**Four Important Tokens:**
| Token | Meaning | Usage |
|-------|---------|-------|
| `<SOS>` | Start Of Sentence | First token fed to the decoder to trigger generation. |
| `<EOS>` | End Of Sentence | Tells the decoder to STOP generating. |
| `<PAD>` | Padding | Fills empty spaces in a batch to make tensors uniform length. |
| `<UNK>` | Unknown Word | Used for out-of-vocabulary words during tokenization. |

**Example:**
Original sentence: `Je t'aime`
Training sentence: `<SOS> Je t'aime <EOS>`
Token IDs: `[1, 34, 92, 2]`

**During Inference:**
Input `<SOS>` → predicts `Je` → predicts `t'aime` → predicts `<EOS>` → **STOP**.

---

## 🚦 Why a Standard RNN Can't Do Translation {#standard-fails}

> **TL;DR:** Standard RNNs process and output simultaneously. Since languages have different grammatical structures (e.g., subject-verb order), the model needs to "read" the entire input sentence before translating it correctly.

**Option A: Many-to-Many RNN (synchronized):**

```
Time:    t=1   t=2    t=3    t=4
Input:  [Je] [t'aime] [X]   [X]     ← French input
Output: [I]  [love]   [you] [X]     ← English output
```

**Problem:** The model must output English word 1 while STILL reading French word 1. It cannot wait to "see" the full French sentence before translating. 

- French "Je ne mange pas" → "I don't eat" (order: I, don't, eat)
- But French grammar is: Subject-Negation-Verb-Negation ("Je ne mange pas")
- Without seeing the WHOLE input, the model cannot know that "ne" means this is a negative sentence!

**Option B: Many-to-One then One-to-Many:**

```
Encoder Phase: Read "Je t'aime" → context vector c
Decoder Phase: c → "I love you"
```

This works! The Encoder first reads the ENTIRE input, then the Decoder generates the translation.

---

## 🏗️ The Encoder-Decoder Architecture: Step by Step {#architecture}

> **TL;DR:** The Encoder reads the full input to create a summary vector. The Decoder uses this vector to predict the translation word-by-word, autoregressively mapping a probability distribution via Softmax at every step.

![Encoder Decoder Architecture](../Visuals/07_encoder_decoder.png)
> 📊 **Graph 06:** The Encoder reads the source sentence and compresses it into the Context Vector (the final hidden state). The Decoder uses the context vector as its starting memory and generates the translated sentence word by word.

### Phase 1: The Encoder

**Input:** English sentence "I love you" → token IDs: `[3, 18, 12]`

The Encoder is a standard LSTM/GRU. It processes each word and updates its hidden state:

```
Initial state:  h_0 = [0, 0, ..., 0]  (zeros, shape: 256D)

Step 1: h_1 = LSTM(embed("I"),    h_0)  → [0.12, -0.45, ...]  (256D)
Step 2: h_2 = LSTM(embed("love"), h_1)  → [0.82,  0.31, ...]  (256D)
Step 3: h_3 = LSTM(embed("you"),  h_2)  → [0.67, -0.12, ...]  (256D)
```

We throw away `h_1` and `h_2`. **Only `h_3` (the final state) is kept.** This is the **Context Vector** `c`.

### ❓ Why does the Encoder use only the Final State?
Many students ask: *"Why do we throw away all the intermediate hidden states?"*
Because in a standard RNN, each hidden state inherently contains the cumulative information seen so far.
- `h_1` → "I"
- `h_2` → "I love"
- `h_3` → "I love you"

Therefore, `h_3` already contains the essence of `h_1` and `h_2`. The classical Seq2Seq model assumes this single final vector is sufficient to summarize the whole sentence. *(Note: Later, Attention mechanisms remove this limitation by using all states).*

### 🧠 LSTM States: Hidden State vs Cell State
While we conceptually refer to the "Context Vector," an LSTM actually produces two distinct state vectors at the final step:
- **Hidden State ($h$)**: The short-term working memory (also the current output).
- **Cell State ($c$)**: The long-term memory track.

The Encoder returns **both** `state_h` and `state_c`. Both are passed together to initialize the Decoder. This explains why Keras returns `output, state_h, state_c` instead of just one vector!

> **Intuition:** `h_3` and `c_3` together are like a summary note that the Encoder passes to the Decoder saying *"This was a sentence expressing loving affection toward someone."*

### Phase 2: The Decoder

The Decoder is initialized with `c = h_3` as its starting state. It then generates words in the target language one at a time.

```
Initial state: s_0 = c = h_3 (from Encoder!)

Step 1: Input = <SOS> token
        Probs = softmax(Dense(LSTM(<SOS>, s_0)))
        → {'Je': 0.72, 'Tu': 0.11, 'Il': 0.08, ...}
        → Select "Je"

Step 2: Input = embed("Je")
        Probs = softmax(Dense(LSTM(embed("Je"), s_1)))
        → {"t'aime": 0.65, "suis": 0.12, "vais": 0.09, ...}
        → Select "t'aime"

Step 3: Input = embed("t'aime")
        Probs = softmax(Dense(LSTM(embed("t'aime"), s_2)))
        → {"<EOS>": 0.89, "bien": 0.06, ...}
        → Select "<EOS>" → STOP
```

**Final output:** "Je t'aime" ✅

### 🔄 Why does the Decoder Predict One Word at a Time?
The decoder cannot predict the whole sentence simultaneously because language is autoregressive. Each prediction strictly depends on:
1. The encoder's context.
2. The *previously generated words*.

Mathematically, it calculates joint probability via the chain rule:
$P(Y|X) = \prod_{t} P(y_t | y_{<t}, X)$

Which means:
*Probability(sentence) = Prob(word1) × Prob(word2 | word1) × Prob(word3 | previous words) ...*

### 🎲 Why Softmax is Used
At every decoder step, the Hidden State is passed through a Dense layer to produce **Logits** (raw scores). We apply the **Softmax** function to convert these logits into a valid probability distribution over the entire vocabulary.
- Suppose Vocabulary is: `Je, Tu, Il, Elle`
- Output logits: `[2.8, 0.9, 0.3, -1.2]`
- After Softmax: `Je (0.78), Tu (0.15), Il (0.05), Elle (0.02)`

The largest probability becomes the selected next word!

### 📐 Shapes of Every Tensor (Crucial for Interviews)
Interviewers frequently ask about tensor shapes. Knowing them prevents many implementation bugs.
Suppose: `Batch size = 64`, `Input length = 12`, `Embedding size = 256`, `Hidden size = 512`

**Encoder Shapes:**
- Input: `(64, 12)`
- Embedding Output: `(64, 12, 256)`
- LSTM Final State Output: `(64, 512)`

**Decoder Shapes:**
- Input: `(64, 15)` *(assuming target length 15)*
- Embedding Output: `(64, 15, 256)`
- LSTM Output (full sequence): `(64, 15, 512)`
- Dense (Softmax) Output: `(64, 15, vocab_size)`

**The information bottleneck:**
The entire meaning of "I love you" is compressed into a 256D vector. For short sentences, this works well. For 50-word sentences, critical information from the beginning gets washed out by the time the encoder reads the end.

---

## 👨‍🏫 Teacher Forcing: Stable Training {#teacher-forcing}

> **TL;DR:** Training without Teacher Forcing is unstable because the model learns from its own mistakes. Teacher Forcing fixes this by always feeding the correct previous word during training.

**The Problem Without Teacher Forcing:**

```
Target:  "Je t'aime <EOS>"
Step 1: Model inputs <SOS>, predicts "Tu" (WRONG! should be "Je")
Step 2: Model inputs "Tu", predicts "veux" (another wrong word, compounding the error!)
Step 3: Model inputs "veux", predicts "pas" (garbage input → garbage output)
```

Each mistake cascades into the next step. The model never gets clean input, making training unstable and slow.

**Teacher Forcing Solution:**

During training, we **IGNORE the model's own predictions** and always feed it the **ground truth previous word**.

```
Target sequence: "Je t'aime <EOS>"
Decoder inputs:  "<SOS> Je t'aime"   ← This is just target, shifted right by 1!
Decoder targets: "Je t'aime <EOS>"   ← What we want the decoder to predict
```

```
Step 1: Input = <SOS>       → Target = "Je"     → Loss computed
Step 2: Input = "Je"        → Target = "t'aime" → Loss computed
Step 3: Input = "t'aime"    → Target = "<EOS>"  → Loss computed
```

Even if step 1 is wrong, step 2 gets the correct "Je" input. Training is fast and stable.

### 🛠️ Complete Training Data Preparation
Many learners understand teacher forcing conceptually but struggle with how to prepare the tensors. Here is the exact "Shift Trick":

```python
# Target sentence: "<SOS> Je t'aime <EOS>"
# Token IDs:       [1, 15, 48, 2]

# Decoder input (shift right)
decoder_input = target_seq[:, :-1]   # [1, 15, 48]

# Decoder target (shift left)
decoder_target = target_seq[:, 1:]   # [15, 48, 2]

# Training call:
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data
)
```
This perfectly illustrates how the decoder learns to predict the next token based on the true previous tokens.

### 📉 Sparse Categorical Crossentropy
Why do we use `loss="sparse_categorical_crossentropy"` instead of Binary Crossentropy?
Because every decoder step is predicting ONE word out of thousands (e.g., a 5,000-word vocabulary). 

The loss function compares:
- The predicted probability distribution (array of 5000 floats).
- The correct word index (e.g., `word #183`).

The `sparse` version allows targets to remain as simple integers (`183`, `91`, `22`) instead of blowing up memory with giant one-hot encoded vectors (`[0, 0, ..., 1, ..., 0]`). It is vastly more memory efficient.

### ⚠️ Exposure Bias
Teacher forcing is fantastic for training, but it introduces a subtle problem.
- **During training:** The decoder always sees the *correct* previous word.
- **During inference:** The decoder sees *its own predictions*.

If the model predicts one wrong word during inference (e.g., predicting `Tu` instead of `Je`), the sentence will drift away into gibberish because it has never been trained on how to recover from its own mistakes. 
This mismatch is called **Exposure Bias**.
> **Solution:** **Scheduled Sampling** (or Professor Forcing). You start training with 100% teacher forcing, and gradually transition to letting the model feed its own predictions back into itself during later epochs.

---

## 🔮 Inference: Generating the Translation {#inference}

> **TL;DR:** Inference is fundamentally different from training. Without ground truth, the model must feed its own predictions back into itself in a slow, step-by-step loop.

### ⚖️ Training vs Inference Difference
This is one of the most frequently asked interview questions! The decoder behaves fundamentally differently during training vs testing.

**TRAINING:**
- Uses **Teacher Forcing** (Ground Truth).
- **Fast:** Parallel over the whole sequence.
- `<SOS> → Je → t'aime → <EOS>` (all at once).

**INFERENCE:**
- Uses **Model Predictions** (No Ground Truth).
- **Slow:** Must wait for Step $t$ to finish before Step $t+1$ can start.
- `<SOS> → Je (predicted) → feed back → t'aime (predicted) → feed back → <EOS>`.

At inference time, we must decode step-by-step:

```python
def decode_sequence(input_seq, encoder_model, decoder_model, 
                    target_tokenizer, max_len=50):
    """
    input_seq: shape [1, input_len]
    """
    # Step 1: Encode the input sentence → get context vector
    states_value = encoder_model.predict(input_seq)  # → [h_state, c_state] for LSTM
    
    # Step 2: Initialize decoder with <SOS> token
    target_seq = np.array([[target_tokenizer.word_index["<SOS>"]]])  # shape: [1, 1]
    
    translation = []
    stop = False
    
    while not stop:
        # Step 3: Decode one step
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Step 4: Sample a token
        sampled_idx = np.argmax(output_tokens[0, -1, :])   # Greedy decoding
        sampled_word = target_tokenizer.index_word.get(sampled_idx, "<UNK>")
        
        if sampled_word == "<EOS>" or len(translation) > max_len:
            stop = True
        else:
            translation.append(sampled_word)
        
        # Step 5: Feed sampled token back as next input
        target_seq = np.array([[sampled_idx]])
        states_value = [h, c]   # Update decoder states
    
    return " ".join(translation)

print(decode_sequence([[3, 18, 12]]))
# → "Je t'aime"
```

---

## 🔦 Beam Search: Smarter Decoding with Numbers {#beam-search}

> **TL;DR:** Greedy search only looks one step ahead and can miss the globally best sentence. Beam search keeps track of the top-k most promising paths at all times, leading to more fluent translations.

### 💡 Why Beam Search Works Better
**Greedy Search** always picks the single best immediate word.
**Beam Search** keeps the top-k complete hypotheses.

Let's look at the math:
- **Greedy Path:** `<SOS> → Je → suis → ...` Probability = $0.72 \times 0.05 = 0.036$
- **Beam Path:** `<SOS> → Tu → aimes → ...` Probability = $0.11 \times 0.60 = 0.066$

Although $0.11 < 0.72$ (so Greedy would never pick "Tu" initially), the *overall sequence* probability for the second path is much higher ($0.066 > 0.036$). Hence, Beam Search often produces much more fluent and accurate translations.

**Beam Search with k=3:**

![Beam Search](../Visuals/08_beam_search.png)
> 📊 **Graph 07:** Beam Search with k=3. At each step, we keep the top-k partial sequences. We expand EACH of the k sequences into the full vocabulary, compute joint probabilities, and keep only the top-k new sequences.

**Step-by-step with real numbers (k=3):**

```
Start: <SOS>

STEP 1: Expand <SOS> → evaluate all vocab words
  Probabilities:
  <SOS> "Je"     : P=0.72  (log: -0.33)
  <SOS> "Tu"     : P=0.11  (log: -2.21)
  <SOS> "Il"     : P=0.08  (log: -2.53)
  <SOS> "Elle"   : P=0.06  (log: -2.81)
  ...
  
  Keep TOP 3: ["Je", "Tu", "Il"]
  Beam scores: [-0.33, -2.21, -2.53]

STEP 2: Expand EACH of the 3 beams → 3 × vocab_size candidates
  From "Je":   "Je t'aime": 0.72×0.65=0.47 (log: -0.76)
               "Je suis"  : 0.72×0.05=0.04 (log: -3.22)
               "Je vais"  : 0.72×0.03=0.02 (log: -3.91)
               ...
  From "Tu":   "Tu t'aimes": 0.11×0.50=0.06 (log: -2.81)
               "Tu es"     : 0.11×0.30=0.03 (log: -3.51)
               ...
  From "Il":   "Il t'aime" : 0.08×0.55=0.04 (log: -3.22)
               ...

  Keep TOP 3: 
  1. "Je t'aime"  (log score: -0.76)
  2. "Tu t'aimes" (log score: -2.81)
  3. "Il t'aime"  (log score: -3.22)

STEP 3: Expand each beam again...
  From "Je t'aime":   "Je t'aime <EOS>": 0.47×0.89=0.42  ← strong candidate!
  From "Tu t'aimes":  ...
  From "Il t'aime":   ...

FINAL: Select sequence with highest total log-probability:
  "Je t'aime <EOS>"  →  log P = -0.76 + log(0.89) = -0.76 + (-0.12) = -0.88
```

**Why log probabilities?**

Multiplying many small probabilities causes numerical underflow:
$0.72 × 0.65 × 0.89 × 0.95 × ... = \text{infinitesimally small}$

Log converts multiplication to addition (no underflow):
$\log(0.72) + \log(0.65) + \log(0.89) + ... = -0.33 + (-0.43) + (-0.12) + ...$

**Length Penalty:**
Without it, Beam Search prefers SHORT sequences (fewer multiplications of < 1). 
Common fix: $\text{score} = \frac{\sum \log P}{L^\alpha}$ where $L$ is length and $\alpha \approx 0.7$.

---

## 💻 Full Keras Implementation {#keras-impl}

> **TL;DR:** The Keras implementation requires building a training model (that uses Teacher Forcing) and separate inference models (that use manual loops).

### 🗺️ End-to-End Data Flow Diagram
A final summary diagram tying everything together before the code:

```text
 English Sentence
        │
        ▼
    Tokenizer
        │
        ▼
   Integer IDs
        │
        ▼
    Embedding
        │
        ▼
   Encoder LSTM
        │
        ▼
Context Vector (h, c)
        │
        ▼
 ┌─────────────────────────────┐
 │ Decoder (Teacher Forcing)   │
 │ Input: <SOS> + target[:-1]  │
 └─────────────────────────────┘
        │
        ▼
   Decoder LSTM
        │
        ▼
  Dense + Softmax
        │
        ▼
Predicted Next Word
        │
        ▼
 Repeat until <EOS>
```

```python
from tensorflow import keras
import numpy as np

# Parameters
encoder_vocab = 5000    # Source language vocab size (English)
decoder_vocab = 5000    # Target language vocab size (French)
embed_dim = 256
latent_dim = 512        # LSTM hidden state size

# ════════════════════════════════════════════
# ENCODER
# ════════════════════════════════════════════
encoder_inputs = keras.layers.Input(shape=[None])
enc_emb = keras.layers.Embedding(encoder_vocab, embed_dim)(encoder_inputs)
_, state_h, state_c = keras.layers.LSTM(
    latent_dim, 
    return_state=True    # Returns: (output, final_h, final_c)
)(enc_emb)
encoder_states = [state_h, state_c]   # The Context Vector (for LSTM = 2 states)

# ════════════════════════════════════════════
# DECODER (training mode — uses teacher forcing)
# ════════════════════════════════════════════
decoder_inputs = keras.layers.Input(shape=[None])  # Teacher-forced targets
dec_emb_layer = keras.layers.Embedding(decoder_vocab, embed_dim)
dec_emb = dec_emb_layer(decoder_inputs)

dec_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = dec_lstm(
    dec_emb,
    initial_state=encoder_states   # Initialize with encoder context!
)

dec_dense = keras.layers.Dense(decoder_vocab, activation="softmax")
decoder_probs = dec_dense(decoder_outputs)

# ════════════════════════════════════════════
# FULL TRAINING MODEL
# ════════════════════════════════════════════
model = keras.Model(
    inputs=[encoder_inputs, decoder_inputs],
    outputs=decoder_probs
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Training on (English_X, French_input), French_target
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=100,
    validation_split=0.2
)
```

---

## 🚧 Limitations of Classical Encoder-Decoder {#limitations}

> **TL;DR:** Compressing an entire sentence into a single vector creates an information bottleneck, causing performance to crash on long sentences.

### 📉 Why Standard Seq2Seq Fails on Long Sentences
**Information Bottleneck:** 
```text
Word1 → Word2 → ... → Word50 → Context Vector → Decoder
```
Everything must pass through one vector. Words at the beginning of the sentence have passed through ~50 LSTM matrix multiplications. Their signal is dramatically attenuated. 

Problems:
- Long sentences lose early context.
- Complex grammar is forgotten.
- Rare words are overpowered by frequent words.
- Distant dependencies are lost.

### 📋 Limitations Summary Checklist
1. ✓ **Fixed-size context vector** cannot scale to infinite length.
2. ✓ **Information bottleneck** drops nuance.
3. ✓ **Weak on long sentences** (performance drops after 30+ words).
4. ✓ **Slow decoding** (inherently word-by-word autoregressive).
5. ✓ **Training/inference mismatch** (Exposure Bias from Teacher Forcing).
6. ✓ **Cannot easily align** source and target words (e.g., matching a French adjective to an English noun).

**Solved later by:** Attention Mechanisms & Transformers!

---

## 📖 Key Terms Dictionary {#terms}

| Term | Simple Definition |
|------|-------------------|
| **`<SOS>` / `<EOS>`** | Special tokens that tell the decoder exactly when to Start and Stop generating text. |
| **Context Vector** | The final hidden state (`h_T`) of the Encoder, acting as a fixed-size mathematical summary of the entire input sentence. |
| **Autoregressive** | A model property where the current prediction depends on all previously generated predictions. |
| **Teacher Forcing** | A training technique where the decoder is fed the *true* previous word instead of its own prediction, stabilizing gradients. |
| **Exposure Bias** | The vulnerability caused by Teacher Forcing where the model crashes during inference because it has never seen its own mistakes. |
| **Beam Search** | A decoding algorithm that maintains the top-$k$ most likely sequences at each step to find a globally better translation than greedy search. |

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. Training model with teacher forcing but forgetting to build inference models** ❌
> **Why it's bad:** During training, the encoder and decoder are fused into one model because teacher forcing provides all inputs at once. During inference, you must feed predictions back in a loop. If you try to use the training model for inference, it will fail because it expects the full target sequence as input.
> **Fix:** You must build SEPARATE `encoder_model` and `decoder_model` objects for inference, where the decoder is explicitly built to accept its own previous states as inputs.

**2. Expecting the Context Vector to remember long sentences** ❌
> **Why it's bad:** Compressing a 50-word sentence into a 512D vector causes severe information loss. Words at the beginning of the sentence have passed through ~50 LSTM matrix multiplications. Their signal is dramatically attenuated.
> **Fix:** Keep sentences short for classical Seq2Seq models, or upgrade to Attention Mechanisms which give the decoder direct access to all encoder states.

---

## 🎤 Interview Q&A (Top 3) {#interview}

**Q1: Explain Teacher Forcing and why it's needed.**
> **A:** Without teacher forcing, the untrained decoder makes mistakes early on. Since its own (wrong) predictions are fed as the next input, errors compound catastrophically — the decoder is essentially trying to learn from garbage inputs. Teacher forcing solves this by feeding the ground-truth previous token at each step during training. This stabilizes gradients and massively speeds up convergence. The cost: a training/inference mismatch (during inference, the model must use its own predictions, having never practiced this). Scheduled sampling is one mitigation strategy.

**Q2: Why do we use log probabilities in Beam Search instead of raw probabilities?**
> **A:** Multiplying many conditional probabilities (each < 1) leads to numerical underflow in floating point arithmetic. For example, $0.7^{50} \approx 1.8 \times 10^{-10}$, which rounds to 0.0 in float32. Taking the logarithm converts multiplication to addition: $\log \prod P_t = \sum \log P_t$, which remains numerically stable for any sequence length.

**Q3: What happens to performance as input sentence length increases in a standard Encoder-Decoder?**
> **A:** Performance degrades sharply. The context vector has fixed capacity (e.g., 512D). As sentence length grows, more information must be compressed into the same sized vector, causing information loss. Bahdanau (2015) showed in experiments that BLEU scores for a standard Encoder-Decoder plateau around 30 words then drop significantly for 40+ word sentences. With Attention, performance stays much more stable with length.

---

## ⚡ Flash Card Cheat Sheet {#revision}

```
╔═════════════════════════════════════════════════════════════════════════╗
║         MODULE 3 CHEAT SHEET: ENCODER-DECODER & TRANSLATION             ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  ARCHITECTURE:                                                            ║
║  Encoder: Reads ALL input → discards intermediate states → keeps h_T    ║
║  Context Vector: h_T (512D) — single bottleneck vector                  ║
║  Decoder: Starts from h_T → generates one word per step until <EOS>    ║
║                                                                           ║
║  TEACHER FORCING (training):                                             ║
║  Decoder input  = <SOS> + target[:-1]   (ground truth shifted right)   ║
║  Decoder target = target[1:]             (ground truth shifted left)    ║
║  ✅ Stable gradients  ❌ Training/inference mismatch                   ║
║                                                                           ║
║  INFERENCE (greedy):                                                      ║
║  1. Encode input → get encoder states                                   ║
║  2. Input <SOS> to decoder                                              ║
║  3. Get prediction → feed back as next input                            ║
║  4. Repeat until <EOS>                                                  ║
║                                                                           ║
║  BEAM SEARCH (k=3):                                                      ║
║  Keep top-k partial sequences at each step.                             ║
║  Score = Σ log P(y_t | y_<t, X) — sum of log probabilities              ║
║  Add length penalty: score / L^α (α≈0.7) to avoid short-seq bias      ║
║                                                                           ║
╚═════════════════════════════════════════════════════════════════════════╝
```

---

---

**🔗 Previous Module →** [02_Sentiment_Analysis_and_Word_Embeddings.md](02_Sentiment_Analysis_and_Word_Embeddings.md)  
**🔗 Next Module →** [04_Attention_Mechanisms.md](04_Attention_Mechanisms.md)
