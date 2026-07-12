# 🧠 Module 5: Processing Sequences Using 1D CNNs and WaveNet
> **Ch. 15 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [The Big Picture](#big-picture)
2. [1D Convolutions for Sequences](#conv1d-foundations)
3. [Causal Padding](#causal-padding)
4. [Hybrid CNN-RNN Architectures](#hybrid-architectures)
5. [WaveNet and Dilated Convolutions](#wavenet)
6. [WaveNet Gated Activation & Residual Blocks](#wavenet-details)
7. [Bidirectional RNNs / BiLSTMs](#bilstm)
8. [Stacked LSTMs](#stacked-lstm)
9. [Stateful LSTMs](#stateful-lstm)
10. [Temporal Convolutional Networks (TCN)](#tcn)
11. [Common Mistakes](#mistakes)
12. [Interview Q&A](#interview)
13. [⚡ Flash Card](#revision)

---

## 🌍 1. The Big Picture {#big-picture}

> **TL;DR:** RNNs process one step at a time. 1D CNNs process a window of steps in parallel — much faster. WaveNet stacks dilated 1D CNNs to see a huge portion of the sequence with very few layers.

**Analogy 📷:**
| Architecture | What it is like |
| :--- | :--- |
| RNN | Watching a video frame by frame, in real time |
| 1D CNN | Laying the tape flat and scanning 5-minute clips side by side |
| WaveNet | Looking at frame 1, 2, 4, 8, 16 — scanning the full 24-hour tape in seconds |

**When to use what:**

| Task | Best Choice |
| :--- | :--- |
| Short sequences, simple patterns | Simple RNN / GRU |
| Long sequences, speed matters | 1D CNN or TCN |
| Audio / raw waveform generation | WaveNet |
| Hybrid: long input + recurrent output | Conv1D + GRU/LSTM |

---

## 🔍 2. 1D Convolutions for Sequences {#conv1d-foundations}

> **TL;DR:** A 1D Conv slides a kernel of size $K$ along the time axis. Every output step is computed independently — fully parallel on GPU.

![WaveNet Architecture](../Visuals/11_wavenet_architecture.png)
> 📊 **Graph 11:** A stack of 1D dilated causal convolutions. Each layer doubles the dilation rate: 1, 2, 4, 8 — expanding how far back each output can see.

**Kernel math:**

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

**Why prefer this over RNNs?**

| Property | RNN | 1D CNN |
| :--- | :--- | :--- |
| Computation | Sequential (step by step) | Parallel (all at once) |
| Training speed | Slow | Fast |
| Long-range memory | Needs LSTM/GRU | Needs dilation |
| Gradient stability | Vanishing gradients | Stable |

---

## 🔍 3. Causal Padding {#causal-padding}

> **TL;DR:** Causal padding adds zeros only on the left side of the sequence. This ensures the output at step $t$ only uses inputs up to and including step $t$ — no future leakage.

**The problem with standard padding:**

```
Standard padding adds zeros on BOTH sides:
[0] [x1] [x2] [x3] [x4] [0]
              ↑
         Output at x2 can "see" x3, x4 → DATA LEAKAGE ❌
```

**Causal padding (left-only zeros):**

```
[0] [0] [x1] [x2] [x3] [x4]
          ↑
     y1 depends only on x1, 0, 0  ✅
```

In Keras, set `padding="causal"` — it handles the zero-padding automatically.

> 💡 **Rule:** Any model that forecasts the future MUST use causal padding. Using `same` or `valid` padding on sequence data is almost always a bug.

---

## 🔍 4. Hybrid CNN-RNN Architectures {#hybrid-architectures}

> **TL;DR:** Use Conv1D with stride > 1 to shorten the sequence first, then pass the result to GRU/LSTM layers. This cuts the number of recurrent steps and saves memory.

**Downsampling formula** (with `padding="valid"` and stride $S$):

$$N_{\text{out}} = \left\lfloor \frac{N_{\text{in}} - K}{S} \right\rfloor + 1$$

**Worked Example:**
- Input: 50 time steps, kernel size 4, stride 2
- $N_{\text{out}} = \lfloor(50 - 4) / 2\rfloor + 1 = 24$ steps
- Result: 50 steps → 24 steps (52% reduction before the RNN)

```python
from tensorflow import keras

model = keras.models.Sequential([
    # Step 1: downsample from 50 → 24 steps
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2,
                        padding="valid", input_shape=[None, 1]),
    # Step 2: recurrent layers now see 24 steps, not 50
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
```

> ⚠️ **Warning:** When using `padding="valid"` + stride, the output length is shorter than the target. You must crop the targets to match:
> ```python
> Y_train_cropped = Y_train[:, 3:]  # adjust index to match output length
> ```

---

## 🔍 5. WaveNet and Dilated Convolutions {#wavenet}

> **TL;DR:** WaveNet stacks causal 1D convolutions with dilation rates that double at each layer (1, 2, 4, 8...). This gives the top layers a very large receptive field while keeping parameter count small.

![WaveNet Architecture](../Visuals/11_wavenet_architecture.png)
> 📊 **Graph 11:** Dilation rates expand the receptive field exponentially. Layer 1 sees 2 steps, layer 2 sees 4, layer 3 sees 8, layer 4 sees 16.

### Dilated Convolution Formula

A dilation rate $d$ spaces the kernel elements $d$ steps apart:

$$y_t = \sum_{\tau=0}^{K-1} w_\tau \cdot x_{t - d \cdot \tau}$$

At $d = 1$ this is a normal causal convolution. At $d = 4$, each kernel tap jumps 4 steps.

### Receptive Field

For $L$ layers with kernel size $K$ and dilation $d_l$:

$$R = 1 + (K-1) \sum_{l=1}^{L} d_l$$

**Worked example** — $K=2$, dilations $\{1, 2, 4, 8\}$ repeated twice:

$$R = 1 + 2 \times (1 + 2 + 4 + 8) = 1 + 30 = 31 \text{ steps}$$

With only 8 layers and kernel size 2, the model can see 31 steps back in history.

### Keras Implementation

```python
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

model_wavenet = keras.models.Sequential()
model_wavenet.add(keras.layers.InputLayer(input_shape=[None, 1]))

# dilations: 1, 2, 4, 8, 1, 2, 4, 8  (two full blocks)
for rate in (1, 2, 4, 8) * 2:
    model_wavenet.add(keras.layers.Conv1D(
        filters=20, kernel_size=2,
        padding="causal", activation="relu",
        dilation_rate=rate
    ))

# 1x1 Conv to project to output size
model_wavenet.add(keras.layers.Conv1D(filters=10, kernel_size=1))
model_wavenet.compile(loss="mse", optimizer="adam",
                      metrics=[last_time_step_mse])
```

---

## 🔍 6. WaveNet Gated Activation & Residual Blocks {#wavenet-details}

> **TL;DR:** The full WaveNet paper replaces ReLU with a gated activation and adds residual + skip connections for deeper, more stable networks.

### Gated Activation

$$z = \tanh(W_f * x) \otimes \sigma(W_g * x)$$

| Part | Role |
| :--- | :--- |
| $\tanh(W_f * x)$ | Filter — what information to pass through |
| $\sigma(W_g * x)$ | Gate — how much of that information to let through |
| $\otimes$ | Element-wise multiplication |

This is the same gating idea used in LSTM and GRU.

### Residual and Skip Connections

```
Input ──────────────────────────────┐
  │                                 │
  ▼                                 │ (residual add)
Gated Conv ──► (Skip to output) ◄──┘
  │
  ▼
Next Layer
```

- **Residual:** adds the input of the block directly to its output, allowing gradients to flow past any layer.
- **Skip:** every layer sends its output directly to the final sum before the output projection. This lets shallow features contribute directly to the prediction.

---

## 🔍 7. Bidirectional RNNs / BiLSTMs {#bilstm}

> **TL;DR:** A BiLSTM runs two LSTMs — one forward, one backward. The hidden states are concatenated, giving each output access to both past and future context.

![LSTM Cell](../Visuals/09_lstm_cell.png)
> 📊 **Diagram 09:** The LSTM cell that forms each directional pass in a BiLSTM.

$$h_t = [\vec{h}_t,\ \overleftarrow{h}_t]$$

**Use cases:**
- Named Entity Recognition (NER)
- Speech recognition
- Text classification

**Do NOT use for live forecasting.** The backward pass requires future values that do not exist yet.

```python
keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))
```

---

## 🔍 8. Stacked LSTMs {#stacked-lstm}

> **TL;DR:** Stack multiple LSTM layers so lower layers extract simple patterns and higher layers combine them into longer-range abstractions.

![Deep RNN Unrolled](../Visuals/07_deep_rnn_unrolled.png)
> 📊 **Diagram 07:** Multiple LSTM layers stacked vertically. Each layer's output sequence feeds into the next layer as its input.

**Architecture:**

```
Input
  ↓
LSTM Layer 1  (return_sequences=True)
  ↓
LSTM Layer 2  (return_sequences=True)
  ↓
LSTM Layer 3  (return_sequences=False or True depending on task)
  ↓
Dense Output
```

- Every layer except the last needs `return_sequences=True`.
- Lower layers → short patterns (individual beats in audio).
- Higher layers → long patterns (rhythmic phrases in audio).

---

## 🔍 9. Stateful LSTMs {#stateful-lstm}

> **TL;DR:** A stateful LSTM carries its hidden state from the end of one batch to the start of the next. This allows the model to remember information across batch boundaries.

**Default (stateless):** hidden state resets to zero at the start of every batch.

**Stateful:** final state of batch $n$ → initial state of batch $n+1$.

| Setting | When to use |
| :--- | :--- |
| Stateless (default) | Most tasks, randomly sampled batches |
| Stateful | Continuous streams: sensor feeds, audio streams, ECG |

**Rules for stateful LSTM:**
1. Set `stateful=True` in the layer.
2. Provide a fixed `batch_size` (cannot vary).
3. Set `shuffle=False` — batches must stay in order.
4. Manually call `model.reset_states()` at the end of each epoch.

```python
model = keras.models.Sequential([
    keras.layers.LSTM(32, stateful=True,
                      batch_input_shape=[batch_size, None, 1],
                      return_sequences=True),
    keras.layers.Dense(1)
])
```

---

## 🔍 10. Temporal Convolutional Networks (TCN) {#tcn}

> **TL;DR:** A TCN combines causal convolutions, dilated convolutions, and residual blocks into a single architecture. It is fully parallel and has stable gradients at any depth.

**TCN vs LSTM comparison:**

| Property | LSTM | TCN |
| :--- | :--- | :--- |
| Computation | Sequential | Parallel |
| Gradient stability | Can vanish/explode | Always stable |
| Long-range memory | Gating required | Dilation required |
| Training speed | Slow | Fast |
| Real-time streaming | Yes (stateful) | Yes (causal) |

**TCN block:**

```
Input
  ├──────────────────────────────┐
  │                              │ (residual)
  ▼                              │
Causal Dilated Conv (d=1)        │
  ↓                              │
Causal Dilated Conv (d=2)        │
  ↓                              │
1×1 Conv (match channels) ───────┘
  ↓
Output
```

TCNs often match or outperform LSTMs on time-series benchmarks.

---

## ❌ Common Mistakes {#mistakes}

**1. Using standard padding instead of causal in forecasting** ❌
> The kernel peeks at future steps. The model gets artificially low training loss but fails at inference time.
> Fix: always use `padding="causal"` for sequence forecasting.

**2. Forgetting to crop targets after Conv1D with stride > 1** ❌
> Conv1D output is shorter than the input. Comparing against the original-length targets causes a shape mismatch error.
> Fix: `Y_train_cropped = Y_train[:, 3:]` (adjust index to match output).

**3. Using BiLSTM for live / real-time forecasting** ❌
> The backward pass needs future values that do not exist in a streaming system.
> Fix: use unidirectional LSTM or causal Conv1D for real-time inference.

**4. Shuffling batches with stateful LSTM** ❌
> The hidden state carries information from batch $n$ to batch $n+1$. Shuffling breaks the ordering and corrupts the state.
> Fix: set `shuffle=False` during training.

---

## 🎤 Interview Q&A {#interview}

**Q1: What is causal padding and why is it required for sequence forecasting?**
> Causal padding adds zeros only on the left side. This ensures the output at step $t$ uses only inputs up to and including $t$. Without it, the kernel sees future inputs, causing data leakage — the model learns to "cheat" during training but fails in production.

**Q2: How does WaveNet expand its receptive field efficiently?**
> By doubling the dilation rate at each layer (1, 2, 4, 8 ...). For $L$ layers with kernel size 2, the receptive field is $R = 2^L$ steps. Parameters grow linearly with $L$, but the context window grows exponentially.

**Q3: When would you pick TCN over LSTM?**
> When training speed matters, when sequences are long, or when you need stable gradients. TCNs are fully parallel and do not suffer from vanishing gradients. LSTMs are preferred for online/streaming tasks where you need to update state one step at a time.

**Q4: What is the difference between residual and skip connections in WaveNet?**
> **Residual:** adds the input of a block to its output — helps gradient flow and keeps the model stable. **Skip:** sends each layer's output directly to the final output layer so every layer contributes to the prediction, not just the last one.

**Q5: Why use a stateful LSTM instead of a stateless one?**
> When the full sequence is too long to fit in one batch. Stateful LSTM carries the hidden state across ordered batches, so the model maintains memory of a continuous stream without truncation.

---

## ⚡ Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║           MODULE 5 — 1D CNNs & WAVENET QUICK REFERENCE          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1D CNN                                                          ║
║  - y_t = sum(w_k * x_{t-k})                                     ║
║  - Parallel computation — faster than RNN on GPU.               ║
║                                                                  ║
║  CAUSAL PADDING                                                  ║
║  - Zeros on LEFT only.  padding="causal" in Keras.              ║
║  - y(t) depends only on x(<= t).  No future leakage.            ║
║                                                                  ║
║  HYBRID CNN-RNN                                                  ║
║  - Conv1D(stride=2) cuts sequence length ~50%.                   ║
║  - Shorter sequence → fewer RNN steps → faster training.         ║
║  - Crop targets to match: Y[:, 3:]                               ║
║                                                                  ║
║  WAVENET DILATED CONV                                            ║
║  - y_t = sum(w_tau * x_{t - d*tau})                             ║
║  - Dilations 1,2,4,8 repeated → R = 1 + 2*(sum of dilations).  ║
║  - Gate: tanh(Wf*x) * sigmoid(Wg*x).                            ║
║                                                                  ║
║  TCN vs LSTM                                                     ║
║  - TCN: parallel, stable gradients, needs dilation.             ║
║  - LSTM: sequential, gating, good for streaming.                 ║
║                                                                  ║
║  STATEFUL LSTM RULES                                             ║
║  - shuffle=False, fixed batch_size, reset_states() each epoch.  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Previous Module →** [04_Long_Term_Dependency_Cells_LSTM_and_GRU.md](04_Long_Term_Dependency_Cells_LSTM_and_GRU.md)
**🔗 Chapter Complete! →** [Back to Chapter Index](../notes.md)
