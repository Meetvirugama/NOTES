# 📚 Chapter 15: Processing Sequences Using RNNs and CNNs
### Complete Study Notes — Professor Level

> **All 26 pages analyzed. All concepts covered. 11 diagrams mapped. Zero shortcuts.**

---

## 🖼️ Visual Gallery (Python-Generated Graphs)

> All visuals are in the [`Visuals/`](Visuals/) folder and are embedded inside their respective modules.
> Re-generate anytime: `python3 generate_visuals.py`

| # | Diagram / Graph Title | Module | File |
|---|---|---|---|
| 01 | Folded vs. Unrolled Recurrent Neurons | 1 | [01_recurrent_neuron_unrolled.png](Visuals/01_recurrent_neuron_unrolled.png) |
| 02 | Unrolled Recurrent Layer Matrix Shapes | 1 | [02_recurrent_layer_unrolled.png](Visuals/02_recurrent_layer_unrolled.png) |
| 03 | Hidden State $h_{(t)}$ vs. Output Prediction $y_{(t)}$ | 1 | [03_hidden_state_vs_output.png](Visuals/03_hidden_state_vs_output.png) |
| 04 | Taxonomy of Sequence Mapping Architectures | 1 | [04_rnn_seq_types.png](Visuals/04_rnn_seq_types.png) |
| 05 | Backpropagation Through Time Gradient Flow | 3 | [05_bptt.png](Visuals/05_bptt.png) |
| 06 | Synthetic Univariate Time Series Examples | 2 | [06_time_series_example.png](Visuals/06_time_series_example.png) |
| 07 | Stacked Deep RNN Layers Unrolled | 2 | [07_deep_rnn_unrolled.png](Visuals/07_deep_rnn_unrolled.png) |
| 08 | Multi-Step Forecasting (Recursive vs. Seq-to-Seq) | 2 | [08_forecasting_ahead.png](Visuals/08_forecasting_ahead.png) |
| 09 | LSTM Cell Internal Gating Mechanics | 4 | [09_lstm_cell.png](Visuals/09_lstm_cell.png) |
| 10 | GRU Cell Simplified Internal Gating | 4 | [10_gru_cell.png](Visuals/10_gru_cell.png) |
| 11 | WaveNet Stacked Dilated 1D Convolutions | 5 | [11_wavenet_architecture.png](Visuals/11_wavenet_architecture.png) |

---

## 🗺️ Master Index

| Module | Topic | File | Pages Covered |
|--------|-------|------|---------------|
| 01 | **Recurrent Foundations**: Folded vs. unrolled neurons, temporal weight sharing, recurrent layer math (Equations 15-1 & 15-2), memory cells, and taxonomy of sequence mapping (seq-to-seq, seq-to-vec, vec-to-seq, delayed seq-to-seq). | [01_Recurrent_Neurons_and_Input_Output_Sequences.md](Detailed_Notes/01_Recurrent_Neurons_and_Input_Output_Sequences.md) | pp. 498–502 |
| 02 | **Time Series Forecasting & Deep RNNs**: Synthetic sequence generation, evaluation baselines (naive, linear), Simple RNN, Deep RNN unrolling guidelines, return_sequences constraint, and multi-step forecasting styles (recursive, sequence-to-vector, sequence-to-sequence via `TimeDistributed`). | [02_Forecasting_Time_Series_and_Deep_RNNs.md](Detailed_Notes/02_Forecasting_Time_Series_and_Deep_RNNs.md) | pp. 503–511 |
| 03 | **Fighting Unstable Gradients**: Exploding/vanishing gradients in BPTT, tanh vs. ReLU bounds, gradient clipping (`clipnorm` vs. `clipvalue`), Layer Normalization in custom cells (`LNSimpleRNNCell`), and stateful vs. stateless RNN state reset rules. | [03_Fighting_Unstable_Gradients_in_RNNs.md](Detailed_Notes/03_Fighting_Unstable_Gradients_in_RNNs.md) | pp. 511–514 |
| 04 | **Long-Term Memory (LSTM & GRU)**: Short-term memory limits, LSTM gate operations (forget, input, output, candidate), mathematical equations (Equation 15-3), peephole connections, GRU structural simplifications, and update/reset gates (Equation 15-4). | [04_Long_Term_Dependency_Cells_LSTM_and_GRU.md](Detailed_Notes/04_Long_Term_Dependency_Cells_LSTM_and_GRU.md) | pp. 514–519 |
| 05 | **1D CNNs & WaveNet**: Parallel sequence processing, causal padding rules to prevent leakage, hybrid downsampling CNN-RNN models, dilated convolutions, and simplified WaveNet architecture implementation. | [05_Processing_Sequences_Using_1D_CNNs_and_WaveNet.md](Detailed_Notes/05_Processing_Sequences_Using_1D_CNNs_and_WaveNet.md) | pp. 519–523 |

---

## ⚡ One-Page Chapter Summary

### The History of Sequential Modeling
```
1997: Hochreiter & Schmidhuber propose LSTM ─────→ Solves vanishing gradient via constant error carousel.
2000: Gers & Schmidhuber introduce Peepholes ───→ Allows gates to peek at current cell states.
2014: Kyunghyun Cho et al. propose GRU ──────────→ Simplifies LSTM by merging states and gating.
2016: DeepMind introduces WaveNet ───────────────→ Dilated causal 1D CNNs replace RNN steps for raw audio.
```

### Core Architecture & Gating Math
```
   [ LSTM CONSTANT ERROR CAROUSEL (C_t) ]
   C(t-1) ─────────── (x Forget Gate f_t) ───────────── (+) ───────────────→ C(t)
                                                         │
                                               (x Input * Candidate)
                                                         │
   h(t-1) ───[ Gate Controllers: f_t, i_t, o_t, g_t ]────┘                  │
   x(t)   ───[   Inputs and previous hidden state   ]───(x Output o_t)──→ [tanh(C_t)] ──→ h(t)
```

### Core Code Snippet (LSTM + WaveNet Architectures)
```python
import tensorflow as tf
from tensorflow import keras

# 1. Custom Cell: Layer Normalization in Recurrent Loop
class LNSimpleRNNCell(keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units, activation=None)
        self.layer_norm = keras.layers.LayerNormalization()
        self.activation = keras.activations.get(activation)

    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]

# 2. Simplified WaveNet Model (Exponentially growing causal receptive field)
def build_wavenet(input_dim=1, output_dim=10):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[None, input_dim]))
    for rate in (1, 2, 4, 8) * 2:
        model.add(keras.layers.Conv1D(
            filters=20, kernel_size=2, padding="causal",
            activation="relu", dilation_rate=rate
        ))
    model.add(keras.layers.Conv1D(filters=output_dim, kernel_size=1))
    return model
```

---

## 🏆 Top 5 Things to Remember

1. **Input Shape Requisites**: Recurrent layers in Keras expect a 3D input tensor of shape `[batch_size, time_steps, features]`. Feeding a flat 2D array of shape `[batch_size, features]` will cause shape errors.
2. **The return_sequences Parameter**: Stacking recurrent layers requires setting `return_sequences=True` on all intermediate layers so they output a 3D sequence tensor rather than a 2D final vector.
3. **Layer Normalization vs. Batch Normalization**: LN normalizes across the features of each instance at each step, making it sequence-length and batch-size invariant. BN is highly unstable in RNNs because sequential stats vary too much over time.
4. **LSTM Constant Error Carousel**: LSTM solves vanishing gradients because its cell state update $\mathbf{C}_{(t)}$ is linear (additive): $\mathbf{f}_{(t)} \otimes \mathbf{C}_{(t-1)} + \mathbf{i}_{(t)} \otimes \mathbf{g}_{(t)}$. The gradient flows back unimpeded when the forget gate is open ($f_{(t)} \approx 1.0$).
5. **Causal Padding Mandatory**: In 1D convolutions for sequence forecasting, you must use `padding="causal"` (shifts padding to the left). Standard padding allows the kernel to peek into the future, causing **data leakage**.

---

## 🔗 Related Chapters
* **Chapter 14**: Convolutions, kernels, and valid/same padding principles apply to 1D sequence convolutions.
* **Chapter 16**: Natural Language Processing utilizes the LSTM, GRU, and attention mechanisms studied here for machine translation and text generation.

---

*Notes created from 26 textbook pages covering pp. 497–523 of Hands-On ML with Scikit-Learn, Keras, and TensorFlow (2nd edition) by Aurélien Géron.*
