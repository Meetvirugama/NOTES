# 🧠 Module 1: Vanishing & Exploding Gradients
> **Ch. 11 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [The Math of Gradient Instability](#gradient-instability)
3. [Glorot, He, and LeCun Weight Initializations](#weight-init)
4. [Nonsaturating Activation Functions (ReLU & Variants)](#activations)
5. [Self-Normalizing Networks & SELU](#self-normalization)
6. [Common Beginner Mistakes](#mistakes)
7. [Interview Q&A (Top 5)](#interview)
8. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** When training deep networks, gradients can shrink (vanish) or grow (explode) as they flow backward. Vanishing gradients freeze lower layers, preventing learning, while exploding gradients make weights wildly fluctuate (common in RNNs). We fix this using mathematically-scaled weight initialization and nonsaturating activation functions.

**The "Microphone Amplifier Chain" Analogy 🎤:**
Imagine a chain of 10 sound amplifiers. If you set the first volume knob slightly too low (e.g., $0.1\times$), and the next ones do the same, your voice at the end will be completely silent (vanishing signal). If you set them slightly too high (e.g., $2.0\times$), the speaker will screech and blow out (exploding signal). 

For the signal to flow perfectly from start to finish, each amplifier must output the voice at the exact same amplitude it came in (variance of output = variance of input). Glorot and He weight initializations do exactly this: they set the "volume knobs" (initial weight variances) based on the number of inputs and outputs of each layer to ensure stable gradient flow.

---

## 🔍 1. The Math of Gradient Instability {#gradient-instability}

### Mathematical Intuition

During backpropagation, we compute the gradient of the loss with respect to the weights of early layers using the **Chain Rule**. In a deep network, this requires multiplying the weights and derivatives of all subsequent layers together:

$$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \frac{\partial L}{\partial \hat{\mathbf{y}}} \cdot \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{h}^{(k)}} \cdots \frac{\partial \mathbf{h}^{(2)}}{\partial \mathbf{h}^{(1)}} \cdot \frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{W}^{(1)}}$$

If we use the logistic sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$, its derivative is:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

The maximum value of $\sigma'(z)$ is **$0.25$** (at $z = 0$). 

If weights are initialized using a standard normal distribution ($\sigma = 1$), they are likely to be greater than $1$. As inputs pass through layers, their variance increases, driving $|z|$ to be very large. When $|z|$ is large, $\sigma'(z) \approx 0$. 

Multiply many terms that are at most $0.25$ by large weights, and the gradient decays exponentially:

$$\text{Gradient} \propto (0.25)^L \to 0 \quad \text{as } L \text{ increases}$$

This causes early layers to receive virtually zero updates, leaving them stuck with random initial weights.

![Vanishing Gradients in Deep Sigmoid Net](../Visuals/17_vanishing_gradients_sigmoid.png)
> 📊 **Graph 17:** Gradient norm decay (vanishing gradients) in a 5-layer sigmoid network. Note how the gradient norm decays exponentially for earlier layers (Layer 1) compared to later layers (Layer 5).

![Initialization Variance](../Visuals/01_weight_initialization_variance.png)
> 📊 **Graph 01:** Simulation of layer output variance propagation. A standard normal initialization leads to variance explosion/decay, whereas Glorot and He initializations keep layer output variance completely stable.

---

## 🔍 2. Glorot, He, and LeCun Weight Initializations {#weight-init}

To keep the signal flowing stably in both directions (forward pass and backward pass), we require:
1. The variance of the outputs of each layer to equal the variance of its inputs.
2. The variance of the gradients to remain equal before and after passing through a layer in the reverse direction.

While mathematically impossible to satisfy both simultaneously unless a layer has the same number of inputs ($fan_{in}$) and outputs ($fan_{out}$), we use the following highly successful compromises:

### The Mathematical Rules

| Initialization Style | Optimized Activation | Variance $\sigma^2$ (Normal Distribution) | Uniform Boundary $r$ ($[-r, +r]$) |
| :--- | :--- | :--- | :--- |
| **Glorot (Xavier)** | Tanh, Sigmoid, Softmax, None | $\sigma^2 = \frac{1}{fan_{avg}}$ | $r = \sqrt{\frac{3}{fan_{avg}}}$ |
| **He (Kaiming)** | ReLU and its variants | $\sigma^2 = \frac{2}{fan_{in}}$ | $r = \sqrt{\frac{6}{fan_{in}}}$ |
| **LeCun** | SELU | $\sigma^2 = \frac{1}{fan_{in}}$ | $r = \sqrt{\frac{3}{fan_{in}}}$ |

*Where:*
$$fan_{avg} = \frac{fan_{in} + fan_{out}}{2}$$

```python
import tensorflow as tf
from tensorflow import keras

# 1. Customizing a layer to use He Normal initialization (for ReLU)
he_layer = keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal")

# 2. Customizing a layer to use LeCun Normal initialization (for SELU)
lecun_layer = keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal")

# 3. Custom Variance Scaling using fan_avg with Uniform distribution
custom_init = keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform")
custom_layer = keras.layers.Dense(100, activation="relu", kernel_initializer=custom_init)
# OUTPUT: Dense layers initialized with custom variance scaling rules.
```

---

## 🔍 3. Nonsaturating Activation Functions (ReLU & Variants) {#activations}

Historically, the logistic sigmoid was preferred because it mimics biological neuron activation. However, it saturates at $0$ and $1$ for large absolute inputs, causing vanishing gradients. Modern deep networks use **nonsaturating activation functions** that maintain high gradients for large inputs.

![Activation Functions](../Visuals/02_activation_functions_comparison.png)
> 📊 **Graph 02:** Nonsaturating activation functions (ReLU, LeakyReLU, ELU, SELU) plotted alongside their gradients. Notice how their gradients do not collapse to zero in the positive input domain.

### 1. ReLU (Rectified Linear Unit)
$$f(z) = \max(0, z)$$
*   **Pros:** Computes blazingly fast; gradient is exactly $1$ for $z > 0$ (no vanishing gradient).
*   **Cons:** **"Dying ReLU"** problem. If a neuron's weights are adjusted such that it outputs $0$ for all instances in the training set, its gradient is permanently $0$. It will never learn again.

### 2. Leaky ReLU
$$f(z) = \max(\alpha z, z)$$
*   **Pros:** Fixes dying ReLU. The parameter $\alpha$ (typically $0.01$ or $0.2$) represents the negative slope, ensuring a small gradient ($0.01$ or $0.2$) is always backpropagated.
*   **Cons:** Adds another hyperparameter $\alpha$ to tune.

### 3. ELU (Exponential Linear Unit)
$$f(z) = \begin{cases} \alpha(e^z - 1) & \text{if } z < 0 \\ z & \text{if } z \geq 0 \end{cases}$$
*   **Pros:** Outputs have an average closer to $0$ (speeds up training). Smooth derivative everywhere (including $z=0$), reducing gradient oscillation.
*   **Cons:** Slower to compute due to the exponential function $e^z$.

![Dying ReLU vs LeakyReLU](../Visuals/03_dying_relu.png)
> 📊 **Graph 03:** Comparison of output distributions. Under ReLU, negative inputs get permanently clipped to exactly zero (causing dead units), whereas LeakyReLU keeps active, non-zero values flowing.

---

## 🔍 4. Self-Normalizing Networks & SELU {#self-normalization}

Introduced by Klambauer et al. in 2017, the **SELU (Scaled Exponential Linear Unit)** activation function is a scaled version of ELU:

$$f(z) = \lambda \begin{cases} \alpha(e^z - 1) & \text{if } z < 0 \\ z & \text{if } z \geq 0 \end{cases}$$

Where constants are fixed to:
*   $\lambda \approx 1.0507$
*   $\alpha \approx 1.6733$

### The Self-Normalization Phenomenon
If you build a neural network consisting **exclusively** of a stack of dense layers, and if all hidden layers use the **SELU** activation function, the network will **self-normalize**: the output of each layer will preserve a mean of $0$ and a standard deviation of $1$ during training. This mathematically solves the vanishing and exploding gradients problems!

### Strict Conditions for SELU Self-Normalization:
1.  **Input Standardization:** Input features must be scaled to mean $0$ and standard deviation $1$.
2.  **LeCun Normal Initialization:** Every hidden layer's weights must be initialized using `lecun_normal`.
3.  **Sequential Architecture:** No skip connections, residual branches, or recurrent links. All layers must be dense.

```python
# Self-Normalizing Keras Stack
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
# OUTPUT: Sequential model built for self-normalization.
```

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. Using the default weight initializer for ReLU layers** ❌
> **Reality:** Keras defaults to Glorot initialization. If you use ReLU layers, you should explicitly set `kernel_initializer="he_normal"` or `"he_uniform"`. Otherwise, training convergence will be slower.

**2. Experiencing dying neurons with standard ReLU** ❌
> **Reality:** If a large fraction (e.g., >20%) of your network's neurons are outputting constant zero (dead), your learning rate is likely too high, pushing weights into permanent negative sums. Lower the learning rate or replace `ReLU` with `LeakyReLU(alpha=0.2)` or `ELU`.

**3. Applying Batch Normalization alongside SELU** ❌
> **Reality:** SELU self-normalizes natively. Adding Batch Normalization layers breaks the mathematical assumptions of SELU and degrades performance. Use one or the other.

---

## 🎤 Interview Q&A (Top 5) {#interview}

**Q1: Why does standard normal weight initialization fail in deep feedforward networks?**
> **A:** Standard normal initialization has a variance of $1.0$. Because a layer's output variance is proportional to $fan_{in} \cdot \sigma^2$, the variance of inputs propagates exponentially through the layers, causing inputs to saturate sigmoid/tanh activations. Saturation shrinks derivatives to near-zero ($\sigma'(z) \approx 0$), making gradients vanish during backpropagation.

**Q2: What is the "symmetry problem" in weight initialization and how do we break it?**
> **A:** If all weights are initialized to the same value (e.g., $0$), all neurons in a hidden layer will compute the exact same output during the forward pass and receive the exact same gradient during backpropagation. They will perform identical weight updates, acting as a single neuron. We break this symmetry by initializing weights with **random values**.

**Q3: How does Leaky ReLU solve the "Dying ReLU" problem?**
> **A:** A standard ReLU outputs exactly $0$ for any negative input, and its derivative is also $0$. If a neuron's weights update such that it always receives negative inputs, its weights will never update again. Leaky ReLU introduces a small slope $\alpha$ (e.g., $0.01$) for $z < 0$. Since the derivative is $\alpha \neq 0$, it ensures that even "dead" neurons receive a small gradient update and can recover.

**Q4: Compare ELU and Leaky ReLU. What is the main trade-off?**
> **A:** ELU is smooth everywhere (including $z=0$), which reduces gradient bouncing and speeds up gradient descent convergence. It also takes negative values, pulling the mean activation closer to zero. However, ELU requires computing an exponential function ($e^z$), which makes it slower during inference than Leaky ReLU.

**Q5: What are the requirements for a SELU network to guarantee self-normalization?**
> **A:** First, the input features must be standardized (mean $0$, standard deviation $1$). Second, weights must be initialized using LeCun normal initialization. Third, the architecture must be strictly sequential (only dense layers, no skip connections or recurrent layers).

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║               MODULE 1 — WEIGHTS & ACTIVATIONS                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WEIGHT INITIALIZATION RULE OF THUMB:                            ║
║  - ReLU & Variants  → He Init     (kernel_initializer="he_normal")║
║  - SELU             → LeCun Init  (kernel_initializer="lecun_normal")
║  - Sigmoid/Tanh     → Glorot Init (kernel_initializer="glorot_...")║
║                                                                  ║
║  MATH DERIVATIONS:                                               ║
║  - Glorot Normal: Variance σ² = 1 / fan_avg                      ║
║  - He Normal:     Variance σ² = 2 / fan_in                       ║
║  - LeCun Normal:  Variance σ² = 1 / fan_in                       ║
║                                                                  ║
║  NONSATURATING ACTIVATIONS:                                      ║
║  - ReLU: max(0, z)               | Fast, but dying ReLU risk.    ║
║  - Leaky ReLU: max(αz, z)        | α=0.01 or 0.2. No dead ReLUs.  ║
║  - ELU: z if z>=0 else α(e^z - 1)| Smooth everywhere, slower exp.║
║  - SELU: λ * ELU(z)              | Guarantees self-normalization. ║
║                                                                  ║
║  SELU CHECKLIST:                                                 ║
║  - Input scaled to N(0,1)                                        ║
║  - lecun_normal initialization                                   ║
║  - Dense layers only (Sequential)                                ║
║  - No Batch Norm or Dropout (use Alpha Dropout instead)          ║
║                                                                  ║
║  COMMON PITFALLS:                                                ║
║  - Forgetting use_bias=False when preceding Batch Normalization  ║
║  - Initializing weights to 0 (causes symmetry lock)              ║
║  - Large learning rate with ReLU (causes dying ReLUs)            ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Next Module →** [02_Batch_Normalization_Clipping.md](02_Batch_Normalization_Clipping.md)
