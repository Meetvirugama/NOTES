# ⚡ Module 4: Autodiff and Custom Training Loops
> **Ch. 12 — Hands-On ML with Scikit-Learn, Keras & TensorFlow**
> **Rewritten: Plain English → Real Numbers → Code → Why It Matters**

---

## 📌 Table of Contents
1. [What is Autodiff? (Plain English)](#autodiff-plain)
2. [tf.GradientTape: How It Works](#gradient-tape)
3. [Gradients with Real Numbers (By Hand)](#gradients-numbers)
4. [Persistent and Nested Tapes](#persistent-nested)
5. [tape.watch() and tf.stop_gradient()](#watch-stop)
6. [Custom Training Loop from Scratch](#custom-loop)
7. [One Full Training Step, Step by Step](#one-step)
8. [Common Mistakes (Wrong vs. Right)](#mistakes)
9. [How It All Connects](#connects)
10. [Flash Card](#flashcard)

---

## 🌍 1. What is Autodiff? (Plain English) {#autodiff-plain}

**The problem:** To train a neural network, you need to know "how much did each weight contribute to the error?" This is the gradient.

Computing gradients by hand for a model with millions of parameters is impossible. That's what **automatic differentiation (Autodiff)** solves — TensorFlow computes all gradients automatically for you.

### 🎯 The GPS Hiker Analogy

Imagine you're hiking in mountains, trying to find the lowest valley (minimum loss):
- `model.fit()` = booking a helicopter. Fast, gets you there, but you can't control the path.
- Custom training loop with `tf.GradientTape` = hiking with a GPS tracker.

The GPS (GradientTape) records **every step you take** (every operation in the forward pass). When you want to go downhill, you look at your GPS log to calculate the slope at your current position (the gradient). Then you take one step down (apply the gradient to update weights). Repeat.

### How Autodiff Works (Conceptually)

```
Forward pass:  input → [operations] → output/loss
               Tape records every operation

Backward pass: Tape "rewinds" through operations
               Applies chain rule automatically
               Computes: d(loss)/d(each_weight)
```

The chain rule in math:
```
If z = f(g(x)), then dz/dx = dz/dg × dg/dx

TF applies this automatically at every layer, every weight.
```

---

## 🔍 2. tf.GradientTape: How It Works {#gradient-tape}

![Autodiff Gradient Tape](../Visuals/07_autodiff_gradient_tape.png)

```python
import tensorflow as tf

w = tf.Variable(3.0)   # a weight (tf.Variable — must be Variable, not constant)

with tf.GradientTape() as tape:
    # Everything inside this block is RECORDED
    y = w * w     # y = w²

# After the block, compute the gradient:
grad = tape.gradient(y, w)   # dy/dw = 2w = 2×3 = 6.0
print(grad.numpy())           # 6.0
```

**What the tape does:**
1. Inside the `with` block: records every operation involving `tf.Variable`s
2. When you call `tape.gradient(target, source)`: computes d(target)/d(source)
3. After `.gradient()` is called: the tape is **deleted automatically** (single use)

**What the tape records (internally):**
```
Operation log:
  [1] y = w × w    (power rule: dy/dw = 2w)

When you call tape.gradient(y, w):
  → applies chain rule through recorded operations
  → result: 2 × 3.0 = 6.0
```

---

## 🔢 3. Gradients with Real Numbers (By Hand) {#gradients-numbers}

Let's compute gradients manually and then verify with TF.

### Example 1: Simple Quadratic

```
Function: z = w1² + w2³
w1 = 5.0, w2 = 3.0

Partial derivatives (by hand):
  dz/dw1 = 2 × w1 = 2 × 5.0 = 10.0
  dz/dw2 = 3 × w2² = 3 × 3.0² = 3 × 9 = 27.0
```

```python
w1 = tf.Variable(5.0)
w2 = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = w1**2 + w2**3

grads = tape.gradient(z, [w1, w2])
print(grads[0].numpy())    # 10.0  ✅ matches hand calculation
print(grads[1].numpy())    # 27.0  ✅ matches hand calculation
```

### Example 2: A Linear Model (What Actually Happens in Training)

```
Model: y_pred = w × x + b
Data:  x = 2.0,  y_true = 7.0
Init:  w = 1.0,  b = 0.0

Forward pass:
  y_pred = 1.0 × 2.0 + 0.0 = 2.0

Loss (MSE for 1 sample):
  loss = (y_true - y_pred)² = (7.0 - 2.0)² = 25.0

Gradients (by hand using chain rule):
  d(loss)/d(y_pred) = -2 × (y_true - y_pred) = -2 × 5 = -10
  d(y_pred)/d(w) = x = 2.0
  d(y_pred)/d(b) = 1.0

  d(loss)/d(w) = d(loss)/d(y_pred) × d(y_pred)/d(w) = -10 × 2.0 = -20.0
  d(loss)/d(b) = d(loss)/d(y_pred) × d(y_pred)/d(b) = -10 × 1.0 = -10.0
```

```python
x = tf.constant(2.0)
y_true = tf.constant(7.0)
w = tf.Variable(1.0)
b = tf.Variable(0.0)

with tf.GradientTape() as tape:
    y_pred = w * x + b
    loss = (y_true - y_pred) ** 2

grads = tape.gradient(loss, [w, b])
print(grads[0].numpy())    # -20.0  ✅ matches hand calculation
print(grads[1].numpy())    # -10.0  ✅ matches hand calculation
```

### One Weight Update (With These Gradients)

```
learning_rate = 0.01

w_new = w - lr × grad_w = 1.0 - 0.01 × (-20.0) = 1.0 + 0.2 = 1.2
b_new = b - lr × grad_b = 0.0 - 0.01 × (-10.0) = 0.0 + 0.1 = 0.1

New prediction: y_pred = 1.2 × 2.0 + 0.1 = 2.5
New loss: (7.0 - 2.5)² = 20.25    ← smaller than 25.0!
```

The gradient tells us: "increase w and b" (negative gradient = increase). And indeed, after the update, we predict 2.5 instead of 2.0 — closer to 7.0.

---

## 🔁 4. Persistent and Nested Tapes {#persistent-nested}

### Default Tape: Single Use

```python
w = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = w**2

grad1 = tape.gradient(z, w)   # works: 6.0
grad2 = tape.gradient(z, w)   # RuntimeError! Tape is already gone after first call
```

**Why?** After `.gradient()` is called, TF deletes the recorded operation log to free memory. It assumes you only need gradients once per forward pass.

### Persistent Tape: Multiple Uses

```python
w1 = tf.Variable(5.0)
w2 = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    z = w1**2 + w2**3

grad_w1 = tape.gradient(z, w1)   # 10.0 ✅
grad_w2 = tape.gradient(z, w2)   # 27.0 ✅  (can call again on persistent tape)

del tape   # ← MANDATORY! Persistent tapes don't auto-release memory!
```

> ⚠️ **Always call `del tape`** after a persistent tape. If you forget, the memory of all recorded operations stays alive in GPU RAM, causing OOM (out-of-memory) errors during long training loops.

### Nested Tapes: Second-Order Gradients (Advanced)

```
Use case: computing the gradient of the gradient
          (e.g., for second-order optimization methods)

Function: y = w²
First derivative:  dy/dw  = 2w
Second derivative: d²y/dw² = 2  (constant — slope of slope)
```

```python
w = tf.Variable(3.0)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = w**2
    dy_dw = inner_tape.gradient(y, w)    # 2×3 = 6.0 (first derivative)

d2y_dw2 = outer_tape.gradient(dy_dw, w)  # 2.0 (second derivative)
print(d2y_dw2.numpy())    # 2.0 ✅
```

---

## 🎛️ 5. tape.watch() and tf.stop_gradient() {#watch-stop}

### tape.watch(): Track Constants Too

The tape automatically tracks `tf.Variable` objects. It does NOT track `tf.constant` by default.

```python
# tf.constant — NOT tracked by default
c = tf.constant(3.0)

with tf.GradientTape() as tape:
    z = c**2

grad = tape.gradient(z, c)
print(grad)    # None  ← c was not tracked!

# FIX: manually watch the constant
with tf.GradientTape() as tape:
    tape.watch(c)    # ← tell tape to watch this constant
    z = c**2

grad = tape.gradient(z, c)
print(grad.numpy())    # 6.0 ✅  (dc²/dc = 2c = 2×3 = 6)
```

**When do you need this?** When computing input gradients (e.g., for saliency maps in interpretability, or adversarial examples).

### tf.stop_gradient(): Block Gradients

![Stop Gradient Adversarial](../Visuals/12_stop_gradient_adversarial.png)

Sometimes you want a gradient to NOT flow through part of the graph. Example: freezing the encoder in a GAN while training the discriminator.

```python
w1 = tf.Variable(5.0)
w2 = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = w1**2             # y depends on w1
    y_frozen = tf.stop_gradient(y)   # cut the gradient here
    z = y_frozen + w2**2  # z depends on frozen y AND w2

grads = tape.gradient(z, [w1, w2])
print(grads[0])            # None  ← w1's gradient path was cut!
print(grads[1].numpy())    # 6.0   ← w2 still gets its gradient (2×3=6)
```

---

## 🔄 6. Custom Training Loop from Scratch {#custom-loop}

![Custom Training Loop Flow](../Visuals/08_custom_training_loop_flow.png)

The `model.fit()` method is actually a for-loop that calls a training step function on each batch. You can write this yourself.

**Why write your own?**
- GAN training (two optimizers alternating)
- Multi-task learning with separate loss weights per task
- Curriculum learning (change data difficulty over time)
- Logging custom metrics at non-standard points

### Full Custom Training Loop Template

```python
import tensorflow as tf
from tensorflow import keras

# Setup
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(1)
])
optimizer = keras.optimizers.SGD(learning_rate=0.01)
loss_fn = keras.losses.mean_squared_error

# Toy dataset
X = tf.random.normal([200, 8])
y = tf.random.normal([200, 1])
dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(200).batch(32)

# Training loop
n_epochs = 3
for epoch in range(n_epochs):
    print(f"\nEpoch {epoch+1}/{n_epochs}")
    
    for step, (X_batch, y_batch) in enumerate(dataset):
        
        # ─── Step 1: Forward pass inside tape ────────────────
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            total_loss = tf.add_n([main_loss] + model.losses)  # include regularization

        # ─── Step 2: Compute gradients ────────────────────────
        gradients = tape.gradient(total_loss, model.trainable_variables)

        # ─── Step 3: Clip gradients (optional, prevents explosion)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        # ─── Step 4: Apply gradients ──────────────────────────
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if step % 2 == 0:
            print(f"  Step {step}: loss = {total_loss.numpy():.4f}")
```

---

## 🔢 7. One Full Training Step, Step by Step {#one-step}

![Custom Training Loop Backpropagation](../Visuals/14_custom_training_loop_backpropagation.png)

Let's trace one mini-batch through the custom loop with actual numbers.

```
Setup:
  model: 2 inputs → Dense(1) → 1 output  (just w and b for simplicity)
  Initial: w = [0.5, -0.3], b = 0.0
  learning_rate = 0.1

Mini-batch (4 samples):
  X_batch = [[1.0, 2.0],
             [0.5, 1.0],
             [2.0, 0.5],
             [1.5, 1.5]]

  y_batch = [4.0, 2.0, 3.5, 5.0]

─── STEP 1: Forward Pass ───────────────────────────────────────────

y_pred = X_batch @ w + b

Sample 1: 1.0×0.5 + 2.0×(-0.3) + 0 = 0.5 - 0.6 = -0.1
Sample 2: 0.5×0.5 + 1.0×(-0.3) + 0 = 0.25 - 0.3 = -0.05
Sample 3: 2.0×0.5 + 0.5×(-0.3) + 0 = 1.0 - 0.15 = 0.85
Sample 4: 1.5×0.5 + 1.5×(-0.3) + 0 = 0.75 - 0.45 = 0.30

y_pred = [-0.10, -0.05, 0.85, 0.30]

─── STEP 2: Compute Loss ───────────────────────────────────────────

MSE errors = (y_batch - y_pred)²
  Sample 1: (4.0 - (-0.10))² = 4.10² = 16.81
  Sample 2: (2.0 - (-0.05))² = 2.05² =  4.20
  Sample 3: (3.5 - 0.85)²    = 2.65² =  7.02
  Sample 4: (5.0 - 0.30)²    = 4.70² = 22.09

mean_loss = (16.81 + 4.20 + 7.02 + 22.09) / 4 = 50.12 / 4 = 12.53

─── STEP 3: Gradients (computed by tape) ───────────────────────────

Tape computes: d(loss)/d(w) and d(loss)/d(b)

─── STEP 4: Update Weights ─────────────────────────────────────────

w_new = w - lr × gradient_w
b_new = b - lr × gradient_b

After update, loss on this batch becomes smaller.

─── KEY INSIGHT ────────────────────────────────────────────────────

After 100 such steps:
  Initial loss: ~12.53
  After 100 steps with lr=0.01: loss might drop to ~0.5

The pattern is: predict → measure error → compute gradient → update → repeat
```

---

## ❌ 8. Common Mistakes (Wrong vs. Right) {#mistakes}

### Mistake 1: Not passing training=True

```python
# ❌ WRONG — Dropout neurons NOT dropped during training
with tf.GradientTape() as tape:
    y_pred = model(X_batch)     # defaults to training=False!
    # Dropout is disabled → model "cheats" during training

# ✅ RIGHT
with tf.GradientTape() as tape:
    y_pred = model(X_batch, training=True)    # Dropout active ✅
```

**Why it matters:** Layers like `Dropout` and `BatchNormalization` behave differently during training vs. inference. Without `training=True`, Dropout doesn't drop neurons, so the model trains without regularization — overfitting is likely.

### Mistake 2: Forgetting to delete a persistent tape

```python
# ❌ WRONG — memory leak!
for epoch in range(100):
    with tf.GradientTape(persistent=True) as tape:
        loss = model(X)
    grad1 = tape.gradient(loss, model.layers[0].trainable_variables)
    grad2 = tape.gradient(loss, model.layers[1].trainable_variables)
    # tape never deleted! GPU RAM fills up after a few epochs.

# ✅ RIGHT
    grad1 = tape.gradient(loss, model.layers[0].trainable_variables)
    grad2 = tape.gradient(loss, model.layers[1].trainable_variables)
    del tape    # ← free memory immediately
```

### Mistake 3: Calling tape.gradient() after the with block closes normally

```python
# ❌ WRONG — tape is gone, gradient returns None
y = model(X)    # outside the tape!
with tf.GradientTape() as tape:
    loss = loss_fn(y, y_true)   # tape only recorded loss computation, not model!

grad = tape.gradient(loss, model.trainable_variables)   # None or zeros!

# ✅ RIGHT — the entire forward pass must be inside the tape
with tf.GradientTape() as tape:
    y = model(X)             # model forward pass inside tape ✅
    loss = loss_fn(y, y_true)  # loss computation inside tape ✅

grad = tape.gradient(loss, model.trainable_variables)   # correct gradients ✅
```

### Mistake 4: Using tf.constant instead of tf.Variable for weights

```python
# ❌ WRONG — constants are not tracked by tape!
w = tf.constant(1.0)
with tf.GradientTape() as tape:
    loss = (w * 2.0 - 5.0)**2

grad = tape.gradient(loss, w)   # None! w is a constant.

# ✅ RIGHT
w = tf.Variable(1.0)   # Variable is tracked automatically
with tf.GradientTape() as tape:
    loss = (w * 2.0 - 5.0)**2

grad = tape.gradient(loss, w)   # 2.0 (correct gradient) ✅
```

---

## 🔗 9. How It All Connects {#connects}

```
THE COMPLETE TRAINING LOOP PICTURE

   Data (X_batch, y_batch)
         │
         ▼
   ┌─────────────────────────────────────────────────────────┐
   │  with tf.GradientTape() as tape:                        │
   │                                                         │
   │    y_pred = model(X_batch, training=True)               │
   │    ← tape records ALL operations involving Variables    │
   │                                                         │
   │    main_loss = loss_fn(y_batch, y_pred)                 │
   │    total_loss = main_loss + sum(model.losses)           │
   │    ← model.losses = any self.add_loss() calls           │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   gradients = tape.gradient(total_loss, model.trainable_variables)
   ← chain rule applied backward through recorded operations

                              │
                              ▼
   optimizer.apply_gradients(zip(gradients, model.trainable_variables))
   ← each weight updated: w -= lr × gradient

                              │
                              ▼
   metric.update_state(y_batch, y_pred)
   ← stateful metric accumulates across batches

   [End of epoch: metric.result() displayed, reset_state() called]
```

---

## ⚡ 10. Flash Card {#flashcard}

```
╔══════════════════════════════════════════════════════════════╗
║      MODULE 4 — AUTODIFF & CUSTOM TRAINING FLASH CARD        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  GRADIENTTAPE:                                               ║
║    Watches: tf.Variable automatically                        ║
║    Ignores: tf.constant (use tape.watch(c) to track)         ║
║    Single use: tape deleted after .gradient()                ║
║    Persistent: GradientTape(persistent=True)                 ║
║      → MUST call del tape after done!                        ║
║                                                              ║
║  GRADIENT COMPUTATION:                                       ║
║    grads = tape.gradient(loss, [w, b])                       ║
║    grads[i] = d(loss)/d(variable_i)                         ║
║    If gradient = None: check if Variable, check if in tape   ║
║                                                              ║
║  CUSTOM LOOP (3 steps every batch):                          ║
║    1. with GradientTape:                                     ║
║         y_pred = model(X, training=True)                     ║
║         loss = loss_fn(y_true, y_pred)                       ║
║    2. grads = tape.gradient(loss, model.trainable_variables)  ║
║    3. optimizer.apply_gradients(zip(grads, variables))        ║
║                                                              ║
║  STOP GRADIENT:                                              ║
║    tf.stop_gradient(tensor) → gradient = None past this point║
║    Use: freeze layers in GANs, multi-task learning           ║
║                                                              ║
║  WHY training=True MATTERS:                                  ║
║    Dropout: drops neurons only when training=True            ║
║    BatchNorm: updates running stats only when training=True  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

**🔗 Previous Module →** [03_Custom_Layers_and_Models.md](03_Custom_Layers_and_Models.md)
**🔗 Next Module →** [05_TensorFlow_Functions_and_Graphs.md](05_TensorFlow_Functions_and_Graphs.md)
