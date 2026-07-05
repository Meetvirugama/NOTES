# 🔧 Module 2: Custom Loss Functions, Metrics, Layers & Components
> **Ch. 12 — Hands-On ML with Scikit-Learn, Keras & TensorFlow**
> **Rewritten: Plain English → Real Numbers → Code → Why It Matters**

---

## 📌 Table of Contents
1. [Why "Custom"? The Big Picture](#big-picture)
2. [Custom Loss Functions](#custom-loss)
3. [Saving Hyperparameters with get_config()](#saving)
4. [Custom Activations, Initializers, Regularizers, Constraints](#custom-activations)
5. [Custom Metrics: Stateless vs. Stateful](#custom-metrics)
6. [Common Mistakes (Wrong vs. Right)](#mistakes)
7. [How It All Connects](#connects)
8. [Flash Card](#flashcard)

---

## 🌍 1. Why "Custom"? The Big Picture {#big-picture}

Keras gives you ready-made losses (`MSE`, `CrossEntropy`), metrics (`Accuracy`), and layers (`Dense`, `Conv2D`).

**But what if you need something Keras doesn't have?**

| You want... | Built-in? | Solution |
|-------------|-----------|---------|
| Huber loss (robust to outliers) | ✅ Actually yes | But custom teaches the pattern |
| Focal loss (for imbalanced data) | ❌ No | Custom loss function |
| mAP (mean average precision) | ❌ No | Custom stateful metric |
| Attention layer (custom formula) | ❌ No | Custom layer |
| GAN training loop | ❌ No | Custom model + training loop |

**The universal Keras customization pattern:**
```python
class MyThing(keras.SomeBaseClass):
    def __init__(self, hyperparams, **kwargs):
        super().__init__(**kwargs)
        self.hyperparams = hyperparams

    def call(self, inputs):         # the actual computation
        return output

    def get_config(self):           # REQUIRED for saving!
        return {**super().get_config(), "hyperparams": self.hyperparams}
```

Once you understand this pattern, you can build anything.

---

## 📉 2. Custom Loss Functions {#custom-loss}

### What is the Huber Loss?

**The problem with MSE:** Squaring errors makes big errors enormous. One bad data point (outlier) can dominate the entire loss.

```
Data: 5 house prices (in $100k):  [2, 3, 4, 5, 100]
                                                 ↑
                                           Outlier!

Predictions:                        [2.1, 3.1, 4.1, 5.1, 5.1]

Errors:                             [-0.1, -0.1, -0.1, -0.1, 94.9]

MSE per sample:                     [0.01, 0.01, 0.01, 0.01, 9006.0]
                                                               ↑
                                           This ONE outlier causes 99.9% of the loss!
```

**The Huber Loss solution:** Use squared error for small errors (gentle), but switch to linear error for large errors (less punishing):

```
                    Error²/2        if |error| ≤ threshold
Huber loss =  {
                    threshold×|error| - threshold²/2    if |error| > threshold
```

### 🔢 Step-by-Step Huber Loss Calculation (threshold = 1.0)

```
Example: 3 predictions vs. true values

      y_true   y_pred   error    |error|  small?   Huber Loss
Row 1:   3.0    3.2     -0.2     0.2      YES (≤1)  0.2²/2 = 0.02
Row 2:   5.0    4.5      0.5     0.5      YES (≤1)  0.5²/2 = 0.125
Row 3:   2.0    5.0     -3.0     3.0      NO  (>1)  1×3 - 1²/2 = 2.5

Total loss = mean = (0.02 + 0.125 + 2.5) / 3 = 0.882
```

Compare to MSE:
```
MSE = (0.2² + 0.5² + 3.0²) / 3 = (0.04 + 0.25 + 9.0) / 3 = 3.097
```

MSE is 3.5x larger — dominated by the outlier. Huber is more balanced.

![Huber Loss with Numbers](../Visuals/03_custom_loss_huber.png)

### Approach 1: Simple Function

```python
import tensorflow as tf
from tensorflow import keras

def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small = tf.abs(error) < 1.0            # threshold = 1.0
    squared_loss = tf.square(error) / 2       # for small errors
    linear_loss  = tf.abs(error) - 0.5        # for large errors: 1*|e| - 1²/2
    return tf.where(is_small, squared_loss, linear_loss)

model.compile(loss=huber_fn, optimizer="nadam")
```

**Problem:** You can't change the threshold. And if you save the model, `huber_fn` must be re-registered when loading.

### Approach 2: Closure with Configurable Threshold

```python
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small = tf.abs(error) < threshold
        squared = tf.square(error) / 2
        linear  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small, squared, linear)
    return huber_fn

model.compile(loss=create_huber(threshold=2.0), optimizer="nadam")
```

**Problem:** When you save and reload the model, the `threshold=2.0` is LOST. You must specify it again.

```python
# Loading: must manually specify threshold again
model = keras.models.load_model("model.h5",
                                 custom_objects={"huber_fn": create_huber(2.0)})
```

### Approach 3: Class — Best Practice (Threshold is SAVED)

```python
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()          # includes 'name', 'reduction'
        return {**base_config, "threshold": self.threshold}

# Usage:
model.compile(loss=HuberLoss(threshold=2.0), optimizer="nadam")
model.save("model.h5")

# Load — threshold=2.0 IS preserved inside the file!
model = keras.models.load_model("model.h5",
                                 custom_objects={"HuberLoss": HuberLoss})
```

### 🔢 What get_config() Actually Does

```python
loss = HuberLoss(threshold=2.0)
print(loss.get_config())
# {'name': 'huber_loss', 'reduction': 'sum_over_batch_size', 'threshold': 2.0}
#  ↑ from super()                                              ↑ your addition

# When you save the model, Keras stores this dictionary as JSON
# When you load, Keras calls:  HuberLoss(**config_dict)
# Which is the same as:        HuberLoss(name='huber_loss', threshold=2.0)
```

---

## 💾 3. Saving Hyperparameters with get_config() {#saving}

**The rule:** Any custom class that you want to save with a model MUST implement `get_config()`.

**Why?** When you call `model.save()`, Keras needs to record "what components are in this model and how were they configured". It does this by calling `get_config()` on each component and saving the result as JSON.

```
model.save("my_model.h5")
    └── calls HuberLoss.get_config()
        └── returns {"name": "huber_loss", "threshold": 2.0}
            └── stored in the HDF5 file as JSON

keras.models.load_model("my_model.h5", custom_objects={"HuberLoss": HuberLoss})
    └── reads JSON: {"threshold": 2.0}
        └── calls HuberLoss(threshold=2.0)  ← reconstructs with same settings
```

**Template — always use this pattern:**
```python
def get_config(self):
    base_config = super().get_config()      # always start with parent's config
    return {**base_config,                  # unpack parent's dict
            "param1": self.param1,          # add your own parameters
            "param2": self.param2}
```

---

## ⚡ 4. Custom Activations, Initializers, Regularizers, Constraints {#custom-activations}

These are simpler than losses. They follow the same pattern but are used in different places.

### Custom Activation Function

**What is an activation function?** After each layer computes `output = input @ weights + bias`, the activation function transforms the result to introduce non-linearity.

**Example: Softplus** — a smooth version of ReLU

```
Regular ReLU:  f(x) = max(0, x)       ← sharp corner at 0
Softplus:      f(x) = log(1 + e^x)    ← smooth version

At x = -3: ReLU = 0,    Softplus = log(1 + e^-3) ≈ log(1.05) ≈ 0.048
At x =  0: ReLU = 0,    Softplus = log(1 + e^0)  = log(2)    ≈ 0.693
At x =  3: ReLU = 3,    Softplus = log(1 + e^3)  ≈ log(21)   ≈ 3.048
```

```python
def my_softplus(z):
    return tf.math.log(1.0 + tf.exp(z))

layer = keras.layers.Dense(30, activation=my_softplus)
```

### Custom Glorot Initializer

**What is initialization?** The starting values of weights before training. Bad initialization can cause training to fail (vanishing/exploding gradients).

**Glorot normal:** Set initial weights based on the layer size.
```
stddev = sqrt(2 / (fan_in + fan_out))

For a layer: input_size=4, output_size=2
stddev = sqrt(2 / (4+2)) = sqrt(0.333) ≈ 0.577
Weights drawn from: Normal(mean=0, stddev=0.577)
```

```python
def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)
```

### Custom L1 Regularizer

**What is regularization?** Adding a penalty to the loss when weights become too large. This prevents overfitting.

**L1 regularization:**
```
Main loss (e.g., 0.5) + regularization penalty

Weights of one layer: [0.8, -0.3, 1.2, -0.1]
L1 penalty = factor × sum(|weights|)
           = 0.01 × (0.8 + 0.3 + 1.2 + 0.1)
           = 0.01 × 2.4
           = 0.024

Total loss = 0.5 + 0.024 = 0.524
```

```python
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor=0.01):
        self.factor = factor

    def __call__(self, weights):             # Note: __call__, NOT call!
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}
```

### Custom Weight Constraint

**What is a constraint?** After each gradient update, forcibly clip weights to satisfy some condition.

```python
def my_positive_weights(weights):
    """Force all weights to be >= 0."""
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
    # Negative weights → 0.0,  Non-negative weights → unchanged
```

### Using All Four Together

```python
layer = keras.layers.Dense(
    30,
    activation=my_softplus,
    kernel_initializer=my_glorot_initializer,
    kernel_regularizer=MyL1Regularizer(factor=0.01),
    kernel_constraint=my_positive_weights
)
```

**What happens during training (in order):**
```
1. Layer created  →  weights set by my_glorot_initializer
2. Forward pass   →  output = my_softplus(input @ W + b)
3. Loss computed  →  total_loss = main_loss + MyL1Regularizer(W)
4. Gradients      →  d(total_loss)/d(W) computed
5. Weights update →  W = W - lr × gradient
6. Constraint     →  W = my_positive_weights(W)   (clip negatives)
7. Next batch     →  repeat from step 2
```

---

## 📊 5. Custom Metrics: Stateless vs. Stateful {#custom-metrics}

### The Problem with Simple Functions (Stateless Metrics)

Suppose you're evaluating a model on 100 batches of 32 samples each.

**Simple approach (WRONG for unequal batch sizes):**
```
Batch 1 (32 samples):  loss = 0.40
Batch 2 (32 samples):  loss = 0.30
...
Batch 99 (32 samples): loss = 0.20
Batch 100 (8 samples): loss = 0.10   ← only 8 samples, not 32!

Wrong: mean of batch means = (0.40 + 0.30 + ... + 0.10) / 100
       The last batch gets equal weight as others, but has fewer samples!
```

**Correct approach (stateful, accumulate):**
```
Running total:  sum of (loss × sample_count) across all batches
Running count:  total number of samples

Correct metric = running_total / running_count
              = sum(batch_loss_i × n_samples_i) / sum(n_samples_i)
```

![Stateful vs Stateless Metric](../Visuals/04_stateful_vs_stateless_metric.png)

### Stateless Metric (Function) — Simple but Limited

```python
def huber_metric(y_true, y_pred):
    error = y_true - y_pred
    is_small = tf.abs(error) < 1.0
    squared = tf.square(error) / 2
    linear  = tf.abs(error) - 0.5
    per_instance = tf.where(is_small, squared, linear)
    return tf.reduce_mean(per_instance)   # just mean of this batch

model.compile(..., metrics=[huber_metric])
```

This works when all batches are the same size. For variable sizes, use stateful.

### Stateful Metric (Class) — Correct for All Cases

```python
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        # These two variables accumulate across batches:
        self.total = self.add_weight("total", initializer="zeros")   # sum of losses
        self.count = self.add_weight("count", initializer="zeros")   # sample count

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Called after EACH batch — accumulate the running totals."""
        error = y_true - y_pred
        is_small = tf.abs(error) < self.threshold
        squared = tf.square(error) / 2
        linear  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        per_sample_loss = tf.where(is_small, squared, linear)

        self.total.assign_add(tf.reduce_sum(per_sample_loss))   # add batch total
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))  # add count

    def result(self):
        """Called at end of epoch — return the accumulated mean."""
        return self.total / self.count

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}

    # reset_state() is inherited — it resets total and count to 0 each epoch

model.compile(..., metrics=[HuberMetric(threshold=1.5)])
```

### 🔢 Stateful Metric: Tracing One Epoch

```
Epoch 1, 3 batches:

Batch 1 (4 samples):  per_sample_losses = [0.02, 0.125, 0.02, 2.5]
  total.assign_add(0.02 + 0.125 + 0.02 + 2.5)  = 2.665
  count.assign_add(4)                             = 4

Batch 2 (4 samples):  per_sample_losses = [0.5, 0.3, 0.18, 0.08]
  total.assign_add(0.5 + 0.3 + 0.18 + 0.08)    = 2.665 + 1.06 = 3.725
  count.assign_add(4)                             = 8

Batch 3 (2 samples):  per_sample_losses = [0.72, 0.45]
  total.assign_add(0.72 + 0.45)                  = 3.725 + 1.17 = 4.895
  count.assign_add(2)                             = 10

result() = total / count = 4.895 / 10 = 0.4895   ← correct epoch mean!

[Next epoch begins: reset_state() is called → total=0, count=0]
```

---

## ❌ 6. Common Mistakes (Wrong vs. Right) {#mistakes}

### Mistake 1: Not implementing get_config()

```python
# ❌ WRONG
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold   # no get_config() !

    def call(self, y_true, y_pred):
        ...

model.compile(loss=HuberLoss(threshold=2.0))
model.save("model.h5")
model2 = keras.models.load_model("model.h5", custom_objects={"HuberLoss": HuberLoss})
# model2 uses threshold=1.0 (default) — the 2.0 was silently lost!

# ✅ RIGHT: add get_config()
def get_config(self):
    return {**super().get_config(), "threshold": self.threshold}
# Now threshold=2.0 is saved and restored correctly ✅
```

### Mistake 2: Using Python `if` inside @tf.function with tensor conditions

```python
# ❌ WRONG — Python if checks VALUE during tracing (only once!)
@tf.function
def my_loss(y_true, y_pred, use_huber):
    if use_huber:           # this is only evaluated ONCE at trace time!
        return huber(y_true, y_pred)
    return mse(y_true, y_pred)

# ✅ RIGHT — use tf.cond for tensor conditionals
@tf.function
def my_loss(y_true, y_pred, use_huber):
    return tf.cond(use_huber,
                   lambda: huber(y_true, y_pred),
                   lambda: mse(y_true, y_pred))
```

### Mistake 3: Stateful metric as a function

```python
# ❌ WRONG — function only sees one batch at a time
def streaming_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(y_true == y_pred, tf.float32))
# This gives mean for each batch, then those means are averaged together
# Wrong if batches have different sizes!

# ✅ RIGHT — use keras.metrics.Accuracy() which is already stateful
model.compile(metrics=["accuracy"])
# Or build a custom stateful metric class as shown above
```

### Mistake 4: Not calling super().__init__(**kwargs)

```python
# ❌ WRONG
class MyLayer(keras.layers.Layer):
    def __init__(self, units):
        self.units = units     # super().__init__() not called!
        # Missing: name, dtype, trainable, and other Layer attributes!

# ✅ RIGHT
class MyLayer(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)   # sets name, dtype, etc.
        self.units = units
```

---

## 🔗 7. How It All Connects {#connects}

```
MODEL TRAINING PIPELINE

Input data (X_batch, y_batch)
        │
        ▼ forward pass
  predictions = model(X_batch)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
  MAIN LOSS                                     REGULARIZATION
  (Custom or built-in)                          (added automatically from layer.losses)
  e.g., HuberLoss(threshold=2.0)                e.g., MyL1Regularizer on weights
        │                                              │
        └──────────────────────────┬───────────────────┘
                                   ▼
                         total_loss = main_loss + reg_losses
                                   │
                                   ▼
                    gradients = tape.gradient(total_loss, trainable_vars)
                                   │
                                   ▼
                    optimizer.apply_gradients(zip(grads, vars))
                                   │
                                   ▼
                    CONSTRAINT applied to weights (clip, normalize)
                                   │
                                   ▼
                    METRIC updated: update_state(y_batch, predictions)
                                   │
                    [After full epoch: metric.result() → displayed]
                    [metric.reset_state() → ready for next epoch]
```

---

## ⚡ 8. Flash Card {#flashcard}

```
╔══════════════════════════════════════════════════════════════╗
║       MODULE 2 — CUSTOM KERAS COMPONENTS FLASH CARD          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  HUBER LOSS:                                                 ║
║    Small error (|e| ≤ threshold): loss = e²/2               ║
║    Large error (|e| > threshold): loss = t×|e| - t²/2       ║
║    Effect: like MSE for small errors, MAE for big ones       ║
║    Benefit: less affected by outliers than pure MSE          ║
║                                                              ║
║  CUSTOM LOSS (function → class):                             ║
║    Function: simple, but threshold lost on model save        ║
║    Class: implement call() + get_config() → threshold SAVED  ║
║                                                              ║
║  get_config() pattern:                                       ║
║    def get_config(self):                                     ║
║        return {**super().get_config(), "param": self.param}  ║
║                                                              ║
║  CUSTOM METRIC:                                              ║
║    Stateless (function): correct only for equal batch sizes  ║
║    Stateful (class): add_weight() to accumulate across batches║
║      update_state() → called per batch (accumulate)         ║
║      result()       → called per epoch (total / count)      ║
║      reset_state()  → called between epochs (reset to 0)    ║
║                                                              ║
║  REGULARIZER: uses __call__(), not call()                    ║
║  CONSTRAINT:  applied AFTER each weight update               ║
║  INITIALIZER: sets starting weight VALUES                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

**🔗 Previous Module →** [01_TensorFlow_Quick_Tour_and_NumPy_Basics.md](01_TensorFlow_Quick_Tour_and_NumPy_Basics.md)
**🔗 Next Module →** [03_Custom_Layers_and_Models.md](03_Custom_Layers_and_Models.md)
