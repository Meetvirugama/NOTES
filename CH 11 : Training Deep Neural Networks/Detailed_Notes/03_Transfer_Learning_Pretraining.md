# 📁 Module 3: Transfer Learning & Pretraining
> **Ch. 11 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [The Mechanics of Transfer Learning](#mechanics)
3. [Transfer Learning with Keras (Step-by-Step Code)](#keras-transfer)
4. [Unsupervised Pretraining](#unsupervised-pretraining)
5. [Pretraining on Auxiliary Tasks & Self-Supervised Learning](#auxiliary-tasks)
6. [Common Beginner Mistakes](#mistakes)
7. [Interview Q&A (Top 5)](#interview)
8. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** Transfer learning lets us reuse the lower hidden layers of an existing model trained on a similar source task, speeding up training and reducing the required volume of labeled training data. If no matching model exists, we can train an unsupervised model (like an autoencoder) on unlabeled data, or a model on an auxiliary task, and reuse those lower layers.

**The "Foreign Language Learning" Analogy 🗣️:**
If you already know Spanish fluently, and you want to learn Italian, you don't start from scratch by learning what nouns, verbs, and pronunciation are. You reuse your existing language "lower layers" (sentence structure, grammatical patterns, Latin vocabulary roots) and focus only on the Italian-specific differences (the "upper layers"). 

This is **Transfer Learning**: reusing generalized, low-level feature detectors (edges, curves, text patterns) instead of learning them from scratch for every new task.

---

## 🔍 1. The Mechanics of Transfer Learning {#mechanics}

When reusing a pretrained neural network, we must decide how many layers to keep, which to freeze, and which to train.

```
PRETRAINED BASE MODEL                        NEW TASK MODEL
┌──────────────────────┐                     ┌──────────────────────┐
│  Input Layer         │                     │  Input Layer         │
├──────────────────────┤                     ├──────────────────────┤
│  Layer 1 (Frozen)    │ ─── Reused ───────→ │  Layer 1 (Frozen)    │  (Detects generic edges)
├──────────────────────┤                     ├──────────────────────┤
│  Layer 2 (Frozen)    │ ─── Reused ───────→ │  Layer 2 (Frozen)    │  (Detects simple shapes)
├──────────────────────┤                     ├──────────────────────┤
│  Layer 3 (Active)    │ ─── Reused ───────→ │  Layer 3 (Trainable) │  (Adapts to task features)
├──────────────────────┤                     ├──────────────────────┤
│  Output Layer (Old)  │ (Discarded)         │  New Output (Active) │  (Task-specific predictions)
└──────────────────────┘                     └──────────────────────┘
```

### Key Strategies:
1.  **Input Resizing:** If your new images don't match the input size of the pretrained model, add a preprocessing step (e.g., resizing layers) to adapt the shape before sending them to the base model.
2.  **Output Replacement:** Discard the original output layer. It is task-specific (e.g., predicting 100 ImageNet classes) and will not match your new task's target categories.
3.  **Task Similarity Rules:**
    *   **High Similarity + Tiny Dataset:** Reuse all hidden layers, replace output, freeze all hidden layers.
    *   **High Similarity + Large Dataset:** Reuse all hidden layers, unfreeze top hidden layers, train with low learning rate.
    *   **Low Similarity + Tiny Dataset:** Drop upper hidden layers, reuse only lower hidden layers, freeze them, train new outputs.
    *   **Low Similarity + Large Dataset:** Reuse lower layers as initialization, unfreeze everything, train normally.

![Transfer Learning Stages](../Visuals/06_transfer_learning_stages.png)
> 📊 **Graph 06:** The two phases of transfer learning fine-tuning. Phase 1 warms up the output layer while keeping pretrained layers frozen. Phase 2 unfreezes top layers for joint training using a very low learning rate.

---

## 🔍 2. Transfer Learning with Keras (Step-by-Step Code) {#keras-transfer}

Consider transferring a model trained on 8 classes of Fashion MNIST (Model A) to a new model (Model B) that classifies Sandals vs. Shirts (2 classes) using only 200 labeled images.

### Step 1: Clone Model A to Avoid Side-Effects
When sharing layers, updating Model B will modify Model A's weights. To prevent this, clone Model A's architecture and copy its weights:
```python
import tensorflow as tf
from tensorflow import keras

model_A = keras.models.load_model("my_model_A.h5")
# Clone the structure (gets new random weights)
model_A_clone = keras.models.clone_model(model_A)
# Copy the weights over
model_A_clone.set_weights(model_A.get_weights())
```

### Step 2: Build Model B by Reusing Hidden Layers
```python
# Create a new sequential model excluding the last output layer of Model A
model_B_on_A = keras.models.Sequential(model_A_clone.layers[:-1])
# Add a new binary output layer
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
```

### Step 3: Freeze Reused Layers & Warm-Up Output Layer
*Must compile the model after freezing or unfreezing layers.*
```python
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

# Compile with standard learning rate
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Warm up for 4 epochs
history_warm = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))
# OUTPUT: Output layer trained to reasonable weights; base weights preserved.
```

### Step 4: Unfreeze and Fine-Tune with Low Learning Rate
```python
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

# Reduce learning rate (e.g., from default 1e-2 to 1e-4) to avoid wrecking weights
optimizer = keras.optimizers.SGD(lr=1e-4)
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Fine-tune model
history_fine = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))
# OUTPUT: Reused layers fine-tuned safely. Final accuracy: 99.25%.
```

---

## 🔍 3. Unsupervised Pretraining {#unsupervised-pretraining}

When labeled data is scarce but unlabeled data is plentiful, and no pretrained model is available, we use **Unsupervised Pretraining**.

![Unsupervised Pretraining Flow](../Visuals/07_unsupervised_pretraining.png)
> 📊 **Graph 07:** Workflow of Unsupervised Pretraining. The model learns general features on a large unlabeled dataset, then transfers the lower layers to be fine-tuned on the small labeled target dataset.

### Historical Context:
*   **Greedy Layer-Wise Pretraining (Hinton 2006):** Historically, training deep networks in one shot failed due to vanishing gradients. Researchers trained a single unsupervised layer (usually a Restricted Boltzmann Machine - RBM), froze it, stacked a new layer on top, trained only that layer, and repeated.
*   **Modern One-Shot Pretraining:** Today, we train the entire unsupervised model (e.g., a Stacked Autoencoder or GAN) in a single shot. We then copy the encoder or discriminator weights to our supervised network and fine-tune it on the labeled dataset.

---

## 🔍 4. Pretraining on Auxiliary Tasks & Self-Supervised Learning {#auxiliary-tasks}

If unlabeled data is not available, you can train a model on an **auxiliary task** for which you can easily generate or obtain labels, then transfer the lower layers.

### Examples:
1.  **Face Recognition:** If you only have 3 photos per employee (too few to train a classifier), gather photos of random people on the web and train a network to predict if two pictures feature the exact same person. This network learns generic face feature detectors (eyes, nose, geometry) that you can reuse to train your employee classifier.
2.  **Self-Supervised Learning (NLP):** Automatically generate labels from the data itself. For example, download a massive text database, randomly mask out words, and train a network to predict the missing words (e.g. "What [are] you doing?"). Reusing the lower layers of this model provides pre-existing grammatical and lexical knowledge for tasks like sentiment analysis.

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. Forgetting to compile the model after changing a layer's `trainable` attribute** ❌
> **Reality:** Keras compiles the execution graph. If you set `layer.trainable = False` or `True` but do not call `model.compile()`, the changes will not be applied, and the weights will not be updated or frozen as expected during the subsequent `fit()` call.

**2. Fine-tuning reused layers using a standard high learning rate** ❌
> **Reality:** The output layer begins with random weights, producing large errors initially. If you unfreeze the base layers and use a high learning rate, these large gradients will overwrite the pretrained weights, destroying the useful feature detectors. Always use a small learning rate (e.g., $10^{-4}$ or $10^{-5}$) for fine-tuning.

**3. Expecting transfer learning to work well with small dense networks** ⚠️
> **Reality:** Transfer learning works poorly with small dense (fully connected) networks. Dense layers learn very specific spatial coordinate combinations that rarely transfer to other tasks. Transfer learning is highly effective in Deep Convolutional Networks (CNNs), which learn generic, spatially invariant feature maps.

---

## 🎤 Interview Q&A (Top 5) {#interview}

**Q1: Why do we freeze pretrained layers before training the new output layer?**
> **A:** The newly added output layer has randomly initialized weights. During the first few epochs of training, it will make large prediction errors, generating massive gradient updates. If the pretrained layers are unfrozen, these large gradients will propagate through them and overwrite their fine-tuned weights, causing the model to forget its learned features. Freezing protects the base weights until the output layer is warmed up.

**Q2: What is the difference between a model clone and a model weight copy in Keras?**
> **A:** `keras.models.clone_model()` clones the architecture (layer types, shapes, activations) but initializes new, random weights. It does not copy the learned weights. To copy the weights, you must explicitly call `new_model.set_weights(old_model.get_weights())`.

**Q3: What is Self-Supervised Learning? Is it supervised or unsupervised?**
> **A:** Self-supervised learning is a technique where the training labels are automatically generated from the raw data itself (e.g., masking words in text, rotating images and predicting rotation angles). Because it requires no human labeling, it is classified as a form of unsupervised learning, though it uses supervised loss functions (like cross-entropy) during the pretraining phase.

**Q4: Under what dataset conditions should we use Unsupervised Pretraining?**
> **A:** We use unsupervised pretraining when: (1) we have a complex, deep target task, (2) we have a very small labeled dataset for that task, and (3) we have access to a large volume of unlabeled data representing a similar distribution.

**Q5: What is "Catastrophic Forgetting" in transfer learning?**
> **A:** It is the phenomenon where a neural network fine-tuned on a new task completely overwrites its previously learned general feature detectors (e.g., edge detectors), causing it to fail on the original task and perform suboptimally on the new task. It is mitigated by layer freezing, low fine-tuning learning rates, and early layers regularization.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║               MODULE 3 — TRANSFER & PRETRAINING                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TRANSFER LEARNING PIPELINE:                                     ║
║  1. Load Base Model:   base_model = load_model("base.h5")        ║
║  2. Splice Layers:     model = Sequential(base_model.layers[:-1])║
║  3. Add Output Layer:  model.add(Dense(classes, "softmax"))       ║
║  4. Freeze Base:       for l in model.layers[:-1]:               ║
║                            l.trainable = False                   ║
║  5. Compile & Warmup:  model.compile(lr=1e-2)                    ║
║                        model.fit(epochs=5)                       ║
║  6. Unfreeze Base:     for l in model.layers[:-1]:               ║
║                            l.trainable = True                    ║
║  7. Compile & Fine-tune:model.compile(lr=1e-4)                   ║
║                        model.fit(epochs=20)                      ║
║                                                                  ║
║  RULE OF THUMB FOR DECAYING LEARNING RATES:                      ║
║  - Output warmup:      Use standard learning rate (e.g., 1e-2).  ║
║  - Fine-tuning:        Use 10x to 100x lower rate (e.g., 1e-4).  ║
║                                                                  ║
║  UNSUPERVISED PRETRAINING:                                       ║
║  - Use Autoencoder/GAN to learn features on unlabeled data.      ║
║  - Transfer encoder/discriminator layers to target classifier.   ║
║                                                                  ║
║  AUXILIARY PRETRAINING:                                          ║
║  - Train on related task with easy labels (e.g., NLP word mask). ║
║                                                                  ║
║  CRITICAL REMINDER:                                              ║
║  - Always run model.compile() AFTER modifying trainable flags!   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Previous Module →** [02_Batch_Normalization_Clipping.md](02_Batch_Normalization_Clipping.md)  
**🔗 Next Module →** [04_Faster_Optimizers.md](04_Faster_Optimizers.md)
