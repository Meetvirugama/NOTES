# 📚 Chapter 10: Introduction to Artificial Neural Networks with Keras
### Complete Study Notes — Professor Level

> **All 26 pages analyzed. All concepts covered. Zero shortcuts.**

---

## 🖼️ Visual Gallery (Python-Generated Graphs)

> All visuals are in the [`Visuals/`](Visuals/) folder and are embedded in each module.
> Re-generate anytime: `python3 generate_visuals.py`

| # | Graph | Module | File |
|---|-------|--------|------|
| 01 | Activation Functions (Step, Sigmoid, Tanh, ReLU, Leaky ReLU, ELU) | 1 | [06_activation_functions.png](Visuals/06_activation_functions.png) |
| 02 | MLP Architecture (Input → Hidden → Output) | 1 | [04_mlp_architecture.png](Visuals/04_mlp_architecture.png) |
| 03 | Training Curves (Good vs Overfit vs Diverge) | 2 | [14_training_curves.png](Visuals/14_training_curves.png) |
| 04 | Gradient Descent (LR Effect on Loss Landscape) | 2 | [09_gradient_descent.png](Visuals/09_gradient_descent.png) |
| 05 | Backpropagation Flow (Forward + Backward pass) | 2 | [12_backpropagation.png](Visuals/12_backpropagation.png) |
| 06 | Learning Rate Finder (LR Range Test) | 7 | [10_lr_finder.png](Visuals/10_lr_finder.png) |
| 07 | Vanishing & Exploding Gradients (Sigmoid vs ReLU) | 2 | [13_gradient_flow.png](Visuals/13_gradient_flow.png) |
| 08 | Hyperparameter Effects (Batch Size + Neurons) | 7 | [28_hyperparameter_effects.png](Visuals/28_hyperparameter_effects.png) |
| 09 | ANN History Timeline (1943 → 2022) | 1 | [02_ann_timeline.png](Visuals/02_ann_timeline.png) |
| 10 | ⭐ Master Summary Dashboard (All concepts) | All | [29_summary_dashboard.png](Visuals/29_summary_dashboard.png) |
| 11 | Biological vs Artificial Neuron | 1 | [01_bio_vs_artificial.png](Visuals/01_bio_vs_artificial.png) |
| 12 | XOR Problem (AND/OR Separability) | 1 | [03_xor_problem.png](Visuals/03_xor_problem.png) |
| 13 | Loss Functions (MSE vs MAE vs Huber vs Cross-Entropy) | 2 | [08_loss_functions.png](Visuals/08_loss_functions.png) |
| 14 | Cross-Entropy Loss Penalty Intuition | 2 | [07_cross_entropy_intuition.png](Visuals/07_cross_entropy_intuition.png) |
| 15 | Fashion MNIST Dataset Class Sample Grid | 3 | [16_fashion_mnist_grid.png](Visuals/16_fashion_mnist_grid.png) |
| 16 | Confusion Matrix (Normalized & Raw counts) | 3 | [17_confusion_matrix.png](Visuals/17_confusion_matrix.png) |
| 17 | Binary Classification Decision Boundary | 3 | [15_binary_decision_boundary.png](Visuals/15_binary_decision_boundary.png) |
| 18 | Wide & Deep Network (Google's Architecture) | 4 | [21_wide_deep_architecture.png](Visuals/21_wide_deep_architecture.png) |
| 19 | Parameter Count Breakdown per Layer | 4 | [19_param_count.png](Visuals/19_param_count.png) |
| 20 | The Three Keras APIs Comparison | 4 | [18_three_apis_comparison.png](Visuals/18_three_apis_comparison.png) |
| 21 | Keras Callback Hook Execution Timeline | 5 | [23_callback_timeline.png](Visuals/23_callback_timeline.png) |
| 22 | Early Stopping and Patience Window | 5 | [24_early_stopping_annotated.png](Visuals/24_early_stopping_annotated.png) |
| 23 | Grid Search vs Random Search vs Bayesian Opt | 6 | [25_grid_vs_random_search.png](Visuals/25_grid_vs_random_search.png) |
| 24 | Transfer Learning (Frozen vs Trainable Layers) | 7 | [26_transfer_learning.png](Visuals/26_transfer_learning.png) |
| 25 | Neurons vs Layers Generalization | 7 | [27_neurons_vs_layers.png](Visuals/27_neurons_vs_layers.png) |
| 26 | Computational Gate Backpropagation Circuit (CS231n) | 2 | [11_backprop_node_circuit.png](Visuals/11_backprop_node_circuit.png) |
| 27 | Activation Functions & Gradient Saturation Zones | 1 | [05_activation_saturation_regions.png](Visuals/05_activation_saturation_regions.png) |
| 28 | Architectural Layout: Standard MLP vs. Wide & Deep | 4 | [20_wide_deep_vs_standard_mlp.png](Visuals/20_wide_deep_vs_standard_mlp.png) |
| 29 | Decision Matrix: Keras API Selection Flowchart | 4 | [22_keras_api_selection_flowchart.png](Visuals/22_keras_api_selection_flowchart.png) |

---

## 🗺️ Master Index

| Module | Topic | File | Pages Covered |
|--------|-------|------|---------------|
| 01 | Biological → Artificial Neurons, Perceptron, MLP, Activation Functions | [01_Biological_to_Artificial_Neurons.md](Detailed_Notes/01_Biological_to_Artificial_Neurons.md) | pp. 286–298 |
| 02 | Backpropagation, Loss Functions, Gradient Descent, Chain Rule | [02_MLPs_and_Backpropagation.md](Detailed_Notes/02_MLPs_and_Backpropagation.md) | pp. 294–302 |
| 03 | Regression vs Classification MLPs, Output Design, Fashion MNIST | [03_Regression_and_Classification_MLPs.md](Detailed_Notes/03_Regression_and_Classification_MLPs.md) | pp. 302–311 |
| 04 | Sequential, Functional, Subclassing APIs, Wide & Deep, Multi-I/O | [04_Implementing_MLPs_with_Keras.md](Detailed_Notes/04_Implementing_MLPs_with_Keras.md) | pp. 311–322 |
| 05 | Saving Models, Callbacks, EarlyStopping, TensorBoard | [05_Saving_Callbacks_TensorBoard.md](Detailed_Notes/05_Saving_Callbacks_TensorBoard.md) | pp. 322–325 |
| 06 | Scikit-Learn Integration, KerasRegressor, Randomized/Grid Search | [05_Saving_Callbacks_TensorBoard.md](Detailed_Notes/05_Saving_Callbacks_TensorBoard.md) | pp. 325–327 |
| 07 | Fine-Tuning HPs, Layer/Neuron/LR/Batch Size, Exercises, 20 IQs | [07_Fine_Tuning_Hyperparameters.md](Detailed_Notes/07_Fine_Tuning_Hyperparameters.md) | pp. 327–329 |

---

## ⚡ One-Page Chapter Summary

### The Journey: From Biology to Keras

```
1943: McCulloch–Pitts → Mathematical neuron model
1958: Rosenblatt → Perceptron (first trainable ANN)
1969: XOR problem → AI Winter (Minsky & Papert)
1986: Backpropagation → Renaissance (Rumelhart et al.)
2006: Deep Learning begins (Hinton)
2012: AlexNet → Deep Learning revolution
2024: GPT-4, Gemini, Stable Diffusion...
```

### Core Architecture

```
INPUT          HIDDEN LAYER(S)          OUTPUT
x₁ ──(w₁₁)──→ h₁ = f(Σwᵢxᵢ + b) ──→ ŷ = softmax or linear
x₂ ──(w₁₂)──→ h₂                       (task-dependent)
...            h₃
xₙ             ...
```

### Three Keras APIs

```python
# Sequential (simple stack):
model = keras.Sequential([Dense(300, "relu"), Dense(100, "relu"), Dense(10, "softmax")])

# Functional (any topology):
inp = Input(shape=[n])
h = Dense(30, "relu")(inp)
out = Dense(1)(h)
model = Model(inputs=inp, outputs=out)

# Subclassing (dynamic):
class MyModel(keras.Model):
    def call(self, inputs): ...
```

### The Training Loop

```python
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid),
          callbacks=[ModelCheckpoint("best.h5", save_best_only=True),
                     EarlyStopping(patience=10),
                     TensorBoard(run_logdir)])
model = keras.models.load_model("best.h5")  # load best!
```

### Output Layer Design (MEMORIZE THIS!)

| Task | Neurons | Activation | Loss |
|------|---------|------------|------|
| Regression | 1 | None | mse |
| Binary classification | 1 | Sigmoid | binary_crossentropy |
| Multi-class | n_classes | Softmax | sparse_categorical_crossentropy |
| Multi-label | n_labels | Sigmoid (each) | binary_crossentropy |

### Hyperparameter Quick Guide

| HP | Start Here | Notes |
|----|-----------|-------|
| # Hidden layers | 1–2 | Add more for complex tasks |
| # Neurons/layer | 50–200 (same all) | "Stretch pants" approach |
| Learning rate | 1e-3 | MOST IMPORTANT HP |
| Optimizer | SGD or Adam | Adam faster, SGD better generalization |
| Batch size | 32 | Larger = faster but less stable |
| Activation (hidden) | ReLU | Default |
| Epochs | 1000 + EarlyStopping | Don't tune manually |

---

## 🏆 Top 5 Things to Remember

1. **ReLU is the default activation** for hidden layers in modern networks
2. **Output layer design** depends entirely on task: regression→linear, binary→sigmoid, multiclass→softmax
3. **Always normalize your inputs** before training
4. **Use EarlyStopping + ModelCheckpoint** together in every training run
5. **Learning rate is the most important hyperparameter** — tune it first

---

## 🔗 Related Chapters

- **Chapter 11**: Training Deep Networks (dropout, batch norm, advanced optimizers)
- **Chapter 12**: TensorFlow lower-level API
- **Chapter 13**: Data loading and preprocessing
- **Chapter 14**: Convolutional Neural Networks (images)
- **Chapter 15**: Recurrent Neural Networks (sequences)

---

*Notes created from 26 textbook pages covering pp. 286–329 of Hands-On ML with Scikit-Learn, Keras, and TensorFlow (2nd edition) by Aurélien Géron.*
