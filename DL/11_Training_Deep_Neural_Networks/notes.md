# 📚 Chapter 11: Training Deep Neural Networks
### Complete Study Notes — Professor Level

> **All 44 pages analyzed. All concepts covered. Zero shortcuts.**

---

## 🖼️ Visual Gallery (Python-Generated Graphs)

> All visuals are in the [`Visuals/`](Visuals/) folder and are embedded in each module.
> Re-generate anytime: `python3 generate_visuals.py`

| # | Graph | Module | File |
|---|-------|--------|------|
| 01 | Layer Output Variance Propagation (Standard vs. Glorot vs. He) | 1 | [01_weight_initialization_variance.png](Visuals/01_weight_initialization_variance.png) |
| 02 | Nonsaturating Activation Functions (ReLU, LeakyReLU, ELU, SELU & Derivatives) | 1 | [02_activation_functions_comparison.png](Visuals/02_activation_functions_comparison.png) |
| 03 | Dying ReLU Output Spike vs. LeakyReLU Continuous Activations | 1 | [03_dying_relu.png](Visuals/03_dying_relu.png) |
| 04 | Batch Normalization Execution Flow (Mean, Var, Normalization, Scale/Shift) | 2 | [04_batch_normalization_flow.png](Visuals/04_batch_normalization_flow.png) |
| 05 | Gradient Clipping: Value Clipping (Rotating) vs. Norm Clipping (Direction Preserved) | 2 | [05_gradient_clipping.png](Visuals/05_gradient_clipping.png) |
| 06 | Transfer Learning Fine-Tuning Stages (Frozen Base Layer vs. Unfrozen Joint Training) | 3 | [06_transfer_learning_stages.png](Visuals/06_transfer_learning_stages.png) |
| 07 | Unsupervised Pretraining & Supervised Fine-Tuning Stages Pipeline | 3 | [07_unsupervised_pretraining.png](Visuals/07_unsupervised_pretraining.png) |
| 08 | Optimizer Trajectories in Narrow Valleys (SGD vs. Momentum vs. NAG Look-Ahead) | 4 | [08_momentum_vs_sgd.png](Visuals/08_momentum_vs_sgd.png) |
| 09 | Adaptive Learning Rates Path Trajectories (AdaGrad Stalling vs. RMSProp vs. Adam) | 4 | [09_adaptive_optimizers.png](Visuals/09_adaptive_optimizers.png) |
| 10 | Loss Curves under Different Learning Rates η (Diverging, Slow, Optimal, Decaying) | 5 | [10_learning_rate_effects.png](Visuals/10_learning_rate_effects.png) |
| 11 | Learning Rate Schedules (Power, Exponential, Piecewise, 1cycle Curves) | 5 | [11_lr_schedules.png](Visuals/11_lr_schedules.png) |
| 12 | Dropout Regularization (Training Drop Node State vs. Testing Scaled Connections) | 6 | [12_dropout_mechanism.png](Visuals/12_dropout_mechanism.png) |
| 13 | MC Dropout Probability Distribution Comparison & Epistemic Uncertainty Mapping | 6 | [13_mc_dropout_uncertainty.png](Visuals/13_mc_dropout_uncertainty.png) |
| 14 | Max-Norm Boundary Hyper-Sphere Projection Rescaling | 6 | [14_max_norm_constraint.png](Visuals/14_max_norm_constraint.png) |
| 15 | ⭐ Master Summary Dashboard (Activation, Schedules, Speed Rank, Best Practice Table) | All | [15_summary_dashboard.png](Visuals/15_summary_dashboard.png) |
| 16 | Normalization Methods Comparison (BN vs. LN vs. IN vs. GN) | 2 | [16_normalization_comparison.png](Visuals/16_normalization_comparison.png) |
| 17 | Gradient Decay (Vanishing Gradients) in Deep Sigmoid Net | 1 | [17_vanishing_gradients_sigmoid.png](Visuals/17_vanishing_gradients_sigmoid.png) |
| 18 | Cosine Learning Rate Schedule with Linear Warm-Up | 5 | [18_learning_rate_warmup.png](Visuals/18_learning_rate_warmup.png) |
| 19 | Optimizer Paths Escaping a Saddle Point Contour | 4 | [19_optimizer_landscape_saddle.png](Visuals/19_optimizer_landscape_saddle.png) |
| 20 | Vanishing Gradient Intuition | 1 | [20_vanishing_gradient_intuition.jpg](Visuals/20_vanishing_gradient_intuition.jpg) |
| 21 | Batch Normalization Concept | 2 | [21_batch_normalization_concept.jpg](Visuals/21_batch_normalization_concept.jpg) |
| 22 | Transfer Learning Using MobileNet | 3 | [22_transfer_learning_mobilenet.jpg](Visuals/22_transfer_learning_mobilenet.jpg) |
| 23 | Evolution of Gradient-Based Optimizers | 4 | [23_optimizer_evolution.jpg](Visuals/23_optimizer_evolution.jpg) |
| 24 | Dropout Concept | 6 | [24_dropout_concept.jpg](Visuals/24_dropout_concept.jpg) |

---

## 🗺️ Master Index

| Module | Topic | File | Pages Covered |
|--------|-------|------|---------------|
| 01 | Vanishing/Exploding Gradients, Glorot/He/LeCun Initialization, ReLU variants, SELU | [01_Vanishing_Exploding_Gradients.md](Detailed_Notes/01_Vanishing_Exploding_Gradients.md) | pp. 362–368 |
| 02 | Batch Normalization (train vs test), Gradient Clipping (value vs norm) | [02_Batch_Normalization_Clipping.md](Detailed_Notes/02_Batch_Normalization_Clipping.md) | pp. 368–375 |
| 03 | Transfer Learning with Keras, Model Cloning, Unsupervised Pretraining, Auxiliary Tasks | [03_Transfer_Learning_Pretraining.md](Detailed_Notes/03_Transfer_Learning_Pretraining.md) | pp. 375–381 |
| 04 | Momentum, NAG, AdaGrad, RMSProp, Adam, AdaMax, Nadam, Jacobian vs Hessian Math | [04_Faster_Optimizers.md](Detailed_Notes/04_Faster_Optimizers.md) | pp. 381–389 |
| 05 | LR finder range test, Power, Exponential, Piecewise, Performance, 1cycle scheduling | [05_Learning_Rate_Scheduling.md](Detailed_Notes/05_Learning_Rate_Scheduling.md) | pp. 389–394 |
| 06 | L1/L2 regularization, Dropout, MC Dropout, Max-Norm constraint, Default DNN Recipes | [06_Regularization_Guidelines.md](Detailed_Notes/06_Regularization_Guidelines.md) | pp. 394–404 |

---

## ⚡ One-Page Chapter Summary

### The History of Training Optimization

```
1964: Polyak Momentum ──────────────────→ Speeds up SGD using historical velocity
1983: Nesterov Accelerated Gradient ────→ Look-ahead gradient dampens oscillations
2010: Glorot & Bengio Weight Init ───────→ Scaled initialization prevents activation saturation
2012: Hinton Dropout & RMSProp ─────────→ Ensemble-regularizer & adaptive learning rates
2015: He Weight Init & Batch Norm ──────→ Stabilizes ReLU activations & layers covariance
2017: Klambauer SELU ───────────────────→ Self-normalizing sequential deep neural networks
2018: Leslie Smith 1cycle Scheduling ───→ Super-convergence using learning rate ramps
```

### Core Architecture: Pipeline of a Deep Layer

```
INPUT  ──────→  WEIGHT MULTIPLICATION  ──────→  BATCH NORMALIZATION  ──────→  ACTIVATION f(z)
x_i             z = W*x                         (Zero-centers, normalizes,     (ELU/ReLU variants
                                                 scales γ, shifts β)            adds non-linearity)
```

### Core Code Snippet (Best-Practice Default Config)

```python
import tensorflow as tf
from tensorflow import keras
from functools import partial

# Reusable constructor for L2-regularized, He-initialized Dense layer
RegularizedDense = partial(
    keras.layers.Dense,
    kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2(0.01),
    use_bias=False # BN shift parameter β handles the bias
)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    
    RegularizedDense(300),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dropout(rate=0.2),
    
    RegularizedDense(100),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dropout(rate=0.2),
    
    keras.layers.Dense(10, activation="softmax", kernel_initializer="glorot_uniform")
])

# Use Nadam with learning rate scheduling & gradient clipping by norm
optimizer = keras.optimizers.Nadam(learning_rate=0.001, clipnorm=1.0)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# OUTPUT: Optimized DNN architecture ready for deep training.
```

### Summary Configuration Decisions

| Scenario | Kernel Initializer | Activation Function | Normalization | Regularization | Optimizer | LR Schedule |
|---|---|---|---|---|---|---|
| **Default DNN** | He initialization | ELU | Batch Norm (if deep) | Early stopping (+ L2) | Nadam (or Adam) | 1cycle |
| **Self-Normalizing** | LeCun initialization | SELU | None (Self-Norm) | Alpha Dropout | Nadam (or SGD+NAG) | 1cycle |
| **Sparse Model** | He initialization | ELU | Batch Norm (if deep) | $\ell_1$ Regularization | Nadam | 1cycle |
| **Low-Latency** | He initialization | LeakyReLU / ReLU | None (or BN fused) | Early stopping | Nadam | 1cycle |
| **Risk-Sensitive**| He initialization | ELU | Batch Norm | MC Dropout | Nadam | 1cycle |

---

## 🏆 Top 5 Things to Remember

1. **Match Initialization to Activation:** Use **He initialization** for ReLU and its variants, **LeCun initialization** for SELU, and **Glorot initialization** for Sigmoid/Tanh.
2. **Batch Normalization Placement:** Place Batch Normalization before or after activation layers. If placing it **before**, set `use_bias=False` on the preceding dense layer to save parameters.
3. **Warm-up Transfer Learning:** When using transfer learning, freeze the reused layers during the initial epochs to prevent the large gradients of the random new output layer from destroying pretrained weights.
4. **Adam is Great, NAG is Stable:** While adaptive optimizers (Adam, Nadam, RMSProp) converge quickly and require minimal learning rate tuning, switch to **SGD + Nesterov Accelerated Gradient (NAG)** if your model struggles to generalize to the test set.
5. **Epoch Reset Warning:** When resuming training from a saved model, Keras resets the epoch counter to 0. Always pass `initial_epoch` to `fit()` to prevent epoch-based learning rate schedulers from spiking the learning rate.

---

## 🔗 Related Chapters

* **Chapter 10**: [Introduction to ANNs with Keras](../CH_10_Introduction_to_Artificial_Neural_Networks_with_Keras/notes.md) - Core concepts of multilayer perceptrons, backpropagation, and basic Keras APIs.
* **Chapter 12**: Custom Models and Training with TensorFlow - Deep dive into customizing loss functions, layers, metrics, and training loops when Keras defaults are insufficient.

---

*Notes created from 44 textbook pages covering pp. 361–404 of Hands-On ML with Scikit-Learn, Keras, and TensorFlow (2nd edition) by Aurélien Géron.*
