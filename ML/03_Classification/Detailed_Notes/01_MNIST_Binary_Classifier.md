# 🏷️ Module 1: MNIST, Binary Classifiers & Accuracy Trap
> **Ch. 3 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [The MNIST Dataset](#concept-1)
3. [Training a Binary Classifier — SGDClassifier](#concept-2)
4. [The Accuracy Trap on Skewed Datasets](#concept-3)
5. [Common Beginner Mistakes](#mistakes)
6. [Interview Q&A](#interview)
7. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** Classification is the other foundational supervised task besides regression. Chapter 3 uses the **MNIST handwritten digit dataset** as the sandbox. The first critical lesson: **accuracy is a dangerous metric for skewed datasets** — a brain-dead "always predict NOT-5" classifier achieves 90% accuracy just because only 10% of images are 5s. This kills accuracy as a classifier evaluation metric.

**Real-World Analogy:**
*   A cancer screening test that always says "no cancer" achieves 99.9% accuracy (because 99.9% of people don't have cancer). But it is medically worthless.
*   This is the **accuracy paradox** with class-imbalanced data — the most dangerous blind spot in classification.

---

## 🔍 1. The MNIST Dataset {#concept-1}

MNIST is the "Hello World" of ML classification — every new algorithm is benchmarked on it.

| Property | Value |
|---|---|
| Source | Handwritten digits by high school students + US Census Bureau employees |
| Size | 70,000 images (60,000 train + 10,000 test — **pre-split**) |
| Image dimensions | 28 × 28 pixels = **784 features** per image |
| Feature values | 0 (white) to 255 (black) pixel intensity |
| Labels | String digits '0'–'9' (must cast to int) |

```python
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# NOTE: In Scikit-Learn >= 1.2, fetch_openml returns a DataFrame by default.
# Use as_frame=False to get NumPy arrays, or convert with .to_numpy().

print(X.shape)   # (70000, 784)
print(y.shape)   # (70000,)
print(y[0])      # '5'   ← NOTE: string, not int!

# Cast to int for use with ML algorithms
y = y.astype(np.uint8)

# Visualize a single digit
import matplotlib.pyplot as plt
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

# Pre-split (already shuffled — important for CV!)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

> [!IMPORTANT]
> The training set is **already shuffled**. This is important: it ensures all cross-validation folds contain a representative mix of all digit classes. Algorithms sensitive to instance ordering (like SGD) can perform poorly if many similar instances appear in sequence.

---

## 🔍 2. Training a Binary Classifier — SGDClassifier {#concept-2}

**Simplify first:** Start with a binary problem — "Is this image a 5?" (5-detector).

```python
# Create binary labels
y_train_5 = (y_train == 5)  # True for 5s, False for all others
y_test_5  = (y_test == 5)

# Train SGDClassifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Predict
sgd_clf.predict([some_digit])
# array([ True]) — correctly identifies the 5
```

**Why SGDClassifier?**
*   Handles very large datasets efficiently (processes one instance at a time).
*   Naturally suited for **online learning**.
*   Relies on randomness (`random_state=42` for reproducibility).

---

## 🔍 3. The Accuracy Trap on Skewed Datasets {#concept-3}

**Step 1 — naive CV accuracy looks great:**
```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.96355, 0.93795, 0.95615])
```

"Wow — 95% accuracy!" Seems amazing. But hold on:

**Step 2 — a brain-dead "Never5Classifier" beats it:**
```python
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)  # Always predicts NOT-5

cross_val_score(Never5Classifier(), X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.91125, 0.90855, 0.90915])
```

**91% accuracy — with zero intelligence!** Why?
*   Only ~10% of MNIST images are 5s.
*   Always predicting NOT-5 = correct 90% of the time.
*   This is the **accuracy paradox** for skewed/imbalanced datasets.

> [!CAUTION]
> **Never use accuracy alone for imbalanced classification problems.** Accuracy is misleading whenever one class significantly outnumbers the other. Always pair it with precision, recall, F1, or ROC AUC.

**The Custom CV Implementation (from the book):**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    clone_clf.fit(X_train[train_index], y_train_5[train_index])
    y_pred = clone_clf.predict(X_train[test_index])
    n_correct = sum(y_pred == y_train_5[test_index])
    print(n_correct / len(y_pred))
# Prints: 0.9502, 0.96565, 0.96495
```

**`StratifiedKFold`** ensures each fold has the same proportion of each class as the full dataset — essential for imbalanced data.

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. "Looking at accuracy and assuming the model is good"** ❌
> In any problem where one class dominates (spam detection, fraud detection, disease screening), accuracy is nearly meaningless. A model that always predicts "not fraud" on a 0.1% fraud rate dataset achieves 99.9% accuracy with zero fraud detection ability.

**2. "Forgetting to cast MNIST labels from string to int"** ❌
> `y = y.astype(np.uint8)` is mandatory. Most Scikit-Learn algorithms will either fail or produce wrong results on string labels. The book explicitly notes this.

---

## 🎤 Interview Q&A {#interview}

**Q1: What is the accuracy paradox and when does it occur?**
> **A:**
> The accuracy paradox occurs with **class-imbalanced datasets** where one class significantly outnumbers another. A trivial classifier that always predicts the majority class achieves very high accuracy (matching the majority class frequency) while having zero ability to detect the minority class. For MNIST 5-detection, only 10% of images are 5s, so a "never-5" classifier achieves 91% accuracy with no ML whatsoever. The solution is to use metrics like precision, recall, F1, or ROC AUC that explicitly measure performance on the minority class.

**Q2: Why does the book use `StratifiedKFold` instead of regular `KFold` for classification cross-validation?**
> **A:**
> Regular `KFold` splits the data randomly, which can produce folds with very unequal class distributions — especially problematic for imbalanced datasets. For example, a fold might accidentally have very few 5s, making the evaluation noisy and unreliable. `StratifiedKFold` ensures that **each fold maintains the same class proportion as the full dataset** (stratified sampling applied to CV folds), producing a much more reliable and consistent evaluation.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║    MODULE 1 FLASH CARD — MNIST & Binary Classification           ║
╠══════════════════════════════════════════════════════════════════╣
║  MNIST FACTS:                                                    ║
║  - 70K images: 60K train + 10K test (pre-split, shuffled).      ║
║  - 28×28 = 784 features. Pixel values 0–255. Labels = strings.  ║
║  - y = y.astype(np.uint8) — ALWAYS cast!                        ║
║                                                                  ║
║  SGDClassifier:                                                  ║
║  - Stochastic Gradient Descent. Handles large datasets.          ║
║  - set random_state for reproducibility.                         ║
║                                                                  ║
║  ACCURACY PARADOX (THE TRAP):                                    ║
║  - SGDClassifier CV accuracy: ~95%. Looks great!                 ║
║  - Never5Classifier CV accuracy: ~91%. Brain-dead!               ║
║  - Root cause: Only 10% of images are 5s (imbalanced).          ║
║  - Fix: Use Precision, Recall, F1, ROC AUC instead.             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

---

**🔗 Previous Module →** [Back to Chapter Index](../notes.md)  
**🔗 Next Module →** [02_Confusion_Matrix_Precision_Recall.md](02_Confusion_Matrix_Precision_Recall.md)
