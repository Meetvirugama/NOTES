# 🏷️ Module 3: Advanced PCA (Compression, Randomized, Incremental)
> **Ch. 8 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [PCA for Image Compression](#concept-1)
3. [Randomized PCA (For speed)](#concept-2)
4. [Incremental PCA (For huge datasets)](#concept-3)
5. [Common Beginner Mistakes](#mistakes)
6. [Interview Q&A](#interview)
7. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** Standard PCA requires the entire dataset to fit into your computer's RAM, and the math (Full SVD) gets incredibly slow on massive datasets. To solve this, we have **Randomized PCA** (a stochastic approximation that is lightning fast) and **Incremental PCA** (which allows you to process datasets that are larger than your RAM by feeding them in tiny batches). We also look at how PCA can be reversed to act as a powerful image compression algorithm.

---

## 🔍 1. PCA for Image Compression {#concept-1}

If you apply PCA to the MNIST dataset (784 features/pixels per image) and preserve 95% of the variance, you will find that it only takes about 154 dimensions to hold that information.
*   The dataset is now **less than 20% of its original size**.
*   This is a massive compression ratio that will drastically speed up any downstream classification algorithm (like an SVM).

**Decompression (Reconstruction Error):**
It is possible to decompress the reduced dataset back to 784 dimensions using the `inverse_transform()` method.
Because we dropped 5% of the variance (the noise, the fine details), we will not get the *exact* original image back. It will be slightly blurry.
The mean squared distance between the original data and the reconstructed data is called the **reconstruction error**.

```python
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train) # Compress to 154D
X_recovered = pca.inverse_transform(X_reduced) # Decompress back to 784D
```

---

## 🔍 2. Randomized PCA {#concept-2}

If the number of dimensions $d$ you want to keep is much smaller than the original number of features $n$, running the full SVD math is a huge waste of time. 

Scikit-Learn offers a stochastic algorithm called **Randomized PCA** that quickly finds an *approximation* of the first $d$ principal components.
*   Complexity of Full SVD: $O(m \times n^2) + O(n^3)$
*   Complexity of Randomized PCA: $O(m \times d^2) + O(d^3)$
*   **Result:** It is dramatically faster.

```python
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)
```
*(By default, Scikit-Learn's `svd_solver` is set to "auto". It will automatically switch to "randomized" if the dataset is large enough and $d$ is small enough. You don't usually need to specify it manually!).*

---

## 🔍 3. Incremental PCA (IPCA) {#concept-3}

Standard PCA (and Randomized PCA) requires the *entire* training set to be loaded into memory (RAM) at the exact same time. If your dataset is 100GB and you only have 16GB of RAM, your computer will crash.

**Incremental PCA (IPCA)** solves this. It allows you to split the training set into mini-batches and feed them to the algorithm one at a time.
*   Perfect for datasets that don't fit in memory (Out-of-core learning).
*   Perfect for online learning (applying PCA on the fly as new instances arrive from a live server).

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

# np.array_split chops the data into 100 batches
for X_batch in np.array_split(X_train, n_batches):
    # CRITICAL: You must use partial_fit, not fit!
    inc_pca.partial_fit(X_batch) 

X_reduced = inc_pca.transform(X_train)
```

*(Alternatively, you can use NumPy's `memmap` class to map a large file on your hard drive to memory as if it were in RAM, and IPCA will handle pulling the data in chunks automatically).*

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. "Using `fit()` instead of `partial_fit()` in Incremental PCA"** ❌
> If you call `fit(X_batch)` inside a loop, Scikit-Learn completely erases the model and restarts training from scratch on just that single batch. By the end of the loop, your model will only have seen the final 1% of your data. You **must** use `partial_fit()` so that the model updates incrementally.

**2. "Assuming `inverse_transform` perfectly recovers the data"** ❌
> When you reduce dimensions with PCA, you are permanently deleting the variance along the dropped axes. When you use `inverse_transform()`, you reconstruct the data in the original feature space, but the deleted information is gone forever. This is why reconstructed images look slightly blurry compared to the originals.

---

## 🎤 Interview Q&A {#interview}

**Q1: If you have a massive dataset that exceeds your computer's RAM, how can you perform dimensionality reduction on it?**
> **A:**
> You must use Incremental PCA (IPCA). Unlike standard PCA which requires the full dataset in memory for Singular Value Decomposition, IPCA allows you to split the dataset into mini-batches. You then feed these batches to the algorithm sequentially using the `partial_fit()` method, keeping memory usage strictly under control.

**Q2: What is the "Reconstruction Error" in PCA?**
> **A:**
> When you compress a dataset to a lower dimension, you intentionally discard some variance (information) to save space. If you decompress the dataset back to its original dimensions using the inverse transformation, the new data points will not exactly match the original points due to that lost information. The mean squared distance between the original data points and the reconstructed data points is the reconstruction error.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║  MODULE 3 FLASH CARD — Advanced PCA                              ║
╠══════════════════════════════════════════════════════════════════╣
║  PCA COMPRESSION:                                                ║
║  - PCA can compress MNIST from 784 to 154 features.              ║
║  - You can decompress it using inverse_transform().              ║
║  - It will be slightly blurry (The Reconstruction Error).        ║
║                                                                  ║
║  RANDOMIZED PCA:                                                 ║
║  - Stochastic math approximation of PCA.                         ║
║  - Exponentially faster if target dimensions are low.            ║
║  - Scikit-Learn uses it automatically by default ("auto").       ║
║                                                                  ║
║  INCREMENTAL PCA (IPCA):                                         ║
║  - Used when the dataset is too big to fit in RAM.               ║
║  - Feed the data in chunks using partial_fit() inside a loop.    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Previous Module →** [02_Principal_Component_Analysis.md](02_Principal_Component_Analysis.md)  
**🔗 Next Module →** [04_Kernel_PCA.md](04_Kernel_PCA.md)
