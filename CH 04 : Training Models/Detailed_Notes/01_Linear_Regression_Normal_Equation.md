# 🏷️ Module 1: Linear Regression & The Normal Equation
> **Ch. 4 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [Linear Regression Model & Cost Function](#concept-1)
3. [The Normal Equation (Closed-form Solution)](#concept-2)
4. [Scikit-Learn's Approach: SVD & Pseudoinverse](#concept-3)
5. [Computational Complexity](#concept-4)
6. [Common Beginner Mistakes](#mistakes)
7. [Interview Q&A](#interview)
8. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** To understand how machine learning works under the hood, we start with the simplest model: **Linear Regression**. Training a model means finding the parameters (weights and bias) that minimize a cost function (usually MSE) over the training set. There are two ways to do this: 
> 1. Compute it instantly using a math formula (The Normal Equation / SVD).
> 2. Iteratively tweak parameters until you reach the bottom (Gradient Descent).
> Module 1 focuses on the direct math approach.

**Real-World Analogy:**
*   Imagine trying to figure out the price of a house based on its size. You want to draw a straight line through all the historical data points. 
*   **The Normal Equation** is like using a massive math formula to perfectly calculate the exact angle and height of that line in one shot.

---

## 🔍 1. Linear Regression Model & Cost Function {#concept-1}

A linear model makes a prediction by computing a **weighted sum** of the input features, plus a constant **bias term** (intercept).

**The Equation (Vectorized Form):**
$$\hat{y} = h_{\theta}(x) = \theta^T \cdot x$$

*   **$\hat{y}$**: Predicted value.
*   **$x$**: Instance's feature vector (with $x_0 = 1$ to handle the bias term).
*   **$\theta$**: The model's parameter vector (containing the bias $\theta_0$ and weights $\theta_1$ to $\theta_n$).
*   **$h_{\theta}$**: The hypothesis function.

**The Cost Function (MSE):**
To train the model, we need to minimize the Mean Squared Error (MSE).
$$\text{MSE}(X, h_{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right)^2$$

> [!NOTE]
> Why minimize MSE instead of RMSE? The value that minimizes a function also minimizes its square root. MSE is mathematically much easier to differentiate.

---

## 🔍 2. The Normal Equation {#concept-2}

To find the value of $\theta$ that minimizes the cost function, there is a closed-form solution — a direct mathematical formula called the **Normal Equation**:

$$\hat{\theta} = (X^T X)^{-1} X^T y$$

**Implementing it from scratch in NumPy:**
```python
import numpy as np

# Generate random linear data: y = 4 + 3x + noise
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add x0 = 1 to each instance for the bias term
X_b = np.c_[np.ones((100, 1)), X] 

# The Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Output:
# array([[4.215],   <- Close to 4 (bias)
#        [2.770]])  <- Close to 3 (weight)
```

**Making a prediction:**
```python
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
```

---

## 🔍 3. Scikit-Learn's Approach: SVD & Pseudoinverse {#concept-3}

While the Normal Equation works, Scikit-Learn's `LinearRegression` class actually uses a different technique under the hood based on **Singular Value Decomposition (SVD)**.

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
# [4.215] [[2.770]]
```

**Why Scikit-Learn doesn't use the pure Normal Equation:**
1. The Normal Equation computes the inverse of $X^T X$. But $X^T X$ might **not be invertible** (singular) — for example, if there are more features than instances ($m < n$), or if some features are perfectly correlated.
2. Scikit-Learn uses the **pseudoinverse** (Moore-Penrose inverse) computed via SVD. 
3. The formula is $\hat{\theta} = X^+ y$. The pseudoinverse $X^+$ is **always defined**, making it much safer.

```python
# How it works under the hood:
np.linalg.pinv(X_b).dot(y)
```

---

## 🔍 4. Computational Complexity {#concept-4}

This is a massive topic for systems design and choosing the right algorithm in production.

| Algorithm | Complexity (Features $n$) | Complexity (Instances $m$) | Out-of-core support |
|---|---|---|---|
| **Normal Equation** | $O(n^{2.4})$ to $O(n^3)$ | $O(m)$ | No (Must fit in memory) |
| **SVD (Scikit-Learn)** | $O(n^2)$ | $O(m)$ | No (Must fit in memory) |

**The Feature Scaling Problem:**
*   If you double the number of features, Scikit-Learn's SVD approach takes **4 times longer**.
*   The Normal Equation takes roughly **5.3 to 8 times longer**.
*   **Conclusion:** Both closed-form solutions are terribly slow when the number of features is large (e.g., 100,000 features). 

> [!TIP]
> Both approaches handle large training sets (millions of instances) efficiently, $O(m)$, **provided the data fits in memory**. If it doesn't fit in memory, or if you have too many features, you must abandon these exact math formulas and use **Gradient Descent** instead.

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. "Assuming Scikit-Learn's LinearRegression uses the Normal Equation"** ❌
> It actually uses SVD and the pseudoinverse. The pseudoinverse handles edge cases gracefully, like when features are highly collinear or $m < n$ (where the standard matrix inverse $X^T X$ would fail).

**2. "Using LinearRegression on text data or image data with 100,000+ features"** ❌
> The computational complexity of the SVD approach is $O(n^2)$. With 100,000 features, $100,000^2$ operations will cause your machine to hang or run out of memory. You must use Stochastic Gradient Descent (`SGDRegressor`) for high-dimensional data.

---

## 🎤 Interview Q&A {#interview}

**Q1: In Linear Regression, why might the Normal Equation fail, and how does Scikit-Learn solve this?**
> **A:** 
> The Normal Equation requires computing the inverse of the matrix $(X^T X)$. If the matrix is singular (non-invertible), the formula crashes. This happens if there are more features than training instances ($m < n$) or if some features are redundant (perfectly collinear). Scikit-Learn avoids this by using Singular Value Decomposition (SVD) to compute the Moore-Penrose pseudoinverse ($X^+$). The pseudoinverse is *always* defined, making the algorithm robust to these edge cases.

**Q2: If you have a training set with millions of features, what training algorithm should you use?**
> **A:**
> You absolutely cannot use the Normal Equation or SVD approaches. Their complexity is $O(n^{2.4})$ to $O(n^3)$ and $O(n^2)$ respectively with regard to features. Millions of features will crash the system. You must use an iterative optimization algorithm like **Stochastic Gradient Descent (SGD)** or **Mini-batch Gradient Descent**, which scale extremely well with the number of features.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║  MODULE 1 FLASH CARD — Linear Regression & Normal Equation       ║
╠══════════════════════════════════════════════════════════════════╣
║  THE MODEL:                                                      ║
║  y_hat = theta^T * X   (Linear combination of weights & inputs)  ║
║                                                                  ║
║  COST FUNCTION:                                                  ║
║  Minimize Mean Squared Error (MSE). Convex bowl shape.           ║
║                                                                  ║
║  NORMAL EQUATION:                                                ║
║  theta_best = inv(X^T * X) * X^T * y                             ║
║  - Exact math solution. No iterations, no learning rate.         ║
║  - Fails if X^T X is non-invertible (e.g., m < n).               ║
║                                                                  ║
║  SCIKIT-LEARN'S WAY:                                             ║
║  theta_best = pinv(X) * y   (Using SVD Pseudoinverse)            ║
║  - Always works. Avoids singular matrix issues.                  ║
║                                                                  ║
║  COMPLEXITY CRASH:                                               ║
║  - Scales O(m) linearly with instances. (Good)                   ║
║  - Scales O(n^2) to O(n^3) with features. (TERRIBLE!)            ║
║  - If >100,000 features → DO NOT USE. Use Gradient Descent.      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Next Module →** [02_Gradient_Descent.md](02_Gradient_Descent.md)
