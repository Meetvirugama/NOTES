# 📚 Chapter 4: Training Models
### Complete Study Notes — Professor Level

> **Opening the Black Box: Linear Models, Optimization, and Regularization**

---

## 🖼️ Visual Gallery

| # | Graph | Module | File |
|---|-------|--------|------|
| 01 | Gradient Descent Paths in Parameter Space | 2 | [01_gd_paths.png](Visuals/01_gd_paths.png) |
| 02 | Underfitting vs Overfitting (Learning Curves) | 3 | [02_learning_curves.png](Visuals/02_learning_curves.png) |
| 03 | Early Stopping Regularization | 4 | [03_early_stopping.png](Visuals/03_early_stopping.png) |
| 04 | The Sigmoid Function | 5 | [04_sigmoid_function.png](Visuals/04_sigmoid_function.png) |

---

## 🗺️ Master Index

| Module | Topic | File |
|--------|-------|------|
| 01 | Linear Regression & The Normal Equation | [01_Linear_Regression_Normal_Equation.md](Detailed_Notes/01_Linear_Regression_Normal_Equation.md) |
| 02 | Gradient Descent (Batch, Stochastic, Mini-batch) | [02_Gradient_Descent.md](Detailed_Notes/02_Gradient_Descent.md) |
| 03 | Polynomial Regression, Learning Curves & Bias/Variance | [03_Polynomial_Regression_Learning_Curves.md](Detailed_Notes/03_Polynomial_Regression_Learning_Curves.md) |
| 04 | Regularized Linear Models | [04_Regularized_Linear_Models.md](Detailed_Notes/04_Regularized_Linear_Models.md) |
| 05 | Logistic Regression & Softmax Regression | [05_Logistic_Softmax_Regression.md](Detailed_Notes/05_Logistic_Softmax_Regression.md) |

---

## ⚡ One-Page Chapter Summary

### Linear Regression (The Math Way vs The Iterative Way)
*   **The Math Way (Normal Eq / SVD):** Finds exact weights in one shot. Great for small datasets. **Fails massively** if you have millions of features ($O(n^2)$ to $O(n^3)$ complexity).
*   **The Iterative Way (Gradient Descent):** Takes baby steps downhill to minimize the cost function. **Requires Feature Scaling**. Perfect for millions of features.

### Gradient Descent Architectures
*   **Batch GD:** Uses 100% of data per step. Smooth path, but terribly slow for large data.
*   **Stochastic GD:** Uses 1 random instance per step. Insanely fast, but bounces around wildly.
*   **Mini-batch GD:** Uses a small chunk of data. Best of both worlds (GPU optimized).

### The Bias / Variance Trade-off
*   **High Bias (Underfitting):** Model is too simple (e.g., straight line on a curve). 
    *   *Fix:* More complex model, polynomial features. (More data won't help).
*   **High Variance (Overfitting):** Model is too complex (e.g., 300-degree polynomial memorizing noise).
    *   *Fix:* More data, or **Regularization**.

### Regularization (Constraining Weights)
*   **Ridge (L2):** Smoothly shrinks weights. (Good default).
*   **Lasso (L1):** Forces weights to EXACTLY ZERO. Output is a sparse model (feature selection).
*   **Elastic Net:** A stable mix of Ridge and Lasso. (Preferred over pure Lasso).
*   **Early Stopping:** Stop Gradient Descent the moment validation error hits rock bottom.

### Classification Models
*   **Logistic Regression:** Linear regression passed through a Sigmoid (S-curve) to output a probability (0 to 1). Uses **Log Loss** cost function.
*   **Softmax Regression:** Generalization of Logistic for mutually exclusive multiclass problems. Probabilities sum to 100%. Uses **Cross Entropy** cost function.

---

## 🏆 Top 5 Things to Remember
1. **Never use the Normal Equation for datasets with >100,000 features.** The algorithm will hang or crash. Use Gradient Descent.
2. **Always scale your features before Gradient Descent or Regularization.** (e.g., `StandardScaler`).
3. **If learning curves plateau close together at a high error rate, the model is underfitting.** Adding more training data is useless.
4. **Lasso ($\ell_1$) performs automatic feature selection.** It eliminates useless features by setting their weights exactly to zero.
5. **Softmax Regression is ONLY for mutually exclusive classes.** If predicting multiple tags on an image (e.g., 'dog' AND 'outdoor'), use multiple binary Logistic Regressions instead.

---

## 🔗 Related Chapters
*   **Chapter 1 & 2:** Linear Regression and MSE introduced.
*   **Chapter 5:** Support Vector Machines (which use a different type of optimization margin).
*   **Chapter 10 & 11:** Deep Neural Networks — which heavily rely on Mini-batch Gradient Descent, Early Stopping, and Cross Entropy loss.
