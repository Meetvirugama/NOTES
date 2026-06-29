# 📚 Chapter 6: Decision Trees
### Complete Study Notes — Professor Level

> **The White Box Model: Making Decisions One Rule at a Time**

---

## 🖼️ Visual Gallery

| # | Graph | Module | File |
|---|-------|--------|------|
| 01 | A Trained Decision Tree | 1 | [01_decision_tree.png](Visuals/01_decision_tree.png) |
| 02 | Unregularized vs Regularized Trees | 2 | [02_regularization.png](Visuals/02_regularization.png) |
| 03 | Step-wise Predictions of a Regression Tree | 3 | [03_regression_predictions.png](Visuals/03_regression_predictions.png) |
| 04 | Sensitivity to Dataset Rotation | 4 | [04_rotation_sensitivity.png](Visuals/04_rotation_sensitivity.png) |

---

## 🗺️ Master Index

| Module | Topic | File |
|--------|-------|------|
| 01 | Training, Predictions & Gini Impurity | [01_Training_and_Predictions.md](Detailed_Notes/01_Training_and_Predictions.md) |
| 02 | The CART Algorithm & Regularization | [02_CART_Algorithm_and_Regularization.md](Detailed_Notes/02_CART_Algorithm_and_Regularization.md) |
| 03 | Regression Trees | [03_Regression_Trees.md](Detailed_Notes/03_Regression_Trees.md) |
| 04 | Limitations and Instability | [04_Limitations_and_Instability.md](Detailed_Notes/04_Limitations_and_Instability.md) |

---

## ⚡ One-Page Chapter Summary

### The Core Concept
*   **Decision Trees:** Split the data by asking yes/no questions (e.g., $x_1 \le 2.45$).
*   **White Box Model:** Highly interpretable. You can look at the tree and know exactly why it made a prediction.
*   **Data Prep:** They require almost ZERO data preparation. **No feature scaling is required!**

### Predictions & Probabilities
*   **Classification:** Traverses the tree to a leaf node. Predicts the most frequent class in that leaf.
*   **Probabilities:** The ratio of the classes inside that leaf node (e.g., $49/54 = 90.7\%$).
*   **Regression:** Predicts the *average target value* of the training instances inside that leaf node (Outputs stair-step plateaus, cannot extrapolate).

### The CART Algorithm
*   **Greedy Approach:** Finds the best split for the *current* node to minimize impurity (Classification) or MSE (Regression). It does not look ahead to find the global optimum.
*   **Gini vs Entropy:** Both measure impurity (how mixed a node is). Gini is faster (default). Entropy sometimes makes slightly more balanced trees. 0 means perfectly pure.

### Regularization (Preventing Overfitting)
*   Decision trees are *non-parametric*. If left unconstrained, they will perfectly memorize the training data (overfit).
*   **To Regularize (Reduce Overfitting):**
    *   Decrease `max_depth`, `max_leaf_nodes`.
    *   Increase `min_samples_leaf`, `min_samples_split`.

### The Fatal Flaw: Instability (High Variance)
*   **Rotation:** Trees only draw perfectly horizontal or vertical boundaries. If you rotate the data 45 degrees, the tree draws a convoluted staircase. (Fix: PCA).
*   **Minor Variations:** Deleting a single data point can cause the algorithm to build a completely different tree.
*   **The Ultimate Fix:** Use Random Forests!

---

## 🏆 Top 5 Things to Remember
1. **Decision Trees do NOT need feature scaling (no `StandardScaler`).**
2. **If a tree is overfitting, you must apply regularization (e.g., decrease `max_depth` or increase `min_samples_leaf`).**
3. **The CART algorithm is greedy.** It finds the best split right now, not the perfect tree overall (because finding the perfect tree is NP-Complete).
4. **Regression Trees predict the average value of a leaf node.** They produce jagged stair-steps and cannot extrapolate outside the training data range.
5. **A single Decision Tree is wildly unstable.** A tiny change in data ruins it. They are best used as building blocks for Random Forests.
