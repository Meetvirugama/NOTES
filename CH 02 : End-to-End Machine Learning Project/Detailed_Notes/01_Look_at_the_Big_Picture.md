# 🏷️ Module 1: Look at the Big Picture
> **Ch. 2 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [The California Housing Dataset](#dataset)
3. [Framing the Problem](#concept-1)
4. [ML Pipelines](#concept-2)
5. [Select a Performance Measure — RMSE & MAE](#concept-3)
6. [Notation Reference](#notation)
7. [Check the Assumptions](#concept-4)
8. [Common Beginner Mistakes](#mistakes)
9. [Interview Q&A](#interview)
10. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** Before writing a single line of code, define the business objective, identify the current baseline, frame the ML task type precisely (supervised/unsupervised, regression/classification, batch/online), and select the mathematical performance measure that aligns with how the system will be used in production.

**The Real-World Analogy 🏡:**
You're a newly hired data scientist at a real estate company. Your model's output — a predicted district housing price — feeds into a **downstream ML system** that decides whether to invest in an area. If you optimize the wrong metric or misframe the task, months of work become worthless before a single user sees it.

---

## 🔍 1. The California Housing Dataset {#dataset}

*   **Source:** StatLib repository (1990 California census data).
*   **Target variable:** `median_house_value` — the median price of houses in a block group.
*   **Block group:** The smallest geographical unit for US Census data (typically 600–3,000 people). Called **"districts"** in the book.
*   **Features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`.
*   **Dataset size:** 20,640 instances.

**Important Quirk Noticed During EDA:**
`total_bedrooms` has only 20,433 non-null values → **207 districts are missing this feature.** Must handle before training.

---

## 🔍 2. Framing the Problem {#concept-1}

**Step 1 — Ask the Right Questions:**
1.  What is the business objective? (How does the company use the prediction?)
2.  What does the current solution look like? (This is your baseline.)
3.  How should the ML problem be framed?

**In this Project:**
*   **Current solution:** Human experts manually estimate prices using complex rules. Off by >20% frequently.
*   **Business objective:** Output predictions feed a downstream ML system that decides real estate investment worth.
*   **Task Framing:**

| Criterion | Answer |
|---|---|
| Supervised or Unsupervised? | **Supervised** — each district has a labeled median house value |
| Classification or Regression? | **Regression** — predicting a continuous numeric value |
| Multiple or Univariate? | **Multiple regression** (multiple input features) + **Univariate** (one output: price) |
| Batch or Online? | **Batch** — data is static, fits in memory, no need for real-time adaptation |

---

## 🔍 3. ML Pipelines {#concept-2}

A **data pipeline** is a sequence of data processing components running asynchronously.

**Key Properties of Pipeline Components:**
*   Each component runs asynchronously, pulling a large amount of data, processing, and outputting to the next data store.
*   Components are **self-contained** (interface = data store).
*   If a component breaks, downstream components can often continue using last known-good output.

> [!WARNING]
> A broken component can go **unnoticed for a long time** if proper monitoring is not in place. The data gets stale and overall performance silently degrades. This is the "model rot" problem.

---

## 🔍 4. Select a Performance Measure {#concept-3}

### Root Mean Square Error (RMSE)
The standard metric for regression tasks. Equation 2-1 from the book:

$$\text{RMSE}(\mathbf{X}, h) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)^2}$$

### Mean Absolute Error (MAE)
$$\text{MAE}(\mathbf{X}, h) = \frac{1}{m} \sum_{i=1}^{m} \left| h(\mathbf{x}^{(i)}) - y^{(i)} \right|$$

### The Norm Connection

| Metric | Norm | Sensitivity to Outliers |
|---|---|---|
| RMSE | ℓ₂ (Euclidean norm) | High — squares errors, heavily penalizes large deviations |
| MAE | ℓ₁ (Manhattan norm) | Low — linear penalty |

*   The **ℓₖ norm** of a vector $\mathbf{v}$ with $n$ elements: $\|\mathbf{v}\|_k = (|v_0|^k + |v_1|^k + \ldots + |v_n|^k)^{1/k}$
*   **ℓ₀** = number of nonzero elements; **ℓ∞** = maximum absolute value.
*   **Higher norm index → more weight to large values → more outlier sensitivity.**
*   **Rule:** When outliers are exponentially rare (bell-shaped curve), RMSE performs well and is generally preferred.

---

## 🔍 5. Notation Reference {#notation}

The book uses standard mathematical notation throughout all chapters:

| Symbol | Meaning | Example |
|---|---|---|
| $m$ | Number of instances in the dataset | 2,000 validation districts |
| $\mathbf{x}^{(i)}$ | Feature vector of the $i$-th instance (bold = vector) | Longitude, Latitude, Income, etc. of district 1 |
| $y^{(i)}$ | Label (target value) of the $i$-th instance | Median house value of district 1 |
| $\mathbf{X}$ | Matrix of all feature vectors (rows = instances) | Full 20,640 × 9 feature matrix |
| $h$ | Hypothesis function / prediction function | The trained ML model |
| $\hat{y}^{(i)} = h(\mathbf{x}^{(i)})$ | Predicted value for instance $i$ | $\hat{y}^{(1)} = \$158,400$ |

**Scalar values:** lowercase italic (m, y)  
**Vectors:** lowercase bold (**x**)  
**Matrices:** uppercase bold (**X**)

---

## 🔍 6. Check the Assumptions {#concept-4}

Before writing any code, **list and verify all assumptions**:

*   The downstream system receives raw price values (not bucketed categories).  
*   If you later discover the downstream system only uses price *categories* (cheap/medium/expensive), the entire problem should be reframed as a **classification** task, not regression.
*   Catching this early can save months of wrong work.

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. "Selecting a performance metric before understanding how the model's output will be used"** ❌
> If the downstream system buckets prices into categories, then predicting the price ±$100 to the nearest dollar is meaningless. The metric must align with business impact.

**2. "Treating median income as a US dollar value"** ❌
> From the book: The median income data has been **scaled and capped** (max 15, min 0.5). The number represents roughly tens of thousands of dollars. Always check how data was computed before using it.

---

## 🎤 Interview Q&A {#interview}

**Q1: Why would you choose MAE over RMSE for the housing regression task?**
> **A:**
> The California housing dataset has some districts with extreme, outlier prices (e.g., Beverly Hills mansions in a mostly middle-class state). Since RMSE squares the error, these outliers receive exponentially larger penalties, pulling the loss function away from optimizing for the majority of normal districts. MAE treats all errors linearly and would produce a model that's more robustly useful for the average district. However, if the downstream system is exceptionally sensitive to large pricing mistakes (e.g., a wrong prediction could cause a $10M bad investment decision), RMSE's heavy outlier penalty may actually be the right choice.

**Q2: What makes the California housing regression task a "multiple regression" and "univariate regression" simultaneously?**
> **A:**
> *   **Multiple regression** refers to the number of input features — the model uses multiple predictors (median_income, total_rooms, population, latitude, etc.) to make a prediction.
> *   **Univariate regression** refers to the number of output variables — the model predicts only one target value: `median_house_value`.
> *   If we were predicting both median house value AND median rent simultaneously, it would be a **multivariate regression** problem.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║       MODULE 1 FLASH CARD — Look at the Big Picture             ║
╠══════════════════════════════════════════════════════════════════╣
║  PROJECT FRAMING:                                                ║
║  - Supervised: labeled median house values.                      ║
║  - Regression: continuous output (price).                        ║
║  - Multiple (input) + Univariate (output).                       ║
║  - Batch: static data, fits in memory.                           ║
║                                                                  ║
║  METRICS:                                                        ║
║  - RMSE = ℓ₂ norm. Squares errors. Outlier-sensitive.           ║
║  - MAE  = ℓ₁ norm. Absolute errors. Outlier-robust.             ║
║  - Higher norm index → more weight on large errors.              ║
║                                                                  ║
║  ML PIPELINE:                                                    ║
║  - Sequence of async components. Robust but silently degrades.   ║
║  - ALWAYS monitor. Dead components → stale data → model rot.     ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Next Module →** [02_Get_the_Data.md](02_Get_the_Data.md)
