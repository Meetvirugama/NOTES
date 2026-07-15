# 🏷️ Module 3: Discover & Visualize the Data to Gain Insights
> **Ch. 2 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [Visualizing Geographical Data](#concept-1)
3. [Looking for Correlations](#concept-2)
4. [Experimenting with Attribute Combinations](#concept-3)
5. [Common Beginner Mistakes](#mistakes)
6. [Interview Q&A](#interview)
7. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** Exploratory Data Analysis (EDA) on the **Training Set only** reveals hidden patterns, correlations, and data quirks that guide feature engineering. The book demonstrates how combining raw features (rooms per household, bedrooms per room) can double the predictive signal. This phase is **iterative** — initial EDA provides a hypothesis; post-prototype analysis refines it.

**Rule from the book:** Make a copy of the training set for exploration. Never modify the original.

```python
housing = strat_train_set.copy()
```

---

## 🔍 1. Visualizing Geographical Data {#concept-1}

Since the dataset has `latitude` and `longitude`, a scatterplot reveals the geographical distribution immediately:

```python
# First attempt — all points equal weight
housing.plot(kind="scatter", x="longitude", y="latitude")

# Better — use alpha to see density
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```

**Insight:** High-density areas appear as Bay Area, Los Angeles, San Diego, and Central Valley (Sacramento/Fresno corridor).

**The Rich Visualization — Price + Population + Location:**
```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100,  # circle radius = population size
             label="population", figsize=(10, 7),
             c="median_house_value",
             cmap=plt.get_cmap("jet"),       # blue=cheap, red=expensive
             colorbar=True)
plt.legend()
```

**Key insight from this plot:** Housing prices are strongly correlated with:
1. **Location** (proximity to ocean/coast).
2. **Population density** (denser = more expensive).

**Recommendation from the book:** A clustering algorithm on lat/lon could create useful new features measuring proximity to cluster centers (e.g., coastal vs. inland clusters).

---

## 🔍 2. Looking for Correlations {#concept-2}

### Pearson's r (Standard Correlation Coefficient)
Computed via `df.corr()` — measures **linear correlation** between all pairs of numerical attributes.

**Full correlation output (target = `median_house_value`):**

| Attribute | Correlation with target |
|---|---|
| `median_income` | **0.687** (strongest positive) |
| `total_rooms` | 0.135 |
| `housing_median_age` | 0.114 |
| `households` | 0.065 |
| `total_bedrooms` | 0.048 |
| `population` | −0.027 |
| `longitude` | −0.047 |
| `latitude` | **−0.143** (strongest negative) |

**Interpretation:**
*   Ranges from −1 (perfect negative linear) to +1 (perfect positive linear).
*   0 = no **linear** relationship.
*   `median_income` is by far the strongest predictor of housing prices.
*   `latitude` has a negative correlation — moving north (higher latitude) = slightly cheaper prices.

> [!WARNING]
> **Pearson's r ONLY detects linear correlations.** Completely misses non-linear relationships. A variable can have r=0 and still be a perfect predictor if the relationship is non-linear (e.g., U-shaped). Always inspect scatterplots visually.

> [!TIP]
> **Spearman's Rank Correlation** (`housing.corr(method='spearman')`) measures *monotonic* relationships (not just linear). It converts values to ranks first, so it can detect "as X increases, Y always increases" even if the relationship isn't a straight line. Use it alongside Pearson's r for a more complete picture.

### Scatter Matrix — Visualizing Multiple Correlations
```python
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```

*   **Main diagonal** = histograms of each attribute.
*   **Off-diagonal** = scatterplot of attribute pairs.

**Most informative single plot:** `median_income` vs `median_house_value`
```python
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
```

Observations:
*   Strong upward linear trend (confirms r=0.687).
*   Visible horizontal line at $500,000 — the **price cap artifact** from data collection.
*   Other horizontal lines at $450K, $350K, $280K — data quirks that could confuse the algorithm. Consider removing these districts.

---

## 🔍 3. Experimenting with Attribute Combinations {#concept-3}

Raw features like `total_rooms` are meaningless without context. A district with 10,000 rooms means nothing without knowing how many households share them.

**Create three engineered features:**

```python
housing["rooms_per_household"]       = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"]         = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"]  = housing["population"] / housing["households"]
```

**Correlation after adding engineered features:**

| New Attribute | Correlation with target |
|---|---|
| `bedrooms_per_room` | **−0.260** (new strong negative!) |
| `rooms_per_household` | 0.146 (improved from raw 0.135) |
| `population_per_household` | −0.022 |

**Key Insight:** `bedrooms_per_room` (houses with fewer bedrooms relative to total rooms = more expensive) is a **far more informative feature** than either `total_bedrooms` or `total_rooms` individually. This demonstrates the power of domain-informed feature engineering.

> [!TIP]
> This round of exploration is just a starting point. Run EDA quickly, get a baseline model running, then analyze the model's errors for the next round of insights. Feature engineering is **iterative**.

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. "Assuming Pearson's r = 0 means independence"** ❌
> A Pearson correlation of 0 only means no **linear** relationship. The book explicitly shows examples where r=0 but the variables are clearly non-linearly dependent (e.g., X² relationship). Always inspect scatterplots.

**2. "Using raw count features without normalizing by household"** ❌
> `total_rooms = 10,000` in one district could mean 5,000 cramped single-rooms or 500 luxury 20-room houses. Without dividing by households, the feature is nearly meaningless as a price predictor. The book shows `rooms_per_household` improves correlation from 0.135 to 0.146, and `bedrooms_per_room` jumps to −0.260.

---

## 🎤 Interview Q&A {#interview}

**Q1: The `median_house_value` histogram shows a sharp cap at $500,000. What are the two solutions the book recommends?**
> **A:**
> The housing dataset caps median house values at $500,000. This is a problem because the model will learn that prices never exceed this limit. The book suggests two options:
> 1. **Collect proper labels** for the capped districts — obtain the actual median house values from a different source.
> 2. **Remove the capped districts** from both the Training Set and the Test Set — the system should not be penalized for predicting values above $500K if that's what the data actually shows.

**Q2: Explain the feature engineering insight of `bedrooms_per_room`. Why does it outperform both `total_bedrooms` and `total_rooms`?**
> **A:**
> `total_bedrooms` and `total_rooms` are district-level aggregates highly correlated with district size (population). Larger districts have more of everything regardless of quality. The **ratio** `bedrooms_per_room` captures housing **quality**: a low ratio means many non-bedroom rooms (living rooms, offices, etc.), characteristic of larger, more expensive houses. A high ratio suggests cheap studio-like dwellings. The ratio removes the confounding "district size" variable and isolates the quality signal, which explains why it achieves r=−0.260 vs. the much weaker signals of its constituent features.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║         MODULE 3 FLASH CARD — Discover & Visualize              ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY CORRELATION FINDINGS:                                       ║
║  - median_income: r=0.687 → strongest predictor                  ║
║  - latitude:      r=-0.143 → north = cheaper                     ║
║                                                                  ║
║  PEARSON'S r LIMITATION:                                         ║
║  - Measures LINEAR relationships ONLY.                           ║
║  - r=0 does NOT mean independent. Always plot.                   ║
║                                                                  ║
║  ENGINEERED FEATURES (most powerful):                            ║
║  - bedrooms_per_room:  r=-0.260 (new strongest negative)         ║
║  - rooms_per_household: r=0.146 (stronger than total_rooms)      ║
║                                                                  ║
║  DATA QUIRK TO FIX:                                              ║
║  - median_house_value capped at $500K → remove or relabel        ║
║  - Horizontal lines in scatter = data artifacts, confuse model   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

---

**🔗 Previous Module →** [02_Get_the_Data.md](02_Get_the_Data.md)  
**🔗 Next Module →** [04_Prepare_the_Data.md](04_Prepare_the_Data.md)
