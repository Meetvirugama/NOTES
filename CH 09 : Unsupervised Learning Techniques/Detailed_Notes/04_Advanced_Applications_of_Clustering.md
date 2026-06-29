# 🏷️ Module 4: Advanced Applications of Clustering
> **Ch. 9 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [Image Segmentation](#concept-1)
3. [Clustering for Preprocessing](#concept-2)
4. [Semi-Supervised Learning (Label Propagation)](#concept-3)
5. [Active Learning](#concept-4)
6. [Interview Q&A](#interview)
7. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** While clustering is commonly used for data analysis (e.g., customer segmentation), it is actually a powerful utility tool for Machine Learning pipelines. We can use it to compress images (color segmentation), reduce dimensionality before training a classifier, or automatically label thousands of instances based on just a handful of human-labeled examples (Semi-Supervised Learning).

---

## 🔍 1. Image Segmentation {#concept-1}

Image segmentation is partitioning an image into multiple segments. For instance, self-driving cars use *semantic segmentation* (labeling all pixels belonging to pedestrians). 
A much simpler form is **Color Segmentation**, which clusters pixels based entirely on their RGB color values.

**How to compress an image with K-Means:**
1.  Load an image. It is represented as a 3D array (Height, Width, 3 RGB color channels).
2.  Reshape the array into a long 2D list of pixels, where each pixel has 3 features (R, G, B).
3.  Run K-Means on this list (e.g., $k=8$ to find the 8 main colors).
4.  Replace every pixel's exact color with the mean color of its assigned cluster.
5.  Reshape back into the original 3D image shape.
*(Result: The image is now composed of exactly 8 colors, heavily compressing the file size).*

---

## 🔍 2. Clustering for Preprocessing {#concept-2}

Clustering is a highly efficient dimensionality reduction technique. You can use it as a preprocessing step *before* a supervised learning algorithm to boost accuracy.

1.  Take your training set and run K-Means (using soft clustering).
2.  The original features are thrown away, and replaced by the distances to the $k$ centroids.
3.  Train a Logistic Regression model on this new, clustered dataset.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Create a pipeline that clusters first, then classifies
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50)),
    ("log_reg", LogisticRegression()),
])

# Since this is a supervised pipeline, you can use GridSearchCV to find the 
# absolute perfect number of clusters!
from sklearn.model_selection import GridSearchCV
param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3)
```
*(In the book's example using the Digits dataset, doing this reduced the classification error rate by almost 30%!).*

---

## 🔍 3. Semi-Supervised Learning (Label Propagation) {#concept-3}

If you have 10,000 unlabeled images, manually labeling them is incredibly expensive. What if you only label 50 of them, and let the algorithm do the rest?

**The Smart Way to Label 50 Images:**
Instead of randomly picking 50 images to label, you should:
1.  Run K-Means to find $k=50$ clusters.
2.  Find the single image closest to each of the 50 centroids. These are the **representative instances**.
3.  Manually look at these 50 representative images and label them.
*(Training a model on these 50 smart, representative instances yields vastly higher accuracy than training on 50 random instances).*

**Label Propagation:**
Now that you have labeled the 50 representative instances, you can automatically propagate their labels to *every other instance* in their respective clusters!
*   **Warning:** Propagating to the edges of the clusters will likely introduce errors (since boundary instances are easily confused). 
*   **Pro-Tip:** Only propagate labels to the 20% of instances that are closest to the centroid in each cluster. Ignore the rest.

---

## 🔍 4. Active Learning {#concept-4}

To continue improving a Semi-Supervised model, you can use **Active Learning**, where a human expert continuously interacts with the algorithm.

**Uncertainty Sampling:**
1.  Train a model on your small, labeled dataset.
2.  Use the model to make predictions on the massive, unlabeled dataset.
3.  Find the instances where the model is *most uncertain* (e.g., the model predicts a 51% chance of Class A, and 49% chance of Class B).
4.  Give these specific, highly-uncertain instances to the human expert to manually label.
5.  Retrain the model, and repeat until the model stops improving.

---

## 🎤 Interview Q&A {#interview}

**Q1: How can K-Means be used as a preprocessing step for a supervised classification algorithm?**
> **A:**
> K-Means can be used for nonlinear dimensionality reduction. Instead of feeding raw features into a classifier, you first run K-Means (e.g., with $k=50$). You then use the `transform()` method to calculate the distance between every instance and all 50 centroids. These 50 distance metrics become the new features for the classifier. This often simplifies the decision boundary for the classifier and significantly boosts accuracy.

**Q2: What is Label Propagation in Semi-Supervised Learning, and how do you mitigate its risks?**
> **A:**
> Label Propagation is the process of manually labeling a small number of representative instances (usually the centroids of clusters), and then automatically assigning those exact same labels to all other unlabeled instances within those clusters. The risk is that instances near the boundary between two clusters might get labeled incorrectly. To mitigate this, you should only propagate labels to the inner core of the cluster (e.g., the 20% of instances closest to the centroid) and leave the boundary instances unlabeled.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║  MODULE 4 FLASH CARD — Advanced Clustering Applications          ║
╠══════════════════════════════════════════════════════════════════╣
║  IMAGE SEGMENTATION:                                             ║
║  - Cluster an image's pixels by RGB values (e.g., k=8).          ║
║  - Replace all pixels with their centroid's color. Compresses!   ║
║                                                                  ║
║  PREPROCESSING FOR CLASSIFICATION:                               ║
║  - Run K-Means, use the distances to centroids as new features.  ║
║  - Because it's part of a supervised pipeline, you can use       ║
║    GridSearchCV to find the optimal 'k' easily!                  ║
║                                                                  ║
║  SEMI-SUPERVISED LEARNING (Label Propagation):                   ║
║  - 1. Cluster unlabeled data into k clusters.                    ║
║  - 2. Manually label the k representative centroids.             ║
║  - 3. Propagate those labels to the core 20% of the clusters.    ║
║                                                                  ║
║  ACTIVE LEARNING (Uncertainty Sampling):                         ║
║  - The model tells the human which specific instances it is      ║
║    confused by, and asks the human to label those manually.      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Previous Module →** [03_Optimal_Number_of_Clusters.md](03_Optimal_Number_of_Clusters.md)  
**🔗 Next Module →** [05_DBSCAN_and_Other_Algorithms.md](05_DBSCAN_and_Other_Algorithms.md)
