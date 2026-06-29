# 📚 Chapter 3: Classification
### Complete Study Notes — Professor Level

> **Every metric from scratch. Every formula. Every trade-off. Every classification type.**

---

## 🖼️ Visual Gallery

| # | Graph | Module | File |
|---|-------|--------|------|
| 01 | Confusion Matrix Anatomy | 2 | [01_confusion_matrix.png](Visuals/01_confusion_matrix.png) |
| 02 | Precision/Recall vs Threshold & PR Curve | 2 | [02_precision_recall_curve.png](Visuals/02_precision_recall_curve.png) |
| 03 | Normalized Error Confusion Matrix | 3 | [03_error_analysis.png](Visuals/03_error_analysis.png) |
| 04 | ROC Curve — SGD vs Random Forest | 3 | [04_roc_curves.png](Visuals/04_roc_curves.png) |

---

## 🗺️ Master Index

| Module | Topic | File |
|--------|-------|------|
| 01 | MNIST, Binary Classifiers & Accuracy Trap | [01_MNIST_Binary_Classifier.md](Detailed_Notes/01_MNIST_Binary_Classifier.md) |
| 02 | Confusion Matrix, Precision, Recall & F1 | [02_Confusion_Matrix_Precision_Recall.md](Detailed_Notes/02_Confusion_Matrix_Precision_Recall.md) |
| 03 | ROC Curve, Multiclass, Error Analysis & Advanced Types | [03_ROC_Multiclass_ErrorAnalysis.md](Detailed_Notes/03_ROC_Multiclass_ErrorAnalysis.md) |

---

## ⚡ One-Page Chapter Summary

### The Classifier Evaluation Pyramid
```
Level 1 (Never use alone)         → ACCURACY (fails on imbalanced data)
Level 2 (Fundamental)             → CONFUSION MATRIX (TN, FP, FN, TP)
Level 3 (From confusion matrix)   → PRECISION, RECALL
Level 4 (Combined)                → F1 SCORE (harmonic mean)
Level 5 (Threshold-independent)   → ROC CURVE + AUC
Level 6 (For rare positives)      → PR CURVE + PR AUC
```

### All Formulas at a Glance
```
Precision  = TP / (TP + FP)          "How good are my positive calls?"
Recall     = TP / (TP + FN)          "How many positives did I find?"
F1         = 2*P*R / (P+R)           Harmonic mean. Punishes imbalanced P/R.
FPR        = FP / (FP + TN)          "How often do I cry wolf?"
ROC AUC    = P(score_pos > score_neg) 1.0=perfect, 0.5=random
```

### Classification Type Taxonomy
```
Binary       → {0,1}     e.g., 5-detector
Multiclass   → {0..N}    e.g., 0-9 digit classifier (OvR or OvO)
Multilabel   → [0,1,1,0] e.g., face recognition (multiple tags)
Multioutput  → [35,128,200,...] e.g., image denoising (pixel intensities)
```

### Actual Numbers from the Book (SGD 5-Detector)
```
Confusion matrix: TN=53057, FP=1522, FN=1325, TP=4096
Precision = 72.9%  |  Recall = 75.6%  |  F1 = 74.2%
At 90% precision target → Recall drops to 43.7%
ROC AUC (SGD) = 0.9612  |  ROC AUC (Random Forest) = 0.9983
Random Forest: Precision=99.0%, Recall=86.6%
Multiclass SGD: 84-87% accuracy → 90% after StandardScaler
```

---

## 🏆 Top 5 Things to Remember
1. **Accuracy is misleading on imbalanced data.** Always use Precision + Recall (or F1).
2. **Confusion matrix layout:** Rows = Actual class, Columns = Predicted class. Main diagonal = correct.
3. **Precision/Recall trade-off:** Raising the threshold → Precision ↑, Recall ↓. Always ask "at what recall?" when someone claims high precision.
4. **ROC vs PR:** Rare positive class → use PR curve (ROC AUC is inflated by large TN pool).
5. **OvR vs OvO:** OvR for most algorithms; OvO for algorithms that scale poorly with data (SVMs).

---

## 🔗 Related Chapters
*   **Chapter 5:** SVMs — uses OvO for multiclass. Kernel trick explained.
*   **Chapter 7:** Random Forests — why they achieve 99% precision on MNIST.
*   **Chapter 10:** Neural Networks for MNIST — ~99%+ with deep learning.
