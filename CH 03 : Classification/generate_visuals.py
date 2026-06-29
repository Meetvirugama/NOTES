import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#121212", "figure.facecolor": "#121212",
    "axes.edgecolor": "#444", "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#A0A0A0", "ytick.color": "#A0A0A0",
    "grid.color": "#2a2a2a", "text.color": "#FFFFFF",
    "font.family": "sans-serif",
})

VISUALS_DIR = "Visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

def save_fig(fig_id):
    path = os.path.join(VISUALS_DIR, fig_id + ".png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=plt.rcParams["figure.facecolor"])
    print(f"Saved: {path}")
    plt.close()

# -----------------------------------------------------------
# Graph 01: Confusion Matrix Anatomy
# -----------------------------------------------------------
def generate_01():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')

    data = np.array([[53057, 1522], [1325, 4096]])
    labels = [['True Negative\n(TN)\n53,057', 'False Positive\n(FP)\n1,522\n(Type I Error)'],
              ['False Negative\n(FN)\n1,325\n(Type II Error)', 'True Positive\n(TP)\n4,096']]
    colors = [['#1a3a4a', '#4a1a1a'], ['#4a1a1a', '#1a4a2a']]
    cell_colors_text = [['#3498db', '#e74c3c'], ['#e74c3c', '#2ecc71']]

    for i in range(2):
        for j in range(2):
            x, y = 0.15 + j * 0.45, 0.08 + (1 - i) * 0.44
            rect = mpatches.FancyBboxPatch((x, y), 0.38, 0.38, boxstyle='round,pad=0.02',
                                           facecolor=colors[i][j], edgecolor='white', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + 0.19, y + 0.19, labels[i][j], ha='center', va='center',
                    fontsize=9.5, color=cell_colors_text[i][j], fontweight='bold', linespacing=1.5)

    ax.text(0.50, 0.99, 'PREDICTED', ha='center', va='top', fontsize=11, fontweight='bold', color='#E0E0E0')
    ax.text(0.34, 0.77, 'NOT-5', ha='center', va='center', fontsize=10, color='#aaa')
    ax.text(0.72, 0.77, '5', ha='center', va='center', fontsize=10, color='#aaa')
    ax.text(0.05, 0.65, 'ACTUAL', ha='center', va='center', fontsize=11, fontweight='bold',
            color='#E0E0E0', rotation=90)
    ax.text(0.10, 0.71, 'NOT-5', ha='center', va='center', fontsize=10, color='#aaa', rotation=90)
    ax.text(0.10, 0.29, '5', ha='center', va='center', fontsize=10, color='#aaa', rotation=90)

    ax.text(0.50, 0.04,
            'Precision = TP/(TP+FP) = 4096/(4096+1522) = 72.9%      Recall = TP/(TP+FN) = 4096/(4096+1325) = 75.6%',
            ha='center', va='center', fontsize=8.5, color='#aaa')

    plt.title("01: Confusion Matrix — SGD 5-Detector on MNIST (Training Set)", fontsize=12, color='#E0E0E0', pad=10)
    save_fig("01_confusion_matrix")

# -----------------------------------------------------------
# Graph 02: Precision/Recall vs Threshold + PR Curve
# -----------------------------------------------------------
def generate_02():
    # Simulate precision-recall-threshold curves
    thresholds = np.linspace(-3, 3, 300)
    precisions = 1 / (1 + np.exp(-thresholds + 0.5)) * 0.5 + 0.5
    recalls    = 1 / (1 + np.exp(thresholds * 1.2)) * 0.8 + 0.05
    precisions = np.clip(precisions, 0, 1)
    recalls    = np.clip(recalls, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: vs Threshold
    ax1.plot(thresholds, precisions, color='#3498db', lw=2.5, linestyle='--', label='Precision')
    ax1.plot(thresholds, recalls, color='#2ecc71', lw=2.5, label='Recall')
    ax1.axvline(x=1.5, color='#e74c3c', linestyle=':', lw=2, label='~90% Prec threshold')
    ax1.axhline(y=0.9, color='#e74c3c', linestyle=':', lw=1, alpha=0.5)
    ax1.set_xlabel('Decision Threshold', fontsize=10)
    ax1.set_ylabel('Score', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title('Precision & Recall vs. Threshold', fontsize=11, color='#E0E0E0')
    ax1.text(1.6, 0.55, 'At 90% Prec\nRecall drops\nto ~43%!', fontsize=8.5, color='#e74c3c')

    # Right: PR Curve
    ax2.plot(recalls, precisions, color='#f1c40f', lw=2.5)
    ax2.scatter([0.44], [0.90], color='#e74c3c', s=80, zorder=5, label='90% Prec operating point')
    ax2.set_xlabel('Recall', fontsize=10)
    ax2.set_ylabel('Precision', fontsize=10)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title('Precision vs. Recall Curve', fontsize=11, color='#E0E0E0')
    ax2.text(0.6, 0.3, 'Sharp\ndrop ~80%\nrecall', fontsize=8.5, color='#aaa')

    plt.suptitle("02: Precision/Recall Trade-off Analysis", fontsize=13, color='#E0E0E0', y=1.01)
    save_fig("02_precision_recall_curve")

# -----------------------------------------------------------
# Graph 03: Normalized Error Confusion Matrix
# -----------------------------------------------------------
def generate_03():
    np.random.seed(42)
    n = 10
    # Simulate a 10x10 confusion matrix
    conf = np.eye(n) * 500
    for _ in range(80):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            conf[i, j] += np.random.randint(5, 80)

    # Add known patterns from the book
    conf[8, :] = conf[8, :] * 0.7  # column 8 gets extra FP (many things called 8)
    for i in range(n):
        conf[i, 8] += np.random.randint(20, 60)
    conf[3, 5] += 100; conf[5, 3] += 100  # 3/5 confusion

    row_sums = conf.sum(axis=1, keepdims=True)
    norm_conf = conf / row_sums
    np.fill_diagonal(norm_conf, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.matshow(norm_conf, cmap='hot')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted Class", fontsize=10, labelpad=10)
    ax.set_ylabel("Actual Class", fontsize=10)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Highlight key errors
    ax.add_patch(mpatches.Rectangle((7.5, -0.5), 1, 10, fill=False, edgecolor='#3498db', lw=2, label='Column 8: many FP'))
    ax.add_patch(mpatches.Rectangle((4.5, 2.5), 1, 1, fill=False, edgecolor='#e74c3c', lw=2))
    ax.add_patch(mpatches.Rectangle((2.5, 4.5), 1, 1, fill=False, edgecolor='#e74c3c', lw=2))

    ax.text(8.0, -1.5, '8', color='#3498db', fontsize=10, ha='center')
    ax.text(11.5, 3.0, '3->5', color='#e74c3c', fontsize=8)
    ax.text(11.5, 5.0, '5->3', color='#e74c3c', fontsize=8)

    plt.title("03: Normalized Error Confusion Matrix\n(Diagonal=0, Brighter=More Errors)", fontsize=12, color='#E0E0E0')
    save_fig("03_error_analysis")

# -----------------------------------------------------------
# Graph 04: ROC Curves — SGD vs Random Forest
# -----------------------------------------------------------
def generate_04():
    # Simulate ROC curves
    fpr_sgd    = np.array([0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0])
    tpr_sgd    = np.array([0, 0.30, 0.60, 0.75, 0.85, 0.92, 0.96, 0.98, 1.0])
    fpr_rf     = np.array([0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.30, 1.0])
    tpr_rf     = np.array([0, 0.70,  0.85,  0.90, 0.92, 0.96, 0.98, 0.99, 1.0])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_sgd, tpr_sgd, color='#3498db', lw=2.5, linestyle=':', label='SGDClassifier (AUC ≈ 0.961)')
    ax.plot(fpr_rf,  tpr_rf,  color='#2ecc71', lw=2.5, label='Random Forest (AUC ≈ 0.998)')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5)')

    ax.fill_between(fpr_rf, tpr_rf, alpha=0.08, color='#2ecc71')
    ax.scatter([0.05], [0.7568], color='#e74c3c', s=80, zorder=5, label='SGD at 43.7% recall')

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    ax.text(0.5, 0.3, 'Top-left corner\n= perfect\nclassifier', ha='center', fontsize=9, color='#aaa')

    plt.title("04: ROC Curves — SGD vs. Random Forest (MNIST 5-Detector)", fontsize=12, color='#E0E0E0', pad=10)
    save_fig("04_roc_curves")

if __name__ == "__main__":
    print("Generating visuals for Chapter 3...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    print("All Chapter 3 visuals generated successfully.")
