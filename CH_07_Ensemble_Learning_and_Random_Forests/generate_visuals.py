import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle, ConnectionPatch

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
# Graph 01: The Law of Large Numbers
# -----------------------------------------------------------
def generate_01():
    np.random.seed(42)
    heads_proba = 0.51
    coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
    cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative_heads_ratio, color='#3498db', alpha=0.5, lw=1)
    ax.plot([0, 10000], [0.51, 0.51], "r--", linewidth=2, label="51% (True Probability of Heads)")
    ax.plot([0, 10000], [0.5, 0.5], "w-", linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Number of coin tosses")
    ax.set_ylabel("Heads ratio")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 10000)
    ax.set_ylim(0.42, 0.58)
    ax.grid(alpha=0.2)
    ax.set_title("01: The Law of Large Numbers (Wisdom of the Crowd)", color='#E0E0E0', pad=15)
    
    save_fig("01_voting_classifiers")

# -----------------------------------------------------------
# Graph 02: Bagging & Pasting (Parallel Training Diagram)
# -----------------------------------------------------------
def generate_02():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Original Data
    ax.text(0.5, 0.9, "Original Training Set\n(m instances)", ha='center', va='center', 
            bbox=dict(boxstyle="square,pad=1", fc="#8e44ad", ec="#9b59b6", lw=2), fontsize=12)
            
    # Subsets
    for i, x in enumerate([0.15, 0.5, 0.85]):
        ax.annotate("", xy=(x, 0.7), xytext=(0.5, 0.85), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
        ax.text(x, 0.65, f"Random Subset {i+1}", ha='center', va='center', 
                bbox=dict(boxstyle="square,pad=0.5", fc="#2980b9", ec="#3498db", lw=2), fontsize=10)
                
    # Predictors
    for i, x in enumerate([0.15, 0.5, 0.85]):
        ax.annotate("", xy=(x, 0.45), xytext=(x, 0.6), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
        ax.text(x, 0.4, f"Predictor {i+1}\n(e.g. Decision Tree)", ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.5", fc="#27ae60", ec="#2ecc71", lw=2), fontsize=10)
                
    # Aggregation
    for i, x in enumerate([0.15, 0.5, 0.85]):
        ax.annotate("", xy=(0.5, 0.15), xytext=(x, 0.35), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
        
    ax.text(0.5, 0.1, "Aggregation\n(Mode or Average)", ha='center', va='center', 
            bbox=dict(boxstyle="circle,pad=0.5", fc="#e67e22", ec="#d35400", lw=2), fontsize=12)
            
    ax.axis('off')
    ax.set_title("02: Bagging & Pasting (Parallel Training)", color='#E0E0E0', fontsize=14, pad=10)
    save_fig("02_bagging_pasting")

# -----------------------------------------------------------
# Graph 03: Feature Importance (Bar chart)
# -----------------------------------------------------------
def generate_03():
    features = ['Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width']
    importances = [0.44, 0.42, 0.11, 0.03]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, importances, color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'])
    ax.invert_yaxis()  # labels read top-to-bottom
    
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.2f}', va='center', color='white', fontweight='bold')
                
    ax.set_xlabel("Relative Feature Importance")
    ax.set_title("03: Random Forest Feature Importance (Iris Dataset)", color='#E0E0E0', pad=15)
    ax.grid(axis='x', alpha=0.2)
    save_fig("03_feature_importance")

# -----------------------------------------------------------
# Graph 04: AdaBoost vs Gradient Boosting
# -----------------------------------------------------------
def generate_04():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: AdaBoost (Instance weights)
    np.random.seed(42)
    X = np.random.rand(15, 2)
    y = (X[:, 0] > 0.5).astype(int)
    
    sizes = np.ones(15) * 50
    sizes[3] = 300 # Misclassified point gets huge weight
    sizes[7] = 200
    
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='#3498db', s=sizes[y==0], marker='o', edgecolors='w', label='Class 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='#e74c3c', s=sizes[y==1], marker='^', edgecolors='w', label='Class 1')
    ax1.axvline(0.6, color='w', lw=2, linestyle='--')
    ax1.set_title("AdaBoost: Updating Instance Weights", color='#E0E0E0')
    ax1.legend(loc='lower right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Right: Gradient Boosting (Residuals)
    X_reg = np.linspace(0, 10, 20)
    y_reg = np.sin(X_reg)
    y_pred1 = np.ones_like(X_reg) * np.mean(y_reg)
    residuals = y_reg - y_pred1
    
    ax2.plot(X_reg, residuals, 'ro', markersize=8, label="Residuals ($y - \hat{y}_1$)")
    ax2.plot(X_reg, residuals, 'r-', alpha=0.3)
    ax2.axhline(0, color='w', lw=2, linestyle='--')
    
    ax2.set_title("Gradient Boosting: Fitting Residual Errors", color='#E0E0E0')
    ax2.legend(loc='lower right')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.suptitle("04: Boosting (Sequential Training)", fontsize=14, color='#E0E0E0', y=1.02)
    save_fig("04_adaboost_vs_gradient")

# -----------------------------------------------------------
# Graph 05: Stacking
# -----------------------------------------------------------
def generate_05():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Layer 1
    ax.text(0.2, 0.7, "Random Forest\nPredictor", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.5", fc="#27ae60", ec="#2ecc71", lw=2), fontsize=10)
    ax.text(0.5, 0.7, "SVM\nPredictor", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.5", fc="#2980b9", ec="#3498db", lw=2), fontsize=10)
    ax.text(0.8, 0.7, "Logistic Reg\nPredictor", ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.5", fc="#8e44ad", ec="#9b59b6", lw=2), fontsize=10)
            
    # Values
    ax.text(0.2, 0.5, "Pred: 3.1", ha='center', va='center', color='#f1c40f', fontweight='bold')
    ax.text(0.5, 0.5, "Pred: 2.7", ha='center', va='center', color='#f1c40f', fontweight='bold')
    ax.text(0.8, 0.5, "Pred: 2.9", ha='center', va='center', color='#f1c40f', fontweight='bold')
    
    # Arrows to blender
    ax.annotate("", xy=(0.4, 0.35), xytext=(0.2, 0.45), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
    ax.annotate("", xy=(0.5, 0.35), xytext=(0.5, 0.45), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
    ax.annotate("", xy=(0.6, 0.35), xytext=(0.8, 0.45), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
    
    # Layer 2
    ax.text(0.5, 0.25, "Layer 2: Blender\n(Meta-Learner)", ha='center', va='center', 
            bbox=dict(boxstyle="square,pad=0.8", fc="#e67e22", ec="#d35400", lw=2), fontsize=12)
            
    # Final output
    ax.annotate("", xy=(0.5, 0.05), xytext=(0.5, 0.15), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
    ax.text(0.5, 0.0, "Final Prediction: 3.0", ha='center', va='center', color='#2ecc71', fontsize=12, fontweight='bold')
    
    ax.axis('off')
    ax.set_title("05: Stacking (Stacked Generalization)", color='#E0E0E0', fontsize=14, pad=10)
    save_fig("05_stacking")

if __name__ == "__main__":
    print("Generating visuals for Chapter 7...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    generate_05()
    print("All Chapter 7 visuals generated successfully.")
