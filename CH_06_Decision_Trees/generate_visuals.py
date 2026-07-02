import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import FancyBboxPatch, Rectangle

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
# Graph 01: A Trained Decision Tree
# -----------------------------------------------------------
def generate_01():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simple mock up of a tree structure
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#2c3e50", ec="#34495e", lw=2)
    
    # Root
    ax.text(0.5, 0.9, "Petal Length <= 2.45 cm\ngini = 0.66\nsamples = 150\nvalue = [50, 50, 50]",
            ha="center", va="center", bbox=bbox_props, color='white', fontsize=10)
    
    # Arrows from root
    ax.annotate("", xy=(0.25, 0.6), xytext=(0.45, 0.8), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
    ax.annotate("", xy=(0.75, 0.6), xytext=(0.55, 0.8), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
    ax.text(0.35, 0.75, "True", color='#2ecc71', fontsize=11, fontweight='bold', ha='center')
    ax.text(0.65, 0.75, "False", color='#e74c3c', fontsize=11, fontweight='bold', ha='center')
    
    # Left Child (Leaf)
    bbox_leaf = dict(boxstyle="round,pad=0.5", fc="#e67e22", ec="#d35400", lw=2)
    ax.text(0.25, 0.5, "gini = 0.0\nsamples = 50\nvalue = [50, 0, 0]\nClass: Setosa",
            ha="center", va="center", bbox=bbox_leaf, color='white', fontsize=10)
            
    # Right Child (Node)
    ax.text(0.75, 0.5, "Petal Width <= 1.75 cm\ngini = 0.5\nsamples = 100\nvalue = [0, 50, 50]",
            ha="center", va="center", bbox=bbox_props, color='white', fontsize=10)
            
    # Arrows from right child
    ax.annotate("", xy=(0.60, 0.2), xytext=(0.7, 0.4), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
    ax.annotate("", xy=(0.90, 0.2), xytext=(0.8, 0.4), arrowprops=dict(arrowstyle="->", color="#A0A0A0", lw=2))
    
    # Left Leaf 2
    bbox_leaf2 = dict(boxstyle="round,pad=0.5", fc="#27ae60", ec="#2ecc71", lw=2)
    ax.text(0.60, 0.1, "gini = 0.168\nsamples = 54\nvalue = [0, 49, 5]\nClass: Versicolor",
            ha="center", va="center", bbox=bbox_leaf2, color='white', fontsize=10)
            
    # Right Leaf 2
    bbox_leaf3 = dict(boxstyle="round,pad=0.5", fc="#8e44ad", ec="#9b59b6", lw=2)
    ax.text(0.90, 0.1, "gini = 0.043\nsamples = 46\nvalue = [0, 1, 45]\nClass: Virginica",
            ha="center", va="center", bbox=bbox_leaf3, color='white', fontsize=10)

    ax.axis('off')
    ax.set_title("01: A Trained Decision Tree (Mockup)", color='#E0E0E0', fontsize=14, pad=10)
    
    save_fig("01_decision_tree")

# -----------------------------------------------------------
# Graph 02: Unregularized vs Regularized Trees
# -----------------------------------------------------------
def generate_02():
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10
    y = ((X[:, 0] > 5) ^ (X[:, 1] > 5)).astype(int)
    
    # Add some noise
    for i in range(10):
        y[np.random.randint(100)] = 1 - y[np.random.randint(100)]
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Unregularized (overfit) - draw arbitrary complex boxes
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='#3498db', marker='s', s=40, edgecolors='k')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='#e74c3c', marker='^', s=40, edgecolors='k')
    
    # Draw crazy boundaries
    ax1.axvline(5, color='w', lw=2)
    ax1.axhline(5, xmin=0, xmax=0.5, color='w', lw=2)
    ax1.axhline(5, xmin=0.5, xmax=1, color='w', lw=2)
    
    # Add overfit squares around noise
    ax1.add_patch(Rectangle((3.8, 6.8), 0.5, 0.5, fill=False, edgecolor='w', lw=1.5))
    ax1.add_patch(Rectangle((7.8, 2.8), 0.5, 0.5, fill=False, edgecolor='w', lw=1.5))
    ax1.add_patch(Rectangle((1.8, 1.8), 0.5, 0.5, fill=False, edgecolor='w', lw=1.5))
    ax1.add_patch(Rectangle((8.8, 8.8), 0.5, 0.5, fill=False, edgecolor='w', lw=1.5))
    
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title("No Restrictions (Overfitting)", color='#E0E0E0')
    
    # Right: Regularized
    ax2.scatter(X[y==0, 0], X[y==0, 1], c='#3498db', marker='s', s=40, edgecolors='k')
    ax2.scatter(X[y==1, 0], X[y==1, 1], c='#e74c3c', marker='^', s=40, edgecolors='k')
    
    # Draw simple clean boundaries
    ax2.axvline(5, color='w', lw=2)
    ax2.axhline(5, xmin=0, xmax=0.5, color='w', lw=2)
    ax2.axhline(5, xmin=0.5, xmax=1, color='w', lw=2)
    
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("min_samples_leaf=4 (Generalizes Better)", color='#E0E0E0')

    plt.suptitle("02: Regularization Hyperparameters", fontsize=14, color='#E0E0E0', y=1.02)
    save_fig("02_regularization")

# -----------------------------------------------------------
# Graph 03: Step-wise Predictions of a Regression Tree
# -----------------------------------------------------------
def generate_03():
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = np.sin(X) + np.random.randn(50) * 0.2
    
    # Simulate a regression tree depth=2
    # thresholds at X=3 and X=7
    # regions: 0-3, 3-7, 7-10
    
    X_pred = np.linspace(0, 10, 500)
    y_pred = np.zeros_like(X_pred)
    
    y_pred[X_pred <= 3] = np.mean(y[X <= 3])
    y_pred[(X_pred > 3) & (X_pred <= 7)] = np.mean(y[(X > 3) & (X <= 7)])
    y_pred[X_pred > 7] = np.mean(y[X > 7])
    
    # Simulate depth=3
    y_pred3 = np.zeros_like(X_pred)
    y_pred3[X_pred <= 1.5] = np.mean(y[X <= 1.5])
    y_pred3[(X_pred > 1.5) & (X_pred <= 3)] = np.mean(y[(X > 1.5) & (X <= 3)])
    y_pred3[(X_pred > 3) & (X_pred <= 5)] = np.mean(y[(X > 3) & (X <= 5)])
    y_pred3[(X_pred > 5) & (X_pred <= 7)] = np.mean(y[(X > 5) & (X <= 7)])
    y_pred3[(X_pred > 7) & (X_pred <= 8.5)] = np.mean(y[(X > 7) & (X <= 8.5)])
    y_pred3[X_pred > 8.5] = np.mean(y[X > 8.5])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(X, y, color='#f1c40f', s=40, edgecolors='k')
    ax1.plot(X_pred, y_pred, color='#e74c3c', lw=3, label=r"$\hat{y}$ (max_depth=2)")
    ax1.set_title("max_depth=2", color='#E0E0E0')
    ax1.legend()
    
    ax2.scatter(X, y, color='#f1c40f', s=40, edgecolors='k')
    ax2.plot(X_pred, y_pred3, color='#3498db', lw=3, label=r"$\hat{y}$ (max_depth=3)")
    ax2.set_title("max_depth=3", color='#E0E0E0')
    ax2.legend()
    
    plt.suptitle("03: Step-wise Predictions of a Regression Tree", fontsize=14, color='#E0E0E0', y=1.02)
    save_fig("03_regression_predictions")

# -----------------------------------------------------------
# Graph 04: Sensitivity to Dataset Rotation
# -----------------------------------------------------------
def generate_04():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    # create linearly separable data vertically
    y = (X[:, 0] > 0.5).astype(int)
    
    # Rotated dataset
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    X_rot = X.dot(R)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Vertical split
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='#3498db', marker='s', s=40, edgecolors='k')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='#e74c3c', marker='^', s=40, edgecolors='k')
    ax1.axvline(0.5, color='w', lw=2)
    ax1.set_title("Straight Data (Perfect 1-split line)", color='#E0E0E0')
    
    # Right: Rotated split (staircase)
    ax2.scatter(X_rot[y==0, 0], X_rot[y==0, 1], c='#3498db', marker='s', s=40, edgecolors='k')
    ax2.scatter(X_rot[y==1, 0], X_rot[y==1, 1], c='#e74c3c', marker='^', s=40, edgecolors='k')
    
    # Draw staircase boundaries
    xs = np.linspace(-0.6, 0.6, 15)
    for i in range(len(xs)-1):
        ax2.plot([xs[i], xs[i+1]], [xs[i]+0.7, xs[i]+0.7], color='w', lw=2)
        if i < len(xs)-2:
            ax2.plot([xs[i+1], xs[i+1]], [xs[i]+0.7, xs[i+1]+0.7], color='w', lw=2)
            
    ax2.set_title("Rotated by 45° (Convoluted Staircase split)", color='#E0E0E0')
    
    plt.suptitle("04: Sensitivity to Dataset Rotation", fontsize=14, color='#E0E0E0', y=1.02)
    save_fig("04_rotation_sensitivity")

if __name__ == "__main__":
    print("Generating visuals for Chapter 6...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    print("All Chapter 6 visuals generated successfully.")
