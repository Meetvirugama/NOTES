import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set global style for a premium dark theme
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#121212",
    "figure.facecolor": "#121212",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#A0A0A0",
    "ytick.color": "#A0A0A0",
    "grid.color": "#2a2a2a",
    "text.color": "#FFFFFF",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

VISUALS_DIR = "Visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

def save_fig(fig_id):
    path = os.path.join(VISUALS_DIR, fig_id + ".png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=plt.rcParams["figure.facecolor"])
    print(f"Saved: {path}")
    plt.close()

# ---------------------------------------------------------
# Graph 01: Traditional vs ML Pipeline
# ---------------------------------------------------------
def generate_01():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    BOX = dict(boxstyle='round,pad=0.5', lw=2)

    # --- Traditional ---
    ax.text(0.27, 0.93, 'TRADITIONAL PROGRAMMING', fontsize=12, fontweight='bold', color='#3498db', ha='center')
    ax.add_patch(mpatches.FancyBboxPatch((0.01, 0.58), 0.16, 0.24, **BOX, edgecolor='#3498db', facecolor='#1a2a3a'))
    ax.text(0.09, 0.70, '📊 Data', ha='center', va='center', fontsize=11)
    ax.add_patch(mpatches.FancyBboxPatch((0.20, 0.58), 0.16, 0.24, **BOX, edgecolor='#3498db', facecolor='#1a2a3a'))
    ax.text(0.28, 0.70, '📜 Rules', ha='center', va='center', fontsize=11)
    ax.add_patch(mpatches.FancyBboxPatch((0.39, 0.62), 0.14, 0.16, **BOX, edgecolor='#9b59b6', facecolor='#2d1b3a'))
    ax.text(0.46, 0.70, '💻 Computer', ha='center', va='center', fontsize=10)
    ax.add_patch(mpatches.FancyBboxPatch((0.57, 0.58), 0.16, 0.24, **BOX, edgecolor='#2ecc71', facecolor='#1a3a25'))
    ax.text(0.65, 0.70, '✅ Answers', ha='center', va='center', fontsize=11)
    for x1, x2 in [(0.17, 0.20), (0.36, 0.39), (0.53, 0.57)]:
        ax.annotate('', xy=(x2, 0.70), xytext=(x1, 0.70), arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # --- ML ---
    ax.text(0.27, 0.42, 'MACHINE LEARNING', fontsize=12, fontweight='bold', color='#2ecc71', ha='center')
    ax.add_patch(mpatches.FancyBboxPatch((0.01, 0.08), 0.16, 0.24, **BOX, edgecolor='#2ecc71', facecolor='#1a3a25'))
    ax.text(0.09, 0.20, '📊 Data', ha='center', va='center', fontsize=11)
    ax.add_patch(mpatches.FancyBboxPatch((0.20, 0.08), 0.16, 0.24, **BOX, edgecolor='#e74c3c', facecolor='#3a1a1a'))
    ax.text(0.28, 0.20, '✅ Answers\n(Labels)', ha='center', va='center', fontsize=10)
    ax.add_patch(mpatches.FancyBboxPatch((0.39, 0.12), 0.14, 0.16, **BOX, edgecolor='#9b59b6', facecolor='#2d1b3a'))
    ax.text(0.46, 0.20, '💻 ML Algo', ha='center', va='center', fontsize=10)
    ax.add_patch(mpatches.FancyBboxPatch((0.57, 0.08), 0.16, 0.24, **BOX, edgecolor='#3498db', facecolor='#1a2a3a'))
    ax.text(0.65, 0.20, '📜 Rules\n(Model)', ha='center', va='center', fontsize=10)
    for x1, x2 in [(0.17, 0.20), (0.36, 0.39), (0.53, 0.57)]:
        ax.annotate('', xy=(x2, 0.20), xytext=(x1, 0.20), arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    ax.axhline(0.50, color='#444', linewidth=0.8, linestyle='--')
    plt.title("01: Traditional Programming vs. Machine Learning", fontsize=14, color='#E0E0E0', pad=15)
    save_fig("01_traditional_vs_ml")

# ---------------------------------------------------------
# Graph 02: ML Taxonomy Tree
# ---------------------------------------------------------
def generate_02():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    def node(ax, x, y, text, color):
        ax.add_patch(mpatches.FancyBboxPatch((x-1.0, y-0.3), 2.0, 0.6, boxstyle='round,pad=0.15', facecolor=color, edgecolor='white', alpha=0.85))
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    def edge(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2+0.3), xytext=(x1, y1-0.3), arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.2))

    node(ax, 5, 5.5, 'Machine Learning Systems', '#34495e')

    # Axis 1: Supervision
    node(ax, 1.5, 4.2, 'Supervision', '#2c3e50')
    node(ax, 0.5, 2.8, 'Supervised', '#3498db')
    node(ax, 1.5, 2.8, 'Unsupervised', '#e67e22')
    node(ax, 2.5, 2.8, 'RL', '#e74c3c')
    # Labels
    ax.text(0.5, 2.3, 'Classification\nRegression', ha='center', fontsize=7, color='#aaa')
    ax.text(1.5, 2.3, 'Clustering\nPCA, Anomaly', ha='center', fontsize=7, color='#aaa')
    ax.text(2.5, 2.3, 'Policy Learning\nAlphaGo', ha='center', fontsize=7, color='#aaa')

    # Axis 2: Batch vs Online
    node(ax, 5, 4.2, 'Learning Style', '#2c3e50')
    node(ax, 4.2, 2.8, 'Batch', '#27ae60')
    node(ax, 5.8, 2.8, 'Online', '#1abc9c')
    ax.text(4.2, 2.3, 'Train once\nAll data', ha='center', fontsize=7, color='#aaa')
    ax.text(5.8, 2.3, 'Incremental\nAdaptive', ha='center', fontsize=7, color='#aaa')

    # Axis 3: Instance vs Model
    node(ax, 8.5, 4.2, 'Generalization', '#2c3e50')
    node(ax, 7.5, 2.8, 'Instance-Based', '#9b59b6')
    node(ax, 9.5, 2.8, 'Model-Based', '#8e44ad')
    ax.text(7.5, 2.3, 'Similarity\nKNN', ha='center', fontsize=7, color='#aaa')
    ax.text(9.5, 2.3, 'Equation (θ)\nLinear Reg', ha='center', fontsize=7, color='#aaa')

    for x2 in [1.5, 5, 8.5]:
        edge(ax, 5, 5.5, x2, 4.2)
    for x2 in [0.5, 1.5, 2.5]:
        edge(ax, 1.5, 4.2, x2, 2.8)
    for x2 in [4.2, 5.8]:
        edge(ax, 5, 4.2, x2, 2.8)
    for x2 in [7.5, 9.5]:
        edge(ax, 8.5, 4.2, x2, 2.8)

    plt.title("02: Machine Learning Systems Taxonomy", fontsize=14, color='#E0E0E0', pad=15)
    save_fig("02_ml_taxonomy")

# ---------------------------------------------------------
# Graph 03: Overfitting vs Good Fit vs Underfitting
# ---------------------------------------------------------
def generate_03():
    np.random.seed(42)
    X = np.sort(np.random.rand(25) * 10 - 1)
    y = np.sin(X * 0.9) + np.random.randn(25) * 0.25

    X_plot = np.linspace(X.min(), X.max(), 200)
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    titles = ['Underfitting\n(High Bias)', 'Good Fit', 'Overfitting\n(High Variance)']
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for i, ax in enumerate(axes):
        ax.scatter(X, y, color='white', s=20, zorder=5, alpha=0.8)
        ax.set_title(titles[i], color=colors[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('GDP per capita', fontsize=9)
        ax.tick_params(colors='#A0A0A0')
        ax.grid(True, alpha=0.2)
        
        if i == 0:  # Underfit: flat line
            p = np.poly1d([0, np.mean(y)])
            ax.plot(X_plot, p(X_plot), color=colors[i], lw=2.5)
        elif i == 1:  # Good fit: smooth sin curve
            ax.plot(X_plot, np.sin(X_plot * 0.9), color=colors[i], lw=2.5)
        else:  # Overfit: high degree polynomial
            z = np.polyfit(X, y, 18)
            p = np.poly1d(z)
            ax.plot(X_plot, p(X_plot), color=colors[i], lw=2)
            ax.set_ylim(-2.5, 2.5)

    if axes[2].get_ylim()[0] < -2:
        axes[2].set_ylim(-2.5, 2.5)

    plt.suptitle("03: Underfitting vs. Good Fit vs. Overfitting", fontsize=13, color='#E0E0E0', y=1.02)
    save_fig("03_overfitting_underfitting")

# ---------------------------------------------------------
# Graph 04: Train/Val/Test Split & K-Fold CV
# ---------------------------------------------------------
def generate_04():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    # Top: Simple split
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1)
    ax1.add_patch(plt.Rectangle((0, 0.2), 7, 0.6, color='#2c5f8a'))
    ax1.add_patch(plt.Rectangle((7, 0.2), 1.5, 0.6, color='#8a5c2c'))
    ax1.add_patch(plt.Rectangle((8.5, 0.2), 1.5, 0.6, color='#2c8a4a'))
    ax1.text(3.5, 0.5, 'Training Set (70%)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(7.75, 0.5, 'Val\n(15%)', ha='center', va='center', fontsize=10)
    ax1.text(9.25, 0.5, 'Test\n(15%)', ha='center', va='center', fontsize=10)
    ax1.set_title("Train / Validation / Test Split", color='#E0E0E0', fontsize=12)

    # Bottom: K-Fold
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1)
    folds = 5
    w = 10 / folds
    fold_colors = ['#2c5f8a'] * folds
    for fold in range(folds):
        for i in range(folds):
            color = '#8a5c2c' if i == fold else '#2c5f8a'
            ax2.add_patch(plt.Rectangle((i * w, (folds - fold - 1) * 0.18), w, 0.15, color=color, ec='black', lw=0.5))
        ax2.text(-0.15, (folds - fold - 1) * 0.18 + 0.075, f'K{fold+1}', ha='right', va='center', fontsize=8, color='#aaa')
    ax2.set_title("5-Fold Cross-Validation (orange = Validation fold)", color='#E0E0E0', fontsize=12)

    plt.suptitle("04: Dataset Splitting Strategies", fontsize=13, color='#E0E0E0', y=1.02)
    save_fig("04_cross_validation")

if __name__ == "__main__":
    print("Generating visuals for Chapter 1...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    print("✅ All Chapter 1 visuals generated successfully.")
