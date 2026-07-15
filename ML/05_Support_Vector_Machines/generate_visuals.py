import matplotlib.pyplot as plt
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
# Graph 01: Large Margin Classification & Soft Margins
# -----------------------------------------------------------
def generate_01():
    np.random.seed(42)
    # Generate somewhat separable data
    X1 = np.random.randn(25, 2) + np.array([1, 1])
    X2 = np.random.randn(25, 2) + np.array([4, 4])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Hard Margin (Perfect Separation)
    ax1.scatter(X1[:, 0], X1[:, 1], color='#3498db', marker='s', edgecolors='k', s=60)
    ax1.scatter(X2[:, 0], X2[:, 1], color='#2ecc71', marker='^', edgecolors='k', s=80)
    
    # Draw Street
    x_val = np.linspace(0, 5, 100)
    ax1.plot(x_val, -x_val + 5, 'w-', lw=2)  # Decision boundary
    ax1.plot(x_val, -x_val + 3.5, 'w--', lw=1.5, alpha=0.7)  # Margin
    ax1.plot(x_val, -x_val + 6.5, 'w--', lw=1.5, alpha=0.7)  # Margin
    
    # Support Vectors
    ax1.scatter([1.5, 3.5], [2.0, 3.0], s=200, facecolors='none', edgecolors='#f1c40f', lw=2)
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.set_title("Hard Margin (0 Violations)", color='#E0E0E0')
    ax1.grid(alpha=0.2)

    # Right: Soft Margin (With outliers/violations)
    ax2.scatter(X1[:, 0], X1[:, 1], color='#3498db', marker='s', edgecolors='k', s=60)
    ax2.scatter(X2[:, 0], X2[:, 1], color='#2ecc71', marker='^', edgecolors='k', s=80)
    # Add violations
    ax2.scatter([3.5], [2.5], color='#3498db', marker='s', edgecolors='k', s=60) 
    ax2.scatter([2.5], [3.5], color='#2ecc71', marker='^', edgecolors='k', s=80)
    
    # Draw wider Street
    ax2.plot(x_val, -x_val + 5.2, 'w-', lw=2)
    ax2.plot(x_val, -x_val + 2.5, 'w--', lw=1.5, alpha=0.7)
    ax2.plot(x_val, -x_val + 7.9, 'w--', lw=1.5, alpha=0.7)
    
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.set_title("Soft Margin (Low C, Wider Street)", color='#E0E0E0')
    ax2.grid(alpha=0.2)
    
    ax2.annotate('Margin Violation', xy=(3.5, 2.5), xytext=(4, 1.5),
             arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=1, headwidth=6),
             color='#e74c3c')

    plt.suptitle("01: Large Margin Classification", fontsize=14, color='#E0E0E0', y=1.02)
    save_fig("01_large_margin")

# -----------------------------------------------------------
# Graph 02: Adding Polynomial Features (1D to 2D)
# -----------------------------------------------------------
def generate_02():
    X1D = np.linspace(-4, 4, 9)
    y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 1D plot
    ax1.axhline(0, color='w', lw=1)
    ax1.scatter(X1D[y==0], np.zeros_like(X1D[y==0]), color='#3498db', marker='s', s=80, zorder=5)
    ax1.scatter(X1D[y==1], np.zeros_like(X1D[y==1]), color='#2ecc71', marker='^', s=100, zorder=5)
    ax1.set_yticks([])
    ax1.set_xlabel(r"$x_1$", fontsize=12)
    ax1.set_title("1D Dataset (Not Linearly Separable)", color='#E0E0E0')
    
    # 2D plot (adding X^2)
    X2D = X1D**2
    ax2.scatter(X1D[y==0], X2D[y==0], color='#3498db', marker='s', s=80, zorder=5)
    ax2.scatter(X1D[y==1], X2D[y==1], color='#2ecc71', marker='^', s=100, zorder=5)
    
    ax2.plot([-4.5, 4.5], [6, 6], 'r--', lw=2, label="Linear Decision Boundary")
    
    ax2.set_xlabel(r"$x_1$", fontsize=12)
    ax2.set_ylabel(r"$x_2 = (x_1)^2$", fontsize=12)
    ax2.set_title("2D Dataset (Linearly Separable)", color='#E0E0E0')
    ax2.legend()
    ax2.grid(alpha=0.2)
    
    plt.suptitle("02: Adding Polynomial Features", fontsize=14, color='#E0E0E0', y=1.05)
    save_fig("02_adding_features")

# -----------------------------------------------------------
# Graph 03: Gaussian RBF Kernel
# -----------------------------------------------------------
def generate_03():
    x = np.linspace(-5, 5, 200)
    landmark = 0
    
    gamma_high = 2.0
    gamma_low = 0.2
    
    rbf_high = np.exp(-gamma_high * (x - landmark)**2)
    rbf_low = np.exp(-gamma_low * (x - landmark)**2)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(x, rbf_high, color='#e74c3c', lw=3, label=r"High $\gamma$ (2.0) - Narrow Influence")
    ax.plot(x, rbf_low, color='#3498db', lw=3, label=r"Low $\gamma$ (0.2) - Wide Influence")
    
    ax.axvline(x=landmark, color='#f1c40f', linestyle='--', lw=2, label="Landmark")
    
    ax.set_xlabel("Distance from landmark", fontsize=12)
    ax.set_ylabel("Similarity (RBF Output)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_title("03: Gaussian RBF Similarity Function", fontsize=14, color='#E0E0E0', pad=15)
    
    save_fig("03_rbf_kernel")

# -----------------------------------------------------------
# Graph 04: SVM Regression
# -----------------------------------------------------------
def generate_04():
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 0.5 * X + 2 + np.random.randn(50) * 0.8
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Large Epsilon
    ax1.scatter(X, y, color='#9b59b6', edgecolors='k', s=50)
    ax1.plot(X, 0.5 * X + 2, 'w-', lw=2, label="Regression Line")
    ax1.plot(X, 0.5 * X + 2 + 1.5, 'w--', lw=1.5, alpha=0.7, label=r"Margin (+$\epsilon$)")
    ax1.plot(X, 0.5 * X + 2 - 1.5, 'w--', lw=1.5, alpha=0.7, label=r"Margin (-$\epsilon$)")
    
    # Identify violations
    violations = np.abs(y - (0.5 * X + 2)) > 1.5
    ax1.scatter(X[violations], y[violations], color='none', edgecolors='#e74c3c', s=150, lw=2, label="Margin Violation")
    
    ax1.set_title(r"Large $\epsilon$ (1.5) - Wider Street", color='#E0E0E0')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(alpha=0.2)
    
    # Small Epsilon
    ax2.scatter(X, y, color='#9b59b6', edgecolors='k', s=50)
    ax2.plot(X, 0.5 * X + 2, 'w-', lw=2)
    ax2.plot(X, 0.5 * X + 2 + 0.5, 'w--', lw=1.5, alpha=0.7)
    ax2.plot(X, 0.5 * X + 2 - 0.5, 'w--', lw=1.5, alpha=0.7)
    
    violations_small = np.abs(y - (0.5 * X + 2)) > 0.5
    ax2.scatter(X[violations_small], y[violations_small], color='none', edgecolors='#e74c3c', s=150, lw=2)
    
    ax2.set_title(r"Small $\epsilon$ (0.5) - Narrow Street", color='#E0E0E0')
    ax2.grid(alpha=0.2)
    
    plt.suptitle("04: SVM Regression Objective", fontsize=14, color='#E0E0E0', y=1.02)
    save_fig("04_svm_regression")

# -----------------------------------------------------------
# Graph 05: Hinge Loss
# -----------------------------------------------------------
def generate_05():
    t = np.linspace(-3, 4, 100)
    hinge = np.maximum(0, 1 - t)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t, hinge, "b-", lw=3, label=r"Hinge Loss: $\max(0, 1 - t)$")
    ax.axhline(y=0, color='w', lw=1)
    ax.axvline(x=1, color='#e74c3c', linestyle='--', lw=2, label="t = 1")
    
    ax.fill_between(t, 0, hinge, where=(t<1), color='#e74c3c', alpha=0.1, label="Penalty Zone (Margin Violation)")
    
    ax.set_xlabel("t (Decision Function Output)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_title("05: The Hinge Loss Function", fontsize=14, color='#E0E0E0', pad=15)
    
    save_fig("05_hinge_loss")

if __name__ == "__main__":
    print("Generating visuals for Chapter 5...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    generate_05()
    print("All Chapter 5 visuals generated successfully.")
