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
# Graph 01: Gradient Descent Paths in Parameter Space
# -----------------------------------------------------------
def generate_01():
    # Simulate the paths
    t = np.linspace(0, 10, 100)
    
    # Batch GD: Smooth path to minimum
    batch_x = 4.5 * np.exp(-0.3 * t) * np.cos(0.5 * t)
    batch_y = 2.5 * np.exp(-0.3 * t) * np.sin(0.5 * t)
    
    # Stochastic GD: Erratically bouncing towards minimum and around it
    np.random.seed(42)
    stoch_x = 4.5 * np.exp(-0.15 * t) * np.cos(0.5 * t) + np.random.randn(100) * 0.4
    stoch_y = 2.5 * np.exp(-0.15 * t) * np.sin(0.5 * t) + np.random.randn(100) * 0.4
    
    # Mini-batch GD: Mildly bouncing
    mini_x = 4.5 * np.exp(-0.2 * t) * np.cos(0.5 * t) + np.random.randn(100) * 0.15
    mini_y = 2.5 * np.exp(-0.2 * t) * np.sin(0.5 * t) + np.random.randn(100) * 0.15

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw contour lines to represent the cost function bowl
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 2*Y**2
    ax.contour(X, Y, Z, levels=15, colors='#333', linewidths=1)
    
    # Plot paths
    ax.plot(stoch_x, stoch_y, 'o-', color='#e74c3c', markersize=3, lw=1, alpha=0.7, label='Stochastic GD')
    ax.plot(mini_x, mini_y, 's-', color='#2ecc71', markersize=3, lw=1, alpha=0.9, label='Mini-batch GD')
    ax.plot(batch_x, batch_y, 'D-', color='#3498db', markersize=4, lw=2, label='Batch GD')
    
    # Mark the minimum
    ax.scatter(0, 0, color='#f1c40f', s=100, zorder=5, marker='*', label='Global Minimum')

    ax.set_xlabel(r"$\theta_0$", fontsize=12)
    ax.set_ylabel(r"$\theta_1$", fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title("01: Gradient Descent Paths in Parameter Space", fontsize=14, color='#E0E0E0', pad=15)
    
    save_fig("01_gd_paths")

# -----------------------------------------------------------
# Graph 02: Underfitting vs Overfitting (Learning Curves)
# -----------------------------------------------------------
def generate_02():
    # Simulate learning curves
    m = np.arange(1, 100)
    
    # Underfitting
    train_under = 1.8 - 1.8 * np.exp(-0.1 * m)
    val_under = 2.2 + 10 * np.exp(-0.15 * m)
    
    # Overfitting
    train_over = 0.5 - 0.4 * np.exp(-0.05 * m)
    val_over = 1.5 + 8 * np.exp(-0.08 * m)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Underfitting
    ax1.plot(m, train_under, "r-+", lw=2, label="Train Error (RMSE)")
    ax1.plot(m, val_under, "b-", lw=2, label="Validation Error (RMSE)")
    ax1.set_ylim(0, 3.5)
    ax1.set_xlabel("Training Set Size", fontsize=11)
    ax1.set_ylabel("RMSE", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)
    ax1.set_title("Underfitting (e.g. Linear Model)", color='#E0E0E0', fontsize=12)
    ax1.text(40, 2.7, "High Plateau\nClose Together", color='#f1c40f', fontsize=10, 
             bbox=dict(facecolor='#222', edgecolor='#f1c40f', boxstyle='round,pad=0.5'))

    # Right: Overfitting
    ax2.plot(m, train_over, "r-+", lw=2, label="Train Error (RMSE)")
    ax2.plot(m, val_over, "b-", lw=2, label="Validation Error (RMSE)")
    ax2.set_ylim(0, 3.5)
    ax2.set_xlabel("Training Set Size", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.2)
    ax2.set_title("Overfitting (e.g. 10th-degree Polynomial)", color='#E0E0E0', fontsize=12)
    
    # Highlight the gap
    ax2.annotate('', xy=(80, 1.5), xytext=(80, 0.5),
                 arrowprops=dict(arrowstyle='<->', color='#2ecc71', lw=2))
    ax2.text(82, 1.0, "Large Gap\n(High Variance)", color='#2ecc71', fontsize=10)

    plt.suptitle("02: Learning Curves — Underfitting vs Overfitting", fontsize=14, color='#E0E0E0', y=1.02)
    save_fig("02_learning_curves")

# -----------------------------------------------------------
# Graph 03: Early Stopping
# -----------------------------------------------------------
def generate_03():
    epochs = np.arange(0, 500)
    
    # Simulate errors
    train_error = 2.0 * np.exp(-epochs/100) + 0.5
    val_error = 2.0 * np.exp(-epochs/70) + 1.2 + 0.0001 * (epochs - 200)**2
    
    best_epoch = np.argmin(val_error)
    min_val_error = val_error[best_epoch]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(epochs, val_error, 'b-', lw=2.5, label="Validation Error")
    ax.plot(epochs, train_error, 'r--', lw=2, label="Training Error")
    
    ax.axvline(x=best_epoch, color='#f1c40f', linestyle=':', lw=2)
    ax.axhline(y=min_val_error, color='#f1c40f', linestyle=':', lw=2)
    
    ax.scatter([best_epoch], [min_val_error], color='#f1c40f', s=100, zorder=5)
    
    ax.annotate('Best Model\n(Stop Here)', xy=(best_epoch, min_val_error), xytext=(best_epoch+50, min_val_error+0.5),
                 arrowprops=dict(facecolor='#f1c40f', shrink=0.05, width=1, headwidth=8),
                 fontsize=11, color='#f1c40f', ha='left')
    
    ax.text(350, 3.0, "Overfitting Phase", color='#e74c3c', fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_ylim(0, 4)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_title("03: Early Stopping Regularization", fontsize=14, color='#E0E0E0', pad=15)
    
    save_fig("03_early_stopping")

# -----------------------------------------------------------
# Graph 04: The Sigmoid Function
# -----------------------------------------------------------
def generate_04():
    t = np.linspace(-10, 10, 100)
    sig = 1 / (1 + np.exp(-t))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t, sig, "b-", lw=3, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    
    ax.axhline(y=0, color='k', lw=1)
    ax.axhline(y=0.5, color='#e74c3c', linestyle=':', lw=2, label="Decision Boundary (0.5)")
    ax.axhline(y=1, color='k', lw=1)
    ax.axvline(x=0, color='k', lw=1)
    
    ax.fill_between(t, 0, sig, where=(t<0), color='#e74c3c', alpha=0.1, label="Predict Class 0")
    ax.fill_between(t, 0, sig, where=(t>=0), color='#2ecc71', alpha=0.1, label="Predict Class 1")
    
    ax.set_xlabel("t (Logit / Score)", fontsize=12)
    ax.set_ylabel(r"$\sigma(t)$ (Probability)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.2)
    ax.set_title("04: The Sigmoid (Logistic) Function", fontsize=14, color='#E0E0E0', pad=15)
    
    save_fig("04_sigmoid_function")

if __name__ == "__main__":
    print("Generating visuals for Chapter 4...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    print("All Chapter 4 visuals generated successfully.")
