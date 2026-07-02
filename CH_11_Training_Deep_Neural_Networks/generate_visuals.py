"""
╔══════════════════════════════════════════════════════════════════╗
║   CH 11: Training Deep Neural Networks — COMPLETE Visuals        ║
║   15 custom dark-theme graphs for all 6 modules                  ║
║   Run: python3 generate_visuals.py                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
from matplotlib.gridspec import GridSpec
import warnings, os
warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visuals")
os.makedirs(OUT, exist_ok=True)

# ── Global dark theme ──────────────────────────────────────────────────────────
DARK   = "#0d1117";  CARD  = "#161b22";  B1 = "#58a6ff"
G1     = "#56d364";  R1   = "#f78166";  P1 = "#d2a8ff"
O1     = "#ffa657";  GOLD = "#e3b341";  TX = "#c9d1d9";  TX2 = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": CARD, "axes.edgecolor": TX2,
    "axes.labelcolor": TX, "xtick.color": TX2, "ytick.color": TX2,
    "text.color": TX, "grid.color": "#21262d", "grid.linestyle": "--",
    "grid.alpha": 0.5, "font.family": "DejaVu Sans",
    "savefig.facecolor": DARK, "savefig.dpi": 150,
})

def save(name):
    p = os.path.join(OUT, name)
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✅  {name}")

def node(ax, x, y, r=0.30, color=B1, label="", fontsize=9, alpha=0.9):
    c = Circle((x, y), r, color=color, zorder=4, linewidth=1.5, ec="white", alpha=alpha)
    ax.add_patch(c)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=5)

def arrow(ax, x1, y1, x2, y2, color=TX2, lw=1.2, alpha=0.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha),
                zorder=2)

def box(ax, x, y, w, h, color=B1, label="", fontsize=9, alpha=0.25, lw=1.8):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.08", fc=color, alpha=alpha,
                       ec=color, lw=lw, zorder=2)
    ax.add_patch(r)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight="bold", zorder=3)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def plot_01_weight_initialization_variance():
    print("[01] Weight Initialization Variance")
    # Simulate variance propagation through a deep network (10 layers)
    layers = np.arange(1, 11)
    
    # 1. Standard normal init (mean=0, std=1) with Sigmoid → variance vanishes/saturates
    np.random.seed(42)
    var_standard = [1.0]
    for _ in range(9):
        # inputs variance is scaled up because weights variance is 1.0 (unscaled)
        # in reality, variance of output of layer L = fan_in * var(W) * var(input)
        # for standard normal var(W) = 1.0, so it explodes if fan_in > 1. Here we simulate.
        var_standard.append(var_standard[-1] * 1.5)
        
    # 2. Glorot init (variance = 1/fan_avg) → stable
    var_glorot = [1.0]
    for _ in range(9):
        var_glorot.append(var_glorot[-1] * (0.95 + np.random.normal(0, 0.02)))
        
    # 3. He init (variance = 2/fan_in) for ReLU → stable
    var_he = [1.0]
    for _ in range(9):
        var_he.append(var_he[-1] * (1.0 + np.random.normal(0, 0.01)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, var_standard, color=R1, marker="o", lw=2, label="Standard Normal Init (unscaled)")
    ax.plot(layers, var_glorot, color=B1, marker="s", lw=2, label="Glorot / Xavier Init (Sigmoid)")
    ax.plot(layers, var_he, color=G1, marker="^", lw=2, label="He Init (ReLU)")
    
    ax.set_title("Layer Output Variance Propagation", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Variance of Layer Outputs")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    
    ax.annotate("Variance explodes\nwithout scaling", xy=(5, var_standard[4]), xytext=(2.5, 30.0),
                color=R1, fontweight="bold", arrowprops=dict(arrowstyle="->", color=R1))
    ax.annotate("Glorot & He keep\nvariance stable", xy=(8, var_he[7]), xytext=(5.0, 0.1),
                color=G1, fontweight="bold", arrowprops=dict(arrowstyle="->", color=G1))
    
    save("01_weight_initialization_variance.png")

def plot_02_activation_functions_comparison():
    print("[02] Activation Functions Comparison")
    z = np.linspace(-3, 3, 300)
    
    # Activations
    relu = np.maximum(0, z)
    lrelu = np.where(z > 0, z, 0.1 * z)
    elu = np.where(z > 0, z, 1.0 * (np.exp(z) - 1))
    # SELU parameters
    scale = 1.0507
    alpha = 1.6733
    selu = scale * np.where(z > 0, z, alpha * (np.exp(z) - 1))
    
    # Derivatives
    d_relu = np.where(z > 0, 1.0, 0.0)
    d_lrelu = np.where(z > 0, 1.0, 0.1)
    d_elu = np.where(z > 0, 1.0, np.exp(z))
    d_selu = scale * np.where(z > 0, 1.0, alpha * np.exp(z))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Functions Plot
    ax = axes[0]
    ax.plot(z, relu, color=B1, lw=2.5, label="ReLU")
    ax.plot(z, lrelu, color=GOLD, lw=2.5, label="Leaky ReLU (α=0.1)", ls="--")
    ax.plot(z, elu, color=O1, lw=2.5, label="ELU (α=1.0)", ls="-.")
    ax.plot(z, selu, color=G1, lw=2.5, label="SELU (Self-Normalizing)")
    ax.set_title("Nonsaturating Activation Functions", fontsize=12, fontweight="bold", color=TX)
    ax.set_xlabel("z"); ax.set_ylabel("f(z)")
    ax.axhline(0, color=TX2, lw=0.8, ls="--"); ax.axvline(0, color=TX2, lw=0.8, ls="--")
    ax.grid(True)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 3)
    
    # Derivatives Plot
    ax = axes[1]
    ax.plot(z, d_relu, color=B1, lw=2.5, label="ReLU'")
    ax.plot(z, d_lrelu, color=GOLD, lw=2.5, label="Leaky ReLU'", ls="--")
    ax.plot(z, d_elu, color=O1, lw=2.5, label="ELU'", ls="-.")
    ax.plot(z, d_selu, color=G1, lw=2.5, label="SELU'")
    ax.set_title("Gradients (Derivatives)", fontsize=12, fontweight="bold", color=TX)
    ax.set_xlabel("z"); ax.set_ylabel("f'(z)")
    ax.axhline(0, color=TX2, lw=0.8, ls="--"); ax.axvline(0, color=TX2, lw=0.8, ls="--")
    ax.grid(True)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    ax.set_xlim(-3, 3); ax.set_ylim(-0.2, 2.0)
    
    plt.tight_layout()
    save("02_activation_functions_comparison.png")

def plot_03_dying_relu():
    print("[03] Dying ReLU Demonstration")
    np.random.seed(42)
    
    # Simulate a distribution of pre-activation inputs z = W*x + b
    # Shifted to negative region to simulate "dead" state
    z_relu_inputs = np.random.normal(-1.0, 1.0, 1000)
    z_lrelu_inputs = np.random.normal(-1.0, 1.0, 1000)
    
    relu_out = np.maximum(0, z_relu_inputs)
    lrelu_out = np.where(z_lrelu_inputs > 0, z_lrelu_inputs, 0.1 * z_lrelu_inputs)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ReLU distribution
    ax = axes[0]
    ax.hist(relu_out, bins=25, color=R1, alpha=0.75, edgecolor="white", zorder=3)
    ax.axvline(0, color=TX, lw=2, ls="--")
    ax.set_title("Standard ReLU Output Distribution", fontsize=12, fontweight="bold", color=R1)
    ax.set_xlabel("Output value"); ax.set_ylabel("Count")
    ax.grid(True)
    ax.text(0.5, ax.get_ylim()[1]*0.8, "⚠️ High spike at 0\n(Dying ReLU: weights frozen)", 
            color=R1, fontweight="bold", bbox=dict(fc=CARD, ec=R1, alpha=0.8, boxstyle="round"))
            
    # LeakyReLU distribution
    ax = axes[1]
    ax.hist(lrelu_out, bins=25, color=G1, alpha=0.75, edgecolor="white", zorder=3)
    ax.axvline(0, color=TX, lw=2, ls="--")
    ax.set_title("Leaky ReLU Output Distribution (α=0.1)", fontsize=12, fontweight="bold", color=G1)
    ax.set_xlabel("Output value"); ax.set_ylabel("Count")
    ax.grid(True)
    ax.text(-0.5, ax.get_ylim()[1]*0.8, "✅ Gradients kept alive\nfor negative inputs", 
            color=G1, fontweight="bold", bbox=dict(fc=CARD, ec=G1, alpha=0.8, boxstyle="round"))
            
    plt.tight_layout()
    save("03_dying_relu.png")

def plot_04_batch_normalization_flow():
    print("[04] Batch Normalization Flow Chart")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    
    fig.suptitle("Batch Normalization (BN) Layer Execution Flow", fontsize=16, fontweight="bold", color=TX)
    
    # Blocks
    box(ax, 1.5, 4.0, 1.8, 1.8, color=B1, label="Input Batch\nX_B\n(m instances)", fontsize=9)
    box(ax, 4.5, 6.0, 2.0, 1.2, color=P1, label="Step 1: Mini-batch Mean\nμ_B = (1/m)Σ x_i", fontsize=8.5)
    box(ax, 4.5, 2.0, 2.0, 1.2, color=P1, label="Step 2: Mini-batch Var\nσ_B^2 = (1/m)Σ (x_i-μ)^2", fontsize=8.5)
    box(ax, 8.0, 4.0, 2.2, 1.5, color=GOLD, label="Step 3: Normalize\nx̂_i = (x_i - μ_B) / √(σ_B^2 + ε)", fontsize=8.5)
    box(ax, 11.5, 4.0, 2.2, 1.5, color=G1, label="Step 4: Scale & Shift\nz_i = γ ⊗ x̂_i + β\n(Output of BN)", fontsize=8.5)
    
    # Arrows
    arrow(ax, 2.5, 4.5, 3.4, 5.8, color=TX2)
    arrow(ax, 2.5, 3.5, 3.4, 2.2, color=TX2)
    arrow(ax, 5.6, 6.0, 6.8, 4.5, color=TX2)
    arrow(ax, 5.6, 2.0, 6.8, 3.5, color=TX2)
    arrow(ax, 9.2, 4.0, 10.3, 4.0, color=TX2)
    arrow(ax, 12.7, 4.0, 13.8, 4.0, color=TX2)
    
    # Test Time EMA info
    ax.text(8.0, 1.0, "Test Time Mode: μ and σ are replaced by running Exponential Moving Averages (EMA)\nestimated during training:  v ← v × momentum + v_batch × (1 - momentum)",
            ha="center", fontsize=9.5, color=O1, fontweight="bold",
            bbox=dict(fc=CARD, ec=O1, alpha=0.8, boxstyle="round"))
            
    save("04_batch_normalization_flow.png")

def plot_05_gradient_clipping():
    print("[05] Gradient Clipping")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, 6); ax.set_ylim(-1, 6)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Gradient Clipping: Value vs Norm", fontsize=14, fontweight="bold", pad=12)
    
    # Draw original large gradient vector
    g_orig = np.array([4.5, 5.0])
    
    # Clip by Value threshold = 2.0
    # Every component clipped individually
    g_val = np.array([np.clip(g_orig[0], -2.0, 2.0), np.clip(g_orig[1], -2.0, 2.0)])
    
    # Clip by Norm threshold = 2.0
    # Rescale vector to length 2.0 if norm > 2.0
    norm = np.linalg.norm(g_orig)
    g_norm = g_orig * (2.0 / norm)
    
    # Plotting vector arrows
    # Origin at 0,0
    ax.quiver(0, 0, g_orig[0], g_orig[1], angles='xy', scale_units='xy', scale=1, color=R1, label="Original Gradient (norm ≈ 6.7)", width=0.012)
    ax.quiver(0, 0, g_val[0], g_val[1], angles='xy', scale_units='xy', scale=1, color=O1, label="Clipped by Value (threshold = 2.0)", width=0.012)
    ax.quiver(0, 0, g_norm[0], g_norm[1], angles='xy', scale_units='xy', scale=1, color=G1, label="Clipped by Norm (threshold = 2.0)", width=0.012)
    
    # Draw constraint box and circle
    # Value box: x in [-2, 2], y in [-2, 2]
    rect = Rectangle((-2, -2), 4.0, 4.0, fill=False, edgecolor=O1, ls="--", lw=1.5, label="Clip Value Boundary")
    ax.add_patch(rect)
    
    # Norm circle: radius = 2.0
    circle = Circle((0,0), 2.0, fill=False, edgecolor=G1, ls=":", lw=1.8, label="Clip Norm Boundary")
    ax.add_patch(circle)
    
    ax.set_xlabel("g_x (Weight 1 Gradient component)")
    ax.set_ylabel("g_y (Weight 2 Gradient component)")
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2, loc="lower right")
    
    ax.text(g_orig[0]+0.1, g_orig[1], "Original\n[4.5, 5.0]", color=R1, fontsize=9, fontweight="bold")
    ax.text(g_val[0]-0.8, g_val[1]+0.2, "Value Clipped\n[2.0, 2.0]\n(Direction changes!)", color=O1, fontsize=8.5, fontweight="bold")
    ax.text(g_norm[0]+0.15, g_norm[1]-0.4, "Norm Clipped\n[1.34, 1.49]\n(Direction preserved)", color=G1, fontsize=8.5, fontweight="bold")
    
    save("05_gradient_clipping.png")

def plot_06_transfer_learning_stages():
    print("[06] Transfer Learning Stages")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Transfer Learning: Freezing & Fine-Tuning Stages", fontsize=16, fontweight="bold", color=TX)
    
    # Phase 1: Frozen Lower Layers
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Phase 1: Warm-up New Output Layer\n(Lower layers FROZEN to protect weights)", fontsize=12, fontweight="bold", color=B1)
    
    # Reused layers (frozen = gray)
    box(ax, 2.0, 4.0, 1.8, 4.0, color="#484f58", label="Frozen Lower Layer 1\n(Low-level edges/shapes)\n[trainable=False]", fontsize=8, alpha=0.5)
    box(ax, 5.0, 4.0, 1.8, 4.0, color="#484f58", label="Frozen Lower Layer 2\n(Mid-level parts)\n[trainable=False]", fontsize=8, alpha=0.5)
    # New layer (active = orange)
    box(ax, 8.0, 4.0, 1.8, 4.0, color=O1, label="New Output Layer\n(Random initialization)\n[trainable=True]", fontsize=8, alpha=0.8)
    
    arrow(ax, 3.0, 4.0, 4.0, 4.0, color=TX2)
    arrow(ax, 6.0, 4.0, 7.0, 4.0, color=TX2)
    
    ax.text(5.0, 1.0, "Gradients propagate back, but only update the Output Layer.\nWarm-up prevents random output gradients from wrecking pretrained weights.",
            ha="center", fontsize=8.5, color=GOLD, style="italic")
            
    # Phase 2: Unfrozen Top Layers
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Phase 2: Joint Fine-Tuning\n(Unfreeze layers + use VERY low learning rate)", fontsize=12, fontweight="bold", color=G1)
    
    box(ax, 2.0, 4.0, 1.8, 4.0, color="#484f58", label="Frozen Lower Layer 1\n(Keep generic features)\n[trainable=False]", fontsize=8, alpha=0.5)
    box(ax, 5.0, 4.0, 1.8, 4.0, color=G1, label="Unfrozen Layer 2\n(Adapt features to new task)\n[trainable=True]", fontsize=8, alpha=0.8)
    box(ax, 8.0, 4.0, 1.8, 4.0, color=O1, label="Fine-Tuning Output\n(Adapting output weights)\n[trainable=True]", fontsize=8, alpha=0.8)
    
    arrow(ax, 3.0, 4.0, 4.0, 4.0, color=TX2)
    arrow(ax, 6.0, 4.0, 7.0, 4.0, color=TX2)
    
    ax.text(5.0, 1.0, "Fine-tune with small learning rate (e.g., η = 1e-4 instead of 1e-2)\nto make gentle adjustments and avoid 'catastrophic forgetting'.",
            ha="center", fontsize=8.5, color=GOLD, style="italic")
            
    plt.tight_layout()
    save("06_transfer_learning_stages.png")

def plot_07_unsupervised_pretraining():
    print("[07] Unsupervised Pretraining Flow")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    fig.suptitle("Unsupervised Pretraining & Fine-Tuning Pipeline", fontsize=16, fontweight="bold", color=TX)
    
    # 1. Unsupervised phase
    box(ax, 2.5, 5.5, 2.8, 1.8, color=B1, label="1. Gather Unlabeled Data\n(Cheap & abundant)", fontsize=9)
    box(ax, 7.0, 5.5, 3.0, 1.8, color=P1, label="2. Train Unsupervised Model\n(e.g., Autoencoder or GAN)\nLearns raw data representations", fontsize=8.5)
    
    # 2. Supervised phase
    box(ax, 7.0, 2.0, 3.0, 1.8, color=GOLD, label="3. Transfer Lower Layers\n(Keep representation weights)\nFreeze them initially", fontsize=8.5)
    box(ax, 12.0, 2.0, 3.0, 1.8, color=G1, label="4. Fine-Tune on Labeled Data\n(Train classifier with few instances)\nUnfreeze & fine-tune top layers", fontsize=8.5)
    
    # Connecting Arrows
    arrow(ax, 4.0, 5.5, 5.4, 5.5, color=TX2)
    arrow(ax, 7.0, 4.5, 7.0, 3.0, color=O1, lw=2)
    arrow(ax, 8.6, 2.0, 10.4, 2.0, color=TX2)
    
    ax.text(7.0, 3.8, "Transfer weight layers", ha="center", fontsize=9, color=O1, fontweight="bold")
    ax.text(7.0, 0.5, "Geoffrey Hinton's 2006 Deep Learning breakthrough recipe.\nExtremely powerful when labeled data is scarce but unlabeled data is plentiful.",
            ha="center", fontsize=10, color=TX2, style="italic")
            
    save("07_unsupervised_pretraining.png")

def plot_08_momentum_vs_sgd():
    print("[08] Momentum vs SGD in Valley")
    # Draw contour of a narrow valley (elongated bowl)
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = 0.1 * X**2 + 2.0 * Y**2  # elongated valley: x is gentle, y is very steep
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Optimizer Trajectories in a Steep Elongated Valley", fontsize=16, fontweight="bold", color=TX)
    
    titles = ["1. Standard SGD (Zig-Zag)", "2. Momentum Optimization", "3. Nesterov Accelerated Gradient (NAG)"]
    colors = [R1, B1, G1]
    
    # Trajectories
    # Standard SGD: bounces back and forth in y, progresses slowly in x
    np.random.seed(42)
    sgd_x, sgd_y = [-4.0], [1.8]
    eta = 0.4
    for _ in range(15):
        # gradient of Z: dZ/dx = 0.2*x, dZ/dy = 4.0*y
        gx = 0.2 * sgd_x[-1]
        gy = 4.0 * sgd_y[-1]
        sgd_x.append(sgd_x[-1] - eta * gx)
        sgd_y.append(sgd_y[-1] - eta * gy)
        
    # Momentum: accumulates velocity in x, dampens in y
    mom_x, mom_y = [-4.0], [1.8]
    vx, vy = 0.0, 0.0
    beta = 0.8
    for _ in range(15):
        gx = 0.2 * mom_x[-1]
        gy = 4.0 * mom_y[-1]
        vx = beta * vx - eta * gx
        vy = beta * vy - eta * gy
        mom_x.append(mom_x[-1] + vx)
        mom_y.append(mom_y[-1] + vy)
        
    # NAG: looks ahead, dampens oscillations even more
    nag_x, nag_y = [-4.0], [1.8]
    vx, vy = 0.0, 0.0
    for _ in range(15):
        # Look ahead position
        lax = nag_x[-1] + beta * vx
        lay = nag_y[-1] + beta * vy
        gx = 0.2 * lax
        gy = 4.0 * lay
        vx = beta * vx - eta * gx
        vy = beta * vy - eta * gy
        nag_x.append(nag_x[-1] + vx)
        nag_y.append(nag_y[-1] + vy)
        
    trajectories = [
        (sgd_x, sgd_y),
        (mom_x, mom_y),
        (nag_x, nag_y)
    ]
    
    for ax, title, color, (tx, ty) in zip(axes, titles, colors, trajectories):
        ax.contour(X, Y, Z, levels=15, cmap="Blues", alpha=0.6)
        ax.plot(tx, ty, color=color, lw=2.5, marker="o", ms=4, label="Optimizer Path")
        ax.scatter([0], [0], color=GOLD, marker="*", s=150, zorder=5, label="Optimum")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("x (gentle slope)"); ax.set_ylabel("y (steep slope)")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
        ax.set_xlim(-5, 5); ax.set_ylim(-2, 2)
        
    plt.tight_layout()
    save("08_momentum_vs_sgd.png")

def plot_09_adaptive_optimizers():
    print("[09] Adaptive Optimizers Comparison")
    # Draw contour of a valley
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = 0.2 * X**2 + 2.0 * Y**2
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Adaptive Learning Rate Optimizer Trajectories", fontsize=16, fontweight="bold", color=TX)
    
    # 1. AdaGrad: scales down learning rates, gets stuck early
    ada_x, ada_y = [-4.0], [1.8]
    sx, sy = 1e-8, 1e-8
    eta = 0.5
    for _ in range(25):
        gx = 0.4 * ada_x[-1]
        gy = 4.0 * ada_y[-1]
        sx += gx**2
        sy += gy**2
        ada_x.append(ada_x[-1] - (eta / np.sqrt(sx)) * gx)
        ada_y.append(ada_y[-1] - (eta / np.sqrt(sy)) * gy)
        
    # 2. RMSProp: uses exponential moving average, avoids getting stuck
    rms_x, rms_y = [-4.0], [1.8]
    sx, sy = 0.0, 0.0
    beta = 0.9
    for _ in range(25):
        gx = 0.4 * rms_x[-1]
        gy = 4.0 * rms_y[-1]
        sx = beta * sx + (1 - beta) * gx**2
        sy = beta * sy + (1 - beta) * gy**2
        rms_x.append(rms_x[-1] - (eta / (np.sqrt(sx) + 1e-8)) * gx)
        rms_y.append(rms_y[-1] - (eta / (np.sqrt(sy) + 1e-8)) * gy)
        
    # 3. Adam: combines RMSProp + momentum
    adam_x, adam_y = [-4.0], [1.8]
    mx, my = 0.0, 0.0
    sx, sy = 0.0, 0.0
    beta1 = 0.9
    beta2 = 0.99
    eta = 0.3
    for t in range(1, 26):
        gx = 0.4 * adam_x[-1]
        gy = 4.0 * adam_y[-1]
        
        # update moments
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy
        sx = beta2 * sx + (1 - beta2) * gx**2
        sy = beta2 * sy + (1 - beta2) * gy**2
        
        # bias correction
        mhat_x = mx / (1 - beta1**t)
        mhat_y = my / (1 - beta1**t)
        shat_x = sx / (1 - beta2**t)
        shat_y = sy / (1 - beta2**t)
        
        adam_x.append(adam_x[-1] - (eta / (np.sqrt(shat_x) + 1e-8)) * mhat_x)
        adam_y.append(adam_y[-1] - (eta / (np.sqrt(shat_y) + 1e-8)) * mhat_y)

    titles = ["1. AdaGrad (Stops early)", "2. RMSProp (Robust)", "3. Adam (Adaptive + Momentum)"]
    colors = [O1, P1, G1]
    trajectories = [
        (ada_x, ada_y),
        (rms_x, rms_y),
        (adam_x, adam_y)
    ]
    
    for ax, title, color, (tx, ty) in zip(axes, titles, colors, trajectories):
        ax.contour(X, Y, Z, levels=15, cmap="Blues", alpha=0.6)
        ax.plot(tx, ty, color=color, lw=2.5, marker="o", ms=4, label="Optimizer Path")
        ax.scatter([0], [0], color=GOLD, marker="*", s=150, zorder=5, label="Optimum")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
        ax.set_xlim(-5, 5); ax.set_ylim(-2, 2)
        
    plt.tight_layout()
    save("09_adaptive_optimizers.png")

def plot_10_learning_rate_effects():
    print("[10] Learning Rate Effects")
    np.random.seed(15)
    epochs = np.arange(1, 41)
    
    # Losses for various scenarios
    too_high = [1.5]
    for _ in range(39):
        too_high.append(too_high[-1] * 1.08 + np.random.normal(0, 0.05))
    too_high = np.clip(too_high, 0, 8)
        
    too_low = 1.6 * np.exp(-0.02 * epochs) + np.random.normal(0, 0.01, 40)
    optimal = 1.6 * np.exp(-0.15 * epochs) + 0.1 + np.random.normal(0, 0.02, 40)
    
    # Decay starts fast then stabilizes at very low loss
    decay = 1.6 * np.exp(-0.08 * epochs)
    decay = np.where(epochs < 15, decay, decay[13] * np.exp(-0.2 * (epochs-14))) + 0.05 + np.random.normal(0, 0.015, 40)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, too_high, color=R1, lw=2.5, label="Too High η (Diverges / Explodes)")
    ax.plot(epochs, too_low, color=GOLD, lw=2.5, label="Too Low η (Extremely slow progress)", ls="--")
    ax.plot(epochs, optimal, color=B1, lw=2.5, label="Optimal Constant η", ls="-.")
    ax.plot(epochs, decay, color=G1, lw=2.5, label="Learning Rate Decay (Best Convergence)")
    
    ax.set_title("Training Loss for Various Learning Rates (η)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training Loss")
    ax.grid(True)
    ax.set_ylim(0, 3.5)
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2)
    
    save("10_learning_rate_effects.png")

def plot_11_lr_schedules():
    print("[11] Learning Rate Schedules")
    steps = np.arange(0, 100)
    
    # 1. Power scheduling: lr = lr0 / (1 + step/s)**c
    lr_power = 0.1 / (1 + steps / 20.0)**1.0
    
    # 2. Exponential scheduling: lr = lr0 * 0.1**(step/s)
    lr_exp = 0.1 * 0.1**(steps / 40.0)
    
    # 3. Piecewise constant
    lr_piecewise = np.where(steps < 30, 0.1, np.where(steps < 70, 0.01, 0.001))
    
    # 4. 1cycle scheduling (Leslie Smith)
    # Ramp up linearly for 45 steps, ramp down linearly to original for 45 steps, drop down to near-zero for last 10 steps
    lr_1cycle = []
    for s in steps:
        if s < 45:
            lr_1cycle.append(0.01 + (0.1 - 0.01) * (s / 45.0))
        elif s < 90:
            lr_1cycle.append(0.1 - (0.1 - 0.01) * ((s - 45.0) / 45.0))
        else:
            lr_1cycle.append(0.01 - (0.01 - 0.0001) * ((s - 90.0) / 10.0))
            
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Learning Rate Scheduling Strategies", fontsize=16, fontweight="bold", color=TX)
    
    configs = [
        ("Power Scheduling\nη(t) = η₀ / (1 + t/s)ᶜ", lr_power, B1),
        ("Exponential Scheduling\nη(t) = η₀ · 0.1^(t/s)", lr_exp, P1),
        ("Piecewise Constant Scheduling", lr_piecewise, GOLD),
        ("1cycle Scheduling (Leslie Smith)", lr_1cycle, G1)
    ]
    
    for ax, (title, lr_curve, color) in zip(axes.flat, configs):
        ax.plot(steps, lr_curve, color=color, lw=3)
        ax.fill_between(steps, 0, lr_curve, alpha=0.1, color=color)
        ax.set_title(title, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Training Progress (%)")
        ax.set_ylabel("Learning Rate (η)")
        ax.grid(True)
        ax.set_ylim(0, 0.12)
        
    plt.tight_layout()
    save("11_lr_schedules.png")

def plot_12_dropout_mechanism():
    print("[12] Dropout Mechanism")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Dropout Regularization Mechanism", fontsize=16, fontweight="bold", color=TX)
    
    # Normal Layer
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Standard Fully-Connected Layer\n(Active during testing)", fontsize=12, fontweight="bold", color=B1)
    
    in_nodes = [(1.5, y) for y in np.linspace(1.5, 6.5, 4)]
    out_nodes = [(4.5, y) for y in np.linspace(2.0, 6.0, 3)]
    
    # Draw connections
    for i_x, i_y in in_nodes:
        for o_x, o_y in out_nodes:
            ax.plot([i_x, o_x], [i_y, o_y], color=TX2, lw=1.2, alpha=0.6, zorder=1)
            
    # Draw nodes
    for i_x, i_y in in_nodes:
        node(ax, i_x, i_y, r=0.28, color=B1)
    for o_x, o_y in out_nodes:
        node(ax, o_x, o_y, r=0.28, color=P1)
        
    ax.text(3.0, 0.5, "All connections active.\nInput weights are scaled: w_test = w * (1 - p)",
            ha="center", fontsize=9.5, color=TX2, style="italic")
            
    # Dropout Active Layer
    ax = axes[1]; ax.set_xlim(0, 6); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Layer with Dropout (rate p = 50%)\n(Active during training step)", fontsize=12, fontweight="bold", color=R1)
    
    # Active nodes
    in_active = [True, False, True, False]
    out_active = [True, True, False]
    
    # Draw connections between active nodes
    for idx_i, (i_x, i_y) in enumerate(in_nodes):
        for idx_o, (o_x, o_y) in enumerate(out_nodes):
            if in_active[idx_i] and out_active[idx_o]:
                ax.plot([i_x, o_x], [i_y, o_y], color=TX2, lw=1.5, alpha=0.8, zorder=1)
            else:
                ax.plot([i_x, o_x], [i_y, o_y], color="#484f58", lw=0.8, alpha=0.25, ls="--", zorder=1)
                
    # Draw nodes with visual states
    for idx_i, (i_x, i_y) in enumerate(in_nodes):
        c = B1 if in_active[idx_i] else "#484f58"
        lbl = "Active" if in_active[idx_i] else "Dropped"
        node(ax, i_x, i_y, r=0.28, color=c, alpha=0.9 if in_active[idx_i] else 0.4)
        ax.text(i_x-0.4, i_y, lbl, ha="right", va="center", fontsize=7.5, color=c)
        
    for idx_o, (o_x, o_y) in enumerate(out_nodes):
        c = P1 if out_active[idx_o] else "#484f58"
        lbl = "Active" if out_active[idx_o] else "Dropped"
        node(ax, o_x, o_y, r=0.28, color=c, alpha=0.9 if out_active[idx_o] else 0.4)
        ax.text(o_x+0.4, o_y, lbl, ha="left", va="center", fontsize=7.5, color=c)
        
    ax.text(3.0, 0.5, "Random sub-network trained in each step.\nForces robust representations & prevents co-adaptation.",
            ha="center", fontsize=9.5, color=TX2, style="italic")
            
    plt.tight_layout()
    save("12_dropout_mechanism.png")

def plot_13_mc_dropout_uncertainty():
    print("[13] MC Dropout Uncertainty")
    # Simulate a case where a classifier predicts footwear classes
    # Compare single prediction vs 100 MC predictions
    classes = ["Sandal", "Shirt", "Sneaker", "Ankle Boot", "Bag"]
    
    # 1. Single prediction (dropout OFF): overly confident about ankle boot (index 3)
    single_probs = [0.01, 0.00, 0.01, 0.98, 0.00]
    
    # 2. MC Dropout (100 runs, dropout ON): reveals uncertainty between shoe types
    np.random.seed(101)
    mc_runs = []
    for _ in range(100):
        # random variations favoring Sandal (0), Sneaker (2), Ankle Boot (3)
        raw = np.array([0.15, 0.0, 0.20, 0.65, 0.0]) + np.random.dirichlet([5, 1, 6, 15, 1]) * 0.3
        mc_runs.append(raw / raw.sum())
    
    mc_runs = np.array(mc_runs)
    mc_mean = mc_runs.mean(axis=0)
    mc_std = mc_runs.std(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Prediction Probability: Standard vs MC Dropout", fontsize=16, fontweight="bold", color=TX)
    
    # Single predictions
    ax = axes[0]
    ax.bar(classes, single_probs, color=R1, alpha=0.8, edgecolor="white", width=0.5)
    ax.set_title("Standard Prediction (Dropout OFF)\nOverconfident & Blind to uncertainty", fontsize=12, fontweight="bold", color=R1)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y")
    for i, p in enumerate(single_probs):
        if p > 0.01:
            ax.text(i, p + 0.02, f"{p:.2%}", ha="center", fontsize=10, fontweight="bold")
            
    # MC Dropout predictions
    ax = axes[1]
    bars = ax.bar(classes, mc_mean, color=G1, alpha=0.8, yerr=mc_std, error_kw=dict(ecolor=TX, lw=1.5, capsize=5), edgecolor="white", width=0.5)
    ax.set_title("Monte Carlo Dropout Prediction (100 runs)\nCaptures model uncertainty", fontsize=12, fontweight="bold", color=G1)
    ax.set_ylabel("Average Probability (with std)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y")
    for i, p in enumerate(mc_mean):
        if p > 0.01:
            ax.text(i, p + mc_std[i] + 0.02, f"{p:.1%}", ha="center", fontsize=9.5, fontweight="bold")
            
    plt.tight_layout()
    save("13_mc_dropout_uncertainty.png")

def plot_14_max_norm_constraint():
    print("[14] Max-Norm Regularization Constraint")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Max-Norm Constraint Projection (radius r = 1.5)", fontsize=13, fontweight="bold", pad=12)
    
    # Constraint boundary
    boundary = Circle((0,0), 1.5, fill=False, edgecolor=G1, ls="--", lw=2, label="Constraint Hypersphere (‖w‖₂ ≤ 1.5)")
    ax.add_patch(boundary)
    
    # Healthy weight inside boundary
    w_healthy = np.array([0.8, 0.7])
    # Weight that gets updated outside boundary
    w_excess = np.array([1.8, 1.5])
    # Projected weight back onto boundary
    norm_excess = np.linalg.norm(w_excess)
    w_projected = w_excess * (1.5 / norm_excess)
    
    # Plot vectors
    ax.quiver(0, 0, w_healthy[0], w_healthy[1], angles='xy', scale_units='xy', scale=1, color=B1, width=0.01)
    ax.quiver(0, 0, w_excess[0], w_excess[1], angles='xy', scale_units='xy', scale=1, color=R1, width=0.01)
    ax.quiver(0, 0, w_projected[0], w_projected[1], angles='xy', scale_units='xy', scale=1, color=G1, width=0.01)
    
    # Labels
    ax.text(w_healthy[0]+0.1, w_healthy[1], f"w_1 (Healthy)\n‖w‖ = {np.linalg.norm(w_healthy):.2f}", color=B1, fontsize=9, fontweight="bold")
    ax.text(w_excess[0]+0.1, w_excess[1], f"w_2 (Violated)\n‖w‖ = {norm_excess:.2f}", color=R1, fontsize=9, fontweight="bold")
    ax.text(w_projected[0]-0.8, w_projected[1]-0.4, f"w_projected\n‖w‖ = 1.50\n(Rescaled)", color=G1, fontsize=9, fontweight="bold")
    
    # Projection line
    ax.plot([w_excess[0], w_projected[0]], [w_excess[1], w_projected[1]], color=TX2, ls=":", lw=1.5)
    
    ax.set_xlabel("Weight Dimension 1")
    ax.set_ylabel("Weight Dimension 2")
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2, loc="upper left")
    
    save("14_max_norm_constraint.png")

def plot_15_summary_dashboard():
    print("[15] Chapter 11 Summary Dashboard")
    # A composite dashboard showcasing key parts of Chapter 11
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("CH 11 Summary Dashboard: Training Deep Neural Networks", fontsize=18, fontweight="bold", color=TX, y=0.98)
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Activation functions
    ax = fig.add_subplot(gs[0, 0])
    z = np.linspace(-3, 3, 200)
    ax.plot(z, np.maximum(0, z), color=B1, lw=2, label="ReLU")
    ax.plot(z, np.where(z > 0, z, 0.1 * z), color=GOLD, lw=2, label="LeakyReLU", ls="--")
    ax.plot(z, np.where(z > 0, z, 1.0 * (np.exp(z) - 1)), color=O1, lw=2, label="ELU", ls="-.")
    ax.plot(z, 1.0507 * np.where(z > 0, z, 1.6733 * (np.exp(z) - 1)), color=G1, lw=2, label="SELU")
    ax.set_title("Nonsaturating Activation Functions", fontsize=11, fontweight="bold", color=TX)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    ax.grid(True); ax.set_xlim(-3, 3); ax.set_ylim(-1.5, 2.5)
    
    # 2. Learning rate schedules
    ax = fig.add_subplot(gs[0, 1])
    steps = np.arange(0, 100)
    lr_exp = 0.1 * 0.1**(steps / 40.0)
    lr_1cycle = [0.01 + (0.1 - 0.01)*(s/45.0) if s < 45 else 0.1 - (0.1 - 0.01)*((s-45.0)/45.0) if s < 90 else 0.01 - (0.01 - 0.0001)*((s-90.0)/10.0) for s in steps]
    ax.plot(steps, lr_exp, color=P1, lw=2.5, label="Exponential Decay")
    ax.plot(steps, lr_1cycle, color=G1, lw=2.5, label="1cycle Schedule")
    ax.set_title("Learning Rate Schedules Comparison", fontsize=11, fontweight="bold", color=TX)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    ax.grid(True); ax.set_ylim(0, 0.12)
    
    # 3. Optimizers performance rank
    ax = fig.add_subplot(gs[1, 0])
    opts = ["SGD", "SGD+Mom", "SGD+NAG", "AdaGrad", "RMSProp", "Adam", "Nadam"]
    scores = [1, 2, 2.5, 1.2, 2.8, 3.0, 3.0] # speed rank
    bars = ax.barh(opts, scores, color=[R1, O1, GOLD, R1, B1, G1, G1], alpha=0.8, height=0.55)
    ax.set_title("Optimizer Convergence Speed Ranking", fontsize=11, fontweight="bold", color=TX)
    ax.set_xlabel("Relative Speed Rank (Higher is Faster)")
    ax.grid(True, axis="x")
    for bar, score in zip(bars, scores):
        ax.text(score + 0.05, bar.get_y() + bar.get_height()/2, f"{score}", va="center", fontsize=9, color=TX, fontweight="bold")
    ax.set_xlim(0, 3.5)
    
    # 4. Default configurations table (ASCII-like drawing in Matplotlib)
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("Best Practice Default DNN Configuration Recipes", fontsize=12, fontweight="bold", color=TX, pad=10)
    
    table_content = [
        ("Hyperparameter", "Standard Deep Net", "Self-Normalizing Net"),
        ("Kernel Initializer", "He initialization", "LeCun initialization"),
        ("Activation", "ELU or LeakyReLU", "SELU"),
        ("Normalization", "Batch Norm (if deep)", "None (Self-Normalizing)"),
        ("Regularization", "Early stopping (+L2/Dropout)", "Alpha Dropout (if needed)"),
        ("Optimizer", "Adam (or Nadam / SGD+NAG)", "Adam (or Nadam / SGD+NAG)"),
        ("LR Schedule", "1cycle (or exponential decay)", "1cycle (or exponential decay)"),
    ]
    
    y_pos = np.linspace(7, 1, len(table_content))
    for idx, (hp, std, self_norm) in enumerate(table_content):
        weight = "bold" if idx == 0 else "normal"
        color = GOLD if idx == 0 else TX
        ax.text(0.5, y_pos[idx], hp, fontsize=10.5, fontweight=weight, color=color, ha="left")
        ax.text(4.5, y_pos[idx], std, fontsize=10.5, fontweight=weight, color=color, ha="left")
        ax.text(8.5, y_pos[idx], self_norm, fontsize=10.5, fontweight=weight, color=color, ha="left")
        ax.axhline(y_pos[idx] - 0.25, color="#21262d", lw=1)
        
    ax.set_xlim(0, 13); ax.set_ylim(0, 8)
    
    save("15_summary_dashboard.png")


def plot_16_normalization_comparison():
    print("[16] Normalization Comparison")
    # Draw a 2x2 grid representing Batch Norm, Layer Norm, Instance Norm, and Group Norm
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Deep Learning Normalization Methods: Normalization Domains", fontsize=15, fontweight="bold", color=TX)
    
    titles = ["Batch Normalization (BN)", "Layer Normalization (LN)", "Instance Normalization (IN)", "Group Normalization (GN)"]
    
    for idx, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_xlim(-1, 6); ax.set_ylim(-1, 6); ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=TX)
        
        # Draw a 3D-like grid representing N (Batch), C (Channels), and [H, W] (Spatial)
        # N: vertical layers (let's draw 3 planes)
        # C: horizontal channels (let's draw 3 columns)
        # For simplicity, represent as boxes
        for n in range(3): # Batch index
            for c in range(3): # Channel index
                # Coordinate base offsets
                x_base = c * 1.8 + n * 0.4
                y_base = (2 - n) * 1.8 + c * 0.2
                
                # Determine color based on normalization domain
                is_shaded = False
                if idx == 0: # Batch Norm: normalizes across N (batch items) for each channel c
                    # Shade if it is channel 1 (c=1) across all batch items
                    if c == 1: is_shaded = True
                elif idx == 1: # Layer Norm: normalizes across C (channels) for each batch item n
                    # Shade all channels for a single batch item (n=1)
                    if n == 1: is_shaded = True
                elif idx == 2: # Instance Norm: normalizes spatial dimensions for one channel and one item
                    # Shade only c=1 and n=1
                    if n == 1 and c == 1: is_shaded = True
                elif idx == 3: # Group Norm: normalizes a group of channels (e.g. c=0 and c=1) for one item
                    # Shade group (c=0 and c=1) for item n=1
                    if n == 1 and c < 2: is_shaded = True
                
                col = G1 if is_shaded else CARD
                alpha = 0.6 if is_shaded else 0.25
                ec = G1 if is_shaded else TX2
                lw = 2.0 if is_shaded else 0.8
                
                # Draw square face representing H x W spatial slice
                rect = Rectangle((x_base, y_base), 1.2, 1.2, facecolor=col, edgecolor=ec, alpha=alpha, lw=lw, zorder=5-n)
                ax.add_patch(rect)
                
                # Add text label for slices
                if n == 0 and c == 0:
                    ax.text(x_base - 0.2, y_base + 0.6, "N (Batch)", ha="right", va="center", fontsize=8.5, color=TX2, rotation=90)
                if n == 2 and c == 0:
                    ax.text(x_base + 0.6, y_base - 0.3, "C (Channels)", ha="center", va="top", fontsize=8.5, color=TX2)
                    
        # Add labels detailing statistics scope
        if idx == 0:
            ax.text(2.5, -0.8, "Statistics calculated across the batch\ndimension (N) for each channel independently.", ha="center", fontsize=9, color=GOLD)
        elif idx == 1:
            ax.text(2.5, -0.8, "Statistics calculated across all channels (C)\nfor each individual batch instance independently.", ha="center", fontsize=9, color=GOLD)
        elif idx == 2:
            ax.text(2.5, -0.8, "Statistics calculated across spatial dims (H, W)\nper instance and per channel independently.", ha="center", fontsize=9, color=GOLD)
        elif idx == 3:
            ax.text(2.5, -0.8, "Statistics calculated across spatial dims and a\nsubset (group) of channels per instance.", ha="center", fontsize=9, color=GOLD)
            
    plt.tight_layout()
    save("16_normalization_comparison.png")


def plot_17_vanishing_gradients_sigmoid():
    print("[17] Vanishing Gradients")
    epochs = np.arange(1, 101)
    
    # Simulate gradient norm decay over epochs for deep sigmoid network layers
    np.random.seed(42)
    grad_layer5 = 0.5 * np.exp(-0.01 * epochs) + np.random.normal(0, 0.01, 100)
    grad_layer3 = 0.1 * np.exp(-0.05 * epochs) + np.random.normal(0, 0.005, 100)
    grad_layer1 = 0.01 * np.exp(-0.15 * epochs) + np.random.normal(0, 0.001, 100)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, grad_layer5, color=G1, lw=2.5, label="Layer 5 (Output adjacent - Stable gradients)")
    ax.plot(epochs, grad_layer3, color=B1, lw=2.5, label="Layer 3 (Middle layer - Diminishing gradients)", ls="--")
    ax.plot(epochs, grad_layer1, color=R1, lw=2.5, label="Layer 1 (Input adjacent - Vanished gradients)", ls="-.")
    
    ax.set_title("Gradient Norm Over Training (5-Layer Sigmoid DNN)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Gradient Norm (‖∇L‖)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2)
    
    ax.annotate("Vanished: Near-zero gradient\nmeans weights do not learn", xy=(40, grad_layer1[39]), xytext=(15, 1e-4),
                color=R1, fontweight="bold", arrowprops=dict(arrowstyle="->", color=R1))
    ax.annotate("Stable gradient values", xy=(60, grad_layer5[59]), xytext=(65, 0.1),
                color=G1, fontweight="bold", arrowprops=dict(arrowstyle="->", color=G1))
    
    save("17_vanishing_gradients_sigmoid.png")


def plot_18_learning_rate_warmup():
    print("[18] Learning Rate Warm-Up Schedule")
    steps = np.arange(1, 101)
    
    # Warmup for first 10 steps, Cosine Decay for remaining 90 steps
    lr_max = 0.01
    lr_min = 0.0001
    lrs = []
    for s in steps:
        if s <= 10:
            # Linear ramp up
            lrs.append(lr_max * (s / 10.0))
        else:
            # Cosine decay
            progress = (s - 10.0) / 90.0
            decayed = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * progress))
            lrs.append(decayed)
            
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(steps, lrs, color=B1, lw=3)
    ax.fill_between(steps, 0, lrs, alpha=0.1, color=B1)
    
    ax.set_title("Cosine Decay Learning Rate Schedule with Linear Warm-Up", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Learning Rate (η)")
    ax.grid(True)
    
    # Annotations
    ax.axvline(10, color=GOLD, ls="--", lw=1.2)
    ax.text(5, lr_max * 0.5, "Linear\nWarm-up\n(First 10%)", color=GOLD, fontsize=9.5, fontweight="bold", ha="center")
    ax.text(55, lr_max * 0.6, "Cosine Decay Phase\n(Stabilizes updates as error drops)", color=B1, fontsize=9.5, fontweight="bold", ha="center")
    
    save("18_learning_rate_warmup.png")


def plot_19_optimizer_landscape_saddle():
    print("[19] Optimizer Landscape Saddle Point")
    # Plot saddle point landscape z = x^2 - y^2
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2 # saddle point contours
    
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contour(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f', colors=TX2)
    
    # Plot hypothetical trajectories escaping the saddle point at (0,0)
    # Start at (-1.8, 0.05) slightly offset from y-axis
    
    # 1. SGD gets stuck on the flat ridge along x-axis
    sgd_x = np.linspace(-1.8, 0.0, 15)
    sgd_y = np.full(15, 0.05)
    
    # 2. Momentum accumulates velocity, escapes along y-axis
    mom_x = [-1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.2, 0.3, 0.35, 0.35]
    mom_y = [0.05, 0.05, 0.05, 0.04, 0.03, 0.01, -0.1, -0.4, -0.8, -1.3, -1.9]
    
    # 3. Adam scales updates, escapes immediately
    adam_x = [-1.8, -1.4, -1.0, -0.6, -0.2, 0.0, 0.1, 0.1, 0.1]
    adam_y = [0.05, 0.05, 0.06, 0.08, 0.15, 0.4, 0.9, 1.4, 1.9]
    
    ax.plot(sgd_x, sgd_y, color=R1, marker="o", ls="-", lw=2, label="SGD (Trapped / Stalled at zero slope)")
    ax.plot(mom_x, mom_y, color=GOLD, marker="s", ls="--", lw=2, label="Momentum (Escapes via accumulated velocity)")
    ax.plot(adam_x, adam_y, color=G1, marker="^", ls="-.", lw=2.5, label="Adam (Escapes rapidly via adaptive learning rates)")
    
    ax.scatter([0], [0], color="white", marker="x", s=150, zorder=10, label="Saddle Point (0, 0)")
    
    ax.set_title("Optimizer Trajectories Escaping a Saddle Point", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Dimension 1 (Loss decreases then increases)")
    ax.set_ylabel("Dimension 2 (Loss decreases strictly)")
    ax.grid(True)
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2, loc="upper right")
    
    save("19_optimizer_landscape_saddle.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Generating Chapter 11 Visual Assets...")
    plot_01_weight_initialization_variance()
    plot_02_activation_functions_comparison()
    plot_03_dying_relu()
    plot_04_batch_normalization_flow()
    plot_05_gradient_clipping()
    plot_06_transfer_learning_stages()
    plot_07_unsupervised_pretraining()
    plot_08_momentum_vs_sgd()
    plot_09_adaptive_optimizers()
    plot_10_learning_rate_effects()
    plot_11_lr_schedules()
    plot_12_dropout_mechanism()
    plot_13_mc_dropout_uncertainty()
    plot_14_max_norm_constraint()
    plot_15_summary_dashboard()
    plot_16_normalization_comparison()
    plot_17_vanishing_gradients_sigmoid()
    plot_18_learning_rate_warmup()
    plot_19_optimizer_landscape_saddle()
    print("🎉 All Chapter 11 Visual Assets Generated Successfully!")

