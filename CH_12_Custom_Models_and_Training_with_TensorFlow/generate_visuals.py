"""
╔══════════════════════════════════════════════════════════════════╗
║   CH 12: Custom Models and Training with TF — Visual Suite v2   ║
║   Rewritten to match updated notes with real worked examples.   ║
║   Run: python3 generate_visuals.py                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import warnings, os
warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visuals")
os.makedirs(OUT, exist_ok=True)

# ── Global dark theme ──────────────────────────────────────────────────────────
DARK = "#0d1117"; CARD = "#161b22"; CARD2 = "#1c2128"
B1   = "#58a6ff"; G1   = "#56d364"; R1   = "#f78166"
P1   = "#d2a8ff"; O1   = "#ffa657"; GOLD = "#e3b341"
TX   = "#c9d1d9"; TX2  = "#8b949e"; TEAL = "#39d353"

plt.rcParams.update({
    "figure.facecolor": DARK,  "axes.facecolor": CARD,
    "axes.edgecolor": TX2,     "axes.labelcolor": TX,
    "xtick.color": TX2,        "ytick.color": TX2,
    "text.color": TX,          "grid.color": "#21262d",
    "grid.linestyle": "--",    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "savefig.facecolor": DARK, "savefig.dpi": 150,
})

# ── Helper utilities ───────────────────────────────────────────────────────────
def save(name):
    p = os.path.join(OUT, name)
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✅  {name}")

def box(ax, x, y, w, h, color=B1, label="", fontsize=9, alpha=0.25, lw=1.8, text_color=None):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.06", fc=color, alpha=alpha,
                       ec=color, lw=lw, zorder=2)
    ax.add_patch(r)
    if label:
        tc = text_color if text_color else color
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=tc, fontweight="bold", zorder=3)

def arr(ax, x1, y1, x2, y2, color=TX2, lw=1.4, alpha=0.85):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha),
                zorder=3)

def circle(ax, x, y, r=0.28, color=B1, label="", fontsize=9):
    c = Circle((x, y), r, color=color, zorder=4, linewidth=1.5,
               ec="white", alpha=0.9)
    ax.add_patch(c)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=5)

def title_tag(ax, text, color=TX2, x=0.5, y=0.02, fontsize=8.5):
    ax.text(x, y, text, transform=ax.transAxes, ha="center",
            fontsize=fontsize, color=color, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# 01 — TensorFlow Architecture Stack
# ══════════════════════════════════════════════════════════════════════════════
def plot_01_tensorflow_api_structure():
    print("[01] TensorFlow API Structure")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("TensorFlow 2.x API Stack & Execution Hierarchy",
                 fontsize=15, fontweight="bold", color=TX)

    layers = [
        (7.0, B1,   "HIGH-LEVEL  ▸  tf.keras  |  tf.data  |  tf.estimator  |  TFX"),
        (5.7, P1,   "CUSTOM COMPONENTS  ▸  tf.losses  |  tf.metrics  |  tf.initializers  |  tf.optimizers"),
        (4.4, GOLD, "LOW-LEVEL PYTHON API  ▸  tf.Tensor  |  tf.Variable  |  tf.GradientTape  |  @tf.function"),
        (3.1, O1,   "C++ EXECUTION ENGINE  ▸  Graph Compiler  |  Memory Allocator  |  AutoGraph  |  Kernel Dispatch"),
    ]
    for y, c, lbl in layers:
        box(ax, 6.0, y, 11.0, 0.85, color=c, label=lbl, fontsize=9.5, alpha=0.28)
        arr(ax, 6.0, y - 0.45, 6.0, y - 0.75, color=c, lw=1.2)

    # Hardware row
    for x, lbl in [(2.0, "CPU"), (6.0, "GPU  (CUDA / ROCm)"), (10.0, "TPU / Mobile / Browser")]:
        box(ax, x, 1.8, 3.2, 0.7, color=G1, label=lbl, fontsize=9.5, alpha=0.3)
    for xf, xt in [(3.5, 4.3), (6.0, 6.0), (8.5, 7.7)]:
        arr(ax, xf, 2.6, xt, 2.15, color=G1)

    # Side notes
    for y, c, note in [
        (7.0, B1,   "Eager by default\n(beginner friendly)"),
        (5.7, P1,   "All custom components\nlive at this level"),
        (4.4, GOLD, "Full control over\nmath and state"),
        (3.1, O1,   "Compiled, optimised C++\n(10-100x faster)"),
    ]:
        ax.text(11.6, y, note, ha="left", va="center", fontsize=8, color=c)

    save("01_tensorflow_api_structure.png")


# ══════════════════════════════════════════════════════════════════════════════
# 02 — tf.constant vs tf.Variable with real numbers
# ══════════════════════════════════════════════════════════════════════════════
def plot_02_tensor_vs_variable():
    print("[02] Tensor vs Variable (with real numbers)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle("tf.constant (Immutable)  vs  tf.Variable (Mutable)",
                 fontsize=15, fontweight="bold", color=TX)

    # ─── LEFT: constant ───────────────────────────────────────────────────────
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("tf.constant  —  LOCKED after creation", fontsize=12,
                 color=R1, fontweight="bold")

    box(ax, 3.0, 5.5, 5.0, 1.1, color=R1,
        label="t = tf.constant([[85., 90.], [72., 68.]])\n"
              "  shape=(2,2)  dtype=float32  id=0x10A7", fontsize=9, alpha=0.3)

    box(ax, 1.5, 3.2, 2.0, 1.1, color=R1,
        label="t[0,0] = 99\n❌  TypeError!\nCannot assign\nto constant", fontsize=8.5, alpha=0.18)

    box(ax, 4.5, 3.2, 2.0, 1.1, color=B1,
        label="result = t + 5\n✅  New tensor\nid=0x10FF (new)\n[[90,95],[77,73]]", fontsize=8.5, alpha=0.25)

    arr(ax, 2.5, 4.9, 1.6, 3.8, color=R1)
    arr(ax, 3.5, 4.9, 4.4, 3.8, color=B1)

    ax.text(3.0, 1.5, "Constants are IMMUTABLE.\nOperations always produce a NEW tensor.\nOriginal values never change.",
            ha="center", fontsize=8.5, color=TX2, style="italic")

    # ─── RIGHT: Variable ─────────────────────────────────────────────────────
    ax = axes[1]; ax.set_xlim(0, 6); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("tf.Variable  —  MUTABLE (weights live here)", fontsize=12,
                 color=G1, fontweight="bold")

    box(ax, 3.0, 5.5, 5.0, 1.1, color=G1,
        label="w = tf.Variable(2.0)\n"
              "  trainable=True  id=0x20C4  tracked by GradientTape", fontsize=9, alpha=0.3)

    ops = [
        (3.0, 3.8, "w.assign(1.95)\n→  w = 1.95\n(same id: 0x20C4)"),
        (3.0, 2.4, "w.assign_sub(0.05)\n→  w = 1.90"),
        (3.0, 1.1, "w.assign_add(0.01)\n→  w = 1.91"),
    ]
    for bx, by, lbl in ops:
        box(ax, bx, by, 4.8, 0.8, color=G1, label=lbl, fontsize=8.5, alpha=0.2)

    arr(ax, 3.0, 4.9, 3.0, 4.25, color=G1)
    arr(ax, 3.0, 3.35, 3.0, 2.85, color=G1)
    arr(ax, 3.0, 1.95, 3.0, 1.55, color=G1)

    ax.text(3.0, 0.2, "Variables are MUTABLE. The same memory location is updated in-place.\n"
            "NEVER do:  w = w - 0.05  (that creates a new Tensor, not a Variable!)",
            ha="center", fontsize=8.0, color=GOLD, style="italic")

    plt.tight_layout()
    save("02_tensor_vs_variable.png")


# ══════════════════════════════════════════════════════════════════════════════
# 03 — Huber Loss with the actual 3-row worked example numbers
# ══════════════════════════════════════════════════════════════════════════════
def plot_03_custom_loss_huber():
    print("[03] Huber Loss with worked-example numbers")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Huber Loss: Why It Beats MSE on Outliers (threshold δ = 1.0)",
                 fontsize=14, fontweight="bold", color=TX)

    # ── LEFT: loss curves ─────────────────────────────────────────────────────
    z = np.linspace(-3, 3, 400)
    mse   = 0.5 * z**2
    mae   = np.abs(z)
    huber = np.where(np.abs(z) <= 1.0, 0.5 * z**2, np.abs(z) - 0.5)

    ax1.plot(z, mse,   color=R1,   lw=2.0, ls="--",  label="MSE   (0.5·e²)")
    ax1.plot(z, mae,   color=GOLD, lw=1.8, ls="-.",   label="MAE   (|e|)")
    ax1.plot(z, huber, color=G1,   lw=3.0,            label="Huber (δ=1.0)")

    ax1.axvline( 1.0, color=TX2, ls=":", lw=1.2)
    ax1.axvline(-1.0, color=TX2, ls=":", lw=1.2)
    ax1.axvspan(-1.0, 1.0, color=G1, alpha=0.08)

    ax1.text(0,   0.22, "Quadratic\n(like MSE)", ha="center", fontsize=8.5, color=G1, fontweight="bold")
    ax1.text(2.0, 1.3,  "Linear\n(like MAE)", ha="center",   fontsize=8.5, color=G1, fontweight="bold")
    ax1.text(-1.05, 2.5, "δ = 1.0", ha="right",             fontsize=8,   color=TX2)

    ax1.set_xlabel("Prediction Error  (y_true − y_pred)"); ax1.set_ylabel("Loss")
    ax1.set_xlim(-3, 3); ax1.set_ylim(0, 3.5); ax1.grid(True)
    ax1.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2)
    ax1.set_title("Loss Curves", fontsize=11, fontweight="bold")

    # ── RIGHT: hand-computed table ─────────────────────────────────────────────
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 9); ax2.axis("off")
    ax2.set_title("Step-by-Step: 3 Samples (δ=1.0)", fontsize=11, fontweight="bold")

    rows = [
        # y_true  y_pred  error  |err|  small?  huber    mse
        ("3.0",   "3.2",  "−0.2", "0.2", "YES ✓", "0.020",  "0.020"),
        ("5.0",   "4.5",  " 0.5", "0.5", "YES ✓", "0.125",  "0.125"),
        ("2.0",   "5.0",  "−3.0", "3.0", "NO  ✗", "2.500",  "4.500"),
    ]

    hdrs = ["y_true", "y_pred", "error", "|error|", "≤ δ?", "Huber", "MSE(0.5e²)"]
    col_x = [0.5, 1.8, 3.1, 4.2, 5.3, 6.5, 8.0]

    ax2.set_facecolor(DARK)

    # Header
    for cx, h in zip(col_x, hdrs):
        ax2.text(cx, 8.3, h, fontsize=9, fontweight="bold", color=GOLD, ha="left")
    ax2.axhline(7.9, color=TX2, lw=0.8)

    colors_row = [G1, G1, R1]
    for i, (row, rc) in enumerate(zip(rows, colors_row)):
        y = 7.0 - i * 1.5
        for cx, val in zip(col_x, row):
            ax2.text(cx, y, val, fontsize=9, color=TX if i < 2 else R1, ha="left")
        ax2.axhline(y - 0.6, color="#21262d", lw=0.5)

    # Totals
    ax2.axhline(2.6, color=TX2, lw=1.2)
    ax2.text(0.5,  2.2, "Mean loss →", fontsize=9, color=TX2)
    ax2.text(6.5,  2.2, "0.882", fontsize=11, fontweight="bold", color=G1)
    ax2.text(8.0,  2.2, "1.548", fontsize=11, fontweight="bold", color=R1)

    ax2.text(5.0, 1.1,
             "Huber = 0.882   (outlier row contributes 2.5)\n"
             "MSE   = 1.548   (outlier row contributes 4.5)\n"
             "→ MSE inflated by 75% due to single outlier!",
             ha="center", fontsize=8.5, color=GOLD,
             bbox=dict(fc=CARD2, ec=GOLD, alpha=0.8, boxstyle="round,pad=0.3"))

    plt.tight_layout()
    save("03_custom_loss_huber.png")


# ══════════════════════════════════════════════════════════════════════════════
# 04 — Stateful Metric: accumulation across 3 batches (from notes)
# ══════════════════════════════════════════════════════════════════════════════
def plot_04_stateful_vs_stateless_metric():
    print("[04] Stateful vs Stateless Metric with 3-batch example")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Stateless vs. Stateful Metric — Why Accumulation Matters",
                 fontsize=14, fontweight="bold", color=TX)

    # ── LEFT: Wrong (stateless average of batch means) ────────────────────────
    ax1.set_xlim(0, 6); ax1.set_ylim(0, 9); ax1.axis("off")
    ax1.set_title("❌  Stateless (function)\n   Averaging batch means", fontsize=11,
                  fontweight="bold", color=R1)

    batches_s = [
        ("Batch 1  (4 samples)", "loss sum = 2.665", "mean = 0.666"),
        ("Batch 2  (4 samples)", "loss sum = 1.060", "mean = 0.265"),
        ("Batch 3  (2 samples)", "loss sum = 1.170", "mean = 0.585"),
    ]
    colors_b = [B1, P1, GOLD]
    for i, ((lbl, s, m), c) in enumerate(zip(batches_s, colors_b)):
        y = 7.5 - i * 1.8
        box(ax1, 3.0, y, 5.2, 1.3, color=c, label=f"{lbl}\n{s}\n{m}", fontsize=8.5, alpha=0.22)
        if i < 2:
            arr(ax1, 3.0, y - 0.7, 3.0, y - 1.1, color=TX2, lw=1.0)

    box(ax1, 3.0, 1.5, 5.2, 1.1, color=R1,
        label="Wrong mean of means:\n(0.666 + 0.265 + 0.585) / 3 = 0.505  ❌\n"
              "Batch 3 (only 2 samples) gets equal weight as Batch 1!", fontsize=8.3, alpha=0.22)
    arr(ax1, 3.0, 3.8, 3.0, 2.1, color=R1)

    # ── RIGHT: Correct (stateful accumulation) ────────────────────────────────
    ax2.set_xlim(0, 6); ax2.set_ylim(0, 9); ax2.axis("off")
    ax2.set_title("✅  Stateful (class)\n   Accumulating total + count", fontsize=11,
                  fontweight="bold", color=G1)

    steps = [
        ("Batch 1 arrives", "total += 2.665  →  total = 2.665\ncount += 4       →  count = 4"),
        ("Batch 2 arrives", "total += 1.060  →  total = 3.725\ncount += 4       →  count = 8"),
        ("Batch 3 arrives", "total += 1.170  →  total = 4.895\ncount += 2       →  count = 10"),
    ]
    for i, (hdr, detail) in enumerate(steps):
        y = 7.6 - i * 1.85
        box(ax2, 3.0, y, 5.4, 1.4, color=G1,
            label=f"{'─'*8}  {hdr}  {'─'*8}\n{detail}", fontsize=8.5, alpha=0.22)
        if i < 2:
            arr(ax2, 3.0, y - 0.75, 3.0, y - 1.1, color=G1, lw=1.0)

    box(ax2, 3.0, 1.5, 5.4, 1.1, color=G1,
        label="result() = total / count = 4.895 / 10 = 0.4895  ✅\n"
              "Each sample weighted equally regardless of batch size!\n"
              "[reset_state() zeros total & count for next epoch]", fontsize=8.3, alpha=0.28)
    arr(ax2, 3.0, 3.75, 3.0, 2.1, color=G1)

    plt.tight_layout()
    save("04_stateful_vs_stateless_metric.png")


# ══════════════════════════════════════════════════════════════════════════════
# 05 — Custom Layer Lifecycle: __init__ → build → call
# ══════════════════════════════════════════════════════════════════════════════
def plot_05_custom_layer_structure():
    print("[05] Custom Layer Lifecycle")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("Custom Keras Layer Lifecycle:  __init__  →  build  →  call",
                 fontsize=14, fontweight="bold", color=TX)

    stages = [
        (2.2, B1,   "__init__(units=30)\n\n"
                    "self.units = 30\n"
                    "self.activation = relu\n\n"
                    "Called when you write:\n"
                    "MyLayer(30)\n\n"
                    "NO weights yet!\n"
                    "(input shape unknown)"),
        (7.5, P1,   "build(input_shape)\n\n"
                    "input_shape = (batch, 10)\n"
                    "  → input_shape[-1] = 10\n\n"
                    "W = add_weight([10, 30])\n"
                    "b = add_weight([30])\n\n"
                    "Called ONCE on first\n"
                    "data call. Never again."),
        (12.8, G1,  "call(inputs)\n\n"
                    "output = inputs @ W + b\n"
                    "output = relu(output)\n\n"
                    "Called EVERY forward\n"
                    "pass (each batch).\n\n"
                    "Gradients flow\n"
                    "through here."),
    ]

    for x, c, lbl in stages:
        box(ax, x, 4.5, 4.2, 5.5, color=c, label=lbl, fontsize=8.8, alpha=0.25)

    arr(ax, 4.4, 4.5, 5.2, 4.5, color=TX, lw=2.0)
    arr(ax, 9.8, 4.5, 10.6, 4.5, color=TX, lw=2.0)

    # When triggered
    ax.text(2.2,  1.3, "▶  MyLayer(30) called",     ha="center", fontsize=8.5, color=B1)
    ax.text(7.5,  1.3, "▶  First model(X) call",    ha="center", fontsize=8.5, color=P1)
    ax.text(12.8, 1.3, "▶  Every model(X_batch)",   ha="center", fontsize=8.5, color=G1)

    ax.text(7.5, 0.4,
            "WHY build() not __init__()? → Input shape is unknown until the first data flows through. "
            "build() receives input_shape automatically, so weights are created with the correct dimensions.",
            ha="center", fontsize=8.5, color=GOLD, style="italic")

    save("05_custom_layer_structure.png")


# ══════════════════════════════════════════════════════════════════════════════
# 06 — Residual Block with actual numbers
# ══════════════════════════════════════════════════════════════════════════════
def plot_06_residual_block_custom_model():
    print("[06] Residual Block with numbers")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("Residual Block  —  output = Dense(x) + x   (skip connection)",
                 fontsize=14, fontweight="bold", color=TX)

    # Input
    box(ax, 1.5, 4.0, 2.0, 1.2, color=B1,
        label="Input  x\n[1.0, 2.0, 3.0]\nshape: (3,)", fontsize=9)

    # Main path
    box(ax, 5.5, 5.5, 2.8, 1.2, color=P1,
        label="Dense (relu)\noutput = relu(x@W+b)\n= [-0.2, 0.5, 0.8]", fontsize=8.5, alpha=0.3)

    # Skip path
    ax.annotate("", xy=(9.5, 2.8), xytext=(2.6, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.0))
    ax.text(6.0, 2.3, "Skip Connection  (original input passed directly)", 
            ha="center", fontsize=8.5, color=GOLD, fontweight="bold")

    # Add node
    circle(ax, 9.5, 4.0, r=0.35, color=O1, label="+", fontsize=12)

    # Output
    box(ax, 12.2, 4.0, 2.8, 1.3, color=G1,
        label="Block Output\n[-0.2+1.0, 0.5+2.0, 0.8+3.0]\n= [0.8, 2.5, 3.8]", fontsize=8.5)

    arr(ax, 2.6, 4.0, 3.5, 4.5,  color=B1,  lw=1.8)
    arr(ax, 3.5, 5.5, 4.0, 5.5,  color=B1,  lw=1.8)
    arr(ax, 7.0, 5.5, 8.5, 4.5,  color=P1,  lw=1.8)
    arr(ax, 8.5, 4.5, 9.1, 4.1,  color=P1,  lw=1.8)
    arr(ax, 9.9, 4.0, 10.7, 4.0, color=G1,  lw=1.8)

    arr(ax, 2.6, 4.0, 2.6, 2.8, color=GOLD, lw=1.8)
    arr(ax, 9.5, 2.8, 9.5, 3.6, color=GOLD, lw=1.8)

    # Why it works
    ax.text(7.0, 0.9,
            "Why residual connections work:\n"
            "  • Gradient flows TWO paths: through Dense AND directly through the skip\n"
            "  • Skip path gradient = 1.0 (no shrinking!) → solves vanishing gradient in deep nets\n"
            "  • ResNet-50/101/152 are all built from stacked residual blocks",
            ha="center", fontsize=8.5, color=TX2,
            bbox=dict(fc=CARD2, ec=TX2, alpha=0.6, boxstyle="round,pad=0.4"))

    save("06_residual_block_custom_model.png")


# ══════════════════════════════════════════════════════════════════════════════
# 07 — GradientTape with the linear model numbers from notes
# ══════════════════════════════════════════════════════════════════════════════
def plot_07_autodiff_gradient_tape():
    print("[07] GradientTape with worked example (y = w*x + b)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("tf.GradientTape:  Automatic Differentiation (Chain Rule)",
                 fontsize=14, fontweight="bold", color=TX)

    # ── LEFT: diagram ─────────────────────────────────────────────────────────
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 9); ax1.axis("off")
    ax1.set_title("How the Tape Works", fontsize=11, fontweight="bold")

    # Tape region
    from matplotlib.patches import Rectangle as Rect
    r = Rect((0.3, 3.5), 9.4, 1.6, fill=True, facecolor=CARD2, ec=GOLD, lw=1.5, ls="--")
    ax1.add_patch(r)
    ax1.text(5.0, 4.95, "📼  tf.GradientTape  Memory Buffer", ha="center", fontsize=9,
             color=GOLD, fontweight="bold")
    ax1.text(5.0, 4.35, "Records:  op1=w*x  |  op2=+b  |  op3=(y_true−y_pred)²",
             ha="center", fontsize=8.5, color=TX)

    # Forward
    ax1.text(5.0, 8.3, "FORWARD PASS  (tape is recording)", ha="center",
             fontsize=10, color=B1, fontweight="bold")
    for x, lbl in [(1.5, "w=1.0,b=0.0\nx=2.0"), (4.5, "y_pred\n= w×x+b\n= 2.0"), (8.0, "loss\n=(7−2)²\n=25.0")]:
        box(ax1, x, 7.0, 2.2, 1.2, color=B1, label=lbl, fontsize=8.5, alpha=0.3)
    arr(ax1, 2.7, 7.0, 3.3, 7.0, color=B1, lw=1.6)
    arr(ax1, 5.7, 7.0, 6.8, 7.0, color=B1, lw=1.6)

    # Backward
    ax1.text(5.0, 2.7, "BACKWARD PASS  (tape replays in reverse)", ha="center",
             fontsize=10, color=R1, fontweight="bold")
    for x, lbl in [(8.0, "dloss/dloss\n= 1.0"), (4.5, "dloss/dw\n= −20.0"), (1.5, "dloss/db\n= −10.0")]:
        box(ax1, x, 1.8, 2.2, 1.2, color=R1, label=lbl, fontsize=8.5, alpha=0.3)
    arr(ax1, 6.8, 1.8, 5.7, 1.8, color=R1, lw=1.6)
    arr(ax1, 3.3, 1.8, 2.7, 1.8, color=R1, lw=1.6)
    arr(ax1, 5.0, 3.4, 5.0, 5.4, color=GOLD, lw=0.9, alpha=0.5)
    arr(ax1, 5.0, 3.5, 5.0, 2.5, color=GOLD, lw=0.9, alpha=0.5)

    # ── RIGHT: hand calculation table ─────────────────────────────────────────
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 9); ax2.axis("off")
    ax2.set_title("Hand Calculation → Verified by TF", fontsize=11, fontweight="bold")

    lines = [
        ("Setup:", "w = 1.0,  b = 0.0,  x = 2.0,  y_true = 7.0", TX),
        ("", "", TX2),
        ("Forward:", "", GOLD),
        ("  y_pred", "= w × x + b  =  1.0 × 2.0 + 0.0  =  2.0", TX),
        ("  loss",   "= (y_true − y_pred)²  =  (7.0 − 2.0)²  =  25.0", TX),
        ("", "", TX2),
        ("Chain Rule:", "", GOLD),
        ("  d(loss)/d(y_pred)", "= −2 × (y_true − y_pred)  =  −10.0", TX),
        ("  d(y_pred)/d(w)",    "= x  =  2.0", TX),
        ("  d(y_pred)/d(b)",    "= 1.0", TX),
        ("", "", TX2),
        ("Gradients:", "", GOLD),
        ("  grad_w", "= −10.0 × 2.0  =  −20.0  ← tape gives this", G1),
        ("  grad_b", "= −10.0 × 1.0  =  −10.0  ← tape gives this", G1),
        ("", "", TX2),
        ("Weight update (lr=0.01):", "", GOLD),
        ("  w_new", "= 1.0 − 0.01 × (−20.0)  =  1.2", B1),
        ("  b_new", "= 0.0 − 0.01 × (−10.0)  =  0.1", B1),
        ("", "", TX2),
        ("New prediction:", "1.2 × 2.0 + 0.1  =  2.5  (closer to 7.0 ✓)", G1),
    ]

    y = 8.7
    for lbl, val, c in lines:
        ax2.text(0.2, y, lbl, fontsize=8.5, color=c, fontweight="bold")
        ax2.text(3.2, y, val, fontsize=8.5, color=TX)
        y -= 0.42

    save("07_autodiff_gradient_tape.png")


# ══════════════════════════════════════════════════════════════════════════════
# 08 — model.fit() vs Custom Loop comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_08_custom_training_loop_flow():
    print("[08] model.fit() vs Custom Loop")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9); ax.axis("off")
    fig.suptitle("model.fit()  vs  Custom Training Loop — When to Use Each",
                 fontsize=14, fontweight="bold", color=TX)

    # Left: model.fit
    box(ax, 3.2, 7.8, 5.5, 1.1, color=P1,
        label="model.fit(X_train, y_train, epochs=10)", fontsize=10, alpha=0.3)
    box(ax, 3.2, 3.8, 5.5, 6.5, color=P1,
        label="⚙️  Keras handles everything automatically:\n\n"
              "✅  Epoch & batch loops\n"
              "✅  GradientTape internally\n"
              "✅  Callbacks (EarlyStopping etc.)\n"
              "✅  Progress bars / metrics\n\n"
              "❌  One optimizer only\n"
              "❌  Can't control each step\n"
              "❌  Hard to implement GANs\n"
              "❌  Hard to do curriculum learning",
        fontsize=9, alpha=0.15)
    arr(ax, 3.2, 7.2, 3.2, 7.1, color=P1)

    # Right: Custom loop
    box(ax, 11.5, 7.8, 5.5, 1.1, color=G1,
        label="Manual loop (for epoch in ... for batch in ...)", fontsize=10, alpha=0.3)
    box(ax, 11.5, 3.8, 5.5, 6.5, color=G1,
        label="🔧  You control every step:\n\n"
              "  for epoch in range(n_epochs):\n"
              "    for X_b, y_b in dataset:\n"
              "      with GradientTape() as tape:\n"
              "        y_pred = model(X_b, training=True)\n"
              "        loss = loss_fn(y_b, y_pred)\n"
              "      grads = tape.gradient(loss, vars)\n"
              "      optimizer.apply_gradients(...)\n"
              "      metric.update_state(y_b, y_pred)",
        fontsize=8.8, alpha=0.15, text_color=G1)
    arr(ax, 11.5, 7.2, 11.5, 7.1, color=G1)

    # When to use
    ax.text(3.2, 0.9, "Use when: standard task, quick prototype, classification/regression",
            ha="center", fontsize=8.5, color=P1, style="italic")
    ax.text(11.5, 0.9, "Use when: GAN, multi-optimizer, custom backprop, RL, research",
            ha="center", fontsize=8.5, color=G1, style="italic")

    # VS divider
    ax.text(7.5, 4.5, "VS", ha="center", va="center", fontsize=20,
            fontweight="bold", color=TX2, alpha=0.4)
    ax.axvline(7.5, color=TX2, lw=0.8, alpha=0.3, ls="--")

    save("08_custom_training_loop_flow.png")


# ══════════════════════════════════════════════════════════════════════════════
# 09 — AutoGraph Tracing Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def plot_09_autograph_tracing_pipeline():
    print("[09] AutoGraph Tracing Pipeline")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("@tf.function Compilation Pipeline:  Python  →  Optimised C++ Graph",
                 fontsize=14, fontweight="bold", color=TX)

    stages = [
        (1.8,  B1,   "1. Python Source\n\ndef f(x):\n  if x > 0:\n    return x * 2\n  for i in tf.range(5):\n    x += i\n  return x"),
        (5.3,  P1,   "2. AutoGraph\nParser\n\nif x > 0\n→ tf.cond(x>0,...)\n\nfor i in tf.range\n→ tf.while_loop(...)"),
        (8.8,  GOLD, "3. Symbolic\nTracing\n\nRun ONCE with\nsymbolic tensors\n(shape+dtype only,\nno real values).\nRecord all TF ops."),
        (12.2, G1,   "4. Compiled\nStatic Graph\n\nFused C++ ops\nCached by signature\n(dtype + shape)\nRuns on GPU/TPU\n10–20x faster!"),
    ]
    for x, c, lbl in stages:
        box(ax, x, 4.5, 2.8, 5.2, color=c, label=lbl, fontsize=8.8, alpha=0.25)
        if x < 12.2:
            arr(ax, x + 1.5, 4.5, x + 2.3, 4.5, color=TX, lw=1.8)

    # Bottom note on caching
    ax.text(7.5, 1.2,
            "Graph is CACHED per input signature (dtype + shape).\n"
            "Same signature → cached graph used (no re-trace).\n"
            "New dtype OR shape → new trace triggered (potential 'trace explosion' if in a loop!).",
            ha="center", fontsize=9, color=GOLD, style="italic",
            bbox=dict(fc=CARD2, ec=GOLD, alpha=0.7, boxstyle="round,pad=0.4"))

    save("09_autograph_tracing_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10 — Master Summary Dashboard (Huber + Eager/Graph benchmark + training step)
# ══════════════════════════════════════════════════════════════════════════════
def plot_10_summary_dashboard():
    print("[10] Chapter 12 Summary Dashboard")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Chapter 12 — Master Summary Dashboard:  Custom Models & Training with TensorFlow",
                 fontsize=16, fontweight="bold", color=TX, y=0.99)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Huber loss curves ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    z = np.linspace(-2.5, 2.5, 300)
    ax.plot(z, 0.5*z**2, color=R1, ls="--", label="MSE")
    ax.plot(z, np.where(np.abs(z)<=1, 0.5*z**2, np.abs(z)-0.5),
            color=G1, lw=2.5, label="Huber (δ=1)")
    ax.axvspan(-1, 1, color=G1, alpha=0.07)
    ax.set_title("Custom Huber Loss", fontsize=11, fontweight="bold")
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    ax.grid(True); ax.set_xlim(-2.5, 2.5); ax.set_ylim(0, 2.5)
    ax.set_xlabel("Error"); ax.set_ylabel("Loss")

    # ── 2. Eager vs Graph speed ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.barh(["Eager\n(Python loop)", "Graph\n(@tf.function)"],
                   [26.4, 1.8], color=[O1, G1], alpha=0.8, height=0.4)
    ax.set_title("Execution Speed\n(100k iterations)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (ms)  —  lower is better"); ax.grid(True, axis="x")
    for bar, t in zip(bars, [26.4, 1.8]):
        ax.text(t + 0.5, bar.get_y() + bar.get_height()/2,
                f"{t} ms", va="center", fontsize=9.5, color=TX, fontweight="bold")
    ax.set_xlim(0, 33)

    # ── 3. Training step loss drop ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    steps = np.arange(0, 101)
    # simulated exponential decay
    loss_curve = 25.0 * np.exp(-0.03 * steps) + 0.2 * np.random.rand(101)
    ax.plot(steps, loss_curve, color=B1, lw=2)
    ax.scatter([0], [25.0], color=R1, s=60, zorder=5, label="Step 0: loss=25.0")
    ax.scatter([100], [loss_curve[-1]], color=G1, s=60, zorder=5, label=f"Step 100: loss≈{loss_curve[-1]:.1f}")
    ax.set_title("Loss During Training\n(linear model, lr=0.01)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Steps"); ax.set_ylabel("Loss"); ax.grid(True)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)

    # ── 4. Tensor types summary ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off"); ax.set_title("Tensor Types Comparison", fontsize=11, fontweight="bold")
    rows = [
        ("Type",         "Mutable?", "Use For"),
        ("tf.constant",  "❌ No",     "Data, hyperparams"),
        ("tf.Variable",  "✅ Yes",    "Weights & state"),
        ("SparseTensor", "❌ No",     "Sparse embeddings"),
        ("RaggedTensor", "❌ No",     "Variable-length seqs"),
    ]
    ys = np.linspace(0.85, 0.1, len(rows))
    xs = [0.05, 0.4, 0.65]
    colors_r = [GOLD] + [TX]*4
    for (t, m, u), y, c in zip(rows, ys, colors_r):
        ax.text(xs[0], y, t, fontsize=9.5, color=c, fontweight="bold", transform=ax.transAxes)
        ax.text(xs[1], y, m, fontsize=9.5, color=c,                   transform=ax.transAxes)
        ax.text(xs[2], y, u, fontsize=9.5, color=c,                   transform=ax.transAxes)

    # ── 5. Custom component pattern ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off"); ax.set_title("Universal Custom Component Pattern", fontsize=11, fontweight="bold")
    code = (
        "class MyComponent(keras.XyzBase):\n\n"
        "  def __init__(self, param, **kwargs):\n"
        "    super().__init__(**kwargs)\n"
        "    self.param = param\n\n"
        "  def call(self, inputs):\n"
        "    # your computation here\n"
        "    return output\n\n"
        "  def get_config(self):   # REQUIRED for saving\n"
        "    return {**super().get_config(),\n"
        "            'param': self.param}"
    )
    ax.text(0.05, 0.95, code, transform=ax.transAxes, fontsize=8.5,
            color=G1, va="top", fontfamily="monospace",
            bbox=dict(fc=CARD2, ec=G1, alpha=0.6, boxstyle="round,pad=0.5"))

    # ── 6. Keras API Selection ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off"); ax.set_title("Keras API Selection Guide", fontsize=11, fontweight="bold")
    rows2 = [
        ("Criterion",    "Sequential",   "Functional",    "Subclassing"),
        ("Layout",       "Linear stack", "DAG / shared",  "Dynamic / loops"),
        ("Debugging",    "Easy",         "Easy",          "Hard (black box)"),
        ("Saving",       "Easy",         "Easy",          "Needs get_config()"),
        ("Use case",     "Simple MLP",   "ResNet / Wide", "GAN / Research"),
    ]
    ys2 = np.linspace(0.9, 0.05, len(rows2))
    xs2 = [0.01, 0.24, 0.52, 0.76]
    colors_h = [GOLD] + [TX]*4
    for row, y2, ch in zip(rows2, ys2, colors_h):
        for xv, val in zip(xs2, row):
            ax.text(xv, y2, val, fontsize=8.5, color=ch,
                    fontweight="bold" if ch == GOLD else "normal",
                    transform=ax.transAxes)
        ax.plot([0.01, 0.99], [y2 - 0.07, y2 - 0.07], color="#21262d", lw=0.5, transform=ax.transAxes)

    save("10_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11 — AutoGraph code side-by-side translation
# ══════════════════════════════════════════════════════════════════════════════
def plot_11_autograph_code_translation():
    print("[11] AutoGraph Code Translation")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("AutoGraph:  Python Syntax  →  TF Graph Operators",
                 fontsize=14, fontweight="bold", color=TX)

    left = (
        "🐍  Python Code (eager)\n\n"
        "def f(x):\n"
        "  if x > 0:\n"
        "    return x * 2\n"
        "  else:\n"
        "    return tf.constant(0.0)\n\n"
        "─────────────────────────\n\n"
        "for i in range(10):\n"
        "  x = x + i\n\n"
        "─────────────────────────\n\n"
        "print('debug:', x)"
    )
    right = (
        "🕸️  Compiled Graph (ops)\n\n"
        "def f(x):\n"
        "  return tf.cond(\n"
        "    tf.greater(x, 0),\n"
        "    lambda: tf.multiply(x, 2),\n"
        "    lambda: tf.constant(0.0)\n"
        "  )\n\n"
        "─────────────────────────\n\n"
        "x = tf.while_loop(\n"
        "  cond=lambda i,x: i<10,\n"
        "  body=lambda i,x: (i+1, x+i),\n"
        "  loop_vars=[0, x])[1]\n\n"
        "─────────────────────────\n\n"
        "tf.print('debug:', x)  ← runs each call\n"
        "print()               ← trace-time ONLY!"
    )

    box(ax, 3.0, 4.2, 5.4, 6.8, color=R1,   label=left,  fontsize=8.8, alpha=0.18)
    box(ax, 11.0, 4.2, 5.4, 6.8, color=G1,  label=right, fontsize=8.8, alpha=0.18)

    box(ax, 7.0, 4.2, 1.6, 2.2, color=GOLD,
        label="⚡ AutoGraph\nParser\n(AST rewrite)", fontsize=8.5, alpha=0.35)
    arr(ax, 5.7, 4.2, 6.2, 4.2, color=TX, lw=2.0)
    arr(ax, 7.8, 4.2, 8.3, 4.2, color=TX, lw=2.0)

    ax.text(7.0, 1.0,
            "Key insight: Python print() only runs at trace time.\n"
            "Use tf.print() for values you need to see every step.",
            ha="center", fontsize=9, color=GOLD)

    save("11_autograph_code_translation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12 — tf.stop_gradient with numbers from notes
# ══════════════════════════════════════════════════════════════════════════════
def plot_12_stop_gradient_adversarial():
    print("[12] tf.stop_gradient")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("tf.stop_gradient():  Blocking Gradient Flow Through Part of the Graph",
                 fontsize=14, fontweight="bold", color=TX)

    # Code example on left
    code = (
        "w1 = tf.Variable(5.0)\n"
        "w2 = tf.Variable(3.0)\n\n"
        "with tf.GradientTape() as tape:\n"
        "    y = w1 ** 2          # y = 25\n"
        "    y_frozen = tf.stop_gradient(y)\n"
        "    z = y_frozen + w2**2  # z = 25 + 9 = 34\n\n"
        "grads = tape.gradient(z, [w1, w2])\n"
        "# grads[0] = None   ← w1 path was cut!\n"
        "# grads[1] = 6.0    ← w2 still gets gradient"
    )
    box(ax, 3.0, 5.0, 5.5, 5.5, color=B1, label=code, fontsize=8.8, alpha=0.18)

    # Diagram on right
    box(ax, 8.5, 6.5, 1.8, 1.0, color=B1,  label="w1 = 5.0", fontsize=9)
    box(ax, 8.5, 4.8, 1.8, 1.0, color=B1,  label="y = w1² = 25", fontsize=9)
    box(ax, 8.5, 3.0, 2.2, 1.0, color=R1,  label="⛔  stop_gradient\n  (y_frozen = 25)", fontsize=8.5, alpha=0.5)
    box(ax, 8.5, 1.4, 1.8, 1.0, color=B1,  label="w2 = 3.0", fontsize=9)

    circle(ax, 11.0, 3.0, r=0.32, color=O1, label="+", fontsize=12)
    box(ax, 13.0, 3.0, 1.8, 1.0, color=G1, label="z = 34", fontsize=9)

    # Forward arrows (green)
    arr(ax, 8.5, 6.0, 8.5, 5.3, color=G1, lw=1.6)
    arr(ax, 8.5, 4.3, 8.5, 3.5, color=G1, lw=1.6)
    arr(ax, 9.6, 3.0, 10.7, 3.0, color=G1, lw=1.6)
    arr(ax, 8.5, 1.9, 9.5, 2.8, color=G1, lw=1.6)
    arr(ax, 11.3, 3.0, 12.1, 3.0, color=G1, lw=1.6)

    # Backward arrows (red)
    arr(ax, 10.7, 2.8, 9.6, 2.8, color=R1, lw=1.4, alpha=0.7)
    arr(ax, 9.5, 2.7, 9.5, 2.7, color=R1, lw=1.4, alpha=0.7)
    # Blocked path
    ax.plot([9.6, 8.5], [2.85, 2.85], color=R1, lw=2.0, ls="--")
    ax.text(8.5, 2.3, "⛔  blocked!\ngrad_w1 = None", ha="center",
            fontsize=8.5, color=R1, fontweight="bold")
    # w2 gradient flows
    arr(ax, 10.7, 3.2, 9.6, 3.2, color=R1, lw=1.4)
    arr(ax, 9.2, 2.8, 8.8, 2.1, color=R1, lw=1.4)
    ax.text(10.5, 1.5, "grad_w2 = 2×3 = 6.0 ✅", fontsize=8.5, color=G1, fontweight="bold")

    ax.text(7.0, 0.4,
            "Use cases: freeze encoder while training decoder  •  GAN training  •  multi-task learning with task isolation",
            ha="center", fontsize=8.5, color=GOLD, style="italic")

    save("12_stop_gradient_adversarial.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13 — Eager vs Graph callstack
# ══════════════════════════════════════════════════════════════════════════════
def plot_13_eager_vs_graph_callstack():
    print("[13] Eager vs Graph Callstack")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Execution Callstack:  Eager Mode  vs  Graph Mode",
                 fontsize=14, fontweight="bold", color=TX)

    # ── Eager ─────────────────────────────────────────────────────────────────
    ax1.set_xlim(0, 6); ax1.set_ylim(0, 8); ax1.axis("off")
    ax1.set_title("Eager Mode\n(back to Python after each op)", fontsize=11,
                  fontweight="bold", color=O1)

    eager_steps = [
        (O1,  "Python: result = a + b + c"),
        (TX2, "C++: compute tf.add(a, b) → ab"),
        (O1,  "Python: receives ab"),
        (TX2, "C++: compute tf.add(ab, c) → result"),
        (O1,  "Python: receives result"),
    ]
    ys = [6.8, 5.6, 4.4, 3.2, 2.0]
    for (c, lbl), y in zip(eager_steps, ys):
        box(ax1, 3.0, y, 5.2, 0.85, color=c, label=lbl, fontsize=9, alpha=0.25)
        if y > 2.0:
            arr(ax1, 3.0, y - 0.45, 3.0, y - 0.85, color=TX2, lw=1.0)

    ax1.text(3.0, 0.9, "4 trips between Python ↔ C++ per expression\n"
             "Python overhead on every operation → slow for complex models",
             ha="center", fontsize=8.5, color=O1, style="italic")

    # ── Graph ─────────────────────────────────────────────────────────────────
    ax2.set_xlim(0, 6); ax2.set_ylim(0, 8); ax2.axis("off")
    ax2.set_title("Graph Mode  (@tf.function)\n(ONE handoff to C++)", fontsize=11,
                  fontweight="bold", color=G1)

    graph_steps = [
        (G1,  "Python: calls compiled ConcreteFunction"),
        (G1,  "C++: runs fused graph (a+b+c in one kernel)"),
        (G1,  "C++ (optional): parallelises independent ops"),
        (G1,  "C++ (optional): dead-code elimination"),
        (G1,  "Python: receives final result"),
    ]
    for (c, lbl), y in zip(graph_steps, ys):
        box(ax2, 3.0, y, 5.2, 0.85, color=c, label=lbl, fontsize=9, alpha=0.25)
        if y > 2.0:
            arr(ax2, 3.0, y - 0.45, 3.0, y - 0.85, color=G1, lw=1.0)

    ax2.text(3.0, 0.9, "2 trips between Python ↔ C++ regardless of model size\n"
             "10–100x faster for repeated calls (training loops, inference)",
             ha="center", fontsize=8.5, color=G1, style="italic")

    plt.tight_layout()
    save("13_eager_vs_graph_callstack.png")


# ══════════════════════════════════════════════════════════════════════════════
# 14 — Custom Training Loop full pipeline with numbers
# ══════════════════════════════════════════════════════════════════════════════
def plot_14_custom_training_loop_backpropagation():
    print("[14] Custom Training Loop with numbers")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15); ax.set_ylim(0, 9); ax.axis("off")
    fig.suptitle("Custom Training Loop — One Mini-Batch Step with Real Values",
                 fontsize=14, fontweight="bold", color=TX)

    # Step boxes
    steps = [
        (1.8,  7.2, B1,  "Mini-batch\n(X_batch, y_batch)\n4 samples"),
        (5.0,  7.2, B1,  "Forward Pass\ny_pred = model(X, training=True)\n[Inside GradientTape]\ny_pred = [-0.10, -0.05, 0.85, 0.30]"),
        (9.0,  7.2, B1,  "Compute Loss\nloss = MSE(y_batch, y_pred)\n= (4.1²+2.05²+2.65²+4.7²)/4\n= 12.53"),
        (12.5, 5.2, R1,  "Compute Gradients\ngrads = tape.gradient(\n  loss, model.trainable_variables)\ngrad_w, grad_b computed\nvia chain rule"),
        (8.5,  3.2, G1,  "Apply Gradients\noptimizer.apply_gradients(\n  zip(grads, variables))\nw -= lr × grad_w\nb -= lr × grad_b"),
        (3.5,  3.2, G1,  "Update Metrics\nmetric.update_state(\n  y_batch, y_pred)\ntotal += batch_loss\ncount += 4"),
    ]

    for x, y, c, lbl in steps:
        box(ax, x, y, 3.4, 2.1, color=c, label=lbl, fontsize=8.3, alpha=0.25)

    # Arrows
    arr(ax, 3.6,  7.2, 3.2,  7.2, color=B1,  lw=1.8)
    arr(ax, 6.8,  7.2, 7.2,  7.2, color=B1,  lw=1.8)
    arr(ax, 10.8, 7.2, 11.5, 6.3, color=B1,  lw=1.8)
    arr(ax, 11.5, 4.2, 10.3, 3.7, color=R1,  lw=1.8)
    arr(ax, 6.8,  3.2, 5.3,  3.2, color=G1,  lw=1.8)

    # Loop-back arrow
    ax.annotate("", xy=(1.8, 8.3), xytext=(3.5, 8.3),
                arrowprops=dict(arrowstyle="-|>", color=P1, lw=1.5, alpha=0.6,
                                connectionstyle="arc3,rad=-0.5"))
    ax.text(2.6, 9.0, "Next batch", ha="center", fontsize=8, color=P1)

    # Labels
    ax.text(7.5, 8.6, "─── GradientTape Context ───", ha="center",
            fontsize=8.5, color=GOLD, style="italic")
    ax.text(7.5, 1.5,
            "After all batches in epoch:  metric.result() → displayed  |  metric.reset_state() → zeroed for next epoch",
            ha="center", fontsize=8.5, color=TX2)

    ax.text(12.5, 2.2,
            "If loss[0] = 12.53 → loss[100] ≈ 0.5\n"
            "(gradient descent works!)",
            ha="center", fontsize=8.5, color=G1, fontweight="bold")

    save("14_custom_training_loop_backpropagation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 15 — NEW: Retracing Diagram (trace explosion)
# ══════════════════════════════════════════════════════════════════════════════
def plot_15_retracing_diagram():
    print("[15] Retracing / Trace Explosion Diagram")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    fig.suptitle("@tf.function Caching:  Same Signature → Cached  |  New Signature → Retrace",
                 fontsize=13, fontweight="bold", color=TX)

    calls = [
        # x,     label,                           dtype_shape,    retrace, color
        (1.2, "Call 1\nfloat32, shape=(2,)",      "Trace!",       True,   R1),
        (3.5, "Call 2\nfloat32, shape=(2,)",      "Cached ✓",    False,   G1),
        (5.8, "Call 3\nfloat32, shape=(2,)",      "Cached ✓",    False,   G1),
        (8.1, "Call 4\nfloat64, shape=(2,)",      "Retrace!",     True,   R1),
        (10.4,"Call 5\nfloat32, shape=(3,)",      "Retrace!",     True,   R1),
        (12.7,"Call 6\nfloat32, shape=(3,)",      "Cached ✓",    False,   G1),
    ]

    cache_boxes = {}  # track graph boxes per signature

    for x, lbl, status, retrace, c in calls:
        box(ax, x, 6.5, 2.0, 2.4, color=c, label=f"{lbl}\n──────\n{status}", fontsize=8.5, alpha=0.28)

    # Cache row
    ax.text(7.0, 4.0, "Compiled Graph Cache", ha="center", fontsize=10,
            color=GOLD, fontweight="bold")
    cache_items = [
        (3.0, "Graph A\nfloat32, (2,)"),
        (7.0, "Graph B\nfloat64, (2,)"),
        (11.0,"Graph C\nfloat32, (3,)"),
    ]
    for cx, cl in cache_items:
        box(ax, cx, 2.8, 3.2, 1.4, color=GOLD, label=cl, fontsize=9, alpha=0.22)

    # Arrows from calls to cache
    for sx, sig, cache_x in [
        (1.2, "float32,(2)", 3.0), (3.5, "float32,(2)", 3.0), (5.8, "float32,(2)", 3.0),
        (8.1, "float64,(2)", 7.0), (10.4, "float32,(3)", 11.0), (12.7, "float32,(3)", 11.0)
    ]:
        arr(ax, sx, 5.2, cache_x, 3.6, color=TX2, lw=0.9, alpha=0.5)

    # Bad pattern example
    ax.text(7.0, 1.2,
            "⚠️  Trace Explosion:  if you pass Python scalars  f(1), f(2), f(3)...  → new trace every call!\n"
            "Fix:  wrap in tf.constant()  →  f(tf.constant(1.0)), f(tf.constant(2.0))  → same signature, cached.",
            ha="center", fontsize=8.5, color=R1,
            bbox=dict(fc=CARD2, ec=R1, alpha=0.7, boxstyle="round,pad=0.4"))

    save("15_retracing_diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
# 16 — NEW: build() vs __init__() timeline
# ══════════════════════════════════════════════════════════════════════════════
def plot_16_build_vs_init_timeline():
    print("[16] build() vs __init__() timeline")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("Custom Layer Timeline:  When does each method run?",
                 fontsize=14, fontweight="bold", color=TX)

    # Timeline line
    ax.axhline(4.0, xmin=0.04, xmax=0.96, color=TX2, lw=2.0, alpha=0.5)

    events = [
        (1.2, B1,  "__init__(30)\n\nself.units = 30\nNo weights yet\n(input size unknown)"),
        (4.5, P1,  "model(X_batch)\n[FIRST call]\n\nbuild() runs:\ninput_shape = (32,10)\nW = add_weight([10,30])\nb = add_weight([30])"),
        (8.0, G1,  "model(X_batch)\n[SECOND call]\n\nbuild() SKIPPED\n(layer.built=True)\ncall() runs forward"),
        (11.5, G1, "model(X_batch)\n[ALL future calls]\n\nbuild() SKIPPED\ncall() runs\ngradients computed"),
    ]

    for x, c, lbl in events:
        # Dot on timeline
        circle(ax, x, 4.0, r=0.22, color=c)
        # Box above or below alternately
        if x in [1.2, 8.0]:
            box(ax, x, 6.3, 2.8, 3.5, color=c, label=lbl, fontsize=8.5, alpha=0.25)
            ax.plot([x, x], [4.25, 4.6], color=c, lw=1.2, ls="--")
        else:
            box(ax, x, 1.7, 2.8, 3.5, color=c, label=lbl, fontsize=8.5, alpha=0.25)
            ax.plot([x, x], [3.75, 3.4], color=c, lw=1.2, ls="--")

    ax.text(7.0, 0.35,
            "Key takeaway: weights are created LAZILY in build() — "
            "not in __init__() — so the layer works with any input shape automatically.",
            ha="center", fontsize=9, color=GOLD, style="italic")

    save("16_build_vs_init_timeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Generating Chapter 12 Visual Assets (v2)...\n")
    plot_01_tensorflow_api_structure()
    plot_02_tensor_vs_variable()
    plot_03_custom_loss_huber()
    plot_04_stateful_vs_stateless_metric()
    plot_05_custom_layer_structure()
    plot_06_residual_block_custom_model()
    plot_07_autodiff_gradient_tape()
    plot_08_custom_training_loop_flow()
    plot_09_autograph_tracing_pipeline()
    plot_10_summary_dashboard()
    plot_11_autograph_code_translation()
    plot_12_stop_gradient_adversarial()
    plot_13_eager_vs_graph_callstack()
    plot_14_custom_training_loop_backpropagation()
    plot_15_retracing_diagram()
    plot_16_build_vs_init_timeline()
    print("\n🎉  All 16 visuals created in Visuals/ folder!")
