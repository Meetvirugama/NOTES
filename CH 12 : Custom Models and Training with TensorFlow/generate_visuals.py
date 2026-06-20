"""
╔══════════════════════════════════════════════════════════════════╗
║   CH 12: Custom Models and Training with TF — COMPLETE Visuals  ║
║   10 custom dark-theme graphs/diagrams for all 5 modules         ║
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

def plot_01_tensorflow_api_structure():
    print("[01] TensorFlow API Structure")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("TensorFlow 2.x Python API & Execution Hierarchy", fontsize=15, fontweight="bold", color=TX)
    
    # Draw layers from top (high level) to bottom (hardware level)
    box(ax, 6.0, 7.0, 10.0, 0.7, color=B1, label="High-Level API (tf.keras, tf.estimator, tf.data, tf.feature_column)", fontsize=10, alpha=0.3)
    box(ax, 6.0, 5.8, 10.0, 0.7, color=P1, label="Custom Components (tf.losses, tf.metrics, tf.optimizers, tf.initializers)", fontsize=10, alpha=0.3)
    box(ax, 6.0, 4.6, 10.0, 0.7, color=GOLD, label="Low-Level Python API (tf.Tensor, tf.Variable, tf.GradientTape, tf.math, tf.linalg)", fontsize=10, alpha=0.3)
    box(ax, 6.0, 3.2, 10.0, 0.8, color=O1, label="TensorFlow C++ Engine (Graph Execution, Memory Allocator, AutoGraph, Kernels)", fontsize=10, alpha=0.2)
    
    # Hardware layer
    box(ax, 2.3, 1.6, 2.5, 0.7, color=G1, label="CPUs", fontsize=9.5, alpha=0.3)
    box(ax, 6.0, 1.6, 2.5, 0.7, color=G1, label="GPUs (CUDA/ROCm)", fontsize=9.5, alpha=0.3)
    box(ax, 9.7, 1.6, 2.5, 0.7, color=G1, label="TPUs / Edge Devices", fontsize=9.5, alpha=0.3)
    
    # Arrows representing call flows
    arrow(ax, 6.0, 6.6, 6.0, 6.2, color=TX)
    arrow(ax, 6.0, 5.4, 6.0, 5.0, color=TX)
    arrow(ax, 6.0, 4.2, 6.0, 3.7, color=TX)
    
    arrow(ax, 3.5, 2.7, 2.5, 2.0, color=TX)
    arrow(ax, 6.0, 2.7, 6.0, 2.0, color=TX)
    arrow(ax, 8.5, 2.7, 9.5, 2.0, color=TX)
    
    # Labels explaining execution style
    ax.text(11.2, 7.0, "Eager by Default\n(User friendly)", ha="left", va="center", fontsize=8.5, color=B1)
    ax.text(11.2, 4.6, "Mathematical primitives\n(Full control)", ha="left", va="center", fontsize=8.5, color=GOLD)
    ax.text(11.2, 3.2, "Compiled Graphs\n(C++ optimization)", ha="left", va="center", fontsize=8.5, color=O1)
    
    save("01_tensorflow_api_structure.png")


def plot_02_tensor_vs_variable():
    print("[02] Tensor vs Variable")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Data Mutability: tf.Tensor vs tf.Variable", fontsize=15, fontweight="bold", color=TX)
    
    # Left: tf.Tensor
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("tf.Tensor (Constant / Immutable)", fontsize=13, color=R1, fontweight="bold")
    box(ax, 3.0, 4.5, 4.2, 1.0, color=R1, label="Tensor Object in Memory\nt = tf.constant([[1, 2], [3, 4]])\nID: 0x10A7B", fontsize=9.5, alpha=0.3)
    
    box(ax, 1.8, 2.2, 2.2, 1.0, color=TX2, label="Modification Attempt:\nt[0,0] = 9\n❌ TypeError raised!", fontsize=8.5, alpha=0.2)
    box(ax, 4.2, 2.2, 2.2, 1.0, color=B1, label="Operations (e.g. t + 1):\nCreates NEW tensor\nID: 0x10FF8 (different)", fontsize=8.5, alpha=0.3)
    
    arrow(ax, 2.5, 3.9, 1.8, 2.8, color=R1)
    arrow(ax, 3.5, 3.9, 4.2, 2.8, color=B1)
    
    # Right: tf.Variable
    ax = axes[1]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("tf.Variable (Mutable / State Manager)", fontsize=13, color=G1, fontweight="bold")
    box(ax, 3.0, 4.5, 4.2, 1.0, color=G1, label="Variable Object in Memory\nv = tf.Variable([[1, 2], [3, 4]])\nID: 0x20C4F", fontsize=9.5, alpha=0.3)
    
    box(ax, 3.0, 2.2, 4.0, 1.4, color=G1, label="In-Place State Modifiers:\n- v.assign([[5, 6], [7, 8]])\n- v[0,0].assign(9)\n- v.assign_add([[1, 1], [1, 1]])\n✅ Memory ID remains unchanged (0x20C4F)", fontsize=8.5, alpha=0.3)
    arrow(ax, 3.0, 3.9, 3.0, 3.0, color=G1)
    
    # Explanations
    axes[0].text(3.0, 0.5, "Tensors represent static mathematical nodes.\nThey cannot change value once allocated.", ha="center", fontsize=9, color=TX2, style="italic")
    axes[1].text(3.0, 0.5, "Variables hold model parameters (weights/biases).\nTheir values are updated dynamically during backpropagation.", ha="center", fontsize=9, color=TX2, style="italic")
    
    plt.tight_layout()
    save("02_tensor_vs_variable.png")


def plot_03_custom_loss_huber():
    print("[03] Custom Loss Huber vs MSE vs MAE")
    z = np.linspace(-3, 3, 300)
    
    # Compute losses
    mse = 0.5 * z**2
    mae = np.abs(z)
    
    delta = 1.0
    huber = np.where(np.abs(z) <= delta, 0.5 * z**2, delta * (np.abs(z) - 0.5 * delta))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, mse, color=R1, lw=2.0, ls="--", label="Mean Squared Error (MSE / L2)")
    ax.plot(z, mae, color=GOLD, lw=2.0, ls="-.", label="Mean Absolute Error (MAE / L1)")
    ax.plot(z, huber, color=G1, lw=3.0, label="Huber Loss (Custom Loss, δ=1.0)")
    
    # Highlight threshold region
    ax.axvline(delta, color=TX2, ls=":", lw=1.2)
    ax.axvline(-delta, color=TX2, ls=":", lw=1.2)
    ax.axvspan(-delta, delta, color=G1, alpha=0.08, label="Quadratic Region (|e| ≤ δ)")
    
    ax.set_title("Huber Loss vs. Classic Losses (MSE & MAE)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Prediction Error (y_true - y_pred)")
    ax.set_ylabel("Loss Value")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 3.5)
    ax.grid(True)
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2)
    
    ax.text(1.5, 2.3, "Linear Region (|e| > δ)\nLess sensitive to outliers\n(Robust regression)", color=G1, fontsize=9.5, fontweight="bold")
    ax.text(0.0, 0.25, "Quadratic\nRegion", color=G1, fontsize=9.5, fontweight="bold", ha="center")
    
    save("03_custom_loss_huber.png")


def plot_04_stateful_vs_stateless_metric():
    print("[04] Stateful vs Stateless Metric")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    
    fig.suptitle("Metric Estimation: Stateless vs. Stateful (Streaming) Comparison", fontsize=15, fontweight="bold", color=TX)
    
    # Scenario details
    ax.text(7.5, 7.1, "Scenario: Tracking Binary Precision across two mini-batches\nBatch 1: 5 True Positives (TP) out of 10 positives prediction. Batch 2: 3 TP out of 3 positive predictions.",
            ha="center", fontsize=9.5, color=GOLD, fontweight="bold", bbox=dict(fc=CARD, ec=GOLD, alpha=0.8, boxstyle="round"))
    
    # Left: Stateless Metric
    box(ax, 3.5, 5.0, 3.2, 1.8, color=R1, label="Stateless Precision\n(Computes batch-by-batch,\nthen averages values)", fontsize=9, alpha=0.3)
    box(ax, 3.5, 2.3, 3.2, 1.8, color=R1, label="Batch 1: 5/10 = 50%\nBatch 2: 3/3 = 100%\nSimple Mean: (50% + 100%)/2\n= 75.0% ❌ (Inaccurate)", fontsize=9, alpha=0.15)
    arrow(ax, 3.5, 4.0, 3.5, 3.3, color=R1)
    
    # Right: Stateful Metric
    box(ax, 11.5, 5.0, 3.2, 1.8, color=G1, label="Stateful / Streaming Metric\n(Maintains internal counters\nacross all batches)", fontsize=9, alpha=0.3)
    box(ax, 11.5, 2.3, 3.2, 1.8, color=G1, label="Batch 1: TP=5, Pos=10\nBatch 2: TP=8, Pos=13\nStreaming Ratio: 8/13\n= 61.5% ✅ (Correct)", fontsize=9, alpha=0.15)
    arrow(ax, 11.5, 4.0, 11.5, 3.3, color=G1)
    
    # Variables indicator
    box(ax, 11.5, 0.4, 3.0, 0.8, color=G1, label="State variables:\ntotal_true_positives, total_positives", fontsize=8, alpha=0.1)
    
    save("04_stateful_vs_stateless_metric.png")


def plot_05_custom_layer_structure():
    print("[05] Custom Layer Lifecycle")
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("Keras Custom Layer Execution & Weights Initialization Lifecycle", fontsize=15, fontweight="bold", color=TX)
    
    # Steps
    box(ax, 2.0, 4.0, 2.2, 2.0, color=B1, label="1. __init__()\nSaves hyperparameters\n(units, activation,\nregularizers)", fontsize=9, alpha=0.3)
    box(ax, 6.5, 4.0, 2.5, 2.0, color=P1, label="2. build(input_shape)\nCalled dynamically on first call.\nAllocates weights using\nself.add_weight()\nbased on input dimensions", fontsize=9, alpha=0.3)
    box(ax, 11.5, 4.0, 2.5, 2.0, color=G1, label="3. call(inputs)\nComputes forward pass\noperations using\ntf.matmul, activations,\nreturning output tensors", fontsize=9, alpha=0.3)
    
    # Connecting Arrows
    arrow(ax, 3.2, 4.0, 5.1, 4.0, color=TX)
    arrow(ax, 7.9, 4.0, 10.1, 4.0, color=TX)
    
    # Highlights
    ax.text(6.5, 1.2, "Delayed weight creation allows layers to infer input shapes automatically (e.g. input_shape=[None, 10]).\nPrevents needing to hardcode input dimensions in subsequent models.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("05_custom_layer_structure.png")


def plot_06_residual_block_custom_model():
    print("[06] Custom Model with Residual Block")
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("Custom Model Subclass: Model containing Custom Layer (ResidualBlock)", fontsize=15, fontweight="bold", color=TX)
    
    # Drawing workflow
    box(ax, 1.5, 4.0, 1.4, 1.5, color=B1, label="Model Input\nx", fontsize=9.5)
    
    # Large outline for ResidualBlock Layer
    rect = Rectangle((3.2, 1.2), 6.5, 5.2, fill=True, facecolor="#1c2128", edgecolor=P1, lw=2.0, ls="-", zorder=1)
    ax.add_patch(rect)
    ax.text(6.45, 6.1, "Custom ResidualBlock(Layer)", color=P1, fontsize=10.5, fontweight="bold", ha="center")
    
    # Inside ResidualBlock: Main Path
    box(ax, 4.5, 4.6, 1.8, 1.0, color=P1, label="Dense Layer 1\n(ELU activation)", fontsize=8.5, alpha=0.5)
    box(ax, 7.5, 4.6, 1.8, 1.0, color=P1, label="Dense Layer 2\n(Linear)", fontsize=8.5, alpha=0.5)
    
    # Addition Node
    node(ax, 8.5, 2.5, r=0.28, color=O1, label="+")
    
    # Output of block
    box(ax, 11.5, 2.5, 1.6, 1.2, color=G1, label="Block Output\ny = ELU(Path2(y1) + x)", fontsize=8.5)
    
    # Connecting Arrows inside block
    arrow(ax, 2.3, 4.0, 3.5, 4.6, color=TX2) # input to dense 1
    arrow(ax, 5.5, 4.6, 6.5, 4.6, color=TX2) # dense 1 to dense 2
    arrow(ax, 8.5, 4.6, 8.5, 2.9, color=TX2) # dense 2 to add
    
    # Skip Connection Path
    arrow(ax, 2.3, 4.0, 3.5, 2.5, color=GOLD, lw=1.8) # input split to bypass
    ax.plot([3.5, 8.1], [2.5, 2.5], color=GOLD, lw=1.8, zorder=2) # skip path line
    arrow(ax, 8.1, 2.5, 8.2, 2.5, color=GOLD, lw=1.8) # skip to add node
    
    # Output arrow
    arrow(ax, 8.8, 2.5, 10.6, 2.5, color=TX2)
    
    ax.text(5.5, 2.1, "Skip Connection (Identity Bypass)", color=GOLD, fontsize=9, fontweight="bold", ha="center")
    
    save("06_residual_block_custom_model.png")


def plot_07_autodiff_gradient_tape():
    print("[07] Autodiff Gradient Tape")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("Autodiff Execution: Operations Recording via tf.GradientTape", fontsize=15, fontweight="bold", color=TX)
    
    # Forward Pass
    box(ax, 1.5, 6.0, 1.6, 1.0, color=B1, label="Input\nx = 3.0", fontsize=9.5)
    box(ax, 5.0, 6.0, 2.0, 1.0, color=B1, label="Node 1\ny = x^2 = 9.0", fontsize=9)
    box(ax, 8.5, 6.0, 2.0, 1.0, color=B1, label="Node 2\nz = y + 2 = 11.0", fontsize=9)
    
    arrow(ax, 2.4, 6.0, 3.9, 6.0, color=B1, lw=1.8)
    arrow(ax, 6.1, 6.0, 7.4, 6.0, color=B1, lw=1.8)
    
    ax.text(5.0, 7.2, "FORWARD PASS (Operations recorded on the tape)", color=B1, fontsize=10.5, fontweight="bold", ha="center")
    
    # Tape Object Visual
    rect = Rectangle((2.5, 3.2), 6.5, 1.4, fill=True, facecolor=CARD, edgecolor=GOLD, lw=1.5, ls="--")
    ax.add_patch(rect)
    ax.text(5.75, 4.1, "📼 tf.GradientTape Memory Buffer", color=GOLD, fontsize=9.5, fontweight="bold", ha="center")
    ax.text(5.75, 3.5, "Recorded ops:  op1 = squared(x)  |  op2 = add(y, 2)", color=TX, fontsize=8.5, ha="center")
    
    # Backward Pass
    box(ax, 8.5, 1.5, 2.0, 1.0, color=R1, label="Output gradient\ndz/dz = 1.0", fontsize=9)
    box(ax, 5.0, 1.5, 2.0, 1.0, color=R1, label="Gradient wrt y\ndz/dy = 1.0", fontsize=9)
    box(ax, 1.5, 1.5, 1.6, 1.0, color=R1, label="Gradient wrt x\ndz/dx = 6.0", fontsize=9.5)
    
    arrow(ax, 7.4, 1.5, 6.1, 1.5, color=R1, lw=1.8)
    arrow(ax, 3.9, 1.5, 2.4, 1.5, color=R1, lw=1.8)
    
    ax.text(5.0, 0.4, "BACKWARD PASS (Tape replayed in reverse order using chain rule)", color=R1, fontsize=10.5, fontweight="bold", ha="center")
    
    # Connections to tape
    arrow(ax, 5.0, 5.4, 5.0, 4.75, color=GOLD, lw=1.0, alpha=0.6)
    arrow(ax, 5.0, 3.1, 5.0, 2.6, color=GOLD, lw=1.0, alpha=0.6)
    
    save("07_autodiff_gradient_tape.png")


def plot_08_custom_training_loop_flow():
    print("[08] Custom Training Loop Flow")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("Execution Logic: Standard model.fit() vs. Custom Training Loop", fontsize=15, fontweight="bold", color=TX)
    
    # Left: model.fit()
    box(ax, 3.0, 5.8, 3.8, 1.2, color=P1, label="High-level model.fit(X, y)\nSimple, standard, abstract", fontsize=10, alpha=0.3)
    box(ax, 3.0, 2.5, 3.8, 2.8, color=P1, label="⚙️ Keras Backend Black Box:\n- Automatic epoch loop\n- Automatic batching\n- Internal callback execution\n- Hard to inject custom learning steps\n- Inflexible target distributions", fontsize=9, alpha=0.15)
    arrow(ax, 3.0, 5.1, 3.0, 4.0, color=P1)
    
    # Right: Custom Loop
    box(ax, 10.5, 5.8, 4.8, 1.2, color=G1, label="Custom training loops (Manual step Control)\nFlexible, customized, clear math execution", fontsize=10, alpha=0.3)
    box(ax, 10.5, 2.5, 4.8, 2.8, color=G1, label="🔧 Explicit Code execution:\n1. Loop through epochs: for epoch in epochs...\n2. Loop through mini-batches: for X, y in dataset...\n3. with tf.GradientTape() as tape:\n     Compute predictions & compute custom loss\n4. grads = tape.gradient(loss, trainable_variables)\n5. optimizer.apply_gradients(zip(grads, variables))\n6. Update metrics manually: metric.update_state(y, pred)", fontsize=8.5, alpha=0.15)
    arrow(ax, 10.5, 5.1, 10.5, 4.0, color=G1)
    
    save("08_custom_training_loop_flow.png")


def plot_09_autograph_tracing_pipeline():
    print("[09] AutoGraph Tracing Pipeline")
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("TF Compilation Pipeline: Python Source to Static Computation Graph", fontsize=15, fontweight="bold", color=TX)
    
    # Pipeline steps
    box(ax, 1.8, 4.0, 2.2, 2.0, color=B1, label="1. Python Code\nStandard function\ncontaining loops (for)\nand branches (if/else)", fontsize=9, alpha=0.3)
    box(ax, 5.0, 4.0, 2.4, 2.0, color=P1, label="2. AutoGraph Analysis\nParses AST, translates\npython loops to tf.while_loop()\nand branches to tf.cond()", fontsize=9, alpha=0.3)
    box(ax, 8.5, 4.0, 2.5, 2.0, color=GOLD, label="3. Symbolic Tracing\nRuns code once with symbolic\ntensor arguments. Records\nall TF operators to a node list", fontsize=9, alpha=0.3)
    box(ax, 12.3, 4.0, 2.4, 2.0, color=G1, label="4. Compiled TF Graph\nStatic, pruned graph\nexecuted fast in C++ engine\n(runs on GPU/TPU directly)", fontsize=9, alpha=0.3)
    
    # Arrows
    arrow(ax, 3.0, 4.0, 3.7, 4.0, color=TX)
    arrow(ax, 6.3, 4.0, 7.1, 4.0, color=TX)
    arrow(ax, 9.8, 4.0, 11.0, 4.0, color=TX)
    
    ax.text(7.5, 1.2, "Traced graphs are cached. If called again with the same input types and dimensions,\nTensorFlow runs the cached graph immediately, bypassing Python interpreter overhead.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("09_autograph_tracing_pipeline.png")


def plot_10_summary_dashboard():
    print("[10] Chapter 12 Summary Dashboard")
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("CH 12 Summary Dashboard: Custom Models & Training with TensorFlow", fontsize=17, fontweight="bold", color=TX, y=0.98)
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Huber Loss Curve
    ax = fig.add_subplot(gs[0, 0])
    z = np.linspace(-2.5, 2.5, 200)
    mse = 0.5 * z**2
    huber = np.where(np.abs(z) <= 1.0, 0.5 * z**2, np.abs(z) - 0.5)
    ax.plot(z, mse, color=R1, ls="--", label="MSE (Quadratic L2)")
    ax.plot(z, huber, color=G1, lw=2.5, label="Huber Loss (δ=1.0)")
    ax.axvspan(-1.0, 1.0, color=G1, alpha=0.08)
    ax.set_title("Custom Huber Loss Performance", fontsize=11, fontweight="bold", color=TX)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    ax.grid(True); ax.set_xlim(-2.5, 2.5); ax.set_ylim(0, 2.5)
    
    # 2. Benchmarking Graph execution speed
    ax = fig.add_subplot(gs[0, 1])
    modes = ["Python Eager Execution", "Compiled TF Graph (@tf.function)"]
    times = [26.4, 1.8] # millisecond fake scale
    bars = ax.barh(modes, times, color=[O1, G1], alpha=0.8, height=0.45)
    ax.set_title("Benchmark: Eager vs. Graph Execution Latency (100k loops)", fontsize=11, fontweight="bold", color=TX)
    ax.set_xlabel("Execution Time (milliseconds) - Lower is Faster")
    ax.grid(True, axis="x")
    for bar, time in zip(bars, times):
        ax.text(time + 0.5, bar.get_y() + bar.get_height()/2, f"{time:.1f} ms", va="center", fontsize=9.5, color=TX, fontweight="bold")
    ax.set_xlim(0, 32)
    
    # 3. Autodiff tape visual mini
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Tape Recording & Replaying (Autodiff)", fontsize=11, fontweight="bold", color=TX)
    box(ax, 2.0, 2.5, 2.2, 1.0, color=B1, label="Forward Pass\nRecord operations", fontsize=8.5, alpha=0.3)
    box(ax, 5.0, 2.5, 2.0, 1.2, color=GOLD, label="📼 tf.GradientTape\nSaves memory of\nall transformations", fontsize=8, alpha=0.2)
    box(ax, 8.0, 2.5, 2.2, 1.0, color=R1, label="Backward Pass\nReplay derivatives", fontsize=8.5, alpha=0.3)
    arrow(ax, 3.2, 2.5, 3.9, 2.5, color=TX)
    arrow(ax, 6.1, 2.5, 6.8, 2.5, color=TX)
    
    # 4. Custom component options comparison table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("Keras APIs: Selection Matrix for custom models", fontsize=11, fontweight="bold", color=TX, pad=10)
    
    table_content = [
        ("Feature", "Sequential API", "Functional API", "Subclassing API"),
        ("Complexity", "Simple linear stack", "DAG / Multi-input / Multi-output", "Highly dynamic / Arbitrary logic"),
        ("Validation", "Static (checked on compile)", "Static (checked on compile)", "Dynamic (checked on runtime)"),
        ("Save/Load", "Extremely easy", "Extremely easy", "Requires custom config overrides"),
        ("Use Case", "Standard MLPs", "Wide & Deep / Residual nets", "Research / Complex custom loops"),
    ]
    
    y_pos = np.linspace(4.5, 0.5, len(table_content))
    for idx, (feat, seq, func, subc) in enumerate(table_content):
        weight = "bold" if idx == 0 else "normal"
        color = GOLD if idx == 0 else TX
        ax.text(0.1, y_pos[idx], feat, fontsize=10, fontweight=weight, color=color, ha="left")
        ax.text(3.1, y_pos[idx], seq, fontsize=10, fontweight=weight, color=color, ha="left")
        ax.text(6.1, y_pos[idx], func, fontsize=10, fontweight=weight, color=color, ha="left")
        ax.text(9.6, y_pos[idx], subc, fontsize=10, fontweight=weight, color=color, ha="left")
        ax.axhline(y_pos[idx] - 0.3, color="#21262d", lw=1)
        
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)
    
    save("10_summary_dashboard.png")


def plot_11_autograph_code_translation():
    print("[11] AutoGraph Code Translation Map")
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("AutoGraph Code-to-Graph Operator Translation Mapping", fontsize=15, fontweight="bold", color=TX)
    
    # Left box: Python Input Code
    box(ax, 3.0, 4.0, 4.2, 3.2, color=R1, label="🐍 Standard Python Syntax\n\ndef f(x):\n    if x > 0:\n        return x * 2\n    else:\n        return tf.constant(0.0)\n\n    # Dynamic control flow\n    # parsed at runtime", fontsize=9.5, alpha=0.25)
    
    # Middle: AutoGraph translation gate
    box(ax, 7.5, 4.0, 2.8, 1.4, color=GOLD, label="⚡ AutoGraph Parser\nAST AST-parsing &\nSyntax Rewrite", fontsize=9, alpha=0.3)
    arrow(ax, 5.2, 4.0, 6.0, 4.0, color=TX)
    arrow(ax, 9.0, 4.0, 9.8, 4.0, color=TX)
    
    # Right box: TensorFlow Node operations
    box(ax, 12.0, 4.0, 4.2, 3.2, color=G1, label="🕸️ Compiled TensorFlow Operators\n\ndef f(x):\n    return tf.cond(\n        tf.greater(x, 0),\n        lambda: tf.multiply(x, 2),\n        lambda: tf.constant(0.0)\n    )\n\n    # Static node graph ready", fontsize=9.5, alpha=0.25)
    
    save("11_autograph_code_translation.png")


def plot_12_stop_gradient_adversarial():
    print("[12] tf.stop_gradient Flow")
    fig, ax = plt.subplots(figsize=(14, 6.0))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("Gradient Flow Control: tf.stop_gradient() Gate Operation", fontsize=15, fontweight="bold", color=TX)
    
    # Nodes
    box(ax, 1.5, 4.0, 1.6, 1.0, color=B1, label="Inputs\nX", fontsize=9)
    box(ax, 4.5, 4.0, 1.8, 1.2, color=P1, label="Base Layers\nTrainable\nWeights θ_base", fontsize=9, alpha=0.3)
    
    # Split paths
    box(ax, 8.5, 6.0, 2.0, 1.0, color=GOLD, label="Task A Head\nTrainable\nWeights θ_A", fontsize=8.5, alpha=0.3)
    box(ax, 8.5, 2.0, 2.0, 1.0, color=O1, label="Task B Head\nTrainable\nWeights θ_B", fontsize=8.5, alpha=0.3)
    
    # Losses
    box(ax, 12.5, 6.0, 1.8, 0.8, color=R1, label="Loss A\n(Supervised)", fontsize=9, alpha=0.2)
    box(ax, 12.5, 2.0, 1.8, 0.8, color=R1, label="Loss B\n(Adversarial)", fontsize=9, alpha=0.2)
    
    # Stop Gradient Gate
    box(ax, 6.8, 2.0, 1.1, 0.8, color=R1, label="STOP\nGRAD", fontsize=8, alpha=0.6)
    
    # Forward Path Arrows (Green/Blue)
    arrow(ax, 2.4, 4.0, 3.5, 4.0, color=G1, lw=1.8, alpha=0.8)
    arrow(ax, 5.5, 4.5, 7.4, 5.8, color=G1, lw=1.8, alpha=0.8) # base to head A
    arrow(ax, 5.5, 3.5, 6.2, 2.3, color=G1, lw=1.8, alpha=0.8) # base to stop grad
    arrow(ax, 7.4, 2.0, 7.4, 2.0, color=G1, lw=1.8, alpha=0.8) # stop grad to head B
    ax.plot([7.4, 7.4], [2.0, 2.0], color=G1, lw=1.8, alpha=0.8)
    arrow(ax, 9.6, 6.0, 11.5, 6.0, color=G1, lw=1.8, alpha=0.8)
    arrow(ax, 9.6, 2.0, 11.5, 2.0, color=G1, lw=1.8, alpha=0.8)
    
    # Backward Gradient Arrows (Red, showing block)
    # Loss A backprop
    arrow(ax, 11.5, 5.8, 9.6, 5.8, color=R1, lw=1.8, alpha=0.8)
    arrow(ax, 7.4, 5.8, 5.5, 4.2, color=R1, lw=1.8, alpha=0.8)
    
    # Loss B backprop
    arrow(ax, 11.5, 1.8, 9.6, 1.8, color=R1, lw=1.8, alpha=0.8)
    # Arrow hits STOP and goes no further
    arrow(ax, 7.4, 1.8, 6.9, 1.8, color=R1, lw=1.8, alpha=0.8)
    ax.text(6.8, 2.8, "❌ Gradient blocked\nhere during backprop\n(θ_base not affected by Loss B)", color=R1, fontsize=8, ha="center", fontweight="bold")
    
    ax.text(7.5, 0.4, "Forward pass executes through both heads normally.\nBackward gradients from Loss B are blocked at tf.stop_gradient(), protecting base layer weights.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("12_stop_gradient_adversarial.png")


def plot_13_eager_vs_graph_callstack():
    print("[13] Eager vs Graph Callstack")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Execution Flow Callstack: Eager Mode vs. Graph Mode", fontsize=15, fontweight="bold", color=TX)
    
    # Left: Eager Mode
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Eager Mode (Python Interpreter Loop)", fontsize=13, color=O1, fontweight="bold")
    
    box(ax, 3.0, 6.0, 4.5, 0.7, color=O1, label="1. Python Execution: result = a + b + c", fontsize=9, alpha=0.3)
    box(ax, 3.0, 4.4, 4.5, 0.7, color=O1, label="2. Python Interpreter calls tf.add(a, b)", fontsize=9, alpha=0.3)
    box(ax, 3.0, 2.8, 4.5, 0.7, color=TX2, label="3. C++ engine computes standard Add operator", fontsize=9, alpha=0.2)
    box(ax, 3.0, 1.2, 4.5, 0.7, color=O1, label="4. Return output to Python. Repeat for second Add.", fontsize=9, alpha=0.3)
    
    arrow(ax, 3.0, 5.6, 3.0, 4.8, color=O1)
    arrow(ax, 3.0, 4.0, 3.0, 3.2, color=O1)
    arrow(ax, 3.0, 2.4, 3.0, 1.6, color=O1)
    
    # Right: Graph Mode
    ax = axes[1]; ax.set_xlim(0, 6); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Graph Mode (C++ Runtime Dispatch)", fontsize=13, color=G1, fontweight="bold")
    
    box(ax, 3.0, 6.0, 4.5, 0.7, color=G1, label="1. Python triggers compiled Concrete Function once", fontsize=9, alpha=0.3)
    box(ax, 3.0, 4.4, 4.5, 0.7, color=G1, label="2. Python Interpreter yields control entirely", fontsize=9, alpha=0.3)
    box(ax, 3.0, 2.8, 4.5, 0.8, color=G1, label="3. C++ Engine executes pre-compiled fused Graph\n(optimizes additions & streams to GPU direct)", fontsize=8.5, alpha=0.3)
    box(ax, 3.0, 1.2, 4.5, 0.7, color=G1, label="4. Yield final output value back to Python context", fontsize=9, alpha=0.3)
    
    arrow(ax, 3.0, 5.6, 3.0, 4.8, color=G1)
    arrow(ax, 3.0, 4.0, 3.0, 3.3, color=G1)
    arrow(ax, 3.0, 2.3, 3.0, 1.6, color=G1)
    
    plt.tight_layout()
    save("13_eager_vs_graph_callstack.png")


def plot_14_custom_training_loop_backpropagation():
    print("[14] Custom Loop Backprop Flow")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("Autodiff & Parameter Update Pipeline in Custom Training Loops", fontsize=15, fontweight="bold", color=TX)
    
    # Components boxes
    box(ax, 1.5, 5.5, 1.8, 1.2, color=B1, label="Mini-batch Data\n(X_batch, y_batch)", fontsize=9)
    box(ax, 5.0, 5.5, 2.2, 1.2, color=B1, label="Model Forward Call\ny_pred = model(X, training=True)\n[Inside GradientTape]", fontsize=8.5, alpha=0.3)
    box(ax, 8.5, 5.5, 2.2, 1.2, color=B1, label="Compute Loss value\nloss = loss_fn(y, y_pred)\n+ model.losses", fontsize=8.5, alpha=0.3)
    
    # Backprop
    box(ax, 12.0, 3.0, 2.4, 1.2, color=R1, label="Calculate Gradients\ngrads = tape.gradient(loss,\nmodel.trainable_variables)", fontsize=8.5, alpha=0.3)
    box(ax, 7.5, 1.8, 2.6, 1.2, color=G1, label="Apply weights optimization\noptimizer.apply_gradients(\nzip(grads, trainable_variables))", fontsize=8.5, alpha=0.3)
    box(ax, 2.5, 1.8, 2.2, 1.2, color=G1, label="Update evaluation metrics\nmetric.update_state(\ny_batch, y_pred)", fontsize=8.5, alpha=0.3)
    
    # Forward arrows
    arrow(ax, 2.5, 5.5, 3.8, 5.5, color=B1, lw=1.5)
    arrow(ax, 6.2, 5.5, 7.3, 5.5, color=B1, lw=1.5)
    arrow(ax, 9.7, 5.5, 11.5, 4.0, color=B1, lw=1.5)
    
    # Backward arrows
    arrow(ax, 12.0, 2.3, 8.9, 1.8, color=R1, lw=1.5)
    arrow(ax, 6.1, 1.8, 3.7, 1.8, color=G1, lw=1.5)
    
    # Labels
    ax.text(12.2, 5.5, "Forward pass stops;\nTape constructs derivative map", color=GOLD, fontsize=8.5, ha="left", fontweight="bold")
    ax.text(6.0, 3.8, "Compute gradients of parameters", color=R1, fontsize=8.5, ha="right")
    ax.text(5.0, 1.0, "Weights updated in-place; Metrics track running performance.", color=G1, fontsize=9.5, ha="center", fontweight="bold")
    
    save("14_custom_training_loop_backpropagation.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Generating Chapter 12 Visual Assets...")
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
    print("🎉 All 14 visuals created successfully!")

