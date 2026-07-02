"""
 ╔══════════════════════════════════════════════════════════════════╗
 ║   CH 14: Deep Computer Vision — COMPLETE Visuals (36 Plots)      ║
 ║   Programmatic dark-theme diagrams for all textbook figures      ║
 ║   Run: python3 generate_visuals.py                              ║
 ╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, ConnectionPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

DOWNLOADED = {
    "01_visual_cortex_pipeline.png",
    "03_biological_inspiration.png",
    "12_lenet5_architecture.png",
    "13_alexnet_architecture.png",
    "15_inception_block.png",
    "16_multiscale_extraction.png",
    "17_vgg16_architecture.png",
    "19_resnet_block.png",
    "32_encoder_decoder_flow.png"
}

def save(name):
    if name in DOWNLOADED:
        plt.close()
        print(f"  ⏭️  Skipping {name} (keeping standard diagram from link)")
        return
    p = os.path.join(OUT, name)
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✅  {name}")

def node(ax, x, y, r=0.25, color=B1, label="", fontsize=8, alpha=0.9):
    c = Circle((x, y), r, color=color, zorder=4, linewidth=1.2, ec="white", alpha=alpha)
    ax.add_patch(c)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=5)

def arrow(ax, x1, y1, x2, y2, color=TX2, lw=1.2, alpha=0.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha),
                zorder=2)

def box(ax, x, y, w, h, color=B1, label="", fontsize=8.5, alpha=0.25, lw=1.5):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.04", fc=color, alpha=alpha,
                       ec=color, lw=lw, zorder=2)
    ax.add_patch(r)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=3)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATORS (1 to 36)
# ══════════════════════════════════════════════════════════════════════════════

def plot_01_visual_cortex_pipeline():
    print("[01] Visual Cortex Pipeline")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    fig.suptitle("Figure 1: Human Eye to Visual Cortex Pipeline", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.5, 2.0, 1.8, 1.2, color=B1, label="Human Eye\n(Retina Stimulus)")
    box(ax, 4.0, 2.0, 1.8, 1.2, color=G1, label="Optic Nerve\n& LGN Relays")
    box(ax, 6.5, 2.0, 1.8, 1.2, color=P1, label="Primary Visual\nCortex (V1)")
    box(ax, 8.8, 2.0, 1.6, 1.2, color=O1, label="Higher Visual\nAreas (V2/V4)")
    
    arrow(ax, 2.5, 2.0, 3.0, 2.0, color=TX)
    arrow(ax, 5.0, 2.0, 5.5, 2.0, color=TX)
    arrow(ax, 7.5, 2.0, 7.9, 2.0, color=TX)
    
    save("01_visual_cortex_pipeline.png")

def plot_02_shape_hierarchy():
    print("[02] Shape Hierarchy")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 2: Edge and Shape Detection Hierarchy", fontsize=12, fontweight="bold", color=TX)
    
    # Layer 1: Edges
    box(ax, 1.5, 2.5, 2.0, 3.0, color=B1, label="V1 Layer\n\nSimple Cells:\nVertical, Horizontal,\nand Diagonal Edges", alpha=0.15)
    # Layer 2: Shapes
    box(ax, 5.0, 2.5, 2.2, 3.0, color=G1, label="V2 & V3 Layers\n\nComplex Cells:\nCurves, Corners,\nand Texture Patterns", alpha=0.15)
    # Layer 3: Objects
    box(ax, 8.5, 2.5, 2.0, 3.0, color=P1, label="V4 & IT Layers\n\nHypercomplex Cells:\nComplex Objects,\nFaces, and Context", alpha=0.15)
    
    arrow(ax, 2.6, 2.5, 3.8, 2.5, color=TX)
    arrow(ax, 6.2, 2.5, 7.4, 2.5, color=TX)
    
    save("02_shape_hierarchy.png")

def plot_03_biological_inspiration():
    print("[03] Biological Inspiration")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 3: Biological Inspiration Behind CNNs", fontsize=12, fontweight="bold", color=TX)
    
    # Overlapping RFs
    c1 = Circle((2.0, 2.5), 0.8, fill=False, edgecolor=B1, ls="--", lw=2)
    c2 = Circle((2.8, 2.1), 0.9, fill=False, edgecolor=G1, ls="--", lw=2)
    ax.add_patch(c1); ax.add_patch(c2)
    ax.text(2.0, 2.5, "Receptive\nField 1", color=B1, fontsize=8, ha="center", va="center")
    ax.text(2.8, 2.1, "Receptive\nField 2", color=G1, fontsize=8, ha="center", va="center")
    
    box(ax, 7.5, 3.5, 2.0, 1.0, color=B1, label="Simple Neuron 1\n(Orientation)")
    box(ax, 7.5, 1.5, 2.0, 1.0, color=G1, label="Simple Neuron 2\n(Orientation)")
    
    arrow(ax, 2.0, 2.5, 6.4, 3.5, color=B1)
    arrow(ax, 2.8, 2.1, 6.4, 1.5, color=G1)
    
    save("03_biological_inspiration.png")

def plot_04_sliding_kernel():
    print("[04] Sliding Kernel Operation")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 9); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 4: Sliding Kernel Operation (Feature Extraction)", fontsize=12, fontweight="bold", color=TX)
    
    # Draw a 5x5 grid
    for r in range(5):
        for c in range(5):
            rect = Rectangle((c*0.6 + 1.0, r*0.6 + 1.0), 0.5, 0.5, fill=True, facecolor=CARD, edgecolor=TX2)
            ax.add_patch(rect)
            
    # Highlight kernel position (3x3 at top-left)
    k_rect = Rectangle((1.0, 2.2), 1.7, 1.7, fill=True, facecolor=B1, alpha=0.3, edgecolor=B1, lw=2)
    ax.add_patch(k_rect)
    ax.text(1.85, 3.0, "3x3 Kernel", color=B1, fontweight="bold", ha="center", fontsize=8)
    
    # Output grid (3x3)
    for r in range(3):
        for c in range(3):
            rect = Rectangle((c*0.6 + 5.5, r*0.6 + 1.6), 0.5, 0.5, fill=True, facecolor=CARD, edgecolor=G1)
            ax.add_patch(rect)
    
    # Target pixel
    node(ax, 5.75, 2.85, r=0.2, color=G1)
    arrow(ax, 2.8, 3.0, 5.5, 2.85, color=O1, lw=1.5, alpha=0.8)
    
    save("04_sliding_kernel.png")

def plot_05_conv_flow():
    print("[05] Conv Flow")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 5: Input Image ──→ Convolution ──→ Feature Map", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.8, 2.5, 2.0, 3.2, color=B1, label="Input Image\n(e.g., 28x28x1)")
    box(ax, 5.0, 2.5, 2.0, 3.2, color=O1, label="Convolutional Layer\n(Filters Stack)")
    box(ax, 8.2, 2.5, 2.0, 3.2, color=G1, label="Feature Maps\n(Activated Regions)")
    
    arrow(ax, 2.9, 2.5, 3.9, 2.5, color=TX, lw=2)
    arrow(ax, 6.1, 2.5, 7.1, 2.5, color=TX, lw=2)
    
    save("05_conv_flow.png")

def plot_06_math_convolution():
    print("[06] Mathematical Convolution Grid")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Figure 6: Mathematical Convolution Visualization", fontsize=12, fontweight="bold", color=TX)
    
    # Input block
    ax = axes[0]; ax.axis("off"); ax.set_title("3x3 Input Slice", color=B1, fontsize=9.5)
    in_arr = np.array([[2, 1, 0], [0, 1, 2], [3, 0, 1]])
    for r in range(3):
        for c in range(3):
            rect = Rectangle((c, 2-r), 1, 1, fill=True, facecolor=CARD, edgecolor=TX2)
            ax.add_patch(rect)
            ax.text(c+0.5, 2.5-r, str(in_arr[r, c]), ha="center", va="center", color=TX, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.2, 3.2); ax.set_ylim(-0.2, 3.2)
    
    # Kernel block
    ax = axes[1]; ax.axis("off"); ax.set_title("3x3 Filter Kernel", color=O1, fontsize=9.5)
    f_arr = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    for r in range(3):
        for c in range(3):
            rect = Rectangle((c, 2-r), 1, 1, fill=True, facecolor=CARD, edgecolor=TX2)
            ax.add_patch(rect)
            ax.text(c+0.5, 2.5-r, str(f_arr[r, c]), ha="center", va="center", color=O1, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.2, 3.2); ax.set_ylim(-0.2, 3.2)
    
    # Output math
    ax = axes[2]; ax.axis("off"); ax.set_title("Dot Product Output", color=G1, fontsize=9.5)
    ax.text(0.5, 1.8, "Element-wise Multiplications:\n\n(2*1) + (1*0) + (0*1) +\n(0*0) + (1*1) + (2*0) +\n(3*1) + (0*0) + (1*1)\n\n= 2 + 1 + 3 + 1 = 7",
            ha="left", va="center", color=TX, fontsize=10)
    ax.set_xlim(0, 4); ax.set_ylim(0, 3)
    
    plt.tight_layout()
    save("06_math_convolution.png")

def plot_07_edge_filters():
    print("[07] Edge Detection Filters")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle("Figure 7: Edge Detection Filters Output", fontsize=12, fontweight="bold", color=TX)
    
    # Input
    ax = axes[0]; ax.axis("off"); ax.set_title("Input (Simple Cross)", color=TX2)
    box(ax, 2, 2, 3.5, 3.5, color=TX2, alpha=0.1)
    ax.plot([2, 2], [0.8, 3.2], color=TX, lw=10)
    ax.plot([0.8, 3.2], [2, 2], color=TX, lw=10)
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    
    # Vertical Output
    ax = axes[1]; ax.axis("off"); ax.set_title("Vertical Filter Output", color=B1)
    box(ax, 2, 2, 3.5, 3.5, color=B1, alpha=0.1)
    ax.plot([2, 2], [0.8, 3.2], color=B1, lw=10)
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    
    # Horizontal Output
    ax = axes[2]; ax.axis("off"); ax.set_title("Horizontal Filter Output", color=G1)
    box(ax, 2, 2, 3.5, 3.5, color=G1, alpha=0.1)
    ax.plot([0.8, 3.2], [2, 2], color=G1, lw=10)
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    
    save("07_edge_filters.png")

def plot_08_filter_effects():
    print("[08] Blur, Sharpen, and Sobel Filters")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Figure 8: Blur, Sharpen, and Sobel Filter Grids", fontsize=12, fontweight="bold", color=TX)
    
    kernels = [
        ("Blur Filter\n(Average)", np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]), G1),
        ("Sharpen Filter\n(Laplacian)", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]), B1),
        ("Sobel Filter\n(Vertical Edges)", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), R1)
    ]
    
    for idx, (title, arr, col) in enumerate(kernels):
        ax = axes[idx]; ax.axis("off"); ax.set_title(title, color=col, fontsize=10, fontweight="bold")
        for r in range(3):
            for c in range(3):
                rect = Rectangle((c, 2-r), 1, 1, fill=True, facecolor=CARD, edgecolor=col)
                ax.add_patch(rect)
                val_str = f"{arr[r, c]:.2f}" if idx==0 else f"{int(arr[r, c])}"
                ax.text(c+0.5, 2.5-r, val_str, ha="center", va="center", color=TX, fontsize=11)
        ax.set_xlim(-0.2, 3.2); ax.set_ylim(-0.2, 3.2)
        
    save("08_filter_effects.png")

def plot_09_feature_maps_depth():
    print("[09] Feature Maps Depth")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 9: Feature Resolution Across Layer Depths", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.8, 2.5, 2.2, 3.2, color=B1, label="Early Layers\n(High Resolution)\n\nExtracts:\nEdges, Lines,\nColor Gradients")
    box(ax, 5.0, 2.5, 2.2, 3.2, color=G1, label="Intermediate Layers\n(Medium Resolution)\n\nExtracts:\nTextures, Shapes,\nPatterns, Corners")
    box(ax, 8.2, 2.5, 2.2, 3.2, color=P1, label="Deep Layers\n(Low Resolution)\n\nExtracts:\nClass Parts, Faces,\nComplex Objects")
    
    arrow(ax, 3.0, 2.5, 3.8, 2.5, color=TX)
    arrow(ax, 6.2, 2.5, 7.0, 2.5, color=TX)
    
    save("09_feature_maps_depth.png")

def plot_10_max_avg_pooling():
    print("[10] Max vs Average Pooling")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Figure 10: Max Pooling vs. Average Pooling (2x2, Stride 2)", fontsize=12, fontweight="bold", color=TX)
    
    in_matrix = np.array([
        [10, 20],
        [8,  14]
    ])
    
    # Max Pool Output
    ax = axes[0]; ax.axis("off"); ax.set_title("Max Pooling\n(Keeps Maximum Activations)", color=B1, fontsize=10)
    rect_in = Rectangle((0, 1), 1, 1, fill=True, facecolor=CARD, edgecolor=B1)
    ax.add_patch(rect_in)
    ax.text(0.5, 1.5, "10   20\n8     14", ha="center", va="center", color=TX, fontsize=14)
    arrow(ax, 1.3, 1.5, 1.8, 1.5, color=B1, lw=2)
    rect_out = Rectangle((2, 1), 0.8, 0.8, fill=True, facecolor=CARD, edgecolor=O1)
    ax.add_patch(rect_out)
    ax.text(2.4, 1.4, "20", ha="center", va="center", color=TX, fontsize=18, fontweight="bold")
    ax.set_xlim(-0.2, 3.2); ax.set_ylim(0.5, 2.5)
    
    # Average Pool Output
    ax = axes[1]; ax.axis("off"); ax.set_title("Average Pooling\n(Smooths/Averages Values)", color=G1, fontsize=10)
    rect_in2 = Rectangle((0, 1), 1, 1, fill=True, facecolor=CARD, edgecolor=G1)
    ax.add_patch(rect_in2)
    ax.text(0.5, 1.5, "10   20\n8     14", ha="center", va="center", color=TX, fontsize=14)
    arrow(ax, 1.3, 1.5, 1.8, 1.5, color=G1, lw=2)
    rect_out2 = Rectangle((2, 1), 0.8, 0.8, fill=True, facecolor=CARD, edgecolor=O1)
    ax.add_patch(rect_out2)
    ax.text(2.4, 1.4, "13", ha="center", va="center", color=TX, fontsize=18, fontweight="bold")
    ax.set_xlim(-0.2, 3.2); ax.set_ylim(0.5, 2.5)
    
    save("10_max_avg_pooling.png")

def plot_11_pooling_reduction():
    print("[11] Pooling Reduction")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 11: Dimensionality Reduction Comparison", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.8, 2.5, 2.2, 3.2, color=B1, label="Input Feature Map\n(Resolution: 112x112)\n\nHigh memory usage\nDetailed spatial grid")
    box(ax, 5.0, 2.5, 2.0, 2.4, color=O1, label="Pooling layer\n(stride = 2)")
    box(ax, 8.2, 2.5, 2.2, 1.6, color=G1, label="Downsampled Output\n(Resolution: 56x56)\n\n75% fewer values\nPreserves features")
    
    arrow(ax, 3.0, 2.5, 3.9, 2.5, color=TX)
    arrow(ax, 6.1, 2.5, 7.0, 2.5, color=TX)
    
    save("11_pooling_reduction.png")

def plot_12_lenet5_architecture():
    print("[12] LeNet-5 Architecture")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 12: LeNet-5 Complete Architecture Flow (Yann LeCun, 1998)", fontsize=12, fontweight="bold", color=TX)
    
    layers = [
        (1.0, 2.5, 0.7, 2.5, "Input\n32x32", B1),
        (2.6, 2.5, 0.8, 2.0, "Conv 5x5\n28x28x6", O1),
        (4.0, 2.5, 0.8, 1.5, "Subsample\n14x14x6", G1),
        (5.5, 2.5, 0.8, 1.2, "Conv 5x5\n10x10x16", O1),
        (7.0, 2.5, 0.8, 0.9, "Subsample\n5x5x16", G1),
        (8.4, 2.5, 0.6, 0.6, "Conv C5\n120", O1),
        (9.5, 2.5, 0.5, 0.5, "FC F6\n84", P1),
        (10.5, 2.5, 0.4, 0.4, "Out\n10", GOLD),
    ]
    
    for x, y, w, h, lbl, col in layers:
        box(ax, x, y, w, h, color=col, label=lbl, fontsize=7.5)
        if x < 10.5:
            arrow(ax, x + w/2 + 0.05, 2.5, x + w/2 + 0.3, 2.5, color=TX2)
            
    save("12_lenet5_architecture.png")

def plot_13_alexnet_architecture():
    print("[13] AlexNet Architecture")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 13: AlexNet Dual-GPU Split Architecture (2012)", fontsize=12, fontweight="bold", color=TX)
    
    # Dual GPU rows representation
    # GPU 1
    ax.text(0.5, 3.8, "GPU 1 (Weights Split)", color=B1, fontweight="bold", fontsize=9)
    box(ax, 1.2, 3.0, 0.8, 1.0, color=B1, label="Conv1\n96ch/2", fontsize=7.5)
    box(ax, 2.6, 3.0, 0.8, 0.9, color=B1, label="Conv2\n256ch/2", fontsize=7.5)
    box(ax, 4.0, 3.0, 0.8, 0.8, color=B1, label="Conv3\n384ch/2", fontsize=7.5)
    box(ax, 5.4, 3.0, 0.8, 0.8, color=B1, label="Conv4\n384ch/2", fontsize=7.5)
    box(ax, 6.8, 3.0, 0.8, 0.8, color=B1, label="Conv5\n256ch/2", fontsize=7.5)
    
    # GPU 2
    ax.text(0.5, 1.0, "GPU 2 (Weights Split)", color=G1, fontweight="bold", fontsize=9)
    box(ax, 1.2, 1.8, 0.8, 1.0, color=G1, label="Conv1\n96ch/2", fontsize=7.5)
    box(ax, 2.6, 1.8, 0.8, 0.9, color=G1, label="Conv2\n256ch/2", fontsize=7.5)
    box(ax, 4.0, 1.8, 0.8, 0.8, color=G1, label="Conv3\n384ch/2", fontsize=7.5)
    box(ax, 5.4, 1.8, 0.8, 0.8, color=G1, label="Conv4\n384ch/2", fontsize=7.5)
    box(ax, 6.8, 1.8, 0.8, 0.8, color=G1, label="Conv5\n256ch/2", fontsize=7.5)
    
    # Shared heads
    box(ax, 8.5, 3.0, 0.8, 0.8, color=P1, label="FC1\n2048", fontsize=7.5)
    box(ax, 8.5, 1.8, 0.8, 0.8, color=P1, label="FC2\n2048", fontsize=7.5)
    box(ax, 9.8, 2.4, 0.8, 0.8, color=GOLD, label="Output\n1000", fontsize=7.5)
    
    # Connections
    for x_idx in [1.2, 2.6, 4.0, 5.4]:
        arrow(ax, x_idx+0.4, 3.0, x_idx+1.0, 3.0, color=B1)
        arrow(ax, x_idx+0.4, 1.8, x_idx+1.0, 1.8, color=G1)
    # Cross connections in Conv3
    arrow(ax, 2.6+0.4, 3.0, 4.0-0.4, 1.8, color=B1, lw=0.8, alpha=0.4)
    arrow(ax, 2.6+0.4, 1.8, 4.0-0.4, 3.0, color=G1, lw=0.8, alpha=0.4)
    
    # To FC
    arrow(ax, 6.8+0.4, 3.0, 8.5-0.4, 3.0, color=B1)
    arrow(ax, 6.8+0.4, 1.8, 8.5-0.4, 1.8, color=G1)
    arrow(ax, 6.8+0.4, 3.0, 8.5-0.4, 1.8, color=B1, lw=0.8, alpha=0.4)
    arrow(ax, 6.8+0.4, 1.8, 8.5-0.4, 3.0, color=G1, lw=0.8, alpha=0.4)
    
    # FC to output
    arrow(ax, 8.5+0.4, 3.0, 9.8-0.4, 2.4, color=P1)
    arrow(ax, 8.5+0.4, 1.8, 9.8-0.4, 2.4, color=P1)
    
    save("13_alexnet_architecture.png")

def plot_14_relu_activation():
    print("[14] ReLU Activation Plot")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle("Figure 14: ReLU vs. Tanh Activation Function", fontsize=12, fontweight="bold", color=TX)
    
    x = np.linspace(-3, 3, 200)
    y_relu = np.maximum(0, x)
    y_tanh = np.tanh(x)
    
    ax.plot(x, y_relu, color=B1, lw=2.5, label="ReLU: f(x) = max(0, x)")
    ax.plot(x, y_tanh, color=R1, lw=2.0, ls="--", label="Tanh: f(x) = tanh(x)")
    
    ax.axhline(0, color=TX2, lw=0.8)
    ax.axvline(0, color=TX2, lw=0.8)
    ax.set_xlabel("Input Activation (x)")
    ax.set_ylabel("Output Response (y)")
    ax.legend(loc="upper left")
    ax.grid(True)
    
    save("14_relu_activation.png")

def plot_15_inception_block():
    print("[15] GoogLeNet Inception Block")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 9); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 15: GoogLeNet Inception Module with 1x1 Convolutions", fontsize=12, fontweight="bold", color=TX)
    
    # Previous Layer
    box(ax, 4.5, 0.7, 1.8, 0.6, color=TX2, label="Previous Layer")
    
    # Convolutions
    box(ax, 1.5, 2.2, 1.4, 0.6, color=B1, label="1x1 Conv")
    
    box(ax, 3.5, 1.7, 1.4, 0.6, color=O1, label="1x1 Conv\n(reduction)")
    box(ax, 3.5, 2.7, 1.4, 0.6, color=B1, label="3x3 Conv")
    
    box(ax, 5.5, 1.7, 1.4, 0.6, color=O1, label="1x1 Conv\n(reduction)")
    box(ax, 5.5, 2.7, 1.4, 0.6, color=B1, label="5x5 Conv")
    
    box(ax, 7.5, 1.7, 1.4, 0.6, color=G1, label="3x3 Max Pool")
    box(ax, 7.5, 2.7, 1.4, 0.6, color=B1, label="1x1 Conv")
    
    # Filter Concat
    box(ax, 4.5, 4.2, 2.5, 0.6, color=P1, label="Filter Concatenation")
    
    # Draw Arrows
    arrow(ax, 4.5, 1.0, 1.5, 1.9, color=TX2)
    arrow(ax, 4.5, 1.0, 3.5, 1.4, color=TX2)
    arrow(ax, 4.5, 1.0, 5.5, 1.4, color=TX2)
    arrow(ax, 4.5, 1.0, 7.5, 1.4, color=TX2)
    
    arrow(ax, 3.5, 2.0, 3.5, 2.4, color=TX2)
    arrow(ax, 5.5, 2.0, 5.5, 2.4, color=TX2)
    arrow(ax, 7.5, 2.0, 7.5, 2.4, color=TX2)
    
    arrow(ax, 1.5, 2.5, 4.0, 3.9, color=TX2)
    arrow(ax, 3.5, 3.0, 4.2, 3.9, color=TX2)
    arrow(ax, 5.5, 3.0, 4.8, 3.9, color=TX2)
    arrow(ax, 7.5, 3.0, 5.0, 3.9, color=TX2)
    
    save("15_inception_block.png")

def plot_16_multiscale_extraction():
    print("[16] Multi-Scale Feature Extraction")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 16: Multi-Scale Feature Extraction Parallelism", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.5, 2.5, 1.6, 2.8, color=B1, label="Large Kernel\n(5x5 Conv)\n\nCaptures large,\nglobal shapes")
    box(ax, 5.0, 2.5, 1.6, 2.8, color=G1, label="Medium Kernel\n(3x3 Conv)\n\nCaptures mid-size\npatterns / parts")
    box(ax, 8.5, 2.5, 1.6, 2.8, color=P1, label="Small Kernel\n(1x1 Conv)\n\nCaptures local,\npixel-wise features")
    
    save("16_multiscale_extraction.png")

def plot_17_vgg16_architecture():
    print("[17] VGG16 Architecture")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 17: VGG-16 Deep Network Channel Progression", fontsize=12, fontweight="bold", color=TX)
    
    vgg_blocks = [
        (1.0, 2.5, 0.8, 3.0, "Input\n224x224x3", B1),
        (2.3, 2.5, 0.8, 2.6, "Block 1\n2x Conv\n64 filters", O1),
        (3.6, 2.5, 0.8, 2.2, "Block 2\n2x Conv\n128 filters", O1),
        (4.9, 2.5, 0.8, 1.8, "Block 3\n3x Conv\n256 filters", O1),
        (6.2, 2.5, 0.8, 1.4, "Block 4\n3x Conv\n512 filters", O1),
        (7.5, 2.5, 0.8, 1.0, "Block 5\n3x Conv\n512 filters", O1),
        (8.8, 2.5, 0.6, 0.6, "FC\n4096", P1),
        (9.7, 2.5, 0.4, 0.4, "Out\n1000", GOLD)
    ]
    
    for x, y, w, h, lbl, col in vgg_blocks:
        box(ax, x, y, w, h, color=col, label=lbl, fontsize=7)
        if x < 9.7:
            arrow(ax, x + w/2 + 0.05, 2.5, x + w/2 + 0.4, 2.5, color=TX2)
            
    save("17_vgg16_architecture.png")

def plot_18_stacked_convolutions():
    print("[18] Stacked Convolutions")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 18: Stacked 3x3 Convs Receptive Field Equivalent to 5x5 Conv", fontsize=12, fontweight="bold", color=TX)
    
    # 5x5 effective grid
    for r in range(5):
        for c in range(5):
            rect = Rectangle((c*0.5 + 0.5, r*0.5 + 1.2), 0.4, 0.4, fill=True, facecolor=CARD, edgecolor=TX2, lw=0.8)
            ax.add_patch(rect)
    ax.text(1.5, 0.8, "Input Grid (5x5)", color=TX2, ha="center", fontsize=8.5)
    
    # Highlighting Conv 1 (3x3 receptive field on input grid)
    rf1 = Rectangle((0.5, 1.2), 1.4, 1.4, fill=True, facecolor=B1, alpha=0.3, edgecolor=B1, lw=1.5)
    ax.add_patch(rf1)
    
    # Intermediate layer (3x3 grid)
    for r in range(3):
        for c in range(3):
            rect = Rectangle((c*0.6 + 4.5, r*0.6 + 1.6), 0.5, 0.5, fill=True, facecolor=CARD, edgecolor=G1, lw=0.8)
            ax.add_patch(rect)
    ax.text(5.1, 1.2, "Layer 2 (3x3)", color=G1, ha="center", fontsize=8.5)
    
    # Highlighting Conv 2 (3x3 receptive field on layer 2)
    rf2 = Rectangle((4.5, 1.6), 1.7, 1.7, fill=True, facecolor=G1, alpha=0.25, edgecolor=G1, lw=1.5)
    ax.add_patch(rf2)
    
    # Output node (1x1)
    node(ax, 8.5, 2.5, r=0.25, color=O1, label="Out")
    ax.text(8.5, 2.0, "Output Layer", color=O1, ha="center", fontsize=8.5)
    
    # Connect
    arrow(ax, 1.9, 2.5, 4.4, 2.5, color=B1)
    arrow(ax, 6.2, 2.5, 8.2, 2.5, color=G1)
    
    save("18_stacked_convolutions.png")

def plot_19_resnet_block():
    print("[19] ResNet Basic Block")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 19: ResNet Basic Skip Connection Unit", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 2.5, 4.2, 1.5, 0.6, color=TX2, label="Input (x)")
    
    box(ax, 2.5, 3.0, 1.5, 0.6, color=B1, label="Weight Layer\n(Conv 3x3)")
    box(ax, 2.5, 1.8, 1.5, 0.6, color=B1, label="Weight Layer\n(Conv 3x3)")
    
    node(ax, 2.5, 0.6, r=0.25, color=O1, label="+")
    
    # Standard flow
    arrow(ax, 2.5, 3.9, 2.5, 3.3, color=TX2)
    arrow(ax, 2.5, 2.7, 2.5, 2.1, color=TX2)
    arrow(ax, 2.5, 1.5, 2.5, 0.9, color=TX2)
    
    # Skip Connection
    ax.annotate("", xy=(2.5, 0.6), xytext=(2.5, 4.2),
                arrowprops=dict(arrowstyle="-|>", color=G1, lw=2, connectionstyle="bar,fraction=0.85"),
                zorder=1)
    ax.text(5.5, 2.4, "Skip Connection\n(Identity Path x)", color=G1, fontweight="bold", va="center", ha="center")
    
    # After Addition
    arrow(ax, 2.5, 0.35, 2.5, 0.1, color=TX2)
    ax.text(3.3, 0.6, "f(x) + x", color=TX, fontweight="bold", va="center")
    
    save("19_resnet_block.png")

def plot_20_resnet_gradient_flow():
    print("[20] ResNet Gradient Flow")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 20: Gradient Bypass Path During Backpropagation", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 2.5, 4.2, 1.5, 0.6, color=TX2, label="Loss Gradients")
    
    box(ax, 2.5, 3.0, 1.5, 0.6, color=R1, label="Weight Layer\n(Vanished/Blocked)", alpha=0.3)
    box(ax, 2.5, 1.8, 1.5, 0.6, color=R1, label="Weight Layer\n(Vanished/Blocked)", alpha=0.3)
    
    node(ax, 2.5, 0.6, r=0.25, color=O1, label="+")
    
    # Backward arrows through weight layers (vanished)
    ax.annotate("", xy=(2.5, 3.3), xytext=(2.5, 3.9), arrowprops=dict(arrowstyle="<-", color=R1, lw=1.2, ls=":"))
    ax.annotate("", xy=(2.5, 2.1), xytext=(2.5, 2.7), arrowprops=dict(arrowstyle="<-", color=R1, lw=1.2, ls=":"))
    
    # Backward arrow through skip connection (unhindered)
    ax.annotate("", xy=(2.5, 4.2), xytext=(2.5, 0.6),
                arrowprops=dict(arrowstyle="<-", color=G1, lw=2.5, connectionstyle="bar,fraction=0.85"),
                zorder=1)
    
    ax.text(5.5, 2.4, "Unhindered Gradient Flow\nBypasses Conv Layers", color=G1, fontweight="bold", va="center", ha="center")
    
    save("20_resnet_gradient_flow.png")

def plot_21_depthwise_separable_conv():
    print("[21] Depthwise Separable Conv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Figure 21: Depthwise Separable Convolution Components", fontsize=12, fontweight="bold", color=TX)
    
    # 1. Depthwise Convolution
    ax = axes[0]; ax.axis("off"); ax.set_title("Step 1: Depthwise Convolution\n(Spatial filter per channel independently)", color=B1, fontsize=9.5)
    for z in range(3):
        x = np.array([0, 2, 2, 0, 0]) + z*0.4
        y = np.array([0, 0, 2, 2, 0]) + z*0.4
        ax.plot(x, y, color=B1, lw=1.5, alpha=0.8)
        ax.text(1.0+z*0.4, 1.0+z*0.4, f"CH {z+1}", color=TX, fontsize=8)
    ax.set_xlim(-0.2, 3.5); ax.set_ylim(-0.2, 3.5)
    
    # 2. Pointwise Convolution
    ax = axes[1]; ax.axis("off"); ax.set_title("Step 2: Pointwise Convolution\n(1x1 convolution across channels)", color=G1, fontsize=9.5)
    box(ax, 1.8, 1.8, 2.0, 1.8, color=G1, label="1x1 Conv Filters\n(Combines channel\nactivations)", alpha=0.2)
    ax.set_xlim(0, 3.6); ax.set_ylim(0, 3.6)
    
    save("21_depthwise_separable_conv.png")

def plot_22_xception_architecture():
    print("[22] Xception Architecture")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 22: Xception Flow Structure Blocks", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.8, 2.5, 2.2, 3.2, color=B1, label="Entry Flow\n\n- Performs standard convs\n- Downsamples spatial grid\n- Doubles depth channels")
    box(ax, 5.0, 2.5, 2.2, 3.2, color=G1, label="Middle Flow\n\n- Repeated 8 times\n- Depthwise separable convs\n- Focuses on representations")
    box(ax, 8.2, 2.5, 2.2, 3.2, color=P1, label="Exit Flow\n\n- Performs final up/down\n- Global Avg Pooling\n- Dense Classifier output")
    
    arrow(ax, 3.0, 2.5, 3.8, 2.5, color=TX)
    arrow(ax, 6.2, 2.5, 7.0, 2.5, color=TX)
    
    save("22_xception_architecture.png")

def plot_23_squeeze_excitation():
    print("[23] Squeeze and Excitation Block")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 23: SENet Squeeze-and-Excitation Recalibration Block", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.2, 2.5, 1.2, 2.8, color=TX2, label="Feature Map\nH x W x C")
    box(ax, 3.2, 2.5, 1.2, 2.8, color=B1, label="Squeeze\n(Global Avg\nPooling)\n1 x 1 x C")
    box(ax, 5.4, 2.5, 1.4, 2.8, color=O1, label="Excitation\n(FC Layers +\nSigmoid)\n1 x 1 x C weights")
    box(ax, 7.8, 2.5, 1.4, 2.8, color=G1, label="Scale\n(Reweight\nChannels)\nRecalibrated maps")
    
    arrow(ax, 1.9, 2.5, 2.5, 2.5, color=TX2)
    arrow(ax, 3.9, 2.5, 4.6, 2.5, color=TX2)
    arrow(ax, 6.2, 2.5, 7.0, 2.5, color=TX2)
    
    save("23_squeeze_excitation.png")

def plot_24_channel_attention():
    print("[24] Channel Attention Mechanism")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 9); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 24: Squeeze-and-Excitation Channel Attention Reweighting", fontsize=12, fontweight="bold", color=TX)
    
    # Stack of feature maps with different weight multipliers
    colors = [R1, G1, B1]
    scales = [0.2, 1.2, 0.6]
    labels = ["CH 1 (Scale: 0.2)", "CH 2 (Scale: 1.2)", "CH 3 (Scale: 0.6)"]
    for z in range(3):
        x = np.array([1, 4, 4, 1, 1])
        y = np.array([1, 1, 3, 3, 1]) + z*0.4
        # Draw base
        ax.plot(x, y, color=colors[z], alpha=0.3)
        
        # Draw Scaled feature map representation
        x_s = np.array([5.5, 8.5, 8.5, 5.5, 5.5])
        y_s = np.array([1, 1, 1+2*scales[z], 1+2*scales[z], 1]) + z*0.4
        ax.plot(x_s, y_s, color=colors[z], lw=2)
        ax.fill(x_s, y_s, color=colors[z], alpha=0.15)
        ax.text(7.0, 1.2 + z*0.4, labels[z], color=colors[z], fontsize=8, ha="center")
        
    arrow(ax, 4.2, 2.5, 5.3, 2.5, color=TX2, lw=2)
    ax.text(4.75, 2.8, "Scale\nExcitation", color=O1, ha="center", fontsize=8.5, fontweight="bold")
    
    save("24_channel_attention.png")

def plot_25_transfer_learning_workflow():
    print("[25] Transfer Learning Workflow")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 25: Pretrained Model Transfer Learning Workflow", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 2.0, 2.5, 2.4, 3.2, color=B1, label="Base Model\n(Pretrained on ImageNet)\n\nExtracted generic features\n(edges, shapes, textures)")
    box(ax, 5.0, 2.5, 1.8, 1.5, color=O1, label="Freeze Base\nLayers")
    box(ax, 8.0, 2.5, 2.4, 3.2, color=G1, label="New Custom Head\n(Trainable Classifiers)\n\nTrains only top layer\non specific class targets")
    
    arrow(ax, 3.3, 2.5, 4.0, 2.5, color=TX)
    arrow(ax, 6.0, 2.5, 6.7, 2.5, color=TX)
    
    save("25_transfer_learning_workflow.png")

def plot_26_frozen_vs_trainable():
    print("[26] Frozen vs Trainable Layers")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 9); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 26: Layer Freezing Timeline vs Fine-Tuning Steps", fontsize=12, fontweight="bold", color=TX)
    
    # Base network
    box(ax, 2.0, 3.5, 2.6, 1.0, color=B1, label="Base Layers: FROZEN\n(Weights locked)", alpha=0.3)
    box(ax, 6.0, 3.5, 2.6, 1.0, color=G1, label="Custom Head: TRAINABLE\n(Warming up weights)")
    ax.text(4.5, 4.6, "Step 1: Warm up the Head", color=TX, fontweight="bold", ha="center")
    arrow(ax, 3.4, 3.5, 4.6, 3.5, color=TX2)
    
    # Fine tuning stage
    box(ax, 2.0, 1.5, 2.6, 1.0, color=O1, label="Top Base Layers: UNFREEZE\n(Train with small LR e.g. 10^-5)")
    box(ax, 6.0, 1.5, 2.6, 1.0, color=G1, label="Custom Head: TRAINABLE\n(Fine tuning convergence)")
    ax.text(4.5, 2.3, "Step 2: Unfreeze top base layers for Fine-Tuning", color=TX, fontweight="bold", ha="center")
    arrow(ax, 3.4, 1.5, 4.6, 1.5, color=TX2)
    
    save("26_frozen_vs_trainable.png")

def plot_27_classification_vs_localization():
    print("[27] Classification vs Localization")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Figure 27: Classification vs. Localization Comparison", fontsize=12, fontweight="bold", color=TX)
    
    # Classification
    ax = axes[0]; ax.axis("off"); ax.set_title("Classification\n(Detects category presence)", color=B1, fontsize=10)
    box(ax, 2, 2, 3.2, 3.2, color=B1, alpha=0.1)
    ax.text(2, 2.4, "🐈", fontsize=48, ha="center")
    ax.text(2, 1.2, "Class Label: 'CAT'\nProbability: 98.4%", color=B1, fontsize=10, fontweight="bold", ha="center")
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    
    # Localization
    ax = axes[1]; ax.axis("off"); ax.set_title("Localization\n(Predicts category boundaries)", color=G1, fontsize=10)
    box(ax, 2, 2, 3.2, 3.2, color=G1, alpha=0.1)
    ax.text(2, 2.4, "🐈", fontsize=48, ha="center")
    # Draw boundary box
    rect = Rectangle((0.8, 1.0), 2.4, 2.4, fill=False, edgecolor=G1, lw=2.5)
    ax.add_patch(rect)
    ax.text(2, 0.6, "Coordinates:\n[x=0.2, y=0.2, w=0.6, h=0.6]", color=G1, fontsize=9.5, fontweight="bold", ha="center")
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    
    save("27_classification_vs_localization.png")

def plot_28_bounding_box_prediction():
    print("[28] Bounding Box Prediction")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 9); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 28: Dual-Head Localization and Classification Network", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.8, 2.5, 2.0, 1.2, color=B1, label="Base CNN Layers\n(e.g., Xception)")
    
    box(ax, 4.5, 2.5, 1.8, 0.8, color=O1, label="Global Average\nPooling")
    
    # Heads
    box(ax, 7.5, 3.5, 2.0, 1.0, color=G1, label="Classification Head\n(Softmax Outputs)")
    box(ax, 7.5, 1.5, 2.0, 1.0, color=P1, label="Regression Head\n(Box Coordinates:\nx, y, w, h)")
    
    arrow(ax, 2.9, 2.5, 3.5, 2.5, color=TX2)
    arrow(ax, 5.5, 2.5, 6.4, 3.5, color=TX2)
    arrow(ax, 5.5, 2.5, 6.4, 1.5, color=TX2)
    
    save("28_bounding_box_prediction.png")

def plot_29_object_detection_pipeline():
    print("[29] Object Detection Pipeline")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 29: Faster R-CNN Detection Pipeline Stages", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.2, 2.5, 1.6, 2.8, color=B1, label="CNN Backbone\n(Shared Map)")
    box(ax, 3.4, 3.8, 1.8, 1.0, color=O1, label="Region Proposal\nNetwork (RPN)")
    box(ax, 5.6, 2.5, 1.8, 1.0, color=G1, label="RoI Pooling\n(Fixed Vectors)")
    box(ax, 8.2, 2.5, 2.0, 2.8, color=P1, label="Fast R-CNN Head\n\n- Class Softmax\n- BBox Regressor")
    
    arrow(ax, 2.1, 2.5, 4.6, 2.5, color=TX2)
    arrow(ax, 2.1, 2.5, 2.8, 3.8, color=TX2)
    arrow(ax, 4.4, 3.8, 5.0, 3.0, color=TX2)
    arrow(ax, 6.6, 2.5, 7.1, 2.5, color=TX2)
    
    save("29_object_detection_pipeline.png")

def plot_30_multiobject_detection():
    print("[30] Multi-Object Detection")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 30: Multi-Object Detection with Class Labels", fontsize=12, fontweight="bold", color=TX)
    
    # Image frame
    box(ax, 4.0, 2.5, 7.0, 3.5, color=TX2, label="", alpha=0.08)
    
    # Objects
    ax.text(2.0, 2.8, "🐈", fontsize=38, ha="center")
    rect1 = Rectangle((1.0, 1.6), 2.0, 2.2, fill=False, edgecolor=B1, lw=2)
    ax.add_patch(rect1)
    ax.text(1.0, 3.9, "Cat: 0.96", color="white", backgroundcolor=B1, fontsize=8, fontweight="bold")
    
    ax.text(5.5, 2.5, "🐕", fontsize=48, ha="center")
    rect2 = Rectangle((4.0, 1.2), 3.0, 2.8, fill=False, edgecolor=G1, lw=2)
    ax.add_patch(rect2)
    ax.text(4.0, 4.1, "Dog: 0.92", color="white", backgroundcolor=G1, fontsize=8, fontweight="bold")
    
    save("30_multiobject_detection.png")

def plot_31_fcn_conversion():
    print("[31] FCN Conversion")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 9); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 31: FCN Dense Layer to 1x1 Convolution Conversion", fontsize=12, fontweight="bold", color=TX)
    
    # Left: Dense Layer
    box(ax, 2.0, 3.0, 2.4, 1.2, color=B1, label="Dense (FC) Layer\n(Flattened vector\nof 4096 units)", alpha=0.25)
    ax.text(2.0, 1.6, "Loses spatial structure\nRequires fixed input dimensions", color=TX2, ha="center", fontsize=8.5)
    
    arrow(ax, 3.6, 3.0, 5.0, 3.0, color=TX, lw=2)
    
    # Right: 1x1 Conv
    box(ax, 6.6, 3.0, 2.4, 1.2, color=G1, label="1x1 Conv Layer\n(4096 filters of\nshape 1x1)", alpha=0.25)
    ax.text(6.6, 1.6, "Retains spatial structure\nAccepts variable inputs size", color=TX2, ha="center", fontsize=8.5)
    
    save("31_fcn_conversion.png")

def plot_32_encoder_decoder_flow():
    print("[32] Encoder Decoder Flow")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 32: Symmetric U-Net Encoder-Decoder Flow", fontsize=12, fontweight="bold", color=TX)
    
    # U-shape blocks
    box(ax, 1.2, 4.0, 1.4, 0.8, color=B1, label="Encoder L1\n224x224")
    box(ax, 2.2, 2.8, 1.4, 0.8, color=B1, label="Encoder L2\n112x112")
    
    box(ax, 4.5, 1.3, 1.6, 0.8, color=O1, label="Bottleneck\n56x56")
    
    box(ax, 6.8, 2.8, 1.4, 0.8, color=G1, label="Decoder L2\n112x112")
    box(ax, 7.8, 4.0, 1.4, 0.8, color=G1, label="Decoder L1\n224x224")
    
    # Connections down
    arrow(ax, 1.6, 3.5, 1.9, 3.3, color=TX2)
    arrow(ax, 2.6, 2.3, 3.8, 1.6, color=TX2)
    
    # Connections up
    arrow(ax, 5.2, 1.6, 6.4, 2.3, color=TX2)
    arrow(ax, 7.2, 3.3, 7.5, 3.5, color=TX2)
    
    # Skip connections (horizontal)
    ax.annotate("", xy=(6.0, 2.8), xytext=(3.0, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=P1, lw=1.5, ls="--"), zorder=1)
    ax.annotate("", xy=(7.0, 4.0), xytext=(2.0, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=P1, lw=1.5, ls="--"), zorder=1)
    ax.text(4.5, 3.0, "Skip Connections (Concatenation)", color=P1, fontsize=8, ha="center")
    
    save("32_encoder_decoder_flow.png")

def plot_33_yolo_grid():
    print("[33] YOLO Grid Concept")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 33: YOLO Grid Cell Mapping", fontsize=12, fontweight="bold", color=TX)
    
    # Draw 7x7 grid
    for r in range(7):
        for c in range(7):
            rect = Rectangle((c*0.6 + 1.0, r*0.6 + 0.5), 0.5, 0.5, fill=True, facecolor=CARD, edgecolor=TX2, lw=0.8)
            ax.add_patch(rect)
            
    # Highlight grid cell containing object center
    cell = Rectangle((4*0.6 + 1.0, 3*0.6 + 0.5), 0.5, 0.5, fill=True, facecolor=B1, alpha=0.4, edgecolor=B1, lw=2.0)
    ax.add_patch(cell)
    
    # Draw predicted bounding box starting from cell center
    bbox = Rectangle((4*0.6 + 1.0 - 0.8, 3*0.6 + 0.5 - 0.6), 2.2, 1.8, fill=False, edgecolor=G1, lw=2.5)
    ax.add_patch(bbox)
    
    ax.text(3.4, 2.6, "Grid Cell\n(Responsible for target)", color=B1, fontsize=7.5, fontweight="bold", ha="center")
    ax.text(4.0, 4.2, "Predicted Bounding Box", color=G1, fontsize=8.5, fontweight="bold")
    
    save("33_yolo_grid.png")

def plot_34_yolo_workflow():
    print("[34] YOLO Workflow")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 34: YOLO Real-Time Single Forward Pass Workflow", fontsize=12, fontweight="bold", color=TX)
    
    box(ax, 1.8, 2.5, 2.2, 3.2, color=B1, label="Input Image\n(Feed Forward Pass)")
    box(ax, 5.0, 2.5, 2.2, 3.2, color=O1, label="Unified Darknet/CNN\n(Predicts grid cells\nboxes & classes simultaneously)")
    box(ax, 8.2, 2.5, 2.2, 3.2, color=G1, label="Final Output\n(Boxes + Categories after\nNon-Max Suppression)")
    
    arrow(ax, 3.0, 2.5, 3.8, 2.5, color=TX, lw=1.5)
    arrow(ax, 6.2, 2.5, 7.0, 2.5, color=TX, lw=1.5)
    
    save("34_yolo_workflow.png")

def plot_35_original_vs_segmentation():
    print("[35] Original vs Segmentation")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Figure 35: Input Image vs. Semantic Segmentation Mask", fontsize=12, fontweight="bold", color=TX)
    
    # Original
    ax = axes[0]; ax.axis("off"); ax.set_title("Input Image (RGB)", color=B1, fontsize=10)
    box(ax, 2, 2, 3.2, 3.2, color=B1, alpha=0.1)
    ax.text(2, 2, "🚗   🌲   🏠", fontsize=38, ha="center", va="center")
    
    # Mask
    ax = axes[1]; ax.axis("off"); ax.set_title("Semantic Segmentation Mask", color=G1, fontsize=10)
    box(ax, 2, 2, 3.2, 3.2, color=G1, alpha=0.1)
    rect1 = Rectangle((0.6, 1.6), 1.2, 0.8, color=R1, label="Car Pixels")
    rect2 = Rectangle((1.8, 1.6), 0.8, 1.2, color=G1, label="Tree Pixels")
    rect3 = Rectangle((2.8, 1.6), 0.8, 1.0, color=P1, label="House Pixels")
    ax.add_patch(rect1); ax.add_patch(rect2); ax.add_patch(rect3)
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    
    save("35_original_vs_segmentation.png")

def plot_36_pixel_segmentation():
    print("[36] Pixel Segmentation Grid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 36: Pixel-Wise Category Classification Map", fontsize=12, fontweight="bold", color=TX)
    
    # Grid of classified pixels
    grid_colors = [
        [R1, R1, B1, B1, G1],
        [R1, B1, B1, G1, G1],
        [B1, B1, G1, G1, P1],
        [B1, G1, G1, P1, P1]
    ]
    
    for r in range(4):
        for c in range(5):
            col = grid_colors[r][c]
            rect = Rectangle((c*1.0 + 1.5, r*0.8 + 0.8), 0.8, 0.6, color=col, alpha=0.75, ec="white")
            ax.add_patch(rect)
            lbl = "Car" if col==R1 else ("Sky" if col==B1 else ("Tree" if col==G1 else "Road"))
            ax.text(c*1.0 + 1.9, r*0.8 + 1.1, lbl, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
            
    save("36_pixel_segmentation.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Generating Chapter 14 Visual Assets...")
    
    plot_01_visual_cortex_pipeline()
    plot_02_shape_hierarchy()
    plot_03_biological_inspiration()
    plot_04_sliding_kernel()
    plot_05_conv_flow()
    plot_06_math_convolution()
    plot_07_edge_filters()
    plot_08_filter_effects()
    plot_09_feature_maps_depth()
    plot_10_max_avg_pooling()
    plot_11_pooling_reduction()
    plot_12_lenet5_architecture()
    plot_13_alexnet_architecture()
    plot_14_relu_activation()
    plot_15_inception_block()
    plot_16_multiscale_extraction()
    plot_17_vgg16_architecture()
    plot_18_stacked_convolutions()
    plot_19_resnet_block()
    plot_20_resnet_gradient_flow()
    plot_21_depthwise_separable_conv()
    plot_22_xception_architecture()
    plot_23_squeeze_excitation()
    plot_24_channel_attention()
    plot_25_transfer_learning_workflow()
    plot_26_frozen_vs_trainable()
    plot_27_classification_vs_localization()
    plot_28_bounding_box_prediction()
    plot_29_object_detection_pipeline()
    plot_30_multiobject_detection()
    plot_31_fcn_conversion()
    plot_32_encoder_decoder_flow()
    plot_33_yolo_grid()
    plot_34_yolo_workflow()
    plot_35_original_vs_segmentation()
    plot_36_pixel_segmentation()
    
    print("🎉 All 36 visuals created successfully!")
