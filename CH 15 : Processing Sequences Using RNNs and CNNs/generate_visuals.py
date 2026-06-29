"""
 ╔══════════════════════════════════════════════════════════════════╗
 ║   CH 15: Processing Sequences — COMPLETE Visuals (11 Plots)      ║
 ║   Programmatic dark-theme diagrams for all textbook figures      ║
 ║   Run: python3 generate_visuals.py                              ║
 ╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, ConnectionPatch
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

def node(ax, x, y, r=0.25, color=B1, label="", fontsize=8, alpha=0.9):
    c = Circle((x, y), r, color=color, zorder=4, linewidth=1.2, ec="white", alpha=alpha)
    ax.add_patch(c)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=5)

def arrow(ax, x1, y1, x2, y2, color=TX2, lw=1.2, alpha=0.5, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, alpha=alpha),
                zorder=2)

def box(ax, x, y, w, h, color=B1, label="", fontsize=8.5, alpha=0.25, lw=1.5, ls="-"):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.04", fc=color, alpha=alpha,
                       ec=color, lw=lw, zorder=2, ls=ls)
    ax.add_patch(r)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=3)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def plot_01_recurrent_neuron_unrolled():
    print("[01] Recurrent Neuron Unrolled")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    fig.suptitle("Figure 15-1: Recurrent Neuron Unrolled through Time", fontsize=11, fontweight="bold", color=TX)
    
    # Left: Recurrent Neuron
    ax.text(1.5, 3.5, "Folded Representation", color=B1, fontweight="bold", ha="center", fontsize=9)
    node(ax, 1.5, 2.0, r=0.4, color=B1, label="RNN\nNeuron")
    arrow(ax, 1.5, 1.0, 1.5, 1.6, color=TX, lw=1.5)
    ax.text(1.5, 0.8, "x(t)", color=TX, ha="center", va="center", fontweight="bold")
    arrow(ax, 1.5, 2.4, 1.5, 3.0, color=G1, lw=1.5)
    ax.text(1.5, 3.2, "y(t)", color=G1, ha="center", va="center", fontweight="bold")
    # Loop arrow
    ax.annotate("", xy=(1.2, 2.3), xytext=(1.8, 2.3),
                arrowprops=dict(arrowstyle="->", color=O1, lw=1.5,
                                connectionstyle="arc3,rad=-1.8"), zorder=3)
    
    # Separator
    ax.plot([3.2, 3.2], [0.5, 3.5], color="#21262d", ls="--")
    
    # Right: Unrolled through time
    ax.text(6.5, 3.5, "Unrolled through Time", color=G1, fontweight="bold", ha="center", fontsize=9)
    time_steps = [("t-1", 4.8), ("t", 6.5), ("t+1", 8.2)]
    
    for idx, (t, x_pos) in enumerate(time_steps):
        node(ax, x_pos, 2.0, r=0.35, color=G1, label=f"Neuron\n{t}")
        arrow(ax, x_pos, 1.0, x_pos, 1.65, color=TX, lw=1.5)
        ax.text(x_pos, 0.8, f"x({t})", color=TX, ha="center", va="center")
        arrow(ax, x_pos, 2.35, x_pos, 3.0, color=G1, lw=1.5)
        ax.text(x_pos, 3.2, f"y({t})", color=G1, ha="center", va="center")
        
        # Horizontal temporal flow
        if idx < len(time_steps) - 1:
            next_x = time_steps[idx+1][1]
            arrow(ax, x_pos + 0.35, 2.0, next_x - 0.35, 2.0, color=O1, lw=1.8)
            ax.text((x_pos + next_x)/2, 2.2, f"h({t})", color=O1, ha="center", fontsize=7.5)
            
    save("01_recurrent_neuron_unrolled.png")

def plot_02_recurrent_layer_unrolled():
    print("[02] Recurrent Layer Unrolled")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    fig.suptitle("Figure 15-2: Recurrent Layer Unrolled through Time", fontsize=11, fontweight="bold", color=TX)
    
    # Left: Folded Layer
    ax.text(1.5, 3.5, "Folded Layer", color=B1, fontweight="bold", ha="center", fontsize=9)
    box(ax, 1.5, 2.0, 1.2, 0.8, color=B1, label="Recurrent\nLayer (n neurons)")
    arrow(ax, 1.5, 1.0, 1.5, 1.6, color=TX, lw=1.5)
    ax.text(1.5, 0.8, "X(t) [m x d]", color=TX, ha="center", va="center")
    arrow(ax, 1.5, 2.4, 1.5, 3.0, color=G1, lw=1.5)
    ax.text(1.5, 3.2, "Y(t) [m x n]", color=G1, ha="center", va="center")
    # Feedback loop
    ax.annotate("", xy=(1.0, 2.2), xytext=(2.0, 2.2),
                arrowprops=dict(arrowstyle="->", color=O1, lw=1.5,
                                connectionstyle="arc3,rad=-1.8"), zorder=3)
    
    # Separator
    ax.plot([3.2, 3.2], [0.5, 3.5], color="#21262d", ls="--")
    
    # Right: Unrolled
    ax.text(6.5, 3.5, "Unrolled Layer Flow", color=G1, fontweight="bold", ha="center", fontsize=9)
    steps = [("t-1", 4.8), ("t", 6.5), ("t+1", 8.2)]
    
    for idx, (t, x_pos) in enumerate(steps):
        box(ax, x_pos, 2.0, 1.1, 0.7, color=G1, label=f"Layer\nstep {t}")
        arrow(ax, x_pos, 1.0, x_pos, 1.65, color=TX, lw=1.5)
        ax.text(x_pos, 0.8, f"X({t})", color=TX, ha="center", va="center")
        arrow(ax, x_pos, 2.35, x_pos, 3.0, color=G1, lw=1.5)
        ax.text(x_pos, 3.2, f"Y({t})", color=G1, ha="center", va="center")
        
        # Recurrent state link
        if idx < len(steps) - 1:
            next_x = steps[idx+1][1]
            arrow(ax, x_pos + 0.55, 2.0, next_x - 0.55, 2.0, color=O1, lw=1.8)
            ax.text((x_pos + next_x)/2, 2.25, f"Y({t})", color=O1, ha="center", fontsize=7.5)
            
    save("02_recurrent_layer_unrolled.png")

def plot_03_hidden_state_vs_output():
    print("[03] Hidden State vs Output")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 15-3: Cell's Hidden State h(t) vs. Output y(t)", fontsize=11, fontweight="bold", color=TX)
    
    # Main Cell Box
    box(ax, 4.0, 2.5, 3.2, 2.4, color=P1, label="", alpha=0.1, lw=2)
    ax.text(4.0, 4.0, "Recurrent Cell", color=P1, fontweight="bold", ha="center", fontsize=10)
    
    # Inputs
    arrow(ax, 1.5, 2.5, 2.4, 2.5, color=TX, lw=1.5)
    ax.text(1.2, 2.5, "Input\nx(t)", color=TX, ha="center", va="center", fontweight="bold")
    
    arrow(ax, 2.8, 1.0, 2.8, 1.7, color=O1, lw=1.5)
    ax.text(2.8, 0.8, "Prev State\nh(t-1)", color=O1, ha="center", va="center")
    
    # Node representing linear combination & activation
    node(ax, 3.2, 2.5, r=0.35, color=B1, label="f(x,h)")
    arrow(ax, 2.8, 1.7, 3.0, 2.2, color=O1, lw=1.2)
    
    # Cell state split
    node(ax, 4.8, 2.5, r=0.3, color=O1, label="h(t)")
    arrow(ax, 3.55, 2.5, 4.5, 2.5, color=B1, lw=1.5)
    
    # Outputs
    arrow(ax, 4.8, 2.8, 4.8, 4.2, color=G1, lw=1.5)
    ax.text(4.8, 4.4, "Output y(t)", color=G1, ha="center", va="center", fontweight="bold")
    
    arrow(ax, 5.1, 2.5, 6.8, 2.5, color=O1, lw=1.5)
    ax.text(7.2, 2.5, "Next State\nh(t)", color=O1, ha="center", va="center")
    
    # Callout about difference
    ax.text(4.0, 0.3, "For simple RNNs: h(t) = y(t)\nFor advanced cells (LSTM): State contains long-term memory distinct from outputs.",
            color=TX2, ha="center", style="italic", fontsize=8)
            
    save("03_hidden_state_vs_output.png")

def plot_04_rnn_seq_types():
    print("[04] RNN Sequence Types")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Figure 15-4: Taxonomy of Sequence Mapping Workflows", fontsize=12, fontweight="bold", color=TX)
    
    # 1. Seq-to-Seq
    ax = axes[0, 0]; ax.axis("off"); ax.set_title("Sequence-to-Sequence (e.g. Time Series Forecasting)", color=B1, fontsize=9.5, fontweight="bold")
    box(ax, 2, 2, 3.5, 2.2, color=B1, alpha=0.05, lw=1)
    for idx, x in enumerate([1.0, 2.0, 3.0]):
        node(ax, x, 2.0, r=0.25, color=B1, label="Cell")
        arrow(ax, x, 1.0, x, 1.75, color=TX)
        arrow(ax, x, 2.25, x, 3.0, color=G1)
        if idx < 2:
            arrow(ax, x+0.25, 2.0, x+0.75, 2.0, color=O1)
    ax.text(2.0, 0.8, "Inputs x(1) to x(T)", color=TX, ha="center", fontsize=8)
    ax.text(2.0, 3.2, "Outputs y(1) to y(T)", color=G1, ha="center", fontsize=8)
    ax.set_xlim(0.2, 3.8); ax.set_ylim(0.5, 3.5)
    
    # 2. Seq-to-Vec
    ax = axes[0, 1]; ax.axis("off"); ax.set_title("Sequence-to-Vector (e.g. Sentiment Classification)", color=G1, fontsize=9.5, fontweight="bold")
    box(ax, 2, 2, 3.5, 2.2, color=G1, alpha=0.05, lw=1)
    for idx, x in enumerate([1.0, 2.0, 3.0]):
        node(ax, x, 2.0, r=0.25, color=G1, label="Cell")
        arrow(ax, x, 1.0, x, 1.75, color=TX)
        if idx < 2:
            arrow(ax, x+0.25, 2.0, x+0.75, 2.0, color=O1)
        else:
            arrow(ax, x, 2.25, x, 3.0, color=G1)
    ax.text(2.0, 0.8, "Inputs x(1) to x(T)", color=TX, ha="center", fontsize=8)
    ax.text(3.0, 3.2, "Final Output Y", color=G1, ha="center", fontsize=8)
    ax.set_xlim(0.2, 3.8); ax.set_ylim(0.5, 3.5)
    
    # 3. Vec-to-Seq
    ax = axes[1, 0]; ax.axis("off"); ax.set_title("Vector-to-Sequence (e.g. Image Captioning)", color=O1, fontsize=9.5, fontweight="bold")
    box(ax, 2, 2, 3.5, 2.2, color=O1, alpha=0.05, lw=1)
    for idx, x in enumerate([1.0, 2.0, 3.0]):
        node(ax, x, 2.0, r=0.25, color=O1, label="Cell")
        if idx == 0:
            arrow(ax, x, 1.0, x, 1.75, color=TX)
        arrow(ax, x, 2.25, x, 3.0, color=G1)
        if idx < 2:
            arrow(ax, x+0.25, 2.0, x+0.75, 2.0, color=O1)
    ax.text(1.0, 0.8, "Single Input X", color=TX, ha="center", fontsize=8)
    ax.text(2.0, 3.2, "Sequence y(1) to y(T)", color=G1, ha="center", fontsize=8)
    ax.set_xlim(0.2, 3.8); ax.set_ylim(0.5, 3.5)
    
    # 4. Encoder-Decoder
    ax = axes[1, 1]; ax.axis("off"); ax.set_title("Delayed Seq-to-Seq (e.g. Machine Translation)", color=P1, fontsize=9.5, fontweight="bold")
    box(ax, 2, 2, 3.5, 2.2, color=P1, alpha=0.05, lw=1)
    # Encoder
    for idx, x in enumerate([0.8, 1.6]):
        node(ax, x, 2.0, r=0.22, color=B1, label="Enc")
        arrow(ax, x, 1.0, x, 1.78, color=TX)
        arrow(ax, x+0.22, 2.0, x+0.58, 2.0, color=B1)
    # Context vector link
    arrow(ax, 2.18, 2.0, 2.42, 2.0, color=GOLD, lw=2)
    ax.text(2.3, 2.3, "Context", color=GOLD, ha="center", fontsize=7, fontweight="bold")
    # Decoder
    for idx, x in enumerate([2.6, 3.4]):
        node(ax, x, 2.0, r=0.22, color=P1, label="Dec")
        arrow(ax, x, 2.22, x, 3.0, color=G1)
        if idx < 1:
            arrow(ax, x+0.22, 2.0, x+0.58, 2.0, color=P1)
    ax.text(1.2, 0.8, "Source Seq", color=TX, ha="center", fontsize=8)
    ax.text(3.0, 3.2, "Target Seq", color=G1, ha="center", fontsize=8)
    ax.set_xlim(0.2, 3.8); ax.set_ylim(0.5, 3.5)
    
    plt.tight_layout()
    save("04_rnn_seq_types.png")

def plot_05_bptt():
    print("[05] Backpropagation Through Time")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Figure 15-5: Backpropagation Through Time (BPTT)", fontsize=11, fontweight="bold", color=TX)
    
    steps = [("t-1", 2.5), ("t", 5.0), ("t+1", 7.5)]
    
    for idx, (t, x_pos) in enumerate(steps):
        box(ax, x_pos, 2.5, 1.2, 0.8, color=B1, label=f"State h({t})")
        # Input
        arrow(ax, x_pos, 1.2, x_pos, 2.05, color=G1, lw=1.5)
        ax.text(x_pos, 1.0, f"x({t})", color=TX, ha="center")
        # Output
        arrow(ax, x_pos, 2.95, x_pos, 3.8, color=G1, lw=1.5)
        ax.text(x_pos, 4.0, f"y({t})", color=G1, ha="center")
        
        # Loss nodes
        node(ax, x_pos, 4.5, r=0.2, color=R1, label=f"L{idx+1}")
        arrow(ax, x_pos, 4.15, x_pos, 4.3, color=TX2, style="-")
        
        # Temporal forward connection
        if idx < len(steps) - 1:
            next_x = steps[idx+1][1]
            arrow(ax, x_pos + 0.65, 2.5, next_x - 0.65, 2.5, color=G1, lw=1.5)
            
            # Backpropagation gradients (dashed red)
            ax.annotate("", xy=(x_pos + 0.65, 2.35), xytext=(next_x - 0.65, 2.35),
                        arrowprops=dict(arrowstyle="-|>", color=R1, lw=1.5, ls="--"),
                        zorder=3)
            ax.text((x_pos+next_x)/2, 2.0, "dState", color=R1, ha="center", fontsize=7.5)
            
        # Downward gradient flow from Loss to Cells
        ax.annotate("", xy=(x_pos, 2.95), xytext=(x_pos, 4.3),
                    arrowprops=dict(arrowstyle="-|>", color=R1, lw=1.5, ls="--"),
                    zorder=3)
        ax.text(x_pos + 0.35, 3.4, "dLoss", color=R1, fontsize=7.5)
        
    save("05_bptt.png")

def plot_06_time_series_example():
    print("[06] Synthetic Time Series Plot")
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    fig.suptitle("Figure 15-6: Synthetic Univariate Time Series Examples", fontsize=12, fontweight="bold", color=TX)
    
    # Simple time series generator
    np.random.seed(42)
    def generate_time_series(batch_size, n_steps):
        freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
        time = np.linspace(0, 1, n_steps)
        series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
        series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
        series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
        return series[..., np.newaxis].astype(np.float32)
        
    series = generate_time_series(3, 50)
    time_steps = np.arange(50)
    colors = [B1, G1, O1]
    
    for i in range(3):
        ax = axes[i]
        ax.plot(time_steps, series[i, :, 0], color=colors[i], lw=2, label=f"Series {i+1}")
        ax.plot(49, series[i, 49, 0], "o", color=R1, markersize=8, label="Target to Forecast" if i==0 else "")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(True)
        ax.set_ylabel("Value")
        
    axes[2].set_xlabel("Time Steps (t)")
    plt.tight_layout()
    save("06_time_series_example.png")

def plot_07_deep_rnn_unrolled():
    print("[07] Deep RNN Unrolled")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 9); ax.set_ylim(0, 6); ax.axis("off")
    fig.suptitle("Figure 15-7: Deep Recurrent Neural Network Unrolled through Time", fontsize=11, fontweight="bold", color=TX)
    
    steps = [("t-1", 2.2), ("t", 4.7), ("t+1", 7.2)]
    layers = [("Layer 3", 4.5, P1), ("Layer 2", 3.0, G1), ("Layer 1", 1.5, B1)]
    
    for l_idx, (l_name, y_pos, col) in enumerate(layers):
        ax.text(0.5, y_pos, l_name, color=col, fontweight="bold", va="center")
        for s_idx, (step_lbl, x_pos) in enumerate(steps):
            box(ax, x_pos, y_pos, 1.0, 0.6, color=col, label=f"Cell {s_idx+1}\nL{3-l_idx}")
            
            # Feedforward upward connection
            if l_idx > 0:
                arrow(ax, x_pos, y_pos + 0.32, x_pos, y_pos + 0.88, color=col, lw=1.2)
            else:
                arrow(ax, x_pos, y_pos + 0.32, x_pos, y_pos + 0.9, color=GOLD, lw=1.5)
                if s_idx == 1:
                    ax.text(x_pos, y_pos + 1.1, f"y({step_lbl})", color=GOLD, ha="center", fontweight="bold")
                    
            if l_idx == len(layers) - 1:
                arrow(ax, x_pos, 0.4, x_pos, y_pos - 0.32, color=TX, lw=1.2)
                if s_idx == 1:
                    ax.text(x_pos, 0.15, f"x({step_lbl})", color=TX, ha="center", fontweight="bold")
                    
            # Temporal horizontal connections
            if s_idx < len(steps) - 1:
                next_x = steps[s_idx+1][1]
                arrow(ax, x_pos + 0.52, y_pos, next_x - 0.52, y_pos, color=O1, lw=1.5)
                
    save("07_deep_rnn_unrolled.png")

def plot_08_forecasting_ahead():
    print("[08] Forecasting Several Steps Ahead")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("Figure 15-8: Multi-Step Forecasting (Sequence-to-Sequence)", fontsize=11, fontweight="bold", color=TX)
    
    t_past = np.arange(40)
    t_future = np.arange(40, 50)
    
    np.random.seed(42)
    y_past = 0.5 * np.sin(t_past * 0.2) + 0.1 * np.random.randn(40)
    y_future_true = 0.5 * np.sin(t_future * 0.2) + 0.1 * np.random.randn(10)
    
    y_pred_step = y_future_true + 0.08 * np.random.randn(10) + np.linspace(0, 0.25, 10)
    y_pred_direct = y_future_true + 0.04 * np.random.randn(10)
    
    ax.plot(t_past, y_past, color=TX, lw=2, label="Historical Data (T=40)")
    ax.plot(t_future, y_future_true, color=G1, lw=2.5, label="Ground Truth Target (10 steps)")
    ax.plot(t_future, y_pred_step, color=R1, ls="--", marker="o", label="1-Step-Ahead (Drifting Errors)")
    ax.plot(t_future, y_pred_direct, color=B1, ls=":", marker="x", label="Seq-to-Seq Forecast (Dense Output Head)")
    
    ax.axvline(39.5, color="#555", ls="-.")
    ax.text(38.5, 0.4, "Forecast Start", color="#aaa", rotation=90, ha="right", va="center")
    
    ax.set_xlabel("Time Step (t)")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)
    
    save("08_forecasting_ahead.png")

def plot_09_lstm_cell():
    print("[09] Detailed LSTM Cell Block")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
    fig.suptitle("Figure 15-9: Detailed LSTM Cell Architecture & Gating", fontsize=12, fontweight="bold", color=TX)
    
    # Outer cell box boundary
    box(ax, 5.2, 3.1, 8.8, 4.8, color=P1, label="", alpha=0.03, lw=2)
    ax.text(5.2, 5.2, "LSTM CELL PIPELINE", color=P1, fontweight="bold", ha="center", fontsize=9.5)
    
    # State channels
    # C(t) line (orange)
    arrow(ax, 0.5, 4.4, 9.8, 4.4, color=O1, lw=2.5, style="-")
    ax.text(0.5, 4.7, "Long-term state\nC(t-1)", color=O1, ha="center", fontsize=8, fontweight="bold")
    ax.text(9.8, 4.7, "Long-term state\nC(t)", color=O1, ha="center", fontsize=8, fontweight="bold")
    
    # h(t) line (green)
    arrow(ax, 0.5, 1.2, 1.8, 1.2, color=G1, lw=2.2, style="-")
    ax.text(0.5, 0.8, "Short-term state\nh(t-1)", color=G1, ha="center", fontsize=8, fontweight="bold")
    
    # Input x(t)
    arrow(ax, 1.8, 0.5, 1.8, 1.2, color=TX, lw=2, style="-")
    ax.text(1.8, 0.25, "Input x(t)", color=TX, ha="center", fontsize=8.5, fontweight="bold")
    
    # Gating variables box definitions
    gates = [
        ("Forget Gate\n(f_t)", 3.2, 2.2, R1, r"$\sigma$", "f_t = sigmoid(...)"),
        ("Input Gate\n(i_t)", 4.4, 2.2, G1, r"$\sigma$", "i_t = sigmoid(...)"),
        ("Candidate\n(g_t)", 5.6, 2.2, B1, r"$\tanh$", "g_t = tanh(...)"),
        ("Output Gate\n(o_t)", 7.0, 2.2, GOLD, r"$\sigma$", "o_t = sigmoid(...)")
    ]
    
    # Merge h(t-1) and x(t) and fan out
    node(ax, 1.8, 1.2, r=0.12, color=TX2, label="")
    arrow(ax, 1.8, 1.2, 2.6, 1.8, color=TX, lw=1.2, style="-")
    arrow(ax, 2.6, 1.8, 7.0, 1.8, color=TX2, lw=1.2, style="-")
    
    # Draw dashed box enclosing the gate controller group
    ax.add_patch(Rectangle((2.6, 1.4), 4.9, 1.5, fill=False, edgecolor=TX2, ls=":", lw=1.2, alpha=0.6))
    ax.text(5.0, 1.5, "Gate Controllers", color=TX2, ha="center", fontsize=8, style="italic")
    
    for name, x, y, col, sym, formula in gates:
        box(ax, x, y, 0.9, 0.5, color=col, label=sym, fontsize=10)
        ax.text(x, y-0.45, name, color=col, ha="center", fontsize=7.5, fontweight="bold")
        arrow(ax, x, 1.8, x, 1.95, color=TX2, lw=1)
        
    # Operator Junctions
    # Forget Gate multiplier (f_t * C(t-1))
    node(ax, 3.2, 4.4, r=0.18, color=R1, label=r"$\otimes$")
    arrow(ax, 3.2, 2.45, 3.2, 4.22, color=R1, lw=1.5)
    ax.text(3.5, 4.1, r"$\otimes$ Forget", color=R1, fontsize=8)
    
    # Input multiplication (i_t * g_t)
    node(ax, 5.0, 3.3, r=0.18, color=B1, label=r"$\otimes$")
    arrow(ax, 4.4, 2.45, 5.0, 3.12, color=G1, lw=1.2)
    arrow(ax, 5.6, 2.45, 5.0, 3.12, color=B1, lw=1.2)
    
    # C(t) Addition (+ C_t-1)
    node(ax, 5.0, 4.4, r=0.18, color=O1, label=r"$\oplus$")
    arrow(ax, 5.0, 3.48, 5.0, 4.22, color=O1, lw=1.5)
    ax.text(5.3, 4.1, r"$\oplus$ Update", color=O1, fontsize=8)
    
    # Output path gating
    node(ax, 8.0, 4.4, r=0.12, color=GOLD)
    arrow(ax, 8.0, 4.4, 8.0, 3.5, color=GOLD, lw=1.2, style="-")
    box(ax, 8.0, 3.2, 0.7, 0.4, color=GOLD, label="tanh", fontsize=8)
    
    node(ax, 8.0, 2.2, r=0.18, color=GOLD, label=r"$\otimes$")
    arrow(ax, 7.45, 2.2, 7.82, 2.2, color=GOLD, lw=1.2)
    arrow(ax, 8.0, 3.0, 8.0, 2.38, color=GOLD, lw=1.2, style="-")
    
    # Output projection
    arrow(ax, 8.18, 2.2, 9.8, 2.2, color=G1, lw=2.5)
    ax.text(9.8, 1.8, "Short-term state\nh(t) / Output y(t)", color=G1, ha="center", fontsize=8, fontweight="bold")
    
    # Persistent loop pathway
    arrow(ax, 9.2, 2.2, 9.2, 0.7, color=TX2, lw=1, style="-")
    arrow(ax, 9.2, 0.7, 0.5, 0.7, color=TX2, lw=1, style="-")
    
    save("09_lstm_cell.png")

def plot_10_gru_cell():
    print("[10] Detailed GRU Cell Block")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
    fig.suptitle("Figure 15-10: Detailed GRU Cell Architecture & Gating", fontsize=12, fontweight="bold", color=TX)
    
    # Outer cell boundary
    box(ax, 5.2, 2.6, 8.8, 3.8, color=G1, label="", alpha=0.03, lw=2)
    ax.text(5.2, 4.2, "GRU CELL PIPELINE", color=G1, fontweight="bold", ha="center", fontsize=9.5)
    
    # State path
    arrow(ax, 0.5, 3.4, 9.8, 3.4, color=G1, lw=2.5, style="-")
    ax.text(0.5, 3.7, "State h(t-1)", color=G1, ha="center", fontsize=8, fontweight="bold")
    ax.text(9.8, 3.7, "State h(t)", color=G1, ha="center", fontsize=8, fontweight="bold")
    
    # Input x(t)
    arrow(ax, 1.8, 0.5, 1.8, 1.2, color=TX, lw=2, style="-")
    ax.text(1.8, 0.25, "Input x(t)", color=TX, ha="center", fontsize=8.5, fontweight="bold")
    
    # Merge input and h(t-1)
    arrow(ax, 0.5, 1.2, 1.8, 1.2, color=TX2, lw=1, style="-")
    node(ax, 1.8, 1.2, r=0.12, color=TX2)
    arrow(ax, 1.8, 1.2, 2.5, 1.8, color=TX2, lw=1, style="-")
    
    # Dashed box for Gate Controllers
    ax.add_patch(Rectangle((2.6, 1.4), 2.8, 1.3, fill=False, edgecolor=TX2, ls=":", lw=1.2, alpha=0.6))
    ax.text(4.0, 1.5, "Gate Controllers", color=TX2, ha="center", fontsize=8, style="italic")
    
    # Reset Gate, Update Gate
    box(ax, 3.2, 2.0, 0.8, 0.5, color=R1, label=r"$\sigma$")
    ax.text(3.2, 2.45, "Reset (r_t)", color=R1, ha="center", fontsize=7.5, fontweight="bold")
    
    box(ax, 4.6, 2.0, 0.8, 0.5, color=B1, label=r"$\sigma$")
    ax.text(4.6, 2.45, "Update (z_t)", color=B1, ha="center", fontsize=7.5, fontweight="bold")
    
    # Connect input line to gates
    arrow(ax, 2.5, 1.8, 4.6, 1.8, color=TX2, lw=1, style="-")
    
    # Gating the state: reset gate multiplier
    node(ax, 3.2, 3.4, r=0.18, color=R1, label=r"$\otimes$")
    arrow(ax, 3.2, 2.25, 3.2, 3.22, color=R1, lw=1.2)
    
    # Candidate activation
    box(ax, 6.0, 2.3, 0.9, 0.5, color=GOLD, label=r"$\tanh$")
    ax.text(6.0, 1.8, "Candidate (g_t)", color=GOLD, ha="center", fontsize=7.5, fontweight="bold")
    arrow(ax, 3.38, 3.4, 6.0, 2.55, color=TX2, lw=1)
    
    # Update gate split (1 - z_t)
    box(ax, 4.6, 2.9, 0.6, 0.3, color=B1, label="1 - z_t", fontsize=7.5)
    arrow(ax, 4.6, 2.25, 4.6, 2.75, color=B1, lw=1)
    
    # Merge junctions on h(t) state
    # (z_t * h(t-1))
    node(ax, 7.0, 3.4, r=0.18, color=B1, label=r"$\otimes$")
    arrow(ax, 4.6, 3.05, 7.0, 3.22, color=B1, lw=1.2)
    
    # ((1-z_t) * g_t)
    node(ax, 8.0, 3.4, r=0.18, color=GOLD, label=r"$\otimes$")
    arrow(ax, 6.0, 2.55, 8.0, 3.22, color=GOLD, lw=1.2)
    
    # Output Adder
    node(ax, 8.7, 3.4, r=0.18, color=G1, label=r"$\oplus$")
    arrow(ax, 7.18, 3.4, 8.52, 3.4, color=TX2, lw=1)
    arrow(ax, 8.18, 3.4, 8.52, 3.4, color=TX2, lw=1)
    
    save("10_gru_cell.png")

def plot_11_wavenet_architecture():
    print("[11] Detailed WaveNet Dilated Convolutions")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    fig.suptitle("Figure 15-11: Detailed WaveNet Stacked Dilated 1D Convolutions", fontsize=12, fontweight="bold", color=TX)
    
    layer_configs = [
        ("Output Layer", 4.2, GOLD, 1),
        ("Dilated Layer (d=4)", 3.1, P1, 4),
        ("Dilated Layer (d=2)", 2.0, B1, 2),
        ("Causal Layer (d=1)", 0.9, G1, 1)
    ]
    
    xs = np.linspace(1.5, 8.5, 12)
    
    # Receptive Field shaded triangle (red) at last node
    # Last node is at xs[-1] = 8.5, y=4.2
    # Its inputs dilate back:
    # y=4.2 (idx 11) -> y=3.1 (idx 11 and idx 7)
    # y=3.1 (idx 11) -> y=2.0 (idx 11 and idx 9); y=3.1 (idx 7) -> y=2.0 (idx 7 and idx 5)
    # y=2.0 -> y=0.9 -> inputs.
    # Receptive field span covers steps from index 4 to 11 at inputs (8 steps).
    poly_x = [xs[-1], xs[-1 - 7], xs[-1]]
    poly_y = [4.2, 0.3, 0.3]
    ax.fill(poly_x, poly_y, color=R1, alpha=0.1, zorder=1)
    
    # Draw nodes
    for name, y, col, d in layer_configs:
        ax.text(0.2, y, name, color=col, fontweight="bold", va="center", fontsize=8)
        for idx, x in enumerate(xs):
            node(ax, x, y, r=0.12, color=col, label="")
            
    # Connections
    # Layer 0 (Inputs) to Causal Layer (d=1)
    for idx, x in enumerate(xs):
        arrow(ax, x, 0.3, x, 0.78, color=TX2, lw=0.8, style="-")
        if idx >= 1:
            arrow(ax, xs[idx-1], 0.3, x, 0.78, color=G1, lw=1)
            
    # Layer 1 to Layer 2 (d=2)
    for idx, x in enumerate(xs):
        arrow(ax, x, 1.02, x, 1.88, color=TX2, lw=0.8, style="-")
        if idx >= 2:
            arrow(ax, xs[idx-2], 1.02, x, 1.88, color=B1, lw=1.2)
            
    # Layer 2 to Layer 3 (d=4)
    for idx, x in enumerate(xs):
        arrow(ax, x, 2.12, x, 2.98, color=TX2, lw=0.8, style="-")
        if idx >= 4:
            arrow(ax, xs[idx-4], 2.12, x, 2.98, color=P1, lw=1.2)
            
    # Layer 3 to Layer 4 (Output d=1)
    for idx, x in enumerate(xs):
        arrow(ax, x, 3.22, x, 4.08, color=TX2, lw=0.8, style="-")
        if idx >= 1:
            arrow(ax, xs[idx-1], 3.22, x, 4.08, color=GOLD, lw=1)
            
    # Receptive Field boundary lines
    ax.plot([xs[-1], xs[-1 - 7]], [4.2, 0.3], color=R1, ls="--", lw=1.5, zorder=2)
    ax.plot([xs[-1], xs[-1]], [4.2, 0.3], color=R1, ls="--", lw=1.5, zorder=2)
    ax.text(xs[-1]-1.5, 4.5, "Receptive Field Cone\n(Expands exponentially with depth: 2^L steps)", color=R1, ha="center", fontsize=8, fontweight="bold")
    
    save("11_wavenet_architecture.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🎨 Generating all Chapter 15 Visuals under Visuals/ directory...")
    plot_01_recurrent_neuron_unrolled()
    plot_02_recurrent_layer_unrolled()
    plot_03_hidden_state_vs_output()
    plot_04_rnn_seq_types()
    plot_05_bptt()
    plot_06_time_series_example()
    plot_07_deep_rnn_unrolled()
    plot_08_forecasting_ahead()
    plot_09_lstm_cell()
    plot_10_gru_cell()
    plot_11_wavenet_architecture()
    print("🎉 All 11 visuals created successfully.")
