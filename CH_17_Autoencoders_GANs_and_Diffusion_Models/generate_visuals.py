"""
generate_visuals.py
CH 17: Autoencoders, GANs, and Diffusion Models
─────────────────────────────────────────────────
Generates 24 accurate matplotlib visuals for Chapter 17 study notes.
Run: python3 generate_visuals.py
All PNGs saved to ./Visuals/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle
from scipy.stats import norm
import os

# ─────────────────────────────────────────────────────────────────────
# GLOBAL DARK THEME
# ─────────────────────────────────────────────────────────────────────
DARK_BG       = "#0d1117"
PANEL_BG      = "#161b22"
BORDER_COLOR  = "#30363d"
ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_PURPLE = "#bc8cff"
ACCENT_ORANGE = "#f78166"
ACCENT_YELLOW = "#e3b341"
ACCENT_CYAN   = "#39d0d8"
ACCENT_PINK   = "#f778ba"
TEXT_COLOR    = "#e6edf3"
MUTED_TEXT    = "#8b949e"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    BORDER_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       MUTED_TEXT,
    "ytick.color":       MUTED_TEXT,
    "grid.color":        BORDER_COLOR,
    "grid.alpha":        0.5,
    "text.color":        TEXT_COLOR,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  BORDER_COLOR,
    "legend.labelcolor": TEXT_COLOR,
    "font.family":       "monospace",
    "font.size":         11,
})

os.makedirs("Visuals", exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# MODULE 01 — BASIC AUTOENCODERS
# ═════════════════════════════════════════════════════════════════════

def graph_01_autoencoder_architecture():
    """Full encoder-decoder flow with layer dims, loss formula, and data funnel."""
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Stacked Autoencoder: Architecture & Data Flow",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR, pad=18)

    # layer specs: (x_center, height, label_top, label_bot, color)
    layers = [
        (1.3,  5.5, "Input", "x\n(784)", ACCENT_BLUE),
        (3.5,  3.8, "Dense", "150\nSELU",  ACCENT_GREEN),
        (5.5,  2.8, "Dense", "100\nSELU",  ACCENT_GREEN),
        (7.5,  1.2, "Bottleneck", "z=30\nSELU", ACCENT_YELLOW),
        (9.5,  2.8, "Dense", "100\nSELU",  ACCENT_PURPLE),
        (11.5, 3.8, "Dense", "150\nSELU",  ACCENT_PURPLE),
        (13.7, 5.5, "Output", "x̂\n(784)", ACCENT_ORANGE),
    ]

    box_w = 0.9
    prev_x = None
    prev_h = None
    for (cx, h, ltop, lbot, color) in layers:
        # Draw bar
        rect = FancyBboxPatch((cx - box_w/2, 4 - h/2), box_w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=color + "28", edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(cx, 4 + h/2 + 0.35, ltop, ha="center", va="bottom",
                fontsize=8.5, color=color, fontweight="bold")
        ax.text(cx, 4, lbot, ha="center", va="center",
                fontsize=8, color=TEXT_COLOR)
        if prev_x is not None:
            ax.annotate("", xy=(cx - box_w/2, 4),
                        xytext=(prev_x + box_w/2, 4),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED_TEXT, lw=1.8))
        prev_x = cx; prev_h = h

    # Encoder / Decoder brace
    ax.annotate("", xy=(7.0, 1.2), xytext=(1.0, 1.2),
                arrowprops=dict(arrowstyle="<->", color=ACCENT_GREEN, lw=2.2))
    ax.text(4.0, 0.75, "ENCODER  f(x)", ha="center", fontsize=11,
            color=ACCENT_GREEN, fontweight="bold")
    ax.annotate("", xy=(14.2, 1.2), xytext=(8.0, 1.2),
                arrowprops=dict(arrowstyle="<->", color=ACCENT_PURPLE, lw=2.2))
    ax.text(11.1, 0.75, "DECODER  g(z)", ha="center", fontsize=11,
            color=ACCENT_PURPLE, fontweight="bold")

    # Loss box
    ax.text(7.5, 7.2,
            r"$\mathcal{L} = \|x - \hat{x}\|^2$   (MSE)   or   $-\sum x_i \log \hat{x}_i$   (BCE)",
            ha="center", fontsize=13, color=ACCENT_YELLOW, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_BG,
                      edgecolor=ACCENT_YELLOW, alpha=0.9))

    plt.tight_layout()
    plt.savefig("Visuals/01_autoencoder_architecture.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [01] Autoencoder architecture")


def graph_02_reconstruction_loss_curve():
    """Training vs validation reconstruction loss for a stacked AE on MNIST."""
    np.random.seed(1)
    epochs = np.arange(1, 51)
    train_loss = 0.38 * np.exp(-0.09 * epochs) + 0.265 + np.random.normal(0, 0.004, 50)
    val_loss   = 0.38 * np.exp(-0.08 * epochs) + 0.278 + np.random.normal(0, 0.005, 50)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_loss, color=ACCENT_BLUE,  lw=2.5, label="Train Loss (BCE)")
    ax.plot(epochs, val_loss,   color=ACCENT_ORANGE, lw=2.5, label="Val Loss (BCE)", linestyle="--")
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.12, color=ACCENT_ORANGE)

    ax.axhline(0.265, color=ACCENT_GREEN, lw=1.5, linestyle=":", label="Converged train ≈ 0.265")
    ax.annotate("Overfitting gap\n(small = good)", xy=(45, (train_loss[-1]+val_loss[-1])/2),
                xytext=(35, 0.32), fontsize=9, color=ACCENT_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=1.2))

    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Binary Cross-Entropy", fontsize=12)
    ax.set_title("AE Reconstruction Loss: Train vs Validation (MNIST, bottleneck=30)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("Visuals/02_reconstruction_loss_curve.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [02] Reconstruction loss curve")


def graph_03_undercomplete_overcomplete():
    """Side-by-side funnel diagrams with capacity annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Undercomplete vs. Overcomplete Autoencoders",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR)

    configs = [
        {
            "title": "Undercomplete AE\n(Bottleneck < Input)",
            "dims": [784, 300, 30, 300, 784],
            "annotation": "30-dim bottleneck FORCES\nmeaningful compression",
            "acolor": ACCENT_GREEN, "note_col": ACCENT_GREEN,
        },
        {
            "title": "Overcomplete AE (No Reg.)\n(Bottleneck >= Input)",
            "dims": [784, 300, 900, 300, 784],
            "annotation": "900-dim bottleneck allows\ntrivial identity mapping!",
            "acolor": ACCENT_ORANGE, "note_col": ACCENT_ORANGE,
        },
    ]

    for ax, cfg in zip(axes, configs):
        ax.set_facecolor(PANEL_BG); ax.set_xlim(0, 10); ax.set_ylim(0, 8)
        ax.axis("off"); ax.set_title(cfg["title"], fontsize=13, color=TEXT_COLOR,
                                     fontweight="bold", pad=10)
        xs = np.linspace(1.2, 8.8, 5)
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_YELLOW, ACCENT_PURPLE, ACCENT_ORANGE]
        for x, dim, color in zip(xs, cfg["dims"], colors):
            h = max(0.35, 4.8 * dim / 900)
            rect = FancyBboxPatch((x - 0.42, 4 - h/2), 0.84, h,
                                   boxstyle="round,pad=0.07",
                                   facecolor=color+"2a", edgecolor=color, lw=2.2)
            ax.add_patch(rect)
            ax.text(x, 4 + h/2 + 0.28, str(dim), ha="center",
                    fontsize=9, color=color, fontweight="bold")
        for i in range(4):
            ax.annotate("", xy=(xs[i+1] - 0.42, 4), xytext=(xs[i] + 0.42, 4),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED_TEXT, lw=1.5))
        ax.text(5, 0.9, cfg["annotation"], ha="center", fontsize=10.5,
                color=cfg["note_col"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG,
                          edgecolor=cfg["note_col"], alpha=0.9))

    plt.tight_layout()
    plt.savefig("Visuals/03_undercomplete_overcomplete.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [03] Undercomplete vs overcomplete")


def graph_04_pca_vs_ae_latent():
    """2D scatter: PCA (linear) vs AE (non-linear) latent space on a Swiss Roll."""
    np.random.seed(42)
    n = 800
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    height = 10 * np.random.rand(n)
    X = np.stack([t * np.cos(t), height, t * np.sin(t)], axis=1)
    colors_data = plt.cm.plasma((t - t.min()) / (t.max() - t.min()))

    # PCA: project to first 2 PCs
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pca_2d = X_centered @ Vt[:2].T

    # AE "latent" — simulate as unrolled manifold (t, height)
    ae_2d = np.stack([t, height], axis=1) + np.random.normal(0, 0.3, (n, 2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("PCA (Linear) vs Autoencoder (Non-Linear) Latent Space — Swiss Roll",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    for ax, z, title, note in zip(
        axes,
        [pca_2d, ae_2d],
        ["PCA — 2 Principal Components", "AE Latent — 2D Bottleneck"],
        ["Manifold still curled\n→ global structure lost", "Manifold unrolled\n→ smooth, continuous code"],
    ):
        ax.scatter(z[:, 0], z[:, 1], c=colors_data, s=12, alpha=0.8)
        ax.set_title(title, fontsize=12, color=TEXT_COLOR, fontweight="bold")
        ax.set_xlabel("Dim 1", fontsize=11); ax.set_ylabel("Dim 2", fontsize=11)
        ax.grid(True, alpha=0.3); ax.tick_params(colors=MUTED_TEXT)
        c = ACCENT_ORANGE if "curled" in note else ACCENT_GREEN
        ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG,
                          edgecolor=c, alpha=0.9))

    sm = plt.cm.ScalarMappable(cmap="plasma")
    sm.set_array(t)
    cbar = plt.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Roll Position", color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=MUTED_TEXT)

    plt.tight_layout()
    plt.savefig("Visuals/04_pca_vs_ae_latent.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [04] PCA vs AE latent space")


# ═════════════════════════════════════════════════════════════════════
# MODULE 02 — SPARSE & DENOISING AUTOENCODERS
# ═════════════════════════════════════════════════════════════════════

def graph_05_sparse_activation_histogram():
    """Accurate histogram: standard AE vs L1-sparse AE activation distributions."""
    np.random.seed(7)
    std_act  = np.abs(np.random.normal(0.45, 0.22, 8000)); std_act = np.clip(std_act, 0, 1)
    l1_act   = np.concatenate([np.zeros(6800),
                               np.abs(np.random.normal(0.72, 0.12, 1200))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Bottleneck Activation Distribution: Standard vs. Sparse AE",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    for ax, act, label, color, note in zip(
        axes,
        [std_act, l1_act],
        ["Standard AE (no regularization)", "Sparse AE (L1 activity_regularizer=1e-4)"],
        [ACCENT_BLUE, ACCENT_ORANGE],
        ["Dense code — every neuron carries info", "Sparse code — 85% neurons ~0\n-> feature detector"],
    ):
        ax.hist(act, bins=45, color=color, alpha=0.82, edgecolor=DARK_BG, linewidth=0.5)
        ax.axvline(np.mean(act), color=ACCENT_YELLOW, lw=2, linestyle="--",
                   label=f"Mean = {np.mean(act):.3f}")
        ax.set_xlabel("Activation Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(label, fontsize=11.5, color=color, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        ax.tick_params(colors=MUTED_TEXT)
        ax.text(0.97, 0.94, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG,
                          edgecolor=color, alpha=0.9))

    plt.tight_layout()
    plt.savefig("Visuals/05_sparse_activation_histogram.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [05] Sparse activation histogram")


def graph_06_kl_sparsity_penalty():
    """Plot KL-divergence sparsity penalty vs actual activation for rho=0.1."""
    rho = 0.1
    rho_hat = np.linspace(0.001, 0.999, 500)
    kl = rho * np.log(rho / rho_hat) + (1 - rho) * np.log((1 - rho) / (1 - rho_hat))
    l1 = rho_hat

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sparsity Penalty Functions: KL-Divergence (rho=0.1) vs L1",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    # KL plot
    axes[0].plot(rho_hat, kl, color=ACCENT_BLUE, lw=2.5)
    axes[0].axvline(rho, color=ACCENT_YELLOW, lw=2, linestyle="--",
                    label=f"Target rho = {rho}")
    axes[0].fill_between(rho_hat, 0, kl, alpha=0.15, color=ACCENT_BLUE)
    axes[0].set_xlabel("Average activation rho_hat", fontsize=11)
    axes[0].set_ylabel("KL Divergence Penalty", fontsize=11)
    axes[0].set_title("KL Divergence Sparsity\nKL(rho || rho_hat)", fontsize=12,
                      color=ACCENT_BLUE, fontweight="bold")
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.4)
    axes[0].set_ylim(0, 3); axes[0].tick_params(colors=MUTED_TEXT)
    axes[0].annotate("Minimum at rho_hat = rho\n(target activation reached)",
                     xy=(rho, 0), xytext=(0.35, 1.5), fontsize=9, color=ACCENT_YELLOW,
                     arrowprops=dict(arrowstyle="->", color=ACCENT_YELLOW, lw=1.5))

    # L1 plot
    axes[1].plot(rho_hat, l1, color=ACCENT_ORANGE, lw=2.5)
    axes[1].fill_between(rho_hat, 0, l1, alpha=0.15, color=ACCENT_ORANGE)
    axes[1].set_xlabel("Activation value |z_j|", fontsize=11)
    axes[1].set_ylabel("L1 Penalty", fontsize=11)
    axes[1].set_title("L1 Activity Regularization\nomega = lambda * sum|z_j|",
                      fontsize=12, color=ACCENT_ORANGE, fontweight="bold")
    axes[1].grid(True, alpha=0.4); axes[1].tick_params(colors=MUTED_TEXT)
    axes[1].text(0.5, 0.35, "Linear — pushes all\nactivations toward 0",
                 transform=axes[1].transAxes, ha="center", fontsize=10,
                 color=ACCENT_ORANGE, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG,
                           edgecolor=ACCENT_ORANGE, alpha=0.9))

    plt.tight_layout()
    plt.savefig("Visuals/06_kl_sparsity_penalty.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [06] KL sparsity penalty curves")


def graph_07_denoising_ae_pipeline():
    """Grid: original | noisy (sigma=0.2) | noisy (sigma=0.5) | reconstructed."""
    np.random.seed(12)
    n = 6

    def make_digit(kind, size=20):
        img = np.zeros((size, size))
        if kind == "bar":
            img[3:17, 8:12] = 1.0; img[3:6, 6:14] = 1.0
        elif kind == "oval":
            for r in range(4, 16):
                for c in range(4, 16):
                    if (r-10)**2/25 + (c-10)**2/16 < 1: img[r, c] = 1.0
            img[8:12, 8:12] = 0.0
        else:
            img[3:6, 4:16] = 1.0; img[3:10, 4:7] = 1.0
            img[8:12, 4:16] = 1.0; img[12:17, 4:7] = 1.0; img[14:17, 4:16] = 1.0
        return img

    kinds = ["bar", "oval", "other", "bar", "oval", "other"]
    originals = [make_digit(k) for k in kinds]
    noisy_lo  = [np.clip(img + np.random.normal(0, 0.25, img.shape), 0, 1) for img in originals]
    noisy_hi  = [np.clip(img + np.random.normal(0, 0.55, img.shape), 0, 1) for img in originals]
    recon     = [np.clip(img + np.random.normal(0, 0.06, img.shape), 0, 1) for img in originals]

    row_labels = ["Original x", "Noisy (σ=0.25)", "Noisy (σ=0.55)", "Reconstructed x̂"]
    row_colors = [ACCENT_BLUE, ACCENT_YELLOW, ACCENT_ORANGE, ACCENT_GREEN]
    row_cmaps  = ["Blues", "YlOrBr", "Reds", "Greens"]
    row_imgs   = [originals, noisy_lo, noisy_hi, recon]

    fig, axes = plt.subplots(4, n, figsize=(15, 9))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Denoising Autoencoder: Input Corruption → Reconstruction",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.99)

    for row, (imgs, label, color, cmap) in enumerate(
        zip(row_imgs, row_labels, row_colors, row_cmaps)
    ):
        axes[row, 0].set_ylabel(label, color=color, fontsize=10, fontweight="bold",
                                rotation=90, labelpad=5)
        for col, img in enumerate(imgs):
            axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, col].axis("off")
            for sp in axes[row, col].spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(1.8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("Visuals/07_denoising_ae_pipeline.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [07] Denoising AE pipeline")


def graph_08_anomaly_detection():
    """Reconstruction error distribution: normal vs anomalous samples."""
    np.random.seed(5)
    normal_err = np.random.lognormal(mean=-1.5, sigma=0.4, size=2000)
    anomaly_err = np.random.lognormal(mean=0.4,  sigma=0.5, size=300)
    threshold  = np.percentile(normal_err, 99)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.hist(normal_err,  bins=60, color=ACCENT_BLUE,   alpha=0.78,
            label="Normal samples (training distribution)", edgecolor=DARK_BG, lw=0.4)
    ax.hist(anomaly_err, bins=40, color=ACCENT_ORANGE,  alpha=0.78,
            label="Anomalous samples (out-of-distribution)", edgecolor=DARK_BG, lw=0.4)
    ax.axvline(threshold, color=ACCENT_YELLOW, lw=2.5, linestyle="--",
               label=f"Detection threshold (99th pct = {threshold:.3f})")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 500],
                     threshold, normal_err.max() + 0.5,
                     alpha=0.08, color=ACCENT_ORANGE)

    ax.set_xlabel("Reconstruction Loss (BCE)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Anomaly Detection via Reconstruction Error\n"
                 "AE trained only on normal data; anomalies have high reconstruction error",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.35); ax.tick_params(colors=MUTED_TEXT)
    ax.annotate("Anomaly region\n(flag as anomalous)",
                xy=(threshold + 0.05, 80), xytext=(threshold + 0.35, 200),
                fontsize=9, color=ACCENT_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=1.5))
    plt.tight_layout()
    plt.savefig("Visuals/08_anomaly_detection.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [08] Anomaly detection distribution")


# ═════════════════════════════════════════════════════════════════════
# MODULE 03 — VARIATIONAL AUTOENCODERS
# ═════════════════════════════════════════════════════════════════════

def graph_09_reparameterization_trick():
    """Visual proof of the reparameterization trick: stochastic node vs deterministic path."""
    fig = plt.figure(figsize=(15, 7))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Reparameterization Trick: Making Sampling Differentiable",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR)

    # Left: naive sampling (no gradient)
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor(DARK_BG); ax1.set_xlim(0, 10); ax1.set_ylim(0, 8)
    ax1.axis("off"); ax1.set_title("Naive Sampling (BROKEN gradient)", fontsize=12,
                                    color=ACCENT_ORANGE, fontweight="bold")

    for (x, y, label, color, style) in [
        (2, 6.5, "Input x",        ACCENT_BLUE,   "solid"),
        (5, 6.5, "Encoder\nmu, sigma", ACCENT_GREEN, "solid"),
        (8, 6.5, "z ~ N(mu,sigma)", ACCENT_YELLOW, "dashed"),
        (8, 3.5, "Decoder\ng(z)", ACCENT_PURPLE, "solid"),
        (5, 1.5, "Loss L", ACCENT_ORANGE, "solid"),
    ]:
        lw = 3 if style == "solid" else 2
        ec = color if style == "solid" else ACCENT_ORANGE
        rect = FancyBboxPatch((x-1, y-0.5), 2, 1.0,
                               boxstyle="round,pad=0.1",
                               facecolor=color+"22", edgecolor=ec, linewidth=lw,
                               linestyle=style)
        ax1.add_patch(rect)
        ax1.text(x, y, label, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    for (x1, y1, x2, y2, broken) in [
        (4, 6.5, 7, 6.5, False),
        (8, 6.0, 8, 4.0, True),   # broken gradient
        (7, 3.5, 6, 1.5, False),
    ]:
        c = ACCENT_ORANGE if broken else MUTED_TEXT
        ls = "dashed" if broken else "solid"
        ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=2, linestyle=ls))
        if broken:
            mid_x = (x1+x2)/2; mid_y = (y1+y2)/2
            ax1.text(mid_x + 0.3, mid_y, "BLOCKED\n(stochastic!)",
                     fontsize=8.5, color=ACCENT_ORANGE, fontweight="bold")

    # Right: reparameterized (gradient flows)
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor(DARK_BG); ax2.set_xlim(0, 10); ax2.set_ylim(0, 8)
    ax2.axis("off"); ax2.set_title("Reparameterized (GRADIENT FLOWS)", fontsize=12,
                                    color=ACCENT_GREEN, fontweight="bold")

    nodes = [
        (2, 6.5, "Input x",         ACCENT_BLUE,   "solid"),
        (5, 6.5, "Encoder\nmu,log_var", ACCENT_GREEN, "solid"),
        (8, 6.5, "z=mu+eps*sigma",  ACCENT_YELLOW, "solid"),
        (8, 3.5, "Decoder g(z)",    ACCENT_PURPLE, "solid"),
        (5, 1.5, "Loss L",          ACCENT_GREEN,  "solid"),
        (5, 5.0, "eps~N(0,I)",      ACCENT_CYAN,   "dashed"),
    ]
    for (x, y, label, color, style) in nodes:
        lw = 2.5 if style == "solid" else 1.8
        rect = FancyBboxPatch((x-1, y-0.5), 2, 1.0,
                               boxstyle="round,pad=0.1",
                               facecolor=color+"22", edgecolor=color, linewidth=lw,
                               linestyle=style)
        ax2.add_patch(rect)
        ax2.text(x, y, label, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    for (x1, y1, x2, y2, color) in [
        (4, 6.5, 7, 6.5, MUTED_TEXT),
        (6, 5.0, 7, 6.3, ACCENT_CYAN),   # eps input
        (8, 6.0, 8, 4.0, ACCENT_GREEN),  # gradient flows!
        (7, 3.5, 6, 1.5, ACCENT_GREEN),
        (4, 1.5, 4, 6.0, ACCENT_GREEN),  # backprop
    ]:
        ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))

    ax2.text(3.2, 3.8, "Backprop\nGradient", ha="center", fontsize=8,
             color=ACCENT_GREEN, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL_BG,
                       edgecolor=ACCENT_GREEN, alpha=0.85))
    ax2.text(8.3, 5.2, "z=mu+eps*sigma\n(differentiable\nw.r.t. mu, sigma)",
             ha="left", fontsize=8, color=ACCENT_YELLOW, fontweight="bold")

    plt.tight_layout()
    plt.savefig("Visuals/09_reparameterization_trick.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [09] Reparameterization trick")


def graph_10_kl_divergence_prior():
    """Plot encoder distribution N(mu,sigma) vs prior N(0,1), showing KL divergence area."""
    x = np.linspace(-5, 5, 500)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("KL Divergence: Encoder Posterior vs Standard Normal Prior",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    cases = [
        {"mu": 2.5, "sigma": 1.2, "label": "Early training\n(far from prior)"},
        {"mu": 1.0, "sigma": 0.8, "label": "Mid training\n(converging)"},
        {"mu": 0.1, "sigma": 1.05, "label": "Converged\n(matches prior)"},
    ]

    for ax, case in zip(axes, cases):
        prior = norm.pdf(x, 0, 1)
        posterior = norm.pdf(x, case["mu"], case["sigma"])
        kl = np.sum(posterior * np.log(posterior / (prior + 1e-10))) * (x[1] - x[0])
        kl = max(0, kl)

        ax.plot(x, prior,     color=ACCENT_BLUE,   lw=2.5, label="Prior p(z) = N(0,1)")
        ax.plot(x, posterior, color=ACCENT_ORANGE,  lw=2.5,
                label=f"q(z|x) = N({case['mu']},{case['sigma']})")
        ax.fill_between(x, 0, np.minimum(prior, posterior), alpha=0.25, color=ACCENT_GREEN)
        ax.fill_between(x, np.minimum(prior, posterior), np.maximum(prior, posterior),
                        alpha=0.22, color=ACCENT_ORANGE)

        ax.set_xlabel("z", fontsize=11); ax.set_ylabel("Density", fontsize=11)
        ax.set_title(case["label"], fontsize=11, color=TEXT_COLOR, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left"); ax.grid(True, alpha=0.3)
        ax.tick_params(colors=MUTED_TEXT); ax.set_ylim(0, 0.55)

        kl_color = ACCENT_ORANGE if kl > 0.5 else (ACCENT_YELLOW if kl > 0.1 else ACCENT_GREEN)
        ax.text(0.97, 0.95, f"KL = {kl:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=kl_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG,
                          edgecolor=kl_color, alpha=0.9))

    plt.tight_layout()
    plt.savefig("Visuals/10_kl_divergence_prior.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [10] KL divergence vs prior")


def graph_11_vae_elbo_loss():
    """ELBO decomposition: reconstruction + KL over training epochs."""
    np.random.seed(3)
    epochs = np.arange(1, 51)
    recon = 0.52 * np.exp(-0.065 * epochs) + 0.28 + np.random.normal(0, 0.004, 50)
    kl    = 0.035 * (1 - np.exp(-0.15 * epochs)) + np.random.normal(0, 0.0015, 50)
    total = recon + kl

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(epochs, total, color=ACCENT_GREEN,  lw=3.0, label="Total ELBO Loss", zorder=3)
    ax.plot(epochs, recon, color=ACCENT_BLUE,   lw=2.5, label="Reconstruction Loss (BCE)")
    ax.plot(epochs, kl,    color=ACCENT_ORANGE,  lw=2.5, label="KL Divergence Loss")
    ax.fill_between(epochs, recon, total, alpha=0.18, color=ACCENT_ORANGE, label="KL contribution")
    ax.fill_between(epochs, 0, recon, alpha=0.10, color=ACCENT_BLUE)

    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("VAE ELBO Loss Decomposition: ELBO = Reconstruction + KL",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10.5); ax.grid(True, alpha=0.4); ax.tick_params(colors=MUTED_TEXT)

    ax.annotate("KL rises: encoder pushes\nq(z|x) toward N(0,I)",
                xy=(20, kl[19]), xytext=(28, 0.025),
                fontsize=9, color=ACCENT_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE, lw=1.5))
    ax.annotate("Reconstruction falls:\ndecoder learns data structure",
                xy=(15, recon[14]), xytext=(22, 0.52),
                fontsize=9, color=ACCENT_BLUE,
                arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=1.5))

    plt.tight_layout()
    plt.savefig("Visuals/11_vae_elbo_loss.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [11] VAE ELBO loss decomposition")


def graph_12_vae_latent_manifold():
    """2D scatter of VAE latent means coloured by digit class (simulated clusters)."""
    np.random.seed(42)
    n_cls = 10
    cmap  = plt.cm.tab10

    # Clusters arranged on a circle (VAE pushes all toward N(0,I) center)
    centers = [(3.5*np.cos(2*np.pi*i/n_cls), 3.5*np.sin(2*np.pi*i/n_cls))
               for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(10, 9))
    handles = []
    for cls, (cx, cy) in enumerate(centers):
        pts = np.random.multivariate_normal([cx, cy],
                                            [[0.65, 0.05], [0.05, 0.65]], 200)
        ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(cls/n_cls)], alpha=0.72, s=18, zorder=2)
        handles.append(mpatches.Patch(color=cmap(cls/n_cls), label=f"Digit {cls}"))

    # Draw 1-sigma circle of prior N(0,I)
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=MUTED_TEXT, lw=1.5,
            linestyle="--", label="1-sigma prior N(0,I)", zorder=1)
    ax.plot(2*np.cos(theta), 2*np.sin(theta), color=MUTED_TEXT, lw=1.0,
            linestyle=":", label="2-sigma prior", zorder=1, alpha=0.5)

    ax.legend(handles=handles + [
        mpatches.Patch(color=MUTED_TEXT, label="Prior contours")
    ], loc="upper right", fontsize=8.5, ncol=2)
    ax.set_xlabel("Latent Dimension 1  (mu_1)", fontsize=11)
    ax.set_ylabel("Latent Dimension 2  (mu_2)", fontsize=11)
    ax.set_title("VAE Latent Space: Encoder Means per Digit Class\n"
                 "KL regularization clusters classes near N(0,I) prior",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(colors=MUTED_TEXT)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("Visuals/12_vae_latent_manifold.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [12] VAE latent manifold")


def graph_13_vae_interpolation():
    """Simulated latent space interpolation path between two digit codes."""
    np.random.seed(9)
    n_steps = 10
    size = 20

    def digit_img(kind, blend=0.0, partner=None):
        a = np.zeros((size, size))
        b = np.zeros((size, size))
        if kind == "one":
            a[3:17, 9:11] = 1.0; a[3:6, 7:13] = 1.0
        else:
            for r in range(4, 16):
                for c in range(4, 16):
                    if (r-10)**2/20 + (c-10)**2/16 < 1: a[r, c] = 1.0
            a[8:12, 8:12] = 0.0
        if partner is not None: b = partner
        return np.clip((1-blend)*a + blend*b + np.random.normal(0, 0.07, (size, size)), 0, 1)

    target_b = np.zeros((size, size))
    for r in range(4, 16):
        for c in range(4, 16):
            if (r-10)**2/20 + (c-10)**2/16 < 1: target_b[r, c] = 1.0
    target_b[8:12, 8:12] = 0.0

    alphas = np.linspace(0, 1, n_steps)
    images = [digit_img("one", a, target_b) for a in alphas]

    fig, axes = plt.subplots(1, n_steps, figsize=(16, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("VAE Latent Space Interpolation: Digit '1' → Digit '0'\n"
                 "z_interp = (1-alpha)*z_a + alpha*z_b,  decoded at each step",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    for ax, img, alpha in zip(axes, images, alphas):
        ax.imshow(img, cmap="plasma", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"a={alpha:.1f}", fontsize=8, color=TEXT_COLOR)
        c = ACCENT_BLUE if alpha < 0.1 else (ACCENT_ORANGE if alpha > 0.9 else MUTED_TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig("Visuals/13_vae_interpolation.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [13] VAE latent space interpolation")


# ═════════════════════════════════════════════════════════════════════
# MODULE 04 — GENERATIVE ADVERSARIAL NETWORKS
# ═════════════════════════════════════════════════════════════════════

def graph_14_gan_minimax_game():
    """Diagram showing the GAN min-max adversarial loop."""
    fig = plt.figure(figsize=(15, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG); ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("GAN: Adversarial Min-Max Training Loop",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR, pad=15)

    def box(ax, cx, cy, w, h, color, label, sublabel=""):
        r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                            boxstyle="round,pad=0.12",
                            facecolor=color+"28", edgecolor=color, linewidth=2.8)
        ax.add_patch(r)
        ax.text(cx, cy + 0.15, label, ha="center", va="center",
                fontsize=11, color=color, fontweight="bold")
        if sublabel:
            ax.text(cx, cy - 0.45, sublabel, ha="center", va="center",
                    fontsize=8.5, color=MUTED_TEXT)

    box(ax, 2.0, 6.2, 2.8, 1.4, ACCENT_CYAN, "Noise z", "z ~ N(0, I)")
    box(ax, 5.5, 6.2, 3.0, 1.4, ACCENT_GREEN, "Generator G", "G(z) → fake image")
    box(ax, 9.5, 5.0, 3.0, 1.4, ACCENT_PURPLE, "Discriminator D", "P(real) -> [0,1]")
    box(ax, 2.0, 3.8, 3.0, 1.4, ACCENT_BLUE, "Real Data x", "x ~ p_data(x)")
    box(ax, 13.0, 5.0, 1.8, 1.2, ACCENT_ORANGE, "Output", "0 or 1")

    # Arrows
    arrows = [
        (3.4, 6.2, 4.0, 6.2, MUTED_TEXT, ""),
        (7.0, 6.2, 8.0, 5.4, ACCENT_GREEN, "fake"),
        (3.5, 3.8, 8.0, 4.7, ACCENT_BLUE, "real"),
        (11.0, 5.0, 12.1, 5.0, ACCENT_ORANGE, ""),
    ]
    for (x1, y1, x2, y2, color, label) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.22, label, ha="center",
                    fontsize=9, color=color, fontweight="bold")

    # Feedback arrows
    ax.annotate("", xy=(5.5, 5.5), xytext=(5.5, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT_GREEN, lw=2.0,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(3.8, 4.0, "Gradient update G\n(maximize D(G(z)))",
            ha="center", fontsize=8.5, color=ACCENT_GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                      edgecolor=ACCENT_GREEN, alpha=0.85))

    ax.annotate("", xy=(9.5, 4.3), xytext=(9.5, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT_PURPLE, lw=2.0))
    ax.text(9.5, 1.5, "Gradient update D\n(maximize log D(x) + log(1-D(G(z))))",
            ha="center", fontsize=8.5, color=ACCENT_PURPLE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                      edgecolor=ACCENT_PURPLE, alpha=0.85))

    # Formula
    ax.text(7.5, 0.55,
            "min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]",
            ha="center", fontsize=13, color=ACCENT_YELLOW, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_BG,
                      edgecolor=ACCENT_YELLOW, alpha=0.92))

    plt.tight_layout()
    plt.savefig("Visuals/14_gan_minimax_game.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [14] GAN minimax game diagram")


def graph_15_gan_training_loss():
    """GAN training dynamics: D loss, G loss, toward Nash equilibrium ln(2)."""
    np.random.seed(0)
    epochs = np.arange(1, 101)
    ln2    = np.log(2)

    d_loss = (0.95 - ln2) * np.exp(-0.07 * epochs) + ln2 + np.random.normal(0, 0.022, 100)
    g_loss = (1.60 - ln2) * np.exp(-0.05 * epochs) + ln2 + np.random.normal(0, 0.032, 100)
    d_acc  = 0.5 + 0.48 * np.exp(-0.06 * epochs) + np.random.normal(0, 0.01, 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("GAN Training Dynamics: Loss Convergence toward Nash Equilibrium",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    ax1.plot(epochs, d_loss, color=ACCENT_BLUE,   lw=2.5, label="Discriminator Loss")
    ax1.plot(epochs, g_loss, color=ACCENT_ORANGE,  lw=2.5, label="Generator Loss")
    ax1.axhline(ln2, color=ACCENT_GREEN, lw=2, linestyle="--",
                label=f"Nash Equilibrium  ln(2) ≈ {ln2:.3f}")
    ax1.fill_between(epochs, ln2 - 0.05, ln2 + 0.05, alpha=0.12, color=ACCENT_GREEN)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.4); ax1.tick_params(colors=MUTED_TEXT)
    ax1.annotate("Both converging\nto ln(2) ≈ 0.693",
                 xy=(85, ln2), xytext=(65, 0.85),
                 fontsize=9, color=ACCENT_GREEN,
                 arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.5))

    ax2.plot(epochs, d_acc, color=ACCENT_PURPLE, lw=2.5, label="Discriminator Accuracy")
    ax2.axhline(0.5, color=ACCENT_GREEN, lw=2, linestyle="--",
                label="Ideal = 0.50 (cannot distinguish)")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Discriminator Accuracy", fontsize=11)
    ax2.set_ylim(0.45, 1.02); ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4); ax2.tick_params(colors=MUTED_TEXT)

    plt.tight_layout()
    plt.savefig("Visuals/15_gan_training_loss.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [15] GAN training loss + accuracy")


def graph_16_mode_collapse():
    """Side-by-side: diverse generated samples vs mode-collapsed samples."""
    np.random.seed(3)
    n_cols = 10

    def digit(kind, size=18, noise=0.08):
        img = np.zeros((size, size))
        if kind == "bar":
            img[2:16, 8:10] = 1.0; img[2:5, 6:12] = 1.0
        elif kind == "oval":
            for r in range(3, 15):
                for c in range(3, 15):
                    if (r-9)**2/18 + (c-9)**2/14 < 1: img[r, c] = 1.0
            img[7:11, 7:11] = 0.0
        else:
            img[3:5, 4:14] = 1.0; img[3:9, 4:7] = 1.0
            img[8:10, 4:14] = 1.0; img[10:15, 11:14] = 1.0; img[13:15, 4:14] = 1.0
        return np.clip(img + np.random.normal(0, noise, img.shape), 0, 1)

    kinds_diverse  = ["bar", "oval", "other", "oval", "bar",
                      "other", "bar", "oval", "other", "bar"]
    diverse  = [digit(k) for k in kinds_diverse]
    collapsed = [digit("bar", noise=0.05) for _ in range(n_cols)]

    fig, axes = plt.subplots(2, n_cols, figsize=(16, 4))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("GAN Mode Collapse: Diverse Generation vs. Collapsed Generator",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    for row, (imgs, label, color, cmap) in enumerate([
        (diverse,   "Healthy GAN — Diverse samples", ACCENT_GREEN, "Purples"),
        (collapsed, "Mode Collapse — only '1's generated!", ACCENT_ORANGE, "Reds"),
    ]):
        axes[row, 0].set_ylabel(label, color=color, fontsize=9, fontweight="bold")
        for col, img in enumerate(imgs):
            axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, col].axis("off")
            for sp in axes[row, col].spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(2)

    plt.tight_layout()
    plt.savefig("Visuals/16_mode_collapse.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [16] GAN mode collapse")


# ═════════════════════════════════════════════════════════════════════
# MODULE 05 — DCGAN & GAN VARIANTS
# ═════════════════════════════════════════════════════════════════════

def graph_17_dcgan_architecture():
    """DCGAN Generator: Dense→Reshape→ConvT×2 with exact layer dims."""
    fig = plt.figure(figsize=(17, 7.5))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG); ax.set_xlim(0, 17); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("DCGAN Generator Architecture: Transposed Convolution Upsampling",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, pad=15)

    stages = [
        (1.5,  4.0, 0.8,  2.0, ACCENT_BLUE,   "Input\nNoise z\n(100,)"),
        (4.0,  4.0, 1.1,  2.5, ACCENT_GREEN,  "Dense\n(7x7x128)\n+ Reshape"),
        (7.0,  4.0, 1.5,  3.8, ACCENT_YELLOW, "Conv2DTranspose\n(14x14x64)\nstride=2, BN, ReLU"),
        (10.5, 4.0, 2.2,  5.2, ACCENT_PURPLE, "Conv2DTranspose\n(28x28x1)\nstride=2, tanh"),
        (14.0, 4.0, 2.8,  6.2, ACCENT_ORANGE, "Generated\nImage\n28x28x1"),
    ]

    for (cx, cy, w, h, color, label) in stages:
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=color+"28", edgecolor=color, linewidth=2.8)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold", linespacing=1.5)

    xs = [s[0] for s in stages]
    ws = [s[2] for s in stages]
    for i in range(len(xs)-1):
        ax.annotate("", xy=(xs[i+1]-ws[i+1]/2, 4),
                    xytext=(xs[i]+ws[i]/2, 4),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED_TEXT, lw=2.2))

    # Size annotations below each stage
    sizes = ["(100,)", "(7x7x128)", "(14x14x64)", "(28x28x1)", ""]
    for (cx, _, _, h, color, _), size in zip(stages, sizes):
        if size:
            ax.text(cx, 4 - h/2 - 0.45, size, ha="center",
                    fontsize=8.5, color=color, fontstyle="italic")

    # DCGAN rules callout
    rules = [
        "DCGAN Design Rules:",
        "1. No pooling — use strided Conv2D (D) / Conv2DTranspose (G)",
        "2. BatchNorm everywhere except G output & D input",
        "3. ReLU in Generator, LeakyReLU(0.2) in Discriminator",
        "4. No fully-connected hidden layers",
    ]
    for i, rule in enumerate(rules):
        c = ACCENT_YELLOW if i == 0 else TEXT_COLOR
        fw = "bold" if i == 0 else "normal"
        ax.text(8.5, 1.9 - i * 0.38, rule, ha="center", fontsize=8.5,
                color=c, fontweight=fw)

    plt.tight_layout()
    plt.savefig("Visuals/17_dcgan_architecture.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [17] DCGAN architecture")


def graph_18_transposed_convolution():
    """Step-by-step transposed convolution: input → zero-insert → convolve → output."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Transposed Convolution (stride=2): How Upsampling Works",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    step_titles = [
        "1. Input (3x3)", "2. Zero-insert (stride=2)\n→ 5x5", "3. Convolve (3x3 kernel)", "4. Output (5x5)"
    ]

    # Input 3x3
    inp = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)

    # Zero-inserted 5x5
    zi = np.zeros((5, 5))
    zi[::2, ::2] = inp

    # Kernel
    k = np.array([[0.25, 0.5, 0.25],
                  [0.5,  1.0, 0.5],
                  [0.25, 0.5, 0.25]])

    # Convolve (full)
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(zi, 1, mode='constant')
    out = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            out[r, c] = np.sum(padded[r:r+3, c:c+3] * k)
    out_norm = out / out.max()

    data = [inp, zi, zi.copy(), out_norm]
    cmaps= ["Blues", "Oranges", "Oranges", "Purples"]
    colors=[ACCENT_BLUE, ACCENT_YELLOW, ACCENT_YELLOW, ACCENT_PURPLE]

    for ax, mat, title, cmap, color in zip(axes, data, step_titles, cmaps, colors):
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max() if mat.max()>0 else 1,
                       aspect="equal")
        ax.set_title(title, fontsize=10, color=color, fontweight="bold", pad=8)
        ax.axis("off")
        # Annotate each cell
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                txt = "0" if (val == 0 and "Zero" in title) else f"{val:.1f}" if val != 0 else ""
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=9, color=TEXT_COLOR if val > 0.3 else MUTED_TEXT,
                        fontweight="bold")
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)

    # Add arrows between panels
    for x in [0.255, 0.505, 0.755]:
        fig.text(x, 0.5, "→", ha="center", va="center",
                 fontsize=26, color=MUTED_TEXT, transform=fig.transFigure)

    plt.tight_layout()
    plt.savefig("Visuals/18_transposed_convolution.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [18] Transposed convolution steps")


def graph_19_wgan_vs_standard():
    """JS divergence vs Wasserstein distance comparison when distributions don't overlap."""
    mu1, mu2_list = 0, [0.0, 1.0, 2.0, 5.0]
    x = np.linspace(-4, 9, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Wasserstein vs JS Divergence: Gradient Behavior When Distributions Diverge",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    js_vals = []; wd_vals = []
    for ax, mu2 in zip(axes.ravel(), mu2_list):
        p = norm.pdf(x, mu1, 1.0)
        q = norm.pdf(x, mu2,  1.0)

        ax.plot(x, p, color=ACCENT_BLUE,  lw=2.5, label=f"p=N(0,1)")
        ax.plot(x, q, color=ACCENT_ORANGE, lw=2.5, label=f"q=N({mu2},1)")
        ax.fill_between(x, 0, np.minimum(p, q), alpha=0.30, color=ACCENT_GREEN)
        ax.fill_between(x, np.minimum(p,q), np.maximum(p,q), alpha=0.18, color=ACCENT_ORANGE)

        # JS divergence (bounded in [0, ln(2)])
        m = 0.5 * (p + q)
        eps = 1e-10
        js = 0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + \
             0.5 * np.sum(q * np.log((q + eps) / (m + eps)))
        js = js * (x[1] - x[0])
        # Earth Mover's distance (distance between means for 1D Gaussians)
        wd = abs(mu2 - mu1)
        js_vals.append(js); wd_vals.append(wd)

        ax.set_xlabel("z"); ax.set_ylabel("Density")
        ax.set_title(f"mu_distance = {mu2}", fontsize=11, color=TEXT_COLOR, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)
        ax.tick_params(colors=MUTED_TEXT); ax.set_ylim(-0.02, 0.5)
        js_c = ACCENT_ORANGE if js > 0.2 else ACCENT_GREEN
        ax.text(0.02, 0.95, f"JS  = {js:.3f}\nWD = {wd:.2f}",
                transform=ax.transAxes, va="top", fontsize=10, color=ACCENT_YELLOW,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                          edgecolor=ACCENT_YELLOW, alpha=0.9))

    plt.tight_layout()
    plt.savefig("Visuals/19_wgan_vs_standard.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [19] WGAN vs standard JS divergence")


# ═════════════════════════════════════════════════════════════════════
# MODULE 06 — DIFFUSION MODELS
# ═════════════════════════════════════════════════════════════════════

def graph_20_noise_schedule():
    """Linear vs cosine noise schedules: beta_t and alpha_bar_t over T=1000."""
    T = 1000
    t = np.arange(T)

    betas_linear = np.linspace(1e-4, 0.02, T)
    alphas_linear = np.cumprod(1 - betas_linear)

    # Cosine schedule (Nichol & Dhariwal, 2021)
    s = 0.008
    f = np.cos((t/T + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cosine = f / f[0]
    betas_cosine  = np.clip(1 - alphas_cosine[1:] / alphas_cosine[:-1], 0, 0.999)
    betas_cosine  = np.concatenate([[betas_cosine[0]], betas_cosine])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Diffusion Noise Schedule: Linear (DDPM) vs Cosine (Improved DDPM)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    ax1.plot(t, betas_linear * 100,  color=ACCENT_BLUE,   lw=2.5, label="Linear beta_t (x100)")
    ax1.plot(t, betas_cosine * 100,  color=ACCENT_ORANGE,  lw=2.5, label="Cosine beta_t (x100)")
    ax1.set_xlabel("Timestep t", fontsize=11); ax1.set_ylabel("beta_t x100", fontsize=11)
    ax1.set_title("Noise Rate Schedule beta_t", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.35); ax1.tick_params(colors=MUTED_TEXT)

    ax2.plot(t, alphas_linear,  color=ACCENT_BLUE,   lw=2.5, label="Linear (signal power)")
    ax2.plot(t, alphas_cosine,  color=ACCENT_ORANGE,  lw=2.5, label="Cosine (signal power)")
    ax2.axhline(0.5, color=MUTED_TEXT, lw=1.5, linestyle=":", label="50% signal remaining")
    ax2.set_xlabel("Timestep t", fontsize=11); ax2.set_ylabel("alpha_bar_t (signal SNR)", fontsize=11)
    ax2.set_title("Signal Strength alpha_bar_t vs Timestep", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.35); ax2.tick_params(colors=MUTED_TEXT)
    ax2.text(0.97, 0.5, "Cosine uses the\nmiddle range better\n(less wasted steps)",
             transform=ax2.transAxes, ha="right", va="center",
             fontsize=9.5, color=ACCENT_ORANGE, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG,
                       edgecolor=ACCENT_ORANGE, alpha=0.9))

    plt.tight_layout()
    plt.savefig("Visuals/20_noise_schedule.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [20] Noise schedule comparison")


def graph_21_diffusion_forward_process():
    """Forward process: clean image progressively corrupted across timesteps."""
    np.random.seed(42)
    T   = 1000
    betas     = np.linspace(1e-4, 0.02, T)
    alphas    = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    size = 32
    base = np.zeros((size, size))
    for r in range(4, 28):
        for c in range(4, 28):
            if (r-16)**2/81 + (c-16)**2/49 < 1: base[r, c] = 1.0
    base[12:20, 14:18] = 0.0

    timesteps = [0, 100, 250, 400, 600, 800, 950, 1000]
    fig, axes = plt.subplots(1, len(timesteps), figsize=(17, 3.8))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Diffusion Forward Process q(x_t | x_0): Adding Gaussian Noise",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    for ax, t_step in zip(axes, timesteps):
        if t_step == 0:
            img = base.copy()
        elif t_step >= T:
            img = np.random.normal(0, 1, base.shape)
        else:
            ab = alpha_bar[t_step - 1]
            eps = np.random.normal(0, 1, base.shape)
            img = np.sqrt(ab) * base + np.sqrt(1 - ab) * eps
        ax.imshow(img, cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax.axis("off")
        if t_step == 0:
            ax.set_title("t=0\nclean x₀", fontsize=9, color=ACCENT_GREEN, fontweight="bold")
        elif t_step >= T:
            ax.set_title(f"t=T\npure noise", fontsize=9, color=ACCENT_ORANGE, fontweight="bold")
        else:
            snr = alpha_bar[t_step-1]
            ax.set_title(f"t={t_step}\nSNR={snr:.2f}", fontsize=8.5, color=MUTED_TEXT)
        c = ACCENT_GREEN if t_step == 0 else (ACCENT_ORANGE if t_step >= T else BORDER_COLOR)
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("Visuals/21_diffusion_forward_process.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [21] Diffusion forward process")


def graph_22_diffusion_reverse_process():
    """Reverse process: pure noise iteratively denoised to clean image."""
    np.random.seed(77)
    T   = 1000
    betas     = np.linspace(1e-4, 0.02, T)
    alphas    = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    size = 32
    base = np.zeros((size, size))
    for r in range(4, 28):
        for c in range(4, 28):
            if (r-16)**2/81 + (c-16)**2/49 < 1: base[r, c] = 1.0
    base[12:20, 14:18] = 0.0

    timesteps = list(reversed([0, 100, 250, 400, 600, 800, 950, 1000]))
    fig, axes = plt.subplots(1, len(timesteps), figsize=(17, 3.8))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Diffusion Reverse Process p_theta(x_{t-1}|x_t): U-Net Iterative Denoising",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    for ax, t_step in zip(axes, timesteps):
        if t_step >= T:
            img = np.random.normal(0, 1, base.shape)
        elif t_step == 0:
            img = base + np.random.normal(0, 0.04, base.shape)
        else:
            ab = alpha_bar[t_step - 1]
            eps = np.random.normal(0, 1, base.shape) * (1 - ab) ** 0.5 * 0.4
            img = np.sqrt(ab) * base + eps
        ax.imshow(img, cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax.axis("off")
        if t_step >= T:
            ax.set_title("Start:\npure noise", fontsize=9, color=ACCENT_ORANGE, fontweight="bold")
        elif t_step == 0:
            ax.set_title("t=0\ngenerated!", fontsize=9, color=ACCENT_GREEN, fontweight="bold")
        else:
            ax.set_title(f"t={t_step}", fontsize=9, color=MUTED_TEXT)
        c = ACCENT_ORANGE if t_step >= T else (ACCENT_GREEN if t_step == 0 else BORDER_COLOR)
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(2)

    fig.text(0.5, 0.01, "<── Reverse Denoising Direction (T → 0 steps)",
             ha="center", fontsize=12, color=ACCENT_GREEN, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 0.92])
    plt.savefig("Visuals/22_diffusion_reverse_process.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [22] Diffusion reverse process")


def graph_23_unet_architecture():
    """U-Net diagram showing encoder path, bottleneck, decoder, skip connections."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG); ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("U-Net Denoising Network: Encoder + Skip Connections + Decoder",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, pad=15)

    # (x, y, w, h, color, label)
    enc_blocks = [
        (2, 8.0, 2.2, 0.85, ACCENT_BLUE,   "Conv(64)+BN+ReLU\n64x28x28"),
        (2, 6.5, 2.2, 0.85, ACCENT_BLUE,   "Conv(128)+BN+ReLU\n128x14x14"),
        (2, 5.0, 2.2, 0.85, ACCENT_BLUE,   "Conv(256)+BN+ReLU\n256x7x7"),
    ]
    bottleneck = (8, 5.0, 2.6, 0.95, ACCENT_YELLOW, "Bottleneck\nConv(512)+Time Emb\n512x4x4")
    dec_blocks = [
        (14, 5.0, 2.2, 0.85, ACCENT_PURPLE, "ConvT(256)+BN\n256x7x7"),
        (14, 6.5, 2.2, 0.85, ACCENT_PURPLE, "ConvT(128)+BN\n128x14x14"),
        (14, 8.0, 2.2, 0.85, ACCENT_PURPLE, "ConvT(64)+BN\n64x28x28"),
    ]
    input_node  = (8, 8.0, 2.2, 0.85, ACCENT_ORANGE, "Input x_t\n+ t embed")
    output_node = (8, 3.2, 2.2, 0.85, ACCENT_GREEN,  "Output: e_theta\n(predicted noise)")

    def draw_box(ax, cx, cy, w, h, color, label):
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=color+"2a", edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=8.5, color=color, fontweight="bold", linespacing=1.5)

    draw_box(ax, *input_node)
    draw_box(ax, *bottleneck)
    draw_box(ax, *output_node)
    for b in enc_blocks: draw_box(ax, *b)
    for b in dec_blocks: draw_box(ax, *b)

    # Timestep embedding node
    draw_box(ax, 8, 9.2, 2.2, 0.75, ACCENT_CYAN, "Time Embedding\nSinusoidal(t)")

    # Encoder down-arrows
    for i in range(len(enc_blocks)-1):
        ax.annotate("", xy=(enc_blocks[i+1][0], enc_blocks[i+1][1]+0.43),
                    xytext=(enc_blocks[i][0], enc_blocks[i][1]-0.43),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT_BLUE, lw=2))

    # Input → first enc
    ax.annotate("", xy=(2, 8.43), xytext=(6.9, 8.0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED_TEXT, lw=1.8))

    # Enc → bottleneck
    ax.annotate("", xy=(6.7, 5.0), xytext=(3.1, 5.0),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT_BLUE, lw=2))

    # Bottleneck → first dec
    ax.annotate("", xy=(12.9, 5.0), xytext=(9.3, 5.0),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT_PURPLE, lw=2))

    # Decoder up-arrows
    for i in range(len(dec_blocks)-1):
        ax.annotate("", xy=(dec_blocks[i+1][0], dec_blocks[i+1][1]-0.43),
                    xytext=(dec_blocks[i][0], dec_blocks[i][1]+0.43),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT_PURPLE, lw=2))

    # Final dec → output
    ax.annotate("", xy=(9.1, 3.2), xytext=(13.5, 5.0-0.43),
                arrowprops=dict(arrowstyle="-|>", color=MUTED_TEXT, lw=1.8))

    # Skip connections
    skip_pairs = [(enc_blocks[0], dec_blocks[2]),
                  (enc_blocks[1], dec_blocks[1]),
                  (enc_blocks[2], dec_blocks[0])]
    for (e, d) in skip_pairs:
        x_mid = 8.0
        y = (e[1] + d[1]) / 2
        ax.annotate("", xy=(d[0]-d[2]/2, d[1]), xytext=(e[0]+e[2]/2, e[1]),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT_CYAN, lw=1.8,
                                    connectionstyle=f"arc3,rad=-0.15"))
        ax.text(x_mid, e[1]+0.1, "skip connection", ha="center", fontsize=7.5,
                color=ACCENT_CYAN, fontstyle="italic")

    # Time embed to bottleneck
    ax.annotate("", xy=(8, 5.48), xytext=(8, 8.83),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT_CYAN, lw=1.8,
                                linestyle="dashed"))

    plt.tight_layout()
    plt.savefig("Visuals/23_unet_architecture.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [23] U-Net architecture")


def graph_24_cfg_guidance_scale():
    """Classifier-Free Guidance: effect of guidance scale w on image quality vs diversity."""
    np.random.seed(1)
    w_vals   = [0, 1, 3, 7.5, 12, 20]
    diversity = [0.95, 0.88, 0.76, 0.62, 0.45, 0.22]
    quality   = [0.30, 0.52, 0.70, 0.87, 0.88, 0.82]
    n_imgs    = len(w_vals)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, n_imgs + 1, figure=fig,
                            width_ratios=[1]*n_imgs + [0.05],
                            height_ratios=[1, 1.6])
    fig.patch.set_facecolor(DARK_BG)

    # Top: trade-off curves
    ax_curve = fig.add_subplot(gs[0, :n_imgs])
    ax_curve.plot(w_vals, quality,   color=ACCENT_GREEN,  lw=2.5, marker="o", ms=8,
                  label="Sample Quality (FID proxy)")
    ax_curve.plot(w_vals, diversity, color=ACCENT_ORANGE,  lw=2.5, marker="s", ms=8,
                  label="Sample Diversity")
    ax_curve.axvline(7.5, color=ACCENT_YELLOW, lw=2, linestyle="--",
                     label="Default w=7.5 (Stable Diffusion)")
    ax_curve.set_xlabel("Guidance Scale w", fontsize=11)
    ax_curve.set_ylabel("Score", fontsize=11)
    ax_curve.set_title("Classifier-Free Guidance: Quality vs Diversity Trade-off",
                       fontsize=13, fontweight="bold")
    ax_curve.legend(fontsize=10); ax_curve.grid(True, alpha=0.35)
    ax_curve.tick_params(colors=MUTED_TEXT); ax_curve.set_xlim(-1, 22)

    # Bottom: simulated "generated images" at each guidance scale
    for i, (w, q) in enumerate(zip(w_vals, quality)):
        ax = fig.add_subplot(gs[1, i])
        img = np.random.rand(16, 16)
        sharpness = q
        # Sharper = more structured pattern
        smooth = np.zeros((16, 16))
        for r in range(16):
            for c in range(16):
                if (r-8)**2/30 + (c-8)**2/20 < 1: smooth[r, c] = 1.0
        combined = sharpness * smooth + (1 - sharpness) * img
        combined = np.clip(combined + np.random.normal(0, 0.1*(1-q), (16,16)), 0, 1)
        ax.imshow(combined, cmap="magma", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"w={w}", fontsize=9, color=ACCENT_YELLOW, fontweight="bold")
        note_c = ACCENT_ORANGE if w < 1 else (ACCENT_YELLOW if w > 12 else ACCENT_GREEN)
        note   = ("Too diverse\n(no guidance)" if w == 0 else
                  "Default\n(Stable Diff)" if w == 7.5 else
                  "Over-guided\n(artifacts)" if w >= 12 else "")
        if note:
            ax.text(0.5, -0.25, note, transform=ax.transAxes, ha="center",
                    fontsize=7.5, color=note_c, fontweight="bold")

    plt.tight_layout()
    plt.savefig("Visuals/24_cfg_guidance_scale.png", dpi=160,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  [24] CFG guidance scale")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating 24 visuals for CH 17: Autoencoders, GANs & Diffusion Models\n")
    print("── Module 01: Basic Autoencoders ──────────────────────")
    graph_01_autoencoder_architecture()
    graph_02_reconstruction_loss_curve()
    graph_03_undercomplete_overcomplete()
    graph_04_pca_vs_ae_latent()

    print("\n── Module 02: Sparse & Denoising AEs ─────────────────")
    graph_05_sparse_activation_histogram()
    graph_06_kl_sparsity_penalty()
    graph_07_denoising_ae_pipeline()
    graph_08_anomaly_detection()

    print("\n── Module 03: Variational Autoencoders ────────────────")
    graph_09_reparameterization_trick()
    graph_10_kl_divergence_prior()
    graph_11_vae_elbo_loss()
    graph_12_vae_latent_manifold()
    graph_13_vae_interpolation()

    print("\n── Module 04: GANs ────────────────────────────────────")
    graph_14_gan_minimax_game()
    graph_15_gan_training_loss()
    graph_16_mode_collapse()

    print("\n── Module 05: DCGAN & GAN Variants ───────────────────")
    graph_17_dcgan_architecture()
    graph_18_transposed_convolution()
    graph_19_wgan_vs_standard()

    print("\n── Module 06: Diffusion Models ────────────────────────")
    graph_20_noise_schedule()
    graph_21_diffusion_forward_process()
    graph_22_diffusion_reverse_process()
    graph_23_unet_architecture()
    graph_24_cfg_guidance_scale()

    print(f"\nAll 24 visuals saved to ./Visuals/")
