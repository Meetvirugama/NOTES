"""
Chapter 16 — NLP with RNNs and Attention
High-quality dark-themed matplotlib visuals.
Run from the CH 16 directory.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ─────────────────────────────────────────────
# GLOBAL THEME
# ─────────────────────────────────────────────
DARK   = "#0d1117"
PANEL  = "#161b22"
PANEL2 = "#1f2937"
ACCENT = "#58a6ff"
TEXT   = "#c9d1d9"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#e3b341"
PURPLE = "#bc8cff"
PINK   = "#d2a8ff"
GRAY   = "#8b949e"
ORANGE = "#ffa657"
TEAL   = "#39d0d0"

plt.rcParams.update({
    "figure.facecolor":  DARK,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRAY,
    "axes.labelcolor":   TEXT,
    "text.color":        TEXT,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GRAY,
    "grid.color":        GRAY,
    "grid.alpha":        0.25,
    "font.size":         11,
})

OUT = "Visuals"
os.makedirs(OUT, exist_ok=True)

def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close("all")
    print(f"  ✓ {name}")


def box(ax, x, y, w, h, text, fc=PANEL2, ec=GRAY, tc=TEXT, fs=11, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w/2, y + h/2, text, color=tc, fontsize=fs,
            ha="center", va="center", zorder=3,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=1)


# ═══════════════════════════════════════════════════════════════════════
# 01 — Char-RNN Workflow
# ═══════════════════════════════════════════════════════════════════════
def gen_01():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    fig.suptitle("Char-RNN: Predicting Next Character", color=TEXT, fontsize=16, fontweight="bold")

    chars  = ["T", "o", " ", "b", "e"]
    nexts  = ["o", " ", "b", "e", "?"]
    colors_ec = [ACCENT, ACCENT, PURPLE, PURPLE, GREEN]

    for i, (ch, nx, ec) in enumerate(zip(chars, nexts, colors_ec)):
        x = 1.5 + i * 1.9
        # Input token
        box(ax, x, 0.5, 1.1, 0.7, f'"{ch}"', ec=ACCENT, tc=ACCENT, fs=14, bold=True)
        ax.text(x + 0.55, 0.15, f"ID={ord(ch)}", color=GRAY, ha="center", fontsize=8)
        # RNN cell
        box(ax, x, 2.0, 1.1, 1.0, f"GRU\nh_{i}", ec=PURPLE, tc=PURPLE, fs=11)
        # Output
        box(ax, x, 3.7, 1.1, 0.7, f'"{nx}"', ec=GREEN, tc=GREEN, fs=14, bold=True)
        ax.text(x + 0.55, 4.55, "P(·|ctx)", color=GRAY, ha="center", fontsize=8)
        # Vertical arrows
        arrow(ax, x+0.55, 1.2, x+0.55, 2.0, color=ACCENT)
        arrow(ax, x+0.55, 3.0, x+0.55, 3.7, color=GREEN)
        # Horizontal hidden state
        if i < 4:
            arrow(ax, x+1.1, 2.5, x+1.9, 2.5, color=PURPLE, lw=2)

    ax.text(0.7, 0.85, "Input\nchars", color=ACCENT, fontsize=9, ha="center")
    ax.text(0.7, 2.5,  "RNN\ncells",  color=PURPLE, fontsize=9, ha="center")
    ax.text(0.7, 4.05, "Output\nprobs",color=GREEN, fontsize=9, ha="center")
    # Hidden state label
    ax.text(5.5, 2.8, "Hidden state h flows →", color=GRAY, fontsize=9)
    save("01_char_rnn_workflow.png")


# ═══════════════════════════════════════════════════════════════════════
# 02 — Dataset Windowing
# ═══════════════════════════════════════════════════════════════════════
def gen_02():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    fig.suptitle("tf.data Windowing (n_steps=5, shift=1)", color=TEXT, fontsize=16, fontweight="bold")

    # Full sequence
    seq = list("To be or not")
    for i, ch in enumerate(seq):
        ec = ACCENT if i < 6 else GRAY
        box(ax, 0.5 + i*0.85, 5.8, 0.8, 0.7, f'"{ch}"', ec=ec, fs=10)
    ax.text(0.9, 6.7, "Full Sequence (1D integer tensor)", color=TEXT, fontsize=11, fontweight="bold")

    # Windows
    rows = [
        ("Window 1:", "\"To be\"",  "\"o be \"",  0, 5),
        ("Window 2:", "\"o be \"",  "\" be o\"",  1, 6),
        ("Window 3:", "\" be o\"",  "\"be or\"",  2, 7),
    ]
    for j, (label, x_txt, y_txt, start, end) in enumerate(rows):
        yy = 4.2 - j * 1.5
        ax.text(0.2, yy + 0.35, label, color=TEXT, fontsize=11, fontweight="bold")
        box(ax, 1.5, yy, 3.5, 0.7, f"X: {x_txt}", ec=ACCENT, tc=ACCENT, fs=11)
        ax.text(5.3, yy + 0.35, "→ target:", color=GRAY, fontsize=10)
        box(ax, 6.4, yy, 3.5, 0.7, f"y: {y_txt}  (X shifted +1)", ec=GREEN, tc=GREEN, fs=11)

        # Highlight in full sequence
        for k in range(start, min(start+5, len(seq))):
            rect = mpatches.FancyBboxPatch(
                (0.5 + k*0.85, 5.7), 0.8, 0.9,
                boxstyle="round,pad=0.03",
                linewidth=2, edgecolor=[ACCENT, PURPLE, ORANGE][j],
                facecolor="none", zorder=4)
            ax.add_patch(rect)

    ax.text(0.2, 0.3, "Key insight: y = X shifted right by 1 position → dense supervision at every step!",
            color=YELLOW, fontsize=10, style="italic")
    save("02_dataset_windowing.png")


# ═══════════════════════════════════════════════════════════════════════
# 03 — Temperature Scaling
# ═══════════════════════════════════════════════════════════════════════
def gen_03():
    logits = np.array([2.5, 1.2, 0.4, -0.5, -1.0])
    labels = ["'e'", "'t'", "'a'", "' '", "'z'"]

    def softmax_t(x, t):
        e = np.exp((x - x.max()) / t)
        return e / e.sum()

    temps = [0.2, 1.0, 2.0]
    colors = [RED, ACCENT, GREEN]
    names  = ["T=0.2 (Greedy)", "T=1.0 (Balanced)", "T=2.0 (Creative)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("Temperature Scaling on Next-Character Probabilities", color=TEXT, fontsize=16, fontweight="bold")

    x = np.arange(len(labels))
    for ax, t, c, name in zip(axes, temps, colors, names):
        p = softmax_t(logits, t)
        bars = ax.bar(x, p, color=c, alpha=0.8, width=0.6, zorder=2)
        ax.set_title(name, color=c, fontsize=13, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability")
        ax.grid(axis="y", alpha=0.3)
        for bar, pi in zip(bars, p):
            ax.text(bar.get_x() + bar.get_width()/2, pi + 0.02,
                    f"{pi:.2f}", ha="center", color=TEXT, fontsize=9)
        entropy = -np.sum(p * np.log(p + 1e-10))
        ax.text(0.5, 0.95, f"H = {entropy:.2f} bits",
                transform=ax.transAxes, ha="center", color=YELLOW, fontsize=10)

    plt.tight_layout()
    save("03_temperature_scaling.png")


# ═══════════════════════════════════════════════════════════════════════
# 04 — Stateful vs Stateless RNN
# ═══════════════════════════════════════════════════════════════════════
def gen_04():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stateless vs Stateful RNN Across Batches", color=TEXT, fontsize=16, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # ── STATELESS (left) ──────────────────────────────
    ax1.set_title("Stateless RNN (Default)", color=ACCENT, fontsize=13, fontweight="bold", pad=10)
    for b in range(3):
        y = 4.5 - b * 1.7
        label = f"Batch {b+1}:\n[Chars {b*100}–{b*100+100}]"
        box(ax1, 0.5, y, 4.5, 1.1, label, ec=ACCENT)
        box(ax1, 5.5, y, 2.0, 1.1, f"h₀ = zeros\n(reset!)", ec=RED, tc=RED, fs=9)
        arrow(ax1, 5.0, y+0.55, 5.5, y+0.55, color=RED, lw=2)
        ax1.text(7.8, y+0.55, "discard", color=GRAY, va="center", fontsize=9)

    ax1.text(5.0, 0.4, "✅ Can shuffle  ✅ Simple  ❌ Forgets across batches",
             color=TEXT, ha="center", fontsize=9)

    # ── STATEFUL (right) ──────────────────────────────
    ax2.set_title("Stateful RNN (stateful=True)", color=GREEN, fontsize=13, fontweight="bold", pad=10)
    for b in range(3):
        y = 4.5 - b * 1.7
        label = f"Batch {b+1}:\n[Chars {b*100}–{b*100+100}]"
        box(ax2, 0.5, y, 4.0, 1.1, label, ec=GREEN)
        hn = f"h₀" if b == 0 else f"h_{b}ₑₙd"
        init_color = GRAY if b == 0 else PURPLE
        box(ax2, 4.8, y, 2.0, 1.1, f"init: {hn}", ec=init_color, tc=init_color, fs=9)

        if b < 2:
            box(ax2, 7.3, y, 1.8, 1.1, f"h_{b+1}ₑₙd →\nnext batch!", ec=PURPLE, tc=PURPLE, fs=8)
            arrow(ax2, 7.3, y+0.55, 6.8, y - 0.6, color=PURPLE, lw=2)

    ax2.text(5.0, 0.4, "❌ No shuffle  ❌ Complex  ✅ Learns long patterns",
             color=TEXT, ha="center", fontsize=9)

    plt.tight_layout()
    save("04_stateful_vs_stateless.png")


# ═══════════════════════════════════════════════════════════════════════
# 05 — Word Embedding Space
# ═══════════════════════════════════════════════════════════════════════
def gen_05():
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor(PANEL)
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
    ax.grid(alpha=0.2)
    ax.set_xlabel("Semantic Dimension 1  (e.g., Royalty ↔ Commonality)", color=GRAY)
    ax.set_ylabel("Semantic Dimension 2  (e.g., Male ↔ Female)", color=GRAY)
    fig.suptitle("Word Embedding Space\n(King − Man + Woman ≈ Queen)", color=TEXT, fontsize=15, fontweight="bold")

    words = {
        "King":   ( 2.8,  2.5, PURPLE),
        "Queen":  (-2.5,  2.8, PINK),
        "Man":    ( 2.8,  0.5, ACCENT),
        "Woman":  (-2.5,  0.8, ACCENT),
        "Prince": ( 1.5,  2.0, PURPLE),
        "Princess":(-1.5, 2.2, PINK),
        "Dog":    ( 2.0, -3.0, GREEN),
        "Cat":    ( 0.5, -3.3, GREEN),
        "Python": (-2.0, -2.5, ORANGE),
        "Java":   (-3.0, -3.0, ORANGE),
        "Paris":  (-0.5,  0.0, YELLOW),
        "France": (-1.0, -0.5, YELLOW),
    }

    for word, (x, y, c) in words.items():
        ax.scatter(x, y, s=120, color=c, zorder=5)
        ax.text(x + 0.15, y + 0.20, word, color=c, fontsize=12, fontweight="bold")

    # Arrows: King-Man+Woman ≈ Queen
    ax.annotate("", xy=(-2.5, 2.8), xytext=(2.8, 2.5),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2.5, linestyle="dashed"))
    ax.text(0.2, 2.9, "King − Man + Woman ≈ Queen", color=RED, fontsize=10, ha="center",
            bbox=dict(fc=DARK, ec=RED, alpha=0.8, pad=3))

    # Cluster labels
    for label, x, y in [("Royalty cluster", 0, 3.5), ("Animals cluster", 1.5, -4.5),
                         ("Languages cluster", -2.8, -3.8), ("Geography", -1, -0.8)]:
        ax.text(x, y, label, color=GRAY, fontsize=9, ha="center", style="italic")

    save("05_word_embeddings.png")


# ═══════════════════════════════════════════════════════════════════════
# 06 — Padding and Masking
# ═══════════════════════════════════════════════════════════════════════
def gen_06():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("Padding and Masking (mask_zero=True)", color=TEXT, fontsize=16, fontweight="bold")

    tokens = ["It", "was", "terrible", "<PAD>", "<PAD>", "<PAD>"]
    ids    = [6, 7, 8, 0, 0, 0]
    masks  = [True, True, True, False, False, False]

    for ax_idx, ax in enumerate((ax1, ax2)):
        ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")
        title_row = ["Input Token", "Token ID", "Mask", "GRU Action"]
        ax.text(6.5, 3.7, ["WITHOUT Masking ❌ (state diluted)", "WITH Masking ✅ (state preserved)"][ax_idx],
                color=[RED, GREEN][ax_idx], ha="center", fontsize=13, fontweight="bold")

        for i, (tok, tid, mask) in enumerate(zip(tokens, ids, masks)):
            x = 0.5 + i*2.0
            tok_color = ACCENT if mask else GRAY
            mid_color = RED if not mask else GRAY
            # Token
            box(ax, x, 2.7, 1.7, 0.7, tok, ec=tok_color, tc=tok_color, fs=10)
            # ID
            ax.text(x+0.85, 2.3, f"id={tid}", ha="center", color=GRAY, fontsize=9)
            # Mask badge
            badge_c = GREEN if mask else RED
            badge_t = "mask=T" if mask else "mask=F"
            box(ax, x, 1.6, 1.7, 0.55, badge_t, ec=badge_c, tc=badge_c, fs=9)
            # Action
            if ax_idx == 0:
                action = "Compute" if mask else "Compute (wrong!)"
                act_c  = GREEN if mask else RED
            else:
                action = "Compute" if mask else "Copy h_{t-1}"
                act_c  = GREEN if mask else YELLOW
            box(ax, x, 0.4, 1.7, 0.7, action, ec=act_c, tc=act_c, fs=8)

    save("06_padding_and_masking.png")


# ═══════════════════════════════════════════════════════════════════════
# 07 — Encoder-Decoder
# ═══════════════════════════════════════════════════════════════════════
def gen_07():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    fig.suptitle("Encoder-Decoder Architecture (Neural Machine Translation)", color=TEXT, fontsize=16, fontweight="bold")

    # Encoder inputs
    eng_words = ["I", "love", "you"]
    for i, w in enumerate(eng_words):
        box(ax, 0.5 + i*1.2, 0.5, 1.0, 0.7, f'"{w}"', ec=ACCENT, tc=ACCENT, fs=12)
        arrow(ax, 1.0 + i*1.2, 1.2, 1.0 + i*1.2, 1.8, color=ACCENT)

    # Encoder
    box(ax, 0.2, 1.8, 4.0, 1.4, "ENCODER (LSTM stack)\nReads all English words →\nCompresses to context vector", ec=ACCENT, tc=TEXT, fs=10)
    ax.text(4.5, 2.5, "h₁", color=GRAY, fontsize=9, ha="center")
    ax.text(4.5, 2.0, "h₂", color=GRAY, fontsize=9, ha="center")
    ax.text(4.5, 1.5, "...", color=GRAY, fontsize=9, ha="center")

    # Context vector
    arrow(ax, 4.2, 2.5, 5.0, 2.5, color=PURPLE, lw=3)
    box(ax, 5.0, 1.9, 2.2, 1.2, "Context\nVector c\n(h_final)", ec=PURPLE, tc=PURPLE, fs=11, bold=True)

    # Decoder
    arrow(ax, 7.2, 2.5, 8.0, 2.5, color=PURPLE, lw=3)
    box(ax, 8.0, 1.8, 4.0, 1.4, "DECODER (LSTM stack)\nInitialized with context c\nGenerates French word-by-word", ec=GREEN, tc=TEXT, fs=10)

    # Decoder outputs
    fra_words = ["Je", "t'aime", "<EOS>"]
    for i, w in enumerate(fra_words):
        box(ax, 8.3 + i*1.2, 4.0, 1.1, 0.7, f'"{w}"', ec=GREEN, tc=GREEN, fs=12)
        arrow(ax, 8.8 + i*1.2, 3.2, 8.8 + i*1.2, 4.0, color=GREEN)

    # Teacher forcing label
    box(ax, 2.0, 4.5, 4.5, 0.8, "Teacher Forcing: Decoder input = Ground truth (shifted right)", ec=YELLOW, tc=YELLOW, fs=9)
    arrow(ax, 4.2, 4.5, 8.0, 3.6, color=YELLOW, lw=1.5)

    save("07_encoder_decoder.png")


# ═══════════════════════════════════════════════════════════════════════
# 08 — Beam Search
# ═══════════════════════════════════════════════════════════════════════
def gen_08():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    fig.suptitle("Beam Search (k=3) — Tracking Top-3 Translation Paths", color=TEXT, fontsize=16, fontweight="bold")

    # Root
    box(ax, 0.2, 3.0, 1.2, 0.8, "<SOS>", ec=GRAY, bold=True)

    # Step 1
    step1 = [
        ("Je",    0.72, GREEN,  5.0),
        ("Tu",    0.11, YELLOW, 3.0),
        ("Il",    0.09, ORANGE, 1.0),
        ("Elle",  0.05, GRAY,   0.0),   # eliminated
    ]
    ax.text(3.5, 6.5, "Step 1: Expand <SOS>", color=TEXT, fontsize=11, ha="center")
    ax.axvline(x=2.8, color=GRAY, alpha=0.3, linestyle="--")
    for (word, prob, c, y) in step1:
        is_kept = y > 0
        alpha = 1.0 if is_kept else 0.3
        box(ax, 3.0, y+0.2, 1.8, 0.7, f"{word}\nlog={np.log(prob):.2f}", ec=c if is_kept else GRAY, tc=c if is_kept else GRAY, fs=10)
        arrow(ax, 1.4, 3.4, 3.0, y + 0.55, color=c if is_kept else GRAY)
        if not is_kept:
            ax.text(4.2, y + 0.1, "✗ eliminated", color=RED, fontsize=8)

    ax.text(3.9, 5.8, "↑ Keep top-3", color=GRAY, fontsize=9, style="italic")

    # Step 2 from "Je"
    ax.text(7.2, 6.5, "Step 2: Expand each beam", color=TEXT, fontsize=11, ha="center")
    ax.axvline(x=5.8, color=GRAY, alpha=0.3, linestyle="--")

    from_je = [
        ("Je t'aime",  0.72*0.65, GREEN,  4.8),
        ("Je suis",    0.72*0.05, YELLOW, 3.8),
    ]
    from_tu = [("Tu aimes",   0.11*0.60, ORANGE, 2.5)]
    from_il = [("Il t'aime",  0.09*0.55, TEAL,   1.2)]

    for (phrase, prob, c, y) in from_je + from_tu + from_il:
        is_top3 = prob > 0.03
        box(ax, 6.0, y, 2.5, 0.7, f"{phrase}\nlogP={np.log(prob):.2f}", ec=c if is_top3 else GRAY, tc=c if is_top3 else GRAY, fs=9)

    # Step 3
    ax.axvline(x=9.5, color=GRAY, alpha=0.3, linestyle="--")
    ax.text(11.0, 6.5, "Step 3: Final sequences", color=TEXT, fontsize=11, ha="center")

    finals = [
        ("Je t'aime\n<EOS>",     0.72*0.65*0.89, GREEN,  4.5),
        ("Je suis\nbeau <EOS>",  0.72*0.05*0.40, YELLOW, 2.8),
        ("Tu aimes\n<EOS>",      0.11*0.60*0.70, ORANGE, 1.2),
    ]
    for i, (phrase, prob, c, y) in enumerate(finals):
        box(ax, 9.8, y, 3.0, 1.0, f"{phrase}\nlogP={np.log(prob):.2f}", ec=c, tc=c, fs=9)
        if i == 0:
            ax.text(13.1, y+0.5, "← WINNER!", color=GREEN, fontsize=10, va="center")

    save("08_beam_search.png")


# ═══════════════════════════════════════════════════════════════════════
# 09 — Attention Alignment Matrix
# ═══════════════════════════════════════════════════════════════════════
def gen_09():
    fig, ax = plt.subplots(figsize=(7, 7))

    eng = ["The", "European", "Economic", "Area", "<EOS>"]
    fra = ["la", "zone", "économique", "européenne", "<EOS>"]

    # Realistic alignment matrix (captures French grammar reversal!)
    matrix = np.array([
        [0.87, 0.04, 0.02, 0.05, 0.02],  # la → The
        [0.02, 0.05, 0.03, 0.86, 0.04],  # zone → Area
        [0.02, 0.04, 0.87, 0.05, 0.02],  # économique → Economic
        [0.03, 0.89, 0.05, 0.02, 0.01],  # européenne → European
        [0.01, 0.01, 0.01, 0.02, 0.95],  # EOS → EOS
    ])

    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention Weight α")

    ax.set_xticks(range(len(eng))); ax.set_xticklabels(eng, rotation=35, ha="right", fontsize=11)
    ax.set_yticks(range(len(fra))); ax.set_yticklabels(fra, fontsize=11)
    ax.set_xlabel("Source: English", color=GRAY, fontsize=11)
    ax.set_ylabel("Target: French", color=GRAY, fontsize=11)
    ax.set_title("Attention Alignment Matrix\n'The European Economic Area' → 'la zone économique européenne'\n(Off-diagonal = French adjective order reversed!)", color=TEXT, fontsize=12)

    for i in range(len(fra)):
        for j in range(len(eng)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=10, fontweight="bold" if v > 0.5 else "normal")

    save("09_attention_alignment.png")


# ═══════════════════════════════════════════════════════════════════════
# 10 — Attention Architecture (Bahdanau)
# ═══════════════════════════════════════════════════════════════════════
def gen_10():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    fig.suptitle("Bahdanau Attention: Step-by-Step at Decoder Time t", color=TEXT, fontsize=16, fontweight="bold")

    # Encoder states
    enc_states = [("h₁\n'I'", ACCENT), ("h₂\n'love'", ACCENT), ("h₃\n'you'", ACCENT)]
    for i, (label, c) in enumerate(enc_states):
        box(ax, 0.5 + i*1.5, 5.0, 1.2, 0.9, label, ec=c, tc=c, fs=11)

    # Alignment scores
    scores = ["-0.12", "+0.34", "+1.73"]
    for i, s in enumerate(scores):
        arrow(ax, 1.1 + i*1.5, 5.0, 3.5, 3.8, color=GRAY)
        ax.text(1.5 + i*1.5, 4.5, f"e={s}", color=YELLOW, fontsize=9, ha="center")

    # Alignment model
    box(ax, 2.5, 3.1, 2.0, 0.7, "Alignment Model\ne = vᵀ tanh(W·s + U·hᵢ)", ec=PURPLE, tc=PURPLE, fs=9)

    # Decoder state
    box(ax, 6.0, 3.1, 1.8, 0.7, "s_{t-1}\n(Decoder)", ec=GREEN, tc=GREEN, fs=10)
    arrow(ax, 6.0, 3.45, 4.5, 3.45, color=GREEN)

    # Softmax
    box(ax, 2.5, 1.9, 2.0, 0.7, "Softmax → α\n[0.11, 0.18, 0.71]", ec=YELLOW, tc=YELLOW, fs=9)
    arrow(ax, 3.5, 3.1, 3.5, 2.6, color=GRAY)

    # Context vector
    arrow(ax, 4.5, 2.25, 5.8, 2.25, color=YELLOW, lw=2)
    box(ax, 5.8, 1.7, 2.2, 1.0, "Context Vector\nc_t = Σ αᵢ·hᵢ\n≈ 71% h₃ + ...", ec=ORANGE, tc=ORANGE, fs=9)

    # Decoder LSTM
    arrow(ax, 7.0, 2.2, 8.0, 2.2, color=ORANGE, lw=2)
    box(ax, 8.0, 1.7, 2.2, 1.0, "Decoder LSTM\ninput: [c_t; y_{t-1}]\n→ new word!", ec=GREEN, tc=GREEN, fs=9)

    # Annotation
    ax.text(3.5, 0.5, "Key: α sums to 1.0. c_t is a weighted average of ALL encoder states.",
            color=GRAY, fontsize=9, ha="center")

    save("10_attention_architecture.png")


# ═══════════════════════════════════════════════════════════════════════
# 11 — Scaled Dot-Product Attention
# ═══════════════════════════════════════════════════════════════════════
def gen_11():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")
    fig.suptitle("Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V", color=TEXT, fontsize=14, fontweight="bold")

    # Input row
    for i, (label, c, x) in enumerate([("Query (Q)", PURPLE, 1.0), ("Key (K)", ACCENT, 4.0), ("Value (V)", GREEN, 7.0)]):
        box(ax, x, 7.5, 2.0, 0.9, label, ec=c, tc=c, fs=12, bold=True)
        arrow(ax, x+1.0, 7.5, x+1.0, 6.5, color=c)

    # MatMul Q@Kᵀ
    box(ax, 2.5, 5.3, 2.8, 0.9, "MatMul: Q @ Kᵀ\nSimilarity matrix [Tq × Tk]", ec=GRAY, fs=9)
    arrow(ax, 2.0, 7.5, 2.5+1.4, 6.2, color=PURPLE)   # Q → MatMul
    arrow(ax, 5.0, 7.5, 2.5+1.4, 6.2, color=ACCENT)   # K → MatMul

    # Scale
    arrow(ax, 3.9, 5.3, 3.9, 4.4, color=GRAY)
    box(ax, 2.5, 3.5, 2.8, 0.8, f"Scale ÷ √dₖ\n(e.g. √64 = 8)\nStabilizes gradients!", ec=YELLOW, tc=YELLOW, fs=9)

    # Softmax
    arrow(ax, 3.9, 3.5, 3.9, 2.6, color=GRAY)
    box(ax, 2.5, 1.8, 2.8, 0.8, "Softmax (row-wise)\nAttention weights α\nΣαᵢⱼ = 1 for each row", ec=RED, tc=RED, fs=9)

    # MatMul with V
    arrow(ax, 4.5, 2.2, 5.8, 2.2, color=RED)
    arrow(ax, 8.0, 7.5, 6.5, 2.6, color=GREEN)
    box(ax, 5.8, 1.6, 2.5, 1.2, "MatMul: α @ V\n= Weighted sum\nof Value vectors", ec=GRAY, fs=9)

    # Output
    arrow(ax, 7.0, 1.6, 7.0, 0.9, color=TEAL, lw=2)
    box(ax, 5.5, 0.2, 3.0, 0.7, "OUTPUT: Context-enriched\nrepresentation for each token", ec=TEAL, tc=TEAL, fs=9, bold=True)

    # Right-side formula
    ax.text(9.5, 5.0, "Formula:\n\nAttention(Q,K,V)\n= softmax(\n    QKᵀ / √dₖ\n  ) · V",
            color=TEXT, fontsize=10, ha="center", va="center",
            bbox=dict(fc=PANEL2, ec=TEAL, pad=8, alpha=0.9))

    save("11_scaled_dot_product.png")


# ═══════════════════════════════════════════════════════════════════════
# 12 — Multi-Head Attention
# ═══════════════════════════════════════════════════════════════════════
def gen_12():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    fig.suptitle("Multi-Head Attention (h=8 heads, d_model=512, d_k=64 per head)", color=TEXT, fontsize=15, fontweight="bold")

    # Input
    box(ax, 4.5, 6.0, 4.0, 0.7, "Input Q, K, V  (shape: [batch, seq_len, 512])", ec=GRAY, fs=10)

    # 3 heads shown
    head_cols = [PURPLE, ACCENT, GREEN, ORANGE]
    head_xs   = [1.0, 4.0, 7.0, 10.0]
    for hi, (c, hx) in enumerate(zip(head_cols[:3], head_xs[:3])):
        label = f"Head {hi+1}" if hi < 2 else "Head 3\n(... ×8 total)"
        # Projection
        box(ax, hx, 4.4, 1.8, 0.8, f"Wᵩ×{chr(81+0)}, Wᴷ, Wᵛ\n({512}→{64}D)", ec=c, fs=8)
        arrow(ax, 6.5, 6.0, hx+0.9, 5.2, color=c)
        arrow(ax, hx+0.9, 4.4, hx+0.9, 3.8, color=c)
        # Scaled Dot Attention
        box(ax, hx, 3.0, 1.8, 0.8, "Scaled Dot\nAttention", ec=c, tc=c, fs=9)
        arrow(ax, hx+0.9, 3.0, hx+0.9, 2.4, color=c)
        # Head output
        box(ax, hx, 1.7, 1.8, 0.6, f"Out_{hi+1}\n[seq,64]", ec=c, tc=c, fs=9)
        arrow(ax, hx+0.9, 1.7, 6.5, 1.0, color=c)

    ax.text(9.5, 2.5, "...\n×8 heads\nin parallel", color=GRAY, fontsize=10, ha="center", style="italic")

    # Concat
    box(ax, 4.0, 0.3, 5.0, 0.8, "Concat([Out₁,...,Out₈]) → shape: [seq, 512]\n+ Linear projection W^O → [seq, 512]", ec=TEAL, tc=TEAL, fs=10)

    ax.text(6.5, 5.7, "↓ Project each head to 64D", color=GRAY, fontsize=9, ha="center")
    ax.text(6.5, 0.0, "Each head learns a different relationship type!", color=YELLOW, fontsize=9, ha="center", style="italic")

    save("12_multi_head_attention.png")


# ═══════════════════════════════════════════════════════════════════════
# 13 — Transformer Block
# ═══════════════════════════════════════════════════════════════════════
def gen_13():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("The Transformer Architecture: Encoder & Decoder Blocks", color=TEXT, fontsize=16, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xlim(0, 5); ax.set_ylim(0, 9); ax.axis("off")

    # ENCODER
    ax1.set_title("Encoder Block (×6)", color=ACCENT, fontsize=13, fontweight="bold")
    enc_layers = [
        ("Input Embeddings\n+ Positional Encoding", GRAY,   0.3, 0.9),
        ("Multi-Head\nSelf-Attention",               ACCENT, 2.0, 1.0),
        ("Add & Layer Norm",                         YELLOW, 3.4, 0.6),
        ("Feed Forward\nDense(512→2048→512)",        PURPLE, 4.4, 1.0),
        ("Add & Layer Norm",                         YELLOW, 5.8, 0.6),
        ("(×6 stacked)",                             GRAY,   6.7, 0.6),
        ("To Decoder\n(Cross-Attention Keys/Values)",TEAL,   7.7, 0.8),
    ]
    prev_y = None
    for label, c, y, h in enc_layers:
        box(ax1, 0.5, y, 4.0, h, label, ec=c, tc=c, fs=10)
        if prev_y:
            arrow(ax1, 2.5, prev_y, 2.5, y, color=GRAY)
        prev_y = y + h

    # DECODER
    ax2.set_title("Decoder Block (×6)", color=GREEN, fontsize=13, fontweight="bold")
    dec_layers = [
        ("Target Embeddings\n+ Positional Encoding",  GRAY,   0.3, 0.9),
        ("MASKED Multi-Head\nSelf-Attention\n(look-ahead mask!)", RED, 1.6, 1.2),
        ("Add & Layer Norm",                          YELLOW, 3.2, 0.6),
        ("Cross-Attention\n(Q=Decoder, K=V=Encoder)", GREEN, 4.2, 1.2),
        ("Add & Layer Norm",                          YELLOW, 5.8, 0.6),
        ("Feed Forward\nDense(512→2048→512)",         PURPLE, 6.8, 1.0),
        ("Add & Layer Norm → Softmax\n→ Output token probs", TEAL, 8.2, 0.7),
    ]
    prev_y = None
    for label, c, y, h in dec_layers:
        box(ax2, 0.5, y, 4.0, h, label, ec=c, tc=c, fs=10)
        if prev_y:
            arrow(ax2, 2.5, prev_y, 2.5, y, color=GRAY)
        prev_y = y + h

    plt.tight_layout()
    save("13_transformer_block.png")


# ═══════════════════════════════════════════════════════════════════════
# 14 — GPT vs BERT Timeline
# ═══════════════════════════════════════════════════════════════════════
def gen_14():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.5); ax.axis("off")
    fig.suptitle("The Pre-Training Revolution (2018–2020)", color=TEXT, fontsize=16, fontweight="bold")

    # Timeline axis
    ax.axhline(y=3.0, xmin=0.02, xmax=0.98, color=GRAY, lw=2)
    for x, yr in zip([1.5, 4.0, 7.0, 10.5, 13.0], ["2018 Q1", "2018 Q2", "2018 Q4", "2019", "2020"]):
        ax.axvline(x=x, ymin=0.4, ymax=0.55, color=GRAY, lw=1.5)
        ax.text(x, 2.5, yr, ha="center", color=GRAY, fontsize=9)

    # Models
    models = [
        (1.5, "ELMo",   "BiLSTM 2-layer\n93M params\nContextual\nembeddings",    ORANGE, 4.0, "↑"),
        (4.0, "GPT-1",  "Decoder-only\n117M params\nNext token\nprediction",     ACCENT, 4.0, "↑"),
        (7.0, "BERT",   "Encoder-only\n110–340M\nMLM + NSP\nBidirectional",      PURPLE, 4.0, "↑"),
        (10.5,"GPT-2",  "Decoder-only\n1.5B params\nZero-shot\ngeneralization",  TEAL,   4.0, "↑"),
        (13.0,"GPT-3",  "Decoder-only\n175B params\nFew-shot\nlearning",         GREEN,  4.0, "↑"),
    ]
    for x, name, desc, c, y, dir in models:
        ax.text(x, y-0.4, name, ha="center", color=c, fontsize=12, fontweight="bold")
        ax.text(x, y+0.3, desc, ha="center", color=GRAY, fontsize=8)
        ax.scatter(x, 3.0, s=120, color=c, zorder=5)
        ax.plot([x, x], [3.0, y-0.5], color=c, lw=1.5, linestyle="--")

    # Annotation boxes at bottom
    box(ax, 0.5, 0.3, 5.5, 0.9, "GPT family: Autoregressive | Decoder-only | Great at Generation", ec=ACCENT, tc=ACCENT, fs=9)
    box(ax, 6.5, 0.3, 5.5, 0.9, "BERT family: Bidirectional | Encoder-only | Great at Understanding", ec=PURPLE, tc=PURPLE, fs=9)

    save("14_ulmfit_elmo_bert_gpt.png")


# ═══════════════════════════════════════════════════════════════════════
# 15 — Chapter 16 Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════
def gen_15():
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    title_data = [
        (0, 0, "Module 1: Char-RNN",        PURPLE, "• Window dataset (n+1, shift=1)\n• y = X shifted right by 1\n• Stateless: reset h every batch\n• Stateful: pass h across batches\n• Temperature T: low=greedy, high=creative"),
        (0, 1, "Module 2: Embeddings",       ACCENT, "• Embedding layer = lookup table [V, D]\n• mask_zero=True → h_t = h_{t-1} at PAD\n• GloVe: 840B tokens, 400K words\n• Phase 1: freeze, Phase 2: fine-tune LR=1e-5\n• Bidirectional GRU sees L→R and R→L"),
        (0, 2, "Module 3: Encoder-Decoder",  GREEN,  "• Encoder reads ALL input → h_T (context)\n• Decoder initialized with h_T → generates\n• Teacher Forcing: feed ground truth\n• Beam Search (k=3): track top-k paths\n• Score = Σ log P(y_t | y_<t, X)"),
        (1, 0, "Module 4: Attention",        YELLOW, "• Bottleneck: 1 fixed vector for N words\n• Bahdanau: e = v tanh(W·s + U·h) (additive)\n• Luong: e = s·h (dot product, faster!)\n• α = softmax(e) → c_t = Σ α·h\n• Alignment matrix = explainability!"),
        (1, 1, "Module 5: Transformer",      ORANGE, "• Attn(Q,K,V) = softmax(QKᵀ/√dₖ)·V\n• Scale by √dₖ to prevent softmax collapse\n• 8 heads, each 64D → concat → project\n• Positional encoding: sin/cos waves\n• Look-ahead mask in decoder self-attn"),
        (1, 2, "Module 6: BERT & GPT",       TEAL,   "• ELMo: BiLSTM, same word=different vec\n• GPT: Decoder-only, next-token, causal\n• BERT: Encoder-only, MLM (mask 15%)\n• 80% [MASK], 10% random, 10% same\n• Fine-tune: lr=2e-5, epochs=3, warmup"),
        (2, 0, "Key Numbers",                RED,    "• BERT-base: 110M params, 12 blocks, d=768\n• GPT-3: 175B params, 96 blocks, d=12288\n• BLEU drop in seq2seq: >30 words → bad\n• Beam width k=3–10 for translation\n• Warmup = 10% of total training steps"),
        (2, 1, "Interview Must-Know",        PINK,   "• Q: Why scale by √dₖ? → Prevent saturation\n• Q: GPT vs BERT? → Causal vs Bidirectional\n• Q: Why MLM not next-word? → BERT cheats\n• Q: Attention complexity? → O(T²·d)\n• Q: LayerNorm vs BatchNorm? → NLP=Layer"),
        (2, 2, "Formula Reference",          GRAY,   "Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V\n\nPositional: PE[p,2i]=sin(p/10000^(2i/d))\n\nTemperature: p_i=exp(z_i/T)/Σexp(z_j/T)\n\nMLM loss: CE on 15% masked tokens\n\nBeam score: Σ log P(y_t|y_<t,X)/L^0.7"),
    ]

    for row, col, title, color, content in title_data:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, color=color, fontsize=11, fontweight="bold", pad=5)
        ax.text(0.05, 0.85, content, transform=ax.transAxes, color=TEXT,
                fontsize=8.5, va="top", family="monospace")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)

    fig.suptitle("CHAPTER 16 — NLP with RNNs and Attention: Complete Summary", color=TEXT, fontsize=18, fontweight="bold", y=1.01)
    save("15_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Chapter 16 visuals...")
    gen_01()
    gen_02()
    gen_03()
    gen_04()
    gen_05()
    gen_06()
    gen_07()
    gen_08()
    gen_09()
    gen_10()
    gen_11()
    gen_12()
    gen_13()
    gen_14()
    gen_15()
    print(f"\nAll 15 visuals saved to ./{OUT}/")

