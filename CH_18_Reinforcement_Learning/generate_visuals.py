"""
generate_visuals.py — Chapter 18: Reinforcement Learning
═══════════════════════════════════════════════════════════════════════════════
Generates 20 richly annotated matplotlib visualizations.
Every graph is self-explanatory — it teaches the concept visually.

Run:  python3 generate_visuals.py
Deps: pip install matplotlib numpy
═══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ─── Global Dark Theme ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "text.color":        "#e6edf3",
    "axes.labelcolor":   "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.alpha":        0.8,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "figure.dpi":        100,
    "savefig.dpi":       150,
})

VISUALS_DIR = Path("Visuals")
VISUALS_DIR.mkdir(exist_ok=True)

# ─── Colour Palette ───────────────────────────────────────────────────────────
RED    = "#ff6b6b"   # Danger / Actor / bad
BLUE   = "#58a6ff"   # Info / Critic / neutral
TEAL   = "#39d353"   # Good / positive
ORANGE = "#ffa657"   # Warning / highlight
PURPLE = "#bc8cff"   # Special
YELLOW = "#f0db4f"   # Accent
PINK   = "#ff7eb6"   # Alternative accent
BG0    = "#0d1117"   # Darkest bg
BG1    = "#161b22"   # Panel bg
BG2    = "#21262d"   # Card bg
BG3    = "#30363d"   # Border
TEXT   = "#e6edf3"   # Primary text
MUTED  = "#8b949e"   # Secondary text


def save(fname):
    plt.savefig(VISUALS_DIR / fname, dpi=150, bbox_inches="tight", facecolor=BG0)
    plt.close("all")
    print(f"  ✓  {fname}")


def section(title):
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ═══════════════════════════════════════════════════════════════════════════════
#  01 ▸ THE RL INTERACTION LOOP  (annotated, step-numbered)
# ═══════════════════════════════════════════════════════════════════════════════
section("01 · RL Interaction Loop")

fig, ax = plt.subplots(figsize=(16.0, 9.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

def rounded_box(ax, x, y, w, h, label, ec, fc=BG2, fs=10, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18",
                       facecolor=fc, edgecolor=ec, linewidth=2.5, zorder=3)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ax.text(x+w/2, y+h/2, label, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=TEXT, zorder=4)

def arr(ax, x1, y1, x2, y2, col, lw=2.2, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                linestyle=ls, mutation_scale=18), zorder=5)

# --- Main boxes ---
rounded_box(ax,  0.4, 2.5, 3.4, 2.0, "AGENT\n(Policy  π_θ)",  BLUE,  fc="#0a1929")
rounded_box(ax,  5.3, 3.5, 3.4, 1.2, "State  s_t",            TEAL,  fc="#0a2a1a", fs=11)
rounded_box(ax,  5.3, 2.0, 3.4, 1.2, "Action  a_t",           RED,   fc="#2a0a0a", fs=11)
rounded_box(ax,  5.3, 0.4, 3.4, 1.2, "Reward  r_t  &  s_{t+1}", ORANGE, fc="#2a1a00", fs=9)
rounded_box(ax, 10.2, 2.5, 3.4, 2.0, "ENVIRONMENT\n(Dynamics)",ORANGE, fc="#1a1200")

# --- Step circles ---
def step_badge(ax, x, y, n, col):
    c = Circle((x, y), 0.28, facecolor=col, edgecolor=TEXT, linewidth=1.5, zorder=6)
    ax.add_patch(c)
    ax.text(x, y, str(n), ha="center", va="center", fontsize=9,
            fontweight="bold", color=BG0, zorder=7)

# --- Arrows & labels ---
# STEP 1: Env sends state to Agent
arr(ax, 5.3, 4.1,  3.8, 4.1, TEAL)
ax.text(4.55, 4.4, "① Observe\nstate  s_t", ha="center", fontsize=8.5, color=TEAL)

# STEP 2: Agent picks action
arr(ax, 3.8, 3.2,  5.3, 2.6, RED)
ax.text(4.55, 2.7, "② Select\naction  a_t", ha="center", fontsize=8.5, color=RED)

# STEP 3: Action goes to env
arr(ax, 8.7, 2.6, 10.2, 3.2, RED)
ax.text(9.45, 2.7, "③ Execute\naction", ha="center", fontsize=8.5, color=RED)

# STEP 4: Env emits reward+state
arr(ax, 10.2, 4.1, 8.7, 4.1, ORANGE)
ax.text(9.45, 4.4, "④ Transition:\nnew  s_{t+1}", ha="center", fontsize=8.5, color=ORANGE)

arr(ax, 8.7, 0.95, 3.8, 0.95, ORANGE)
ax.text(6.3, 0.6, "⑤ Receive reward  r_t  →  Agent updates policy", ha="center",
        fontsize=9, color=ORANGE, style="italic")

# Step badges
step_badge(ax, 4.55, 4.1,  "①", TEAL)
step_badge(ax, 4.55, 2.75, "②", RED)
step_badge(ax, 9.45, 2.75, "③", RED)
step_badge(ax, 9.45, 4.1,  "④", ORANGE)
step_badge(ax, 9.45, 0.95, "⑤", ORANGE)

# --- Legend sidebar ---
legend_items = [
    (TEAL,   "State  s_t  :  what agent observes"),
    (RED,    "Action  a_t :  decision made by policy"),
    (ORANGE, "Reward  r_t :  scalar feedback signal"),
    (BLUE,   "Policy  π_θ :  neural network parameters θ"),
]
for i, (col, lbl) in enumerate(legend_items):
    y_l = 6.1 - i * 0.45
    ax.add_patch(Rectangle((0.3, y_l - 0.1), 0.25, 0.28, facecolor=col, zorder=5))
    ax.text(0.65, y_l + 0.05, lbl, fontsize=8, color=TEXT, va="center")

ax.set_title("The Reinforcement Learning Interaction Loop\n"
             "Agent ↔ Environment exchange: observe → decide → act → reward → repeat",
             fontsize=13, fontweight="bold", color=TEXT, pad=12)
plt.tight_layout(pad=2.5)
save("01_rl_interaction_loop.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  02 ▸ DISCOUNT FACTOR γ  —  visual intuition on a timeline
# ═══════════════════════════════════════════════════════════════════════════════
section("02 · Discount Factor γ")

fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.5))
fig.patch.set_facecolor(BG0)

# LEFT: decay curves
ax = axes[0]
ax.set_facecolor(BG1)
steps = np.arange(0, 51)
cfgs = [(0.0, MUTED, "--", "γ=0  (myopic: only NOW matters)"),
        (0.75, ORANGE, "-",  "γ=0.75"),
        (0.95, BLUE,   "-",  "γ=0.95  ← typical default"),
        (0.99, TEAL,   "-",  "γ=0.99  (long-horizon)")]
for g, col, ls, lbl in cfgs:
    lw = 3 if "0.95" in lbl else 2
    ax.plot(steps, [g**k for k in steps], color=col, lw=lw, ls=ls, label=lbl)

ax.axhline(1/np.e, color=MUTED, ls=":", lw=1.2, alpha=0.7)
ax.text(51.5, 1/np.e, "1/e≈0.37", va="center", fontsize=8, color=MUTED)
ax.fill_between(steps, [0.95**k for k in steps], 0, alpha=0.08, color=BLUE)
ax.annotate("at k=20:\nγ=0.95 → 0.36×\nstill 36% credit!", xy=(20, 0.95**20),
            xytext=(30, 0.6), fontsize=8, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
ax.set_xlabel("Steps into the Future  (k)"); ax.set_ylabel("Weight of future reward  γ^k")
ax.set_title("Effect of γ on Future Reward Weight", fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50); ax.set_ylim(-0.03, 1.08)

# RIGHT: timeline bar chart — visual reward weighting
ax2 = axes[1]
ax2.set_facecolor(BG1)
k_vals = np.arange(0, 10)
r_vals = np.ones(10)   # assume reward=1 each step for clarity

for gamma, col, label in [(0.95, BLUE, "γ=0.95"), (0.75, ORANGE, "γ=0.75")]:
    weights = [gamma**k for k in k_vals]
    offset = -0.2 if gamma == 0.95 else 0.2
    bars = ax2.bar(k_vals + offset, weights, 0.35, color=col, alpha=0.85,
                   label=f"{label}  (total={sum(weights):.2f})")
    for bar, w in zip(bars, weights):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{w:.2f}", ha="center", fontsize=6.5, color=col, rotation=45)

ax2.set_xlabel("Time Step  (k steps ahead)"); ax2.set_ylabel("Discount Weight  γ^k")
ax2.set_title("Per-Step Discount Weight Comparison\n(Episode reward = +1 every step)",
              fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, axis="y", alpha=0.3)
ax2.set_xticks(k_vals)
ax2.set_xticklabels([f"t+{k}" for k in k_vals], fontsize=8)

fig.suptitle("Discount Factor γ — How Much Does Future Reward Matter?",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
plt.tight_layout(pad=2.5)
save("02_discount_factor_gamma.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  03 ▸ ε-GREEDY EXPLORATION  —  what happens at each value
# ═══════════════════════════════════════════════════════════════════════════════
section("03 · ε-Greedy Exploration Visual")

fig, axes = plt.subplots(1, 3, figsize=(17.0, 7.0))
fig.patch.set_facecolor(BG0)

actions = ["Left", "Right"]
eps_vals = [1.0, 0.5, 0.05]
Q_left, Q_right = 3.2, 7.8    # example learned Q-values

for ax, eps in zip(axes, eps_vals):
    ax.set_facecolor(BG1)
    # softmax-like display: eps fraction = random, (1-eps) = greedy
    p_right_greedy = 1 - eps   # goes to best action
    p_left_greedy  = 0.0
    p_random = eps             # split evenly between actions

    p_left  = p_left_greedy  + p_random / 2
    p_right = p_right_greedy + p_random / 2

    colors = [ORANGE if p_right > p_left else RED,
              ORANGE if p_left  > p_right else RED]
    bars = ax.barh(actions, [p_left, p_right], color=[RED, TEAL], alpha=0.88, height=0.5)
    for bar, p in zip(bars, [p_left, p_right]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{p:.0%}", va="center", fontsize=14, fontweight="bold",
                color=bar.get_facecolor())

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probability of Selecting Action")
    ax.set_title(f"ε = {eps:.2f}\n"
                 f"{'Pure Exploration' if eps==1.0 else 'Mixed' if eps==0.5 else 'Mostly Greedy'}",
                 fontweight="bold")
    ax.axvline(0.5, color=MUTED, ls="--", lw=1, alpha=0.5)

    # annotation
    ax.text(0.5, -0.55, f"Explore {eps:.0%}  |  Exploit {1-eps:.0%}",
            ha="center", fontsize=9, color=MUTED, transform=ax.transData)

    # Q-value note
    ax.text(1.0, 1.6, f"Q(Left)={Q_left}\nQ(Right)={Q_right}",
            fontsize=8.5, color=MUTED, ha="right")
    ax.grid(True, axis="x", alpha=0.3)

fig.suptitle("ε-Greedy Exploration: How ε Controls Explore vs Exploit",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
plt.tight_layout(pad=2.5)
save("03_epsilon_greedy_visual.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  04 ▸ BELLMAN EQUATION  —  data-flow diagram
# ═══════════════════════════════════════════════════════════════════════════════
section("04 · Bellman Equation Flow")

fig, ax = plt.subplots(figsize=(16.0, 8.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

def label_box(ax, x, y, w, h, top_text, bot_text, ec, fc=BG2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.14",
                       facecolor=fc, edgecolor=ec, linewidth=2, zorder=3)
    ax.add_patch(p)
    ax.text(x+w/2, y+h*0.65, top_text, ha="center", va="center",
            fontsize=10, fontweight="bold", color=ec, zorder=4)
    ax.text(x+w/2, y+h*0.25, bot_text, ha="center", va="center",
            fontsize=8, color=MUTED, zorder=4)

def horz_arrow(ax, x1, x2, y, col, label="", label_y_off=0.25, lw=2):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=16))
    if label:
        ax.text((x1+x2)/2, y+label_y_off, label, ha="center", fontsize=8, color=col)

# Boxes
label_box(ax, 0.3, 2.0, 2.5, 2.0, "State  s",         "current position",  BLUE, "#051a30")
label_box(ax, 3.5, 2.0, 2.5, 2.0, "Action  a",        "chosen from π(s)",  RED,  "#300505")
label_box(ax, 6.7, 2.0, 2.5, 2.0, "Reward  r",        "immediate signal",  ORANGE, "#301500")
label_box(ax, 6.7, 0.1, 2.5, 1.5, "Next State  s'",   "transition result", TEAL,   "#00200a")
label_box(ax,10.0, 2.0, 3.5, 2.0, "Q*(s, a)",         "optimal Q-value",   PURPLE, "#1a0030")

# Arrows
horz_arrow(ax, 2.8, 3.5, 3.0, RED, "select\naction")
horz_arrow(ax, 6.0, 6.7, 3.0, ORANGE, "execute → get")
horz_arrow(ax, 9.2, 10.0, 3.0, PURPLE, "→ optimal\nQ update")
ax.annotate("", xy=(8.0, 2.0), xytext=(8.0, 1.6),
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2))
ax.annotate("", xy=(10.0, 2.5), xytext=(9.2, 1.4),
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2))

# Formula box
formula_box = FancyBboxPatch((0.2, -0.05), 13.6, 1.0,
    boxstyle="round,pad=0.1", facecolor="#0a0a20", edgecolor=PURPLE, lw=2, zorder=3)
ax.add_patch(formula_box)
ax.text(7.0, 0.45,
        "Q*(s, a)  =  r  +  γ · max_{a'} Q*(s', a')"
        "        [Bellman Optimality Equation]",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=PURPLE, zorder=4)

# Colour-coded annotation of formula components
annotations = [
    (1.8,  0.45, "r", "immediate\nreward", ORANGE),
    (5.2,  0.45, "γ·max Q*(s',a')", "discounted best\nfuture value", TEAL),
    (10.5, 0.45, "=", "", MUTED),
]

ax.set_title("Bellman Optimality Equation — Data Flow\n"
             "Q*(s,a) = immediate reward + discounted best future value",
             fontsize=12, fontweight="bold", color=TEXT, pad=10)
plt.tight_layout(pad=2.5)
save("04_bellman_equation_flow.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  05 ▸ Q-TABLE UPDATE  —  before/after with TD error highlighted
# ═══════════════════════════════════════════════════════════════════════════════
section("05 · Q-Table Update Visual")

fig, axes = plt.subplots(1, 3, figsize=(17.0, 8.0))
fig.patch.set_facecolor(BG0)
states   = ["s0\n(start)", "s1\n(mid)", "s2\n(near\ngoal)", "s3\n(goal)"]
actions_lbl = ["Left", "Right"]

# Q-table values (before update)
Q_before = np.array([
    [0.00, 0.10],   # s0
    [0.20, 0.50],   # s1
    [0.60, 0.90],   # s2 ← agent is here; takes Right → s3 gets reward
    [0.00, 0.00],   # s3 terminal
])
# After Q-learning update for (s2, Right):
# r=1.0, γ=0.9, max Q(s3,·)=0.0  → target = 1.0+0.9*0=1.0
# Q_new(s2,Right) = 0.90 + 0.1*(1.0 - 0.90) = 0.91
Q_after = Q_before.copy()
Q_after[2, 1] = 0.91   # updated cell

for idx, (ax, Q, title, highlight) in enumerate(zip(
        axes[:2],
        [Q_before, Q_after],
        ["Q-Table  BEFORE Update", "Q-Table  AFTER Update"],
        [False, True])):
    ax.set_facecolor(BG1)
    cmap = LinearSegmentedColormap.from_list("rl", ["#0a1929", "#0a3a6e", "#58a6ff", "#39d353"])
    im = ax.imshow(Q, cmap=cmap, vmin=0, vmax=1.0, aspect="auto")

    for i in range(len(states)):
        for j in range(len(actions_lbl)):
            col = BG0 if Q[i,j] < 0.5 else TEXT
            weight = "bold" if (highlight and i==2 and j==1) else "normal"
            ax.text(j, i, f"{Q[i,j]:.2f}", ha="center", va="center",
                    fontsize=13, color=col, fontweight=weight)
            if highlight and i==2 and j==1:
                rect = Rectangle((-0.5+j, -0.5+i), 1, 1, lw=3,
                                  edgecolor=TEAL, facecolor="none", zorder=5)
                ax.add_patch(rect)
                ax.text(j+0.5, i-0.4, "UPDATED!", fontsize=7.5,
                        color=TEAL, ha="center", fontweight="bold")

    ax.set_xticks(range(len(actions_lbl))); ax.set_xticklabels(actions_lbl, fontsize=10)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states, fontsize=9)
    ax.set_title(title, fontweight="bold", color=TEAL if highlight else TEXT)
    plt.colorbar(im, ax=ax, fraction=0.04, label="Q-value")

# Right panel: TD error breakdown
ax3 = axes[2]; ax3.set_facecolor(BG1); ax3.axis("off")
lines = [
    ("Q-LEARNING UPDATE FORMULA", 14, "bold",   BLUE),
    ("",                          8,  "normal",  TEXT),
    ("Q(s, a)  ←  Q(s, a)  +  α · δ", 13, "bold", TEXT),
    ("",                          8,  "normal",  TEXT),
    ("where  δ  =  TD Error:",    10, "normal",  ORANGE),
    ("δ = r + γ·maxQ(s',a') − Q(s,a)", 12, "bold", ORANGE),
    ("",                          8,  "normal",  TEXT),
    ("─" * 32,                    9,  "normal",  MUTED),
    ("This example:",             10, "bold",    TEAL),
    ("  s=s2,  a=Right",          9,  "normal",  TEXT),
    ("  r=1.0  (reached goal!)",  9,  "normal",  TEAL),
    ("  gamma=0.90,  alpha=0.10", 9,  "normal",  TEXT),
    ("  Q(s3,*)=0.0 (terminal)",  9,  "normal",  TEXT),
    ("",                          8,  "normal",  TEXT),
    ("TD Target = 1.0+0.9*0 = 1.00", 10, "bold", YELLOW),
    ("TD Error  = 1.00-0.90 = +0.10", 10, "bold", ORANGE),
    ("",                          8,  "normal",  TEXT),
    ("Q_new = 0.90+0.1*0.10 = 0.91", 11, "bold", TEAL),
]
y_pos = 0.98
for (txt, fs, fw, col) in lines:
    ax3.text(0.05, y_pos, txt, transform=ax3.transAxes,
             fontsize=fs, fontweight=fw, color=col, va="top")
    y_pos -= 0.065 if fs >= 11 else 0.05

fig.suptitle("Q-Table Update: How Q-Learning Learns One Step at a Time",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
plt.tight_layout(pad=2.5)
save("05_q_table_update.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  06 ▸ REINFORCE ALGORITHM  —  full episode timeline
# ═══════════════════════════════════════════════════════════════════════════════
section("06 · REINFORCE Episode Timeline")

np.random.seed(0)
T = 12
rewards = [1]*9 + [-1, -1, -1]   # pole falls at t=9
np.random.shuffle(rewards[:9])

gamma = 0.95
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)

returns_arr = np.array(returns, dtype=float)
norm = (returns_arr - returns_arr.mean()) / (returns_arr.std() + 1e-8)

fig, axes = plt.subplots(3, 1, figsize=(15.0, 11.0), sharex=True)
fig.patch.set_facecolor(BG0)
steps_t = np.arange(T)

# Panel 1: raw rewards per step
ax = axes[0]; ax.set_facecolor(BG1)
cols = [TEAL if r > 0 else RED for r in rewards]
ax.bar(steps_t, rewards, color=cols, alpha=0.85, width=0.7)
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylabel("Reward  r_t", fontsize=10)
ax.set_title("① Raw Rewards per Timestep  (CartPole: +1 per step balanced, −1 on fall)",
             fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
for t, r in enumerate(rewards):
    ax.text(t, r + 0.05*(1 if r>0 else -1), f"{r:+d}", ha="center", fontsize=8, color=TEXT)

# Panel 2: discounted return G_t
ax = axes[1]; ax.set_facecolor(BG1)
ax.plot(steps_t, returns_arr, color=BLUE, lw=2.5, marker="o",
        markersize=6, markerfacecolor=BG0, markeredgecolor=BLUE, label="G_t")
ax.fill_between(steps_t, 0, returns_arr, alpha=0.15, color=BLUE)
ax.axhline(returns_arr.mean(), color=ORANGE, ls="--", lw=1.5, label=f"mean G = {returns_arr.mean():.2f}")
ax.set_ylabel("Discounted Return  G_t", fontsize=10)
ax.set_title("② Discounted Return G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …  (γ=0.95)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
for t, g in enumerate(returns_arr):
    ax.text(t, g + 0.15, f"{g:.1f}", ha="center", fontsize=7.5, color=BLUE)

# Panel 3: normalised return (gradient weight)
ax = axes[2]; ax.set_facecolor(BG1)
bar_cols = [TEAL if v > 0 else RED for v in norm]
ax.bar(steps_t, norm, color=bar_cols, alpha=0.88, width=0.7)
ax.axhline(0, color=MUTED, lw=1.5)
ax.set_xlabel("Timestep  t"); ax.set_ylabel("Normalised Return\n(gradient weight)", fontsize=10)
ax.set_title("③ Normalised G_t used as gradient weight in REINFORCE\n"
             ">0 → increase action prob  |  <0 → decrease action prob",
             fontweight="bold")
ax.set_xticks(steps_t); ax.set_xticklabels([f"t={t}" for t in steps_t], fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
for t, v in enumerate(norm):
    ax.text(t, v + 0.05*(1 if v>0 else -1), f"{v:+.2f}",
            ha="center", fontsize=7.5, color=TEXT)

fig.suptitle("REINFORCE Algorithm — 3-Panel Episode Walkthrough\n"
             "Collect full episode → compute returns → normalise → update policy",
             fontsize=13, fontweight="bold", color=TEXT, y=1.01)
plt.tight_layout(pad=2.5)
save("06_reinforce_episode_walkthrough.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  07 ▸ TD vs MC vs DP  —  update diagrams side by side
# ═══════════════════════════════════════════════════════════════════════════════
section("07 · TD vs MC vs DP Update Diagrams")

fig, axes = plt.subplots(1, 3, figsize=(17.0, 9.0))
fig.patch.set_facecolor(BG0)

def draw_backup_diagram(ax, method, depth, branching, col):
    """Draw a backup/update tree for the given RL method."""
    ax.set_xlim(-3, 3); ax.set_ylim(-depth - 0.5, 0.8); ax.axis("off")
    ax.set_facecolor(BG1)

    # root state
    circle = Circle((0, 0), 0.3, facecolor=col, edgecolor=TEXT, lw=2, zorder=5)
    ax.add_patch(circle)
    ax.text(0, 0, "s", ha="center", va="center", fontsize=11, fontweight="bold",
            color=BG0, zorder=6)

    if method == "DP":
        # full width tree — sample ALL actions and ALL next states
        for a_idx, a_x in enumerate([-1.5, 1.5]):
            sq = FancyBboxPatch((a_x-0.22, -0.9-0.22), 0.44, 0.44,
                                boxstyle="square,pad=0", facecolor=RED,
                                edgecolor=TEXT, lw=1.5, zorder=5)
            ax.add_patch(sq)
            ax.plot([0, a_x], [0, -0.9], color=RED, lw=1.5, zorder=4)
            for s_idx, s_x in enumerate([a_x-0.9, a_x+0.9]):
                c2 = Circle((s_x, -2.2), 0.28, facecolor=TEAL, edgecolor=TEXT, lw=1.5, zorder=5)
                ax.add_patch(c2)
                ax.plot([a_x, s_x], [-0.9, -2.2], color=TEAL, lw=1.2, ls="--", zorder=4)
        ax.text(0, -3.0, "Uses P(s'|s,a) and ALL branches\nNo sampling needed", ha="center",
                fontsize=8.5, color=MUTED, style="italic")

    elif method == "TD(0)":
        # one action, one sampled next state
        sq = FancyBboxPatch((-0.22, -0.9-0.22), 0.44, 0.44,
                            boxstyle="square,pad=0", facecolor=RED,
                            edgecolor=TEXT, lw=1.5, zorder=5)
        ax.add_patch(sq)
        ax.plot([0, 0], [0, -0.9], color=RED, lw=2, zorder=4)
        c2 = Circle((0, -2.2), 0.28, facecolor=BLUE, edgecolor=TEXT, lw=2, zorder=5)
        ax.add_patch(c2)
        ax.plot([0, 0], [-0.9, -2.2], color=BLUE, lw=2, zorder=4)
        # Bootstrap arrow back
        ax.annotate("", xy=(0.4, -0.05), xytext=(0.4, -2.1),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8,
                                    connectionstyle="arc3,rad=-0.4"))
        ax.text(1.0, -1.1, "bootstrap\nfrom V(s')", fontsize=8, color=ORANGE)
        ax.text(0, -3.0, "One step sampled.\nBootstrap from V(s').\nUpdate V(s).",
                ha="center", fontsize=8.5, color=MUTED, style="italic")

    elif method == "Monte Carlo":
        y = 0
        for k in range(depth):
            sq = FancyBboxPatch((-0.22, y-0.9-0.22), 0.44, 0.44,
                                boxstyle="square,pad=0", facecolor=RED,
                                edgecolor=TEXT, lw=1.2, zorder=5)
            ax.add_patch(sq)
            ax.plot([0, 0], [y, y-0.9], color=RED, lw=1.5, zorder=4)
            y -= 0.9
            c2 = Circle((0, y-0.8), 0.22, facecolor=BLUE if k < depth-1 else TEAL,
                         edgecolor=TEXT, lw=1.5, zorder=5)
            ax.add_patch(c2)
            ax.plot([0, 0], [y, y-0.8], color=BLUE, lw=1.5, zorder=4)
            y -= 0.8
        ax.text(-0.4, y + 0.4, "TERMINAL\nG_T = r_T", fontsize=7.5, color=TEAL, ha="center")
        arrow_y_start = y + 0.3
        ax.annotate("", xy=(0, -0.3), xytext=(0, arrow_y_start),
                    arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=2,
                                    connectionstyle="arc3,rad=0.5"))
        ax.text(1.5, arrow_y_start/2, "propagate G_t\nback to s", fontsize=8, color=YELLOW)
        ax.text(0, arrow_y_start - 0.5, "Wait for episode end.\nUse actual return G_t.\nNo bootstrap.",
                ha="center", fontsize=8.5, color=MUTED, style="italic")

    title_col = {
        "DP": RED, "TD(0)": BLUE, "Monte Carlo": YELLOW
    }[method]
    ax.set_title(method, fontsize=13, fontweight="bold", color=title_col, pad=10)

draw_backup_diagram(axes[0], "DP", 2, 2, RED)
draw_backup_diagram(axes[1], "TD(0)", 2, 1, BLUE)
draw_backup_diagram(axes[2], "Monte Carlo", 3, 1, YELLOW)

# Legend row
for ax in axes:
    ax.add_patch(Circle((0, 0), 0, facecolor=BG0))  # invisible placeholder

fig.text(0.01, 0.02, "⚫ State node    ■ Action node    ──  sampled    ----  all transitions",
         fontsize=9, color=MUTED)
fig.suptitle("Backup Diagrams: How DP, TD(0) & Monte Carlo Update Value Estimates",
             fontsize=13, fontweight="bold", color=TEXT, y=1.01)
plt.tight_layout(pad=2.5)
save("07_backup_diagrams_td_mc_dp.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  08 ▸ BIAS-VARIANCE SPECTRUM  —  annotated comparison
# ═══════════════════════════════════════════════════════════════════════════════
section("08 · Bias-Variance Spectrum")

fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.5))
fig.patch.set_facecolor(BG0)

categories = ["Monte Carlo\nλ=1", "TD(λ=0.95)\nGAE", "TD(λ=0.5)", "TD(0)\nλ=0",
              "Dynamic\nProgramming"]
bias_v  = [0.0, 0.15, 0.45, 0.80, 1.0]
var_v   = [1.0, 0.75, 0.45, 0.15, 0.0]
x_pos   = np.arange(len(categories))

ax = axes[0]; ax.set_facecolor(BG1)
w = 0.35
b1 = ax.bar(x_pos-w/2, bias_v, w, color=RED,  alpha=0.88, label="Bias  (→wrong direction)")
b2 = ax.bar(x_pos+w/2, var_v,  w, color=BLUE, alpha=0.88, label="Variance  (→noisy updates)")
for bar, v in zip(b1, bias_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.2f}",
            ha="center", fontsize=8, color=RED, rotation=45)
for bar, v in zip(b2, var_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.2f}",
            ha="center", fontsize=8, color=BLUE, rotation=45)
ax.annotate("PPO uses\nGAE here!", xy=(1, 0.75), xytext=(1.8, 0.85),
            fontsize=8.5, color=TEAL,
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
ax.set_xticks(x_pos); ax.set_xticklabels(categories, fontsize=8.5)
ax.set_ylabel("Relative Level  (0=low, 1=high)"); ax.set_ylim(0, 1.25)
ax.set_title("Bias-Variance Trade-off by Algorithm", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

# Right: λ sweep curve
ax2 = axes[1]; ax2.set_facecolor(BG1)
lam = np.linspace(0, 1, 200)
bias_curve = lam**3
var_curve  = (1 - lam)**1.5
ax2.plot(lam, bias_curve, color=RED,  lw=2.5, label="Bias (↑ more λ → less bias)")
ax2.plot(lam, var_curve,  color=BLUE, lw=2.5, label="Variance (↑ more λ → more variance)")
ax2.fill_between(lam, bias_curve, var_curve,
                 where=bias_curve < var_curve, alpha=0.1, color=TEAL, label="Variance dominates")
ax2.fill_between(lam, bias_curve, var_curve,
                 where=bias_curve >= var_curve, alpha=0.1, color=RED, label="Bias dominates")
ax2.axvline(0.95, color=TEAL, ls="--", lw=2, label="λ=0.95  (PPO/GAE default)")
ax2.axvline(0.0,  color=MUTED, ls=":",  lw=1.5, alpha=0.6)
ax2.axvline(1.0,  color=MUTED, ls=":",  lw=1.5, alpha=0.6)
ax2.text(0.02, 0.92, "λ=0\n= TD(0)\nhigh bias", fontsize=8, color=RED, transform=ax2.transAxes)
ax2.text(0.85, 0.92, "λ=1\n= MC\nhigh var", fontsize=8, color=BLUE, transform=ax2.transAxes)
ax2.set_xlabel("λ  (TD-lambda / GAE parameter)"); ax2.set_ylabel("Relative level")
ax2.set_title("GAE λ Sweep: Sweet Spot at λ≈0.95", fontweight="bold")
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)

fig.suptitle("Bias-Variance Spectrum — The Fundamental Trade-off in RL Estimation",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
plt.tight_layout(pad=2.5)
save("08_bias_variance_spectrum.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  09 ▸ DQN ARCHITECTURE  —  annotated layer diagram
# ═══════════════════════════════════════════════════════════════════════════════
section("09 · DQN Architecture Annotated")

fig, ax = plt.subplots(figsize=(17.0, 9.0))
ax.set_xlim(0, 15); ax.set_ylim(0, 7); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

def layer_block(ax, x, y, w, h, name, detail, col, note=""):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=BG2, edgecolor=col, lw=2.5, zorder=3)
    ax.add_patch(p)
    ax.text(x+w/2, y+h*0.65, name, ha="center", va="center",
            fontsize=10, fontweight="bold", color=col, zorder=4)
    ax.text(x+w/2, y+h*0.28, detail, ha="center", va="center",
            fontsize=8, color=MUTED, zorder=4)
    if note:
        ax.text(x+w/2, y-0.42, note, ha="center", fontsize=7.5, color=ORANGE,
                style="italic", zorder=4)

def connect(ax, x1, x2, y_mid, col, lw=2):
    ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=14))

# CartPole DQN
y_base = 3.6
layers = [
    (0.3,  y_base, 1.8, 1.8, "Input", "4 features\n(CartPole obs)", TEAL, "Cart pos\nCart vel\nPole angle\nPole vel"),
    (3.0,  y_base, 1.8, 1.8, "Dense\n128", "ReLU\n128 neurons", BLUE, "16,512 params\n(640+128 bias)"),  # wait no it's 4*128+128
    (5.7,  y_base, 1.8, 1.8, "Dense\n128", "ReLU\n128 neurons", BLUE, "16,512 params"),
    (8.4,  y_base, 1.8, 1.8, "Output\nDense 2", "LINEAR\n2 outputs", RED, "258 params\nNO activation!"),
]
# actually params: 4*128+128=640, 128*128+128=16512, 128*2+2=258
layer_params = ["", "4×128+128\n= 640 params", "128×128+128\n= 16,512 params", "128×2+2\n= 258 params"]
for i, (x, y, w, h, name, detail, col, note) in enumerate(layers):
    layer_block(ax, x, y, w, h, name, detail, col, layer_params[i])

for i in range(len(layers)-1):
    x1 = layers[i][0] + layers[i][2]
    x2 = layers[i+1][0]
    connect(ax, x1, x2, y_base + 0.9, MUTED)

# Output annotation
ax.text(9.2, y_base+1.4, f"Q(s, Left)\nQ(s, Right)", fontsize=9, color=RED,
        ha="center", va="center")
ax.annotate("", xy=(11.3, y_base+0.9), xytext=(10.2, y_base+0.9),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
ax.text(12.8, y_base+1.5, "argmax → action\n(greedy policy)", fontsize=9, color=TEAL,
        ha="center")
ax.text(12.8, y_base+0.9, "Training target:\n|y - Q(s,a)|²", fontsize=9, color=ORANGE,
        ha="center")

# Total params
ax.add_patch(FancyBboxPatch((0.2, 2.5), 10.2, 0.85, boxstyle="round,pad=0.1",
             facecolor="#0a1929", edgecolor=BLUE, lw=1.5, zorder=3))
ax.text(5.3, 2.93, "Total trainable parameters = 640 + 16,512 + 258 = 17,410  (tiny but effective!)",
        ha="center", va="center", fontsize=9.5, color=BLUE, zorder=4)

# Key rule highlight
ax.add_patch(FancyBboxPatch((0.2, 1.5), 10.2, 0.80, boxstyle="round,pad=0.1",
             facecolor="#1a0a00", edgecolor=RED, lw=1.8, zorder=3))
ax.text(5.3, 1.9,
        "⚠  CRITICAL: Output layer MUST be LINEAR (no activation). "
        "Q-values ∈ (−∞, +∞).",
        ha="center", va="center", fontsize=9.5, color=RED, fontweight="bold", zorder=4)

# Code snippet
ax.add_patch(FancyBboxPatch((11.0, 1.0), 3.8, 5.8, boxstyle="round,pad=0.1",
             facecolor="#0d1117", edgecolor=BG3, lw=1.5, zorder=3))
code_lines = [
    ("model = keras.Sequential([", MUTED),
    ("  Dense(128,", TEXT),
    ("    activation='relu',", TEAL),
    ("    input_shape=[4]),", TEXT),
    ("  Dense(128,", TEXT),
    ("    activation='relu'),", TEAL),
    ("  Dense(2),  # linear!", RED),
    ("])", MUTED),
    ("", TEXT),
    ("# All Q-values at once:", ORANGE),
    ("q_vals = model(obs[None])", TEXT),
    ("# → shape: (1, 2)", MUTED),
    ("action = tf.argmax(q_vals[0])", TEAL),
]
y_code = 6.5
for line, col in code_lines:
    ax.text(11.15, y_code, line, fontsize=7.5, color=col,
            fontfamily="monospace", va="top")
    y_code -= 0.42

ax.set_title("DQN Q-Network Architecture (CartPole)\n"
             "One forward pass computes Q-values for ALL actions simultaneously",
             fontsize=12, fontweight="bold", color=TEXT, pad=12)
plt.tight_layout(pad=2.5)
save("09_dqn_architecture_annotated.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  10 ▸ EXPERIENCE REPLAY BUFFER  —  circular buffer animation frames
# ═══════════════════════════════════════════════════════════════════════════════
section("10 · Experience Replay Buffer Visual")

fig, axes = plt.subplots(1, 2, figsize=(16.0, 8.0))
fig.patch.set_facecolor(BG0)

def draw_replay_buffer(ax, n_slots=12, filled=8, highlight_indices=None, title=""):
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor(BG1)
    angles = np.linspace(0, 2*np.pi, n_slots, endpoint=False)
    r = 1.1
    for i, theta in enumerate(angles):
        x, y = r * np.cos(theta), r * np.sin(theta)
        is_filled = i < filled
        is_highlighted = highlight_indices is not None and i in highlight_indices
        col = BLUE if is_filled else BG3
        ec  = YELLOW if is_highlighted else (TEXT if is_filled else MUTED)
        lw  = 3 if is_highlighted else 1.5
        rect = FancyBboxPatch((x-0.2, y-0.14), 0.4, 0.28,
                              boxstyle="round,pad=0.04",
                              facecolor=col if not is_highlighted else ORANGE,
                              edgecolor=ec, lw=lw, zorder=4)
        ax.add_patch(rect)
        idx_txt = f"e{i}" if is_filled else "  "
        ax.text(x, y, idx_txt, ha="center", va="center", fontsize=7,
                color=BG0 if is_filled or is_highlighted else MUTED, fontweight="bold", zorder=5)

    # Center label
    ax.text(0, 0.15, "REPLAY", ha="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.text(0, -0.15, "BUFFER", ha="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.text(0, -0.45, f"{filled}/{n_slots} filled", ha="center", fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=10.5, fontweight="bold")

# Left: buffer with random mini-batch highlighted
draw_replay_buffer(axes[0], filled=10,
                   highlight_indices={1, 4, 7, 9},
                   title="Replay Buffer\n4 random transitions sampled for mini-batch")
axes[0].text(0, -1.55, "Random sampling → breaks temporal correlation!",
             ha="center", fontsize=9, color=TEAL)

# Right: sequential (correlated) problem illustration
ax = axes[1]; ax.set_facecolor(BG1); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title("Why Random Sampling Matters", fontsize=10.5, fontweight="bold")

# Draw sequential transitions (correlated)
corr_cols = [BLUE]*5 + [RED]*4
for i, col in enumerate(corr_cols):
    x = 0.5 + i * 1.0
    ax.add_patch(FancyBboxPatch((x, 5.0), 0.8, 0.9,
                 boxstyle="round,pad=0.05", facecolor=col, edgecolor=TEXT, lw=1.2))
    ax.text(x+0.4, 5.45, f"t={i}", ha="center", fontsize=7.5, color=BG0, fontweight="bold")
    if i < len(corr_cols)-1:
        ax.annotate("", xy=(x+0.9, 5.45), xytext=(x+0.8, 5.45),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
ax.text(4.5, 6.2, "Sequential (CORRELATED) — like training a sorted dataset!",
        ha="center", fontsize=9, color=RED, fontweight="bold")
ax.text(4.5, 4.7, "→ Network forgets previous states, oscillates, may diverge",
        ha="center", fontsize=8.5, color=RED, style="italic")

# Draw random transitions (decorrelated)
random_order = [3, 7, 0, 5, 2, 8, 1, 4, 6]
mix_cols = [BLUE if i < 5 else RED for i in random_order]
for j, (orig_i, col) in enumerate(zip(random_order, mix_cols)):
    x = 0.5 + j * 1.0
    ax.add_patch(FancyBboxPatch((x, 2.5), 0.8, 0.9,
                 boxstyle="round,pad=0.05", facecolor=col, edgecolor=TEAL, lw=1.5))
    ax.text(x+0.4, 2.95, f"t={orig_i}", ha="center", fontsize=7.5, color=BG0, fontweight="bold")
ax.text(4.5, 3.7, "Random Mini-Batch (DECORRELATED) — IID like supervised learning!",
        ha="center", fontsize=9, color=TEAL, fontweight="bold")
ax.text(4.5, 2.2, "→ Stable gradients, network trains on diverse experiences",
        ha="center", fontsize=8.5, color=TEAL, style="italic")

ax.add_patch(FancyBboxPatch((0.1, 1.3), 9.0, 0.7, boxstyle="round,pad=0.08",
             facecolor="#0a1a00", edgecolor=TEAL, lw=1.5))
ax.text(4.6, 1.65,
        "Buffer size: 10K–1M  |  Batch size: 32–256  |  Warmup: 1K steps before training",
        ha="center", va="center", fontsize=9, color=TEAL)

fig.suptitle("Experience Replay Buffer — Breaking Temporal Correlation",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
plt.tight_layout(pad=2.5)
save("10_experience_replay_buffer.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  11 ▸ TARGET NETWORK  —  online vs target comparison
# ═══════════════════════════════════════════════════════════════════════════════
section("11 · Target Network Mechanism")

fig, axes = plt.subplots(1, 2, figsize=(16.0, 8.5))
fig.patch.set_facecolor(BG0)

# Left: WITHOUT target network (chasing own tail)
ax = axes[0]; ax.set_facecolor(BG1)
t = np.linspace(0, 4*np.pi, 300)
q_val   = np.sin(t) * np.exp(-t*0.08) + 3
target_no = q_val + 0.4*np.sin(t*2.3 + 0.5)   # both oscillate together
ax.plot(t, q_val,    color=BLUE, lw=2.5, label="Q_θ(s,a)  prediction")
ax.plot(t, target_no, color=RED, lw=2.5, ls="--", label="Target = r+γ·max Q_θ(s',a')")
ax.fill_between(t, q_val, target_no, alpha=0.15, color=RED)
ax.set_xlabel("Training steps"); ax.set_ylabel("Q-value estimate")
ax.set_title("WITHOUT Target Network\n(both networks share θ → chasing own tail!)",
             fontweight="bold", color=RED)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.text(6, 4.2, "Target keeps moving\n→ oscillation → divergence!", fontsize=9,
        color=RED, style="italic", ha="center")

# Right: WITH target network (stable target steps)
ax2 = axes[1]; ax2.set_facecolor(BG1)
steps2 = np.arange(0, 300)
q_online = 1 + 3.0 / (1 + np.exp(-0.03 * (steps2 - 100)))
q_online += 0.1 * np.sin(steps2 * 0.3) * np.exp(-steps2 * 0.006)

# Staircase target (hard updates every 50 steps)
C = 50
q_target_vals = []
for s in steps2:
    freeze_pt = (s // C) * C
    q_target_vals.append(q_online[freeze_pt])
q_target = np.array(q_target_vals)

ax2.plot(steps2, q_online, color=BLUE, lw=2.5, label="Q_θ  (online, updated every step)")
ax2.step(steps2, q_target, color=ORANGE, lw=2.5, where="post",
         label=f"Q_θ−  (target, frozen for {C} steps)")

for c in range(0, 300, C):
    ax2.axvline(c, color=TEAL, ls=":", lw=1, alpha=0.5)
ax2.text(25, 1.3, "Target\nfrozen", fontsize=8, color=TEAL, ha="center")
ax2.text(75, 1.3, "Next\nfreezepoint", fontsize=8, color=TEAL, ha="center")

ax2.set_xlabel("Training steps"); ax2.set_ylabel("Q-value estimate")
ax2.set_title(f"WITH Target Network  (C={C} steps per hard update)\nStable, monotone convergence ✓",
              fontweight="bold", color=TEAL)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

fig.suptitle("Target Network: Stabilizing the Moving-Target Problem",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
plt.tight_layout(pad=2.5)
save("11_target_network_mechanism.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  12 ▸ DOUBLE DQN vs VANILLA DQN — overestimation demonstration
# ═══════════════════════════════════════════════════════════════════════════════
section("12 · Double DQN vs Vanilla DQN")

fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.5))
fig.patch.set_facecolor(BG0)

# Left: overestimation visual
ax = axes[0]; ax.set_facecolor(BG1)
actions_n = 8
true_q   = np.array([5.0, 7.0, 4.5, 6.0, 8.0, 5.5, 6.5, 7.5])
noise    = np.random.RandomState(42).randn(actions_n) * 1.2
noisy_q  = true_q + noise
max_true = true_q.max()
max_noisy = noisy_q.max()

x = np.arange(actions_n)
ax.bar(x - 0.22, true_q,  0.4, color=BLUE,   alpha=0.85, label="True Q*(s,a)")
ax.bar(x + 0.22, noisy_q, 0.4, color=ORANGE, alpha=0.85, label="Estimated Q̂(s,a)  + noise")
ax.axhline(max_true,  color=BLUE,   ls="--", lw=2, label=f"True max = {max_true:.1f}")
ax.axhline(max_noisy, color=RED,    ls="--", lw=2, label=f"Estimated max = {max_noisy:.1f}  (OVER!)")
ax.annotate("", xy=(6, max_noisy), xytext=(6, max_true),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax.text(6.5, (max_true+max_noisy)/2, f"+{max_noisy-max_true:.1f}\nover\nestimate",
        fontsize=9, color=RED, va="center")
ax.set_xlabel("Action index"); ax.set_ylabel("Q-value")
ax.set_title("Vanilla DQN: max Operation Overestimates\nbecause noise always selects upward outlier",
             fontweight="bold")
ax.legend(fontsize=8.5); ax.grid(True, axis="y", alpha=0.3)

# Right: Double DQN fix explanation
ax2 = axes[1]; ax2.set_facecolor(BG1); ax2.axis("off")
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)

def info_box(ax, x, y, w, h, header, body_lines, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                 facecolor=BG2, edgecolor=col, lw=2))
    ax.text(x+w/2, y+h-0.35, header, ha="center", fontsize=10,
            fontweight="bold", color=col)
    for i, line in enumerate(body_lines):
        ax.text(x+0.2, y+h-0.75-i*0.52, line, fontsize=8.5, color=TEXT, va="top")

info_box(ax2, 0.2, 5.5, 9.6, 4.2, "VANILLA DQN  (overestimates)",
         ["Target = r  +  γ · max_{a'} Q_θ−(s', a')",
          "• SAME network θ− selects AND evaluates best action",
          "• If Q̂(a*)  is noise-inflated → target inflated too",
          "• Overestimation compounds through Bellman backups"],
         RED)

info_box(ax2, 0.2, 0.8, 9.6, 4.2, "DOUBLE DQN  (unbiased estimate ✓)",
         ["Step 1:  a* = argmax_{a'} Q_θ(s', a')    ← online network SELECTS",
          "Step 2:  y  = r  +  γ · Q_θ−(s', a*)     ← target network EVALUATES",
          "• Two independent networks cross-check each other",
          "• Independent noise → averaging reduces bias → accurate Q*"],
         TEAL)

ax2.text(5.0, 5.2, "▼  Code change: just 3 lines  ▼", ha="center",
         fontsize=9, color=YELLOW, fontweight="bold")

fig.suptitle("Double DQN: Fixing Q-Value Overestimation with Decoupled Selection/Evaluation",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
plt.tight_layout(pad=2.5)
save("12_double_dqn_overestimation.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  13 ▸ DUELING DQN  —  V + A decomposition
# ═══════════════════════════════════════════════════════════════════════════════
section("13 · Dueling DQN Architecture")

fig, ax = plt.subplots(figsize=(16.0, 9.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

# Shared trunk
rounded_box(ax, 0.3, 2.8, 2.5, 1.5, "Input\nstate  s", TEAL, fc="#001a10")
rounded_box(ax, 3.5, 3.0, 2.5, 1.1, "Dense 128\nReLU", BLUE, fc=BG2)
rounded_box(ax, 3.5, 1.8, 2.5, 1.1, "(Shared\nfeature trunk)", MUTED, fc=BG2, bold=False)

# Value stream
rounded_box(ax, 7.2, 4.8, 2.5, 1.1, "Dense 64\nReLU", PURPLE, fc=BG2)
rounded_box(ax, 7.2, 3.5, 2.5, 1.1, "V(s)  ← 1 output\nlinear", PURPLE, fc="#0a0020")

# Advantage stream
rounded_box(ax, 7.2, 2.0, 2.5, 1.1, "Dense 64\nReLU", RED, fc=BG2)
rounded_box(ax, 7.2, 0.7, 2.5, 1.1, "A(s,a) ← N outputs\nlinear (N=actions)", RED, fc="#200000")

# Combine layer
rounded_box(ax, 10.8, 2.8, 2.8, 1.5, "Combine\nQ(s,a) = V(s) + A(s,a)\n− mean(A)", ORANGE, fc="#1a0800")

# Arrows
arr(ax, 2.8, 3.55, 3.5, 3.55, MUTED)
# shared to value
arr(ax, 6.0, 3.55, 7.2, 5.35, PURPLE)
arr(ax, 6.0, 3.55, 7.2, 4.05, PURPLE)
# shared to advantage
arr(ax, 6.0, 3.55, 7.2, 2.55, RED)
arr(ax, 6.0, 3.55, 7.2, 1.25, RED)
# value + adv to combine
arr(ax, 9.7, 4.05, 10.8, 3.55, PURPLE)
arr(ax, 9.7, 1.25, 10.8, 3.0, RED)
arr(ax, 13.6, 3.55, 14.0, 3.55, ORANGE)

# Labels
ax.text(6.6, 5.8, "Value\nStream", ha="center", fontsize=9, color=PURPLE, fontweight="bold")
ax.text(6.6, 0.2, "Advantage\nStream", ha="center", fontsize=9, color=RED, fontweight="bold")

# Intuition boxes
intuit = [
    (0.3, 6.3, 6.0, 0.55, "V(s) = how good is this STATE regardless of action", PURPLE),
    (0.3, 5.7, 6.0, 0.55, "A(s,a) = how much BETTER is action a vs average action", RED),
    (0.3, 5.1, 6.0, 0.55, "Q(s,a) = V(s) + A(s,a) - mean(A)    [mean subtracted for identifiability]", ORANGE),
]
for x, y, w, h, txt, col in intuit:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                 facecolor=BG2, edgecolor=col, lw=1.5))
    ax.text(x+0.15, y+h/2, txt, va="center", fontsize=8.5, color=col)

# Right panel: example values
ax.add_patch(FancyBboxPatch((7.0, -0.1), 6.8, 6.8, boxstyle="round,pad=0.1",
             facecolor=BG2, edgecolor=BG3, lw=1.5))
lines_r = [
    ("Example (N=4 actions):", YELLOW, 9, "bold"),
    ("", TEXT, 7, "normal"),
    ("V(s)   = 6.0   ← state quality", PURPLE, 9, "normal"),
    ("", TEXT, 7, "normal"),
    ("A(s, Left)   = −1.5", RED, 9, "normal"),
    ("A(s, Right)  = +0.8", RED, 9, "normal"),
    ("A(s, Up)     = +0.5", RED, 9, "normal"),
    ("A(s, Down)   = +0.2", RED, 9, "normal"),
    ("mean(A)      = 0.0", MUTED, 9, "normal"),
    ("", TEXT, 7, "normal"),
    ("Q(s, Left)  = 6.0 − 1.5 − 0.0 = 4.5", ORANGE, 9, "normal"),
    ("Q(s, Right) = 6.0 + 0.8 − 0.0 = 6.8", ORANGE, 9, "normal"),
    ("Q(s, Up)    = 6.0 + 0.5 − 0.0 = 6.5", ORANGE, 9, "normal"),
    ("Q(s, Down)  = 6.0 + 0.2 − 0.0 = 6.2", ORANGE, 9, "normal"),
    ("", TEXT, 7, "normal"),
    ("→ argmax = Right  (correct!)", TEAL, 10, "bold"),
]
y_r = 6.5
for (txt, col, fs, fw) in lines_r:
    ax.text(7.2, y_r, txt, fontsize=fs, color=col, fontweight=fw, va="top")
    y_r -= (0.45 if fs >= 9 else 0.3)

ax.set_title("Dueling DQN Architecture: Decouple State Value V(s) from Action Advantage A(s,a)",
             fontsize=12, fontweight="bold", color=TEXT, pad=12)
plt.tight_layout(pad=2.5)
save("13_dueling_dqn_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  14 ▸ ACTOR-CRITIC  —  two-network annotated flow
# ═══════════════════════════════════════════════════════════════════════════════
section("14 · Actor-Critic Flow")

fig, ax = plt.subplots(figsize=(16.0, 9.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

# Boxes
rounded_box(ax, 0.3, 2.8, 2.2, 1.5,  "State  s_t",         TEAL,   fc="#001a10")
rounded_box(ax, 3.8, 4.5, 3.0, 1.5,  "ACTOR\nπ_θ(a|s)",   RED,    fc="#200000")
rounded_box(ax, 3.8, 1.5, 3.0, 1.5,  "CRITIC\nV_φ(s)",    BLUE,   fc="#001030")
rounded_box(ax, 8.8, 2.8, 2.5, 1.5,  "ENV",                ORANGE, fc="#1a0800")
rounded_box(ax, 3.8, 0.1, 3.0, 1.0,  "Advantage  A_t\n= δ (TD error)", PURPLE, fc="#100020")

# Arrows
arr(ax, 2.5,  3.55, 3.8,  5.25, MUTED)   # state → actor
arr(ax, 2.5,  3.55, 3.8,  2.25, MUTED)   # state → critic
arr(ax, 6.8,  5.25, 8.8,  3.8,  RED)     # actor → env (action)
ax.text(7.8, 5.3, "a_t", fontsize=10, color=RED, fontweight="bold", ha="center")
arr(ax, 8.8,  3.2,  2.5,  3.2,  ORANGE)  # env → state (new state + reward)
ax.text(5.7, 3.5, "r_t, s_{t+1}", fontsize=9, color=ORANGE, ha="center")
arr(ax, 5.3,  1.5,  5.3,  1.1,  PURPLE)  # critic → advantage
arr(ax, 4.8,  0.6,  3.5,  3.5,  PURPLE, lw=1.5)  # advantage → actor loss

# Formula boxes
formulas = [
    (0.2, 6.2, 6.0, 0.65, "TD Error  δ = r_t + γ·V_φ(s_{t+1}) − V_φ(s_t)   ← Advantage estimate", PURPLE),
    (0.2, 5.5, 6.0, 0.65, "Actor Loss:  L_actor = −δ · log π_θ(a_t | s_t)         [gradient ASCENT]", RED),
    (0.2, 4.8, 6.0, 0.65, "Critic Loss:  L_critic = δ²  = (r + γV(s') − V(s))²   [MSE regression]", BLUE),
    (0.2, 4.1, 6.0, 0.65, "Entropy bonus:  L_H = −Σ π log π   [exploration incentive]", MUTED),
]
for x, y, w, h, txt, col in formulas:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                 facecolor=BG2, edgecolor=col, lw=1.5))
    ax.text(x+0.15, y+h/2, txt, va="center", fontsize=8.8, color=col)

# Update arrows
ax.text(7.5, 1.5, "ACTOR updates θ\n(gradient ascent)", ha="center", fontsize=9,
        color=RED, style="italic")
ax.text(7.5, 0.6, "CRITIC updates φ\n(gradient descent)", ha="center", fontsize=9,
        color=BLUE, style="italic")
arr(ax, 6.8, 5.25, 7.5, 2.2, RED, lw=1.5)
arr(ax, 6.8, 2.25, 7.5, 0.9, BLUE, lw=1.5)

# Code box (right side)
ax.add_patch(FancyBboxPatch((10.5, 0.5), 3.3, 6.2, boxstyle="round,pad=0.1",
             facecolor=BG2, edgecolor=BG3, lw=1.5))
code = [
    "# Shared network",
    "logits, V_s = model(obs)",
    "",
    "# TD Error = Advantage",
    "_, V_next = model(next_obs)",
    "delta = r + γ*V_next - V_s",
    "",
    "# Actor gradient",
    "log_pi = log_softmax(logits)",
    "actor_loss = -delta * log_pi[a]",
    "",
    "# Critic gradient",
    "critic_loss = delta ** 2",
    "",
    "# Entropy bonus",
    "pi = softmax(logits)",
    "entropy = -(pi*log_pi).sum()",
    "",
    "loss = (actor_loss",
    "       + 0.5*critic_loss",
    "       - 0.01*entropy)",
]
y_c = 6.5
for line in code:
    col = TEAL if line.startswith("#") else (ORANGE if "loss" in line.lower() and "#" not in line else TEXT)
    ax.text(10.65, y_c, line, fontsize=7, color=col, fontfamily="monospace", va="top")
    y_c -= 0.28

ax.set_title("Actor-Critic: Two Networks, One Shared Advantage Signal\n"
             "Actor (policy π_θ) + Critic (value V_φ) trained simultaneously",
             fontsize=12, fontweight="bold", color=TEXT, pad=12)
plt.tight_layout(pad=2.5)
save("14_actor_critic_flow.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  15 ▸ PPO CLIPPED OBJECTIVE  —  detailed annotated plot
# ═══════════════════════════════════════════════════════════════════════════════
section("15 · PPO Clipped Objective Detailed")

fig, axes = plt.subplots(1, 2, figsize=(16.0, 8.0))
fig.patch.set_facecolor(BG0)
eps = 0.2
ratios = np.linspace(0.3, 2.2, 500)

for ax, A, lbl in zip(axes, [1.0, -1.0],
                       ["Positive Advantage  A > 0  (GOOD action → increase prob)",
                        "Negative Advantage  A < 0  (BAD action → decrease prob)"]):
    ax.set_facecolor(BG1)
    raw_obj     = ratios * A
    clipped_r   = np.clip(ratios, 1-eps, 1+eps)
    clipped_obj = clipped_r * A
    ppo_obj     = np.minimum(raw_obj, clipped_obj)

    ax.plot(ratios, raw_obj,     color=BLUE,   lw=2.2, ls="--", label="r(θ)·A  (unclipped)")
    ax.plot(ratios, clipped_obj, color=ORANGE, lw=2.2,          label=f"clip(r,{1-eps},{1+eps})·A")
    ax.plot(ratios, ppo_obj,     color=RED,    lw=3.5,          label="PPO  min(…)  ← actual signal")

    # Trust region
    ax.axvspan(1-eps, 1+eps, alpha=0.08, color=TEAL)
    ax.axvline(1.0,   color=MUTED, ls=":",  lw=1.5, alpha=0.8)
    ax.axvline(1-eps, color=TEAL,  ls="--", lw=1.3, alpha=0.6)
    ax.axvline(1+eps, color=TEAL,  ls="--", lw=1.3, alpha=0.6)

    # Annotations
    ax.text(1.0, ax.get_ylim()[0]*0.7 if A < 0 else ax.get_ylim()[1]*0.95,
            "r=1\n(no change)", ha="center", fontsize=8, color=MUTED)
    ax.text(1-eps-0.05, 0, f"{1-eps:.1f}", ha="right", fontsize=9, color=TEAL, fontweight="bold")
    ax.text(1+eps+0.05, 0, f"{1+eps:.1f}", ha="left",  fontsize=9, color=TEAL, fontweight="bold")
    ax.text(1.0, 0.05 if A > 0 else -0.05, "  Trust\n  Region", ha="left",
            fontsize=8, color=TEAL, style="italic")

    # Region annotations
    if A > 0:
        ax.annotate("Clamped:\nalready increased\nenough → stop!",
                    xy=(1.8, ppo_obj[np.argmin(np.abs(ratios-1.8))]),
                    xytext=(1.6, 0.4), fontsize=8, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    else:
        ax.annotate("Clamped:\nalready decreased\nenough → stop!",
                    xy=(0.5, ppo_obj[np.argmin(np.abs(ratios-0.5))]),
                    xytext=(0.4, 0.4), fontsize=8, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax.set_xlabel("Probability ratio  r(θ) = π_new / π_old")
    ax.set_ylabel("Gradient signal (objective value)")
    ax.set_title(lbl, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3); ax.axhline(0, color=MUTED, lw=0.8)

fig.suptitle("PPO Clipped Surrogate Objective  (ε=0.2)\n"
             "Prevents policy from changing too fast — no matter how large the advantage",
             fontsize=13, fontweight="bold", color=TEXT, y=1.02)
plt.tight_layout(pad=2.5)
save("15_ppo_clipped_objective.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  16 ▸ DQN LEARNING CURVES  —  4 algorithms comparison
# ═══════════════════════════════════════════════════════════════════════════════
section("16 · DQN Learning Curves")

np.random.seed(7)
eps_c = np.arange(0, 601)

def scurve(center, steepness, noise, maxval=200, seed=7):
    np.random.seed(seed)
    raw = maxval / (1 + np.exp(-steepness*(eps_c-center))) + np.random.randn(len(eps_c))*noise
    return np.clip(raw, 0, maxval)

def roll(arr, w=25):
    return np.convolve(arr, np.ones(w)/w, mode="valid")

curves = [
    (scurve(300, 0.018, 22, 185, 1), RED,    "REINFORCE",    "--"),
    (scurve(250, 0.022, 14, 200, 2), ORANGE, "DQN",          "-"),
    (scurve(210, 0.026, 11, 200, 3), BLUE,   "Double DQN",   "-"),
    (scurve(185, 0.030, 10, 200, 4), TEAL,   "Dueling DQN",  "-"),
]
x_r = eps_c[12:-12]

fig, ax = plt.subplots(figsize=(14.0, 8.0))
ax.set_facecolor(BG1)
fig.patch.set_facecolor(BG0)

for (raw, col, lbl, ls) in curves:
    sm = roll(raw)
    lw = 2 if ls == "--" else 2.5
    ax.plot(x_r, sm, color=col, lw=lw, ls=ls, label=lbl)
    # Add shaded std band
    if ls != "--":
        ax.fill_between(x_r, sm-12, sm+12, alpha=0.07, color=col)

ax.axhline(195, color=MUTED, ls=":", lw=1.5, alpha=0.8)
ax.text(605, 195, "  195\n  (solved)", va="center", fontsize=8.5, color=MUTED)

# Annotation: convergence speeds
converge_eps = [330, 270, 230, 200]
for (_, col, lbl, ls), ep in zip(curves, converge_eps):
    ax.annotate(f"{ep} ep", xy=(ep, 195), xytext=(ep, 165),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3),
                fontsize=8, color=col, ha="center")

ax.set_xlabel("Episode", fontsize=11); ax.set_ylabel("Mean Reward (25-ep rolling avg)", fontsize=11)
ax.set_title("CartPole-v1: Algorithm Convergence Comparison\n"
             "Dueling DQN fastest  |  REINFORCE slowest (high variance, no replay)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 600); ax.set_ylim(0, 215)
plt.tight_layout(pad=2.5)
save("16_dqn_learning_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  17 ▸ A3C PARALLEL WORKERS  —  architecture diagram
# ═══════════════════════════════════════════════════════════════════════════════
section("17 · A3C Parallel Workers")

fig, ax = plt.subplots(figsize=(16.0, 9.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

# Global network
rounded_box(ax, 5.5, 2.8, 3.0, 1.5, "GLOBAL\nNETWORK  θ_global", YELLOW, fc="#1a1400")
ax.text(7.0, 4.6, "Shared parameters: θ_global, φ_global", ha="center",
        fontsize=9, color=YELLOW, style="italic")

# Worker boxes
worker_positions = [(0.3, 4.8), (0.3, 2.5), (0.3, 0.2), (10.2, 4.8), (10.2, 2.5), (10.2, 0.2)]
worker_envs      = ["Env_1", "Env_2", "Env_3", "Env_4", "Env_5", "Env_6"]
worker_colors    = [BLUE, RED, TEAL, PURPLE, ORANGE, PINK]

for (x, y), env, col in zip(worker_positions, worker_envs, worker_colors):
    rounded_box(ax, x, y, 2.8, 2.0,
                f"Worker\nπ_θ_local\n{env}", col, fc=BG2, fs=8.5)
    # Push gradients
    arrow_x2 = 5.5 if x < 7 else 8.5
    arrow_y2 = 3.55
    arr(ax, x+2.8 if x < 7 else x, y+1.0, arrow_x2, arrow_y2, col, lw=1.5)
    ax.text((x+2.8+arrow_x2)/2 + (0 if x<7 else 0),
            (y+1.0+arrow_y2)/2 + 0.25,
            "push\ngrads", ha="center", fontsize=7, color=col)

    # Pull weights
    arrow_x1 = 8.5 if x > 7 else 5.5
    arr(ax, arrow_x1, 3.1, x+2.8 if x < 7 else x, y+0.5, col, lw=1.2)
    ax.text((arrow_x1+x+2.8 if x<7 else arrow_x1+x)/2,
            (3.1+y+0.5)/2 - 0.25,
            "pull θ", ha="center", fontsize=7, color=MUTED)

# Key insights
insights = [
    "• Each worker runs its own copy of the environment independently",
    "• Workers collect n-step experience, compute gradients LOCALLY",
    "• Gradients pushed to global network ASYNCHRONOUSLY (no lock!)",
    "• Global weights pulled at start of each worker episode",
    "• N workers ≈ N× more gradient updates per wall-clock second",
    "• Decorrelated experience: different env copies → diverse data",
]
ax.add_patch(FancyBboxPatch((4.5, -0.1), 9.3, 2.7, boxstyle="round,pad=0.12",
             facecolor=BG2, edgecolor=YELLOW, lw=1.5))
for i, line in enumerate(insights):
    ax.text(4.7, 2.45 - i*0.41, line, fontsize=8.5, color=TEXT, va="top")

ax.set_title("A3C: Asynchronous Advantage Actor-Critic\n"
             "N parallel workers push gradients to shared global network asynchronously",
             fontsize=12, fontweight="bold", color=TEXT, pad=12)
plt.tight_layout(pad=2.5)
save("17_a3c_parallel_workers.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  18 ▸ PPO TRAINING LOOP  —  rollout → multi-epoch annotated
# ═══════════════════════════════════════════════════════════════════════════════
section("18 · PPO Training Loop")

fig, ax = plt.subplots(figsize=(16.0, 8.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

# Phases
phases = [
    (0.2, 1.5, 2.8, 3.0, "COLLECT\nRollout\n(N_steps=2048)", TEAL,
     "Run env under\ncurrent π_θ_old\nStore: s,a,r,s',done"),
    (3.5, 1.5, 2.8, 3.0, "COMPUTE\nAdvantages\n(GAE λ=0.95)", PURPLE,
     "Backward pass\nthrough episode\nA_t = Σ(γλ)^l·δ_l"),
    (6.8, 1.5, 2.8, 3.0, "MINI-BATCH\nShuffle\n(batch=64)", BLUE,
     "Shuffle rollout\nCreate K mini-batches\nfrom 2048 transitions"),
    (10.1, 1.5, 2.8, 3.0, "PPO UPDATE\n×10 epochs\n(reuse data!)", RED,
     "Clip ratio r(θ)\nUpdate Actor + Critic\nCheck KL divergence"),
]
for x, y, w, h, name, col, detail in phases:
    rounded_box(ax, x, y, w, h, name, col, fc=BG2)
    ax.text(x+w/2, y+h+0.18, detail, ha="center", fontsize=7.5,
            color=MUTED, style="italic")
    if name != phases[-1][3]:
        arr(ax, x+w, y+h/2, x+w+0.5, y+h/2, MUTED, lw=2)

# Key advantage text
ax.add_patch(FancyBboxPatch((0.1, 0.2), 13.8, 1.0, boxstyle="round,pad=0.1",
             facecolor="#0a1a00", edgecolor=TEAL, lw=1.5))
ax.text(7.0, 0.7,
        "KEY: PPO reuses the SAME rollout for 10 gradient epochs  →  10× more updates per env interaction!\n"
        "Standard policy gradient uses data ONCE and discards it. PPO safely reuses it with the clipping constraint.",
        ha="center", va="center", fontsize=9, color=TEAL)

# Phase numbers
for i, (x, y, w, h, *_) in enumerate(phases):
    circle = Circle((x+w/2, y+h+0.68), 0.2, facecolor=phases[i][5], zorder=5)
    ax.add_patch(circle)
    ax.text(x+w/2, y+h+0.68, str(i+1), ha="center", va="center",
            fontsize=9, fontweight="bold", color=BG0, zorder=6)

# Loop arrow
ax.annotate("", xy=(0.6, 4.7), xytext=(12.9, 4.7),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5,
                            connectionstyle="arc3,rad=-0.4"))
ax.text(7.0, 5.5, "← Repeat for M iterations  (policy improves each loop)", ha="center",
        fontsize=9, color=ORANGE, style="italic")

ax.set_title("PPO Training Loop — Sample-Efficient Policy Gradient\n"
             "Collect 2048 steps  →  Compute GAE  →  Shuffle  →  10 gradient epochs",
             fontsize=12, fontweight="bold", color=TEXT, pad=12)
plt.tight_layout(pad=2.5)
save("18_ppo_training_loop.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  19 ▸ ALPHAZERO MCTS LOOP  —  self-play diagram
# ═══════════════════════════════════════════════════════════════════════════════
section("19 · AlphaZero MCTS Loop")

fig, ax = plt.subplots(figsize=(16.0, 9.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

# Steps in a cycle
cycle_boxes = [
    (6.0, 5.0, 2.0, 1.2, "Self-Play\n(MCTS + π,V net)", BLUE),
    (10.0, 3.0, 2.0, 1.2, "Store\n(s, π_MCTS, z)", TEAL),
    (8.0, 0.8, 2.0, 1.2, "Train Network\non stored games", RED),
    (3.5, 3.0, 2.0, 1.2, "Evaluate\nnew vs old π", PURPLE),
]
for x, y, w, h, lbl, col in cycle_boxes:
    rounded_box(ax, x, y, w, h, lbl, col, fc=BG2)

# Cycle arrows
cycle_coords = [(7.0, 5.0), (11.0, 3.6), (9.0, 0.8), (4.5, 3.6)]
for i in range(len(cycle_coords)):
    x1, y1 = cycle_coords[i]
    x2, y2 = cycle_coords[(i+1) % len(cycle_coords)]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5,
                                connectionstyle="arc3,rad=0.3", mutation_scale=16))

# MCTS sub-diagram
ax.add_patch(FancyBboxPatch((0.2, 3.5), 3.0, 3.3, boxstyle="round,pad=0.12",
             facecolor=BG2, edgecolor=BLUE, lw=2))
ax.text(1.7, 6.55, "MCTS per Move", ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
mcts_steps = [
    "① Select: UCB = Q + c·P·√N/(1+n)",
    "② Expand: neural net prior P(a|s)",
    "③ Evaluate: V(s) from neural net",
    "④ Backup: update Q along path",
    "   Repeat 800×, then pick action",
]
for i, step in enumerate(mcts_steps):
    col = TEAL if i == 0 else (ORANGE if i == 1 else (PURPLE if i == 2 else
          (RED if i == 3 else MUTED)))
    ax.text(0.35, 6.1 - i*0.52, step, fontsize=8, color=col)

# Network output box
ax.add_patch(FancyBboxPatch((10.0, 5.0), 3.8, 1.8, boxstyle="round,pad=0.12",
             facecolor=BG2, edgecolor=YELLOW, lw=2))
ax.text(11.9, 6.55, "Neural Network", ha="center", fontsize=9.5, color=YELLOW, fontweight="bold")
ax.text(10.15, 6.1, "Input: board state  s", fontsize=8.5, color=TEXT)
ax.text(10.15, 5.7, "Output 1: P(a|s)  (policy prior)", fontsize=8.5, color=RED)
ax.text(10.15, 5.3, "Output 2: V(s)    (win probability)", fontsize=8.5, color=BLUE)
ax.text(10.15, 4.9, "Loss = (z−V)²  +  CE(π_MCTS, P)", fontsize=8, color=ORANGE)

# Training data label
ax.add_patch(FancyBboxPatch((0.2, 0.2), 3.0, 3.0, boxstyle="round,pad=0.12",
             facecolor=BG2, edgecolor=RED, lw=2))
ax.text(1.7, 3.0, "Training Targets", ha="center", fontsize=9.5, color=RED, fontweight="bold")
targets = [
    "s  : board position",
    "π_MCTS : visit distribution",
    "         from 800 simulations",
    "z : game outcome",
    "    +1=win, 0=draw, -1=loss",
    "",
    "No human data needed!",
    "Start from random play.",
]
for i, t in enumerate(targets):
    col = TEAL if "No human" in t or "random" in t else TEXT
    ax.text(0.35, 2.65 - i*0.32, t, fontsize=8, color=col)

ax.set_title("AlphaZero: Self-Play + MCTS + Neural Network — Closed-Loop Learning\n"
             "Achieves superhuman Chess/Shogi/Go with ZERO human knowledge",
             fontsize=12, fontweight="bold", color=TEXT, pad=12)
plt.tight_layout(pad=2.5)
save("19_alphazero_mcts_loop.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  20 ▸ RL ALGORITHM TAXONOMY  —  visual tree
# ═══════════════════════════════════════════════════════════════════════════════
section("20 · RL Algorithm Taxonomy")

fig, ax = plt.subplots(figsize=(17.0, 10.0))
ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
fig.patch.set_facecolor(BG0); ax.set_facecolor(BG0)

def tbox(ax, x, y, w, h, text, col, fc=BG2, fs=9.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=fc, edgecolor=col, lw=2, zorder=3)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=col, zorder=4)

def line(ax, x1, y1, x2, y2, col=MUTED, lw=1.5):
    ax.plot([x1, x2], [y1, y2], color=col, lw=lw, zorder=2)

# Root
tbox(ax, 5.5, 7.0, 4.0, 0.85, "REINFORCEMENT LEARNING", TEXT, fc="#0a0a0a", fs=11)

# Level 1
tbox(ax, 0.2, 5.4, 4.0, 0.85, "MODEL-FREE", BLUE, fc="#001020")
tbox(ax, 5.5, 5.4, 4.0, 0.85, "MODEL-BASED", ORANGE, fc="#1a0800")
tbox(ax, 10.8, 5.4, 4.0, 0.85, "OFF-POLICY HYBRID", PURPLE, fc="#100020")

# Lines from root
for x2c in [2.2, 7.5, 12.8]:
    line(ax, 7.5, 7.0, x2c, 6.25)

# Level 2: model-free sub-types
tbox(ax, 0.1, 3.8, 1.8, 0.8, "Value-\nBased", BLUE, "#001520")
tbox(ax, 2.1, 3.8, 1.8, 0.8, "Policy-\nBased", RED, "#200010")
tbox(ax, 4.1, 3.8, 2.2, 0.8, "Actor-\nCritic", PURPLE, "#100020")
for x2c in [1.0, 3.0, 5.2]:
    line(ax, 2.2, 5.4, x2c, 4.6)

# Level 3: algorithms
algo_rows = [
    # (x, y, label, parent_x, col)
    (0.05, 2.1, "DQN\n(off-policy)", 1.0, BLUE),
    (1.15, 2.1, "Double\nDQN", 1.0, BLUE),
    (2.25, 2.1, "Dueling\nDQN", 1.0, BLUE),
    (1.5, 2.1,  "REINFORCE\n(on-policy)", 3.0, RED),
    (3.8, 2.1,  "A2C/A3C\n(parallel)", 5.2, PURPLE),
    (5.4, 2.1,  "PPO  ⭐\n(clipped)", 5.2, PURPLE),
    (6.9, 2.1,  "SAC\n(max-entropy)", 5.2, PURPLE),
]
# Rewrite more carefully
algos = [
    (0.05, 2.1, 1.55, 0.85, "DQN",       BLUE,   1.0),
    (1.75, 2.1, 1.55, 0.85, "Double DQN",BLUE,   1.0),
    (2.0,  0.9, 1.55, 0.85, "Dueling\nDQN", BLUE, 2.2),
    (2.2,  2.1, 1.55, 0.85, "REINFORCE", RED,    3.0),
    (3.9,  2.1, 1.7,  0.85, "A2C / A3C", PURPLE, 5.2),
    (5.75, 2.1, 1.55, 0.85, "PPO  ⭐",   PURPLE, 5.2),
    (7.4,  2.1, 1.55, 0.85, "SAC",       PURPLE, 5.2),
]
for x, y, w, h, name, col, px in algos:
    tbox(ax, x, y, w, h, name, col, fs=8.5)
    line(ax, px, 3.8, x+w/2, y+h, col)

# Model-based algorithms
tbox(ax, 6.5, 3.8, 1.8, 0.8, "Dyna", ORANGE, "#1a0800")
tbox(ax, 8.4, 3.8, 1.8, 0.8, "AlphaZero\n(MCTS)", ORANGE, "#1a0800")
for xc in [7.4, 9.3]:
    line(ax, 7.5, 5.4, xc, 4.6)
tbox(ax, 6.5, 2.1, 1.55, 0.85, "World\nModels", ORANGE, "#1a0800")
tbox(ax, 8.3, 2.1, 1.55, 0.85, "Dreamer", ORANGE, "#1a0800")
line(ax, 7.4, 3.8, 7.3, 2.95, ORANGE)
line(ax, 9.3, 3.8, 9.0, 2.95, ORANGE)

# Hybrid
tbox(ax, 10.9, 3.8, 1.7, 0.8, "TD3", PURPLE, "#100020")
tbox(ax, 12.8, 3.8, 1.7, 0.8, "DDPG", PURPLE, "#100020")
for xc in [11.75, 13.65]:
    line(ax, 12.8, 5.4, xc, 4.6)

# Legend
legend_data = [
    (BLUE, "Value-Based  (DQN family)  → Discrete actions"),
    (RED, "Policy-Based  (REINFORCE)  → Direct π optimization"),
    (PURPLE, "Actor-Critic  (PPO, SAC)   → Best of both worlds"),
    (ORANGE, "Model-Based   (AlphaZero)  → Plan with learned dynamics"),
]
for i, (col, lbl) in enumerate(legend_data):
    ax.add_patch(Rectangle((10.9, 2.8 - i*0.38), 0.3, 0.25, facecolor=col))
    ax.text(11.3, 2.92 - i*0.38, lbl, fontsize=8, color=TEXT, va="center")

ax.set_title("Reinforcement Learning Algorithm Taxonomy\n"
             "⭐ PPO is the go-to general-purpose algorithm (OpenAI, Google, Meta)",
             fontsize=13, fontweight="bold", color=TEXT, pad=14)
plt.tight_layout(pad=2.5)
save("20_rl_algorithm_taxonomy.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Final summary
# ═══════════════════════════════════════════════════════════════════════════════
filenames = [
    "01_rl_interaction_loop.png",
    "02_discount_factor_gamma.png",
    "03_epsilon_greedy_visual.png",
    "04_bellman_equation_flow.png",
    "05_q_table_update.png",
    "06_reinforce_episode_walkthrough.png",
    "07_backup_diagrams_td_mc_dp.png",
    "08_bias_variance_spectrum.png",
    "09_dqn_architecture_annotated.png",
    "10_experience_replay_buffer.png",
    "11_target_network_mechanism.png",
    "12_double_dqn_overestimation.png",
    "13_dueling_dqn_architecture.png",
    "14_actor_critic_flow.png",
    "15_ppo_clipped_objective.png",
    "16_dqn_learning_curves.png",
    "17_a3c_parallel_workers.png",
    "18_ppo_training_loop.png",
    "19_alphazero_mcts_loop.png",
    "20_rl_algorithm_taxonomy.png",
]

print("\n" + "═"*65)
print("  ✅  ALL 20 VISUALS GENERATED  →  Visuals/ directory")
print("═"*65)
for i, f in enumerate(filenames, 1):
    print(f"  {i:2d}.  {f}")
print("═"*65)
