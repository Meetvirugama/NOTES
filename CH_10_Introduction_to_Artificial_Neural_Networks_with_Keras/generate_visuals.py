"""
╔══════════════════════════════════════════════════════════════════╗
║   CH 10: ANN with Keras — COMPLETE Visual Generator v2          ║
║   25 real matplotlib graphs for all 7 modules                   ║
║   Run: python3 generate_visuals.py                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
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
# EXISTING GRAPHS (1-10) — kept from v1
# ══════════════════════════════════════════════════════════════════════════════

def plot_activation_functions():
    print("\n[01] Activation Functions")
    z = np.linspace(-5, 5, 400)
    funcs  = [lambda z: np.where(z>=0,1.,0.), lambda z:1/(1+np.exp(-z)),
              np.tanh, lambda z:np.maximum(0,z),
              lambda z:np.where(z>=0,z,0.01*z), lambda z:np.where(z>=0,z,np.exp(z)-1)]
    names  = ["Step", "Sigmoid σ(z)", "Tanh", "ReLU ⭐ (Default)", "Leaky ReLU", "ELU"]
    colors = [R1, B1, P1, G1, GOLD, O1]
    ranges = ["{0,1}", "(0,1)", "(-1,1)", "[0,∞)", "(-∞,∞)", "(-1,∞)"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Activation Functions — Complete Visual Guide", fontsize=18, fontweight="bold", color=TX, y=1.01)
    for ax, f, name, c, rng in zip(axes.flat, funcs, names, colors, ranges):
        y = f(z)
        ax.plot(z, y, color=c, lw=2.8, zorder=3)
        ax.fill_between(z, 0, y, alpha=0.12, color=c)
        ax.axhline(0, color=TX2, lw=0.8, ls="--"); ax.axvline(0, color=TX2, lw=0.8, ls="--")
        ax.set_title(name, fontsize=13, fontweight="bold", color=c, pad=8)
        ax.set_xlim(-5,5); ax.grid(True)
        ax.text(0.98, 0.05, f"Range: {rng}", transform=ax.transAxes,
                ha="right", fontsize=8, color=TX2,
                bbox=dict(fc=CARD, ec=c, alpha=0.7, boxstyle="round"))
    plt.tight_layout(); save("01_activation_functions.png")

def plot_mlp_architecture():
    print("[02] MLP Architecture")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis("off")
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    fig.suptitle("Multi-Layer Perceptron (MLP) — Architecture", fontsize=17, fontweight="bold", color=TX)
    layers = [{"x":1.2,"n":4,"label":"Input Layer\n(x₁…x₄)","c":B1},
              {"x":3.8,"n":5,"label":"Hidden Layer 1\n(ReLU)","c":G1},
              {"x":6.3,"n":5,"label":"Hidden Layer 2\n(ReLU)","c":P1},
              {"x":8.8,"n":3,"label":"Output Layer\n(Softmax)","c":O1}]
    positions = []
    for ld in layers:
        ys = np.linspace(1.0, 7.0, ld["n"])
        positions.append([(ld["x"], y) for y in ys])
    for li in range(len(positions)-1):
        for (x1,y1) in positions[li]:
            for (x2,y2) in positions[li+1]:
                ax.plot([x1,x2],[y1,y2], color=TX2, lw=0.4, alpha=0.35, zorder=1)
    for li, ld in enumerate(layers):
        for (x,y) in positions[li]:
            node(ax, x, y, r=0.28, color=ld["c"])
        ax.text(ld["x"], 0.1, ld["label"], ha="center", fontsize=10,
                color=ld["c"], fontweight="bold")
    for i in range(len(layers)-1):
        ax.annotate("", xy=(layers[i+1]["x"]-0.35, 4), xytext=(layers[i]["x"]+0.35, 4),
                    arrowprops=dict(arrowstyle="-|>", color=TX, lw=1.8))
    ax.text(5, 7.7, "Fully Connected — Every neuron connects to every neuron in next layer",
            ha="center", fontsize=10, color=TX2, style="italic")
    legend = [mpatches.Patch(color=ld["c"], label=ld["label"].split("\n")[0]) for ld in layers]
    ax.legend(handles=legend, loc="upper right", framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    save("02_mlp_architecture.png")

def plot_training_curves():
    print("[03] Training / Validation Curves")
    np.random.seed(42); ep = np.arange(1,51)
    tg = 1.2*np.exp(-0.12*ep)+0.08+np.random.normal(0,.010,50)
    vg = 1.3*np.exp(-0.10*ep)+0.12+np.random.normal(0,.020,50)
    to = 1.2*np.exp(-0.18*ep)+0.03+np.random.normal(0,.010,50)
    vo = np.where(ep<20, 1.3*np.exp(-0.08*ep)+0.15, 0.6+0.008*(ep-20))+np.random.normal(0,.015,50)
    td = np.abs(np.sin(ep*0.5))*1.5+0.8
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Training Dynamics — Three Scenarios", fontsize=16, fontweight="bold", color=TX)
    configs = [("✅ Good Training", tg, vg, G1, B1),
               ("⚠️ Overfitting",   to, vo, R1, GOLD),
               ("❌ High LR (Diverging)", td, td*1.1+0.3, R1, O1)]
    for ax,(title,tr,vl,ct,cv) in zip(axes, configs):
        ax.plot(ep, tr, color=ct, lw=2.2, label="Train Loss")
        ax.plot(ep, vl, color=cv, lw=2.2, ls="--", label="Val Loss")
        ax.set_title(title, fontsize=12, fontweight="bold", color=TX)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2); ax.grid(True)
    axes[1].axvline(20, color=R1, ls=":", lw=2)
    axes[1].text(21, 0.55, "← Early\nStop Here!", fontsize=8, color=R1, fontweight="bold")
    plt.tight_layout(); save("03_training_curves.png")

def plot_gradient_descent():
    print("[04] Gradient Descent")
    x = np.linspace(-3,3,300)
    y = x**2 + 0.5*np.sin(3*x)*x
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("Gradient Descent — Learning Rate Effect", fontsize=16, fontweight="bold", color=TX)
    scenarios = [("❌ Too High η=0.9", R1,  [2.8,-2.5,2.1,-1.9,1.7]),
                 ("✅ Just Right η=0.1",G1, [2.8,1.8,1.0,0.4,0.1,0.01]),
                 ("⚠️ Too Low η=0.005",GOLD,[2.8,2.6,2.4,2.2,2.0,1.8])]
    for ax,(title,c,pts) in zip(axes, scenarios):
        ax.plot(x,y,color=B1,lw=2.5,zorder=2)
        ax.fill_between(x,y.min()-0.5,y,alpha=0.10,color=B1)
        ys=[xi**2+0.5*np.sin(3*xi)*xi for xi in pts]
        ax.scatter(pts,ys,color=c,s=90,zorder=5)
        for i in range(len(pts)-1):
            ax.annotate("",xy=(pts[i+1],ys[i+1]),xytext=(pts[i],ys[i]),
                        arrowprops=dict(arrowstyle="-|>",color=c,lw=1.8))
        ax.set_title(title, fontsize=11, fontweight="bold", color=c)
        ax.set_xlabel("Weight θ"); ax.set_ylabel("Loss"); ax.grid(True)
    plt.tight_layout(); save("04_gradient_descent.png")

def plot_backprop():
    print("[05] Backpropagation Flow")
    fig, ax = plt.subplots(figsize=(14,7))
    ax.set_xlim(0,14); ax.set_ylim(0,7); ax.axis("off")
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    fig.suptitle("Backpropagation — Forward & Backward Pass", fontsize=16, fontweight="bold", color=TX)
    blocks = [(1.5,3.5,"INPUT\nx",B1),(4.0,3.5,"Layer 1\nW₁,b₁\nReLU",G1),
              (7.0,3.5,"Layer 2\nW₂,b₂\nReLU",P1),(10.0,3.5,"Output\nSoftmax\nŷ",O1),
              (12.5,3.5,"LOSS\nL(y,ŷ)",R1)]
    for (x,y,label,c) in blocks:
        box(ax,x,y,1.7,2.0,color=c,label=label,fontsize=10)
    xs = [b[0] for b in blocks]
    for i in range(len(xs)-1):
        ax.annotate("",xy=(xs[i+1]-0.9,5.0),xytext=(xs[i]+0.9,5.0),
                    arrowprops=dict(arrowstyle="-|>",color=G1,lw=2.2))
    for i in range(len(xs)-1,0,-1):
        ax.annotate("",xy=(xs[i-1]+0.9,1.5),xytext=(xs[i]-0.9,1.5),
                    arrowprops=dict(arrowstyle="-|>",color=R1,lw=2.2))
    ax.text(7,5.5,"➡  FORWARD PASS  (compute predictions)",ha="center",fontsize=12,color=G1,fontweight="bold")
    ax.text(7,0.8,"⬅  BACKWARD PASS  (propagate gradients — Chain Rule)",ha="center",fontsize=12,color=R1,fontweight="bold")
    for gx,gl in zip([2.7,5.5,8.5,11.3],["∂L/∂W₁","∂L/∂W₂","∂L/∂ŷ","∂L/∂L=1"]):
        ax.text(gx,1.2,gl,ha="center",fontsize=9,color=R1,style="italic")
    ax.text(7,0.1,"Chain Rule: ∂L/∂W = ∂L/∂ŷ × ∂ŷ/∂h × ∂h/∂W",
            ha="center",fontsize=10,color=GOLD,fontweight="bold")
    save("05_backpropagation.png")

def plot_lr_finder():
    print("[06] LR Finder")
    np.random.seed(7); lr_log=np.linspace(-5,0,300); lr=10**lr_log
    loss=(2.5-1.8*np.tanh((lr_log+2.5)*2.2)+0.5*np.maximum(0,lr_log+1.5)**2
          +np.random.normal(0,0.05,300))
    opt_idx=np.argmin(loss)-30; best_lr=lr[opt_idx]
    fig,ax=plt.subplots(figsize=(12,6))
    fig.suptitle("Learning Rate Finder — LR Range Test",fontsize=16,fontweight="bold",color=TX)
    ax.plot(lr,loss,color=B1,lw=2.5); ax.fill_between(lr,loss,loss.max(),alpha=0.10,color=B1)
    ax.set_xscale("log"); ax.set_xlabel("Learning Rate (log scale)"); ax.set_ylabel("Loss"); ax.grid(True)
    ax.axvspan(lr[opt_idx-20],lr[opt_idx+10],alpha=0.18,color=G1)
    ax.axvline(best_lr,color=G1,lw=2,ls="--")
    ax.text(best_lr*1.5,loss.max()*0.85,f"✅ Optimal LR ≈ {best_lr:.1e}",
            fontsize=10,color=G1,fontweight="bold",
            bbox=dict(fc=CARD,ec=G1,alpha=0.8,boxstyle="round"))
    div_idx=np.argmax(np.gradient(loss)>0.05)
    ax.axvline(lr[div_idx],color=R1,lw=2,ls="--")
    ax.text(lr[div_idx]*1.1,loss.max()*0.6,"❌ Divergence\nstarts here",fontsize=9,color=R1,fontweight="bold")
    ax.text(1e-5,loss[0]*0.95,"⚠️ Too slow",fontsize=9,color=GOLD,fontweight="bold")
    plt.tight_layout(); save("06_lr_finder.png")

def plot_gradient_flow():
    print("[07] Vanishing & Exploding Gradients")
    layers=np.arange(1,11)
    van=0.5**layers*10; exp=1.5**layers*0.01; rl=np.ones(10)*0.8+np.random.normal(0,0.05,10)
    fig,ax=plt.subplots(figsize=(12,6))
    fig.suptitle("Gradient Flow Across Layers",fontsize=16,fontweight="bold",color=TX)
    ax.plot(layers,van,color=R1,lw=2.5,marker="o",ms=7,label="Vanishing (sigmoid layers)")
    ax.plot(layers,exp,color=GOLD,lw=2.5,marker="s",ms=7,label="Exploding (bad init)")
    ax.plot(layers,rl, color=G1, lw=2.5,marker="^",ms=7,label="Healthy (ReLU)")
    ax.fill_between(layers,0,rl,alpha=0.08,color=G1)
    ax.axhline(0.05,color=R1,ls=":",lw=1.2,alpha=0.7)
    ax.text(10.1,0.05,"Danger\nZone",fontsize=8,color=R1,va="center")
    ax.set_xlabel("Layer (output → input)"); ax.set_ylabel("Gradient Magnitude")
    ax.legend(framealpha=0.3,facecolor=CARD,edgecolor=TX2,fontsize=11)
    ax.grid(True); ax.set_yscale("log")
    ax.annotate("≈ 0! Weights frozen",xy=(8,van[7]),xytext=(5,0.02),fontsize=9,color=R1,
                arrowprops=dict(arrowstyle="->",color=R1))
    ax.annotate("Explodes → NaN!",xy=(8,exp[7]),xytext=(6,100),fontsize=9,color=GOLD,
                arrowprops=dict(arrowstyle="->",color=GOLD))
    plt.tight_layout(); save("07_gradient_flow.png")

def plot_hyperparameter_effects():
    print("[08] Hyperparameter Effects")
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    fig.suptitle("Hyperparameter Effects on Training",fontsize=16,fontweight="bold",color=TX)
    bs=[2,4,8,16,32,64,128,256,512,1024]
    ta=[0.94,0.95,0.96,0.97,0.975,0.978,0.98,0.981,0.982,0.983]
    va=[0.91,0.93,0.94,0.95,0.955,0.952,0.945,0.93,0.91,0.88]
    ax=axes[0]; ax2=ax.twinx()
    ax.plot(bs,ta,color=G1,lw=2.5,marker="o",ms=6,label="Train Acc")
    ax.plot(bs,va,color=B1,lw=2.5,marker="s",ms=6,label="Val Acc",ls="--")
    ax2.bar(bs,[t-v for t,v in zip(ta,va)],alpha=0.22,color=R1,width=[b*0.3 for b in bs])
    ax2.set_ylabel("Train-Val Gap",color=R1); ax2.tick_params(axis="y",colors=R1)
    ax.axvline(32,color=GOLD,ls=":",lw=2); ax.text(32,0.88,"  32\n  ⭐",fontsize=9,color=GOLD,fontweight="bold")
    ax.set_xscale("log",base=2); ax.set_xlabel("Batch Size"); ax.set_ylabel("Accuracy")
    ax.set_title("Batch Size vs Accuracy",fontsize=13,color=TX)
    ax.legend(loc="lower left",framealpha=0.3,facecolor=CARD,edgecolor=TX2); ax.grid(True)
    ns=[10,25,50,100,150,200,300,500,800,1000]
    vn=[0.80,0.87,0.91,0.94,0.955,0.962,0.967,0.968,0.969,0.969]
    tt=[0.5,1.0,1.8,3.0,4.5,5.5,7.5,11.0,16.0,19.5]
    ax=axes[1]; ax3=ax.twinx()
    ax.plot(ns,vn,color=P1,lw=2.5,marker="^",ms=7,label="Val Accuracy")
    ax.fill_between(ns,0.75,vn,alpha=0.10,color=P1)
    ax3.plot(ns,tt,color=O1,lw=2,ls="--",marker="D",ms=5,label="Train Time")
    ax3.set_ylabel("Train Time (s/epoch)",color=O1); ax3.tick_params(axis="y",colors=O1)
    ax.axvline(128,color=GOLD,ls=":",lw=2); ax.text(130,0.77,"128\n⭐",fontsize=9,color=GOLD,fontweight="bold")
    ax.set_xlabel("Neurons/Layer"); ax.set_ylabel("Val Accuracy")
    ax.set_title("Neurons per Layer — Diminishing Returns",fontsize=13,color=TX)
    ax.legend(loc="lower right",framealpha=0.3,facecolor=CARD,edgecolor=TX2); ax.grid(True)
    plt.tight_layout(); save("08_hyperparameter_effects.png")

def plot_ann_timeline():
    print("[09] ANN History Timeline")
    events=[
        (1943,"McCulloch & Pitts\nFirst math neuron model",B1,True),
        (1958,"Rosenblatt\nPerceptron invented",G1,False),
        (1969,"Minsky & Papert\nXOR → AI Winter",R1,True),
        (1986,"Rumelhart et al.\nBackpropagation",P1,False),
        (2006,"Hinton\nDeep Belief Nets",O1,True),
        (2012,"AlexNet\nImageNet Revolution",GOLD,False),
        (2017,"Transformers\nAttention is All You Need",B1,True),
        (2022,"ChatGPT / GPT-4\nLLM Revolution 🚀",G1,False),
    ]
    fig,ax=plt.subplots(figsize=(17,7))
    fig.suptitle("History of Artificial Neural Networks",fontsize=17,fontweight="bold",color=TX)
    ax.axis("off"); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    years=[e[0] for e in events]
    ax.set_xlim(min(years)-15,max(years)+15); ax.set_ylim(-4,5)
    ax.axhline(0,color=TX2,lw=1.5,xmin=0.02,xmax=0.98)
    ax.arrow(max(years)+5,0,12,0,head_width=0.12,head_length=4,fc=TX2,ec=TX2)
    for year,label,color,up in events:
        yo=1.8 if up else -2.0
        ax.plot([year,year],[0.15 if up else -0.15,yo*0.85],color=color,lw=1.8)
        ax.scatter([year],[0],color=color,s=100,zorder=4,ec="white",lw=1.2)
        ax.text(year,yo,f"{year}\n{label}",ha="center",va="center",
                fontsize=8.5,color=color,fontweight="bold",multialignment="center",
                bbox=dict(fc=CARD,ec=color,alpha=0.5,boxstyle="round,pad=0.3"),zorder=5)
    save("09_ann_timeline.png")

def plot_summary_dashboard():
    print("[10] Summary Dashboard")
    z=np.linspace(-4,4,300)
    fig=plt.figure(figsize=(20,12))
    fig.suptitle("CH 10: ANNs with Keras — Complete Visual Summary",fontsize=19,fontweight="bold",color=TX,y=0.99)
    gs=GridSpec(3,4,figure=fig,hspace=0.5,wspace=0.4)
    ax_a=fig.add_subplot(gs[0,:2])
    for f,lbl,c in [(lambda z:1/(1+np.exp(-z)),"Sigmoid",B1),(np.tanh,"tanh",P1),(lambda z:np.maximum(0,z),"ReLU ⭐",G1)]:
        ax_a.plot(z,f(z),color=c,lw=2.5,label=lbl)
    ax_a.axhline(0,color=TX2,lw=0.8,ls="--"); ax_a.axvline(0,color=TX2,lw=0.8,ls="--")
    ax_a.set_title("Activation Functions",fontsize=12,fontweight="bold",color=TX)
    ax_a.legend(framealpha=0.3,facecolor=CARD,edgecolor=TX2); ax_a.grid(True); ax_a.set_xlim(-4,4)
    ax_b=fig.add_subplot(gs[0,2:])
    xp=np.linspace(-3,3,200); yp=xp**2+0.3*np.sin(5*xp)
    ax_b.plot(xp,yp,color=B1,lw=2); ax_b.fill_between(xp,yp,10,alpha=0.08,color=B1)
    path=[2.7,1.9,1.1,0.4,0.05,-0.01]; py=[xi**2+0.3*np.sin(5*xi) for xi in path]
    ax_b.plot(path,py,color=G1,lw=1.5,ls="--"); ax_b.scatter(path,py,color=G1,s=60,zorder=5)
    ax_b.set_title("Gradient Descent on Loss Surface",fontsize=12,fontweight="bold",color=TX)
    ax_b.set_xlabel("Weight θ"); ax_b.set_ylabel("Loss L"); ax_b.grid(True)
    ax_c=fig.add_subplot(gs[1,:2])
    np.random.seed(42); ep=np.arange(1,41)
    tr=1-0.5*np.exp(-0.1*ep)+np.random.normal(0,0.005,40)
    vl=1-0.55*np.exp(-0.08*ep)+np.random.normal(0,0.007,40)
    ax_c.plot(ep,tr,color=G1,lw=2,label="Train Acc"); ax_c.plot(ep,vl,color=B1,lw=2,ls="--",label="Val Acc")
    ax_c.axhline(0.98,color=TX2,lw=1,ls=":",alpha=0.6); ax_c.text(1,0.985,"98% target",fontsize=8,color=TX2)
    ax_c.set_title("Accuracy vs Epochs",fontsize=12,fontweight="bold",color=TX)
    ax_c.set_xlabel("Epoch"); ax_c.set_ylabel("Accuracy")
    ax_c.legend(framealpha=0.3,facecolor=CARD,edgecolor=TX2); ax_c.grid(True); ax_c.set_ylim(0.45,1.05)
    ax_d=fig.add_subplot(gs[1,2])
    logits=np.array([2.0,0.5,-0.3,1.0,0.1,-0.5,1.5,0.3,-1.0,0.2])
    sm=np.exp(logits)/np.exp(logits).sum()
    ax_d.bar([f"C{i}" for i in range(10)],sm,color=[G1 if i==0 else B1 for i in range(10)],alpha=0.8)
    ax_d.set_title("Softmax Output\n(10-class)",fontsize=11,fontweight="bold",color=TX)
    ax_d.set_ylabel("Probability"); ax_d.set_ylim(0,0.5); ax_d.grid(True,axis="y")
    ax_e=fig.add_subplot(gs[1,3])
    hps=["LR","# Layers","# Neurons","Batch","Optimizer","Activation"]
    imps=[9.5,8.0,7.0,6.5,6.0,5.0]
    bars=ax_e.barh(hps,imps,color=[R1,B1,P1,G1,O1,GOLD],alpha=0.8,height=0.6)
    ax_e.set_xlim(0,11); ax_e.set_title("HP Importance\nRanking",fontsize=11,fontweight="bold",color=TX)
    for b,v in zip(bars,imps):
        ax_e.text(v+0.1,b.get_y()+b.get_height()/2,f"{v}",va="center",fontsize=9,color=TX)
    ax_e.grid(True,axis="x")
    ax_f=fig.add_subplot(gs[2,:2])
    ep2=np.arange(1,51); np.random.seed(10)
    for c,lbl,alpha in [(B1,"SGD",0.12),(G1,"Adam ⭐",0.18),(P1,"RMSProp",0.12)]:
        loss=2.0*np.exp(-0.06*(1+(c==G1)*1.3)*ep2)+0.10+np.random.normal(0,0.015,50)
        ax_f.plot(ep2,loss,color=c,lw=2,label=lbl); ax_f.fill_between(ep2,loss,2.2,alpha=alpha,color=c)
    ax_f.set_title("Optimizer Convergence Speed",fontsize=12,fontweight="bold",color=TX)
    ax_f.set_xlabel("Epoch"); ax_f.set_ylabel("Loss")
    ax_f.legend(framealpha=0.3,facecolor=CARD,edgecolor=TX2); ax_f.grid(True)
    ax_g=fig.add_subplot(gs[2,2:])
    lay=np.arange(1,9); sg=0.5**lay*2; rl2=np.ones(8)*0.9+np.random.normal(0,0.03,8)
    x_pos=np.arange(8)
    ax_g.bar(x_pos-0.2,sg,0.38,label="Sigmoid (vanishing)",color=R1,alpha=0.8)
    ax_g.bar(x_pos+0.2,rl2,0.38,label="ReLU ⭐ (healthy)",color=G1,alpha=0.8)
    ax_g.set_xticks(x_pos); ax_g.set_xticklabels([f"L{i+1}" for i in range(8)])
    ax_g.set_title("Vanishing Gradient: Sigmoid vs ReLU",fontsize=12,fontweight="bold",color=TX)
    ax_g.set_xlabel("Layer"); ax_g.set_ylabel("Gradient Magnitude")
    ax_g.legend(framealpha=0.3,facecolor=CARD,edgecolor=TX2); ax_g.grid(True,axis="y")
    save("10_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# NEW GRAPHS (11-25)
# ══════════════════════════════════════════════════════════════════════════════

# ── 11. Biological vs Artificial Neuron (side-by-side) ───────────────────────
def plot_bio_vs_artificial():
    print("[11] Biological vs Artificial Neuron")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Biological Neuron  vs  Artificial Neuron",
                 fontsize=17, fontweight="bold", color=TX, y=1.02)

    # LEFT: Biological neuron
    ax = axes[0]; ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis("off")
    ax.set_facecolor(DARK); ax.set_title("🧠 Biological Neuron", fontsize=14, color=B1, fontweight="bold")

    # Dendrites
    for i,(dx,dy,a) in enumerate([(1.2,6.5,130),(0.8,5.0,160),(1.0,3.5,190)]):
        ax.annotate("", xy=(3.0,4.8), xytext=(dx,dy),
                    arrowprops=dict(arrowstyle="-|>",color=G1,lw=2))
        ax.text(dx-0.3,dy,f"Dendrite {i+1}\n(Input Signal)",ha="right",fontsize=8,color=G1)

    # Cell body
    cell = Circle((3.8,4.8),0.9,color=O1,alpha=0.8,zorder=3,ec="white",lw=2)
    ax.add_patch(cell)
    ax.text(3.8,4.8,"Cell\nBody\n(Σ sum)",ha="center",va="center",fontsize=8,color="white",fontweight="bold",zorder=4)

    # Axon
    ax.annotate("",xy=(7.0,4.8),xytext=(4.7,4.8),arrowprops=dict(arrowstyle="-|>",color=B1,lw=3))
    ax.text(5.85,5.2,"Axon\n(transmit)",ha="center",fontsize=8,color=B1)

    # Synapse
    syn = Circle((7.5,4.8),0.5,color=P1,alpha=0.8,zorder=3,ec="white",lw=1.5)
    ax.add_patch(syn)
    ax.text(7.5,4.8,"Synapse\n(⚡)",ha="center",va="center",fontsize=7,color="white",fontweight="bold",zorder=4)

    # Next neuron
    ax.annotate("",xy=(9.2,4.8),xytext=(8.0,4.8),arrowprops=dict(arrowstyle="-|>",color=TX2,lw=1.5))
    ax.text(9.5,4.8,"Next\nNeuron",ha="center",fontsize=8,color=TX2)

    # Threshold label
    ax.text(3.8,3.5,"Fires only if\ninput > threshold",ha="center",fontsize=8,color=GOLD,
            bbox=dict(fc=CARD,ec=GOLD,boxstyle="round",alpha=0.7))

    # RIGHT: Artificial neuron
    ax = axes[1]; ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis("off")
    ax.set_facecolor(DARK); ax.set_title("🤖 Artificial Neuron (Perceptron)", fontsize=14, color=R1, fontweight="bold")

    inputs = [(1.0,6.5,"x₁\n(0.5)"),(1.0,4.8,"x₂\n(0.3)"),(1.0,3.1,"x₃\n(0.8)")]
    weights = ["w₁=0.4","w₂=0.6","w₃=0.2"]
    for (ix,iy,ilbl),wlbl in zip(inputs,weights):
        node(ax,ix,iy,r=0.35,color=G1,label=ilbl,fontsize=8)
        ax.annotate("",xy=(3.5,4.8),xytext=(ix+0.35,iy),
                    arrowprops=dict(arrowstyle="-|>",color=TX2,lw=1.5,alpha=0.7))
        mx,my=(ix+3.5)/2,(iy+4.8)/2
        ax.text(mx,my+0.15,wlbl,ha="center",fontsize=8,color=GOLD,fontweight="bold",
                bbox=dict(fc=DARK,ec=GOLD,alpha=0.6,boxstyle="round"))

    # Bias
    node(ax,1.0,1.4,r=0.35,color=TX2,label="b=1\n(bias)",fontsize=8)
    ax.annotate("",xy=(3.5,4.5),xytext=(1.35,1.4),arrowprops=dict(arrowstyle="-|>",color=TX2,lw=1.2,alpha=0.5))

    # Summation node
    node(ax,4.2,4.8,r=0.55,color=O1,label="Σ\nz=Σwᵢxᵢ+b",fontsize=8)

    # Activation
    node(ax,6.5,4.8,r=0.55,color=R1,label="f(z)\nReLU/\nSigmoid",fontsize=7)
    ax.annotate("",xy=(5.9,4.8),xytext=(4.75,4.8),arrowprops=dict(arrowstyle="-|>",color=O1,lw=2))

    # Output
    node(ax,9.0,4.8,r=0.45,color=B1,label="ŷ\nOutput",fontsize=8)
    ax.annotate("",xy=(8.5,4.8),xytext=(7.05,4.8),arrowprops=dict(arrowstyle="-|>",color=B1,lw=2))

    ax.text(5,1.5,"Formula: z = w₁x₁ + w₂x₂ + w₃x₃ + b\nOutput: ŷ = f(z)",
            ha="center",fontsize=9,color=GOLD,
            bbox=dict(fc=CARD,ec=GOLD,alpha=0.8,boxstyle="round"))

    # Mapping arrows between panels
    for bio,art in [("Dendrites","Inputs (x)"),("Cell Body (Σ)","Summation node"),
                    ("Axon","Activation f(z)"),("Synapse","Weights (w)"),("Fires?","Output ŷ")]:
        pass  # concept map is shown via side-by-side layout

    plt.tight_layout(); save("11_bio_vs_artificial.png")

# ── 12. XOR Problem ───────────────────────────────────────────────────────────
def plot_xor_problem():
    print("[12] XOR Problem")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("The XOR Problem — Why Single Perceptron Fails",
                 fontsize=16, fontweight="bold", color=TX)

    # Data
    X = np.array([[0,0],[0,1],[1,0],[1,1]]); y_xor = np.array([0,1,1,0])
    y_or  = np.array([0,1,1,1]); y_and = np.array([0,0,0,1])

    for ax,(title,y,solvable) in zip(axes,[
        ("AND — ✅ Linearly Separable",y_and,True),
        ("OR — ✅ Linearly Separable",y_or,True),
        ("XOR — ❌ NOT Linearly Separable",y_xor,False)]):

        ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=G1 if solvable else R1)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.grid(True)
        ax.set_facecolor(DARK)

        for xi,yi,yval in zip(X[:,0],X[:,1],y):
            c=G1 if yval==1 else R1
            ax.scatter(xi,yi,color=c,s=250,zorder=5,ec="white",lw=1.5)
            ax.text(xi+0.06,yi+0.06,f"({xi},{yi})→{yval}",fontsize=9,color=c)

        if solvable:
            if title.startswith("AND"):
                lx=np.array([-0.3,1.3]); ax.plot(lx, 1.5-lx, color=B1,lw=2.5,ls="--",label="Decision boundary")
            else:
                lx=np.array([-0.3,1.3]); ax.plot(lx,-0.3+lx*0.0+0.5,color=B1,lw=2.5,ls="--")
                ax.plot([-0.3,1.3],[0.5,0.5],color=B1,lw=2.5,ls="--",label="Decision boundary")
            ax.text(0.5,-0.35,"One straight line works! ✅",ha="center",fontsize=9,color=G1)
        else:
            # Show failed attempts
            ax.text(0.5,-0.35,"No single straight line can separate! ❌",ha="center",fontsize=9,color=R1)
            ax.text(0.5,1.35,"→ Need MULTIPLE LAYERS (MLP)!",ha="center",fontsize=9,color=GOLD,fontweight="bold")
            for slope,intercept,alpha in [(1.5,-0.3,0.5),(-1.5,2.0,0.5),(0,0.5,0.5)]:
                xline=np.linspace(-0.3,1.3,50)
                ax.plot(xline,slope*xline+intercept,color=R1,lw=1.5,ls=":",alpha=alpha)

    plt.tight_layout(); save("12_xor_problem.png")

# ── 13. Loss Functions Comparison ────────────────────────────────────────────
def plot_loss_functions():
    print("[13] Loss Functions")
    y_true = 0.0
    err = np.linspace(-3, 3, 300)  # error = y_pred - y_true

    mse   = err**2
    mae   = np.abs(err)
    huber = np.where(np.abs(err)<=1, 0.5*err**2, np.abs(err)-0.5)

    # Cross-entropy: predicted prob vs true label=1
    p = np.linspace(0.01, 0.99, 300)
    ce_pos = -np.log(p)      # loss when true label=1
    ce_neg = -np.log(1-p)    # loss when true label=0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Loss Functions — Visual Comparison", fontsize=16, fontweight="bold", color=TX)

    # Regression losses
    ax = axes[0]
    ax.plot(err, mse,   color=B1,   lw=2.5, label="MSE = (y−ŷ)²")
    ax.plot(err, mae,   color=G1,   lw=2.5, label="MAE = |y−ŷ|")
    ax.plot(err, huber, color=O1,   lw=2.5, label="Huber (smooth MAE)", ls="--")
    ax.set_title("Regression Loss Functions", fontsize=13, fontweight="bold", color=TX)
    ax.set_xlabel("Prediction Error (y − ŷ)"); ax.set_ylabel("Loss Value")
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2, fontsize=11); ax.grid(True)
    ax.set_ylim(0, 5)
    ax.text(0,4.5,"MSE penalizes large errors MUCH more",ha="center",fontsize=9,color=B1,style="italic")
    ax.text(0,4.1,"MAE treats all errors equally",ha="center",fontsize=9,color=G1,style="italic")
    ax.annotate("MSE=4×MAE\nat error=2",xy=(2,4),xytext=(1.2,3.5),fontsize=8,color=B1,
                arrowprops=dict(arrowstyle="->",color=B1))

    # Classification losses (Cross-Entropy)
    ax = axes[1]
    ax.plot(p, ce_pos, color=G1, lw=2.5, label="Cross-Entropy (true=1): −log(p)")
    ax.plot(p, ce_neg, color=R1, lw=2.5, label="Cross-Entropy (true=0): −log(1−p)")
    ax.axvline(0.5, color=TX2, lw=1, ls=":")
    ax.set_title("Binary Cross-Entropy Loss", fontsize=13, fontweight="bold", color=TX)
    ax.set_xlabel("Predicted Probability (p̂)"); ax.set_ylabel("Loss Value")
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2, fontsize=11); ax.grid(True)
    ax.set_ylim(0, 5); ax.set_xlim(0,1)
    ax.fill_between(p[:150], ce_pos[:150], alpha=0.08, color=R1)
    ax.text(0.1,3.5,"99% confident\nbut WRONG?",ha="center",fontsize=9,color=R1,
            bbox=dict(fc=CARD,ec=R1,alpha=0.7,boxstyle="round"))
    ax.annotate("→ HUGE penalty!",xy=(0.01,4.5),xytext=(0.2,4.2),fontsize=9,color=R1,
                arrowprops=dict(arrowstyle="->",color=R1))
    plt.tight_layout(); save("13_loss_functions.png")

# ── 14. Cross-Entropy Intuition ───────────────────────────────────────────────
def plot_cross_entropy_intuition():
    print("[14] Cross-Entropy Intuition")
    p = np.linspace(0.01, 0.99, 200)
    ce = -np.log(p)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Cross-Entropy: Exponential Punishment for Wrong Confident Predictions",
                 fontsize=14, fontweight="bold", color=TX)

    ax = axes[0]
    ax.plot(p, ce, color=R1, lw=3)
    ax.fill_between(p, 0, ce, alpha=0.15, color=R1)
    ax.set_xlabel("Model's Confidence (Predicted Probability)", fontsize=11)
    ax.set_ylabel("Loss = −log(p)", fontsize=11)
    ax.set_title("Loss when True Label = 1", fontsize=12, fontweight="bold", color=TX)
    ax.grid(True)

    examples = [(0.9,"90% confident\nCorrect ✅",G1),(0.5,"50% confident\nUncertain ⚠️",GOLD),(0.1,"10% confident\nWrong ❌",R1)]
    for pr,lbl,c in examples:
        loss_val = -np.log(pr)
        ax.scatter([pr],[loss_val],color=c,s=180,zorder=5,ec="white",lw=1.5)
        ax.annotate(f"{lbl}\nloss={loss_val:.2f}",xy=(pr,loss_val),
                    xytext=(pr-0.15,loss_val+0.3+0.5*(pr<0.5)),fontsize=8,color=c,
                    arrowprops=dict(arrowstyle="->",color=c),
                    bbox=dict(fc=CARD,ec=c,alpha=0.8,boxstyle="round"))

    # Bar chart comparison
    ax = axes[1]
    scenarios = ["99% wrong\n(overconfident)", "90% wrong", "50% uncertain", "90% correct", "99% correct"]
    losses    = [-np.log(0.01), -np.log(0.10), -np.log(0.50), -np.log(0.90), -np.log(0.99)]
    colors    = [R1, R1, GOLD, G1, G1]
    bars = ax.bar(scenarios, losses, color=colors, alpha=0.8, width=0.6)
    ax.set_title("Loss by Confidence Level", fontsize=12, fontweight="bold", color=TX)
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, axis="y")
    for bar, loss_val in zip(bars, losses):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{loss_val:.2f}", ha="center", fontsize=9, color=TX, fontweight="bold")
    ax.text(0.5, 4.2, "Wrong with high confidence = MAXIMUM punishment!",
            transform=ax.transAxes, ha="center", fontsize=10, color=R1, fontweight="bold")

    plt.tight_layout(); save("14_cross_entropy_intuition.png")

# ── 15. Fashion MNIST Sample Grid ────────────────────────────────────────────
def plot_fashion_mnist_grid():
    print("[15] Fashion MNIST Sample Grid")
    class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
                   "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    class_colors = [B1,G1,P1,R1,O1,GOLD,B1,G1,P1,R1]

    # Simulate realistic Fashion MNIST-like pixel patterns for each class
    np.random.seed(123)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Fashion MNIST — 10 Classes (Simulated Sample Images)",
                 fontsize=16, fontweight="bold", color=TX, y=1.01)

    patterns = {
        0: lambda: np.clip(np.random.normal(0.6,0.2,(28,28)) * np.outer(np.hanning(28),np.ones(28)),0,1),
        1: lambda: np.clip(np.random.normal(0.7,0.15,(28,28)) * np.outer(np.ones(28),np.hanning(28)**0.3),0,1),
        2: lambda: np.clip(np.random.normal(0.65,0.18,(28,28)) * (np.outer(np.hanning(28)**0.5,np.hanning(28)**0.5)),0,1),
        3: lambda: np.clip(np.random.normal(0.6,0.2,(28,28)) * np.outer(np.ones(28)*0.9,np.hanning(28)),0,1),
        4: lambda: np.clip(np.random.normal(0.7,0.15,(28,28)) * np.outer(np.hanning(28)**0.3,np.hanning(28)**0.3),0,1),
        5: lambda: np.clip(np.random.normal(0.5,0.3,(28,28)) * np.outer(np.hanning(28),np.hanning(28)**2),0,1),
        6: lambda: np.clip(np.random.normal(0.65,0.18,(28,28)) * np.outer(np.hanning(28)**0.4,np.ones(28)),0,1),
        7: lambda: np.clip(np.random.normal(0.6,0.22,(28,28)) * np.outer(np.hanning(28)**2,np.hanning(28)**0.5),0,1),
        8: lambda: np.clip(np.random.normal(0.55,0.25,(28,28)) * np.outer(np.ones(28),np.ones(28))*np.eye(28)[::-1],0,1),
        9: lambda: np.clip(np.random.normal(0.7,0.15,(28,28)) * np.outer(np.hanning(28)**0.5,np.hanning(28)**2),0,1),
    }

    emojis = ["👕","👖","🧥","👗","🧥","👡","👔","👟","👜","👢"]

    for i, ax in enumerate(axes.flat):
        img = patterns[i]()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{emojis[i]} {i}: {class_names[i]}", fontsize=10,
                     color=class_colors[i], fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(class_colors[i]); spine.set_linewidth(2)

    fig.text(0.5, -0.02,
             "60,000 training + 10,000 test images | 28×28 grayscale pixels | 10 classes",
             ha="center", fontsize=11, color=TX2, style="italic")
    plt.tight_layout(); save("15_fashion_mnist_grid.png")

# ── 16. Confusion Matrix ──────────────────────────────────────────────────────
def plot_confusion_matrix():
    print("[16] Confusion Matrix")
    class_names = ["T-shirt","Trouser","Pullover","Dress","Coat",
                   "Sandal","Shirt","Sneaker","Bag","Boot"]
    np.random.seed(42)
    n = 10
    # Realistic confusion matrix: high diagonal, small off-diagonal confusions
    cm = np.random.randint(2, 12, (n, n))
    np.fill_diagonal(cm, np.random.randint(180, 250, n))
    # Add realistic confusions (Shirt↔Pullover, Coat↔Pullover)
    cm[2,6] = 45; cm[6,2] = 38; cm[2,4] = 30; cm[4,2] = 28
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("Confusion Matrix — Fashion MNIST Model Evaluation",
                 fontsize=16, fontweight="bold", color=TX)

    for ax, data, title, fmt in zip(axes,
                                    [cm, cm_norm],
                                    ["Raw Counts", "Normalized (per true class)"],
                                    [".0f", ".2f"]):
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", color=TX)

        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                txt = f"{val:{fmt}}"
                c = "white" if val > thresh else TX2
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color=c, fontweight="bold" if i==j else "normal")

    axes[1].text(0.5, -0.18,
                 "Diagonal = Correct predictions  |  Off-diagonal = Errors  |  Shirt↔Pullover most confused",
                 transform=axes[1].transAxes, ha="center", fontsize=9, color=TX2, style="italic")
    plt.tight_layout(); save("16_confusion_matrix.png")

# ── 17. Binary Decision Boundary ─────────────────────────────────────────────
def plot_decision_boundary():
    print("[17] Binary Decision Boundary")
    np.random.seed(42)
    n = 120
    X0 = np.random.randn(n, 2) * 0.8 + [-1.5, -1.0]
    X1 = np.random.randn(n, 2) * 0.8 + [1.5,  1.0]
    X  = np.vstack([X0, X1])
    y  = np.array([0]*n + [1]*n)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Binary Classification — Decision Boundary (Sigmoid Output)",
                 fontsize=15, fontweight="bold", color=TX)

    xx, yy = np.meshgrid(np.linspace(-4,4,200), np.linspace(-4,4,200))
    Xg = np.c_[xx.ravel(), yy.ravel()]

    # Simple logistic boundary: P(y=1) = sigmoid(w^T x + b)
    w = np.array([1.2, 0.9]); b = -0.3
    z_grid = Xg @ w + b
    p_grid = 1 / (1 + np.exp(-z_grid))
    P = p_grid.reshape(xx.shape)

    for ax, show_prob in zip(axes, [False, True]):
        if show_prob:
            cf = ax.contourf(xx, yy, P, levels=20, cmap="RdYlGn", alpha=0.55, vmin=0, vmax=1)
            plt.colorbar(cf, ax=ax, label="P(class=1)")
        else:
            ax.contourf(xx, yy, P, levels=[-0.1, 0.5, 1.1],
                        colors=[R1+"44", G1+"44"])

        ax.contour(xx, yy, P, levels=[0.5], colors="white", linewidths=2.5, linestyles="--")
        ax.scatter(X0[:,0], X0[:,1], color=R1, s=50, alpha=0.8, label="Class 0", ec="white", lw=0.5, zorder=4)
        ax.scatter(X1[:,0], X1[:,1], color=G1, s=50, alpha=0.8, label="Class 1", ec="white", lw=0.5, zorder=4)
        ax.set_xlabel("Feature x₁"); ax.set_ylabel("Feature x₂")
        ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2)
        ax.grid(True, alpha=0.3); ax.set_xlim(-4,4); ax.set_ylim(-4,4)

    axes[0].set_title("Hard Decision Boundary (threshold=0.5)", fontsize=12, fontweight="bold", color=TX)
    axes[0].text(0,-3.5,"White dashed = decision boundary (P=0.5)",ha="center",fontsize=9,color=TX2)

    axes[1].set_title("Probability Gradient (soft output)", fontsize=12, fontweight="bold", color=TX)
    axes[1].text(0,-3.5,"Green = high confidence class 1 | Red = high confidence class 0",
                 ha="center",fontsize=9,color=TX2)

    plt.tight_layout(); save("17_binary_decision_boundary.png")

# ── 18. Wide & Deep Architecture ─────────────────────────────────────────────
def plot_wide_deep():
    print("[18] Wide & Deep Architecture")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    fig.suptitle("Wide & Deep Network — Google's Architecture (2016)",
                 fontsize=16, fontweight="bold", color=TX)

    for ax_idx, (ax, title, has_aux) in enumerate(zip(axes,
        ["Wide & Deep (Basic)", "Wide & Deep (Multi-Input + Auxiliary Output)"],
        [False, True])):

        ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")
        ax.set_facecolor(DARK)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TX, pad=8)

        if not has_aux:
            # Input
            for i, y in enumerate([6.5, 5.5, 4.5, 3.5]):
                node(ax, 1.0, y, r=0.3, color=B1, label=f"x{i+1}", fontsize=9)

            # Wide path direct to concat
            ax.annotate("",xy=(6.0,7.0),xytext=(1.35,5.5),
                        arrowprops=dict(arrowstyle="-|>",color=GOLD,lw=2.2))
            ax.text(3.5,6.8,"Wide Path\n(direct)",ha="center",fontsize=9,color=GOLD,fontweight="bold")

            # Deep path: H1, H2
            for i,(hx,hy,lbl,c) in enumerate([(3.0,4.5,"H1\n30 neurons\nReLU",G1),(5.0,4.5,"H2\n30 neurons\nReLU",P1)]):
                node(ax,hx,hy,r=0.55,color=c,label=lbl,fontsize=8)
            ax.annotate("",xy=(2.4,4.5),xytext=(1.35,4.9),arrowprops=dict(arrowstyle="-|>",color=TX2,lw=1.5))
            ax.annotate("",xy=(4.4,4.5),xytext=(3.6,4.5),arrowprops=dict(arrowstyle="-|>",color=G1,lw=2))
            ax.text(4.0,3.5,"Deep Path",ha="center",fontsize=9,color=G1)

            # Concat
            box(ax,6.5,5.5,1.4,1.0,color=O1,label="Concat\n(merge)",fontsize=9)
            ax.annotate("",xy=(5.8,5.5),xytext=(5.55,4.5),arrowprops=dict(arrowstyle="-|>",color=P1,lw=2))

            # Output
            node(ax,8.5,5.5,r=0.5,color=R1,label="Output\nŷ",fontsize=9)
            ax.annotate("",xy=(8.0,5.5),xytext=(7.2,5.5),arrowprops=dict(arrowstyle="-|>",color=O1,lw=2))

            ax.text(5,1.5,
                    "Wide path = Memorization (remembers exact patterns)\nDeep path = Generalization (learns abstract features)",
                    ha="center",fontsize=9,color=TX2,
                    bbox=dict(fc=CARD,ec=TX2,alpha=0.5,boxstyle="round"))

        else:
            # Two inputs
            box(ax,1.0,7.0,1.4,0.8,color=B1,label="Input A\n(5 features)",fontsize=8)
            box(ax,1.0,5.0,1.4,0.8,color=G1,label="Input B\n(6 features)",fontsize=8)

            # Deep path from B
            for hx,hy,lbl,c in [(3.5,5.0,"H1\n30",G1),(5.0,5.0,"H2\n30",P1)]:
                node(ax,hx,hy,r=0.4,color=c,label=lbl,fontsize=9)
            ax.annotate("",xy=(3.1,5.0),xytext=(1.7,5.0),arrowprops=dict(arrowstyle="-|>",color=G1,lw=2))
            ax.annotate("",xy=(4.6,5.0),xytext=(3.9,5.0),arrowprops=dict(arrowstyle="-|>",color=G1,lw=2))

            # Wide path from A
            ax.annotate("",xy=(6.5,6.5),xytext=(1.7,7.0),arrowprops=dict(arrowstyle="-|>",color=GOLD,lw=2.2))
            ax.text(4.0,7.0,"Wide path",ha="center",fontsize=9,color=GOLD,fontweight="bold")

            # Concat
            box(ax,7.0,5.8,1.5,1.2,color=O1,label="Concat",fontsize=9)
            ax.annotate("",xy=(6.25,5.8),xytext=(5.4,5.0),arrowprops=dict(arrowstyle="-|>",color=P1,lw=2))

            # Main output
            node(ax,9.0,6.2,r=0.5,color=R1,label="Main\nOutput",fontsize=8)
            ax.annotate("",xy=(8.5,6.2),xytext=(7.75,5.8),arrowprops=dict(arrowstyle="-|>",color=O1,lw=2))

            # Auxiliary output
            node(ax,7.5,3.2,r=0.5,color=TX2,label="Aux\nOutput",fontsize=8)
            ax.annotate("",xy=(7.1,3.5),xytext=(5.4,4.7),arrowprops=dict(arrowstyle="-|>",color=TX2,lw=1.5,alpha=0.7),)
            ax.text(6.0,2.5,"Auxiliary Output\n(for regularization)\nloss_weight=0.1",ha="center",fontsize=8,color=TX2,
                    bbox=dict(fc=CARD,ec=TX2,alpha=0.5,boxstyle="round"))
            ax.text(9.0,5.0,"Main Output\nloss_weight=0.9",ha="center",fontsize=8,color=R1,
                    bbox=dict(fc=CARD,ec=R1,alpha=0.5,boxstyle="round"))

    plt.tight_layout(); save("18_wide_deep_architecture.png")

# ── 19. Parameter Count Breakdown ────────────────────────────────────────────
def plot_param_count():
    print("[19] Parameter Count Breakdown")
    layers = ["Input\n(Flatten)", "Hidden 1\nDense(300)", "Hidden 2\nDense(100)", "Output\nDense(10)"]
    units  = [784, 300, 100, 10]
    prev   = [0, 784, 300, 100]
    params = [0, 784*300+300, 300*100+100, 100*10+10]
    param_labels = ["0\n(no params)", "784×300+300\n=235,500", "300×100+100\n=30,100", "100×10+10\n=1,010"]
    colors = [TX2, B1, G1, O1]
    formulas = ["—", "W:(784×300)\n+b:(300)", "W:(300×100)\n+b:(100)", "W:(100×10)\n+b:(10)"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Model Parameter Count — Fashion MNIST MLP (266,610 total params)",
                 fontsize=15, fontweight="bold", color=TX)

    # Bar chart of params
    ax = axes[0]
    bars = ax.barh(layers, params, color=colors, alpha=0.85, height=0.55, edgecolor="white", lw=1)
    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_title("Parameters per Layer", fontsize=13, fontweight="bold", color=TX)
    ax.grid(True, axis="x", alpha=0.5)
    for bar, p, lbl in zip(bars, params, param_labels):
        ax.text(p+500, bar.get_y()+bar.get_height()/2, f" {p:,}\n({lbl.split(chr(10))[0]})",
                va="center", fontsize=9, color=TX)
    ax.axvline(sum(params), color=GOLD, lw=2, ls="--")
    ax.text(sum(params)-2000, -0.7, f"Total: {sum(params):,}", fontsize=11, color=GOLD, fontweight="bold")

    # Neural network param flow diagram
    ax = axes[1]; ax.set_xlim(0,10); ax.set_ylim(0,9); ax.axis("off")
    ax.set_title("How Parameters are Computed", fontsize=13, fontweight="bold", color=TX)

    layer_x = [1.5, 4.0, 6.5, 9.0]
    layer_sizes = [4, 5, 3, 2]  # visual neurons
    layer_colors = colors

    positions = []
    for lx, ln, lc in zip(layer_x, layer_sizes, layer_colors):
        ys = np.linspace(2.0, 7.5, ln)
        positions.append([(lx, y) for y in ys])
        for y in ys:
            node(ax, lx, y, r=0.28, color=lc)

    for li in range(len(positions)-1):
        for (x1,y1) in positions[li]:
            for (x2,y2) in positions[li+1]:
                ax.plot([x1,x2],[y1,y2],color=TX2,lw=0.3,alpha=0.3)

    for lx,lbl,param,c in zip(layer_x, layers, params, colors):
        ax.text(lx, 1.2, lbl, ha="center", fontsize=8, color=c, fontweight="bold")
        if param > 0:
            ax.text(lx, 0.5, f"{param:,} params", ha="center", fontsize=8, color=c,
                    bbox=dict(fc=CARD, ec=c, alpha=0.7, boxstyle="round"))

    for i,(p1,p2,fp) in enumerate(zip(layer_x,layer_x[1:],formulas[1:])):
        ax.text((p1+p2)/2, 8.3, fp, ha="center", fontsize=7.5, color=TX2,
                bbox=dict(fc=DARK, ec=TX2, alpha=0.6, boxstyle="round"))

    plt.tight_layout(); save("19_param_count.png")

# ── 20. Three Keras APIs Comparison ──────────────────────────────────────────
def plot_three_apis():
    print("[20] Three Keras APIs")
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.suptitle("Three Keras APIs — When to Use Which?",
                 fontsize=17, fontweight="bold", color=TX, y=1.01)

    configs = [
        ("Sequential API", "⭐ Beginner", "Linear stacks only\nOne input → one output", B1, [
            "model = Sequential([",
            "  Flatten(28,28),",
            "  Dense(300,'relu'),",
            "  Dense(100,'relu'),",
            "  Dense(10,'softmax')",
            "])",
        ], ["✅ Easiest to use","✅ Great for learning","❌ No branching","❌ Single I/O only"]),
        ("Functional API", "⭐⭐ Intermediate", "Any graph topology\nMultiple I/O, skip connections", G1, [
            "inp = Input([8])",
            "h1 = Dense(30,'relu')(inp)",
            "h2 = Dense(30,'relu')(h1)",
            "merged = Concat()([inp,h2])",
            "out = Dense(1)(merged)",
            "model = Model(inp,out)",
        ], ["✅ Any architecture","✅ Multiple I/O","✅ Skip connections","✅ Production standard"]),
        ("Subclassing API", "⭐⭐⭐ Advanced", "Full Python control\nDynamic architectures", P1, [
            "class MyModel(Model):",
            "  def __init__(self):",
            "    self.h1=Dense(30,'relu')",
            "    self.h2=Dense(30,'relu')",
            "  def call(self,x):",
            "    return self.h2(self.h1(x))",
        ], ["✅ Dynamic forward pass","✅ Research use","⚠️ Harder to save","⚠️ Less inspectable"]),
    ]

    for ax, (title, level, desc, color, code, pros) in zip(axes, configs):
        ax.set_xlim(0,10); ax.set_ylim(0,12); ax.axis("off"); ax.set_facecolor(DARK)

        # Header
        box(ax, 5, 11.2, 9.5, 1.0, color=color, label=f"{title}\n{level}", fontsize=11)

        # Description
        ax.text(5, 10.0, desc, ha="center", va="center", fontsize=9, color=TX2,
                multialignment="center")

        # Code block
        code_bg = FancyBboxPatch((0.3, 5.8), 9.4, 3.8, boxstyle="round,pad=0.1",
                                  fc="#1a1a2e", ec=color, lw=1.5, alpha=0.9)
        ax.add_patch(code_bg)
        ax.text(0.6, 9.4, "# Code:", fontsize=8, color=TX2)
        for i, line in enumerate(code):
            ax.text(0.6, 9.0 - i*0.53, line, fontsize=8.2, color=TX,
                    fontfamily="monospace")

        # Pros/cons
        for i, item in enumerate(pros):
            c2 = G1 if item.startswith("✅") else (GOLD if item.startswith("⚠️") else R1)
            ax.text(0.5, 5.3 - i*0.75, item, fontsize=9, color=c2)

        # Use case label
        use_cases = ["Learning\nSimple models","Production\nComplex models","Research\nCustom architectures"]
        idx = ["Sequential","Functional","Subclassing"].index(title.split()[0])
        box(ax, 5, 1.0, 8.5, 1.4, color=color, label=f"Best For:\n{use_cases[idx]}", fontsize=10)

    plt.tight_layout(); save("20_three_apis_comparison.png")

# ── 21. Callback Timeline ─────────────────────────────────────────────────────
def plot_callback_timeline():
    print("[21] Callback Timeline")
    fig, ax = plt.subplots(figsize=(17, 8))
    fig.suptitle("Keras Callbacks — When Each Hook Fires During Training",
                 fontsize=16, fontweight="bold", color=TX)
    ax.set_xlim(0, 18); ax.set_ylim(0, 8); ax.axis("off"); ax.set_facecolor(DARK)

    # Training timeline bar
    ax.barh(4, 16, left=1, height=0.4, color=CARD, edgecolor=TX2, lw=1.5)
    ax.text(9, 3.3, "Training Timeline (Epochs)", ha="center", fontsize=12, color=TX2)

    # Events with positions
    events = [
        (1.0, 6.5, "on_train_begin", B1, "Called once\nbefore training starts"),
        (3.0, 6.5, "on_epoch_begin", G1, "Called at start\nof every epoch"),
        (4.5, 5.5, "on_batch_begin", P1, "Every\nbatch start"),
        (6.0, 5.5, "on_batch_end",   O1, "Every\nbatch end"),
        (7.5, 6.5, "on_epoch_end",   GOLD, "Called at end\nof every epoch"),
        (9.5, 5.0, "ModelCheckpoint", R1,  "→ save if val_loss improved"),
        (9.5, 4.0, "EarlyStopping",  G1,  "→ stop if no improvement for N epochs"),
        (9.5, 3.0, "TensorBoard",    B1,  "→ log metrics to file"),
        (14.0,6.5, "on_train_end",   R1,  "Called once\nafter training ends"),
    ]

    for ex, ey, label, c, desc in events[:5] + [events[-1]]:
        ax.scatter([ex], [4.2], color=c, s=120, zorder=5, ec="white", lw=1.5)
        ax.plot([ex, ex], [4.2, ey-0.5], color=c, lw=1.5, ls="--", alpha=0.7)
        ax.text(ex, ey, label, ha="center", fontsize=8.5, color=c, fontweight="bold",
                bbox=dict(fc=CARD, ec=c, alpha=0.8, boxstyle="round"))
        ax.text(ex, ey-0.7, desc, ha="center", fontsize=7.5, color=TX2)

    # Callbacks box
    box(ax, 9.5, 4.0, 6.0, 2.8, color=TX2, label="", fontsize=9, alpha=0.1)
    ax.text(9.5, 5.6, "Called at on_epoch_end:", ha="center", fontsize=9, color=TX2, fontweight="bold")
    for (ex,ey,label,c,desc) in events[5:8]:
        ax.text(8.0, ey, f"  {label}  → {desc}", fontsize=8.5, color=c)

    # Epoch markers
    for ex in [3.0, 7.5, 12.0]:
        ax.axvline(ex, color=TX2, lw=0.8, ls=":", alpha=0.4, ymin=0.44, ymax=0.56)

    ax.text(9, 1.5,
            "model.fit(X, y, callbacks=[ModelCheckpoint(...), EarlyStopping(...), TensorBoard(...)])",
            ha="center", fontsize=9, color=GOLD, fontfamily="monospace",
            bbox=dict(fc=CARD, ec=GOLD, alpha=0.8, boxstyle="round"))

    save("21_callback_timeline.png")

# ── 22. Early Stopping Annotated ──────────────────────────────────────────────
def plot_early_stopping():
    print("[22] Early Stopping Annotated")
    np.random.seed(42)
    epochs = np.arange(1, 81)

    train_loss = 2.0*np.exp(-0.08*epochs) + 0.05 + np.random.normal(0, 0.015, 80)
    val_loss   = 2.0*np.exp(-0.06*epochs) + 0.12 + np.random.normal(0, 0.025, 80)
    # Add overfitting after epoch 30
    val_loss[30:] += np.linspace(0, 0.35, 50) + np.random.normal(0, 0.02, 50)

    best_epoch = np.argmin(val_loss[:45]) + 1  # best at ~epoch 30
    stop_epoch = best_epoch + 10               # patience=10

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Early Stopping — Preventing Overfitting",
                 fontsize=16, fontweight="bold", color=TX)

    ax.plot(epochs, train_loss, color=G1, lw=2.5, label="Training Loss")
    ax.plot(epochs, val_loss,   color=B1, lw=2.5, label="Validation Loss", ls="--")
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.08, color=R1)

    # Best epoch marker
    best_val = val_loss[best_epoch-1]
    ax.axvline(best_epoch, color=GOLD, lw=2, ls="--")
    ax.scatter([best_epoch], [best_val], color=GOLD, s=200, zorder=6, ec="white", lw=2)
    ax.text(best_epoch+0.5, best_val+0.02,
            f"Best model\n(Epoch {best_epoch})\nval_loss={best_val:.3f}",
            fontsize=9, color=GOLD, fontweight="bold",
            bbox=dict(fc=CARD, ec=GOLD, alpha=0.85, boxstyle="round"))

    # Patience window
    ax.axvspan(best_epoch, stop_epoch, alpha=0.12, color=O1)
    ax.text((best_epoch+stop_epoch)/2, 0.85, f"Patience window\n(patience=10 epochs)",
            ha="center", fontsize=9, color=O1, fontweight="bold")

    # Stop marker
    ax.axvline(stop_epoch, color=R1, lw=2.5)
    ax.text(stop_epoch+0.5, 0.9, f"⛔ Training\nStopped\n(Epoch {stop_epoch})",
            fontsize=9, color=R1, fontweight="bold",
            bbox=dict(fc=CARD, ec=R1, alpha=0.85, boxstyle="round"))

    # Overfitting zone
    ax.axvspan(stop_epoch, 80, alpha=0.06, color=R1)
    ax.text(65, 1.0, "Would have\noverfitted here\n(saved time + quality!)",
            ha="center", fontsize=9, color=R1,
            bbox=dict(fc=CARD, ec=R1, alpha=0.5, boxstyle="round"))

    # Annotations
    ax.annotate("",xy=(stop_epoch-0.5, 0.6),xytext=(stop_epoch+8, 0.6),
                arrowprops=dict(arrowstyle="-|>",color=O1,lw=1.5))
    ax.annotate("",xy=(stop_epoch+8, 0.6),xytext=(stop_epoch-0.5, 0.6),
                arrowprops=dict(arrowstyle="-|>",color=O1,lw=1.5))
    ax.text(stop_epoch+9, 0.59, f"Loads weights\nfrom Epoch {best_epoch}",
            fontsize=8, color=O1)

    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Loss", fontsize=12)
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2, fontsize=12)
    ax.grid(True); ax.set_xlim(1,80); ax.set_ylim(0.0, 1.2)
    ax.text(40, 1.12,
            "EarlyStopping(patience=10, restore_best_weights=True)  →  Automatically stopped at Epoch 40!",
            ha="center", fontsize=9.5, color=GOLD, fontfamily="monospace",
            bbox=dict(fc=CARD, ec=GOLD, alpha=0.8, boxstyle="round"))
    plt.tight_layout(); save("22_early_stopping_annotated.png")

# ── 23. Grid vs Random Search ─────────────────────────────────────────────────
def plot_grid_vs_random():
    print("[23] Grid vs Random Search")
    np.random.seed(7)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Hyperparameter Search Strategies — Grid vs Random vs Bayesian",
                 fontsize=15, fontweight="bold", color=TX)

    # True "performance landscape"
    def perf(lr, neurons):
        return np.exp(-((np.log10(lr)+3.5)**2/0.5 + (neurons-60)**2/800))

    lr_range = np.logspace(-5, -1, 200)
    n_range  = np.linspace(10, 200, 200)
    LR, NN   = np.meshgrid(lr_range, n_range)
    Z        = perf(LR, NN)

    titles = ["Grid Search\n(9 combinations)", "Random Search\n(9 random trials)", "Bayesian Opt\n(9 smart trials)"]
    configs = []

    # Grid points
    g_lr = np.logspace(-5,-1,3); g_n = np.array([30,100,180])
    GLR,GN = np.meshgrid(g_lr,g_n)
    configs.append((GLR.ravel(), GN.ravel()))

    # Random points
    configs.append((np.random.uniform(-5,-1,9), np.random.uniform(10,200,9)))

    # Bayesian (focus around optimal)
    opt_lr, opt_n = 3e-4, 60
    configs.append((np.random.normal(np.log10(opt_lr)*0.85, 0.5, 9).clip(-5,-1),
                    np.random.normal(opt_n, 20, 9).clip(10,200)))

    for ax,(title,pts) in zip(axes, zip(titles, configs)):
        ax.contourf(np.log10(lr_range), n_range, Z, levels=20, cmap="viridis", alpha=0.7)
        ax.contour(np.log10(lr_range),  n_range, Z, levels=5,  colors="white", alpha=0.3, linewidths=0.8)

        if isinstance(pts[0], np.ndarray) and pts[0][0] > -10:  # random/bayes
            ax.scatter(pts[0], pts[1], color=R1, s=120, zorder=5, ec="white", lw=1.5, label="Trials")
        else:
            ax.scatter(np.log10(pts[0]), pts[1], color=R1, s=120, zorder=5, ec="white", lw=1.5)

        # Mark best region
        ax.axvline(np.log10(opt_lr), color=GOLD, lw=1.5, ls="--", alpha=0.6)
        ax.axhline(opt_n,            color=GOLD, lw=1.5, ls="--", alpha=0.6)
        ax.scatter([np.log10(opt_lr)],[opt_n],color=GOLD,s=300,marker="*",zorder=6,label="Optimal")

        ax.set_xlabel("log₁₀(Learning Rate)"); ax.set_ylabel("Neurons per Layer")
        ax.set_title(title, fontsize=12, fontweight="bold", color=TX)
        ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2)

    axes[0].text(-3, 190, "Misses optimal if\nnot in grid!", ha="center", fontsize=8, color=R1,
                 bbox=dict(fc=DARK, ec=R1, alpha=0.7, boxstyle="round"))
    axes[1].text(-3, 190, "Better coverage!", ha="center", fontsize=8, color=G1,
                 bbox=dict(fc=DARK, ec=G1, alpha=0.7, boxstyle="round"))
    axes[2].text(-3, 190, "Smart: focuses on\npromising regions!", ha="center", fontsize=8, color=B1,
                 bbox=dict(fc=DARK, ec=B1, alpha=0.7, boxstyle="round"))

    plt.tight_layout(); save("23_grid_vs_random_search.png")

# ── 24. Transfer Learning Diagram ─────────────────────────────────────────────
def plot_transfer_learning():
    print("[24] Transfer Learning")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    fig.suptitle("Transfer Learning — Reusing Pretrained Networks",
                 fontsize=16, fontweight="bold", color=TX)

    for ax_idx, (ax, title) in enumerate(zip(axes,
        ["Step 1: Pretrained Model (ImageNet, 1000 classes)",
         "Step 2: Fine-tuned for Your Task"])):
        ax.set_xlim(0,12); ax.set_ylim(0,10); ax.axis("off")
        ax.set_facecolor(DARK)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TX, pad=8)

        # Layer blocks
        layer_defs = [
            (2, [2.0,3.5,5.0,6.5,8.0], "Low-level\nFeatures\n(edges)", G1, "FREEZE ❄️" if ax_idx==1 else "Trained"),
            (5, [2.5,4.0,5.5,7.0],     "Mid-level\nFeatures\n(shapes)", B1, "FREEZE ❄️" if ax_idx==1 else "Trained"),
            (8, [3.5,5.0,6.5],         "High-level\nFeatures\n(objects)", P1, "FREEZE ❄️" if ax_idx==1 else "Trained"),
            (11,[4.5,5.5],             "Output\n(1000 cls)" if ax_idx==0 else "New Head\n(10 cls)",
             TX2 if ax_idx==0 else R1, "Replace 🔄" if ax_idx==1 else "Original"),
        ]

        for lx, lys, lbl, c, status in layer_defs:
            frozen = status.startswith("FREEZE") and ax_idx==1
            alpha  = 0.35 if frozen else 0.85
            for ly in lys:
                n_color = c if not frozen else TX2
                node(ax, lx, ly, r=0.32, color=n_color, alpha=alpha)

            if frozen:
                ax.text(lx, 1.1, "❄️ Frozen", ha="center", fontsize=8, color=TX2)
            else:
                ax.text(lx, 1.1, "🔥 Training" if ax_idx==1 and lx==11 else status,
                        ha="center", fontsize=8, color=c)

            ax.text(lx, 0.4, lbl.split("\n")[0], ha="center", fontsize=8, color=c, fontweight="bold")

        # Connections
        all_pos = []
        for lx, lys, *_ in layer_defs:
            all_pos.append([(lx,y) for y in lys])
        for li in range(len(all_pos)-1):
            for (x1,y1) in all_pos[li]:
                for (x2,y2) in all_pos[li+1]:
                    col = TX2 if (ax_idx==1 and li<3) else B1
                    ax.plot([x1,x2],[y1,y2],color=col,lw=0.3,alpha=0.25)

        # Labels
        ax.text(2, 9.3, "Layer 1-5\n(General patterns)", ha="center", fontsize=8, color=G1)
        ax.text(5, 9.3, "Layer 6-10\n(Complex patterns)", ha="center", fontsize=8, color=B1)
        ax.text(8, 9.3, "Layer 11-15\n(Task-specific)", ha="center", fontsize=8, color=P1)

        if ax_idx == 1:
            ax.text(6, 9.0,
                    "❄️ Freeze pretrained layers → ⚡ Train only the new head\n"
                    "Optionally unfreeze top layers for fine-tuning",
                    ha="center", fontsize=9, color=GOLD,
                    bbox=dict(fc=CARD, ec=GOLD, alpha=0.7, boxstyle="round"))

    plt.tight_layout(); save("24_transfer_learning.png")

# ── 25. Neurons vs Layers (Same Budget) ───────────────────────────────────────
def plot_neurons_vs_layers():
    print("[25] Neurons vs Layers Comparison")
    np.random.seed(42)
    epochs = np.arange(1, 51)

    # Same ~5000 params: 1 wide layer vs 3 deep layers
    wide_train  = 1.0-0.7*np.exp(-0.12*epochs)+np.random.normal(0,0.008,50)
    wide_val    = 1.0-0.65*np.exp(-0.09*epochs)+np.random.normal(0,0.012,50)
    deep_train  = 1.0-0.7*np.exp(-0.16*epochs)+np.random.normal(0,0.008,50)
    deep_val    = 1.0-0.68*np.exp(-0.14*epochs)+np.random.normal(0,0.012,50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Wide vs Deep: Same Parameter Budget (~5,000 params)",
                 fontsize=15, fontweight="bold", color=TX)

    # Accuracy curves
    ax = axes[0]
    ax.plot(epochs, wide_val,  color=O1, lw=2.5, label="Wide: 1 layer × 70 neurons (val)")
    ax.plot(epochs, deep_val,  color=G1, lw=2.5, label="Deep: 3 layers × 15 neurons (val)")
    ax.plot(epochs, wide_train,color=O1, lw=1.5, ls="--", alpha=0.6, label="Wide (train)")
    ax.plot(epochs, deep_train,color=G1, lw=1.5, ls="--", alpha=0.6, label="Deep (train)")
    ax.set_title("Validation Accuracy: Wide vs Deep", fontsize=13, fontweight="bold", color=TX)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(framealpha=0.4, facecolor=CARD, edgecolor=TX2, fontsize=9); ax.grid(True)
    ax.set_ylim(0.2, 1.05)
    ax.text(26, 0.56, "Deep converges\nfaster & higher!", ha="center", fontsize=9,
            color=G1, bbox=dict(fc=CARD, ec=G1, alpha=0.7, boxstyle="round"))
    ax.annotate("",xy=(40,deep_val[39]),xytext=(36,0.6),arrowprops=dict(arrowstyle="->",color=G1))

    # Architecture visual
    ax = axes[1]; ax.set_xlim(0,10); ax.set_ylim(0,8); ax.axis("off")
    ax.set_title("Architecture Comparison (Same ~5000 Params)", fontsize=13, fontweight="bold", color=TX)

    # Wide network
    ax.text(2, 7.6, "Wide: 1×70", ha="center", fontsize=11, color=O1, fontweight="bold")
    for i,y in enumerate(np.linspace(1.0,7.0,7)):  # Show 7 of 70
        node(ax,1.0,y,r=0.25,color=O1,alpha=0.6)
        node(ax,3.0,y,r=0.25,color=TX2,alpha=0.6)
    ax.text(2.0,0.3,"Input → [70 neurons] → Output",ha="center",fontsize=8,color=O1)
    ax.text(2.0,-0.1,"~5,040 params",ha="center",fontsize=8,color=TX2)

    # Deep network
    ax.text(7.5, 7.6, "Deep: 3×15", ha="center", fontsize=11, color=G1, fontweight="bold")
    for lx, ln, c in [(5.5,4,B1),(7.0,3,G1),(8.5,3,P1)]:
        for y in np.linspace(2.5,6.0,ln):
            node(ax,lx,y,r=0.28,color=c)
    node(ax,10.0,4.25,r=0.28,color=TX2,label="out",fontsize=7)
    ax.text(7.5,0.3,"Input → [15] → [15] → [15] → Output",ha="center",fontsize=8,color=G1)
    ax.text(7.5,-0.1,"~4,950 params",ha="center",fontsize=8,color=TX2)

    ax.text(5, 8.2,
            "💡 Key insight: 'More layers > more neurons per layer' for complex tasks",
            ha="center", fontsize=10, color=GOLD,
            bbox=dict(fc=CARD, ec=GOLD, alpha=0.7, boxstyle="round"))

    plt.tight_layout(); save("25_neurons_vs_layers.png")


def plot_26_backprop_node_circuit():
    print("[26] Backprop Node Circuit")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_facecolor(DARK)
    
    fig.suptitle("Computational Gate Backpropagation Mechanics (CS231n Style)", fontsize=13, fontweight="bold", color=TX)
    
    # Gate Circle
    node(ax, 5.0, 3.0, r=0.8, color=GOLD, label="*", fontsize=20)
    ax.text(5.0, 4.0, "Multiplication Gate\nf(x, y) = x · y", ha="center", fontsize=9.5, color=GOLD, fontweight="bold")
    
    # Input lines
    ax.plot([1.5, 4.2], [4.5, 3.5], color=TX2, lw=2.0) # upper input
    ax.plot([1.5, 4.2], [1.5, 2.5], color=TX2, lw=2.0) # lower input
    # Output line
    ax.plot([5.8, 8.5], [3.0, 3.0], color=TX2, lw=2.0)
    
    # Values labels
    # Forward pass (Green)
    ax.text(1.3, 4.7, "Input x = -2.0", color=G1, fontsize=9.5, fontweight="bold", ha="right")
    ax.text(1.3, 1.3, "Input y = 3.0", color=G1, fontsize=9.5, fontweight="bold", ha="right")
    ax.text(8.7, 3.2, "Output z = -6.0", color=G1, fontsize=9.5, fontweight="bold", ha="left")
    
    # Upstream gradient (Blue)
    ax.text(8.7, 2.6, "Upstream Gradient\n∂L/∂z = 1.0", color=B1, fontsize=9, fontweight="bold", ha="left")
    
    # Downstream gradients calculated (Red)
    # local gradient for x is y = 3.0. Downstream: (∂L/∂z) * (∂z/∂x) = 1.0 * 3.0 = 3.0
    # local gradient for y is x = -2.0. Downstream: (∂L/∂z) * (∂z/∂y) = 1.0 * -2.0 = -2.0
    ax.text(2.8, 4.6, "Local grad: ∂z/∂x = y = 3.0\nDownstream: ∂L/∂x = 3.0", color=R1, fontsize=8.5, fontweight="bold", ha="center")
    ax.text(2.8, 1.4, "Local grad: ∂z/∂y = x = -2.0\nDownstream: ∂L/∂y = -2.0", color=R1, fontsize=8.5, fontweight="bold", ha="center")
    
    # Legend panel
    box(ax, 5.0, 0.6, 8.0, 0.8, color=TX2, label="", alpha=0.1)
    ax.text(5.0, 0.7, "Forward values (green) flow left-to-right. Gradients (red) propagate right-to-left.\nChain Rule: Downstream Gradient = Upstream Gradient × Local Gradient", 
            ha="center", va="center", fontsize=8.5, color=GOLD, style="italic")
    
    save("26_backprop_node_circuit.png")


def plot_27_activation_saturation_regions():
    print("[27] Activation Saturation")
    z = np.linspace(-4, 4, 300)
    
    # Sigmoid and its derivative
    sig = 1.0 / (1.0 + np.exp(-z))
    d_sig = sig * (1.0 - sig)
    
    # Tanh and its derivative
    tanh_val = np.tanh(z)
    d_tanh = 1.0 - tanh_val**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Activation Functions & Gradient Saturation Regions", fontsize=14, fontweight="bold", color=TX)
    
    # Sigmoid Plot
    ax = axes[0]
    ax.plot(z, sig, color=B1, lw=2.5, label="Sigmoid σ(z)")
    ax.plot(z, d_sig, color=R1, lw=2.0, ls="--", label="Sigmoid derivative σ'(z)")
    ax.axvspan(-4, -2, color=R1, alpha=0.08)
    ax.axvspan(2, 4, color=R1, alpha=0.08)
    ax.set_title("Sigmoid Saturation Zones", fontsize=11, fontweight="bold", color=TX)
    ax.set_xlabel("z"); ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    ax.text(-3, 0.4, "Saturation Zone\nDerivative ≈ 0\n(Gradient vanishes)", color=R1, fontsize=8, ha="center", fontweight="bold")
    ax.text(3, 0.4, "Saturation Zone\nDerivative ≈ 0\n(Gradient vanishes)", color=R1, fontsize=8, ha="center", fontweight="bold")
    
    # Tanh Plot
    ax = axes[1]
    ax.plot(z, tanh_val, color=G1, lw=2.5, label="Tanh(z)")
    ax.plot(z, d_tanh, color=R1, lw=2.0, ls="--", label="Tanh derivative Tanh'(z)")
    ax.axvspan(-4, -2, color=R1, alpha=0.08)
    ax.axvspan(2, 4, color=R1, alpha=0.08)
    ax.set_title("Tanh Saturation Zones", fontsize=11, fontweight="bold", color=TX)
    ax.set_xlabel("z")
    ax.grid(True)
    ax.legend(framealpha=0.3, facecolor=CARD, edgecolor=TX2)
    ax.text(-3, 0.4, "Saturation Zone\nDerivative ≈ 0", color=R1, fontsize=8, ha="center", fontweight="bold")
    ax.text(3, 0.4, "Saturation Zone\nDerivative ≈ 0", color=R1, fontsize=8, ha="center", fontweight="bold")
    
    save("27_activation_saturation_regions.png")


def plot_28_wide_deep_vs_standard_mlp():
    print("[28] Wide & Deep vs Standard MLP")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Architectural Layout: Standard MLP vs. Wide & Deep Network", fontsize=14, fontweight="bold", color=TX)
    
    # Left: Standard MLP
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Standard Sequential MLP\n(Sequential information abstraction)", fontsize=11, fontweight="bold", color=B1)
    
    box(ax, 2.0, 4.0, 1.8, 1.2, color=B1, label="Inputs\nX", fontsize=9.5)
    box(ax, 5.0, 4.0, 1.8, 1.5, color=P1, label="Hidden Layers\n(Learns abstract\nconcepts)", fontsize=9, alpha=0.3)
    box(ax, 8.0, 4.0, 1.8, 1.2, color=R1, label="Output Layer\nŷ", fontsize=9.5)
    
    arrow(ax, 2.9, 4.0, 4.1, 4.0, color=TX2)
    arrow(ax, 5.9, 4.0, 7.1, 4.0, color=TX2)
    
    # Right: Wide & Deep
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Wide & Deep Architecture\n(Memorization + Generalization)", fontsize=11, fontweight="bold", color=G1)
    
    box(ax, 1.5, 4.0, 1.6, 1.2, color=B1, label="Inputs\nX", fontsize=9.5)
    box(ax, 4.5, 2.5, 1.8, 1.5, color=P1, label="Deep Path\nHidden Layers\n(Generalizes rules)", fontsize=8.5, alpha=0.3)
    box(ax, 7.5, 4.0, 1.6, 1.2, color=O1, label="Concatenate &\nOutput ŷ", fontsize=9)
    
    # Deep arrows
    arrow(ax, 2.3, 3.5, 3.6, 2.5, color=P1, lw=1.5)
    arrow(ax, 5.4, 2.5, 6.7, 3.5, color=P1, lw=1.5)
    
    # Wide arrows (direct shortcut bypass)
    arrow(ax, 2.3, 4.5, 6.7, 4.5, color=GOLD, lw=2.2)
    ax.text(4.5, 4.8, "Wide Path (Memorizes rules)", color=GOLD, fontsize=8.5, fontweight="bold", ha="center")
    
    plt.tight_layout()
    save("28_wide_deep_vs_standard_mlp.png")


def plot_29_keras_api_selection_flowchart():
    print("[29] Keras API Selection Flowchart")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    
    fig.suptitle("Decision Matrix: Selecting the Right Keras API", fontsize=14, fontweight="bold", color=TX)
    
    # Start
    box(ax, 6.0, 7.2, 2.0, 0.7, color=B1, label="START", fontsize=10)
    
    # Question 1
    box(ax, 6.0, 5.6, 4.5, 1.0, color=GOLD, label="Is your model structure a simple\nlinear stack of layers (single-I/O)?", fontsize=9.5, alpha=0.3)
    arrow(ax, 6.0, 6.8, 6.0, 6.1, color=TX)
    
    # Branch 1 (Yes -> Sequential)
    arrow(ax, 3.75, 5.6, 2.5, 5.6, color=G1)
    ax.text(3.1, 5.8, "YES", color=G1, fontsize=8.5, fontweight="bold")
    box(ax, 1.5, 5.6, 1.8, 0.8, color=G1, label="Sequential API", fontsize=9.5)
    
    # Question 2
    arrow(ax, 6.0, 5.1, 6.0, 4.2, color=R1)
    ax.text(6.1, 4.7, "NO", color=R1, fontsize=8.5, fontweight="bold")
    box(ax, 6.0, 3.6, 4.5, 1.0, color=GOLD, label="Does it require dynamic loops,\nconditionals, or runtime checks?", fontsize=9.5, alpha=0.3)
    
    # Branch 2 (Yes -> Subclassing)
    arrow(ax, 8.25, 3.6, 9.5, 3.6, color=G1)
    ax.text(8.9, 3.8, "YES", color=G1, fontsize=8.5, fontweight="bold")
    box(ax, 10.5, 3.6, 1.8, 0.8, color=P1, label="Subclassing API", fontsize=9.5)
    
    # Branch 3 (No -> Functional)
    arrow(ax, 6.0, 3.1, 6.0, 2.2, color=R1)
    ax.text(6.1, 2.7, "NO", color=R1, fontsize=8.5, fontweight="bold")
    box(ax, 6.0, 1.6, 4.5, 1.0, color=G1, label="Functional API\n(Production Standard: handles skip\nconnections, multiple inputs/outputs)", fontsize=9.5)
    
    save("29_keras_api_selection_flowchart.png")


# ── 1206. RUN ALL 29
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  CH 10: ANN Visual Generator v2 — 29 Graphs")
    print("=" * 60)

    # Original 10
    plot_activation_functions()
    plot_mlp_architecture()
    plot_training_curves()
    plot_gradient_descent()
    plot_backprop()
    plot_lr_finder()
    plot_gradient_flow()
    plot_hyperparameter_effects()
    plot_ann_timeline()
    plot_summary_dashboard()

    # New 15
    plot_bio_vs_artificial()
    plot_xor_problem()
    plot_loss_functions()
    plot_cross_entropy_intuition()
    plot_fashion_mnist_grid()
    plot_confusion_matrix()
    plot_decision_boundary()
    plot_wide_deep()
    plot_param_count()
    plot_three_apis()
    plot_callback_timeline()
    plot_early_stopping()
    plot_grid_vs_random()
    plot_transfer_learning()
    plot_neurons_vs_layers()
    
    # Extra 4
    plot_26_backprop_node_circuit()
    plot_27_activation_saturation_regions()
    plot_28_wide_deep_vs_standard_mlp()
    plot_29_keras_api_selection_flowchart()

    print("\n" + "=" * 60)
    print(f"  ✅  All 29 graphs saved to: {OUT}")
    print("=" * 60)

