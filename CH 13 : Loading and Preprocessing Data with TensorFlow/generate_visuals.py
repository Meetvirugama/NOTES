"""
 ╔══════════════════════════════════════════════════════════════════╗
 ║   CH 13: Loading & Preprocessing Data — COMPLETE Visuals         ║
 ║   12 custom dark-theme graphs/diagrams for all 5 modules         ║
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

def plot_01_dataset_chaining():
    print("[01] Dataset Chaining repeat(3).batch(7)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    fig.suptitle("Dataset Transformations Chaining: repeat(3).batch(7)", fontsize=14, fontweight="bold", color=TX)
    
    # Raw source slices
    box(ax, 1.5, 5.0, 2.0, 0.6, color=B1, label="Source: tf.range(10)", fontsize=9.5)
    ax.text(1.5, 4.2, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", color=TX2, fontsize=9, ha="center")
    
    # Repeat block
    arrow(ax, 2.6, 5.0, 4.0, 5.0, color=TX)
    box(ax, 5.5, 5.0, 2.2, 0.6, color=P1, label="repeat(3) Transformation", fontsize=9.5)
    ax.text(5.5, 4.0, "Stream: 0..9, 0..9, 0..9\n(30 items total, lazy stream)", color=TX2, fontsize=9, ha="center")
    
    # Batch block
    arrow(ax, 6.7, 5.0, 8.2, 5.0, color=TX)
    box(ax, 9.8, 5.0, 2.2, 0.6, color=G1, label="batch(7) Transformation", fontsize=9.5)
    
    # Batches visualization
    ax.text(9.8, 3.5, "Batch 1: [0, 1, 2, 3, 4, 5, 6] (size 7)\nBatch 2: [7, 8, 9, 0, 1, 2, 3] (size 7)\nBatch 3: [4, 5, 6, 7, 8, 9, 0] (size 7)\nBatch 4: [1, 2, 3, 4, 5, 6, 7] (size 7)\nBatch 5: [8, 9] (size 2)", color=TX, fontsize=8.5, ha="center")
    
    # Footer disclaimer on memory copies
    ax.text(6.0, 0.8, "💡 repeat() and batch() do NOT copy data in memory. They represent lazy execution pipelines.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("01_dataset_chaining.png")


def plot_02_ingestion_pipeline():
    print("[02] CSV Ingestion Pipeline Flow")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("California Housing Ingestion & Preprocessing Pipeline with tf.data", fontsize=15, fontweight="bold", color=TX)
    
    # Ingestion steps
    box(ax, 1.8, 6.5, 2.4, 0.8, color=B1, label="1. list_files(filepaths)\nFile paths dataset shuffled", fontsize=8.5)
    box(ax, 1.8, 4.5, 2.4, 0.8, color=P1, label="2. interleave()\nReads 5 files in parallel,\ncycles line-by-line", fontsize=8.5)
    box(ax, 1.8, 2.5, 2.4, 0.8, color=GOLD, label="3. map(preprocess)\nParses CSV via decode_csv\nStacks & standardizes", fontsize=8.5)
    
    box(ax, 7.5, 6.5, 2.4, 0.8, color=O1, label="4. shuffle(buffer_size)\nUses RAM buffer to\nrandomize instances", fontsize=8.5)
    box(ax, 7.5, 4.5, 2.4, 0.8, color=O1, label="5. repeat(epochs)\nRepeats stream indices\nfor training iteration", fontsize=8.5)
    box(ax, 7.5, 2.5, 2.4, 0.8, color=G1, label="6. batch(batch_size)\nGroups elements into tensors\n(defaults to keeping remainder)", fontsize=8.5)
    
    box(ax, 13.0, 4.5, 2.4, 1.0, color=G1, label="7. prefetch(1)\nPre-loads 1 batch ahead\nto prevent GPU starvation", fontsize=9)
    
    # Connecting Arrows
    arrow(ax, 1.8, 6.1, 1.8, 4.9, color=TX)
    arrow(ax, 1.8, 4.1, 1.8, 2.9, color=TX)
    arrow(ax, 1.8, 2.1, 7.5, 2.1, color=TX) # bottom connection line
    ax.plot([1.8, 7.5], [1.5, 1.5], color=TX, lw=1.2, alpha=0.5)
    arrow(ax, 7.5, 1.5, 7.5, 2.1, color=TX)
    
    arrow(ax, 7.5, 2.9, 7.5, 4.1, color=TX)
    arrow(ax, 7.5, 4.9, 7.5, 6.1, color=TX)
    arrow(ax, 7.5, 6.9, 13.0, 6.9, color=TX)
    ax.plot([7.5, 13.0], [7.2, 7.2], color=TX, lw=1.2, alpha=0.5)
    arrow(ax, 13.0, 7.2, 13.0, 5.0, color=TX)
    
    ax.text(7.5, 0.5, "Pipeline constructs a computational DAG of lazy iterators. Execution starts only when model pulls batches.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("02_ingestion_pipeline.png")


def plot_03_prefetching_timeline():
    print("[03] Prefetching Timeline Comparison")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    fig.suptitle("Performance Tuning: Pipeline Execution Timeline with and without Prefetching", fontsize=14, fontweight="bold", color=TX)
    
    # Top plot: Without Prefetching
    ax = axes[0]
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["GPU (Train)", "CPU (Prep)"])
    ax.set_title("Without Prefetching (Sequential execution leads to hardware stalling)", fontsize=11, color=R1, fontweight="bold")
    ax.grid(True, axis="x")
    
    # Intervals without prefetching
    # T0-T1: CPU 1, T1-T2: GPU 1, T2-T3: CPU 2, T3-T4: GPU 2
    # draw boxes
    rects = [
        # CPU 1
        Rectangle((0, 1.6), 2, 0.8, color=P1, alpha=0.5),
        # GPU 1
        Rectangle((2, 0.6), 2, 0.8, color=B1, alpha=0.5),
        # CPU 2
        Rectangle((4, 1.6), 2, 0.8, color=P1, alpha=0.5),
        # GPU 2
        Rectangle((6, 0.6), 2, 0.8, color=B1, alpha=0.5),
    ]
    for r in rects:
        ax.add_patch(r)
    
    ax.text(1.0, 2.0, "Prepare 1", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(3.0, 1.0, "Train 1", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(5.0, 2.0, "Prepare 2", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(7.0, 1.0, "Train 2", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    
    # Bottom plot: With Prefetching
    ax = axes[1]
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["GPU (Train)", "CPU (Prep)"])
    ax.set_title("With Prefetching (Parallel processing matches throughput)", fontsize=11, color=G1, fontweight="bold")
    ax.grid(True, axis="x")
    ax.set_xlabel("Timeline (Arbitrary Units)")
    
    # Intervals with prefetching
    # CPU 1: 0-2, CPU 2: 2-4, CPU 3: 4-6
    # GPU 1: 2-4, GPU 2: 4-6
    rects2 = [
        # CPU 1
        Rectangle((0, 1.6), 2, 0.8, color=P1, alpha=0.5),
        # CPU 2
        Rectangle((2, 1.6), 2, 0.8, color=P1, alpha=0.5),
        # GPU 1
        Rectangle((2, 0.6), 2, 0.8, color=B1, alpha=0.5),
        # CPU 3
        Rectangle((4, 1.6), 2, 0.8, color=P1, alpha=0.5),
        # GPU 2
        Rectangle((4, 0.6), 2, 0.8, color=B1, alpha=0.5),
    ]
    for r in rects2:
        ax.add_patch(r)
        
    ax.text(1.0, 2.0, "Prepare 1", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(3.0, 2.0, "Prepare 2", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(3.0, 1.0, "Train 1", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(5.0, 2.0, "Prepare 3", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(5.0, 1.0, "Train 2", color=TX, ha="center", va="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    save("03_prefetching_timeline.png")


def plot_04_tfrecord_structure():
    print("[04] TFRecord Format Structure")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
    fig.suptitle("Anatomy of a TFRecord Binary Record", fontsize=14, fontweight="bold", color=TX)
    
    # Draw blocks
    # Length (8 bytes) -> Length CRC (4 bytes) -> Data (var bytes) -> Data CRC (4 bytes)
    box(ax, 2.0, 3.0, 2.2, 1.5, color=B1, label="Length\n(8 bytes / uint64)\nSize of data payload", fontsize=9.5)
    box(ax, 4.5, 3.0, 1.8, 1.5, color=R1, label="Length CRC\n(4 bytes)\nChecksum", fontsize=9.5)
    box(ax, 7.5, 3.0, 3.0, 1.5, color=G1, label="Data Payload\n(Variable bytes)\ne.g. Serialized Protobuf", fontsize=9.5)
    box(ax, 10.3, 3.0, 1.8, 1.5, color=R1, label="Data CRC\n(4 bytes)\nChecksum", fontsize=9.5)
    
    # Arrows
    arrow(ax, 3.2, 3.0, 3.5, 3.0, color=TX)
    arrow(ax, 5.5, 3.0, 5.9, 3.0, color=TX)
    arrow(ax, 9.1, 3.0, 9.3, 3.0, color=TX)
    
    ax.text(6.0, 1.2, "CRCs safeguard against file corruption on disk or across network streams.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("04_tfrecord_structure.png")


def plot_05_example_protobuf_schema():
    print("[05] Example Protobuf Schema Hierarchy")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    fig.suptitle("Hierarchical Structure of the Example Protobuf Schema", fontsize=14, fontweight="bold", color=TX)
    
    # Nodes
    box(ax, 6.0, 6.0, 2.5, 0.7, color=B1, label="Example\n(Root Wrapper)", fontsize=10)
    box(ax, 6.0, 4.6, 2.5, 0.7, color=P1, label="Features\n(Dictionary / Map)", fontsize=10)
    
    # Feature nodes
    box(ax, 2.5, 3.0, 2.0, 0.7, color=GOLD, label="Feature: \"name\"\n(Feature Object)", fontsize=9)
    box(ax, 6.0, 3.0, 2.0, 0.7, color=GOLD, label="Feature: \"id\"\n(Feature Object)", fontsize=9)
    box(ax, 9.5, 3.0, 2.0, 0.7, color=GOLD, label="Feature: \"emails\"\n(Feature Object)", fontsize=9)
    
    # Lists
    box(ax, 2.5, 1.4, 2.0, 0.7, color=G1, label="BytesList\n[b\"Alice\"]", fontsize=8.5, alpha=0.3)
    box(ax, 6.0, 1.4, 2.0, 0.7, color=G1, label="Int64List\n[123]", fontsize=8.5, alpha=0.3)
    box(ax, 9.5, 1.4, 2.0, 0.7, color=G1, label="BytesList\n[b\"a@b.com\", b\"c@d.com\"]", fontsize=8.5, alpha=0.3)
    
    # Arrows
    arrow(ax, 6.0, 5.6, 6.0, 5.0, color=TX)
    arrow(ax, 5.5, 4.2, 3.0, 3.4, color=TX)
    arrow(ax, 6.0, 4.2, 6.0, 3.4, color=TX)
    arrow(ax, 6.5, 4.2, 9.0, 3.4, color=TX)
    
    arrow(ax, 2.5, 2.6, 2.5, 1.8, color=TX)
    arrow(ax, 6.0, 2.6, 6.0, 1.8, color=TX)
    arrow(ax, 9.5, 2.6, 9.5, 1.8, color=TX)
    
    save("05_example_protobuf_schema.png")


def plot_06_sequence_example_schema():
    print("[06] SequenceExample Protobuf Schema Hierarchy")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    fig.suptitle("SequenceExample Protobuf Schema (Lists of Lists & Sequential Data)", fontsize=14, fontweight="bold", color=TX)
    
    # Root
    box(ax, 6.5, 6.0, 3.0, 0.7, color=B1, label="SequenceExample\n(Root Document)", fontsize=10)
    
    # Split: Context & FeatureLists
    box(ax, 3.5, 4.5, 3.2, 0.8, color=P1, label="context: Features\n(Single attributes like Author, Date)", fontsize=9.5)
    box(ax, 9.5, 4.5, 3.2, 0.8, color=P1, label="feature_lists: FeatureLists\n(Sequential dictionary lists)", fontsize=9.5)
    
    # Details context
    box(ax, 3.5, 2.5, 2.8, 0.8, color=GOLD, label="Map {\"author\": Feature,\n       \"date\": Feature}", fontsize=8.5)
    
    # Details sequences
    box(ax, 8.0, 2.5, 2.4, 0.8, color=GOLD, label="FeatureList: \"content\"\n(Text contents)", fontsize=8.5)
    box(ax, 11.0, 2.5, 2.4, 0.8, color=GOLD, label="FeatureList: \"comments\"\n(User reviews)", fontsize=8.5)
    
    # Sub lists
    box(ax, 8.0, 0.8, 2.4, 0.8, color=G1, label="Repeated Feature\ne.g., Sentence 1, Sentence 2", fontsize=8, alpha=0.3)
    box(ax, 11.0, 0.8, 2.4, 0.8, color=G1, label="Repeated Feature\ne.g., Comment 1, Comment 2", fontsize=8, alpha=0.3)
    
    # Arrows
    arrow(ax, 5.5, 5.6, 4.0, 5.0, color=TX)
    arrow(ax, 7.5, 5.6, 9.0, 5.0, color=TX)
    
    arrow(ax, 3.5, 4.0, 3.5, 3.0, color=TX)
    arrow(ax, 9.0, 4.0, 8.5, 3.0, color=TX)
    arrow(ax, 10.0, 4.0, 10.5, 3.0, color=TX)
    
    arrow(ax, 8.0, 2.0, 8.0, 1.3, color=TX)
    arrow(ax, 11.0, 2.0, 11.0, 1.3, color=TX)
    
    save("06_sequence_example_schema.png")


def plot_07_sparse_to_dense_tensor():
    print("[07] Sparse to Dense Representation mapping")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Data Mutability: SparseTensor vs. Dense Tensor Conversion", fontsize=15, fontweight="bold", color=TX)
    
    # Left: Sparse Tensor details
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("SparseTensor Representation\n(Highly memory efficient)", fontsize=12, color=O1, fontweight="bold")
    box(ax, 3.0, 4.8, 4.5, 1.0, color=O1, label="indices = [[0, 0], [0, 1], [1, 0]]\nvalues = [b\"a@b.com\", b\"c@d.com\", b\"x@y.com\"]\ndense_shape = [2, 3]", fontsize=9)
    
    # Visual grid of coordinates
    rect1 = Rectangle((0.8, 1.0), 4.4, 2.5, fill=True, facecolor=CARD, edgecolor=TX2, lw=1)
    ax.add_patch(rect1)
    ax.text(1.2, 3.1, "0,0 -> a@b.com", color=TX, fontsize=9.5)
    ax.text(1.2, 2.5, "0,1 -> c@d.com", color=TX, fontsize=9.5)
    ax.text(1.2, 1.9, "1,0 -> x@y.com", color=TX, fontsize=9.5)
    ax.text(1.2, 1.3, "(Others implicit empty / default)", color=TX2, fontsize=8.5, style="italic")
    
    # Right: Dense Tensor padded
    ax = axes[1]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Dense padded reconstruction\ntf.sparse.to_dense(..., default_value=b\"\")", fontsize=12, color=G1, fontweight="bold")
    
    box(ax, 3.0, 4.8, 4.5, 1.0, color=G1, label="Reconstructed matrix\nshape: (2, 3)\ndtype: tf.string", fontsize=9)
    
    # Matrix grid
    rect2 = Rectangle((0.8, 1.0), 4.4, 2.5, fill=True, facecolor=CARD, edgecolor=G1, lw=1.5)
    ax.add_patch(rect2)
    ax.text(3.0, 3.0, "[[ b\"a@b.com\",   b\"c@d.com\",   b\"\" ],", color=TX, ha="center", fontsize=9.5)
    ax.text(3.0, 2.0, " [ b\"x@y.com\",   b\"\",          b\"\" ]]", color=TX, ha="center", fontsize=9.5)
    ax.text(3.0, 1.3, "Filled with default padding", color=G1, ha="center", fontsize=8.5, style="italic")
    
    plt.tight_layout()
    save("07_sparse_to_dense_tensor.png")


def plot_08_lookup_table_oov_buckets():
    print("[08] StaticVocabularyTable and OOV Buckets")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("Categorical Encoding: StaticVocabularyTable & Out-Of-Vocabulary (OOV) Buckets", fontsize=15, fontweight="bold", color=TX)
    
    # Vocabulary list
    box(ax, 2.2, 5.0, 3.0, 3.2, color=B1, label="Vocabulary List (Known)\n0: \"<1H OCEAN\"\n1: \"INLAND\"\n2: \"NEAR OCEAN\"\n3: \"NEAR BAY\"\n4: \"ISLAND\"", fontsize=9)
    
    # Input batch
    box(ax, 6.5, 7.0, 3.5, 0.8, color=P1, label="Input Categories Batch:\n[\"NEAR BAY\", \"DESERT\", \"INLAND\"]", fontsize=9)
    
    # Lookup Gate
    box(ax, 6.5, 4.5, 2.2, 1.0, color=GOLD, label="StaticVocabularyTable\nLookup Operation", fontsize=9.5)
    arrow(ax, 6.5, 6.5, 6.5, 5.1, color=TX)
    
    # OOV Buckets
    box(ax, 10.8, 5.0, 3.0, 2.0, color=R1, label="OOV Buckets (Unknown)\nnum_oov_buckets = 2\n5: hash(\"DESERT\") % 2 + 5\n6: hash(\"other\") % 2 + 5", fontsize=9)
    
    # Results
    box(ax, 6.5, 1.8, 4.5, 1.2, color=G1, label="Output Category Indices:\n- \"NEAR BAY\"  -->  index 3  (Known)\n- \"DESERT\"    -->  index 5  (OOV Bucket 1)\n- \"INLAND\"    -->  index 1  (Known)", fontsize=9.5)
    arrow(ax, 6.5, 3.9, 6.5, 2.5, color=TX)
    
    # Linking vocabulary to table
    arrow(ax, 3.8, 5.0, 5.3, 4.7, color=B1)
    # Linking OOV to table
    arrow(ax, 9.2, 5.0, 7.7, 4.7, color=R1)
    
    ax.text(6.5, 0.4, "OOV buckets use category string hashing to prevent collisions for unknown variables in test datasets.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("08_lookup_table_oov_buckets.png")


def plot_09_embedding_lookup_efficiency():
    print("[09] Embedding Lookup Computational Efficiency")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Computational Efficiency: Dense One-Hot Multiplier vs. Direct Index Retrieval", fontsize=15, fontweight="bold", color=TX)
    
    # Left: Dense Multiplication path
    ax = axes[0]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Standard Dense Layer (One-Hot Multiplier)\n(Computationally wasteful)", fontsize=11, color=R1, fontweight="bold")
    box(ax, 3.0, 4.8, 4.2, 1.0, color=R1, label="Inputs: One-Hot Vectors\ne.g., [0, 0, 0, 1, 0, 0, 0]\n(Shape: [4, 7])", fontsize=9)
    
    box(ax, 3.0, 2.8, 4.0, 1.0, color=TX2, label="Multiplication Operation:\n[4, 7] @ [7, 2] weights matrix\nRequires massive O(N*D) flops", fontsize=9, alpha=0.2)
    box(ax, 3.0, 1.0, 3.8, 0.8, color=B1, label="Resulting Embeddings\n(Shape: [4, 2])", fontsize=9)
    arrow(ax, 3.0, 4.2, 3.0, 3.4, color=TX)
    arrow(ax, 3.0, 2.2, 3.0, 1.5, color=TX)
    
    # Right: Direct lookup
    ax = axes[1]; ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Embedding Layer (Direct Index Lookup)\n(High execution speed)", fontsize=11, color=G1, fontweight="bold")
    box(ax, 3.0, 4.8, 4.2, 1.0, color=G1, label="Inputs: Integer Indices\ne.g., [3]\n(Shape: [4, 1])", fontsize=9)
    
    box(ax, 3.0, 2.8, 4.0, 1.0, color=G1, label="Lookup Operation:\ntf.nn.embedding_lookup(matrix, indices)\nRetrieves matrix row directly (O(1) complexity)", fontsize=9, alpha=0.3)
    box(ax, 3.0, 1.0, 3.8, 0.8, color=B1, label="Resulting Embeddings\n(Shape: [4, 2])", fontsize=9)
    arrow(ax, 3.0, 4.2, 3.0, 3.4, color=TX)
    arrow(ax, 3.0, 2.2, 3.0, 1.5, color=TX)
    
    save("09_embedding_lookup_efficiency.png")


def plot_10_tf_transform_architecture():
    print("[10] TF Transform Architecture")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("TF Transform Architecture: Resolving Training / Serving Skew", fontsize=15, fontweight="bold", color=TX)
    
    # Outline 1: Training Phase (Apache Beam)
    rect1 = Rectangle((0.5, 2.5), 6.5, 4.5, fill=True, facecolor="#101827", edgecolor=B1, lw=2.0)
    ax.add_patch(rect1)
    ax.text(3.75, 6.6, "1. TRAINING PHASE (Apache Beam)", color=B1, fontsize=10.5, fontweight="bold", ha="center")
    
    box(ax, 3.75, 5.5, 5.0, 0.8, color=B1, label="Raw Dataset (Large text / CSV logs)\nStored on cloud buckets", fontsize=9)
    box(ax, 3.75, 4.0, 5.0, 0.8, color=P1, label="Beam Pipeline + TFT Preprocessing\ntft.scale_to_z_score() & compute_vocabulary()", fontsize=9)
    box(ax, 3.75, 3.0, 5.0, 0.6, color=G1, label="Saves global stats & vocabulary metadata", fontsize=8.5)
    arrow(ax, 3.75, 5.0, 3.75, 4.5, color=TX)
    arrow(ax, 3.75, 3.5, 3.75, 3.4, color=TX)
    
    # Connection: export preprocessing function
    arrow(ax, 7.1, 4.0, 7.9, 4.0, color=GOLD, lw=2)
    ax.text(7.5, 4.5, "Generates & Exports\nTF Function\nwith Constants", color=GOLD, fontsize=8.5, ha="center", fontweight="bold")
    
    # Outline 2: Serving Phase (production model deployment)
    rect2 = Rectangle((8.0, 2.5), 6.5, 4.5, fill=True, facecolor="#1b1c1e", edgecolor=G1, lw=2.0)
    ax.add_patch(rect2)
    ax.text(11.25, 6.6, "2. SERVING PHASE (Production Deployment)", color=G1, fontsize=10.5, fontweight="bold", ha="center")
    
    box(ax, 11.25, 5.5, 5.0, 0.8, color=O1, label="New Production Instances (Raw Inputs)\nSubmitted to API or mobile browser client", fontsize=9)
    box(ax, 11.25, 4.0, 5.0, 0.8, color=GOLD, label="Deploys Preprocessing TF Function\nPerforms scaling on the fly using stats constants", fontsize=8.5)
    box(ax, 11.25, 2.9, 5.0, 0.6, color=G1, label="Keras Model Predictions (Inference ready)", fontsize=8.5)
    arrow(ax, 11.25, 5.0, 11.25, 4.5, color=TX)
    arrow(ax, 11.25, 3.5, 11.25, 3.3, color=TX)
    
    ax.text(7.5, 1.2, "TF Transform eliminates training/serving skew by exporting preprocessing variables directly as frozen tensor constants.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("10_tf_transform_architecture.png")


def plot_11_tfds_loading_pipeline():
    print("[11] TFDS Loading and Ingestion Pipeline")
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.axis("off")
    fig.suptitle("TFDS Loading Pipeline: Downloader to Keras Model Integration", fontsize=15, fontweight="bold", color=TX)
    
    # Pipeline steps
    box(ax, 2.0, 4.0, 2.8, 2.0, color=B1, label="1. tfds.load(name=\"mnist\")\nDownloads dataset shards,\nextracts, and caches to local\ndisk (~/tensorflow_datasets)", fontsize=8.5)
    box(ax, 6.0, 4.0, 2.8, 2.0, color=P1, label="2. Dataset Operations\n- shuffle(10000)\n- batch(32)\nCreates dataset iterators", fontsize=8.5)
    box(ax, 10.0, 4.0, 2.8, 2.0, color=GOLD, label="3. map(as_supervised=True)\nExtracts dictionary values\ninto standard Keras tuples:\n(image_tensor, label)", fontsize=8.5)
    box(ax, 13.6, 4.0, 2.0, 2.0, color=G1, label="4. model.fit()\nFeeds training set\ndirectly with\nprefetch(1) loop", fontsize=8.5)
    
    # Arrows
    arrow(ax, 3.5, 4.0, 4.5, 4.0, color=TX)
    arrow(ax, 7.5, 4.0, 8.5, 4.0, color=TX)
    arrow(ax, 11.5, 4.0, 12.5, 4.0, color=TX)
    
    ax.text(7.5, 1.2, "TFDS provides pre-sharded datasets with automated streaming interop for standard benchmark workloads.",
            ha="center", fontsize=9.5, color=GOLD, style="italic")
    
    save("11_tfds_loading_pipeline.png")


def plot_12_summary_dashboard():
    print("[12] Chapter 13 Summary Dashboard")
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("CH 13 Summary Dashboard: Loading & Preprocessing Pipelines with TensorFlow", fontsize=17, fontweight="bold", color=TX, y=0.98)
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Pipeline performance curve
    ax = fig.add_subplot(gs[0, 0])
    modes = ["Standard Python Ingestion", "tf.data (No Prefetching)", "tf.data (Prefetched & Cached)"]
    latencies = [42.5, 18.2, 3.5]
    bars = ax.barh(modes, latencies, color=[R1, O1, G1], alpha=0.8, height=0.45)
    ax.set_title("Benchmarking: Data Pipeline Ingestion Latency (per batch)", fontsize=11, fontweight="bold", color=TX)
    ax.set_xlabel("Latency (milliseconds) - Lower is Faster")
    ax.grid(True, axis="x")
    for bar, time in zip(bars, latencies):
        ax.text(time + 0.5, bar.get_y() + bar.get_height()/2, f"{time:.1f} ms", va="center", fontsize=9.5, color=TX, fontweight="bold")
    ax.set_xlim(0, 48)
    
    # 2. Dataset chaining flowchart mini
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Sequential Transformation Stack (DAG)", fontsize=11, fontweight="bold", color=TX)
    box(ax, 2.0, 2.5, 2.2, 1.0, color=B1, label="list_files()\nShuffled paths", fontsize=8.5, alpha=0.3)
    box(ax, 5.0, 2.5, 2.0, 1.0, color=GOLD, label="interleave().map()\nParsed & scaled", fontsize=8.5, alpha=0.2)
    box(ax, 8.0, 2.5, 2.2, 1.0, color=G1, label="prefetch(1)\nParallel output", fontsize=8.5, alpha=0.3)
    arrow(ax, 3.2, 2.5, 3.9, 2.5, color=TX)
    arrow(ax, 6.1, 2.5, 6.8, 2.5, color=TX)
    
    # 3. TFRecord Protobuf comparisons
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Protobuf Selection Matrix (TFRecord)", fontsize=11, fontweight="bold", color=TX)
    box(ax, 2.5, 2.5, 4.0, 2.8, color=B1, label="Example Protobuf\n\n- Maps feature_name -> values\n- Flat structural attributes\n- Best for: Tabular rows,\n  images, flattened text tokens", fontsize=8.5, alpha=0.3)
    box(ax, 7.5, 2.5, 4.0, 2.8, color=P1, label="SequenceExample Protobuf\n\n- Splitted: context vs feature_lists\n- Supports lists of lists (nested)\n- Best for: Sequential NLP tokens,\n  videos, frame-by-frame inputs", fontsize=8.5, alpha=0.3)
    
    # 4. Ingestion APIs Comparison table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("Preprocessing Strategies: Key Alternatives Matrix", fontsize=11, fontweight="bold", color=TX, pad=10)
    
    table_content = [
        ("Criterion", "dataset.map()", "Keras Layers", "TF Transform (TFT)"),
        ("Compute Time", "On the fly (per epoch)", "On the fly (per epoch)", "Ahead of time (once total)"),
        ("Model Export", "Requires manual layers", "Automatic (built in)", "Automatic (from code stats)"),
        ("Scale Scope", "Single CPU client", "Single CPU client", "Distributed Apache Beam"),
        ("Outlier Risk", "High (no global cache)", "High (no global cache)", "Zero (computed analyzers)"),
    ]
    
    y_pos = np.linspace(4.5, 0.5, len(table_content))
    for idx, (feat, map_a, keras_a, tft_a) in enumerate(table_content):
        weight = "bold" if idx == 0 else "normal"
        color = GOLD if idx == 0 else TX
        ax.text(0.1, y_pos[idx], feat, fontsize=10, fontweight=weight, color=color, ha="left")
        ax.text(3.1, y_pos[idx], map_a, fontsize=10, fontweight=weight, color=color, ha="left")
        ax.text(6.1, y_pos[idx], keras_a, fontsize=10, fontweight=weight, color=color, ha="left")
        ax.text(9.6, y_pos[idx], tft_a, fontsize=10, fontweight=weight, color=color, ha="left")
        ax.axhline(y_pos[idx] - 0.3, color="#21262d", lw=1)
        
    ax.set_xlim(0, 13); ax.set_ylim(0, 5)
    
    save("12_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Generating Chapter 13 Visual Assets...")
    plot_01_dataset_chaining()
    plot_02_ingestion_pipeline()
    plot_03_prefetching_timeline()
    plot_04_tfrecord_structure()
    plot_05_example_protobuf_schema()
    plot_06_sequence_example_schema()
    plot_07_sparse_to_dense_tensor()
    plot_08_lookup_table_oov_buckets()
    plot_09_embedding_lookup_efficiency()
    plot_10_tf_transform_architecture()
    plot_11_tfds_loading_pipeline()
    plot_12_summary_dashboard()
    print("🎉 All 12 visuals created successfully!")
