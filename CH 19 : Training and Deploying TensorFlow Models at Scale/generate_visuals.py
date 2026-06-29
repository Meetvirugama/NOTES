import matplotlib.pyplot as plt
import numpy as np
import os

# Set global style for a premium dark theme
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#121212",
    "figure.facecolor": "#121212",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#A0A0A0",
    "ytick.color": "#A0A0A0",
    "grid.color": "#333333",
    "text.color": "#FFFFFF",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
})

VISUALS_DIR = "Visuals"
if not os.path.exists(VISUALS_DIR):
    os.makedirs(VISUALS_DIR)

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(VISUALS_DIR, fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")

# ---------------------------------------------------------
# Graph 01: TensorFlow Serving Architecture
# ---------------------------------------------------------
def generate_01():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    
    # Client
    ax.add_patch(plt.Rectangle((0.1, 0.4), 0.2, 0.2, color='#4CAF50', alpha=0.8))
    ax.text(0.2, 0.5, 'Client\n(REST/gRPC)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrow
    ax.annotate('', xy=(0.4, 0.5), xytext=(0.3, 0.5),
                arrowprops=dict(facecolor='#00BCD4', shrink=0.05, width=2, headwidth=10))
    
    # TF Serving Server
    ax.add_patch(plt.Rectangle((0.4, 0.2), 0.5, 0.6, color='#2c3e50', alpha=0.8, ec='#34495e', lw=3))
    ax.text(0.65, 0.75, 'TensorFlow Serving', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Model Versions
    ax.add_patch(plt.Rectangle((0.45, 0.5), 0.4, 0.15, color='#e74c3c', alpha=0.9))
    ax.text(0.65, 0.575, 'Model (v1)\nDefault', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((0.45, 0.3), 0.4, 0.15, color='#f39c12', alpha=0.9))
    ax.text(0.65, 0.375, 'Model (v2)\nA/B Testing', ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.title("01: TensorFlow Serving Architecture", fontsize=16, color="#E0E0E0", pad=20)
    save_fig("01_tfs_architecture")
    plt.close()

# ---------------------------------------------------------
# Graph 02: TFLite Conversion Pipeline
# ---------------------------------------------------------
def generate_02():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    # Keras Model
    ax.add_patch(plt.Rectangle((0.05, 0.4), 0.2, 0.2, color='#9b59b6', alpha=0.9))
    ax.text(0.15, 0.5, 'SavedModel\n(float32)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow 1
    ax.annotate('Convert &\nQuantize', xy=(0.4, 0.5), xytext=(0.25, 0.5),
                arrowprops=dict(facecolor='#bdc3c7', shrink=0.05, width=2, headwidth=8), ha='center', va='bottom', fontsize=9)
    
    # TFLite Flatbuffer
    ax.add_patch(plt.Rectangle((0.4, 0.4), 0.2, 0.2, color='#e67e22', alpha=0.9))
    ax.text(0.5, 0.5, '.tflite\n(int8)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow 2
    ax.annotate('Deploy', xy=(0.75, 0.5), xytext=(0.6, 0.5),
                arrowprops=dict(facecolor='#bdc3c7', shrink=0.05, width=2, headwidth=8), ha='center', va='bottom', fontsize=9)
    
    # Edge Device
    ax.add_patch(plt.Rectangle((0.75, 0.35), 0.2, 0.3, color='#2ecc71', alpha=0.9))
    ax.text(0.85, 0.5, 'Edge Device\n(Mobile/IoT)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.title("02: TFLite Conversion & Deployment Pipeline", fontsize=16, color="#E0E0E0", pad=20)
    save_fig("02_tflite_conversion")
    plt.close()

# ---------------------------------------------------------
# Graph 03: Mixed Precision
# ---------------------------------------------------------
def generate_03():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    # Master Weights
    ax.add_patch(plt.Rectangle((0.1, 0.6), 0.2, 0.2, color='#3498db', alpha=0.9))
    ax.text(0.2, 0.7, 'Master Weights\n(float32)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Forward Pass
    ax.annotate('Cast to', xy=(0.4, 0.7), xytext=(0.3, 0.7),
                arrowprops=dict(facecolor='#bdc3c7', shrink=0.05, width=1, headwidth=6))
    
    ax.add_patch(plt.Rectangle((0.4, 0.6), 0.2, 0.2, color='#1abc9c', alpha=0.9))
    ax.text(0.5, 0.7, 'Forward Pass\n(float16)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Backward Pass
    ax.annotate('Calc Gradients', xy=(0.7, 0.7), xytext=(0.6, 0.7),
                arrowprops=dict(facecolor='#bdc3c7', shrink=0.05, width=1, headwidth=6))
    
    ax.add_patch(plt.Rectangle((0.7, 0.6), 0.2, 0.2, color='#1abc9c', alpha=0.9))
    ax.text(0.8, 0.7, 'Gradients\n(float16)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Update Weights
    ax.annotate('Cast & Update', xy=(0.2, 0.6), xytext=(0.8, 0.6),
                arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=2, headwidth=8, connectionstyle="arc3,rad=-0.5"))
    
    plt.title("03: Mixed Precision Execution Flow", fontsize=16, color="#E0E0E0", pad=20)
    save_fig("03_mixed_precision")
    plt.close()

# ---------------------------------------------------------
# Graph 04: Data Parallelism
# ---------------------------------------------------------
def generate_04():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    
    # Nodes
    colors = ['#ff7675', '#74b9ff', '#55efc4', '#ffeaa7']
    labels = ['GPU 1\n(Batch 1)', 'GPU 2\n(Batch 2)', 'GPU 3\n(Batch 3)', 'GPU 4\n(Batch 4)']
    
    for i in range(4):
        x = 0.1 + i * 0.22
        ax.add_patch(plt.Circle((x, 0.7), 0.08, color=colors[i], alpha=0.9))
        ax.text(x, 0.7, labels[i], ha='center', va='center', fontsize=9, fontweight='bold', color='black')
        
        # Arrows pointing to Ring
        ax.annotate('', xy=(x, 0.4), xytext=(x, 0.62),
                    arrowprops=dict(facecolor='#dfe6e9', shrink=0.05, width=1.5, headwidth=7))
                    
        # Arrows pointing up from Ring
        ax.annotate('', xy=(x+0.05, 0.62), xytext=(x+0.05, 0.4),
                    arrowprops=dict(facecolor='#a29bfe', shrink=0.05, width=1.5, headwidth=7))
    
    # AllReduce Ring
    ax.add_patch(plt.Rectangle((0.05, 0.2), 0.85, 0.2, color='#2d3436', alpha=0.9, ec='#636e72', lw=2))
    ax.text(0.475, 0.3, 'Ring AllReduce (Gradient Synchronization)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.title("04: Data Parallelism with AllReduce", fontsize=16, color="#E0E0E0", pad=20)
    save_fig("04_data_parallelism")
    plt.close()

if __name__ == "__main__":
    print("Generating visuals for Chapter 19...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    print("All visuals generated successfully.")
