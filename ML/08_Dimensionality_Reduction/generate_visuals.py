import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#121212", "figure.facecolor": "#121212",
    "axes.edgecolor": "#444", "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#A0A0A0", "ytick.color": "#A0A0A0",
    "grid.color": "#2a2a2a", "text.color": "#FFFFFF",
    "font.family": "sans-serif",
})

VISUALS_DIR = "Visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

def save_fig(fig_id):
    path = os.path.join(VISUALS_DIR, fig_id + ".png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=plt.rcParams["figure.facecolor"])
    print(f"Saved: {path}")
    plt.close()

# -----------------------------------------------------------
# Graph 01: Projection vs Manifold Learning (Swiss Roll)
# -----------------------------------------------------------
def generate_01():
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    
    fig = plt.figure(figsize=(12, 5))
    
    # Left: Projection (Squashed)
    ax1 = fig.add_subplot(121)
    ax1.scatter(X[:, 0], X[:, 2], c=t, cmap=plt.cm.viridis, s=20, alpha=0.5)
    ax1.set_title("Projection onto 2D Plane\n(Squashed / Information Lost)", color='#E0E0E0')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Right: Manifold Unrolled
    ax2 = fig.add_subplot(122)
    ax2.scatter(t, X[:, 1], c=t, cmap=plt.cm.viridis, s=20, alpha=0.8)
    ax2.set_title("Manifold Learning\n(Unrolled / Structure Preserved)", color='#E0E0E0')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.suptitle("01: Projection vs Manifold Learning", fontsize=14, color='#E0E0E0')
    save_fig("01_projection_vs_manifold")

# -----------------------------------------------------------
# Graph 02: PCA Preserving Variance
# -----------------------------------------------------------
def generate_02():
    np.random.seed(4)
    m = 60
    w1, w2 = 0.1, 0.3
    noise = 0.1
    
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    X = np.empty((m, 2))
    X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
    X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    X = X - X.mean(axis=0) # center
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(X[:, 0], X[:, 1], "bo", alpha=0.5, color='#3498db')
    
    # Draw PCs
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        ax.annotate('', pca.mean_ + v, pca.mean_, arrowprops=dict(arrowstyle='->', linewidth=2, color='#e74c3c'))
        
    ax.plot([-1.5, 1.5], [-1.5 * pca.components_[0, 1]/pca.components_[0, 0], 1.5 * pca.components_[0, 1]/pca.components_[0, 0]], 
            '--', color='#e74c3c', label="PC1 (Max Variance)")
            
    ax.plot([-1.5, 1.5], [-1.5 * pca.components_[1, 1]/pca.components_[1, 0], 1.5 * pca.components_[1, 1]/pca.components_[1, 0]], 
            ':', color='#2ecc71', label="PC2 (Less Variance)")
            
    ax.axis('equal')
    ax.legend(loc='lower right')
    ax.set_title("02: PCA Preserving Maximum Variance", color='#E0E0E0', pad=15)
    save_fig("02_pca_variance")

# -----------------------------------------------------------
# Graph 03: Explained Variance Ratio (Elbow)
# -----------------------------------------------------------
def generate_03():
    # Simulate MNIST PCA curve
    d = np.arange(1, 400)
    cumsum = 1 - np.exp(-d/50) # Simulated curve shape
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d, cumsum, linewidth=3, color='#3498db')
    
    # 95% line
    d_95 = np.argmax(cumsum >= 0.95)
    ax.plot([d_95, d_95], [0, 0.95], "r--")
    ax.plot([0, d_95], [0.95, 0.95], "r--")
    ax.plot(d_95, 0.95, "ro", markersize=8)
    
    ax.annotate("95% Variance\nPreserved", xy=(d_95, 0.95), xytext=(d_95+20, 0.8),
                arrowprops=dict(arrowstyle="->", color="#FFFFFF"), color="#FFFFFF")
                
    ax.set_xlabel("Dimensions (Principal Components)")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("03: Explained Variance Ratio (The Elbow Plot)", color='#E0E0E0', pad=15)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    save_fig("03_explained_variance")

# -----------------------------------------------------------
# Graph 04: Kernel PCA Pre-image
# -----------------------------------------------------------
def generate_04():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Original space
    ax.add_patch(plt.Circle((2, 2), 1, color='#3498db', alpha=0.3))
    ax.plot(2, 2, 'wo', markersize=8, label="Original Data X")
    ax.plot(2.5, 2.5, 'rx', markersize=12, markeredgewidth=3, label="Pre-image (Guessed X)")
    ax.annotate("", xy=(2.4, 2.4), xytext=(2.1, 2.1), arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=2))
    ax.text(2.6, 2.1, "Reconstruction\nError", color='#e74c3c')
    
    # Feature space (infinite dim conceptually)
    ax.add_patch(plt.Polygon([[6,1], [9,1], [10,3], [7,3]], color='#9b59b6', alpha=0.3))
    ax.plot(8, 2, 'wo', markersize=8, label="$\phi(X)$ (Infinite Dims)")
    ax.plot(8, 2, 'rx', markersize=12, markeredgewidth=3)
    
    # Arrows
    ax.annotate("Kernel Trick", xy=(7, 2), xytext=(3, 2), arrowprops=dict(arrowstyle="->", color="#FFFFFF", lw=2))
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.legend(loc="upper left")
    ax.set_title("04: Kernel PCA Pre-Image Concept", color='#E0E0E0')
    save_fig("04_kpca_preimage")

# -----------------------------------------------------------
# Graph 05: LLE Unrolling Swiss Roll (Conceptual diagram)
# -----------------------------------------------------------
def generate_05():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Draw points in a mesh to simulate local neighbors
    X = np.linspace(0, 10, 10)
    Y = np.linspace(0, 5, 5)
    XX, YY = np.meshgrid(X, Y)
    
    ax.scatter(XX, YY, color='#3498db', s=50)
    
    # Highlight a neighborhood
    ax.scatter(XX[2, 4], YY[2, 4], color='#e74c3c', s=150, zorder=5) # Center point
    neighbors_x = [XX[1,4], XX[3,4], XX[2,3], XX[2,5]]
    neighbors_y = [YY[1,4], YY[3,4], YY[2,3], YY[2,5]]
    ax.scatter(neighbors_x, neighbors_y, color='#2ecc71', s=100, zorder=5) # Neighbors
    
    for nx, ny in zip(neighbors_x, neighbors_y):
        ax.plot([XX[2,4], nx], [YY[2,4], ny], 'w-', lw=2, alpha=0.8)
        
    ax.text(XX[2,4]+0.2, YY[2,4]+0.2, "$x^{(i)}$", color='white', fontsize=14)
    ax.text(XX[2,5]+0.2, YY[2,5]+0.2, "Neighbors", color='#2ecc71', fontsize=12)
    
    ax.set_title("05: LLE Preserving Local Neighborhood Distances", color='#E0E0E0', pad=15)
    ax.axis('off')
    save_fig("05_lle_swiss_roll")

if __name__ == "__main__":
    print("Generating visuals for Chapter 8...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    generate_05()
    print("All Chapter 8 visuals generated successfully.")
