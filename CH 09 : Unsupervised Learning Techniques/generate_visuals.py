import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import silhouette_samples, silhouette_score

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
# Graph 01: K-Means Voronoi Tessellation
# -----------------------------------------------------------
def generate_01():
    X, *_ = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=42)
    kmeans = KMeans(n_clusters=5, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Voronoi using Meshgrid Contour (Much cleaner than scipy.spatial.Voronoi)
    mins = X.min(axis=0) - 1
    maxs = X.max(axis=0) + 1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),
                         np.linspace(mins[1], maxs[1], 1000))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    ax.contour(xx, yy, Z, linewidths=1, colors='w', alpha=0.5)
    
    # Plot points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.8, edgecolor='k')
    
    # Plot centroids
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               s=250, marker='X', c='white', edgecolor='black', linewidth=1.5, label="Centroids", zorder=5)
               
    ax.set_title("01: K-Means Voronoi Tessellation", color='#E0E0E0', pad=15)
    ax.legend()
    ax.axis('off')
    
    # Set limits to perfectly bound the contour
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    
    save_fig("01_voronoi_tessellation")

# -----------------------------------------------------------
# Graph 02: Inertia (Elbow Method)
# -----------------------------------------------------------
def generate_02():
    X, *_ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
    
    inertias = []
    k_values = range(1, 10)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(kmeans.inertia_)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, 'bo-', color='#3498db', linewidth=2, markersize=8)
    
    # Mark the elbow
    ax.plot(4, inertias[3], 'ro', markersize=12, label="The 'Elbow' (k=4)")
    ax.annotate("Optimal k", xy=(4.1, inertias[3] + 100), xytext=(5, inertias[3] + 1000),
                arrowprops=dict(facecolor='white', arrowstyle="->", color='#FFFFFF'))
    
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("02: The Inertia Elbow Method", color='#E0E0E0', pad=15)
    ax.grid(alpha=0.2)
    ax.legend()
    save_fig("02_inertia_elbow")

# -----------------------------------------------------------
# Graph 03: Silhouette Diagrams (Conceptual)
# -----------------------------------------------------------
def generate_03():
    # We will simulate a simplified view of silhouette diagrams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Good Silhouette (k=4)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, 100])
    
    y_lower = 10
    mean_score_good = 0.65
    for i in range(4):
        size = 20
        y_upper = y_lower + size
        color = plt.cm.viridis(float(i) / 4)
        
        # Simulate good cluster shape
        x_vals = np.sort(np.random.normal(mean_score_good + np.random.uniform(-0.1, 0.1), 0.1, size))
        x_vals = np.clip(x_vals, 0, 1) # bound it
        
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, x_vals, facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 2
        
    ax1.axvline(x=mean_score_good, color="red", linestyle="--")
    ax1.set_title("Good Model (k=4)\nAll clusters cross mean line", color='#2ecc71')
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Cluster label")
    ax1.set_yticks([])
    
    # Right: Bad Silhouette (k=6)
    ax2.set_xlim([-0.1, 1])
    ax2.set_ylim([0, 100])
    
    y_lower = 5
    mean_score_bad = 0.45
    for i in range(6):
        size = 15 if i % 2 == 0 else 5 # Uneven sizes
        y_upper = y_lower + size
        color = plt.cm.viridis(float(i) / 6)
        
        # Simulate bad cluster shape (some very low)
        base = 0.3 if i % 2 == 0 else 0.7
        x_vals = np.sort(np.random.normal(base, 0.15, size))
        x_vals = np.clip(x_vals, -0.1, 1)
        
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, x_vals, facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 2
        
    ax2.axvline(x=mean_score_bad, color="red", linestyle="--")
    ax2.set_title("Bad Model (k=6)\nSome clusters fall short, uneven sizes", color='#e74c3c')
    ax2.set_xlabel("Silhouette Coefficient")
    ax2.set_yticks([])
    
    plt.suptitle("03: Silhouette Diagrams", fontsize=14, color='#E0E0E0')
    save_fig("03_silhouette_diagram")

# -----------------------------------------------------------
# Graph 04: DBSCAN (Core vs Anomaly)
# -----------------------------------------------------------
def generate_04():
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    # Add some obvious outliers
    outliers = np.array([[-1.5, 1.5], [2.5, -1], [1.5, 1.5], [-0.5, -0.8]])
    X = np.vstack([X, outliers])
    
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    y_pred = dbscan.fit_predict(X)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot core clusters
    ax.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], color='#3498db', label="Cluster 0", s=30)
    ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], color='#2ecc71', label="Cluster 1", s=30)
    
    # Plot anomalies
    ax.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], color='#e74c3c', marker='x', s=100, linewidth=2, label="Anomalies (Noise)")
    
    # Draw an epsilon circle around one core point to illustrate
    core_idx = dbscan.core_sample_indices_[0]
    core_point = X[core_idx]
    circle = plt.Circle(core_point, 0.2, color='white', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    ax.annotate("$\epsilon$-neighborhood", xy=core_point, xytext=(core_point[0]-1, core_point[1]+0.5),
                arrowprops=dict(facecolor='white', arrowstyle="->", color='#FFFFFF'))
    
    ax.set_title("04: DBSCAN Clustering & Anomaly Detection", color='#E0E0E0', pad=15)
    ax.legend(loc='lower right')
    ax.axis('off')
    save_fig("04_dbscan")

# -----------------------------------------------------------
# Graph 05: Gaussian Mixture Model (Ellipsoids)
# -----------------------------------------------------------
def generate_05():
    # Create elongated blobs
    np.random.seed(42)
    n_samples = 300
    
    # Generate random sample, two components
    C = np.array([[0.0, -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X)
    y_pred = gmm.predict(X)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot data
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.6)
    
    # Function to draw ellipses (simulated density contours)
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        ax = ax or plt.gca()
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        for nsig in range(1, 4):
            ax.add_patch(plt.matplotlib.patches.Ellipse(position, width=nsig * width, height=nsig * height, angle=angle, **kwargs))
            
    # Draw GMM components
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=0.2, color='#e74c3c', fill=False, linewidth=2)
        
    ax.set_title("05: GMM Capturing Elongated Ellipsoidal Clusters", color='#E0E0E0', pad=15)
    ax.axis('off')
    save_fig("05_gmm_ellipsoids")

if __name__ == "__main__":
    print("Generating visuals for Chapter 9...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    generate_05()
    print("All Chapter 9 visuals generated successfully.")
