import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#121212",
    "figure.facecolor": "#121212",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#A0A0A0",
    "ytick.color": "#A0A0A0",
    "grid.color": "#2a2a2a",
    "text.color": "#FFFFFF",
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

# ---------------------------------------------------------
# Graph 01: ML Pipeline Architecture
# ---------------------------------------------------------
def generate_01():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)

    steps = [
        ("1. Frame\nProblem", "#3498db"),
        ("2. Get\nData", "#2ecc71"),
        ("3. EDA\nVisualize", "#e67e22"),
        ("4. Prepare\nData", "#9b59b6"),
        ("5. Train\nModel", "#e74c3c"),
        ("6. Fine-\nTune", "#1abc9c"),
        ("7. Present\nSolution", "#f1c40f"),
        ("8. Launch\nMonitor", "#e74c3c"),
    ]

    for i, (text, color) in enumerate(steps):
        x = 0.3 + i * 1.7
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.2), 1.3, 1.6, boxstyle='round,pad=0.1',
                                              facecolor=color + '40', edgecolor=color, lw=1.5))
        ax.text(x + 0.65, 2.0, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 1.3 + 0.4, 2.0), xytext=(x + 1.3, 2.0),
                        arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.5))

    ax.text(7, 0.5, 'Each step feeds into the next. Steps 3-6 are iterative.', ha='center',
            fontsize=9, color='#aaa', style='italic')

    plt.title("01: The 8-Stage End-to-End ML Project Workflow", fontsize=13, color='#E0E0E0', pad=10)
    save_fig("01_ml_pipeline")

# ---------------------------------------------------------
# Graph 02: Stratified vs Random Sampling
# ---------------------------------------------------------
def generate_02():
    # Simulate income categories proportions
    categories = ['Cat 1\n(<$15K)', 'Cat 2\n($15-30K)', 'Cat 3\n($30-45K)', 'Cat 4\n($45-60K)', 'Cat 5\n(>$60K)']
    full_data = [3.97, 31.88, 35.05, 17.64, 11.46]          # Ground truth
    stratified = [3.97, 31.88, 35.05, 17.64, 11.46]         # Stratified matches perfectly
    random_sample = [4.25, 30.10, 34.20, 18.50, 12.95]      # Random drifts

    x = np.arange(len(categories))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    r1 = ax.bar(x - width, full_data, width, label='Full Dataset', color='#3498db', alpha=0.85)
    r2 = ax.bar(x, stratified, width, label='Stratified Test Set', color='#2ecc71', alpha=0.85)
    r3 = ax.bar(x + width, random_sample, width, label='Random Test Set', color='#e74c3c', alpha=0.85)

    ax.set_xlabel('Income Category', fontsize=11)
    ax.set_ylabel('Proportion (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Annotate the mismatch
    ax.text(3.78, 12.5, 'Drift!', color='#e74c3c', fontsize=9, fontweight='bold')

    plt.title("02: Stratified vs. Random Sampling — Income Category Proportions", fontsize=13, color='#E0E0E0', pad=10)
    save_fig("02_stratified_sampling")

# ---------------------------------------------------------
# Graph 03: ColumnTransformer Pipeline Architecture
# ---------------------------------------------------------
def generate_03():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)

    BOX = dict(boxstyle='round,pad=0.4', lw=1.5)

    # Input
    ax.add_patch(mpatches.FancyBboxPatch((0.1, 1.5), 1.8, 2.0, **BOX, edgecolor='white', facecolor='#2c3e50'))
    ax.text(1.0, 2.5, 'Raw\nHousing\nData', ha='center', va='center', fontsize=10)

    # Split arrow
    ax.annotate('', xy=(2.8, 3.5), xytext=(1.9, 3.0), arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.5))
    ax.annotate('', xy=(2.8, 1.5), xytext=(1.9, 2.0), arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.5))

    # Numerical branch
    ax.text(4.0, 4.5, 'Numerical Branch', ha='center', fontsize=9, color='#3498db', fontweight='bold')
    for i, (txt, col) in enumerate([("Imputer\n(median)", '#3498db'), ("Feature\nEngineer", '#3498db'), ("Standard\nScaler", '#3498db')]):
        x = 2.8 + i * 2.0
        ax.add_patch(mpatches.FancyBboxPatch((x, 3.0), 1.7, 1.1, **BOX, edgecolor=col, facecolor=col+'33'))
        ax.text(x + 0.85, 3.55, txt, ha='center', va='center', fontsize=8.5)
        if i < 2:
            ax.annotate('', xy=(x + 1.7 + 0.3, 3.55), xytext=(x + 1.7, 3.55),
                        arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.2))

    # Categorical branch
    ax.text(4.0, 2.2, 'Categorical Branch', ha='center', fontsize=9, color='#e67e22', fontweight='bold')
    ax.add_patch(mpatches.FancyBboxPatch((2.8, 0.8), 1.7, 1.1, **BOX, edgecolor='#e67e22', facecolor='#e67e2233'))
    ax.text(3.65, 1.35, 'OneHot\nEncoder', ha='center', va='center', fontsize=8.5)

    # Concatenate
    ax.annotate('', xy=(9.0, 3.55), xytext=(8.5, 3.55), arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.5))
    ax.annotate('', xy=(9.0, 1.35), xytext=(4.5, 1.35), arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.5))
    ax.add_patch(mpatches.FancyBboxPatch((9.0, 1.0), 1.5, 3.0, **BOX, edgecolor='#2ecc71', facecolor='#2ecc7133'))
    ax.text(9.75, 2.5, 'Concat\n(ColumnTrans\nformer)', ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(11.2, 2.5), xytext=(10.5, 2.5), arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    ax.add_patch(mpatches.FancyBboxPatch((11.2, 1.8), 0.7, 1.4, **BOX, edgecolor='#9b59b6', facecolor='#9b59b633'))
    ax.text(11.55, 2.5, 'ML\nReady', ha='center', va='center', fontsize=7.5)

    plt.title("03: Full ColumnTransformer Pipeline Architecture", fontsize=13, color='#E0E0E0', pad=10)
    save_fig("03_pipeline_architecture")

# ---------------------------------------------------------
# Graph 04: Model Comparison Bar Chart
# ---------------------------------------------------------
def generate_04():
    models = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest\n(default)', 'Random\nForest\n(GridSearch)']
    cv_rmse_mean = [69052, 71408, 50182, 49682]
    cv_rmse_std = [2732, 2439, 2097, 1900]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, cv_rmse_mean, color=colors, alpha=0.85, width=0.55,
                  yerr=cv_rmse_std, capsize=6, error_kw={'ecolor': 'white', 'lw': 1.5})

    ax.set_ylabel('CV RMSE ($)', fontsize=11)
    ax.set_ylim(40000, 80000)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(47730, color='#1abc9c', linewidth=2, linestyle='--', label='Final Test RMSE: 47,730')
    ax.legend(fontsize=9)

    for bar, val in zip(bars, cv_rmse_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                f'${val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title("04: Model Comparison — 10-Fold CV RMSE (Lower is Better)", fontsize=13, color='#E0E0E0', pad=10)
    save_fig("04_model_comparison")

# ---------------------------------------------------------
# Graph 05: Feature Importances
# ---------------------------------------------------------
def generate_05():
    features = ['median\nincome', 'INLAND', 'pop_per\nhhold', 'longitude', 'latitude',
                'rooms_per\nhhold', 'bedrooms\nper_room', 'housing\nmed_age',
                'population', 'total\nrooms', 'households', 'total\nbedrooms', '<1H\nOCEAN']
    importances = [0.366, 0.165, 0.109, 0.073, 0.063, 0.056, 0.053, 0.041,
                   0.015, 0.015, 0.014, 0.014, 0.010]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['#f1c40f' if i <= 2 else '#3498db' for i in range(len(features))]
    bars = ax.barh(features[::-1], importances[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel('Feature Importance Score', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, importances[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

    yellow_patch = mpatches.Patch(color='#f1c40f', alpha=0.85, label='Top 3 Most Important')
    blue_patch = mpatches.Patch(color='#3498db', alpha=0.85, label='Other Features')
    ax.legend(handles=[yellow_patch, blue_patch], fontsize=9)

    plt.title("05: Random Forest Feature Importances (Best Model)", fontsize=13, color='#E0E0E0', pad=10)
    save_fig("05_feature_importances")

if __name__ == "__main__":
    print("Generating visuals for Chapter 2...")
    generate_01()
    generate_02()
    generate_03()
    generate_04()
    generate_05()
    print("All Chapter 2 visuals generated successfully.")
