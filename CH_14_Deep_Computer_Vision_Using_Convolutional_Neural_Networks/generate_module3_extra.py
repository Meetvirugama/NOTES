import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import os

# Set up the same premium dark theme colors as generate_visuals.py
BG = "#1A1A1A"
TX = "#F0F0F0"
TX2 = "#B0B0B0"
B1 = "#2A3B4C"
B1_HL = "#3D5A80"
B2 = "#2E4A35"
B2_HL = "#4A7A55"
B3 = "#4C2A2A"
B3_HL = "#803D3D"
B4 = "#4A3B2E"
B4_HL = "#7A5535"

def setup_fig(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    return fig, ax

def add_glow(alpha=0.3):
    return [pe.withStroke(linewidth=3, foreground="#000000", alpha=alpha)]

def draw_box(ax, x, y, w, h, text, color, text_color=TX, fontsize=12, alpha=1.0):
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, color=text_color, fontsize=fontsize, weight='bold',
            ha='center', va='center', zorder=3, path_effects=add_glow())

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=TX2, lw=2, shrinkA=5, shrinkB=5))

def plot_37_efficientnet_scaling():
    fig, ax = setup_fig((10, 5))
    
    # Base network
    draw_box(ax, 1, 2, 1.5, 1, "Base\nNetwork", B1, fontsize=11)
    
    # Width Scaling
    draw_arrow(ax, 2.5, 2.5, 3.5, 3.5)
    draw_box(ax, 3.5, 3, 2.5, 1, "Width Scaling (w)\n(More Channels)", B2, fontsize=10)
    
    # Depth Scaling
    draw_arrow(ax, 2.5, 2.5, 3.5, 2.5)
    draw_box(ax, 3.5, 2, 2.5, 1, "Depth Scaling (d)\n(More Layers)", B3, fontsize=10)
    
    # Resolution Scaling
    draw_arrow(ax, 2.5, 2.5, 3.5, 1.5)
    draw_box(ax, 3.5, 1, 2.5, 1, "Resolution (r)\n(Larger Image)", B4, fontsize=10)
    
    # Compound Scaling
    draw_arrow(ax, 6.0, 2.5, 7.0, 2.5)
    draw_box(ax, 7.0, 1.5, 2.5, 2, "Compound Scaling\n(w * d * r)\nEfficientNet", B1_HL, fontsize=11)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    plt.savefig('Visuals/37_efficientnet_scaling.png', facecolor=BG, bbox_inches='tight', dpi=300)
    plt.close()

def plot_38_mbconv_block():
    fig, ax = setup_fig((12, 4))
    
    draw_box(ax, 0.5, 1.5, 1.5, 1, "Input", B1)
    draw_arrow(ax, 2.0, 2.0, 2.5, 2.0)
    
    draw_box(ax, 2.5, 1.0, 2.0, 2.0, "1x1 Conv\nExpansion", B2)
    draw_arrow(ax, 4.5, 2.0, 5.0, 2.0)
    
    draw_box(ax, 5.0, 1.0, 2.0, 2.0, "3x3 Depthwise\nConv", B3)
    draw_arrow(ax, 7.0, 2.0, 7.5, 2.0)
    
    draw_box(ax, 7.5, 1.0, 2.0, 2.0, "SE Block\n(Attention)", B4)
    draw_arrow(ax, 9.5, 2.0, 10.0, 2.0)
    
    draw_box(ax, 10.0, 1.5, 1.5, 1.0, "1x1 Conv\nProjection", B1_HL)
    
    # Skip connection
    ax.plot([1.25, 1.25, 10.75, 10.75], [2.5, 3.5, 3.5, 2.5], color=TX2, lw=2, zorder=1)
    ax.annotate("", xy=(10.75, 2.5), xytext=(10.75, 3.0), arrowprops=dict(arrowstyle="->", color=TX2, lw=2))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    plt.savefig('Visuals/38_mbconv_block.png', facecolor=BG, bbox_inches='tight', dpi=300)
    plt.close()

def plot_39_vit_architecture():
    fig, ax = setup_fig((12, 6))
    
    draw_box(ax, 0.5, 3.5, 2, 2, "Image\n224x224", B1)
    draw_arrow(ax, 2.5, 4.5, 3.0, 4.5)
    
    draw_box(ax, 3.0, 3.5, 2.5, 2, "Split into\n16x16 Patches", B2)
    draw_arrow(ax, 5.5, 4.5, 6.0, 4.5)
    
    draw_box(ax, 6.0, 3.5, 2.5, 2, "Flatten & Linear\nProjection", B3)
    draw_arrow(ax, 8.5, 4.5, 9.0, 4.5)
    
    draw_box(ax, 9.0, 3.5, 2.5, 2, "+ Positional\nEmbeddings", B4)
    
    draw_arrow(ax, 10.25, 3.5, 10.25, 2.5)
    
    draw_box(ax, 6.0, 0.5, 8.5, 2, "Transformer Encoder\n(Multi-Head Self Attention -> MLP)", B1_HL)
    
    draw_arrow(ax, 6.0, 1.5, 5.5, 1.5)
    draw_box(ax, 3.5, 1.0, 2.0, 1.0, "MLP Head\n(Classifier)", B2_HL)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    plt.savefig('Visuals/39_vit_architecture.png', facecolor=BG, bbox_inches='tight', dpi=300)
    plt.close()

def plot_40_convnext_block():
    fig, ax = setup_fig((14, 4))
    
    draw_box(ax, 0.5, 1.5, 1.5, 1, "Input", B1)
    draw_arrow(ax, 2.0, 2.0, 2.5, 2.0)
    
    draw_box(ax, 2.5, 1.0, 2.5, 2.0, "7x7 Depthwise\nConv", B2)
    draw_arrow(ax, 5.0, 2.0, 5.5, 2.0)
    
    draw_box(ax, 5.5, 1.5, 1.5, 1.0, "Layer\nNorm", B3)
    draw_arrow(ax, 7.0, 2.0, 7.5, 2.0)
    
    draw_box(ax, 7.5, 1.5, 1.5, 1.0, "1x1 Conv", B4)
    draw_arrow(ax, 9.0, 2.0, 9.5, 2.0)
    
    draw_box(ax, 9.5, 1.5, 1.5, 1.0, "GELU", B1_HL)
    draw_arrow(ax, 11.0, 2.0, 11.5, 2.0)
    
    draw_box(ax, 11.5, 1.5, 1.5, 1.0, "1x1 Conv", B2_HL)
    
    ax.plot([1.25, 1.25, 13.5, 13.5], [2.5, 3.5, 3.5, 2.0], color=TX2, lw=2, zorder=1)
    ax.annotate("", xy=(13.5, 2.0), xytext=(13.5, 2.5), arrowprops=dict(arrowstyle="->", color=TX2, lw=2))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    plt.savefig('Visuals/40_convnext_block.png', facecolor=BG, bbox_inches='tight', dpi=300)
    plt.close()

def plot_41_evolution_timeline():
    fig, ax = setup_fig((12, 7))
    
    y = 5.5
    draw_box(ax, 1.0, y, 4.0, 0.8, "GoogLeNet (2014) - Parallel Inception", B1)
    draw_arrow(ax, 5.0, y+0.4, 6.0, y-0.1)
    
    y -= 1.0
    draw_box(ax, 6.0, y, 4.0, 0.8, "ResNet (2015) - Skip Connections", B2)
    draw_arrow(ax, 6.0, y+0.4, 5.0, y-0.1)
    
    y -= 1.0
    draw_box(ax, 1.0, y, 4.0, 0.8, "Xception (2016) - Depthwise Sep", B3)
    draw_arrow(ax, 5.0, y+0.4, 6.0, y-0.1)
    
    y -= 1.0
    draw_box(ax, 6.0, y, 4.0, 0.8, "SENet (2017) - Channel Attention", B4)
    draw_arrow(ax, 6.0, y+0.4, 5.0, y-0.1)
    
    y -= 1.0
    draw_box(ax, 1.0, y, 4.0, 0.8, "EfficientNet (2019) - Compound Scaling", B1_HL)
    draw_arrow(ax, 5.0, y+0.4, 6.0, y-0.1)
    
    y -= 1.0
    draw_box(ax, 6.0, y, 4.0, 0.8, "Vision Transformer (2020) - Attention", B2_HL)
    draw_arrow(ax, 6.0, y+0.4, 5.0, y-0.1)
    
    y -= 1.0
    draw_box(ax, 1.0, y, 4.0, 0.8, "ConvNeXt (2022) - Modern CNN", B3_HL)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 7)
    plt.savefig('Visuals/41_evolution_timeline.png', facecolor=BG, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    os.makedirs("Visuals", exist_ok=True)
    plot_37_efficientnet_scaling()
    plot_38_mbconv_block()
    plot_39_vit_architecture()
    plot_40_convnext_block()
    plot_41_evolution_timeline()
    print("Successfully generated extra module 3 visuals.")
