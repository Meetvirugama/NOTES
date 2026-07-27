import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Create Visuals directory
os.makedirs("../notes/Visuals", exist_ok=True)

def setup_plot(title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    return fig, ax

def save_plot(fig, filename):
    plt.tight_layout()
    plt.savefig(f"../notes/Visuals/{filename}", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', text_color='black', title=None, alpha=0.9):
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='#1565C0', facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(rect)
    
    if title:
        ax.text(x + w/2, y + h - 0.2, title, ha='center', va='top', fontsize=12, fontweight='bold', color=text_color, zorder=3)
        ax.text(x + w/2, y + h/2 - 0.1, text, ha='center', va='center', fontsize=11, color=text_color, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', color=text_color, zorder=3)

def draw_arrow(ax, start, end, color='#1565C0', text=""):
    ax.annotate(text, xy=end, xytext=start,
                arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=10, shrink=0.05),
                ha='center', va='center', fontsize=10, zorder=1,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

def generate_namespace_diagram():
    fig, ax = setup_plot("Docker Linux Kernel Isolation (50 LPA)", "01_namespace_isolation.png")
    
    # Host OS
    draw_box(ax, 0, 0, 10, 6, "", color='#EEEEEE', title="Host Physical Machine (Linux Kernel)")
    
    # Global Resources
    draw_box(ax, 0.5, 0.5, 9, 1, "Global Kernel Resources (CPU, Memory, Network Cards)", color='#BDBDBD')
    
    # Container Boundary
    draw_box(ax, 2, 2, 6, 3.5, "", color='#FFF9C4', title="Container Process ('nginx')")
    
    # Namespaces
    draw_box(ax, 2.5, 2.5, 2, 1, "PID Namespace\n(Thinks it's PID 1)", color='#FFCDD2')
    draw_box(ax, 5.5, 2.5, 2, 1, "NET Namespace\n(Private eth0)", color='#C8E6C9')
    draw_box(ax, 2.5, 4, 2, 1, "MNT Namespace\n(Chroot isolated)", color='#BBDEFB')
    draw_box(ax, 5.5, 4, 2, 1, "Cgroups\n(Throttles RAM to 500MB)", color='#E1BEE7')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    save_plot(fig, "01_namespace_isolation.png")

def generate_overlayfs_diagram():
    fig, ax = setup_plot("OverlayFS & The Copy-on-Write Penalty (50 LPA)", "02_overlayfs_cow.png")
    
    # Host Volume
    draw_box(ax, 7, 1, 2.5, 4, "Host Directory\n/var/lib/postgresql/data\n\n(BYPASSES OVERLAYFS)", color='#C8E6C9', title="Bind Mount / Volume")
    
    # Image Layers
    draw_box(ax, 0.5, 1, 5, 1, "Layer 1: Base OS (Ubuntu)", color='#E0E0E0', title="Read-Only Image Layers")
    draw_box(ax, 0.5, 2.2, 5, 1, "Layer 2: Install PostgreSQL", color='#E0E0E0')
    
    # R/W Layer
    draw_box(ax, 0.5, 3.8, 5, 1.2, "Read/Write Container Layer\n(Temporary Data)", color='#FFCDD2', title="Running Container Layer")
    
    # CoW Arrow
    draw_arrow(ax, (3, 2.2), (3, 3.8), color='#D32F2F', text="Copy-on-Write (CoW)\nMassive I/O Penalty!")
    
    # Mount Arrow
    draw_arrow(ax, (5.5, 4.4), (7, 4.4), color='#388E3C', text="Writes directly to disk\n(Fast I/O)")
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    save_plot(fig, "02_overlayfs_cow.png")

def generate_networking_diagram():
    fig, ax = setup_plot("Docker Port Forwarding Internals (50 LPA)", "03_iptables_veth.png")
    
    # External Request
    draw_box(ax, 0, 3, 2, 1, "Internet Request\n(Port 8080)", color='#FFE082')
    
    # Iptables
    draw_box(ax, 3, 2.5, 2.5, 2, "Linux Kernel iptables\nPREROUTING DNAT\nRewrite -> 172.17.0.2:80", color='#FFCDD2')
    
    # docker0 bridge
    draw_box(ax, 6.5, 2.5, 1.5, 2, "docker0\nBridge", color='#E0E0E0')
    
    # Container
    draw_box(ax, 9, 2.5, 2, 2, "Container\n(Nginx Port 80)\n172.17.0.2", color='#C8E6C9')
    
    # Flow
    draw_arrow(ax, (2, 3.5), (3, 3.5), text="")
    draw_arrow(ax, (5.5, 3.5), (6.5, 3.5), text="")
    draw_arrow(ax, (8, 3.5), (9, 3.5), text="veth pair", color='#1565C0')
    
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 6)
    save_plot(fig, "03_iptables_veth.png")

if __name__ == "__main__":
    generate_namespace_diagram()
    generate_overlayfs_diagram()
    generate_networking_diagram()
    print("✅ Docker architecture diagrams generated successfully in Docker/notes/Visuals/")
