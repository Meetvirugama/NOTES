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

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', text_color='black', title=None):
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='#1565C0', facecolor=color, alpha=0.9, zorder=2)
    ax.add_patch(rect)
    
    if title:
        ax.text(x + w/2, y + h - 0.1, title, ha='center', va='top', fontsize=12, fontweight='bold', color=text_color, zorder=3)
        ax.text(x + w/2, y + h/2 - 0.1, text, ha='center', va='center', fontsize=11, color=text_color, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', color=text_color, zorder=3)

def draw_arrow(ax, start, end, color='#1565C0', text=""):
    ax.annotate(text, xy=end, xytext=start,
                arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=10, shrink=0.05),
                ha='center', va='center', fontsize=10, zorder=1,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

def generate_firecracker_diagram():
    fig, ax = setup_plot("AWS Lambda: Firecracker MicroVM Isolation (50 LPA)", "01_firecracker_isolation.png")
    
    # Physical Host
    draw_box(ax, 0, 0, 10, 5, "", color='#EEEEEE', title="EC2 Bare Metal Host (Nitro System)")
    
    # Kernel
    draw_box(ax, 0.5, 0.5, 9, 0.8, "Host Linux Kernel (KVM)", color='#BDBDBD')
    
    # MicroVMs
    draw_box(ax, 1, 2, 2.5, 2.5, "Lambda Env A\n(Customer 1)", color='#FFCDD2', title="MicroVM 1")
    draw_box(ax, 4, 2, 2.5, 2.5, "Lambda Env B\n(Customer 1)", color='#FFCDD2', title="MicroVM 2")
    draw_box(ax, 7, 2, 2.5, 2.5, "Lambda Env C\n(Customer 2)", color='#C8E6C9', title="MicroVM 3")
    
    draw_arrow(ax, (2.25, 2), (2.25, 1.3), text="Hardware Isolation")
    draw_arrow(ax, (5.25, 2), (5.25, 1.3), text="Hardware Isolation")
    draw_arrow(ax, (8.25, 2), (8.25, 1.3), text="Hardware Isolation")
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    save_plot(fig, "01_firecracker_isolation.png")

def generate_transit_gateway_diagram():
    fig, ax = setup_plot("AWS Transit Gateway: Hub & Spoke BGP Routing", "02_transit_gateway.png")
    
    # TGW
    draw_box(ax, 4, 3, 2, 2, "Transit Gateway\n(Distributed Hyperplane)", color='#FFE082')
    
    # Spoke VPCs
    draw_box(ax, 0.5, 5, 2, 1, "VPC A (App)", color='#E3F2FD')
    draw_box(ax, 7.5, 5, 2, 1, "VPC B (DB)", color='#E3F2FD')
    draw_box(ax, 0.5, 1, 2, 1, "VPC C (Logs)", color='#E3F2FD')
    
    # On-Premise
    draw_box(ax, 7.5, 1, 2, 1, "On-Prem Data Center", color='#E8F5E9')
    
    # Connections
    draw_arrow(ax, (2.5, 5.5), (4, 4.5), text="VPC Attach")
    draw_arrow(ax, (7.5, 5.5), (6, 4.5), text="VPC Attach")
    draw_arrow(ax, (2.5, 1.5), (4, 3.5), text="VPC Attach")
    
    # BGP
    draw_arrow(ax, (7.5, 1.5), (6, 3.5), color='#D32F2F', text="Direct Connect\n(BGP Propagation)")
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    save_plot(fig, "02_transit_gateway.png")

def generate_dynamodb_diagram():
    fig, ax = setup_plot("DynamoDB: The Hot Partition Bottleneck", "04_dynamodb_hot_partition.png")
    
    # API Request
    draw_box(ax, 4, 5, 2, 1, "API Request\nPK: 2023-10-01", color='#E0E0E0')
    
    # Consistent Hash Ring representation
    draw_box(ax, 1, 1, 2, 3, "Partition 1\n(Normal)", color='#C8E6C9')
    draw_box(ax, 4, 1, 2, 3, "Partition 2\n🔥 HOT 🔥\nThrottling!", color='#EF9A9A')
    draw_box(ax, 7, 1, 2, 3, "Partition 3\n(Idle)", color='#C8E6C9')
    
    draw_arrow(ax, (5, 5), (5, 4), color='#D32F2F', text="Hash(2023-10-01)\npoints to P2")
    
    ax.text(5, 0.5, "To Fix: Append _1, _2 to PK to shard traffic across P1, P2, P3", ha='center', fontsize=12, style='italic')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    save_plot(fig, "04_dynamodb_hot_partition.png")

if __name__ == "__main__":
    generate_firecracker_diagram()
    generate_transit_gateway_diagram()
    generate_dynamodb_diagram()
    print("✅ AWS architecture diagrams generated successfully in AWS/notes/Visuals/")
