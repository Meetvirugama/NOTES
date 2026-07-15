import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import matplotlib.patches as patches

# Create Visuals directory if it doesn't exist (failsafe)
os.makedirs('/Users/meetvirugama/Desktop/NOTES/CN/Visuals', exist_ok=True)

# Set global style
plt.style.use('dark_background')
COLORS = ['#00ffcc', '#ff00ff', '#ffff00', '#00ccff', '#ff3333']

def generate_topologies():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Network Topologies', fontsize=24, fontweight='bold', color='white', y=0.98)
    
    # 1. Star Topology
    ax = axes[0, 0]
    G_star = nx.star_graph(6)
    pos_star = nx.spring_layout(G_star, seed=42)
    nx.draw(G_star, pos_star, ax=ax, with_labels=False, node_color=COLORS[0], node_size=800, edge_color='white', width=2)
    ax.set_title('Star Topology', fontsize=16, color=COLORS[0], pad=15)
    ax.text(0, -1.2, "Central hub with spokes.\nFails if hub dies.", color='white', ha='center', fontsize=12)

    # 2. Mesh Topology
    ax = axes[0, 1]
    G_mesh = nx.complete_graph(6)
    pos_mesh = nx.circular_layout(G_mesh)
    nx.draw(G_mesh, pos_mesh, ax=ax, with_labels=False, node_color=COLORS[1], node_size=800, edge_color='white', width=1.5, alpha=0.9)
    ax.set_title('Mesh Topology', fontsize=16, color=COLORS[1], pad=15)
    ax.text(0, -1.3, "Every node connected to every other.\nHighest reliability.", color='white', ha='center', fontsize=12)

    # 3. Ring Topology
    ax = axes[0, 2]
    G_ring = nx.cycle_graph(6)
    pos_ring = nx.circular_layout(G_ring)
    nx.draw(G_ring, pos_ring, ax=ax, with_labels=False, node_color=COLORS[2], node_size=800, edge_color='white', width=2)
    ax.set_title('Ring Topology', fontsize=16, color=COLORS[2], pad=15)
    ax.text(0, -1.3, "Closed loop.\nData travels in one direction.", color='white', ha='center', fontsize=12)

    # 4. Bus Topology
    ax = axes[1, 0]
    ax.plot([-1, 5], [0, 0], color='white', lw=4) # Backbone
    for i in range(5):
        ax.plot([i, i], [0, 1], color='white', lw=2) # Drop lines
        ax.scatter([i], [1], color=COLORS[3], s=800, zorder=5) # Nodes
    ax.set_xlim(-1.5, 5.5)
    ax.set_ylim(-1, 2)
    ax.set_title('Bus Topology', fontsize=16, color=COLORS[3], pad=15)
    ax.text(2, -0.8, "Single shared backbone.\nFails if backbone snaps.", color='white', ha='center', fontsize=12)
    ax.axis('off')

    # 5. Leaf-Spine (Modern)
    ax = axes[1, 1]
    G_ls = nx.Graph()
    spines = ['S1', 'S2']
    leaves = ['L1', 'L2', 'L3', 'L4']
    G_ls.add_nodes_from(spines)
    G_ls.add_nodes_from(leaves)
    for s in spines:
        for l in leaves:
            G_ls.add_edge(s, l)
    
    pos_ls = {}
    for i, s in enumerate(spines): pos_ls[s] = (i*2-1, 1)
    for i, l in enumerate(leaves): pos_ls[l] = (i-1.5, 0)
    
    nx.draw(G_ls, pos_ls, ax=ax, with_labels=False, node_color=COLORS[4], node_size=800, edge_color='white', width=2)
    ax.set_title('Leaf-Spine (Modern AI Standard)', fontsize=16, color=COLORS[4], pad=15)
    ax.text(0, -0.4, "Every leaf connects to every spine.\nUltra-low latency for AI clusters.", color='white', ha='center', fontsize=12)

    # Hide the empty 6th subplot
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('/Users/meetvirugama/Desktop/NOTES/CN/Visuals/topologies.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_osi_model():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    layers = [
        "7. Application (HTTP, DNS)",
        "6. Presentation (TLS/SSL)",
        "5. Session (Sockets)",
        "4. Transport (TCP, UDP)",
        "3. Network (IP, Routers)",
        "2. Data Link (MAC, Switches)",
        "1. Physical (Cables, Bits)"
    ]
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, 7))
    
    for i, (layer, color) in enumerate(zip(layers, colors)):
        rect = plt.Rectangle((0, i), 1, 0.8, color=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(0.5, i + 0.4, layer, ha='center', va='center', color='white', fontsize=16, fontweight='bold')
        
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 7)
    ax.axis('off')
    plt.title('The OSI Reference Model', fontsize=22, fontweight='bold', color='white', pad=20)
    plt.text(0.5, 7.2, "Mnemonic: Please Do Not Throw Sausage Pizza Away", ha='center', color='#aaaaaa', fontsize=12, style='italic')
    
    plt.savefig('/Users/meetvirugama/Desktop/NOTES/CN/Visuals/osi_model.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_switching():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Switching Techniques', fontsize=22, fontweight='bold', color='white')
    
    # Circuit Switching
    ax1 = axes[0]
    ax1.plot([0, 10], [5, 5], color=COLORS[0], lw=4) # Dedicated line
    ax1.plot([0, 10], [8, 8], color='#333333', lw=2, linestyle='--') 
    ax1.plot([0, 10], [2, 2], color='#333333', lw=2, linestyle='--')
    ax1.scatter([0, 10], [5, 5], color='white', s=200, zorder=5)
    ax1.text(0, 5.5, "Source", color='white', fontsize=12, ha='center')
    ax1.text(10, 5.5, "Dest", color='white', fontsize=12, ha='center')
    ax1.set_title('Circuit Switching', fontsize=16, color=COLORS[0])
    ax1.text(5, 4, "Dedicated Physical Path\n(High setup time, no delay during transfer)", color='white', ha='center', fontsize=10)
    ax1.axis('off')
    
    # Packet Switching
    ax2 = axes[1]
    
    # Paths
    ax2.plot([0, 3, 7, 10], [5, 8, 8, 5], color='#444444', lw=2, linestyle='--')
    ax2.plot([0, 5, 10], [5, 5, 5], color='#444444', lw=2, linestyle='--')
    ax2.plot([0, 3, 7, 10], [5, 2, 2, 5], color='#444444', lw=2, linestyle='--')
    
    # Nodes
    ax2.scatter([0, 10], [5, 5], color='white', s=200, zorder=5)
    ax2.scatter([3, 7, 5, 3, 7], [8, 8, 5, 2, 2], color='#777777', s=100, zorder=5) # Routers
    
    # Packets
    ax2.scatter([2], [7], color=COLORS[1], s=150, marker='s', zorder=10) # P1
    ax2.scatter([6], [5], color=COLORS[2], s=150, marker='s', zorder=10) # P2
    ax2.scatter([4], [2], color=COLORS[3], s=150, marker='s', zorder=10) # P3
    
    ax2.text(2, 7.5, "P1", color='white', fontsize=10, ha='center')
    ax2.text(6, 5.5, "P2", color='white', fontsize=10, ha='center')
    ax2.text(4, 2.5, "P3", color='white', fontsize=10, ha='center')
    
    ax2.set_title('Packet Switching', fontsize=16, color=COLORS[1])
    ax2.text(5, 0, "Data broken into packets.\nEach packet finds its own best route.", color='white', ha='center', fontsize=10)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/meetvirugama/Desktop/NOTES/CN/Visuals/switching.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_network_types():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw nested circles for scale
    circle_wan = plt.Circle((0, 0), 4, color=COLORS[3], alpha=0.3)
    circle_man = plt.Circle((0, -1), 3, color=COLORS[1], alpha=0.5)
    circle_lan = plt.Circle((0, -2), 2, color=COLORS[0], alpha=0.8)
    
    ax.add_patch(circle_wan)
    ax.add_patch(circle_man)
    ax.add_patch(circle_lan)
    
    ax.text(0, -2, "LAN\n(Local Area Network)\nHome, Office\nFastest Speed", color='black', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(0, 1, "MAN\n(Metropolitan Area Network)\nCity-wide\nModerate Speed", color='white', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(0, 3.2, "WAN\n(Wide Area Network)\nGlobal, Internet\nHigh Latency", color='white', ha='center', va='center', fontweight='bold', fontsize=14)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis('off')
    plt.title('Types of Networks (Scale)', fontsize=22, fontweight='bold', color='white')
    
    plt.savefig('/Users/meetvirugama/Desktop/NOTES/CN/Visuals/network_types.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_tcp_udp():
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('TCP vs UDP', fontsize=24, fontweight='bold', color='white', y=0.95)
    
    # TCP Sequence
    ax1 = axes[0]
    ax1.plot([0, 0], [10, 0], color='white', lw=2) # Sender timeline
    ax1.plot([10, 10], [10, 0], color='white', lw=2) # Receiver timeline
    
    ax1.text(0, 10.5, "Sender", color='white', fontsize=14, ha='center', fontweight='bold')
    ax1.text(10, 10.5, "Receiver", color='white', fontsize=14, ha='center', fontweight='bold')
    
    # 3-way handshake
    ax1.annotate('', xy=(10, 9), xytext=(0, 10), arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=2))
    ax1.text(5, 9.7, "SYN", color=COLORS[0], fontsize=12, ha='center')
    
    ax1.annotate('', xy=(0, 8), xytext=(10, 9), arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=2))
    ax1.text(5, 8.7, "SYN-ACK", color=COLORS[0], fontsize=12, ha='center')
    
    ax1.annotate('', xy=(10, 7), xytext=(0, 8), arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=2))
    ax1.text(5, 7.7, "ACK", color=COLORS[0], fontsize=12, ha='center')
    
    # Data transfer
    ax1.annotate('', xy=(10, 5), xytext=(0, 6), arrowprops=dict(arrowstyle="->", color='white', lw=2))
    ax1.text(5, 5.7, "Data Packet 1", color='white', fontsize=12, ha='center')
    
    ax1.annotate('', xy=(0, 4), xytext=(10, 5), arrowprops=dict(arrowstyle="->", color=COLORS[2], lw=2))
    ax1.text(5, 4.7, "ACK 1", color=COLORS[2], fontsize=12, ha='center')
    
    ax1.set_title('TCP (Reliable, Connection-Oriented)', fontsize=16, color=COLORS[0], pad=20)
    ax1.text(5, -1, "Guarantees delivery.\nUsed for Web, Email, File Transfer.", color='white', ha='center', fontsize=12)
    ax1.axis('off')
    
    # UDP Sequence
    ax2 = axes[1]
    ax2.plot([0, 0], [10, 0], color='white', lw=2) # Sender timeline
    ax2.plot([10, 10], [10, 0], color='white', lw=2) # Receiver timeline
    
    ax2.text(0, 10.5, "Sender", color='white', fontsize=14, ha='center', fontweight='bold')
    ax2.text(10, 10.5, "Receiver", color='white', fontsize=14, ha='center', fontweight='bold')
    
    # Data blast
    ax2.annotate('', xy=(10, 9), xytext=(0, 10), arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=2))
    ax2.text(5, 9.7, "Data Packet 1", color=COLORS[1], fontsize=12, ha='center')
    
    ax2.annotate('', xy=(10, 8), xytext=(0, 9), arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=2))
    ax2.text(5, 8.7, "Data Packet 2", color=COLORS[1], fontsize=12, ha='center')
    
    ax2.annotate('', xy=(8, 7), xytext=(0, 8), arrowprops=dict(arrowstyle="->", color='red', lw=2)) # Dropped packet
    ax2.text(4, 7.7, "Data Packet 3 (Dropped)", color='red', fontsize=12, ha='center')
    
    ax2.annotate('', xy=(10, 6), xytext=(0, 7), arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=2))
    ax2.text(5, 6.7, "Data Packet 4", color=COLORS[1], fontsize=12, ha='center')
    
    ax2.set_title('UDP (Fast, Connectionless)', fontsize=16, color=COLORS[1], pad=20)
    ax2.text(5, -1, "Fire and forget. No acknowledgments.\nUsed for Live Video, Gaming.", color='white', ha='center', fontsize=12)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/meetvirugama/Desktop/NOTES/CN/Visuals/tcp_vs_udp.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_encoding():
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Line Encoding: Manchester vs Differential', fontsize=20, fontweight='bold', color='white', y=0.95)
    
    bits = [0, 1, 0, 0, 1]
    time = np.arange(0, len(bits), 0.5)
    
    # Standard Manchester (0 = high to low, 1 = low to high)
    manchester = []
    for b in bits:
        if b == 0: manchester.extend([1, 0])
        else: manchester.extend([0, 1])
        
    ax1 = axes[0]
    ax1.step(time, manchester, where='post', color=COLORS[0], lw=3)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_xlim(0, len(bits))
    ax1.set_title("Standard Manchester Encoding", color=COLORS[0], fontsize=14)
    for i, b in enumerate(bits):
        ax1.text(i + 0.25, 1.2, str(b), color='white', fontsize=16, fontweight='bold')
        ax1.axvline(i, color='#333333', linestyle='--')
        
    # Differential Manchester (0 = transition at start, 1 = no transition at start. Always transition in middle)
    diff_manchester = []
    current_level = 1
    for b in bits:
        if b == 0:
            current_level = 1 - current_level # Transition at start
        
        # First half of bit
        diff_manchester.append(current_level)
        
        # Transition in middle
        current_level = 1 - current_level
        diff_manchester.append(current_level)
        
    ax2 = axes[1]
    ax2.step(time, diff_manchester, where='post', color=COLORS[1], lw=3)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_xlim(0, len(bits))
    ax2.set_title("Differential Manchester Encoding", color=COLORS[1], fontsize=14)
    for i, b in enumerate(bits):
        ax2.text(i + 0.25, 1.2, str(b), color='white', fontsize=16, fontweight='bold')
        ax2.axvline(i, color='#333333', linestyle='--')
        
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('/Users/meetvirugama/Desktop/NOTES/CN/Visuals/encoding.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_devices():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Networking Devices Comparison', fontsize=24, fontweight='bold', color='white', y=0.98)
    
    # 1. Repeater
    ax1 = axes[0, 0]
    ax1.scatter([5], [5], color='#777777', s=1500, marker='s', zorder=5) # Repeater
    ax1.text(5, 5, "REPEATER\n(Layer 1)", color='white', ha='center', va='center', fontweight='bold', fontsize=10)
    
    ax1.plot([1, 5], [5, 5], color='gray', lw=1) # Weak signal cable
    ax1.plot([5, 9], [5, 5], color='white', lw=4) # Strong signal cable
    
    # Weakened Signal (Decaying square wave)
    x_weak = [1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4]
    y_weak = [5, 8, 8,   5,   5, 7, 7,   5,   5, 6, 6,   5,   5, 5]
    ax1.plot(x_weak, y_weak, color=COLORS[0], lw=2)
    ax1.text(2.5, 9, "Weakened Signal", color='white', ha='center', fontsize=12)
    
    # Regenerated Signal (Full square wave)
    x_regen = [6, 6, 6.5, 6.5, 7, 7, 7.5, 7.5, 8, 8, 8.5, 8.5, 9, 9]
    y_regen = [5, 8, 8,   5,   5, 8, 8,   5,   5, 8, 8,   5,   5, 5]
    ax1.plot(x_regen, y_regen, color=COLORS[0], lw=2)
    ax1.text(7.5, 9, "Regenerated Signal", color='white', ha='center', fontsize=12)
    
    ax1.set_title('Repeater (Megaphone)', color=COLORS[0], fontsize=16, pad=20)
    ax1.text(5, 1, "Boosts weak signals back to full strength.", color='white', ha='center', fontsize=12)
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # 2. Hub
    ax2 = axes[0, 1]
    ax2.scatter([5], [5], color='#777777', s=1500, marker='s', zorder=5)
    ax2.text(5, 5, "HUB", color='white', ha='center', va='center', fontweight='bold', fontsize=12)
    
    pc_pos = [(2,5), (8,5), (3.5, 7.5), (6.5, 7.5), (3.5, 2.5), (6.5, 2.5)]
    for px, py in pc_pos:
        ax2.plot([5, px], [5, py], color='white', lw=2)
        ax2.scatter([px], [py], color='white', s=500, zorder=5)
        
    ax2.text(2, 6, "Sender", color=COLORS[0], ha='center')
    
    # Broadcast arrows
    ax2.annotate('', xy=(4, 5), xytext=(2.5, 5), arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=3))
    
    for px, py in pc_pos[1:]:
        dx = (px - 5) * 0.4
        dy = (py - 5) * 0.4
        ax2.annotate('', xy=(5+dx, 5+dy), xytext=(5, 5), arrowprops=dict(arrowstyle="->", color=COLORS[3], lw=3))
        
    ax2.set_title('Hub (Shouting in a room)', color=COLORS[3], fontsize=16, pad=20)
    ax2.text(5, 0, "Broadcasts incoming data to ALL ports.\nCauses massive collisions.", color='white', ha='center', fontsize=12)
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # 3. Bridge
    ax3 = axes[0, 2]
    ax3.scatter([5], [5], color='#666666', s=1500, marker='s', zorder=5)
    ax3.text(5, 5, "BRIDGE", color='white', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Hubs
    ax3.scatter([3, 7], [5, 5], color='#777777', s=1000, marker='s', zorder=5)
    ax3.text(3, 4, "Hub", color='white', ha='center')
    ax3.text(7, 4, "Hub", color='white', ha='center')
    
    # Bridge to Hubs
    ax3.plot([3, 5], [5, 5], color='white', lw=2)
    ax3.plot([5, 7], [5, 5], color='white', lw=2)
    
    # Hubs to PCs
    ax3.plot([3, 1], [5, 7], color='white', lw=2)
    ax3.plot([3, 1], [5, 3], color='white', lw=2)
    ax3.plot([7, 9], [5, 7], color='white', lw=2)
    ax3.plot([7, 9], [5, 3], color='white', lw=2)
    
    ax3.scatter([1, 1, 9, 9], [7, 3, 7, 3], color='white', s=500, zorder=5)
    ax3.text(1, 8, "PC", color='white', ha='center')
    ax3.text(1, 2, "PC", color='white', ha='center')
    ax3.text(9, 8, "PC", color='white', ha='center')
    ax3.text(9, 2, "PC", color='white', ha='center')
    
    ax3.set_title('Bridge (Security Guard)', color=COLORS[2], fontsize=16, pad=20)
    ax3.text(5, 0, "Connects 2 Hubs (LAN segments).\nFilters traffic via MAC address.", color='white', ha='center', fontsize=12)
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    # 4. Switch
    ax4 = axes[1, 0]
    ax4.scatter([3, 7], [5, 5], color='#444444', s=1200, marker='o', zorder=5)
    ax4.text(3, 4, "Switch A", color='white', ha='center')
    ax4.text(7, 4, "Switch B", color='white', ha='center')
    
    ax4.plot([1, 3], [8, 5], color='white', lw=2)
    ax4.plot([1, 3], [2, 5], color='white', lw=2)
    ax4.plot([3, 7], [5, 5], color='white', lw=4)
    ax4.plot([7, 9], [5, 8], color='white', lw=2)
    ax4.plot([7, 9], [5, 2], color='white', lw=2)
    
    ax4.scatter([1, 1, 9, 9], [8, 2, 8, 2], color=COLORS[0], s=600, marker='s', zorder=5)
    ax4.text(1, 9, "Node 1", color='white', ha='center')
    ax4.text(1, 1, "Node 2", color='white', ha='center')
    ax4.text(9, 9, "Node 3", color='white', ha='center')
    ax4.text(9, 1, "Node 4", color='white', ha='center')
    
    ax4.set_title('Switch (Targeted Delivery)', color=COLORS[0], fontsize=16, pad=20)
    ax4.text(5, 0, "Uses MAC Tables to forward frames\nspecifically to the intended Node.", color='white', ha='center', fontsize=12)
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    # 5. Router
    ax5 = axes[1, 1]
    
    # Computer A and Router
    ax5.scatter([1], [5], color='#4488ff', s=500, marker='s', zorder=5)
    ax5.text(1, 4, "Comp A", color='white', ha='center')
    ax5.plot([1, 2.5], [5, 5], color='white', lw=2)
    
    ax5.scatter([2.5], [5], color='#333333', s=1000, marker='s', zorder=5)
    ax5.text(2.5, 4, "Router", color='white', ha='center')
    
    # Computer B
    ax5.scatter([9], [5], color='#4488ff', s=500, marker='s', zorder=5)
    ax5.text(9, 4, "Comp B", color='white', ha='center')
    
    # Top Path (Networks 1, 3, 5)
    nx_top = [4.5, 6, 7.5]
    ny_top = [7, 7, 7]
    ax5.scatter(nx_top, ny_top, color=COLORS[1], s=400, zorder=5)
    ax5.text(4.5, 8, "Net 1", color='white', ha='center', fontsize=9)
    ax5.text(6, 8, "Net 3", color='white', ha='center', fontsize=9)
    ax5.text(7.5, 8, "Net 5", color='white', ha='center', fontsize=9)
    
    ax5.plot([2.5, 4.5], [5, 7], color='white', lw=2)
    ax5.plot([4.5, 6], [7, 7], color='white', lw=2)
    ax5.plot([6, 7.5], [7, 7], color='white', lw=2)
    ax5.plot([7.5, 9], [7, 5], color='white', lw=2)
    
    # Bottom Path (Networks 2, 4)
    nx_bot = [5, 7]
    ny_bot = [3, 3]
    ax5.scatter(nx_bot, ny_bot, color=COLORS[1], s=400, zorder=5)
    ax5.text(5, 2, "Net 2", color='white', ha='center', fontsize=9)
    ax5.text(7, 2, "Net 4", color='white', ha='center', fontsize=9)
    
    ax5.plot([2.5, 5], [5, 3], color='white', lw=2)
    ax5.plot([5, 7], [3, 3], color='white', lw=2)
    ax5.plot([7, 9], [3, 5], color='white', lw=2)
    
    # Highlight the chosen path (e.g., bottom path is shorter)
    ax5.plot([2.5, 5], [5, 3], color=COLORS[0], lw=3)
    ax5.plot([5, 7], [3, 3], color=COLORS[0], lw=3)
    ax5.plot([7, 9], [3, 5], color=COLORS[0], lw=3)
    
    ax5.set_title('Router (Path Finding)', color=COLORS[1], fontsize=16, pad=20)
    ax5.text(5, 0, "Calculates the best path across multiple\nnetworks using IP routing tables.", color='white', ha='center', fontsize=12)
    ax5.axis('off')
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('/Users/meetvirugama/Desktop/NOTES/CN/Visuals/devices.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Topologies...")
    generate_topologies()
    print("Generating OSI Model...")
    generate_osi_model()
    print("Generating Switching Techniques...")
    generate_switching()
    print("Generating Network Types...")
    generate_network_types()
    print("Generating TCP vs UDP...")
    generate_tcp_udp()
    print("Generating Encoding...")
    generate_encoding()
    print("Generating Devices...")
    generate_devices()
    print("All visuals generated successfully in Visuals/ folder!")
