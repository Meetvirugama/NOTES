# 🌐 Computer Networks: Network Domains (2026 Edition)
> **Source**: Gate Smashers CN Playlist (Video 16)
> **Focus**: Placement Aptitude, Software Engineering (SWE), and AI/ML Interviews

---

## 💥 1. Collision Domain & Broadcast Domain
*Understanding how network devices manage traffic and collisions on a shared network.*

### 📊 Domain Definitions & Causes
| Domain Type | Definition | Common Causes / Traffic Types |
| :--- | :--- | :--- |
| **Collision Domain** | A network segment where data packets can collide when sent simultaneously. | Two devices on a shared medium transmitting at the exact same time. |
| **Broadcast Domain** | A logical division where all nodes can reach each other via a broadcast. | ARP (Address Resolution Protocol), DHCP discovery (`FF:FF:FF:FF:FF:FF`). |

### 🛠️ Effect of Network Devices on Domains
| Device | OSI Layer | Impact on Collision Domains | Impact on Broadcast Domains |
| :--- | :--- | :--- | :--- |
| **Hub** | Layer 1 | None (All ports share **1** collision domain) | None (All ports share **1** broadcast domain) |
| **Switch** | Layer 2 | **Separates** (1 collision domain *per port*) | None (All ports share **1** broadcast domain by default) |
| **Router** | Layer 3 | **Separates** (Each interface is a collision domain) | **Separates** (Routers do not forward broadcast packets) |

### 🎯 Network Segmentation & Reducing Collisions
- **Reducing Collisions:** Replacing Hubs with Switches eliminates most collisions by isolating ports.
- **Reducing Broadcasts:** Using Routers or VLANs (on switches) segments large broadcast domains into smaller ones, reducing network radiation.

### 📝 Quick Example
- **Hub:** Has **one** collision domain and **one** broadcast domain.
- **Switch:** Creates **one collision domain per port**, but all ports share **one broadcast domain** by default.
