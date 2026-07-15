# 🌐 Computer Networks: Switching Techniques (2026 Edition)
> **Source**: Gate Smashers CN Playlist (Videos 17-20)
> **Focus**: Placement Aptitude, Software Engineering (SWE), and AI/ML Interviews

---

## 🔀 1. Switching Techniques Overview
*Methods used in computer networks to route data from a source to a destination.*

### 📊 Switching Types Comparison
| Switching Technique | Core Concept | Pros | Cons | Real-World Example |
| :--- | :--- | :--- | :--- | :--- |
| **Circuit Switching** | Dedicated, reserved physical path (setup $\rightarrow$ transfer $\rightarrow$ teardown). | Guaranteed bandwidth, no out-of-order packets. | Inefficient resource use (wasted when idle), high setup delay. | Traditional PSTN Telephone |
| **Packet Switching** | Data chopped into packets; bandwidth shared dynamically; routed individually. | Highly efficient, resilient to link failures. | Variable delay (jitter), out-of-order packets, header overhead. | The Internet (TCP/IP) |
| **Message Switching** | Entire message transmitted and stored at each intermediate node (store-and-forward). | Better bandwidth use than Circuit Switching. | Very high latency, requires huge node storage. | Early telegraph/email relay |

---

## 📦 2. Packet Switching In-Depth
*Packet switching is the backbone of modern networks, split into Connectionless and Connection-Oriented types.*

### 📊 Datagram vs. Virtual Circuit
| Feature | Datagram Network (Connectionless) | Virtual Circuit (Connection-Oriented) |
| :--- | :--- | :--- |
| **Concept** | Packets treated independently; no setup phase. | Logical connection established before transfer. |
| **Path** | Routing decisions made per packet; path varies. | All packets follow the exact same pre-established path. |
| **Packet Ordering** | Packets may arrive out of order. | Packets arrive in perfect sequence. |
| **Overhead** | High (requires full source/destination IP in header). | Low (uses a small Virtual Circuit Identifier - VCID). |
| **State** | Routers do not keep state. | Routers maintain state for the Virtual Circuit. |
| **Example** | **IP (Internet Protocol)** | **Frame Relay, ATM, X.25** |
