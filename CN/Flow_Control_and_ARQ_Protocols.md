# 🌐 Computer Networks: Flow Control & ARQ (2026 Edition)
> **Source**: Gate Smashers CN Playlist (Videos 25-30)
> **Focus**: Placement Aptitude, Software Engineering (SWE), and AI/ML Interviews

---

## 🚦 1. Flow Control Basics
*Synchronizing a fast sender with a slow receiver to prevent buffer overflow and dropped packets.*

- **Why it's needed:** If a gigabit server blasts data to a slow smartphone, the phone's memory buffer fills up. Flow control forces the server to pause and wait.

---

## 🔄 2. Flow Control & ARQ Protocols
*ARQ (Automatic Repeat reQuest) adds error control (timeouts and retransmissions) to flow control over noisy links.*

### 📊 Protocol Comparison
| Protocol | Mechanism / Concept | Sender Window | Receiver Window | Pros & Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Stop-and-Wait** | Send 1 frame, wait for ACK. If timeout, retransmit. | 1 | 1 | **Pros**: Very simple.<br>**Cons**: Terrible efficiency on long-distance links (mostly idle). |
| **Go-Back-N ARQ** | Send $N$ frames (Sliding Window). Receiver strictly accepts in-order. Cumulative ACKs. | $N$ | 1 | **Pros**: Simple receiver.<br>**Cons**: If 1 frame drops, must retransmit it AND all subsequent valid frames. |
| **Selective Repeat ARQ** | Send $N$ frames. Receiver buffers out-of-order frames. Individual ACKs. | $N$ (e.g., $2^{m-1}$) | $N$ | **Pros**: Highly efficient bandwidth (only retransmits dropped frame).<br>**Cons**: Complex receiver with memory buffering. |

---

## 🧮 3. Aptitude Math & Problem Solving (Flow Control)

**Formula Cheat Sheet:**
- **Link Utilization ($U$) for Stop-and-Wait:** $U = \frac{T_t}{T_t + 2 \times T_p}$ 
*(where $T_t$ = Transmission time, $T_p$ = Propagation delay)*

#### 📝 Practice Questions

**Q1: In Go-Back-N ARQ, if a sender transmits frames 0, 1, 2, 3, 4 and Frame 1 is lost, what happens?**
- **Solution:** The receiver gets 0 and ACKs it. When 2, 3, 4 arrive, the receiver **discards** them because it expects 1. The sender times out on Frame 1 and is forced to retransmit **Frames 1, 2, 3, and 4**.

**Q2: How does Selective Repeat handle the same scenario (Frame 1 lost out of 0, 1, 2, 3, 4)?**
- **Solution:** The receiver gets 0 and ACKs it. It **buffers** frames 2, 3, and 4. The sender times out on Frame 1 and retransmits **ONLY Frame 1**. Once Frame 1 arrives, the receiver passes 1, 2, 3, and 4 up to the OS.
