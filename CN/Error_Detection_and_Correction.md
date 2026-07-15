# 🌐 Computer Networks: Error Detection & Correction (2026 Edition)
> **Source**: Gate Smashers CN Playlist (Videos 21-24)
> **Focus**: Placement Aptitude, Software Engineering (SWE), and AI/ML Interviews

---

## 🛡️ 1. Error Detection & Correction Techniques
*Techniques used primarily at the Data Link and Transport layers to identify and fix corrupted data.*

### 📊 Error Detection Methods Comparison
| Method | How it works | Capabilities & Limitations | Common Use Case |
| :--- | :--- | :--- | :--- |
| **Parity Check** | Adds a single bit to make total 1s even (Even Parity) or odd (Odd Parity). | Detects single-bit errors. Misses burst errors (even # of flips). | Basic character transmission (e.g., ASCII). |
| **Checksum** | Divides data into 16-bit words, adds via 1s complement, appends inverted sum. | Better than parity, but fails if word order swaps or bits flip complementarily. | IPv4, UDP, and TCP Headers. |
| **CRC (Cyclic Redundancy)** | Treats data as polynomials, uses modulo-2 division by a generator, appends remainder (FCS). | Highly robust. Detects single, double, odd, and most burst errors. | Ethernet (CRC-32 FCS). |

---

## 🛠️ 2. Hamming Code (Forward Error Correction)
*Unlike Parity or Checksum, FEC techniques can both detect AND automatically correct errors on the fly.*

- **Core Mechanism:** Inserts multiple redundant parity bits at positions that are **powers of 2** (1, 2, 4, 8...). Data fills the rest (3, 5, 6, 7...).
- **Syndrome Calculation:** The receiver recalculates parity checks to form a binary **syndrome**.
- **Correction:** If syndrome is `000`, no error. If non-zero, its decimal value indicates the **exact bit position** that flipped.
- **Real-World Example:** **ECC RAM** in servers uses Hamming codes to instantly fix single-bit memory flips caused by cosmic rays or interference without system crashes.

### 📝 Practice Questions
**Q1: In an Even Parity system, if the data is `1011001`, what will the parity bit be?**
- **Solution:** The data has four 1s. To keep it even, the parity bit must be **`0`**.

**Q2: What is the main advantage of CRC over Checksum?**
- **Solution:** CRC is based on polynomial division, making it vastly superior at detecting burst errors (multiple contiguous flipped bits) compared to Checksum's simple addition.
