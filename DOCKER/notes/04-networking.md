# Docker: Networking Internals (Advanced Notes)

> **Target:** SDE-3 | DevOps Engineer | FAANG | 20–50+ LPA

---

# 1. Docker Network Drivers
- **Bridge (Default):** Creates a private internal network on the host. Containers can talk to each other via IP.
- **Host:** Removes network isolation. The container shares the host's networking namespace directly (faster, but port conflicts are possible).
- **None:** Completely disables all networking for the container (maximum security).
- **Overlay:** Connects multiple Docker daemons together across physical hosts (used by Docker Swarm).

---

# 🚀 50 LPA Senior Engineer Deep Dive: Iptables & Veth Pairs

When you run `docker run -p 8080:80 nginx`, what exactly happens in the Linux kernel?

### Virtual Ethernet Pairs (veth)
Because the container is in its own isolated `NET Namespace`, it cannot talk to the host's physical `eth0` network card.
Docker solves this by creating a **veth pair** (Virtual Ethernet Pair). Think of it as a virtual ethernet cable:
- One end of the cable is plugged into the container (`eth0` inside the container).
- The other end is plugged into the host's virtual Bridge network (`docker0`).

Now the container can send packets to the host. But how does external traffic reach the container?

### Iptables and NAT (Network Address Translation)
When you map a port (`-p 8080:80`), Docker does NOT spin up a proxy process to forward traffic. Instead, Docker talks directly to the Linux Kernel's firewall: **iptables**.

Docker injects a **DNAT (Destination NAT)** rule into the `PREROUTING` chain of the host's `iptables`.
1. An external request hits the physical host on port `8080`.
2. Before the packet even reaches any user-space application, the Kernel intercepts it via `iptables`.
3. The DNAT rule says: *"Ah, port 8080! Rewrite the destination IP of this packet to the Container's private IP (172.17.0.2) and change the port to 80."*
4. The Kernel routes the rewritten packet across the `docker0` bridge.
5. The packet crosses the `veth` pair and arrives at the container's `eth0`.
6. Nginx processes the request!

This kernel-level routing is why Docker port forwarding is incredibly fast and incurs almost zero CPU overhead compared to running a software proxy.

---

# 24. Advanced Interview Questions

### Basic
- What is the difference between Bridge and Host networking?
- How do two containers on a custom bridge network resolve each other's IPs? (Docker's embedded DNS server resolves container names to IPs).

### Advanced (50 LPA FAANG)
- **Explain exactly how a packet flows from the external internet into a Docker container mapped with `-p`.** (Physical NIC -> iptables DNAT PREROUTING -> docker0 bridge -> veth pair -> container eth0).
- **If you run a container with `--network host`, what happens if you try to start two Nginx containers on port 80?** (The second one crashes with 'Address already in use' because they share the physical host's network namespace).
