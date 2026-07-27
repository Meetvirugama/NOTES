# Docker: Linux Kernel Mechanics (Advanced Notes)

> **Target:** SDE-3 | DevOps Engineer | FAANG | 20–50+ LPA

---

# 1. Namespaces (The "What you can see" Barrier)
Namespaces are a feature of the Linux kernel that partitions kernel resources such that one set of processes sees one set of resources while another set of processes sees a different set of resources.

- **PID (Process ID):** Isolates the process ID number space. A process inside the container thinks it is PID 1, but on the host, it might be PID 34982.
- **NET (Network):** Isolates network interfaces, routing tables, and iptables rules. The container gets its own `eth0`.
- **MNT (Mount):** Isolates file system mount points.
- **IPC (Inter-Process Communication):** Isolates shared memory and semaphores.
- **UTS (UNIX Time-sharing System):** Isolates hostname and domain name.
- **USER:** Isolates user and group IDs. (UID 0 inside the container might map to UID 1000 on the host).

---

# 2. Cgroups (The "What you can use" Barrier)
Control Groups (cgroups) limit, account for, and isolate the resource usage (CPU, memory, disk I/O, network) of a collection of processes.

- If a container tries to use more memory than its cgroup limit (e.g., `docker run -m 500m`), the kernel's Out-Of-Memory (OOM) Killer steps in and terminates the process.

---

# 🚀 50 LPA Senior Engineer Deep Dive: The Root Privilege Vector

A classic FAANG Security/DevOps interview question: *"Why is adding a developer to the `docker` group dangerous?"*

### The Docker Daemon is Root
The Docker daemon (`dockerd`) runs as `root` because it needs permission to configure kernel namespaces, cgroups, and iptables rules.
When you type a `docker` command, the CLI talks to the daemon via a Unix socket (`/var/run/docker.sock`). 
If a user is in the `docker` group, they have write access to that socket.

### The Privilege Escalation Attack
A developer (who does not have sudo rights) can simply do this:
```bash
docker run -v /:/host -it ubuntu bash
```
This command tells the Docker daemon (which runs as root) to mount the host's entire root filesystem (`/`) into the container at `/host`.
Once inside the container, the developer is the `root` user of the container. They can now simply edit `/host/etc/shadow` or `/host/etc/sudoers` to give their standard host user account infinite passwordless sudo access. 
**Conclusion:** Having access to the Docker socket is mathematically equivalent to having passwordless root access to the physical host.

### Rootless Docker
To solve this, modern enterprises deploy **Rootless Docker**. It utilizes the `USER` namespace to map the `root` user inside the container to a standard, unprivileged user (e.g., UID 1000) on the host. If the container is compromised, the hacker only gains the rights of the unprivileged user on the host machine.

---

# 24. Advanced Interview Questions

### Basic
- What does the PID namespace do?
- What happens if a container exceeds its memory limit? (OOM Killed).

### Advanced (50 LPA FAANG)
- **Why shouldn't you run your application as the `root` user inside a Dockerfile (`USER root`)?** (If a breakout occurs, they are root on the host if User Namespaces aren't configured).
- **How does a container get its own hostname?** (Through the UTS Namespace).
- **Explain exactly how mounting the host root directory into a container allows for privilege escalation.**
