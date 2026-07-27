# Docker: Containers vs VMs (Advanced Interview Notes)

> **Target:** SDE-3 | Cloud Architect | FAANG | 20–50+ LPA

---

# 1. Virtual Machines
A Virtual Machine (VM) is an emulation of a physical computer system. It runs a full "guest" operating system (OS) on top of the host's hardware, managed by a Hypervisor (like VMware, VirtualBox, or AWS Nitro).

- **Pros:** Total isolation. High security. Can run Windows on Linux.
- **Cons:** Extremely heavy. A VM requires gigabytes of RAM just for the guest OS kernel before you even run your application. Booting takes minutes.

---

# 2. Containers (Docker)
Containers are a standardized unit of software that package up code and all its dependencies so the application runs quickly and reliably from one computing environment to another.

- **Pros:** Lightweight. Boots in milliseconds. Shares the Host OS Kernel.
- **Cons:** Less isolation than VMs. A Linux container cannot run on a Windows kernel natively (Docker Desktop uses a hidden Linux VM on Windows/Mac to work around this).

---

# 🚀 50 LPA Senior Engineer Deep Dive: "Containers Don't Exist"

At the junior level, people think of a Docker container as a "lightweight VM". At the 50 LPA level, you must understand that **containers do not actually exist in the Linux kernel**. 

There is no "container" object in Linux. A container is simply a standard Linux process (like running `top` or `bash`) that has been **lied to**.

### The Illusion of Isolation
Docker uses three native Linux features to trick a process into thinking it is running alone on a completely independent computer:

1. **Namespaces:** Determines *what* a process can see. (It gives the process an illusion that it has its own process tree, network interfaces, and mount points).
2. **Cgroups (Control Groups):** Determines *how much* a process can use. (It restricts the process to 500MB of RAM or 1 CPU core).
3. **chroot (Change Root):** Changes the apparent root directory for the current running process so it cannot access files outside its designated directory tree.

When you type `docker run nginx`, Docker is just making API calls to the Linux kernel to start the `nginx` process wrapped in these isolation barriers. If you log into the host machine and type `ps aux`, you will see the `nginx` process running right alongside all your normal host processes!

### The Security Implication
Because containers share the host kernel, they are vulnerable to kernel exploits. If a vulnerability like **Dirty COW (CVE-2016-5195)** allows a process to escalate privileges within the kernel, a hacker inside your Docker container can break out and take over the entire physical host machine. 
This is why AWS Lambda does not use standard Docker containers; they use Firecracker MicroVMs to provide hardware-level kernel isolation.

---

# 24. Advanced Interview Questions

### Basic
- What is the difference between a Container and a VM?
- What is a Docker Image?

### Advanced (50 LPA FAANG)
- **If I run a container with 2GB of RAM, does it reserve 2GB of RAM on the host immediately like a VM?** (No. Cgroups only set a *limit*. The container only uses what it actively allocates).
- **Can I run a Windows container natively on a Linux host?** (No. Containers share the host kernel. A Linux kernel cannot execute Windows system calls).
- **Explain the phrase: "A container is just an isolated process."**
