# Docker: Storage & OverlayFS (Advanced Notes)

> **Target:** SDE-3 | DevOps Engineer | FAANG | 20–50+ LPA

---

# 1. Images and Layers
A Docker image is built up from a series of layers. Each instruction in a `Dockerfile` (like `RUN`, `COPY`, `ADD`) creates a new layer.
- **Read-Only:** All layers of a Docker image are read-only.
- **Caching:** If a layer hasn't changed, Docker reuses the cached layer during the next build, saving massive amounts of time.

---

# 2. Volumes and Bind Mounts
When a container is deleted, all data written inside it is lost forever. To persist data (like a database), you must use mounts.
- **Volumes:** Managed entirely by Docker (stored in `/var/lib/docker/volumes/`). Best practice for persistence.
- **Bind Mounts:** Maps a specific directory on the host machine to a directory in the container (e.g., mapping your local `./src` to `/app` for live reloading during development).

---

# 🚀 50 LPA Senior Engineer Deep Dive: OverlayFS & CoW Penalty

### Union File Systems (UFS)
How can a container write data if the Image layers are read-only?
Docker uses **OverlayFS** (a Union File System). It takes the read-only image layers and stacks a thin, invisible **Read-Write (R/W) Layer** on top of them for the running container.
The container looks at the stack and sees a single, unified filesystem.

### The Copy-on-Write (CoW) Penalty
If your container needs to modify a file that exists down in the read-only image layers, it cannot modify it directly. 
OverlayFS performs a **Copy-on-Write**:
1. It searches down through the layers to find the file.
2. It physically copies the entire file up into the Read-Write layer.
3. It modifies the copied file.

**The Catastrophic Performance Impact:**
Imagine you run a PostgreSQL database inside a container WITHOUT a Volume. The database files are stored in the Container's R/W Layer.
Every time Postgres updates a 10GB database file, OverlayFS must copy the entire 10GB file up to the R/W layer before the modification can happen. The I/O latency will completely destroy your database performance.

**The Solution:**
Volumes and Bind Mounts **bypass the Union File System entirely**. They write directly to the Host OS filesystem. This is why you MUST mount a volume for `/var/lib/postgresql/data` when running databases in Docker—it completely eliminates the CoW penalty!

---

# 24. Advanced Interview Questions

### Basic
- What is the difference between a Volume and a Bind Mount?
- Why should you put `RUN apt-get update` and `RUN apt-get install` on the same line in a Dockerfile? (To prevent layer caching issues).

### Advanced (50 LPA FAANG)
- **Explain the Copy-on-Write (CoW) strategy in OverlayFS.**
- **Why is running a database in Docker without a Volume considered a critical performance flaw?** (Because all writes hit the UFS Read-Write layer, triggering massive CoW latency overheads. Volumes bypass the UFS completely).
