# AWS Storage (Advanced Interview Notes)

> **Target:** SDE-3 | Cloud Architect | FAANG | 20–50+ LPA

---

# 1. Amazon S3 (Simple Storage Service)
S3 is an object storage service offering industry-leading scalability, data availability, security, and performance.

## Storage Classes
- **S3 Standard:** Frequent access.
- **S3 Standard-IA (Infrequent Access):** Lower storage cost, but you are charged a retrieval fee.
- **S3 Glacier Flexible Retrieval:** Archive storage. Takes minutes to hours to restore data.
- **S3 Glacier Deep Archive:** Lowest cost storage class. Takes 12+ hours to restore.

---

# 2. Amazon EBS (Elastic Block Store)
EBS provides block-level storage volumes for use with EC2 instances.
- **gp3 (General Purpose SSD):** Baseline of 3,000 IOPS regardless of volume size.
- **io2 (Provisioned IOPS SSD):** High performance for mission-critical databases (up to 256,000 IOPS).
- **st1 (Throughput Optimized HDD):** Big data, data warehouses, log processing.

---

# 🚀 50 LPA Senior Engineer Deep Dive: CAP Theorem & Prefix Limits

At the SDE-3 level, S3 is not just "a bucket". It is an immensely complex distributed system spanning multiple data centers, governed by the CAP Theorem.

### Strong Read-After-Write Consistency
Historically, S3 offered *Eventual Consistency* for overwrites/deletes. If you overwrote `image.jpg` and immediately read it, you might get the old version.
In 2020, AWS released **Strong Read-After-Write Consistency** for all operations without any performance penalty. 
**How?** AWS built a new metadata tracking system using CPU cache-coherency protocols (like MESI) adapted for distributed networks, ensuring that any subsequent read is mathematically guaranteed to hit the newly written object, sacrificing a microscopic amount of Availability to guarantee Consistency (CP in CAP theorem).

### The S3 TPS Scaling Bottleneck
If you dump millions of files into S3 like this:
`s3://my-bucket/logs/2023-10-01/log1.txt`
`s3://my-bucket/logs/2023-10-01/log2.txt`

S3 will throttle you (HTTP 503 Slow Down).
**Why?** S3 partitions its internal indexing servers based on the **Prefix** (the folder path). A single prefix can only handle **3,500 PUTs/sec** or **5,500 GETs/sec**.
Because all your logs share the exact same prefix (`/logs/2023-10-01/`), they all hit the exact same physical partition server in AWS, melting it down.

**The Solution:**
Add entropy (randomness) to the start of the prefix!
`s3://my-bucket/4f2b/logs/2023-10-01/log1.txt`
`s3://my-bucket/9a1e/logs/2023-10-01/log2.txt`

Now, AWS scales the traffic across hundreds of partitions horizontally.

---

# 24. Advanced Interview Questions

### Basic
- Difference between EBS and EFS? (EBS is block storage attached to one instance; EFS is a network file system attached to many).
- What is an S3 Multipart Upload?

### Advanced (50 LPA FAANG)
- **Why did AWS historically use Eventual Consistency for S3 overwrites, and how does the CAP Theorem apply?**
- **You need to ingest 20,000 images per second into S3. How do you design the bucket structure to prevent throttling?** (Prefix entropy hashing).
