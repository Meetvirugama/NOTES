# AWS Compute (Advanced Interview Notes)

> **Target:** SDE-3 | Cloud Architect | FAANG | 20–50+ LPA

---

# 1. Amazon EC2 (Elastic Compute Cloud)
EC2 provides scalable computing capacity in the AWS Cloud. It allows you to provision virtual servers (instances) quickly.

## Instance Types
- **Compute Optimized (C-family):** High-performance processors (e.g., Gaming servers, batch processing).
- **Memory Optimized (R/X-family):** Fast performance for workloads processing large datasets in memory (e.g., Redis, SAP HANA).
- **Storage Optimized (I/D-family):** High, sequential read/write access to large datasets (e.g., NoSQL databases).

## Tenancy Models
- **Shared:** Run on shared hardware (Default).
- **Dedicated Instances:** Run on hardware dedicated to a single AWS account.
- **Dedicated Hosts:** Physical servers with EC2 instance capacity fully dedicated to you. (Required for BYOL - Bring Your Own License models).

---

# 2. AWS Lambda (Serverless Compute)
Lambda runs code without provisioning or managing servers. It charges per millisecond of compute time.

## Execution Environment
When a function is triggered, Lambda allocates an execution environment, downloads your code, and runs it.
- **Cold Start:** The time taken to initialize this environment if one isn't already available.
- **Warm Start:** If a function is invoked repeatedly, Lambda reuses the existing environment, bypassing the cold start latency.

## Best Practices
- **Global Variables:** Initialize DB connections *outside* the handler function so warm starts can reuse the connection pool.
- **Memory Allocation:** Increasing memory also proportionally increases CPU power and network bandwidth.

---

# 🚀 50 LPA Senior Engineer Deep Dive: Firecracker & Multitenancy

At FAANG, you aren't asked "What is Lambda?". You are asked **"How does AWS safely run my code on the same physical CPU as a malicious hacker's code?"**

### The Nitro Hypervisor & Firecracker MicroVMs
Historically, booting a Virtual Machine took minutes. AWS built **Firecracker**, an open-source virtualization technology written in Rust, which can boot a MicroVM in **125 milliseconds**.

When you trigger a Lambda function, AWS does NOT spin up a traditional EC2 instance or a Docker container. Docker containers share the host OS Kernel, meaning a kernel exploit (like Dirty COW) would allow a hacker to break out of the container and read memory from other AWS customers.

Instead, Firecracker uses Linux KVM (Kernel-based Virtual Machine) to spin up a completely isolated **MicroVM** for every single Lambda execution environment. 
- **Security:** Hardware-level isolation. You cannot break out.
- **Speed:** Boots fast enough to handle unpredictable serverless traffic spikes.

### The Cold Start Bottleneck
In VPCs, Lambda cold starts used to take 10+ seconds. Why? Because AWS had to attach an Elastic Network Interface (ENI) to the Lambda MicroVM so it could access your private subnets. 
AWS solved this in 2019 by creating **Hyperplane ENIs**—a shared network interface that maps to multiple execution environments, dropping VPC cold starts back down to milliseconds.

### Provisioned Concurrency
If your trading platform requires strict 50ms latency, you cannot tolerate any cold starts. You use **Provisioned Concurrency**, which forces AWS to keep `N` Firecracker MicroVMs pre-warmed and ready to execute instantly.

---

# 24. Advanced Interview Questions

### Basic
- Difference between EC2 and Lambda?
- What are EC2 purchasing options? (Spot, Reserved, On-Demand).

### Advanced (50 LPA FAANG)
- **Why doesn't AWS use standard Docker containers for Lambda?** (Kernel sharing is a multitenant security risk; Firecracker provides hardware isolation).
- **Explain how ENI attachment limits affect Lambda scaling in a VPC.**
- **If your Lambda function is CPU-bound, how do you make it faster?** (You must allocate more Memory, which implicitly unlocks more CPU shares).
