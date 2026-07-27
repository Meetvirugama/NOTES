# AWS Databases (Advanced Interview Notes)

> **Target:** SDE-3 | Cloud Architect | FAANG | 20–50+ LPA

---

# 1. Amazon RDS (Relational Database Service)
Managed relational database service (MySQL, PostgreSQL, Oracle, SQL Server).
- **Multi-AZ:** Synchronous replication to a standby instance for high availability.
- **Read Replicas:** Asynchronous replication used to scale read traffic.

---

# 2. Amazon DynamoDB
Fully managed, serverless, NoSQL key-value database designed to run high-performance applications at any scale.

- **Partition Key:** Determines the physical server (partition) the data is stored on.
- **Sort Key:** Allows sorting of data within the same partition.

---

# 🚀 50 LPA Senior Engineer Deep Dive: Hot Partitions & Quorum Writes

### DynamoDB: The "Hot Partition" Problem
DynamoDB distributes your data across hundreds of physical SSD partitions using **Consistent Hashing**. 
The hash of your Partition Key determines the server.

If your Partition Key is `Date` (e.g., `2023-10-01`), then ALL writes for today will hit **exactly one server**. That server has a hard physical limit (1000 WCU or 3000 RCU). The server will melt down and throttle your writes (ProvisionedThroughputExceededException), even if your table has 100,000 WCU provisioned overall!

**The 50 LPA Fix: Write Sharding**
You append a random suffix to the partition key (e.g., `2023-10-01_1`, `2023-10-01_2` ... `2023-10-01_10`). This forces DynamoDB to distribute today's traffic across 10 physical partitions evenly.

### Amazon Aurora: Decoupling Compute from Storage
Standard MySQL writes data to local disk (EBS). Amazon Aurora fundamentally rewrote the database engine.
Aurora EC2 instances do NOT write data to disk. They write **Log Records** across the network to a massive, multi-tenant distributed storage fleet.

**The Quorum System:**
Aurora's storage fleet spans 3 Availability Zones. Your data is replicated 6 times (2 copies per AZ).
To achieve a successful write, Aurora requires a **4/6 Write Quorum**.
To achieve a successful read, Aurora requires a **3/6 Read Quorum**.

This means an entire AWS Availability Zone can go offline, AND another storage node can fail, and Aurora will still accept writes without losing data. 

---

# 24. Advanced Interview Questions

### Basic
- Difference between DynamoDB and RDS?
- What is an RDS Read Replica?

### Advanced (50 LPA FAANG)
- **Explain Consistent Hashing and why choosing 'UserID' is better than 'Country' for a DynamoDB Partition Key.** (Country causes Hot Partitions due to uneven traffic; UserID has high cardinality and distributes traffic evenly).
- **How does Amazon Aurora achieve faster failovers than standard RDS Multi-AZ?** (Aurora separates compute from storage. The storage fleet is shared. The read replica just promotes to primary without needing to synchronously copy disk blocks).
