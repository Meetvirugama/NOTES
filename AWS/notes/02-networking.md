# AWS Networking (Advanced Interview Notes)

> **Target:** SDE-3 | Cloud Architect | FAANG | 20–50+ LPA

---

# 1. Amazon VPC (Virtual Private Cloud)
A VPC is a logically isolated section of the AWS Cloud where you launch AWS resources in a virtual network.

## Subnets
- **Public Subnet:** Has a route to the Internet Gateway (IGW).
- **Private Subnet:** Does NOT have a route to the IGW. Uses a NAT Gateway to access the internet.

## Security
- **Security Groups:** Stateful, operates at the Instance level. (If a request is allowed in, the response is automatically allowed out).
- **Network ACLs:** Stateless, operates at the Subnet level. (You must explicitly allow both inbound and outbound traffic).

---

# 2. Load Balancing (ELB)
AWS provides managed load balancers to distribute incoming traffic.

- **ALB (Application Load Balancer):** Layer 7 (HTTP/HTTPS). Routes based on paths, headers, or query strings.
- **NLB (Network Load Balancer):** Layer 4 (TCP/UDP). Handles millions of requests per second with ultra-low latency.
- **GWLB (Gateway Load Balancer):** Layer 3. Used to deploy third-party virtual firewalls inline with traffic.

---

# 🚀 50 LPA Senior Engineer Deep Dive: Transit Gateways & BGP

At scale (hundreds of AWS accounts and VPCs), peering them together creates an unmanageable `N(N-1)/2` mesh network. 

### AWS Transit Gateway
Transit Gateway acts as a central cloud router. Instead of peering VPC A to VPC B, VPC C, and VPC D, all VPCs connect to the Transit Gateway (a Hub and Spoke topology).

**Under the hood:** Transit Gateway is built on the **AWS Hyperplane** (the same distributed system that powers NLB and NAT Gateways). It is not a single EC2 instance; it is a fleet of packet-processing nodes that scale horizontally.

### BGP (Border Gateway Protocol) and Direct Connect
When linking an on-premise Corporate Data Center to AWS via **Direct Connect** (a physical fiber line), routing tables are not updated manually. 
AWS uses **BGP (Border Gateway Protocol)** over an IPSec VPN or Direct Connect. 
- The Corporate Router advertises its IP prefixes (e.g., `10.1.0.0/16`) to the AWS Transit Gateway.
- The Transit Gateway dynamically propagates these routes to all attached VPCs.
- If a fiber line is cut, BGP detects the failure and instantly re-routes traffic to a backup VPN connection using AS (Autonomous System) Path prepending.

### NAT Gateway Port Exhaustion
A highly common senior interview scenario: "Our lambdas in a private subnet are dropping packets to external APIs during peak hours. Why?"
**Answer:** A NAT Gateway maps private IPs to its single Public IP using ports. It has a limit of **55,000 concurrent connections** to a single unique destination IP. If you exceed this, you hit SNAT port exhaustion.
**Fix:** You must deploy multiple NAT Gateways in different AZs and split traffic, or assign public IPs if security permits.

---

# 24. Advanced Interview Questions

### Basic
- Difference between Security Group and NACL?
- What is an Internet Gateway?

### Advanced (50 LPA FAANG)
- **What happens if a Security Group allows port 80 Inbound, but blocks port 80 Outbound?** (Traffic flows normally because SGs are stateful. NACLs are stateless and would block it).
- **Explain SNAT Port Exhaustion and how to mitigate it in massive EKS clusters.**
- **How does AWS prevent IP Spoofing at the hardware level?** (The Nitro controller enforces Source/Destination checks before packets ever leave the physical hypervisor).
