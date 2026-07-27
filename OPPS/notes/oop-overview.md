# Object-Oriented Programming (OOP) Overview

## Definition

**Object-Oriented Programming System (OOPs)** is a programming paradigm based on the concept of **"Objects"**. These objects act as containers that bind together data in the form of fields (attributes/properties) and code in the form of procedures (methods/behaviors).

**Deep Dive:** At a theoretical level, OOP shifts the design focus from *logical operations* (verbs) to *stateful entities* (nouns). Instead of treating data as passive memory blocks passed between active functions, OOP treats data as active entities that inherently know how to operate upon themselves.

## Why It Is Needed

### The Problem with Procedural Programming
In procedural languages (like C), data and functions are completely disjointed. As a codebase scales to millions of lines, managing which function modifies which piece of global or passed data becomes a cognitive nightmare. This leads to **Spaghetti Code**—where changes in one function break seemingly unrelated parts of the system because they share access to the same exposed memory structures.

### The OOP Solution
OOP was created to enforce **Modularity and Encapsulation**.
- **State Isolation:** By forcing data to be hidden inside an object, you guarantee that only the object's own methods can mutate its state.
- **Cognitive Mapping:** Humans understand the world through entities (a Car, a Bank Account, a User). OOP allows developers to map business logic directly 1:1 with real-world nouns.

## Key Concepts (The 4 Pillars)

1. **Encapsulation:** Wrapping data and methods into a secure, single unit.
2. **Abstraction:** Exposing only high-level, essential mechanisms and hiding internal complex implementation details.
3. **Inheritance:** Establishing an "IS-A" hierarchy to reuse code and define relationships between parent and child entities.
4. **Polymorphism:** The ability of a single interface or base entity to represent multiple underlying concrete forms.

## How It Works

OOP maps real-world entities to code representation.

```mermaid
flowchart LR
    A[Real World Entity: Car] -->|Translates to| B[OOP Representation]
    B --> C[State/Data: color, speed]
    B --> D[Behavior/Code: start(), stop()]
```

## Syntax & Practical Example

Let's look at how OOP encapsulates state and behavior, and analyze it line-by-line.

### C++ Example

```cpp
#include <iostream>
using namespace std;

class BankAccount {
private:
    double balance; // 1. Encapsulated state

public:
    // 2. Constructor
    BankAccount(double initial_balance) : balance(initial_balance) {}

    // 3. Behavior operating on internal state
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            cout << "Deposited: " << amount << ". New Balance: " << balance << endl;
        }
    }
};

int main() {
    BankAccount myAcc(100.0); // 4. Object Instantiation
    myAcc.deposit(50.0);      // 5. Method Invocation
    return 0;
}
```

### Line-by-Line Breakdown:
1. `private: double balance;`: The state is hidden. The `main` function cannot execute `myAcc.balance = -1000`. This prevents illegal states.
2. `BankAccount(double...`: The constructor ensures the object cannot exist without a valid initial state.
3. `void deposit...`: The method provides a safe, validated gateway to mutate the internal state.
4. `BankAccount myAcc(100.0);`: Memory is allocated on the Stack for `myAcc`, and the constructor is immediately invoked.
5. `myAcc.deposit(50.0);`: The compiler implicitly passes a pointer to `myAcc` (the `this` pointer) into the `deposit` function so it knows which account's balance to update.

## Real-World Example

Think of a **Restaurant System**.
- **Procedural approach:** You have arrays of `order_ids`, `prices`, `table_numbers`, and massive functions like `calculate_bill(order_ids[], prices[])` that loop through parallel arrays.
- **OOP approach:** You have a `Table` object that *contains* an array of `Order` objects. You simply call `Table.calculateBill()`. The data and the logic to process that data are grouped logically.

## Edge Cases

### Edge Case 1: Breaking Encapsulation via Reflection
- **What happens:** In languages like Java, a developer can use the Reflection API (`field.setAccessible(true)`) to forcefully read or modify `private` variables, bypassing encapsulation entirely.
- **Expected Behavior:** The private state is mutated from the outside, potentially corrupting the object.
- **Best Practice:** While Reflection is useful for frameworks (like Spring or Hibernate), application developers should *never* use it to bypass access modifiers. In modern Java (Project Jigsaw), the module system restricts reflection on internals by default to prevent this.

## Advantages vs Disadvantages

| Feature | Details |
| :--- | :--- |
| **Maintainability** | Bugs are localized to specific classes rather than spread across a global procedural script. |
| **Reusability** | Code can be reused via inheritance and composition. |
| **Overhead (Con)** | OOP requires more boilerplate code. Object creation, garbage collection, and dynamic dispatch (v-tables) introduce slight runtime overhead. |
| **Rigidity (Con)** | Deep inheritance hierarchies can become incredibly fragile (e.g., changing a base class breaks 50 subclasses). |

## Tricky FAANG Interview Questions

### Question 1: Procedural vs OOP (Advanced)
**Q:** *If OOP is so great, why is the Linux Kernel written entirely in C (Procedural) and not C++ (OOP)?*
**Answer & Explanation:** OOP introduces hidden overhead. Features like Virtual Functions require Virtual Tables (v-tables), meaning every virtual function call requires a pointer dereference at runtime, destroying CPU cache locality and pipelining efficiency. Additionally, C++ compilers traditionally struggled with predictable memory layouts compared to C. In extremely low-level systems programming (like OS kernels), developers need absolute deterministic control over memory and CPU cycles, making procedural C superior.
**Why it's asked:** To see if you understand the *cost* of OOP abstractions, separating junior developers (who think OOP is a silver bullet) from seniors (who understand trade-offs).

### Question 2: Paradigm Misconceptions
**Q:** *Is C++ a purely Object-Oriented language? What about Java?*
**Answer & Explanation:** No, neither is *purely* Object-Oriented.
- **C++:** Is a multi-paradigm language. You can write global functions and variables completely outside of any class (just like C).
- **Java:** Is highly Object-Oriented (everything must be in a class), but it still has primitive types (`int`, `boolean`) which are not objects. A *pure* OOP language (like Smalltalk or Ruby) treats absolutely everything, even the number `1`, as an object.
**Why it's asked:** Tests deep understanding of language design philosophy.

## OA Tips

- In Online Assessments, if a problem involves multiple related properties (e.g., sorting intervals where each interval has a `start`, `end`, and `cost`), **always create a custom Class/Struct** to hold these values rather than using parallel arrays or confusing 2D arrays. It makes custom comparators much easier to write and debug.

## 2-Minute Revision

- **OOPs:** Paradigm shifting focus from functions to stateful entities (Objects).
- **Encapsulation:** Hides state. Protects integrity.
- **Abstraction:** Hides complexity. Simplifies interface.
- **Inheritance:** Code reuse via IS-A relationships.
- **Polymorphism:** One interface, many forms.
- **Overhead:** V-tables, dynamic dispatch, and object creation add slight performance costs compared to raw procedural code.


---
## 🚀 50 LPA Senior Engineer Deep Dive: CPU Cache & DOD

At the Staff/Senior level (L5/L6), interviewers expect you to understand the hardware implications of Object-Oriented Programming, particularly in latency-sensitive environments like High-Frequency Trading (HFT) and Game Engines.

### The True Cost of OOP: Cache Misses
Modern CPUs are incredibly fast, but RAM is extremely slow. To bridge this gap, CPUs load data into L1/L2/L3 caches in chunks called **Cache Lines** (typically 64 bytes).

In traditional OOP, you often have an **Array of Objects** (Struct of Arrays / SoA vs Array of Structs / AoS).
```cpp
class Particle {
    double x, y, z;      // 24 bytes
    double velocity;     // 8 bytes
    bool isActive;       // 1 byte (padded to 8)
    // ... total 64 bytes
};
std::vector<Particle*> particles;
```
If you loop through `particles` to update just the `velocity`, the CPU must load the entire 64-byte `Particle` object into the L1 cache, update 8 bytes, and throw the rest away. Because the objects are allocated dynamically on the heap (pointers), they are scattered across RAM. Every pointer dereference causes an **L1 Cache Miss**, forcing the CPU to wait hundreds of cycles to fetch data from RAM.

### Data-Oriented Design (DOD) as the Solution
In HFT and AAA game engines, traditional OOP is abandoned in favor of DOD (Struct of Arrays).
```cpp
struct Particles {
    std::vector<double> x, y, z;
    std::vector<double> velocity;
    std::vector<bool> isActive;
};

int main() {
    Particles p;
    // ... initialize 10 million particles ...
    
    // The CPU fetches 64 bytes of velocity at a time (8 doubles).
    // This loop hits the L1 cache 100% of the time, causing zero stalls.
    for (size_t i = 0; i < p.velocity.size(); i++) {
        p.velocity[i] *= 1.1; // Increase speed by 10%
    }
    return 0;
}
```
Now, if you update velocity, you loop over a contiguous array of `double`s. The CPU loads 8 velocities into a single 64-byte cache line simultaneously. You achieve **100% cache utilization**, running orders of magnitude faster than the OOP approach.

### 50 LPA FAANG Questions
**Q:** *Why do trading systems at Optiver or Jane Street avoid traditional OOP design patterns?*
**A:** Because OOP relies heavily on heap allocation and polymorphism (v-tables). Heap allocation scatters data, ruining CPU cache spatial locality. Virtual functions ruin branch prediction and instruction caching. HFT requires deterministic latency, achieved by contiguous stack allocation and data-oriented design to maximize L1 cache hits.
