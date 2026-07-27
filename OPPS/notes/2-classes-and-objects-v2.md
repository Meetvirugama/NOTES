# Classes and Objects — Industry-Level Interview Notes

> **Target:** SDE-1 → SDE-3 | Google · Goldman Sachs · Microsoft · Amazon · Meta · Apple · Uber · LinkedIn  
> **Level:** Production-grade, interview-ready, zero college fluff

---

## Table of Contents
1. [Definition](#1-definition)
2. [Why It Exists](#2-why-it-exists)
3. [Internal Working](#3-internal-working)
4. [Syntax](#4-syntax)
5. [Visual Diagrams](#5-visual-diagrams)
6. [Real World Analogy](#6-real-world-analogy)
7. [Interview Explanation](#7-interview-explanation)
8. [Interview Follow-up Questions](#8-interview-follow-up-questions)
9. [Coding Examples](#9-coding-examples)
10. [Common Mistakes](#10-common-mistakes)
11. [Best Practices](#11-best-practices)
12. [Complexity](#12-complexity)
13. [Advantages](#13-advantages)
14. [Disadvantages](#14-disadvantages)
15. [Comparison Table](#15-comparison-table)
16. [Design Pattern Connection](#16-design-pattern-connection)
17. [System Design Connection](#17-system-design-connection)
18. [Multithreading Connection](#18-multithreading-connection)
19. [Company Interview Perspective](#19-company-interview-perspective)
20. [Tricky Interview Questions](#20-tricky-interview-questions)
21. [Coding Problems](#21-coding-problems)
22. [Revision Sheet](#22-revision-sheet)
23. [Flashcards](#23-flashcards)
24. [Cheat Sheet](#24-cheat-sheet)
25. [Final Interview Summary](#25-final-interview-summary)

---

## 1. Definition

### Simply
A **class** is a blueprint. An **object** is an instance of that blueprint brought to life in memory.

### Technically
- **Class**: A user-defined type that encapsulates **data (fields/attributes)** and **behavior (methods)**. Compiled into type metadata. Exists at compile time.
- **Object**: A runtime entity — a concrete block of memory allocated on the **heap** (usually) that conforms to the class blueprint. Has its own **state** independent of other objects.

### From an Interviewer's Perspective
> "When I ask 'what's a class vs object?' I'm really probing if you understand **type vs instance**, **compile time vs runtime**, **stack vs heap**, and **memory layout**. A senior engineer explains vtable pointers, object size, and identity vs equality."

⭐ **Key distinction**: Class = template at compile time. Object = allocated memory at runtime.

---

## 2. Why It Exists

### Problem It Solves
Before OOP (C, Assembly), code was a sea of global variables and functions. There was **no grouping** of related data + logic.

**Problems with procedural code:**
- Data and functions are disconnected
- No access control (anyone can modify any variable)
- No reusability primitives
- Spaghetti code at scale (impossible to maintain 1M+ LOC)

### What Happens Without Classes
```c
// Procedural C: data + functions live separately
int account_balance = 0;
char account_owner[50] = "Alice";

void deposit(int amount) { account_balance += amount; }
void withdraw(int amount) { account_balance -= amount; } // No protection!
```
Anyone can set `account_balance = -999999`. No protection. No grouping.

### Real Software Examples
- **Java's `String` class**: Groups character array + length + 40+ methods
- **Python's `dict`**: Groups hash table internals + `get()`, `put()`, `keys()`
- **Linux kernel** (C): Simulates OOP via `struct` + function pointers (proving the concept predates keywords)
- **Android `Activity` class**: Manages UI state + lifecycle callbacks as one unit

---

## 3. Internal Working

### Memory Layout

When you create an object, the runtime allocates a **contiguous block of memory** on the heap. The layout is determined at compile time.

```
Object in Heap Memory (C++ BankAccount):
+-----------------------------+
|  vptr (8 bytes)             |  <- Points to vtable (if virtual methods)
+-----------------------------+
|  balance (8 bytes, double)  |
+-----------------------------+
|  accountId (4 bytes, int)   |
+-----------------------------+
|  [3 bytes padding]          |  <- Compiler alignment
+-----------------------------+
Total size: 24 bytes
```

### Stack vs Heap

| Aspect | Stack | Heap |
|--------|-------|------|
| Who allocates? | Compiler (automatic) | Developer / GC |
| Speed | Very fast (stack pointer move) | Slower (malloc/new) |
| Lifetime | Scope-bound | Manual / GC-managed |
| Size limit | ~1-8 MB typically | GBs |
| Thread safety | Per-thread (safe) | Shared (not safe) |

```
Stack Frame (main):               Heap:
+------------------+              +------------------+
| local int x = 5  |              | BankAccount obj  |
| BankAccount* ptr |---pointer--->| balance: 1000.0  |
+------------------+              | accountId: 42    |
                                  +------------------+
```

### Object Lifecycle

```
1. Class Loading     -> JVM reads bytecode; C++ compiler parses header
2. Memory Allocation -> heap.allocate(sizeof(MyClass))
3. Constructor Call  -> initializes fields, calls super()
4. Object in Use     -> methods called, state changes
5. Destructor/GC     -> memory reclaimed
```

### JVM Internals (Java)
```
Source: BankAccount.java
         | javac
Bytecode: BankAccount.class
         | ClassLoader
JVM Method Area: class metadata
         | new BankAccount()
JVM Heap: object instance
         | GC (when no references)
Memory freed
```

**Object Header in JVM (HotSpot, 64-bit):**
```
+------------------+
| Mark Word (8B)   |  <- GC age, hash code, lock state
+------------------+
| Class Pointer(8B)|  <- Points to class metadata
+------------------+
| Fields...        |
+------------------+
```
⭐ **Interview Gold**: Every Java object has 12-16 bytes of overhead from the object header alone — critical for memory optimization at Google/Amazon scale.

### Virtual Table (vtable)
```
BankAccount vtable:
+-------------------------+
| BankAccount::deposit()  | <- address
| BankAccount::withdraw() | <- address
| BankAccount::toString() | <- address (overriding Object)
+-------------------------+
      ^
  vptr inside every BankAccount object
```

### Method Dispatch
- **Static Binding** (compile-time): `private`, `static`, `final` methods — compiler bakes in the address directly
- **Dynamic Binding** (runtime): `virtual` methods (C++) or non-final instance methods (Java) — lookup via vtable at runtime

---

## 4. Syntax

### C++
```cpp
#include <iostream>
#include <string>
using namespace std;

class BankAccount {
private:
    double balance;
    string owner;

public:
    // Constructor
    BankAccount(string owner, double initialBalance)
        : owner(owner), balance(initialBalance) {}

    // Methods
    void deposit(double amount) {
        if (amount <= 0) throw invalid_argument("Amount must be positive");
        balance += amount;
    }

    double getBalance() const { return balance; }

    // Destructor
    ~BankAccount() {
        cout << "Account for " << owner << " closed." << endl;
    }
};

int main() {
    BankAccount acc("Alice", 1000.0); // Stack allocated
    BankAccount* accPtr = new BankAccount("Bob", 500.0); // Heap allocated

    acc.deposit(250.0);
    accPtr->deposit(100.0);

    delete accPtr; // Manual cleanup
    return 0;
} // acc automatically destroyed here
```

### Java
```java
public class BankAccount {
    private double balance;
    private String owner;

    // Constructor
    public BankAccount(String owner, double initialBalance) {
        this.owner = owner;
        this.balance = initialBalance;
    }

    // Methods
    public void deposit(double amount) {
        if (amount <= 0) throw new IllegalArgumentException("Amount must be positive");
        this.balance += amount;
    }

    public double getBalance() { return balance; }

    @Override
    public String toString() {
        return String.format("BankAccount[owner=%s, balance=%.2f]", owner, balance);
    }

    public static void main(String[] args) {
        BankAccount acc = new BankAccount("Alice", 1000.0); // Always heap in Java
        acc.deposit(250.0);
        System.out.println(acc); // Calls toString()
    } // GC handles cleanup
}
```

### Python
```python
class BankAccount:
    # Class variable (shared across all instances)
    interest_rate = 0.05

    def __init__(self, owner: str, initial_balance: float):
        # Instance variables (unique per object)
        self.owner = owner
        self._balance = initial_balance  # _ = convention for "protected"

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    @property
    def balance(self) -> float:
        return self._balance

    def __repr__(self) -> str:
        return f"BankAccount(owner='{self.owner}', balance={self._balance:.2f})"

# Usage
acc = BankAccount("Alice", 1000.0)  # __init__ called
acc.deposit(250.0)
print(acc)  # __repr__ called
print(type(acc))   # <class '__main__.BankAccount'>
print(id(acc))     # Memory address
```

### Key Differences

| Feature | C++ | Java | Python |
|---------|-----|------|--------|
| Memory mgmt | Manual (new/delete) | GC (automatic) | GC (ref counting + cyclic GC) |
| Stack objects | Yes | No (always heap) | No |
| Object header | Minimal (just vptr if virtual) | 12-16 bytes | 28+ bytes (ref count, type ptr) |
| Class variables | `static` members | `static` fields | Class-level attributes |
| Private enforcement | Compile-time hard | Compile-time hard | Convention only (`_`) |
| `this` keyword | Pointer (this->) | Reference (this.) | Explicit (self) |

---

## 5. Visual Diagrams

### Class Blueprint vs Object Instances
```
CLASS (Blueprint - compile time)
+----------------------------------+
|  BankAccount                     |
+----------------------------------+
|  - balance: double               |
|  - owner: string                 |
+----------------------------------+
|  + deposit(amount): void         |
|  + withdraw(amount): void        |
|  + getBalance(): double          |
+----------------------------------+
           |                |
    (instance)          (instance)
           v                v
OBJECT 1 (runtime)    OBJECT 2 (runtime)
+-----------------+  +-----------------+
| balance: 1000.0 |  | balance: 500.0  |
| owner: "Alice"  |  | owner: "Bob"    |
+-----------------+  +-----------------+
```

### Heap and Stack Relationship
```
STACK                          HEAP
+----------------+            +------------------+
| main()         |            |                  |
| +------------+ |            |  +------------+  |
| | acc1 ptr --+-+------------+->| BankAcct   |  |
| |            | |            |  | bal: 1000  |  |
| | acc2 ptr --+-+------------+->| BankAcct   |  |
| +------------+ |            |  | bal: 500   |  |
+----------------+            |  +------------+  |
                              +------------------+
```

### Object Identity vs Equality
```
                  HEAP
a -------------> [BankAccount: balance=1000]
b -------------> [BankAccount: balance=1000]
c -------------> (same object as a) ^

a == b  -> false (different identity, equal value if equals() defined)
a == c  -> true  (same reference, same identity)
a.equals(b) -> true (if equals() compares balance)
```

### vtable Diagram
```
OBJECT (on heap)           vtable (in code segment)
+---------------+          +---------------------------+
| vptr          |--------->| BankAccount::deposit()    |
+---------------+          | BankAccount::withdraw()   |
| balance: 1000 |          | Object::toString()        |
+---------------+          +---------------------------+
| owner: "Alice"|
+---------------+
```

---

## 6. Real World Analogy

### Car Manufacturing (Class = Blueprint, Object = Physical Car)
```
Blueprint (Class):           Car (Object):
+-------------------+       +-------------------+
| Toyota Camry Spec |       | VIN: ABC-123       |
| - Color: ?        |  ->   | Color: Red         |
| - Engine: 2.5L    |       | Engine: 2.5L       |
| - start()         |       | Mileage: 45,230    |
| - accelerate()    |       | [starts, drives]   |
+-------------------+       +-------------------+
    One blueprint             Millions of instances
```

### Hospital
- **Class**: `PatientRecord` defines fields (name, age, blood type, diagnosis) and behaviors (admit, discharge, prescribe)
- **Object**: "John Doe, 34, A+, diabetes, admitted 2024-01-15" — a specific patient

### E-Commerce (Amazon)
- **Class**: `Product` (id, name, price, category, inventory_count, addToCart, review)
- **Objects**: iPhone 15, AirPods Pro, Kindle — each an independent instance with its own inventory count

### Bank (Goldman Sachs)
- **Class**: `TradeOrder` (symbol, quantity, price, timestamp, execute, cancel, modify)
- **Objects**: 10M trade orders created and destroyed per trading day

### Operating System
- **Class**: `Process` (pid, memory pages, file descriptors, CPU registers, state)
- **Objects**: Every running program (Chrome, VS Code, Terminal) is a `Process` object

### Game Development
- **Class**: `GameObject` (position, velocity, health, render(), update(), collide())
- **Objects**: Every bullet, enemy, wall, player in the game

### Company Organization
- **Class**: `Employee` (id, name, salary, department, promote(), terminate(), getBonus())
- **Objects**: Each individual employee in the company database

---

## 7. Interview Explanation

### 30 Seconds
> "A class is a compile-time blueprint defining data and behavior. An object is a runtime instance with actual memory allocated, holding its own independent state. Every object shares the class's method code but has its own copy of instance variables."

### 1 Minute
> "Think of a class like a cookie cutter and objects as the cookies. The class defines structure — fields for state, methods for behavior — but it doesn't exist in memory at runtime by itself. When you call `new BankAccount()`, the runtime allocates heap memory sized by the class definition, runs the constructor to initialize fields, and hands you a reference. Multiple references can point to the same object — which is the aliasing problem you need to watch for in multithreaded code."

### 3 Minutes
> "At the language level, a class is a type definition: field names and types, method signatures, access control, and inheritance. The compiler uses this to determine memory layout — field offsets, total size, alignment requirements, and whether to embed a vtable pointer for virtual dispatch.

> At runtime, when you instantiate with `new`, the allocator finds a contiguous heap block of the right size. In Java, every object gets a 12-byte header: an 8-byte mark word holding GC age, hash code, and lock state, and a 4-byte class pointer referencing the method area. Then come your fields. In C++, the layout is leaner — just a vptr if virtual methods exist, then fields, with alignment padding.

> The object's methods live in code memory — not inside the object itself. All instances share the same method bytecode. What distinguishes instances is their field values. This is why an object reference is just a pointer — the methods are looked up via the class pointer or vtable, not stored redundantly per object.

> For senior-level interviews, I'd highlight identity vs equality: `==` compares references (identity) in Java; `.equals()` compares semantic content. And always override `hashCode` when overriding `equals` — the contract is: if `a.equals(b)`, then `a.hashCode() == b.hashCode()`."

---

## 8. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What is the difference between a class and an object? | Class is blueprint (compile-time); object is instance (runtime) with memory |
| Where are objects stored? | Heap (Java, Python always; C++ by default with `new`) |
| Can you have a class without an object? | Yes — static methods/fields can be accessed via class name |
| What is `this`? | Reference to the current object instance |
| What is the default constructor? | No-arg constructor auto-generated by compiler if none is defined |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| What is object identity vs equality? | Identity = same reference (`==`); Equality = same semantic value (`.equals()`) |
| How much memory does a Java object take? | Minimum 16 bytes (12-byte header + 4 bytes padding); varies by fields |
| What is shallow copy vs deep copy? | Shallow = copy references; Deep = recursively copy all referenced objects |
| Why are strings immutable in Java? | Safety (shared constant pool, thread-safe, caching hashCode) |
| What happens if you don't call `delete` in C++? | Memory leak — heap memory never reclaimed |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| Explain object layout with virtual functions in C++ | Each polymorphic object gets hidden vptr (8 bytes on 64-bit) pointing to class vtable |
| How does the JVM represent an object internally? | Object header (mark word + class pointer) + instance fields. Methods in Method Area |
| What is object interning? | Reusing identical immutable objects from a pool (String pool, Integer cache -128..127) |
| What is the difference between `==` and `.equals()` for Strings? | `==` compares references; `.equals()` compares char content |
| Explain memory alignment and padding | CPU accesses memory at aligned boundaries; compiler adds padding bytes for performance |

### Expert
| Question | Expected Answer |
|----------|-----------------|
| How does HotSpot JVM compress object pointers? | Compressed OOPs: on <32GB heaps, 4-byte class pointers with 3-bit shift trick |
| What is false sharing in object field layout? | Two threads accessing different fields of same cache line causes cache invalidation |
| How would you design a zero-copy object pool? | Pre-allocate fixed array, track free list with indices, avoid allocation in hot path |
| Explain the difference between value types and reference types | Value: stored inline on stack/embedded; Reference: heap-allocated, pointer held |

### 💼 Google Level
> *"Design a memory-efficient class to represent a graph node where we have 100M nodes in memory. What fields do you choose and what memory layout decisions do you make?"*
- Use int32 vs int64 for IDs, array-of-arrays vs adjacency list, cache line packing, primitive arrays vs Object arrays, off-heap memory (ByteBuffer)

### 💼 Goldman Sachs Level
> *"In a high-frequency trading system processing 10M orders/sec, what are the object allocation concerns?"*
- GC pressure, object pooling, flyweight pattern, value types (Java records / C++ structs), off-heap allocation, avoiding boxing

---

## 9. Coding Examples

### Basic Example
```java
public class Circle {
    private final double radius;

    public Circle(double radius) {
        if (radius <= 0) throw new IllegalArgumentException("Radius must be positive");
        this.radius = radius;
    }

    public double area() { return Math.PI * radius * radius; }
    public double perimeter() { return 2 * Math.PI * radius; }

    @Override
    public String toString() { return String.format("Circle(r=%.2f)", radius); }
}
```

### Intermediate — Value Object Pattern
```java
// Immutable value object (used heavily in financial systems)
public final class Money {
    private final long amountInCents; // Avoid floating point for money!
    private final Currency currency;

    public Money(long amountInCents, Currency currency) {
        this.amountInCents = amountInCents;
        this.currency = Objects.requireNonNull(currency);
    }

    public Money add(Money other) {
        if (!this.currency.equals(other.currency))
            throw new IllegalArgumentException("Currency mismatch");
        return new Money(this.amountInCents + other.amountInCents, this.currency);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Money m)) return false;
        return amountInCents == m.amountInCents && currency.equals(m.currency);
    }

    @Override
    public int hashCode() { return Objects.hash(amountInCents, currency); }
}
```

### Advanced — Object Pool Pattern
```java
// Production: avoid GC pressure in high-throughput systems
public class ObjectPool<T> {
    private final Queue<T> pool;
    private final Supplier<T> factory;
    private final int maxSize;

    public ObjectPool(Supplier<T> factory, int maxSize) {
        this.factory = factory;
        this.maxSize = maxSize;
        this.pool = new ArrayDeque<>(maxSize);
        for (int i = 0; i < maxSize; i++) pool.offer(factory.get());
    }

    public T borrow() {
        T obj = pool.poll();
        return obj != null ? obj : factory.get();
    }

    public void release(T obj) {
        if (pool.size() < maxSize) pool.offer(obj);
    }
}

// Usage in HFT system
ObjectPool<TradeOrder> orderPool = new ObjectPool<>(TradeOrder::new, 10_000);
TradeOrder order = orderPool.borrow();
order.fill("AAPL", 100, 150.50);
processOrder(order);
orderPool.release(order); // Reuse, avoid GC
```

### Production-Level — Entity Base Class
```java
@MappedSuperclass
public abstract class BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @CreatedDate
    private Instant createdAt;

    @LastModifiedDate
    private Instant updatedAt;

    @Version // Optimistic locking
    private Long version;

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BaseEntity that = (BaseEntity) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() { return Objects.hashCode(id); }
}
```

### Interview Coding Example — MinStack O(1) getMin
```java
class MinStack {
    private final Deque<int[]> stack; // [value, currentMin]

    public MinStack() { stack = new ArrayDeque<>(); }

    public void push(int val) {
        int min = stack.isEmpty() ? val : Math.min(val, stack.peek()[1]);
        stack.push(new int[]{val, min});
    }

    public void pop() { stack.pop(); }
    public int top() { return stack.peek()[0]; }
    public int getMin() { return stack.peek()[1]; }
}
```

---

## 10. Common Mistakes

### ⚠️ Mistake 1: Comparing Objects with `==` in Java
```java
String a = new String("hello");
String b = new String("hello");
System.out.println(a == b);       // FALSE — different references
System.out.println(a.equals(b));  // TRUE — same content
```

### ⚠️ Mistake 2: Mutable Public Fields
```java
class Config {
    public List<String> allowedHosts = new ArrayList<>(); // PUBLIC MUTABLE!
}
// Thread 1 adds to it, Thread 2 reads — race condition
// Fix: Return unmodifiable view
public List<String> getAllowedHosts() {
    return Collections.unmodifiableList(allowedHosts);
}
```

### ⚠️ Mistake 3: Memory Leak via Static Collections
```java
class Cache {
    private static final Map<String, Object> cache = new HashMap<>();
    // Nobody ever removes! Memory grows forever.
}
// Fix: Use WeakHashMap or an eviction policy (Caffeine)
```

### ⚠️ Mistake 4: Forgetting `delete` in C++ (Memory Leak)
```cpp
// BAD
void process() {
    MyObject* obj = new MyObject();
    if (someCondition) return; // LEAK!
    delete obj;
}
// GOOD: Use RAII smart pointers
void process() {
    auto obj = std::make_unique<MyObject>(); // Auto-deleted on scope exit
    if (someCondition) return; // Safe
}
```

### ⚠️ Mistake 5: Overriding `equals` without `hashCode`
```java
// Violates contract: if a.equals(b) then a.hashCode() == b.hashCode()
class User {
    String email;
    @Override
    public boolean equals(Object o) { ... } // Overridden
    // hashCode NOT overridden — HashMap/HashSet will BREAK
}
```

---

## 11. Best Practices

### Naming
- Classes: `PascalCase` — `BankAccount`, `UserService`, `PaymentProcessor`
- Objects/variables: `camelCase` — `bankAccount`, `currentUser`
- Constants: `UPPER_SNAKE_CASE` — `MAX_RETRY_COUNT`

### Design
- **Single Responsibility**: One class, one reason to change
- **Immutability**: Especially for value objects (`Money`, `UserId`)
- **Favor composition over inheritance**
- **Program to interfaces**: `List<T> list = new ArrayList<>()` not `ArrayList<T>`

### 🚀 Performance
- Avoid unnecessary object creation in hot paths (use object pools)
- Use primitive arrays over `Object[]` arrays when possible
- `int[]` is 4x more cache-efficient than `Integer[]` (no pointer chasing)
- Consider `record` types in Java 16+ for immutable data classes (zero boilerplate)

### Maintainability
- Always override `toString()` for debugging
- Always override `equals()` and `hashCode()` together
- Use `final` fields where possible
- Validate in constructors — fail fast

---

## 12. Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| Object creation (`new`) | O(1) amortized | Heap allocation + constructor |
| Field access | O(1) | Direct memory offset |
| Virtual method call | O(1) | One vtable indirection |
| GC collection (Java) | O(n) live objects | Stop-the-world pauses |
| `equals()` comparison | O(n) worst case | Depends on implementation |

### Memory Overhead per Object
| Language | Object overhead |
|----------|----------------|
| C++ (no virtuals) | 0 extra bytes |
| C++ (with virtuals) | 8 bytes (vptr) |
| Java | 12-16 bytes (object header) |
| Python | 28+ bytes (ob_refcnt, ob_type, etc.) |

---

## 13. Advantages

| Advantage | Example |
|-----------|---------|
| **Modularity** | Each class is self-contained, independently testable |
| **Reusability** | `BankAccount` reused across savings, checking, loan accounts |
| **Maintainability** | Change `deposit()` once, all code benefits |
| **Encapsulation** | Balance can't be directly mutated |
| **Domain modeling** | Code mirrors business domain (DDD) |
| **Testability** | Mock objects replace real dependencies in unit tests |

---

## 14. Disadvantages

| Disadvantage | When it hurts |
|-------------|---------------|
| **Memory overhead** | 100M Java objects = 1.6GB+ just in headers |
| **GC pressure** | High allocation rate causes GC pauses |
| **Indirection cost** | Virtual dispatch hurts cache performance |
| **Overkill for simple data** | A 3-field DTO doesn't need a complex hierarchy |
| **Complex debugging** | State spread across heap, hard to trace |

🚀 **Production insight**: In ultra-high-performance systems (HFT, game engines), engineers bypass OOP and use flat arrays of primitives (Data-Oriented Design / ECS pattern).

---

## 15. Comparison Table

### Object vs Class
| Aspect | Class | Object |
|--------|-------|--------|
| Exists at | Compile time | Runtime |
| Stored in | Method Area (JVM) | Heap |
| Count | One per type | Multiple instances |
| Has state? | Only static fields | Yes (instance fields) |

### Stack Object vs Heap Object (C++)
| Aspect | Stack | Heap |
|--------|-------|------|
| Declaration | `BankAccount acc;` | `BankAccount* acc = new BankAccount();` |
| Lifetime | Scope-limited | Manual / smart pointer |
| Speed | Faster | Slower |
| Risk | Stack overflow | Memory leak |

### Reference vs Value Semantics
| Aspect | Reference (Java default) | Value (C++ stack) |
|--------|--------------------------|-------------------|
| Copy behavior | Copies reference (alias) | Copies data |
| Mutation sharing | Yes | No |
| Memory | Heap | Stack/inline |

---

## 16. Design Pattern Connection

| Pattern | How Classes/Objects are Used |
|---------|------------------------------|
| **Singleton** | Class ensures only one object instance exists globally |
| **Factory** | Factory class abstracts which concrete class to instantiate |
| **Builder** | Builder class assembles a complex object step-by-step |
| **Prototype** | Object clones itself (`clone()`) |
| **Flyweight** | Share one object across many contexts to save memory |
| **Object Pool** | Pre-create and reuse objects to avoid allocation overhead |
| **Value Object** | Immutable objects representing domain values (Money, Address) |

---

## 17. System Design Connection

### Large Backend Systems
- Every **REST request** maps to a DTO (Data Transfer Object) class
- **Entity classes** map to database rows via ORM (Hibernate, SQLAlchemy)
- **Service classes** hold business logic (stateless singleton beans in Spring)

### Microservices
- **Event classes**: Messages published to Kafka are serialized objects
- **Client classes**: gRPC stubs, REST clients wrapped in typed classes

### Distributed Systems
- **Distributed caching**: Objects must be serializable
- **Protocol Buffers**: Generates typed classes from `.proto` schemas

### Cloud
- **AWS SDK**: `S3Client`, `DynamoDbClient` are typed class wrappers
- **CDK**: Cloud resources represented as classes

---

## 18. Multithreading Connection

### Thread Safety of Objects
```
Shared Object (Heap)
        ^
Thread1 reads balance   <- RACE CONDITION if no sync
Thread2 writes balance  <-
```

### Making Objects Thread-Safe
```java
// Option 1: Synchronized method (coarse-grained)
public synchronized void deposit(double amount) { balance += amount; }

// Option 2: Atomic fields (fine-grained, lock-free)
private final AtomicLong balance = new AtomicLong(0);
public void deposit(long amount) { balance.addAndGet(amount); }

// Option 3: Immutable object (no synchronization needed!)
public final class Money {
    private final long amount; // final = immutable = thread-safe
    public Money add(Money other) { return new Money(this.amount + other.amount); }
}
```

### Key Rules
- ⭐ **Immutable objects are always thread-safe**
- ⭐ **Stateless service classes (no fields) are always thread-safe**
- ⚠️ **Shared mutable state requires explicit synchronization**

---

## 19. Company Interview Perspective

### Google
- Focus on **memory layout** and **GC behavior** for large-scale systems
- Expect questions about **object identity vs equality** in distributed caches
- May ask to design classes for a specific system component (rate limiter, cache)

### Goldman Sachs
- Heavy focus on **object lifecycle** in high-throughput trading systems
- Object pooling, avoiding GC pauses, primitive vs boxed types
- Value objects for financial amounts (immutable, no floating point)

### Microsoft
- C++ memory management: smart pointers, RAII, copy vs move semantics
- COM-style interface design (pure virtual classes)
- .NET CLR: value types vs reference types

### Amazon
- Class design for **scalable services** (stateless service classes, serializable entities)
- **OOP applied to system design**: "Design the class hierarchy for a warehouse system"

### Meta
- Python and C++ — Python object model, metaclasses, descriptors
- Low-level memory in C++: custom allocators, arena allocation
- Performance at Facebook scale

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Can a class be an object in Java? | Yes — `Class` objects (metadata) are objects themselves |
| 2 | Is `null` an object? | No — it is the absence of a reference |
| 3 | Can you call an instance method on a `null` reference? | Throws `NullPointerException` |
| 4 | What's the size of an empty class in C++? | 1 byte — C++ mandates unique addresses for distinct objects |
| 5 | In Java, where is `String "hello"` stored? | String Pool in Heap (modern JDK) |
| 6 | Can two objects have the same `hashCode`? | Yes — hash collision is allowed |
| 7 | Can an object be GC'd while its method runs? | No — the method call stack holds a strong reference |
| 8 | `clone()` vs copy constructor? | `clone()` = Object mechanism + casts; copy constructor = explicit, type-safe |
| 9 | Why is `hashCode` not unique? | Infinite objects, finite hash space — pigeonhole principle |
| 10 | What is an anonymous object? | `new BankAccount().deposit(100)` — no reference, immediately GC eligible |
| 11 | ⚠️ `equals()` true but `hashCode()` different? | HashMap/HashSet breaks — contract violation |
| 12 | Virtual method from C++ constructor? | Calls base class version — derived class not yet constructed |
| 13 | What is object slicing in C++? | Assigning derived to base by value — loses derived state |
| 14 | How does Python simulate private? | Name mangling: `__field` becomes `_ClassName__field` |
| 15 | ⚠️ Does Java pass objects by reference? | No — passes the **reference by value** |
| 16 | What is a phantom reference? | Enqueued after finalization; used for cleanup without resurrection |
| 17 | `Integer.valueOf(127) == Integer.valueOf(127)` true? | Yes — Integer cache. `128` would be false |
| 18 | How does Python know `a.method()` receives `a` as `self`? | Descriptor protocol: bound method wraps `method + a` |
| 19 | `delete` a pointer twice in C++? | Undefined behavior — heap corruption, segfault |
| 20 | What is a POD type? | Trivially copyable C++ struct — can `memcpy`, no vtable, no custom constructor |

---

## 21. Coding Problems

### Easy — Counter Class
```java
class Counter {
    private int count = 0;
    public void increment() { count++; }
    public void decrement() { count--; }
    public void reset() { count = 0; }
    public int getCount() { return count; }
}
```

### Medium — LRU Cache (LeetCode #146)
```java
class LRUCache {
    private final int capacity;
    private final LinkedHashMap<Integer, Integer> cache;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.cache = new LinkedHashMap<>(capacity, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
                return size() > capacity;
            }
        };
    }

    public int get(int key) { return cache.getOrDefault(key, -1); }
    public void put(int key, int value) { cache.put(key, value); }
}
```

### Hard — Thread-Safe BoundedBlockingQueue (LeetCode #1188)
```java
class BoundedBlockingQueue {
    private final Queue<Integer> queue;
    private final int capacity;
    private final Object lock = new Object();

    public BoundedBlockingQueue(int capacity) {
        this.capacity = capacity;
        this.queue = new LinkedList<>();
    }

    public void enqueue(int element) throws InterruptedException {
        synchronized (lock) {
            while (queue.size() == capacity) lock.wait();
            queue.offer(element);
            lock.notifyAll();
        }
    }

    public int dequeue() throws InterruptedException {
        synchronized (lock) {
            while (queue.isEmpty()) lock.wait();
            int val = queue.poll();
            lock.notifyAll();
            return val;
        }
    }

    public int size() { synchronized (lock) { return queue.size(); } }
}
```

---

## 22. Revision Sheet

| Concept | Key Point |
|---------|-----------|
| Class | Compile-time blueprint: fields + methods + access control |
| Object | Runtime instance: heap memory with actual field values |
| Constructor | Initializes object state; runs after memory allocation |
| `this` | Reference to current object |
| Stack vs Heap | Stack = local/auto (fast, limited); Heap = objects (slow, large) |
| GC | Java/Python: automatic; C++: manual (smart pointers) |
| Object header | Java: 12-16 bytes overhead per object |
| `equals/hashCode` | Always override together; if equal, same hashCode |
| Static members | Belong to class, not instance |
| Immutability | Makes objects thread-safe automatically |

### Things Interviewer Expects
- Explain object memory layout
- Distinguish stack vs heap
- Know equals/hashCode contract
- Understand static vs instance
- Apply single responsibility principle
- Handle null safely
- Use immutability appropriately

### Common Pitfalls
- ⚠️ `==` vs `.equals()` for objects
- ⚠️ Overriding `equals` without `hashCode`
- ⚠️ Mutable public fields
- ⚠️ Not validating in constructors
- ⚠️ Object slicing in C++
- ⚠️ Memory leaks in C++ (not using RAII)

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| Class vs Object | Class = blueprint (compile); Object = instance (runtime) |
| Where are Java objects stored? | Heap (always) |
| What is `this`? | Reference to current object instance |
| Java object header size? | 12-16 bytes (mark word + class pointer) |
| `==` vs `.equals()` in Java | `==` compares references; `.equals()` compares values |
| What is a constructor? | Special method to initialize object state |
| What is object slicing? | C++ assigning derived to base by value loses derived fields |
| What is RAII? | C++ idiom: Resource lifetime tied to object scope |
| What makes an object thread-safe? | Immutability, synchronization, or atomic operations |
| What is an anonymous object? | Object created without a reference variable |
| What is a value object? | Immutable object defined by its values, not identity |
| Why are immutable objects thread-safe? | No mutation = no race conditions |
| What is Integer cache in Java? | -128 to 127 Integer objects are pooled |
| What is `hashCode` contract? | If `a.equals(b)` then `a.hashCode() == b.hashCode()` |
| What is object interning? | Reusing existing identical immutable objects (String pool) |
| What is a copy constructor? | Constructor creating a new object as copy of existing |
| What is shallow vs deep copy? | Shallow: copy references; Deep: recursively copy everything |
| What is GC in Java? | Automatic heap memory reclamation for unreachable objects |
| Python object overhead? | 28+ bytes (ref count + type pointer + dict pointer) |
| What is `__slots__` in Python? | Replace `__dict__` with fixed array — reduces memory per object |
| What is `final` class in Java? | Cannot be subclassed (String, Integer are final) |
| What is `record` in Java 16+? | Immutable data class with auto-generated equals/hashCode/toString |
| What is a POD type in C++? | Trivially copyable struct — can be memcpy'd safely |
| What is object pooling? | Pre-creating and reusing objects to reduce allocation overhead |
| Phantom reference? | Weakest reference; enqueued after finalization for cleanup |

---

## 24. Cheat Sheet

### Top 20 Facts
1. Class = compile-time type; Object = runtime instance
2. Java always allocates objects on the heap
3. Java object header = 12-16 bytes overhead
4. C++ empty class = 1 byte minimum
5. Virtual methods add 8-byte vptr to object
6. `==` = reference identity; `.equals()` = semantic equality
7. Always override `hashCode` when overriding `equals`
8. Integer cache: -128 to 127 objects are pooled in Java
9. `final` fields are thread-safe once constructed
10. Static fields are per-class, not per-instance
11. Java passes object references by value
12. Python uses name mangling for `__field`
13. C++ RAII: object destruction triggers resource cleanup
14. Immutable objects are always thread-safe
15. Object slicing is a C++ pitfall
16. String literals in Java go to the string pool
17. `clone()` is shallow by default
18. Java records (Java 16+) are immutable value objects
19. Python `__slots__` reduces per-object memory
20. Object pool pattern eliminates GC pressure in hot paths

### Top 20 Interview Questions
1. Class vs Object?
2. Stack vs Heap for objects?
3. `==` vs `.equals()` in Java?
4. What is the Java object header?
5. How does GC know when to collect?
6. What is immutability and why does it matter?
7. Override `equals` without `hashCode` — what breaks?
8. What is object slicing in C++?
9. What is RAII?
10. What is the Integer cache?
11. What is a value object?
12. How to make a class thread-safe?
13. Shallow vs deep copy?
14. What is object interning?
15. Can constructors call virtual methods in C++?
16. What is `this`?
17. Static vs instance members?
18. What is a phantom reference?
19. What is `__slots__` in Python?
20. Design a thread-safe singleton.

### Top 20 Mistakes
1. `==` for object comparison (use `.equals()`)
2. Overriding `equals` without `hashCode`
3. Mutable public fields
4. Not validating constructor arguments
5. Memory leaks in C++ (forgetting `delete`)
6. Object slicing in C++
7. Using `float/double` for money values
8. Not making value objects immutable
9. Static mutable state
10. Unnecessary object creation in hot loops
11. Not overriding `toString()`
12. Shallow copy when deep copy needed
13. Circular references defeating GC
14. Integer boxing in tight loops
15. Not using `final` for immutable fields
16. Calling virtual methods from C++ constructors
17. Ignoring thread safety of shared objects
18. Returning mutable internal collections
19. Not using `Objects.requireNonNull`
20. Over-engineering simple data classes

### Top 20 Keywords
`class`, `object`, `instance`, `constructor`, `heap`, `stack`, `GC`, `vtable`, `vptr`, `equals`, `hashCode`, `immutable`, `static`, `final`, `RAII`, `object pool`, `value object`, `object header`, `reference`, `identity`

---

## 25. Final Interview Summary

### 5-Minute Revision
- Class = blueprint; Object = runtime instance with heap memory
- Java object header = 12-16 bytes overhead (mark word + class pointer)
- Stack = scope-bound, fast; Heap = GC-managed, for objects
- Virtual methods: vptr in object -> vtable lookup
- `==` = reference identity; `.equals()` = semantic equality
- Override `hashCode` when overriding `equals` (contract)
- Immutable objects = thread-safe automatically
- RAII in C++: resource lifetime tied to object scope
- Object pool: avoid allocation in hot paths

### 15-Minute Revision
Add to the above:
- Integer cache (-128..127), String pool interning
- Object slicing in C++ (assignment to base type)
- Shallow vs deep copy
- Java records, sealed classes (Java 17+)
- GC basics: generational GC, mark-and-sweep
- Thread-safety: synchronization, atomics, immutability
- Value objects in DDD
- Object pool pattern for HFT/game engines
- Python `__slots__` for memory reduction
- `this` vs `super` usage

### Night-Before Interview Revision
1. ⭐ Class vs Object: Blueprint vs runtime instance
2. ⭐ Heap layout: Header (12B) + fields + padding
3. ⭐ `==` vs equals: Identity vs equality
4. ⭐ equals/hashCode contract: If equal, same hash
5. ⭐ Immutability: Thread-safe by design
6. ⭐ Static: Class-level, not instance-level
7. ⭐ RAII: C++ deterministic cleanup
8. ⭐ Object pooling: Reduce GC in hot paths
9. ⭐ Virtual dispatch: vptr -> vtable -> function
10. ⭐ Value objects: Immutable, equality by value not identity
