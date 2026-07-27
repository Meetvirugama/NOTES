# OOP Overview — Industry-Level Interview Notes

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
Object-Oriented Programming is a **programming paradigm** that organizes code around **objects** — units that bundle both **data** (state) and **behavior** (methods) together. Instead of writing separate functions that operate on separate data structures, you model your domain as interacting objects.

### Technically
OOP is a software engineering methodology built on four core pillars — **Encapsulation**, **Abstraction**, **Inheritance**, and **Polymorphism** — that collectively enable:
- **Loose coupling**: Components depend on interfaces, not concrete implementations
- **High cohesion**: Related data and behavior live together in one unit
- **Information hiding**: Internal state protected from uncontrolled mutation
- **Code reuse**: Shared behavior through inheritance and composition

### From an Interviewer's Perspective
> "I don't want the dictionary definition. I want to know: why did OOP exist before Agile, DDD, and microservices and still survive? What do the four pillars mean at a machine level? What are OOP's limits in a concurrent system? When should you NOT use OOP? That's the conversation I want."

⭐ **Key Insight**: OOP solves **complexity management** in large codebases. The four pillars are the tools. The goal is always maintainable, extensible, testable software.

⭐ **Meta-insight**: OOP is not just about syntax. Every SOLID principle, every design pattern, every DDD aggregate root — all are applications of the four pillars at different levels of abstraction.

---

## 2. Why It Exists

### The Problem Before OOP

#### Procedural Programming (Pre-1980s)
```c
// C-style: Global data structures + standalone functions
struct BankAccount {
    double balance;
    char owner[50];
    int accountId;
};

void deposit(struct BankAccount* acc, double amount) {
    acc->balance += amount; // No validation. Anyone can call this.
}

void applyFraudulentChange(struct BankAccount* acc) {
    acc->balance = -999999.0; // VALID C. No protection!
    acc->accountId = 0;       // Corruption possible at any time!
}
```

**Problems:**
- Any function can modify any data structure — no protection
- Adding validation means hunting down EVERY place the struct is accessed
- Testing requires setting up the entire global state
- Parallel teams working on the same data structure = constant conflicts

#### The OOP Solution (1980s–90s, Smalltalk → C++ → Java)
```java
public class BankAccount {
    private double balance;      // Hidden state
    private final String owner;  // Immutable after construction

    public BankAccount(String owner, double initial) {
        if (initial < 0) throw new IllegalArgumentException();
        this.owner = owner;
        this.balance = initial;
    }

    public void deposit(double amount) {
        if (amount <= 0) throw new IllegalArgumentException();
        this.balance += amount;  // Only path to mutation
    }
    // No direct balance write possible from outside!
}
```

**OOP solved:**
1. **Accidental corruption** → Private fields + validated methods
2. **Find all call sites** → Change the method once, all callers benefit
3. **Parallel team conflicts** → Each team owns their class's internals
4. **Testing** → Mock interfaces, isolate components

### Real Software Examples
- **Java Collections Framework**: `HashMap` hides its hash table, bucket array, load factor, and rehashing behind `put()`/`get()`. You never touch the internals.
- **Android's `Activity`**: Every screen is an object with state (views, data) and behavior (`onCreate()`, `onPause()`, `onDestroy()`). The OS calls your lifecycle methods — Polymorphism in action.
- **Spring Framework**: `ApplicationContext` manages thousands of beans with complex wiring — all hidden behind simple `getBean()` calls.
- **JVM itself**: The JVM is an OOP masterpiece — `ClassLoader`, `Thread`, `GarbageCollector` are all modeled as extensible objects.

---

## 3. Internal Working

### How the Four Pillars Work at Machine Level

#### Encapsulation — Compile-Time Enforcement
```
At compile time:
  acc.balance = 100; // If balance is private → COMPILE ERROR
  
At binary level:
  // Same memory address as public fields!
  // "Private" is purely compiler annotation
  // → Zero runtime overhead
```

#### Abstraction — Virtual Table Indirection
```
Interface PaymentGateway {
    charge() -> vtable pointer at offset 0
    refund()  -> vtable pointer at offset 1
}

StripeGateway vtable:
[0] → StripeGateway::charge() machine code at 0x4A2F
[1] → StripeGateway::refund() machine code at 0x4B11

PaymentGateway* gw = new StripeGateway();
gw->charge(100); // → dereference vptr → vtable[0] → jump 0x4A2F
// Cost: 2 pointer dereferences + 1 indirect jump ≈ 3-5ns
```

#### Inheritance — Memory Layout
```
Animal object in heap:
+--[Animal fields]-----------+
|  name: String              | ← Animal portion
|  age: int                  |
+--[Dog fields]--------------+
|  breed: String             | ← Dog portion appended
|  isVaccinated: boolean     |
+----------------------------+

Dog* d = new Dog();
Animal* a = d;   // Same pointer — Animal portion begins at same offset
d->bark();       // Uses Dog vtable entry
a->makeSound();  // Uses Dog vtable entry via dynamic dispatch
```

#### Polymorphism — vtable Dispatch at Runtime
```
Animal[] zoo = { new Dog(), new Cat(), new Parrot() };

for (Animal a : zoo) {
    a.makeSound(); // For each iteration:
}

// Iteration 1: zoo[0].vptr → Dog::vtable → Dog::makeSound() → "Woof"
// Iteration 2: zoo[1].vptr → Cat::vtable → Cat::makeSound() → "Meow"
// Iteration 3: zoo[2].vptr → Parrot::vtable → Parrot::makeSound() → "Squawk"
// Runtime type, not declared type, determines behavior
```

### JVM Memory Model for Objects
```
Java Object Header (12–16 bytes):
+--[Mark Word: 8 bytes]---+
|  GC age (4 bits)        |
|  Bias lock info (54 bits)|
|  Hash code (25 bits)    |
+--[Klass Pointer: 4–8B]--+  ← Points to Class metadata
|  → class name, fields   |
|  → method tables (vtable)|
+--[Field Data]-----------+  ← Your actual fields start here
|  balance: double (8B)   |
|  owner: ref (4–8B)      |
+-------------------------+
```

### JIT Compiler Optimizations on OOP
```
Cold path (interpreter):
  account.getBalance() → push frame → execute → pop frame → return

Hot path after JIT compilation:
  account.getBalance() → // JIT detects: only one implementation loaded!
  → Inline: balance field access directly in caller
  → Zero method call overhead: equals public field access speed

→ OOP abstraction can become ZERO overhead after JIT warmup
```

---

## 4. Syntax

### All Four Pillars Together (Java)
```java
// ====== ABSTRACTION: Contract without implementation ======
public interface PaymentProcessor {
    PaymentResult process(Payment payment);  // WHAT, not HOW
    void refund(String transactionId, Money amount);
}

// ====== ABSTRACTION: Partial implementation with shared state ======
public abstract class AuditedProcessor implements PaymentProcessor {
    private final AuditLog auditLog;

    protected AuditedProcessor(AuditLog auditLog) {
        this.auditLog = auditLog;
    }

    // Template method: shared audit logic
    protected void logTransaction(Payment p, PaymentResult r) {
        auditLog.record(p, r, Instant.now());
    }
}

// ====== INHERITANCE + ENCAPSULATION: Concrete implementation ======
public final class StripeProcessor extends AuditedProcessor {
    // ENCAPSULATION: Hidden implementation details
    private final String secretKey;
    private final StripeClient client;
    private final RateLimiter rateLimiter;

    public StripeProcessor(String secretKey, AuditLog auditLog) {
        super(auditLog);  // Initialize parent state
        if (secretKey == null || secretKey.isBlank())
            throw new IllegalArgumentException("Secret key required");
        this.secretKey = secretKey;
        this.client = new StripeClient(secretKey);
        this.rateLimiter = RateLimiter.create(100.0); // 100 requests/sec
    }

    // ====== POLYMORPHISM: Runtime-resolved override ======
    @Override
    public PaymentResult process(Payment payment) {
        rateLimiter.acquire();
        PaymentResult result = client.charge(payment.getAmount(), payment.getCurrency());
        logTransaction(payment, result);  // Inherited behavior
        return result;
    }

    @Override
    public void refund(String transactionId, Money amount) {
        client.refund(transactionId, amount.getCents());
    }
}
```

### All Four Pillars Together (C++)
```cpp
#include <iostream>
#include <string>
#include <memory>
using namespace std;

// ABSTRACTION: Pure virtual interface
class Logger {
public:
    virtual ~Logger() = default;
    virtual void log(const string& level, const string& message) = 0;
    virtual void flush() = 0;
};

// ENCAPSULATION + INHERITANCE
class FileLogger : public Logger {
private:                              // ENCAPSULATION: hidden
    string filename;
    FILE* fileHandle;
    int logCount;

public:
    explicit FileLogger(const string& filename)
        : filename(filename), logCount(0) {
        fileHandle = fopen(filename.c_str(), "a");
        if (!fileHandle) throw runtime_error("Cannot open log file");
    }

    ~FileLogger() override {
        if (fileHandle) fclose(fileHandle);
    }

    // POLYMORPHISM: virtual override
    void log(const string& level, const string& message) override {
        fprintf(fileHandle, "[%s] %s\n", level.c_str(), message.c_str());
        ++logCount;
    }

    void flush() override { fflush(fileHandle); }

    int getLogCount() const { return logCount; }
};

// Client uses ABSTRACTION — doesn't know it's a FileLogger
void processRequest(Logger& logger, const string& data) {
    logger.log("INFO", "Processing: " + data);  // POLYMORPHISM
}
```

### Python
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ABSTRACTION
class Repository(ABC):
    @abstractmethod
    def save(self, entity) -> None:
        pass

    @abstractmethod
    def find_by_id(self, entity_id: str):
        pass

# ENCAPSULATION + INHERITANCE
class UserRepository(Repository):
    def __init__(self):
        self.__users: dict = {}  # ENCAPSULATION: name-mangled

    def save(self, user) -> None:  # POLYMORPHISM: override
        if not user.user_id:
            raise ValueError("User ID required")
        self.__users[user.user_id] = user

    def find_by_id(self, user_id: str):
        return self.__users.get(user_id)

    @property
    def count(self) -> int:
        return len(self.__users)
```

---

## 5. Visual Diagrams

### The Four Pillars Relationship
```
                    +--[OOP]---------------------------+
                    |                                  |
          +---------+--[4 Pillars]---+                 |
          |                         |                  |
    [Encapsulation]           [Abstraction]            |
    Hide STATE                Hide COMPLEXITY          |
    Access Modifiers          Interfaces/ABCs          |
          |                         |                  |
    [Inheritance]             [Polymorphism]           |
    Share CODE                Many BEHAVIORS           |
    IS-A relationship         Same interface           |
          |                         |                  |
          +-----------+-------------+                  |
                      |                                |
             [SOLID Principles]                        |
             [Design Patterns]                         |
             [Clean Architecture]                      |
                    +----------------------------------+
```

### OOP vs Procedural Memory View
```
PROCEDURAL (C):
Code Segment:   [deposit()]  [withdraw()]  [getBalance()]
Data Segment:   [account1]   [account2]   [account3]
                ^ Functions and data are separate.
                ^ Any function can access any data structure.

OOP (Java/C++):
Heap:           [BankAccount object]
                 ├── balance: 1000.0   ← data
                 ├── owner: "Alice"    ← data
                 ├── deposit()         ← method (code pointer)
                 ├── withdraw()        ← method (code pointer)
                 └── getBalance()      ← method (code pointer)
                Data and behavior are bundled and protected.
```

### Message Passing (Pillars in Action)
```
Client Code:

PaymentProcessor gw = new StripeProcessor("key");
↑                ↑    ↑
Abstraction    (Polymorphism)  Encapsulation
(Interface)  (compile-time     hides "key"
              type declared)

gw.process(payment);
↑
Polymorphism at runtime:
  gw.vptr → StripeProcessor vtable → StripeProcessor::process()
```

---

## 6. Real World Analogy

### Car (All 4 Pillars)
- **Encapsulation**: Engine compartment — hood closes, you can't touch fuel injectors, ABS module directly. You only use the gas pedal, steering wheel, brake pedal.
- **Abstraction**: Steering wheel hides rack-and-pinion gears, power steering pump, tie rods. Same wheel interface regardless of model year.
- **Inheritance**: SportsCar IS-A Car. Inherits all Car properties (4 wheels, engine, steering) and adds a turbocharger and sport suspension.
- **Polymorphism**: Tell a `Truck`, a `SportsCar`, and an `ElectricCar` to "accelerate" — each does it differently (diesel engine, turbo boost, electric motor) but responds to the same command.

### Hospital System
- **Encapsulation**: Patient's medical record — only doctors with authorization can view or modify. You can't directly edit your own diagnosis.
- **Abstraction**: The nurse is your interface. You tell her your symptoms; she translates to doctor, runs tests, brings results. You don't deal with lab equipment or billing systems.
- **Inheritance**: A `Cardiologist` IS-A `Doctor`. Inherits patient examination skills, adds specialized cardiac knowledge.
- **Polymorphism**: Call `doctor.treat(patient)` on a Cardiologist, a Neurologist, an Oncologist — each runs completely different treatment protocols.

---

## 7. Interview Explanation

### 30 Seconds
> "OOP organizes code around objects that bundle state and behavior. The four pillars are: Encapsulation — hide and protect internal state; Abstraction — expose only what clients need; Inheritance — reuse code via IS-A relationships; and Polymorphism — allow the same interface to have multiple implementations resolved at runtime."

### 1 Minute
> "OOP was created to solve the complexity problems of procedural code, where any function could modify any data structure without control. By binding data and methods together in objects, and using access modifiers to protect that data, OOP gave us the ability to enforce invariants. Abstraction via interfaces decouples callers from implementations — you can swap a MySQL database for DynamoDB without changing business logic. Inheritance reuses code but should be used carefully; modern practice favors composition. Polymorphism, implemented via virtual tables at the machine level, is what makes design patterns like Strategy and Observer possible."

### 3 Minutes
> "Let me go from the big picture down to the machine level.
>
> At the architecture level, OOP is a complexity management strategy. A 100K-line codebase broken into cohesive objects — each responsible for one thing — is maintainable. The same 100K lines as procedural spaghetti is a disaster. 
>
> At the language level, the four pillars translate to specific mechanisms. Encapsulation is compile-time access modifier enforcement — zero runtime cost. Abstraction is interface declarations that decouple callers from implementations. Inheritance is the IS-A hierarchy where child class memory layouts physically embed parent fields. Polymorphism is virtual dispatch: when you call a method on an interface reference, the JVM follows the object's `vptr` pointer to its class's `vtable`, looks up the method offset, and jumps to the specific implementation. This costs ~3-5 nanoseconds per call — a small but real overhead.
>
> That overhead matters at scale. The JIT compiler monitors call sites — if it sees that `PaymentProcessor.process()` is always called on a `StripeProcessor`, it can inline the specific implementation, eliminating the vtable lookup entirely. This is called devirtualization, and it's why well-written Java can approach C++ performance in hot paths.
>
> The major critique of OOP is mutable state. Encapsulated, mutable objects require synchronization in multithreaded systems. That's why modern systems blend OOP with functional ideas — immutable value objects (like Java `record`, Kotlin `data class`) that can be shared across threads without locks.
>
> And at the system design level, microservices *are* OOP — each service encapsulates its own data (private database), exposes a public API (interface), inherits team standards, and behaves polymorphically based on the domain."

---

## 8. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What are the 4 pillars of OOP? | Encapsulation, Abstraction, Inheritance, Polymorphism |
| What is a class vs an object? | Class = blueprint/type definition; Object = allocated instance in heap memory |
| What is the difference between Encapsulation and Abstraction? | Encapsulation = hiding/protecting DATA (access modifiers); Abstraction = hiding COMPLEXITY (interfaces) |
| Can you have OOP without inheritance? | Yes — composition is often preferred over inheritance |
| Is Java 100% OOP? | No — primitives (`int`, `double`) are not objects; static methods/fields don't require objects |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| What is the "Fragile Base Class" problem? | Changes to a base class can unexpectedly break subclasses that relied on the previous behavior |
| When does polymorphism happen at compile-time vs runtime? | Compile-time: overloading (method signature); Runtime: overriding (vtable dispatch based on actual object type) |
| Why might OOP be bad for concurrent systems? | Mutable state requires synchronization; immutable objects (FP approach) are inherently thread-safe |
| What is Composition over Inheritance? | Building objects from component objects (HAS-A) rather than extending base classes (IS-A) for flexibility |
| What is an Anemic Domain Model? | Objects with only fields and getters/setters — behavior lives in service classes instead. Anti-OOP pattern. |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| Explain virtual dispatch at the machine level | vptr in object → vtable of actual class → method address → indirect jump. Prevents CPU branch prediction. |
| What is JIT devirtualization? | JVM detects single implementation at a call site → inlines it directly, removing vtable lookup overhead |
| What is Data-Oriented Design and why is it faster than OOP? | DOD organizes data by type (arrays of positions, arrays of health) for cache locality vs OOP's object-per-entity (scattered heap) |
| How does OOP map to microservices? | Each service = encapsulated object: private DB (encapsulation), REST API (abstraction), domain hierarchy (inheritance), multiple implementations (polymorphism) |
| What is the Circle-Ellipse problem and how does it relate to OOP? | A Circle IS-A Ellipse mathematically, but Circle's `setRadius()` violates Ellipse's invariant of independent semi-axes. Demonstrates when inheritance is semantically incorrect. |

### 💼 Google Level
> *"Google's Protocol Buffers team considered making the generated classes fully OOP-encapsulated (private fields, accessor methods). They ultimately exposed raw mutable fields. What are the performance and API tradeoffs?"*

Expected: OOP-style: invariant enforcement, versioned API safety, but accessor overhead (JIT inlines these, but adds generated code). Raw fields: zero overhead, max performance for serialization hot paths, but no validation — callers must know invariants.

### 💼 Goldman Sachs Level
> *"Our trading system processes 1M orders/second. We noticed that using polymorphism (Strategy pattern) for order types caused a 15% performance regression compared to the if-else approach. How do you debug and resolve this?"*

Expected: Profile vtable call sites with async-profiler. If monomorphic → JIT should devirtualize (check if doing so with JITWatch). If polymorphic: consider sealed classes (Java 17) for exhaustive pattern matching. In extreme cases, C++ templates (zero-overhead abstraction) or data-oriented rewrite with type tags.

---

## 9. Coding Examples

### Demonstrating All Four Pillars: Payment System
```java
// ===== PILLAR 1: ABSTRACTION =====
// Client code depends on this — never on concrete implementation
public interface OrderProcessor {
    ProcessingResult processOrder(Order order);
    boolean canProcess(Order order);
}

// ===== PILLAR 2: ENCAPSULATION =====
// Internal state is protected; state transitions are validated
public class Order {
    private final String orderId;
    private final Money amount;
    private OrderStatus status;
    private final List<OrderItem> items;
    private Instant createdAt;

    public Order(String orderId, List<OrderItem> items) {
        if (orderId == null || orderId.isBlank())
            throw new IllegalArgumentException("Order ID required");
        if (items == null || items.isEmpty())
            throw new IllegalArgumentException("Order must have items");
        this.orderId = orderId;
        this.items = List.copyOf(items);
        this.amount = items.stream()
            .map(OrderItem::getPrice)
            .reduce(Money.ZERO, Money::add);
        this.status = OrderStatus.PENDING;
        this.createdAt = Instant.now();
    }

    // Controlled state transitions — not free setters
    public void confirm() {
        if (status != OrderStatus.PENDING)
            throw new IllegalStateException("Only PENDING orders can be confirmed");
        this.status = OrderStatus.CONFIRMED;
    }

    public void ship() {
        if (status != OrderStatus.CONFIRMED)
            throw new IllegalStateException("Only CONFIRMED orders can be shipped");
        this.status = OrderStatus.SHIPPED;
    }

    // No setter for amount, orderId, items, createdAt — immutable after construction
    public String getOrderId() { return orderId; }
    public Money getAmount() { return amount; }
    public OrderStatus getStatus() { return status; }
    public List<OrderItem> getItems() { return items; } // Already unmodifiable
}

// ===== PILLAR 3: INHERITANCE =====
// Shared validation and audit logic in base class
public abstract class BaseOrderProcessor implements OrderProcessor {
    private final AuditService auditService;
    protected final RateLimiter rateLimiter;

    protected BaseOrderProcessor(AuditService auditService, double rateLimit) {
        this.auditService = auditService;
        this.rateLimiter = RateLimiter.create(rateLimit);
    }

    // Template Method: shared algorithm, customizable steps
    @Override
    public final ProcessingResult processOrder(Order order) {
        rateLimiter.acquire(); // Inherited rate limiting
        if (!canProcess(order)) {
            return ProcessingResult.rejected("Order cannot be processed by this processor");
        }
        ProcessingResult result = doProcess(order); // Polymorphic step
        auditService.record(order, result, getClass().getSimpleName());
        return result;
    }

    protected abstract ProcessingResult doProcess(Order order);
}

// ===== PILLAR 4: POLYMORPHISM =====
// Multiple implementations — same interface, different behaviors

public class StripeOrderProcessor extends BaseOrderProcessor {
    private final StripeClient stripeClient;

    public StripeOrderProcessor(StripeClient client, AuditService audit) {
        super(audit, 100.0); // 100 req/sec
        this.stripeClient = client;
    }

    @Override
    public boolean canProcess(Order order) {
        return order.getAmount().getCurrency().equals(Currency.USD)
            && order.getAmount().getCents() <= 100_000_00L; // $100K limit
    }

    @Override
    protected ProcessingResult doProcess(Order order) {
        StripeCharge charge = stripeClient.charge(order.getAmount(), order.getOrderId());
        order.confirm();
        return ProcessingResult.success(charge.getId());
    }
}

public class CryptoOrderProcessor extends BaseOrderProcessor {
    private final BlockchainClient blockchainClient;

    public CryptoOrderProcessor(BlockchainClient client, AuditService audit) {
        super(audit, 10.0); // 10 req/sec (blockchain is slower)
        this.blockchainClient = client;
    }

    @Override
    public boolean canProcess(Order order) {
        return order.getAmount().getCents() >= 50_000_00L; // Min $50K for crypto
    }

    @Override
    protected ProcessingResult doProcess(Order order) {
        Transaction txn = blockchainClient.submit(order.getAmount(), order.getOrderId());
        order.confirm();
        return ProcessingResult.success(txn.getHash());
    }
}

// Routing: polymorphism in action
public class OrderService {
    private final List<OrderProcessor> processors; // All behind interface

    public OrderService(List<OrderProcessor> processors) {
        this.processors = processors;
    }

    public ProcessingResult process(Order order) {
        return processors.stream()
            .filter(p -> p.canProcess(order)) // Each processor decides
            .findFirst()
            .map(p -> p.processOrder(order))  // Runtime dispatch
            .orElseThrow(() -> new UnsupportedOperationException("No processor for order"));
    }
}
```

### Composition over Inheritance
```java
// INHERITANCE ANTI-PATTERN: Deep hierarchy, brittle
class Vehicle { void start() {} }
class MotorVehicle extends Vehicle { void refuel() {} }
class Car extends MotorVehicle { void park() {} }
class ElectricCar extends Car {
    @Override void refuel() { throw new UnsupportedOperationException("No refueling!"); }
    // Liskov Substitution VIOLATED!
}

// COMPOSITION PATTERN: Flexible, no hierarchy problems
interface Startable { void start(); }
interface Fuelable { void refuel(FuelType type); }
interface Chargeable { void charge(ChargerType type); }
interface Parkable { void park(ParkingSpot spot); }

class Car implements Startable, Fuelable, Parkable {
    private final Engine engine; // HAS-A Engine
    private final FuelTank tank; // HAS-A FuelTank

    public void start() { engine.start(); }
    public void refuel(FuelType type) { tank.fill(type); }
    public void park(ParkingSpot spot) { /* parallel park logic */ }
}

class ElectricCar implements Startable, Chargeable, Parkable {
    private final ElectricMotor motor;  // HAS-A ElectricMotor
    private final Battery battery;      // HAS-A Battery

    public void start() { motor.start(); }
    public void charge(ChargerType type) { battery.charge(type); }
    public void park(ParkingSpot spot) { /* auto-park logic */ }
}
// No hierarchy. No forced method implementations. Each class is exactly what it is.
```

---

## 10. Common Mistakes

### ⚠️ Mistake 1: Anemic Domain Model
```java
// ANTI-PATTERN: Classes are just data bags. Behavior lives in service classes.
class Order {
    private String id;
    private String status;
    private double amount;
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; } // Free mutation!
    public double getAmount() { return amount; }
    public void setAmount(double amount) { this.amount = amount; } // Negative allowed!
}

class OrderService {
    public void shipOrder(Order order) {
        // Validation externalized — anyone can bypass this:
        if ("CONFIRMED".equals(order.getStatus())) {
            order.setStatus("SHIPPED");
        }
    }
}

// PROBLEM: Another team writes:
order.setStatus("SHIPPED"); // Bypasses validation entirely!

// FIX: Move behavior into the entity (Rich Domain Model)
class Order {
    private OrderStatus status;
    public void ship() {
        if (status != OrderStatus.CONFIRMED) throw new IllegalStateException();
        this.status = OrderStatus.SHIPPED;
    }
}
```

### ⚠️ Mistake 2: Inheritance for Code Reuse Alone
```java
// WRONG: Inheriting just to reuse utility methods
class HttpClient {
    protected String buildHeaders() { /* ... */ }
    protected void parseResponse() { /* ... */ }
}

class UserService extends HttpClient { // UserService IS NOT an HttpClient!
    public User getUser(String id) {
        buildHeaders(); // Just wants to reuse this...
        return parseResponse();
    }
}
// Breaking: UserService now carries all public API of HttpClient
// Fix: Inject HttpClient as dependency (Composition)
class UserService {
    private final HttpClient httpClient; // HAS-A
    public UserService(HttpClient httpClient) { this.httpClient = httpClient; }
}
```

### ⚠️ Mistake 3: God Class
```java
// ANTI-PATTERN: One class does everything
class SystemManager {
    public void saveUser(User user) { /* DB access */ }
    public void sendEmail(String to, String subject) { /* Email sending */ }
    public void generateInvoicePDF(Order order) { /* PDF generation */ }
    public void processPayment(Payment payment) { /* Stripe calls */ }
    public void updateInventory(Product p, int qty) { /* Inventory */ }
    // 50 more methods...
    // NO cohesion. Multiple reasons to change. Untestable.
}
// Fix: Apply SRP. Split into UserRepository, EmailService, InvoiceGenerator,
//      PaymentService, InventoryService
```

### ⚠️ Mistake 4: Overusing Inheritance (IS-A vs CAN-DO)
```java
// WRONG: Stack extends Vector (actual Java SDK mistake!)
class Stack<E> extends Vector<E> {
    public E push(E item) { addElement(item); return item; }
    public E pop() { /* last element */ }
    // BUG: Stack IS-A Vector? No! But now callers can do:
    // stack.add(0, element); // Inserts at beginning — violates stack semantics!
    // stack.remove(3);       // Removes arbitrary element — not a stack!
}
// Fix: Stack SHOULD have been a wrapper (composition) around a private Deque
```

### ⚠️ Mistake 5: Not Using Polymorphism (if-else for types)
```java
// WRONG: Switch on type — violates Open-Closed Principle
class ShapeRenderer {
    public void render(Shape shape) {
        if (shape instanceof Circle) {
            renderCircle((Circle) shape);
        } else if (shape instanceof Square) {
            renderSquare((Square) shape);
        } else if (shape instanceof Triangle) {  // Adding = modifying!
            renderTriangle((Triangle) shape);
        }
    }
}

// CORRECT: Polymorphism — adding new shape = adding new class
interface Shape {
    void render(GraphicsContext ctx);
}
class Circle implements Shape { public void render(GraphicsContext ctx) { /* ... */ } }
class Square implements Shape { public void render(GraphicsContext ctx) { /* ... */ } }
// New type: just implement Shape. Zero changes to existing code.
```

---

## 11. Best Practices

### OOP Design Principles
- **Single Responsibility**: One class = one reason to change
- **Open-Closed**: Open for extension via interfaces, closed for modification
- **Liskov Substitution**: Subtype must be fully substitutable for its supertype
- **Interface Segregation**: Small, focused interfaces > one fat interface
- **Dependency Inversion**: High-level modules depend on abstractions

### 🚀 Production Rules
- **Composition over Inheritance**: Prefer HAS-A over IS-A for flexibility
- **Program to Interfaces**: `List<String>` not `ArrayList<String>` in declarations
- **Fail-Fast in Constructors**: Validate all arguments; throw early
- **Immutable Where Possible**: `final` fields = thread-safe by default
- **Tell, Don't Ask**: Let the object act on its own data rather than asking for data externally
- **Law of Demeter**: Only call methods on direct collaborators — not chains of getters

### Anti-Patterns to Avoid
- God Class, Anemic Domain Model, Deep Inheritance Trees
- Singleton overuse (hidden global state)
- Service Locator (hidden dependency injection)
- `instanceof` chains (replace with polymorphism)

---

## 12. Complexity

### Runtime Costs of OOP Mechanisms

| Mechanism | Cost | Notes |
|-----------|------|-------|
| Field access | O(1), 1-4ns | Direct memory read |
| Static method call | O(1), 2-5ns | Direct jump, stack frame |
| Virtual method call | O(1), 5-10ns | vtable lookup + indirect jump |
| Interface call (JVM) | O(1), 5-15ns | `invokeinterface` — slightly slower than `invokevirtual` |
| JIT-inlined method | O(1), 0-1ns | After JIT: same as field access |
| Object allocation | O(1), ~5ns amortized | JVM TLAB bump pointer |
| GC overhead | O(N) pause | Varies by collector: G1 <1ms, ZGC <0.1ms |

### Memory Overhead
```
JVM Object Header: 12–16 bytes per object
→ 1,000,000 Integer objects = 16MB just for headers!
→ vs 1,000,000 ints = 4MB
→ Object density matters at scale: use primitives in hot loops
```

---

## 13. Advantages

| Advantage | Why it Matters |
|-----------|---------------|
| **Invariant Enforcement** | Constructors + private fields = object always in valid state |
| **Maintainability** | Internal changes don't break callers (encapsulation) |
| **Extensibility** | New implementations via interfaces (OCP) |
| **Testability** | Interface dependencies → mock in tests, no real infrastructure |
| **Domain Modeling** | Classes map 1:1 to business entities — readable, communicative code |
| **Team Scalability** | Each team owns their class's internals; clear API boundaries |
| **Reuse** | Inheritance + composition — don't re-implement common logic |

---

## 14. Disadvantages

| Disadvantage | Severity | When It Hurts |
|-------------|----------|---------------|
| **Mutable state threading** | High | Concurrent systems need locks on every shared mutable object |
| **Object overhead** | Medium | 16-byte headers × millions of objects = GB of memory overhead |
| **Cache misses** | High | Scattered heap objects → CPU cache misses in tight loops (DOD wins here) |
| **Boilerplate** | Medium | Getters, setters, constructors, interfaces, implementations |
| **Over-engineering risk** | Medium | Premature abstraction creates wrong APIs |
| **Fragile Base Class** | High | Base class changes can silently break subclasses |
| **Vtable overhead** | Low | ~5-15ns per virtual call; JIT eliminates in many cases |

---

## 15. Comparison Table

### OOP vs Other Paradigms

| Paradigm | Focus | State | Best For | Worst For |
|---------|-------|-------|----------|-----------|
| **OOP** | Objects | Mutable | Enterprise apps, UI, games | High concurrency, data pipelines |
| **Procedural** | Functions | Global | OS kernels, embedded, scripts | Large codebases (hard to manage) |
| **Functional** | Functions | Immutable | Concurrency, data transformation | Stateful UI, complex domain models |
| **Data-Oriented** | Memory layout | Structured | Game engines, HFT, realtime | Complex business logic |

### Inheritance vs Composition

| Aspect | Inheritance | Composition |
|--------|-------------|-------------|
| **Relationship** | IS-A | HAS-A |
| **Coupling** | Tight (shares base class changes) | Loose (interface-based) |
| **Flexibility** | Low (hierarchy fixed at compile time) | High (swap at runtime) |
| **Code reuse** | Yes (but limited) | Yes (any component anywhere) |
| **Multiple behaviors** | Hard (single inheritance in Java) | Easy (implement multiple interfaces) |
| **Testing** | Hard (mock entire hierarchy) | Easy (mock individual component) |
| **When to use** | Genuine IS-A, stable hierarchies | Most cases |

---

## 16. Design Pattern Connection

All GoF Design Patterns are applications of the 4 OOP pillars:

| Category | Pattern | Pillars Used |
|---------|---------|-------------|
| **Creational** | Factory Method | Abstraction, Polymorphism |
| | Builder | Encapsulation, Abstraction |
| | Singleton | Encapsulation |
| **Structural** | Adapter | Abstraction, Polymorphism |
| | Decorator | Inheritance, Composition |
| | Facade | Abstraction, Encapsulation |
| | Composite | Polymorphism, Abstraction |
| **Behavioral** | Strategy | Abstraction, Polymorphism |
| | Observer | Abstraction, Polymorphism |
| | Template Method | Inheritance, Polymorphism |
| | Command | Abstraction, Encapsulation |
| | Iterator | Encapsulation, Abstraction |

---

## 17. System Design Connection

### Microservices as OOP at Scale
```
OOP Concept          → Microservice Equivalent
─────────────────────────────────────────────
Class                → Service (Order Service, User Service)
Encapsulation        → Private Database (no shared DB)
Public Method        → REST/gRPC API endpoint
Interface            → API contract (OpenAPI, Protobuf schema)
Inheritance          → Shared libraries, sidecar patterns
Polymorphism         → Multiple implementations behind API Gateway
```

### DDD (Domain-Driven Design) — Pure Applied OOP
- **Entity** = Class with identity and mutable state (Encapsulation)
- **Value Object** = Immutable class defined by its values (Encapsulation + Immutability)
- **Aggregate Root** = Class that controls access to child objects (Encapsulation at cluster level)
- **Repository** = Interface (Abstraction) with multiple implementations (Polymorphism)
- **Domain Event** = Command/Observer pattern (Behavioral patterns)

### Clean Architecture Layers = Abstraction Layers
```
[Entities/Domain] ← [Use Cases] ← [Interface Adapters] ← [Frameworks/DB]

Dependency Inversion: Inner layers depend on abstractions
Each layer = higher-level abstraction hiding lower-level details
```

---

## 18. Multithreading Connection

### OOP's Core Problem: Shared Mutable State
```java
// OOP mutable object — UNSAFE across threads
class Counter {
    private int count = 0; // Mutable state!
    public void increment() { count++; } // i++ = read → add → write = 3 ops = race condition
}

// Fix 1: Synchronization
class SynchronizedCounter {
    private int count = 0;
    public synchronized void increment() { count++; }
    // Works but: only one thread at a time = bottleneck
}

// Fix 2: Atomic operations
class AtomicCounter {
    private final AtomicInteger count = new AtomicInteger(0);
    public void increment() { count.incrementAndGet(); } // CAS — lock-free
}

// Fix 3: Immutability (FP approach) — the ultimate thread safety
final class ImmutableCounter {
    private final int count;
    public ImmutableCounter(int count) { this.count = count; }
    public ImmutableCounter increment() { return new ImmutableCounter(count + 1); }
    // No mutation → no synchronization needed → freely shared across threads
}
```

### OOP + FP Hybrid (Modern Best Practice)
```java
// Commands are value objects (immutable) — FP style
record PlaceOrderCommand(String orderId, List<OrderItem> items, CustomerId customer) {}

// Handlers are objects with injected dependencies — OOP style
@Service
public class PlaceOrderHandler {
    private final OrderRepository repo;
    private final PaymentService payments;
    private final EventPublisher events;

    public PlaceOrderHandler(OrderRepository repo, PaymentService payments, EventPublisher events) {
        this.repo = repo;
        this.payments = payments;
        this.events = events;
    }

    public OrderId handle(PlaceOrderCommand cmd) {
        Order order = new Order(cmd.orderId(), cmd.items()); // Immutable value creation
        payments.charge(order.getAmount(), cmd.customer());
        repo.save(order);
        events.publish(new OrderPlaced(order.getOrderId())); // Immutable event
        return order.getOrderId();
    }
}
```

---

## 19. Company Interview Perspective

### Google
- "How does OOP's vtable mechanism affect CPU branch prediction?"
- Data-oriented design vs OOP for Dremel/Spanner-scale performance
- Protocol Buffers design: why not full OOP encapsulation on generated fields?

### Goldman Sachs
- Model a trading system using OOP. `TradeOrder`, `Portfolio`, `RiskEngine`, `PriceQuote`.
- How does encapsulation prevent position limit violations?
- Event sourcing: how do you model state transitions using OOP invariants?

### Amazon
- DDD + OOP: model an Amazon product catalog with aggregate roots, entities, value objects.
- Why is the Anemic Domain Model an anti-pattern for complex domains?
- How does Spring IoC map to OOP's Dependency Inversion?

### Meta
- React moved from OOP class components to functional hooks — what were the tradeoffs?
- Python ABC vs Protocol (structural subtyping) — when does each apply?
- GraphQL schema = abstraction layer — how does it hide datastore complexity?

### Microsoft
- C# vs Java OOP: properties vs getters, structs vs classes, `sealed` vs `final`.
- COM interfaces as stable ABI: how pure virtual C++ classes achieve binary compatibility.
- WPF MVVM: how does OOP enable the ViewModel/View separation?

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Is Java 100% Object-Oriented? | No — primitives (`int`, `long`) are not objects; static members exist without instances |
| 2 | ⚠️ Can you achieve Abstraction without Encapsulation? | Yes — an interface (abstraction) with a concrete class having all `public` fields (no encapsulation). Different pillars. |
| 3 | ⚠️ Does OOP guarantee thread safety? | No — OOP encapsulates mutable state; mutable state requires explicit synchronization |
| 4 | What is the Circle-Ellipse problem? | Mathematically Circle IS-A Ellipse, but `setWidth()` on a Circle forces height = width, violating Ellipse invariant. Shows limits of IS-A |
| 5 | ⚠️ What is the Yo-Yo problem? | Deep inheritance: understanding method behavior requires bouncing up/down 6+ class levels in the hierarchy |
| 6 | Why did Java's `Stack` extend `Vector`? | Historical mistake — implemented IS-A for code reuse. Should have used composition. Now `Stack.add(0, e)` violates stack semantics. |
| 7 | What is a "vtable" and how many vtables does a class have? | One vtable per class (not per object). Objects have a `vptr` pointer to their class's vtable. |
| 8 | ⚠️ Does `final` class make it thread-safe? | No — `final` prevents subclassing; thread safety requires `final` fields + immutability or synchronization |
| 9 | What happens if two interfaces have the same default method? | Compile error — implementing class must override and explicitly resolve the conflict |
| 10 | ⚠️ Can an interface have state? | Not instance state. Only `public static final` constants. Default methods can access these. |
| 11 | What is the Fragile Base Class problem? | Parent class is changed (even safely); child class depended on old behavior → subtle runtime bugs without compile error |
| 12 | What is Structural vs Nominal typing in OOP? | Java = Nominal (must explicitly declare `implements`); Python/Go = Structural (if it has the methods, it's compatible — "Duck typing") |
| 13 | ⚠️ When does JIT devirtualization NOT work? | When a call site is polymorphic (megamorphic — >2 types seen). JIT gives up inlining; reverts to vtable dispatch. |
| 14 | What is the difference between `instanceof` and polymorphism? | `instanceof` = type-checking then casting = procedural style. Polymorphism = no type check, method resolves itself — OOP style |
| 15 | Can abstract classes have constructors? | Yes — they run when subclass calls `super()`. Cannot instantiate directly, but constructors initialize shared state |

---

## 21. Coding Problems

### Easy — OOP Design: Library System
Design a basic Library system with Books, Members, and Borrowing.
```java
// Value object
public record ISBN(String value) {
    public ISBN { if (!value.matches("\\d{13}")) throw new IllegalArgumentException(); }
}

// Entity
public class Book {
    private final ISBN isbn;
    private final String title;
    private final String author;
    private BookStatus status;

    public Book(ISBN isbn, String title, String author) {
        this.isbn = isbn;
        this.title = title;
        this.author = author;
        this.status = BookStatus.AVAILABLE;
    }

    public void checkout() {
        if (status != BookStatus.AVAILABLE)
            throw new IllegalStateException("Book not available");
        this.status = BookStatus.CHECKED_OUT;
    }

    public void returnBook() { this.status = BookStatus.AVAILABLE; }
    public boolean isAvailable() { return status == BookStatus.AVAILABLE; }
}

// Repository abstraction
public interface BookRepository {
    void save(Book book);
    Optional<Book> findByISBN(ISBN isbn);
}

// Service — uses abstraction
public class LibraryService {
    private final BookRepository books;

    public LibraryService(BookRepository books) { this.books = books; }

    public void checkout(ISBN isbn, Member member) {
        Book book = books.findByISBN(isbn)
            .orElseThrow(() -> new NotFoundException("Book not found"));
        book.checkout();
        member.recordCheckout(book);
        books.save(book);
    }
}
```

### Medium — OOP Design: Vending Machine (State Pattern)
```java
public class VendingMachine {
    private VendingState state;
    private final Map<String, Integer> inventory;
    private double balance;

    public VendingMachine(Map<String, Integer> inventory) {
        this.inventory = new HashMap<>(inventory);
        this.state = new IdleState(this);
    }

    // Delegate behavior to current state (State Pattern = Polymorphism)
    public void insertCoin(double amount) { state.insertCoin(amount); }
    public void selectProduct(String productId) { state.selectProduct(productId); }
    public void cancel() { state.cancel(); }

    void setState(VendingState state) { this.state = state; }
    void addBalance(double amount) { this.balance += amount; }
    double getBalance() { return balance; }
    boolean hasProduct(String productId) { return inventory.getOrDefault(productId, 0) > 0; }
    void dispenseProduct(String productId) { inventory.merge(productId, -1, Integer::sum); }
    void resetBalance() { this.balance = 0; }
}

interface VendingState {
    void insertCoin(double amount);
    void selectProduct(String productId);
    void cancel();
}

class IdleState implements VendingState {
    private final VendingMachine machine;
    IdleState(VendingMachine machine) { this.machine = machine; }

    public void insertCoin(double amount) {
        machine.addBalance(amount);
        machine.setState(new HasMoneyState(machine));
    }
    public void selectProduct(String id) { System.out.println("Insert coin first"); }
    public void cancel() { System.out.println("Nothing to cancel"); }
}
// HasMoneyState, DispensingState implementations...
```

### Hard — OOP Design: Parking Lot
```java
// Abstract vehicle hierarchy
public abstract class Vehicle {
    private final String licensePlate;
    public abstract VehicleSize getSize();
    protected Vehicle(String licensePlate) { this.licensePlate = licensePlate; }
    public String getLicensePlate() { return licensePlate; }
}

public class Motorcycle extends Vehicle {
    public Motorcycle(String plate) { super(plate); }
    public VehicleSize getSize() { return VehicleSize.SMALL; }
}

public class Car extends Vehicle {
    public Car(String plate) { super(plate); }
    public VehicleSize getSize() { return VehicleSize.MEDIUM; }
}

// Encapsulated parking spot
public class ParkingSpot {
    private final int spotId;
    private final VehicleSize spotSize;
    private Vehicle parkedVehicle;

    public ParkingSpot(int spotId, VehicleSize size) {
        this.spotId = spotId;
        this.spotSize = size;
    }

    public boolean canFit(Vehicle vehicle) {
        return parkedVehicle == null && vehicle.getSize().ordinal() <= spotSize.ordinal();
    }

    public void park(Vehicle vehicle) {
        if (!canFit(vehicle)) throw new IllegalStateException("Spot not available");
        this.parkedVehicle = vehicle;
    }

    public Vehicle unpark() {
        Vehicle v = this.parkedVehicle;
        this.parkedVehicle = null;
        return v;
    }

    public boolean isEmpty() { return parkedVehicle == null; }
}

// Abstraction layer
public interface ParkingLotService {
    Optional<ParkingSpot> findAvailableSpot(Vehicle vehicle);
    Ticket parkVehicle(Vehicle vehicle);
    Money checkout(Ticket ticket);
}
```

---

## 22. Revision Sheet

| Concept | Key Rule |
|---------|----------|
| Encapsulation | Private fields + validated methods. Object controls its own state. |
| Abstraction | Interface = what; Implementation = how. Client sees only interface. |
| Inheritance | IS-A relationship. Child IS a more specific version of parent. |
| Polymorphism | Same interface, different implementations resolved at runtime via vtable. |
| Composition | HAS-A relationship. Preferred over inheritance for flexibility. |
| Anemic Domain Model | Anti-pattern: classes have data but no behavior. |
| Tell Don't Ask | Tell object to do something; don't ask for its data and act externally. |
| Fragile Base Class | Parent changes break child classes unexpectedly. |
| vtable | Per-class table of method pointers. vptr in each object points to its class's vtable. |
| JIT Devirtualization | JVM inlines virtual calls when only one implementation is seen. |

### OOP in One Sentence Per Pillar
- **Encapsulation**: "My data is mine — here's the only way to interact with it."
- **Abstraction**: "You don't need to know how I do it — just what I can do."
- **Inheritance**: "I am everything my parent is, plus more."
- **Polymorphism**: "Call me; I'll figure out the right behavior for my type."

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| 4 pillars of OOP? | Encapsulation, Abstraction, Inheritance, Polymorphism |
| What is a vtable? | Per-class table of method pointers; objects have vptr pointing to it |
| Compile-time vs Runtime polymorphism? | Compile = overloading; Runtime = overriding via vtable |
| Composition vs Inheritance? | Composition = HAS-A (flexible); Inheritance = IS-A (tightly coupled) |
| What is an Anemic Domain Model? | Objects with only data + getters/setters; behavior in separate service classes |
| Fragile Base Class problem? | Base class changes silently break subclasses |
| What is JIT devirtualization? | JVM inlines monomorphic virtual calls — zero overhead |
| OOP and thread safety? | OOP doesn't guarantee thread safety; mutable state requires synchronization |
| When does OOP hurt performance? | Cache misses (scattered heap), vtable overhead, GC pressure |
| Is Java 100% OOP? | No — primitives, static methods |
| Tell Don't Ask? | Tell object to perform action; don't ask for data and act externally |
| What is the Circle-Ellipse problem? | Shows IS-A inheritance can be semantically wrong despite mathematical relationship |
| Law of Demeter? | Only talk to immediate collaborators |
| What is DDD? | Domain-Driven Design — OOP applied to complex business domains |
| What is the Anemic Domain Model's fix? | Rich Domain Model: move behavior into domain entities |
| `invokeinterface` vs `invokevirtual`? | `invokeinterface` is slower — searches for method offset (no fixed offset in multiple-interface scenarios) |
| What is structural subtyping? | Duck typing: if it has the methods, it's compatible. Used in Go, Python (Protocol) |
| What is SOLID? | 5 OOP design principles: SRP, OCP, LSP, ISP, DIP |
| What is a Value Object in DDD? | Immutable class defined by its values, no identity (e.g., Money, Address) |
| What is microservices vs OOP relationship? | Microservices = OOP at system scale (encapsulated state, public API, polymorphic implementations) |

---

## 24. Cheat Sheet

### Top 20 OOP Facts
1. OOP solves complexity in large codebases by binding data + behavior into cohesive units
2. Encapsulation = compile-time enforcement via access modifiers (zero runtime cost)
3. Abstraction = interface decouples caller from implementation (small runtime cost)
4. Inheritance = IS-A; child object's memory layout contains parent portion at start
5. Polymorphism = runtime vtable dispatch; ~5-15ns per virtual call
6. JIT devirtualization eliminates vtable overhead for monomorphic call sites
7. Java is NOT 100% OOP — primitives and static members exist without objects
8. Composition (HAS-A) is almost always preferred over Inheritance (IS-A)
9. Anemic Domain Model = anti-pattern; behavior belongs with data
10. OOP's biggest weakness: mutable state is dangerous in concurrent systems
11. Immutability is the FP solution to OOP concurrency problems
12. Data-Oriented Design beats OOP in cache-sensitive tight loops (game engines)
13. Fragile Base Class problem: parent changes silently break subclasses
14. SOLID principles = how to apply the 4 pillars correctly
15. All GoF design patterns are OOP pillar applications
16. Microservices = OOP at system architecture level
17. DDD = OOP applied to complex business domain modeling
18. Java `record` = auto-encapsulated immutable value object
19. `instanceof` chains = procedural thinking in OOP code → use polymorphism
20. OOP is not the only truth: blend with FP (immutability) for best results

---

## 25. Final Interview Summary

### Night-Before Revision
1. ⭐ OOP = Encapsulation + Abstraction + Inheritance + Polymorphism
2. ⭐ Encapsulation: hide state, protect invariants (compile-time, zero runtime cost)
3. ⭐ Abstraction: interface separates WHAT from HOW (vtable dispatch at runtime)
4. ⭐ Inheritance: IS-A, memory layout includes parent fields
5. ⭐ Polymorphism: vtable lookup — vptr → class vtable → method address
6. ⭐ JIT devirtualization: monomorphic virtual call = inlined = zero overhead
7. ⭐ Composition over Inheritance: HAS-A > IS-A in most cases
8. ⭐ Anemic Domain Model = behavior separated from data = anti-OOP
9. ⭐ OOP mutable state = threading headache → use immutability (final fields)
10. ⭐ Microservices = OOP at system scale (encapsulated service = encapsulated class)
