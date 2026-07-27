# Abstraction — Industry-Level Interview Notes

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
Abstraction means **hiding HOW something is done** and showing only **WHAT it can do**. It lets you work with complex systems through a simple interface without needing to understand the internal machinery.

### Technically
Abstraction is an OOP mechanism that separates the **contract** (what operations are available and what they promise) from the **implementation** (how those operations are carried out). It's realized through:
- **Abstract classes**: Partially implemented classes that define a contract with shared code
- **Interfaces** (Java/C#): Pure contracts — no implementation, only method signatures and constants
- **Abstract Base Classes (ABC)** in Python
- **Pure virtual functions** in C++

The key property: **clients depend on the abstraction, not the concrete type**. This makes implementations swappable.

### From an Interviewer's Perspective
> "Don't confuse Abstraction with Encapsulation. Encapsulation hides STATE. Abstraction hides COMPLEXITY and decouples callers from implementations. I want to hear about: abstract classes vs interfaces (and when to use which), the Dependency Inversion Principle, API vs ABI stability in C++, leaky abstractions, and how good abstraction design prevents ripple effects when implementations change."

⭐ **Core rule**: Encapsulation → "Who can access my data?" Abstraction → "What do I expose to my callers?"

⭐ **The most important benefit**: When you abstract, you can change implementations without breaking callers.

---

## 2. Why It Exists

### Problem Without Abstraction

#### Tight Coupling
```java
// NO ABSTRACTION: Business logic directly depends on concrete infrastructure
public class OrderService {
    // Hardcoded concrete dependency
    private MySQLOrderRepository repository = new MySQLOrderRepository(
        "jdbc:mysql://prod-db.internal:3306/orders",
        "user", "password"
    );
    private SMTPEmailSender emailSender = new SMTPEmailSender(
        "smtp.gmail.com", 587, "user@company.com"
    );

    public void placeOrder(Order order) {
        repository.save(order);           // Coupled to MySQL
        emailSender.send(order.getCustomerEmail(), "Order confirmed");
    }

    // Problems:
    // 1. Testing requires actual MySQL running + real SMTP server
    // 2. Switch to DynamoDB = rewrite OrderService
    // 3. Switch to Twilio SMS = rewrite OrderService
    // 4. New requirement: A/B test two email providers = rewrite OrderService
}
```

#### What Happens Without Abstraction
| Problem | Impact |
|---------|--------|
| No swappable implementations | Switching DB requires rewriting all calling code |
| No testability | Unit tests require real external services running |
| No extensibility | Adding new behavior requires modifying existing code |
| High coupling | Changes in MySQL library break OrderService |
| No parallel development | Team can't build UI until DB layer is ready |

### With Abstraction
```java
// WITH ABSTRACTION: OrderService depends on interfaces
public class OrderService {
    private final OrderRepository repository;  // Interface — any implementation
    private final NotificationService notifier; // Interface — any implementation

    public OrderService(OrderRepository repository, NotificationService notifier) {
        this.repository = repository;
        this.notifier = notifier;
    }

    public void placeOrder(Order order) {
        repository.save(order);      // Works with MySQL, DynamoDB, or InMemory
        notifier.notify(order);      // Works with SMTP, SMS, Push, or Fake
    }
}

// Production: inject real implementations
OrderService prod = new OrderService(
    new MySQLOrderRepository(dbConfig),
    new SMTPNotificationService(smtpConfig)
);

// Testing: inject fakes — no real infrastructure!
OrderService test = new OrderService(
    new InMemoryOrderRepository(),
    new FakeNotificationService()
);
```

### Real Software Examples
- **JDBC (Java Database Connectivity)**: `Connection`, `Statement`, `ResultSet` are interfaces. Your code works with MySQL, PostgreSQL, Oracle, H2 — any driver implementing the JDBC interface.
- **Java Collections**: `List<String> list = new ArrayList<>()`. Switch to `LinkedList` for O(1) insertions — zero change to calling code.
- **JPA/Hibernate**: `EntityManager` interface abstracts SQL generation and connection management. Swap Hibernate for EclipseLink — code unchanged.
- **Spring's `PlatformTransactionManager`**: Abstract over JDBC transactions, JPA transactions, JMS transactions — same `@Transactional` annotation, any backend.
- **AWS SDK**: `S3Client` interface hides HTTP calls, retry logic, signing, and region routing.

---

## 3. Internal Working

### How Abstract Classes Work at Machine Level

#### Java Bytecode
```
Abstract class Animal:
  ACC_ABSTRACT flag set in class file
  makeSound() has ACC_ABSTRACT flag — no bytecode body

Dog extends Animal:
  makeSound() has bytecode implementation
  Dog's vtable entry for makeSound() → Dog::makeSound() address
  
Dog d = new Dog();
Animal a = d;      // Reference typed as Animal
a.makeSound();     // invokevirtual
                   // → follow d.vptr → Dog vtable → Dog::makeSound()
                   // Abstract class can't be instantiated but can hold reference
```

#### vtable Layout for Abstract Classes
```
Animal's vtable (incomplete — abstract methods marked):
[0] → Object::equals()    (inherited)
[1] → Object::hashCode()  (inherited)
[2] → Object::toString()  (inherited)
[3] → ??? (abstract — must be filled by subclass)  ← makeSound()

Dog's vtable (complete):
[0] → Object::equals()
[1] → Object::hashCode()
[2] → Object::toString()
[3] → Dog::makeSound()    ← filled in by Dog
[4] → Dog::fetch()        ← Dog-specific method

// new Animal() → JVM: "vtable incomplete → InstantiationError"
// new Dog()    → JVM: "vtable complete → OK"
```

#### C++ Pure Virtual Function
```cpp
class Animal {
public:
    virtual void makeSound() = 0;  // = 0 → pure virtual → abstract class

    // This creates a vtable entry initialized to __cxa_pure_virtual
    // Calling it without override → program terminates
};

// At compile time:
// sizeof(Animal) includes vptr pointer (8 bytes on 64-bit)
// sizeof(Dog)    includes vptr + Animal fields + Dog fields

Dog d;
Animal& a = d;
a.makeSound();  // vptr → Dog vtable → Dog::makeSound() ✓
```

### Interface Dispatch — How It Works in JVM
```
Java Interface dispatch uses: invokeinterface (bytecode instruction)

Why different from invokevirtual?
→ invokevirtual: method offset in vtable is FIXED (same offset in every class)
→ invokeinterface: offset varies by class! A class can implement multiple
                   interfaces in any order.

JVM Interface Table (itable) per class:
Dog's itable:
  Implements Animal → offset 3 is makeSound()
  Implements Runnable → offset 4 is run()
  Implements Comparable → offset 5 is compareTo()

// invokeinterface must search/cache the correct offset
// Slightly slower than invokevirtual (~2-5ns extra)
// JVM caches the offset: first call slow, subsequent calls fast
```

### Compile-Time Abstraction: C++ Templates
```cpp
// Runtime abstraction: virtual dispatch (costs ~5ns per call)
class Sorter { virtual void sort(vector<int>& data) = 0; };
class QuickSort : public Sorter { void sort(vector<int>& data) override { /* */ } };

void sortData(Sorter& sorter, vector<int>& data) {
    sorter.sort(data); // Virtual call — JIT cannot inline!
}

// Compile-time abstraction: zero overhead!
template<typename SortStrategy>
void sortData(SortStrategy& sorter, vector<int>& data) {
    sorter.sort(data); // Resolved at compile time — inlined!
}

QuickSortImpl qs;
sortData(qs, data); // Compiled to QuickSortImpl::sort() direct call — zero overhead
```

### API vs ABI Stability (C++ FAANG Level)
```
API (Application Programming Interface): Source-level contract
  → If you change method signature, calling code fails to compile
  → Caught by developer immediately

ABI (Application Binary Interface): Binary-level contract
  → Method offsets in vtable, class size, calling conventions
  → If you add a virtual method to Animal, Dog's vtable shifts
  → Code compiled against old Animal.so + new Dog.so = silent crash

Production Rule (C++):
  → Pure virtual classes as plugin interfaces are ABI-stable!
  → They're just vtable pointers — no instance state to shift
  → This is why COM, XPCOM, and plugin SDKs use pure virtual classes
```

---

## 4. Syntax

### Java — Abstract Class vs Interface

```java
// ===== ABSTRACT CLASS =====
// Use when: Shared state, shared code, IS-A relationship, default implementations
public abstract class DataProcessor {
    // Shared state (interfaces can't have this)
    private final Logger logger = LoggerFactory.getLogger(getClass());
    private final Metrics metrics;
    private int processedCount = 0;

    protected DataProcessor(Metrics metrics) {
        this.metrics = Objects.requireNonNull(metrics);
    }

    // Template Method: shared algorithm skeleton
    public final ProcessingResult process(DataRecord record) {
        logger.debug("Processing record: {}", record.getId());
        validateInput(record);            // Hook — can override
        ProcessingResult result = doProcess(record); // Abstract — must override
        processedCount++;
        metrics.recordSuccess();
        logger.info("Processed record {} in {}ms", record.getId(), result.getDurationMs());
        return result;
    }

    // Abstract method: contract
    protected abstract ProcessingResult doProcess(DataRecord record);

    // Optional hook with default behavior — override if needed
    protected void validateInput(DataRecord record) {
        if (record == null) throw new IllegalArgumentException("Record cannot be null");
    }

    // Non-abstract: shared utility for all processors
    protected Logger getLogger() { return logger; }
    public int getProcessedCount() { return processedCount; }
}

// Concrete implementations
public class JSONDataProcessor extends DataProcessor {
    private final ObjectMapper mapper;

    public JSONDataProcessor(ObjectMapper mapper, Metrics metrics) {
        super(metrics);
        this.mapper = mapper;
    }

    @Override
    protected ProcessingResult doProcess(DataRecord record) {
        try {
            Map<String, Object> data = mapper.readValue(record.getContent(), Map.class);
            // Process JSON...
            return ProcessingResult.success(data);
        } catch (JsonProcessingException e) {
            return ProcessingResult.failure("Invalid JSON: " + e.getMessage());
        }
    }

    @Override
    protected void validateInput(DataRecord record) {
        super.validateInput(record); // Call parent validation
        if (!record.getContentType().equals("application/json"))
            throw new IllegalArgumentException("Expected JSON content");
    }
}
```

```java
// ===== INTERFACE =====
// Use when: Capability contract, multiple inheritance needed, no shared state, stable API

public interface Repository<T, ID> {
    // Abstract methods (contract)
    void save(T entity);
    Optional<T> findById(ID id);
    List<T> findAll();
    void delete(ID id);
    boolean existsById(ID id);

    // Default method (Java 8+): optional behavior, override if needed
    default long count() { return findAll().size(); }

    // Static factory method (Java 8+): utility without needing instance
    static <T, ID> Repository<T, ID> inMemory() {
        return new InMemoryRepository<>();
    }
}

// Multiple interface implementation — impossible with abstract classes
public class UserRepository implements Repository<User, UUID>, Auditable, Cacheable {
    @Override public void save(User user) { /* DB write */ }
    @Override public Optional<User> findById(UUID id) { /* DB read */ }
    // ... etc
}
```

### C++ — Pure Virtual Interface
```cpp
#include <vector>
#include <optional>
#include <string>
using namespace std;

// Pure abstract class = interface in C++
class Cache {
public:
    virtual ~Cache() = default;                           // Virtual destructor REQUIRED
    virtual optional<string> get(const string& key) = 0;
    virtual void set(const string& key, const string& value, int ttlSeconds) = 0;
    virtual bool del(const string& key) = 0;
    virtual bool exists(const string& key) = 0;
    virtual void flushAll() = 0;
};

class RedisCache : public Cache {
private:
    string host;
    int port;
    RedisClient* client;  // Hidden implementation

public:
    RedisCache(const string& host, int port)
        : host(host), port(port) {
        client = RedisClient::connect(host, port);
    }

    ~RedisCache() override { delete client; }

    optional<string> get(const string& key) override {
        auto result = client->get(key);
        if (result.is_nil()) return nullopt;
        return result.as_string();
    }

    void set(const string& key, const string& value, int ttlSeconds) override {
        client->setex(key, ttlSeconds, value);
    }

    bool del(const string& key) override { return client->del(key) > 0; }
    bool exists(const string& key) override { return client->exists(key); }
    void flushAll() override { client->flushall(); }
};

// Abstract template class (compile-time abstraction)
template<typename Hasher>
class InMemoryCache : public Cache {
private:
    unordered_map<string, string, Hasher> store;
    // Hasher is resolved at compile time — zero virtual dispatch
};
```

### Python — Abstract Base Class (ABC)
```python
from abc import ABC, abstractmethod
from typing import Optional, Generic, TypeVar
from datetime import datetime

T = TypeVar('T')

class EventStore(ABC):
    """Abstract contract for event sourcing store."""

    @abstractmethod
    def append(self, stream_id: str, events: list, expected_version: int) -> None:
        """Append events to stream. Raises ConcurrencyError if version mismatch."""
        ...

    @abstractmethod
    def load(self, stream_id: str, from_version: int = 0) -> list:
        """Load events from stream starting at given version."""
        ...

    @abstractmethod
    def subscribe(self, stream_id: str, handler) -> None:
        """Subscribe to new events on a stream."""
        ...

    # Default implementation — concrete method in abstract class
    def exists(self, stream_id: str) -> bool:
        try:
            events = self.load(stream_id, from_version=0)
            return len(events) > 0
        except StreamNotFoundError:
            return False

# Concrete implementation
class PostgresEventStore(EventStore):
    def __init__(self, connection_string: str):
        self._conn = psycopg2.connect(connection_string)  # Hidden detail

    def append(self, stream_id: str, events: list, expected_version: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(version) FROM events WHERE stream_id = %s",
                (stream_id,)
            )
            current = cur.fetchone()[0] or -1
            if current != expected_version:
                raise ConcurrencyError(f"Expected version {expected_version}, got {current}")
            for i, event in enumerate(events):
                cur.execute(
                    "INSERT INTO events (stream_id, version, data) VALUES (%s, %s, %s)",
                    (stream_id, expected_version + i + 1, json.dumps(event))
                )
        self._conn.commit()

    def load(self, stream_id: str, from_version: int = 0) -> list:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM events WHERE stream_id = %s AND version >= %s ORDER BY version",
                (stream_id, from_version)
            )
            return [json.loads(row[0]) for row in cur.fetchall()]

    def subscribe(self, stream_id: str, handler) -> None:
        # LISTEN/NOTIFY implementation
        pass
```

---

## 5. Visual Diagrams

### Abstraction Layer Diagram
```
+==========================================+
|           CLIENT CODE                    |
|   List<User> users = repo.findAll();     |
|   notifier.notify(order);               |
+==========================================+
              |           |
     =========|===========|============
     =    ABSTRACTION BARRIER         =
     =  (Interface / Abstract Class)  =
     =========|===========|============
              |           |
    +---------+   +-------+--------+
    |                              |
[MySQLRepo]              [InMemoryRepo]
[DynamoRepo]             [MongoRepo]
[JpaRepo]                [TestDoubleRepo]

Client never knows which implementation runs.
Swap implementations = zero client changes.
```

### Abstract Class vs Interface Memory/Dispatch
```
ABSTRACT CLASS:
+-[Animal object in heap]-----------+
| vptr -----> Animal vtable         |
|             [0] Object::equals()  |
|             [1] Object::hashCode()|
|             [2] Animal::eat()     | ← concrete method
|             [3] Dog::makeSound()  | ← override slot
| name: "Rex"                       | ← shared state
| age: 3                            | ← shared state
+-----------------------------------+

INTERFACE (Java itable):
+-[Dog object in heap]---------------+
| vptr -----> Dog vtable             |
| itable_ptr → [Animal methods]      |
|              → [Runnable methods]  |
|              → [Serializable]      |
| name: "Rex"                        |
| breed: "Lab"                       |
+------------------------------------+
Multiple interface tables per object;
invokeinterface searches the correct one
```

### Abstraction Enables Dependency Inversion
```
WITHOUT DIP (bad):                    WITH DIP (good):

OrderService                          OrderService
    |                                     |
    ↓ depends on                         ↓ depends on
MySQLRepo (concrete)              OrderRepository (interface)
    ↑ (cannot swap)                       ↑
                                MySQLRepo  MongoRepo  InMemoryRepo
                                (choose at runtime via injection)
```

---

## 6. Real World Analogy

### Electrical Power Socket
- **Abstraction**: The wall socket (two holes, standard voltage/current spec)
- **Implementation**: Coal plant, nuclear plant, solar array, wind turbines — anything that generates the right voltage
- **Client**: Your laptop charger — works in any country with the right adapter
- **Benefit**: Switched from coal to solar — no change to your laptop

### Restaurant Menu
- **Abstraction**: The menu (list of dishes with descriptions and prices)
- **Implementation**: Kitchen recipes, cooking techniques, chef's skill, ingredient sourcing
- **Client**: Customer — orders "Pasta Carbonara" without knowing cooking process
- **Swap**: New chef, new recipe, new suppliers → dish still called "Pasta Carbonara" → client unaffected

### ATM Interface
- **Abstraction**: Insert card, enter PIN, select amount, collect cash
- **Implementation**: Mainframe communication, ledger debiting, Bill Counter Mechanism, network encryption
- **Client**: Customer using the interface
- **Swap**: Bank switches from COBOL mainframe to modern cloud system → ATM interface unchanged

### Java JDBC
- **Abstraction**: `Connection`, `Statement`, `ResultSet` interfaces (JDBC API)
- **Implementations**: MySQL Connector/J, PostgreSQL JDBC, H2 (in-memory), Oracle JDBC
- **Client**: Your DAO class — uses `Connection.createStatement()`, never `MySQLConnection`
- **Swap**: Change one line in config: `jdbc:mysql://` → `jdbc:postgresql://` — zero code change in DAO

---

## 7. Interview Explanation

### 30 Seconds
> "Abstraction is the OOP pillar that hides implementation details and exposes only the essential interface to callers. It's achieved through abstract classes and interfaces. The key benefit: clients depend on the abstraction, not the concrete implementation, so implementations can be swapped — different databases, different email providers — without changing the calling code."

### 1 Minute
> "Abstraction is about designing stable contracts that clients depend on, while implementations remain free to change. In Java, interfaces represent a pure contract — just what operations are available. Abstract classes add shared code and state on top. The critical distinction from Encapsulation: Encapsulation hides and protects STATE within an object; Abstraction hides IMPLEMENTATION COMPLEXITY between layers. They work together: a class can encapsulate its state AND be abstracted behind an interface.
>
> The power shows in testing: if `OrderService` depends on `OrderRepository` (interface), I can inject `InMemoryOrderRepository` in tests — no real database needed. And if I switch from MySQL to DynamoDB in production, `OrderService` is unchanged."

### 3 Minutes
> "At a production level, abstraction design is where senior engineers earn their keep. Let me go deep.
>
> At the language level: abstract classes and interfaces compile to vtable-based dispatch. When you call a method on an interface reference in Java, the JVM uses `invokeinterface` bytecode — it searches the object's interface table (itable) for the right method offset, which is slightly slower than `invokevirtual` because a class can implement multiple interfaces at varying offsets. After the first call, this offset is cached, so subsequent calls are near-`invokevirtual` speed.
>
> In C++, pure virtual functions are the interface mechanism. Pure virtual classes are critical for plugin architectures because they're ABI-stable — they're just a vtable pointer, no state. Concrete C++ classes have unstable ABIs (adding a virtual function shifts the entire vtable). So cross-library boundaries always use pure virtual class interfaces in production C++ codebases.
>
> The Dependency Inversion Principle (DIP) is abstraction in action: high-level modules (OrderService) depend on abstractions (OrderRepository), not low-level modules (MySQLOrderRepository). Low-level modules implement the abstraction. This is what makes Spring IoC work — the container injects the right concrete implementation at runtime.
>
> The failure mode to know: Leaky Abstractions. A leaky abstraction is when implementation details bleed through the interface. Classic example: JDBC's `SQLException` — exposes SQL-specific error codes even through an abstraction layer. Good abstraction design means your interface throws domain exceptions (`RepositoryException`), not infrastructure ones.
>
> And there's a cost trade-off: compile-time abstraction (C++ templates, Java generics) resolves at compile time — zero runtime overhead, but larger binaries. Runtime abstraction (interfaces, virtual dispatch) adds vtable lookup overhead (~5ns), but enables runtime swapping, which is what makes DI, mocking, and plugin systems possible."

---

## 8. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What is Abstraction in OOP? | Hiding implementation details and exposing only the essential interface. Clients depend on contract, not implementation. |
| Abstraction vs Encapsulation? | Encapsulation = hiding STATE (access modifiers); Abstraction = hiding IMPLEMENTATION COMPLEXITY (interfaces/abstract classes) |
| Abstract class vs Interface? | Abstract class: can have state, constructors, concrete methods, single inheritance. Interface: purely contract, multiple implementation, default methods (Java 8+) |
| Can you instantiate an abstract class? | No — it has abstract methods without implementations. Only concrete subclasses can be instantiated. |
| What is "programming to an interface"? | Declare variables/parameters as interface types, not concrete types: `List<String>` not `ArrayList<String>` |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| When abstract class, when interface? | Abstract class: IS-A + shared state/code. Interface: CAN-DO + multiple inheritance + stable contract across unrelated classes |
| What is a Leaky Abstraction? | When implementation details escape through the interface. JDBC's SQLExceptions leaking SQL error codes is the classic example. |
| Can abstract classes have constructors? | Yes. Subclass calls `super(args)`. Constructor initializes shared fields declared in the abstract class. |
| What are Java 8 default methods for? | Allow adding new methods to existing interfaces without breaking all implementing classes — backward-compatible API evolution |
| What is the Dependency Inversion Principle (DIP)? | High-level modules depend on abstractions; low-level modules implement abstractions. Both depend on the abstract layer. |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| Explain `invokeinterface` vs `invokevirtual` at JVM bytecode level | `invokevirtual`: fixed vtable offset per method (fast). `invokeinterface`: offset varies per class implementing the interface; JVM searches/caches offset (slightly slower) |
| What is ABI stability and how do pure virtual classes achieve it? | ABI = binary-level contract. C++ concrete classes have unstable ABIs (adding virtual fn shifts vtable). Pure virtual = only vtable pointer — no state, stable layout across compiles |
| What is structural subtyping vs nominal subtyping for abstraction? | Nominal: must declare `implements Interface` (Java, C#). Structural: if object has matching methods, it's compatible — "duck typing" (Go interfaces, Python Protocol) |
| How does JIT handle interface dispatch performance? | Inline cache: JVM tracks which concrete type appears at each call site. Monomorphic (one type) → devirtualize + inline. Polymorphic (few types) → dispatch table. Megamorphic → full invokeinterface lookup |
| What is the Open-Closed Principle and how does abstraction enable it? | Open for extension, closed for modification. Interfaces let you add new implementations without changing existing calling code. |

### 💼 Google Level
> *"You're designing the storage layer abstraction for Bigtable. What properties must the abstraction expose, which must it hide, and what happens when the abstraction must leak for performance (like enforcing row-key design for hot spot avoidance)?"*

Expected: Discuss storage contract (read/write/scan), hiding node routing, compaction, load balancing. Acknowledge forced leak: clients MUST understand row-key design for performance — that's an intentional leaky abstraction to enable locality-aware access patterns. Trade-off: clean abstraction vs performance-critical client awareness.

### 💼 Goldman Sachs Level
> *"Design the abstraction layer for a pricing engine that must support Equities, Options, Fixed Income, and Derivatives — with zero if-else statements and extensible to new instrument types."*

Expected: `PricingModel` interface with `Price calculateFairValue(MarketData market, InstrumentSpec spec)`. Each instrument type implements its pricing model (Black-Scholes for options, DCF for bonds, etc.). `PricingEngine` depends on `Map<InstrumentType, PricingModel>` — register new types without touching engine. Polymorphism = zero if-else.

---

## 9. Coding Examples

### Basic — Building a Notification System
```java
// === ABSTRACTION LAYER ===
public interface NotificationSender {
    void send(Notification notification);
    boolean supports(NotificationChannel channel);
}

public record Notification(
    String recipient,
    String subject,
    String body,
    NotificationChannel channel,
    Map<String, String> metadata
) {}

// === CONCRETE IMPLEMENTATIONS ===
public class EmailSender implements NotificationSender {
    private final JavaMailSender mailSender;
    private final TemplateEngine templates;

    public EmailSender(JavaMailSender mailSender, TemplateEngine templates) {
        this.mailSender = mailSender;
        this.templates = templates;
    }

    @Override
    public void send(Notification notification) {
        MimeMessage message = mailSender.createMimeMessage();
        MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");
        helper.setTo(notification.recipient());
        helper.setSubject(notification.subject());
        helper.setText(templates.process(notification.body(), notification.metadata()), true);
        mailSender.send(message);
    }

    @Override
    public boolean supports(NotificationChannel channel) {
        return channel == NotificationChannel.EMAIL;
    }
}

public class SMSSender implements NotificationSender {
    private final TwilioClient twilio;

    @Override
    public void send(Notification notification) {
        twilio.messages.create(
            notification.recipient(),
            new MessageCreator(from, notification.body())
        );
    }

    @Override
    public boolean supports(NotificationChannel channel) { return channel == NotificationChannel.SMS; }
}

public class PushNotificationSender implements NotificationSender {
    private final FCMClient fcm;

    @Override
    public void send(Notification notification) {
        Message msg = Message.builder()
            .setToken(notification.recipient())
            .setNotification(com.google.firebase.messaging.Notification.builder()
                .setTitle(notification.subject())
                .setBody(notification.body())
                .build())
            .build();
        fcm.send(msg);
    }

    @Override
    public boolean supports(NotificationChannel channel) { return channel == NotificationChannel.PUSH; }
}

// === CLIENT CODE: Depends only on abstraction ===
@Service
public class NotificationService {
    private final List<NotificationSender> senders; // All behind interface!

    public NotificationService(List<NotificationSender> senders) {
        this.senders = senders;
    }

    public void send(Notification notification) {
        senders.stream()
            .filter(s -> s.supports(notification.channel()))
            .findFirst()
            .orElseThrow(() -> new UnsupportedChannelException(notification.channel()))
            .send(notification);
    }
}
// Adding WhatsApp? Create WhatsAppSender implementing NotificationSender.
// Zero changes to NotificationService!
```

### Intermediate — Abstract Class + Template Method
```java
// Report generation: fixed algorithm skeleton, customizable steps
public abstract class ReportGenerator {
    // Template Method — final: algorithm is fixed, steps are customizable
    public final Report generate(ReportRequest request) {
        List<ReportRow> rawData = fetchData(request);       // Hook: customizable
        List<ReportRow> filtered = applyFilters(rawData, request.getFilters()); // Shared
        List<ReportRow> formatted = format(filtered);        // Hook: customizable
        String output = render(formatted);                   // Abstract: must implement
        ReportMetadata metadata = buildMetadata(request, formatted.size()); // Shared
        return new Report(output, metadata);
    }

    // Shared implementation
    private List<ReportRow> applyFilters(List<ReportRow> data, List<Filter> filters) {
        return data.stream()
            .filter(row -> filters.stream().allMatch(f -> f.test(row)))
            .collect(Collectors.toList());
    }

    private ReportMetadata buildMetadata(ReportRequest request, int rowCount) {
        return new ReportMetadata(request.getTitle(), rowCount, Instant.now());
    }

    // Hooks with default implementations (override if needed)
    protected List<ReportRow> fetchData(ReportRequest request) {
        return getDefaultDataSource().query(request.getQuery());
    }

    protected List<ReportRow> format(List<ReportRow> data) { return data; } // Default: no-op

    // Abstract: subclass MUST implement
    protected abstract String render(List<ReportRow> data);
    protected abstract DataSource getDefaultDataSource();
}

// PDF Report
public class PDFReportGenerator extends ReportGenerator {
    private final PDFWriter pdfWriter;
    private final DataSource reportDB;

    @Override
    protected String render(List<ReportRow> data) {
        Document doc = new Document();
        PdfWriter.getInstance(doc, outputStream);
        doc.open();
        // Build PDF table from data
        doc.close();
        return outputStream.toString("ISO-8859-1");
    }

    @Override
    protected DataSource getDefaultDataSource() { return reportDB; }
}

// Excel Report
public class ExcelReportGenerator extends ReportGenerator {
    private final DataSource warehouse;

    @Override
    protected String render(List<ReportRow> data) {
        Workbook wb = new XSSFWorkbook();
        Sheet sheet = wb.createSheet("Report");
        // Build Excel rows
        return encode(wb);
    }

    @Override
    protected List<ReportRow> format(List<ReportRow> data) {
        // Excel-specific: format dates and currencies
        return data.stream().map(this::applyExcelFormatting).collect(Collectors.toList());
    }

    @Override
    protected DataSource getDefaultDataSource() { return warehouse; }
}
```

### Advanced — Abstraction for Testability
```java
// === Infrastructure Interfaces ===
public interface UserRepository {
    void save(User user);
    Optional<User> findById(UUID id);
    Optional<User> findByEmail(String email);
    boolean existsByEmail(String email);
}

public interface PasswordHasher {
    String hash(String plaintext);
    boolean verify(String plaintext, String hashed);
}

public interface TokenGenerator {
    String generateJWT(User user, Duration expiry);
    Optional<UUID> extractUserId(String token);
}

// === Service: depends ONLY on abstractions ===
public class AuthService {
    private final UserRepository users;
    private final PasswordHasher hasher;
    private final TokenGenerator tokens;

    public AuthService(UserRepository users, PasswordHasher hasher, TokenGenerator tokens) {
        this.users = users;
        this.hasher = hasher;
        this.tokens = tokens;
    }

    public AuthResult register(RegisterRequest request) {
        if (users.existsByEmail(request.email()))
            return AuthResult.failure("Email already registered");
        String hash = hasher.hash(request.password());
        User user = new User(UUID.randomUUID(), request.email(), hash, Instant.now());
        users.save(user);
        String token = tokens.generateJWT(user, Duration.ofHours(24));
        return AuthResult.success(token, user.getId());
    }

    public AuthResult login(LoginRequest request) {
        return users.findByEmail(request.email())
            .filter(u -> hasher.verify(request.password(), u.getPasswordHash()))
            .map(u -> AuthResult.success(tokens.generateJWT(u, Duration.ofHours(8)), u.getId()))
            .orElse(AuthResult.failure("Invalid credentials"));
    }
}

// === Test: inject fakes — no real DB, no real hashing, no real JWT library! ===
@Test
class AuthServiceTest {
    private final InMemoryUserRepository users = new InMemoryUserRepository();
    private final FakePasswordHasher hasher = new FakePasswordHasher();
    private final FakeTokenGenerator tokens = new FakeTokenGenerator();
    private final AuthService authService = new AuthService(users, hasher, tokens);

    @Test
    void register_withDuplicateEmail_fails() {
        users.save(existingUser("alice@example.com"));
        AuthResult result = authService.register(new RegisterRequest("alice@example.com", "pass"));
        assertThat(result.isSuccess()).isFalse();
        assertThat(result.getError()).contains("already registered");
    }

    @Test
    void login_withValidCredentials_returnsToken() {
        authService.register(new RegisterRequest("bob@example.com", "secret123"));
        AuthResult result = authService.login(new LoginRequest("bob@example.com", "secret123"));
        assertThat(result.isSuccess()).isTrue();
        assertThat(result.getToken()).isNotBlank();
    }
}
```

---

## 10. Common Mistakes

### ⚠️ Mistake 1: Abstraction for Its Own Sake (Over-Engineering)
```java
// WRONG: Creating an interface that will NEVER have a second implementation
public interface UserService {
    User getUser(String id);
    void createUser(User user);
}

public class UserServiceImpl implements UserService {
    public User getUser(String id) { /* ... */ }
    public void createUser(User user) { /* ... */ }
}
// If there's only ever one UserServiceImpl, the interface is pure boilerplate.
// Add the interface WHEN you have a second implementation or need to mock for tests.

// CORRECT: Concrete class directly — add interface when actually needed
public class UserService {
    public User getUser(String id) { /* ... */ }
    public void createUser(User user) { /* ... */ }
}
```

### ⚠️ Mistake 2: Leaky Abstraction
```java
// WRONG: Repository interface leaks database-specific concern
public interface OrderRepository {
    void save(Order order) throws SQLException;              // LEAKED: SQL concern!
    List<Order> findByStatus(String status, int limit, int offset); // LEAKED: SQL paging!
}

// CORRECT: Domain-level abstraction
public interface OrderRepository {
    void save(Order order);  // Implementation decides how to handle failures
    Page<Order> findByStatus(OrderStatus status, PageRequest pageRequest);
    // Domain types, domain pagination — no SQL leakage
}
```

### ⚠️ Mistake 3: Fat Interface (ISP Violation)
```java
// WRONG: Forces every repository to implement every operation
public interface DataStore {
    void create(Object entity);
    Object read(String id);
    void update(Object entity);
    void delete(String id);
    List<Object> findAll();
    List<Object> findByFilter(Map<String, Object> filter);
    void bulkInsert(List<Object> entities);       // Not all stores support bulk!
    void transactional(Runnable operation);        // Not all stores support transactions!
    byte[] exportToCSV();                          // Not a store concern!
}

// CORRECT: Segregated interfaces
public interface Readable<T, ID> { Optional<T> findById(ID id); List<T> findAll(); }
public interface Writable<T> { void save(T entity); void delete(T entity); }
public interface Queryable<T> { List<T> findByFilter(QuerySpec spec); }
public interface Transactional { void executeInTransaction(Runnable unit); }
```

### ⚠️ Mistake 4: Using Abstract Class Where Interface Fits Better
```java
// WRONG: Forces single inheritance, prevents User from extending anything else
public abstract class Serializable {
    public abstract String serialize();
    public abstract void deserialize(String data);
}

public class User extends Serializable { /* now can't extend BaseEntity */ }

// CORRECT: Interface allows multiple implementation
public interface Serializable {
    String serialize();
    void deserialize(String data);
}

public class User extends BaseEntity implements Serializable, Auditable { /* flexible! */ }
```

### ⚠️ Mistake 5: Returning Concrete Type from Abstract Method
```java
// WRONG: Leaks concrete type through abstraction
public interface MessageQueue {
    KafkaMessage receive(); // Exposes Kafka-specific type to all callers!
}

// CORRECT: Domain-level type
public interface MessageQueue {
    Message receive(); // Callers only know Message, not Kafka
}
```

---

## 11. Best Practices

### Design
- **Interface first**: Design the contract before writing the implementation
- **Program to interfaces**: `List<T>`, `Map<K,V>` in declarations, not `ArrayList`, `HashMap`
- **Small, focused interfaces**: One cohesive capability per interface (ISP)
- **Domain exception**: Throw `RepositoryException` not `SQLException` from repository interface
- **No `instanceof` through abstraction**: If you need instanceof, the abstraction is probably wrong

### Abstract Class vs Interface Decision
```
Use INTERFACE when:
  ✓ Pure behavioral contract with no shared state
  ✓ Unrelated classes need to implement (Comparable, Serializable)
  ✓ Multiple inheritance needed
  ✓ Stable API that must not break existing implementations

Use ABSTRACT CLASS when:
  ✓ Shared state (fields) across all implementations
  ✓ Template Method pattern with shared algorithm skeleton
  ✓ IS-A relationship where all subclasses truly are the base type
  ✓ Protected methods as hooks for subclass customization
```

### 🚀 Performance Notes
- `invokevirtual` (abstract class method): ~5ns, JIT can inline if monomorphic
- `invokeinterface` (interface method): ~5-15ns first call, ~5ns after inline cache warms
- C++ template abstraction: 0ns — resolved at compile time
- Both are negligible vs I/O operations (~100μs DB, ~50ms network)

---

## 12. Complexity

| Mechanism | Call Cost | JIT Optimization | Memory |
|-----------|-----------|-----------------|--------|
| Interface call (`invokeinterface`) | ~5-15ns | Inline cache → ~5ns after warmup | itable per class (extra memory) |
| Abstract class (`invokevirtual`) | ~5-10ns | Devirtualize if monomorphic → 0ns | Standard vtable |
| C++ pure virtual | ~3-5ns | No JIT; branch predictor helps | vtable per class |
| C++ template (compile-time) | ~0ns | Fully inlined | Larger binary (code bloat) |
| Java Generic (type erasure) | ~0ns for dispatch | Same as concrete type | No extra at runtime |

---

## 13. Advantages

| Advantage | Concrete Benefit |
|-----------|-----------------|
| **Implementation independence** | Swap MySQL → DynamoDB → H2 (tests) without touching `OrderService` |
| **Testability** | Inject fakes/mocks behind interfaces — no real infrastructure in unit tests |
| **Extensibility** | New implementations don't change existing callers (OCP) |
| **Parallel development** | Team A codes to interface; Team B implements it — simultaneous |
| **Versioned evolution** | Change abstract class: add default methods; calling code unchanged |
| **Dependency Inversion** | High-level modules depend on abstractions — stable even as details change |

---

## 14. Disadvantages

| Disadvantage | When It Hurts |
|-------------|---------------|
| **Premature abstraction** | Designing for extensions that never come — YAGNI violation |
| **Indirection overhead** | Jumping through interface → harder to trace code in IDE |
| **Leaky abstractions** | All abstractions fail at some point; choose abstractions wisely |
| **Interface bloat** | Too many small interfaces = hard to navigate |
| **Performance cost** | `invokeinterface` slightly slower than direct call (JIT mitigates) |
| **Serialization complexity** | Serializing through interfaces requires type metadata |

---

## 15. Comparison Table

### Abstract Class vs Interface

| Feature | Abstract Class | Interface |
|---------|---------------|-----------|
| **Instance variables** | Yes | No (only `static final`) |
| **Constructors** | Yes (called via `super()`) | No |
| **Concrete methods** | Yes | Default/static (Java 8+) only |
| **Multiple inheritance** | No (single) | Yes (multiple) |
| **Access modifiers** | Any (`protected`, `private`) | `public` only |
| **IS-A relationship** | Yes | CAN-DO (capability) |
| **Use case** | Shared state + behavior | Pure contract across hierarchy |
| **When to use** | Template Method, shared code | DIP, multiple capability |

### Abstraction vs Encapsulation

| Aspect | Abstraction | Encapsulation |
|--------|-------------|---------------|
| **Question answered** | "What can I do?" | "Who can access my state?" |
| **Hides** | Implementation complexity | Internal data/state |
| **Mechanism** | Interfaces, abstract classes | Access modifiers |
| **Level** | Between components | Within one component |
| **Goal** | Decoupled callers | Invariant protection |
| **Example** | `PaymentGateway` interface | `private double balance;` |

---

## 16. Design Pattern Connection

| Pattern | Abstraction Role |
|---------|-----------------|
| **Strategy** | Abstract algorithm interface; swap implementations at runtime |
| **Factory Method** | Abstract object creation; concrete factories decide what to instantiate |
| **Abstract Factory** | Family of related abstract factories |
| **Proxy** | Proxy implements same interface as real subject; transparent substitution |
| **Decorator** | Wraps same interface; adds behavior without changing callers |
| **Adapter** | Makes incompatible interface compatible behind target interface |
| **Bridge** | Separates abstraction hierarchy from implementation hierarchy |
| **Template Method** | Abstract class defines algorithm skeleton; abstract methods are extension points |
| **Repository** | Abstract data access — technology-agnostic for domain layer |
| **Command** | Abstract command interface; execute() is polymorphic |

---

## 17. System Design Connection

### Microservices — Abstraction at Service Level
```
Internal implementation (hidden):
  OrderService: PostgreSQL for data, Redis for cache, Kafka for events

Public abstraction (exposed):
  REST API: POST /orders, GET /orders/{id}, DELETE /orders/{id}
  
Clients (other services) only know the REST contract.
Swap PostgreSQL → CockroachDB? REST API unchanged. Clients unaffected.
```

### API Gateway
```
Client → [API Gateway] → Service A (hidden)
                      → Service B (hidden)
                      → Service C (hidden)

API Gateway IS the abstraction layer.
It hides: service addresses, authentication, rate limiting, load balancing.
Clients have one stable endpoint.
```

### Plugin Architecture
```
Core App ──> [Plugin Interface]
              ↑         ↑         ↑
         Plugin A   Plugin B   Plugin C  (loaded at runtime)

Abstraction enables runtime loading of unknown implementations.
Used by: IntelliJ, Eclipse, WordPress, Chrome Extensions.
```

### Event-Driven Systems
```
Producer → [Message (abstraction)] → [Broker]
                                         ↓
Consumer A (processes as-needed)
Consumer B (processes as-needed)

Message is the abstraction: producer and consumers never know about each other.
Swap RabbitMQ → Kafka: change broker, not producers or consumers.
```

---

## 18. Multithreading Connection

### Abstraction Enables Thread-Safe Swap
```java
// Interface allows thread-safe implementations to be swapped in
public interface Counter {
    void increment();
    long get();
}

// Not thread-safe — single-threaded use
public class SimpleCounter implements Counter {
    private long count = 0;
    public void increment() { count++; }
    public long get() { return count; }
}

// Thread-safe — concurrent use
public class AtomicCounter implements Counter {
    private final AtomicLong count = new AtomicLong(0);
    public void increment() { count.incrementAndGet(); }
    public long get() { return count.get(); }
}

// Distributed — across JVMs
public class RedisCounter implements Counter {
    private final RedisTemplate redis;
    private final String key;
    public void increment() { redis.opsForValue().increment(key); }
    public long get() { return (Long) redis.opsForValue().get(key); }
}

// Client code never changes when switching implementations:
Counter counter = new AtomicCounter(); // Thread-safe in same JVM
// Counter counter = new RedisCounter(redis, "global:count"); // Cross-JVM
```

### Abstract Class Thread-Safety Consideration
```java
public abstract class ThreadSafeProcessor {
    // Shared state declared in abstract class → must synchronize
    private final AtomicLong processedCount = new AtomicLong(0);
    private final Object lock = new Object();

    public final void process(Event event) {
        ProcessingResult result = doProcess(event); // Subclass implementation
        processedCount.incrementAndGet();           // Thread-safe increment
        if (result.shouldCache()) {
            synchronized (lock) {
                updateCache(event, result); // Protected shared state
            }
        }
    }

    protected abstract ProcessingResult doProcess(Event event); // Subclass fills this
    private void updateCache(Event event, ProcessingResult result) { /* ... */ }
    public long getProcessedCount() { return processedCount.get(); }
}
```

---

## 19. Company Interview Perspective

### Google
- "How does abstraction enable backward-compatible API evolution? How did you evolve JDBC or gRPC APIs?"
- Protocol Buffers: how does the Protobuf schema serve as an abstraction layer for cross-language communication?
- Compile-time abstraction: when would you use C++ template abstraction vs runtime interface abstraction?

### Goldman Sachs
- "Design an abstract pricing engine that handles equities, derivatives, and fixed income — extensible to new asset classes with zero if-else"
- FIX Protocol as abstraction layer for order routing
- Event sourcing: `EventStore` abstraction and its failure modes (concurrent writes)

### Amazon
- "How does DynamoDB abstract over its distributed storage and index structures? What leaks through?"
- AWS SDK client interface vs regional endpoint management
- Why is `AmazonS3` interface critical for localstack-based testing?

### Microsoft
- C++ COM as the ultimate ABI-stable abstraction layer: why pure virtual classes?
- .NET `IEnumerable<T>` and deferred execution — how abstraction enables lazy evaluation
- LINQ's `IQueryable<T>` vs `IEnumerable<T>` — abstraction that leaks the execution model

### Meta
- Python Protocol (structural subtyping) vs ABC (nominal): when does each apply?
- React component as abstraction: props = interface, implementation = render tree
- GraphQL schema as abstraction layer hiding multiple backend data sources

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | ⚠️ Can an interface extend another interface? | Yes — interfaces can extend multiple interfaces. Class implementing child must implement all methods from entire hierarchy. |
| 2 | Can abstract classes have static methods? | Yes. Static methods don't participate in polymorphism — they're resolved at compile time. |
| 3 | ⚠️ Java 8 default method conflict — two interfaces with same default method | Compile error. Implementing class must override and explicitly choose or define its own. |
| 4 | What is a Marker Interface? | Interface with no methods (e.g., `Serializable`, `Cloneable`). Used for type-tagging. Modern alternative: annotations. |
| 5 | ⚠️ Can you use `instanceof` with interfaces? | Yes. `obj instanceof Serializable` checks if object's class implements Serializable. |
| 6 | What is the difference between an interface and an abstract class in terms of when to use? | Interface: CAN-DO capability across unrelated classes. Abstract class: IS-A with shared state/code in related hierarchy. |
| 7 | ⚠️ What is a Leaky Abstraction? Give a real example | TCP abstracts network unreliability as reliable stream. But when network drops, your application hangs — TCP's complexity leaks through. |
| 8 | Can an abstract class implement an interface without implementing all methods? | Yes — abstract class can leave interface methods abstract for concrete subclasses to implement. |
| 9 | ⚠️ What is structural subtyping? | Type compatibility based on shape (methods) rather than declared type hierarchy. Python `Protocol`, Go interfaces. "If it walks like a duck..." |
| 10 | What is the difference between Abstraction and Generalization? | Generalization = finding common base type among related types (Inheritance). Abstraction = hiding implementation behind a contract (Interfaces). |
| 11 | ⚠️ What happens if an abstract class doesn't declare any abstract methods? | Allowed — it just prevents instantiation. Can still use as base class for shared code. |
| 12 | How does Java `Optional` relate to Abstraction? | `Optional<T>` abstracts the concept of "value might not be present" — callers handle absence via API, not null-checking. |
| 13 | What is the difference between `invokeinterface` and `invokevirtual` JVM bytecodes? | `invokevirtual` uses fixed vtable offset per method. `invokeinterface` searches itable (slower first call, cached after). |
| 14 | ⚠️ When should you NOT abstract? | When there's only one implementation and no plan for more; when testing doesn't require mocking; when abstraction would leak anyway. YAGNI. |
| 15 | What is API surface area and why minimize it? | Number of public methods in your abstraction. Fewer = smaller contract = easier to change internals = less for callers to depend on. |

---

## 21. Coding Problems

### Easy — Implement a Cache Abstraction
```java
// Design a cache abstraction with multiple implementations
public interface Cache<K, V> {
    Optional<V> get(K key);
    void put(K key, V value, Duration ttl);
    void evict(K key);
    boolean containsKey(K key);
    void clear();
}

// In-memory (for development/tests)
public class InMemoryCache<K, V> implements Cache<K, V> {
    private final Map<K, CacheEntry<V>> store = new ConcurrentHashMap<>();

    @Override
    public Optional<V> get(K key) {
        CacheEntry<V> entry = store.get(key);
        if (entry == null || entry.isExpired()) {
            store.remove(key);
            return Optional.empty();
        }
        return Optional.of(entry.getValue());
    }

    @Override
    public void put(K key, V value, Duration ttl) {
        store.put(key, new CacheEntry<>(value, Instant.now().plus(ttl)));
    }

    @Override public void evict(K key) { store.remove(key); }
    @Override public boolean containsKey(K key) { return get(key).isPresent(); }
    @Override public void clear() { store.clear(); }

    private record CacheEntry<V>(V value, Instant expiresAt) {
        boolean isExpired() { return Instant.now().isAfter(expiresAt); }
    }
}

// Redis (for production)
public class RedisCache<K, V> implements Cache<K, V> {
    private final RedisTemplate<K, V> redis;

    @Override
    public Optional<V> get(K key) {
        return Optional.ofNullable(redis.opsForValue().get(key));
    }

    @Override
    public void put(K key, V value, Duration ttl) {
        redis.opsForValue().set(key, value, ttl);
    }

    @Override public void evict(K key) { redis.delete(key); }
    @Override public boolean containsKey(K key) { return Boolean.TRUE.equals(redis.hasKey(key)); }
    @Override public void clear() { redis.execute(conn -> conn.serverCommands().flushDb()); }
}
```

### Medium — Design a Payment Gateway Abstraction
```java
// Payment gateway abstraction for multiple providers
public interface PaymentGateway {
    ChargeResult charge(ChargeRequest request);
    RefundResult refund(RefundRequest request);
    boolean supports(Currency currency);
    String getProviderName();
}

public record ChargeRequest(
    Money amount,
    String customerId,
    String idempotencyKey,
    Map<String, String> metadata
) {}

public record ChargeResult(
    String transactionId,
    ChargeStatus status,
    Optional<String> failureReason,
    Instant processedAt
) {
    public static ChargeResult success(String txnId) {
        return new ChargeResult(txnId, ChargeStatus.SUCCESS, Optional.empty(), Instant.now());
    }
    public static ChargeResult failed(String reason) {
        return new ChargeResult(null, ChargeStatus.FAILED, Optional.of(reason), Instant.now());
    }
}

public class StripeGateway implements PaymentGateway {
    private final StripeClient client;
    private final RateLimiter rateLimiter;
    private final CircuitBreaker circuitBreaker;

    @Override
    public ChargeResult charge(ChargeRequest request) {
        rateLimiter.acquire();
        return circuitBreaker.run(() -> {
            PaymentIntentCreateParams params = PaymentIntentCreateParams.builder()
                .setAmount(request.amount().getCents())
                .setCurrency(request.amount().getCurrency().code())
                .setCustomer(request.customerId())
                .setIdempotencyKey(request.idempotencyKey())
                .build();
            PaymentIntent intent = PaymentIntent.create(params);
            return ChargeResult.success(intent.getId());
        }, e -> ChargeResult.failed("Stripe error: " + e.getMessage()));
    }

    @Override public boolean supports(Currency currency) {
        return Set.of(Currency.USD, Currency.EUR, Currency.GBP).contains(currency);
    }

    @Override public String getProviderName() { return "stripe"; }
    // refund() implementation...
}
```

### Hard — Abstract Event Sourcing Store
```java
public interface EventStore {
    // Append events atomically; throw OptimisticConcurrencyException if wrong version
    void append(StreamId streamId, List<DomainEvent> events, long expectedVersion);

    // Load all events for an aggregate
    List<DomainEvent> load(StreamId streamId);

    // Load events from a specific version
    List<DomainEvent> load(StreamId streamId, long fromVersion);

    // Subscribe to all events on a stream (for projections)
    Subscription subscribe(StreamId streamId, EventHandler handler);

    // Check if stream exists
    boolean exists(StreamId streamId);

    // Get current version without loading events
    long currentVersion(StreamId streamId);
}

// Domain aggregate uses the abstraction
public class OrderAggregate {
    private String orderId;
    private OrderStatus status;
    private Money total;
    private long version = -1;
    private final List<DomainEvent> uncommittedEvents = new ArrayList<>();

    private OrderAggregate() {}

    // Reconstitute from events
    public static OrderAggregate reconstitute(List<DomainEvent> events) {
        OrderAggregate order = new OrderAggregate();
        events.forEach(order::apply);
        order.uncommittedEvents.clear();
        return order;
    }

    public void placeOrder(String orderId, List<OrderItem> items) {
        if (this.orderId != null) throw new IllegalStateException("Order already placed");
        apply(new OrderPlaced(orderId, items, Instant.now()));
    }

    public void confirm() {
        if (status != OrderStatus.PENDING) throw new IllegalStateException();
        apply(new OrderConfirmed(orderId, Instant.now()));
    }

    private void apply(DomainEvent event) {
        if (event instanceof OrderPlaced placed) {
            this.orderId = placed.orderId();
            this.status = OrderStatus.PENDING;
            this.total = calculateTotal(placed.items());
        } else if (event instanceof OrderConfirmed) {
            this.status = OrderStatus.CONFIRMED;
        }
        this.version++;
        this.uncommittedEvents.add(event);
    }

    public List<DomainEvent> getUncommittedEvents() { return List.copyOf(uncommittedEvents); }
    public void clearUncommittedEvents() { uncommittedEvents.clear(); }
    public long getVersion() { return version; }
}

// Repository uses EventStore abstraction
public class OrderRepository {
    private final EventStore store; // ABSTRACTED — works with Postgres, EventStoreDB, InMemory

    public void save(OrderAggregate order) {
        store.append(
            new StreamId("order-" + order.getOrderId()),
            order.getUncommittedEvents(),
            order.getVersion() - order.getUncommittedEvents().size()
        );
        order.clearUncommittedEvents();
    }

    public Optional<OrderAggregate> findById(String orderId) {
        StreamId stream = new StreamId("order-" + orderId);
        if (!store.exists(stream)) return Optional.empty();
        List<DomainEvent> events = store.load(stream);
        return Optional.of(OrderAggregate.reconstitute(events));
    }
}
```

---

## 22. Revision Sheet

| Concept | Key Rule |
|---------|----------|
| Abstraction | Hide implementation details; expose only the contract (interface) |
| Abstract class | Can have state, constructors, concrete methods. IS-A. Single inheritance. |
| Interface | Pure contract. No state. Multiple implementation. CAN-DO. |
| DIP | High-level depends on abstraction; low-level implements it |
| Leaky abstraction | Implementation details escape through the interface |
| ISP | One focused capability per interface — no fat interfaces |
| `invokeinterface` | Slightly slower than `invokevirtual` — interface table search |
| Template Method | Abstract class defines algorithm skeleton; abstract methods are hooks |
| Program to interface | Declare as interface type, not concrete: `List<T>` not `ArrayList<T>` |
| YAGNI | Don't create abstractions for extensions that don't exist yet |

### Decision Matrix: Interface vs Abstract Class
```
Need multiple inheritance?                    → Interface
No shared state across implementations?       → Interface
Pure behavioral capability (CAN-DO)?          → Interface
Need shared fields/constructors?              → Abstract class
Template Method pattern needed?               → Abstract class
IS-A with shared code?                        → Abstract class
Java 8+ with some default behavior?           → Interface + default methods
```

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| What is Abstraction? | Hiding implementation details; exposing only the contract |
| Abstraction vs Encapsulation? | Abstraction hides complexity between layers; Encapsulation hides state within one object |
| Abstract class vs Interface? | Abstract class: state + code + IS-A. Interface: pure contract + CAN-DO |
| What is DIP? | High-level depends on abstractions; low-level implements them |
| What is a Leaky Abstraction? | Implementation details escape through the interface |
| `invokeinterface` vs `invokevirtual`? | `invokeinterface`: searches itable (slower first call). `invokevirtual`: fixed vtable offset |
| What is a Marker Interface? | Interface with no methods — for type tagging (Serializable, Cloneable) |
| Can abstract classes implement interfaces? | Yes — they can leave interface methods abstract for subclasses |
| What is structural subtyping? | Type compatibility by shape/methods, not declared hierarchy. Python Protocol, Go interfaces |
| When NOT to abstract? | When there's only one implementation and no test/swap need. YAGNI. |
| C++ ABI stability via pure virtual? | Pure virtual class = just vtable pointer — stable across compiles and compiler versions |
| Template Method pattern | Abstract class defines algorithm skeleton; abstract methods are extension points for subclasses |
| Java default methods (Java 8+)? | Concrete methods in interfaces — for backward-compatible API evolution without breaking implementors |
| What is the ISP? | Interface Segregation: one focused capability per interface — clients don't implement unused methods |
| What is fat interface? | Interface with too many unrelated methods — forces implementors to stub methods they don't need |
| How does abstraction enable testability? | Interface dependency → inject InMemory/Fake in tests — no real infrastructure needed |
| What is API surface area? | Number of public methods in your abstraction. Smaller = easier to evolve. |
| Java `Optional` and abstraction? | Abstracts "value might not be present" — callers use API instead of null checks |
| What is `invokeinterface` in JVM? | Bytecode for interface method call — must search itable; JVM caches for repeated calls |
| Plugin architecture and abstraction? | Core defines interface; plugins implement it; runtime loads dynamically. IntelliJ, Eclipse, Chrome. |

---

## 24. Cheat Sheet

### Top 20 Facts
1. Abstraction = contract (WHAT) separate from implementation (HOW)
2. Clients depend on abstractions — implementations are swappable
3. Java: interface (pure contract) + abstract class (partial implementation)
4. C++: pure virtual class (`= 0`) = interface; crucial for ABI stability
5. Python: ABC (nominal) vs Protocol (structural/duck typing)
6. `invokeinterface` slightly slower than `invokevirtual` — JIT inline cache closes the gap
7. DIP: high-level modules depend on abstractions; low-level implements them
8. Leaky abstraction: implementation details escape through interface
9. ISP: many small interfaces > one fat interface
10. Template Method: abstract class skeleton + abstract method extension points
11. YAGNI: don't abstract until you need a second implementation or testability
12. Abstract class: state + code + single IS-A hierarchy
13. Interface: no state + multiple implementation + CAN-DO capabilities
14. Java default methods (Java 8+): add methods to interface without breaking implementors
15. C++ pure virtual → ABI-stable cross-DLL boundary interfaces
16. Structural subtyping (Go/Python): type-compatible by shape, not declaration
17. Program to interface: `List<T>` not `ArrayList<T>` in variable declarations
18. Abstraction enables testability: inject fake implementations behind interface
19. Marker interfaces: no methods, just type-tagging (modern: use annotations)
20. Domain exceptions over infrastructure exceptions in interface contracts

---

## 25. Final Interview Summary

### Night-Before Revision
1. ⭐ Abstraction = hide HOW, expose WHAT via interfaces/abstract classes
2. ⭐ Abstraction ≠ Encapsulation: Abstraction hides complexity; Encapsulation hides state
3. ⭐ Interface: pure contract, no state, multiple impl, CAN-DO
4. ⭐ Abstract class: state + code + IS-A, Template Method
5. ⭐ DIP: high-level → abstraction ← low-level (Dependency Inversion)
6. ⭐ Program to interfaces: `List<T>` not `ArrayList<T>`
7. ⭐ Leaky abstraction: implementation detail escapes through interface
8. ⭐ `invokeinterface` vs `invokevirtual`: interface search vs fixed offset
9. ⭐ C++ pure virtual = ABI-stable interface for plugin/DLL boundaries
10. ⭐ Abstraction enables testability: inject fakes — no real DB/SMTP in unit tests
