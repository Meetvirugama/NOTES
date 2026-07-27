# Constructors — Industry-Level Interview Notes

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
A constructor is a **special initialization routine** that runs **exactly once** when an object is created. It sets the object's initial state so the object is ready to use immediately after `new`.

### Technically
A constructor is a subroutine that:
- **Same name as the class** (Java/C++) or `__init__` (Python)
- **No return type** — it doesn't explicitly return; the `new` operator returns the reference
- **Runs immediately after memory allocation**: `new` allocates heap memory → zeroes out fields → calls constructor to initialize state
- **Enforces invariants**: The first and strongest opportunity to guarantee the object is always in a valid state
- **Can be overloaded**: Multiple constructors with different parameter lists

### From an Interviewer's Perspective
> "I want to hear: 'new' allocates; constructor initializes — those are different steps. Tell me about constructor chaining (`this()`/`super()`), why constructors can't be virtual, why final fields must be initialized in constructors, copy constructors vs `clone()`, and the JVM Safe Publication guarantee for `final` fields. Private constructors for Singleton and Factory patterns are a must."

⭐ **Core contract**: After the constructor completes, every field must satisfy the class invariant. If it can't — throw an exception. Never let an invalid object enter the system.

---

## 2. Why It Exists

### Problem Without Constructors

#### Two-Step Initialization (Error-Prone)
```java
// Without constructor guarantee:
BankAccount account = new BankAccount(); // Memory allocated. Fields are null/0.
// ⚠️ Thread context switch RIGHT HERE
// Another thread calls account.withdraw(500); → balance is 0 → overdraft!
account.setOwner("Alice");
account.setBalance(1000.0);
// Now valid — but too late if something ran in between

// Without constructors in C++: raw memory = garbage!
struct Account { double balance; char* owner; };
Account acc; // balance is garbage (whatever is in that memory address!)
acc.balance  // Could be 1e308 or -42.0 or NaN
```

#### What Happens Without Forced Initialization
| Problem | Impact |
|---------|--------|
| Partial initialization | Object accessed before fully set up |
| Race conditions | Thread uses object before init completes |
| Null references | Methods called before fields set |
| No validation window | Invalid state enters system silently |
| No invariant enforcement | Any caller can skip required setup |

### With Constructors
```java
public class BankAccount {
    private final String accountId;   // final = MUST be set in constructor
    private final String owner;       // final = immutable after construction
    private double balance;

    // Constructor: fails loudly on invalid input, guarantees valid state on exit
    public BankAccount(String owner, double initialBalance) {
        if (owner == null || owner.isBlank())
            throw new IllegalArgumentException("Owner name required");
        if (initialBalance < 0)
            throw new IllegalArgumentException("Initial balance cannot be negative");
        this.accountId = UUID.randomUUID().toString();
        this.owner = owner;
        this.balance = initialBalance;
    }
    // Invariant guaranteed: accountId non-null, owner valid, balance >= 0 after construction
}
```

### Real Software Examples
- **`java.time.LocalDate`**: Constructor validates that date components are legal. `LocalDate.of(2024, 2, 30)` throws — impossible date never enters the system.
- **`java.net.URL`**: Constructor validates URL syntax. Invalid URL string → `MalformedURLException` in constructor.
- **Spring `ApplicationContext`**: Constructor starts dependency injection. If a required bean is missing, startup fails — Fail-Fast.
- **`DatabaseConnectionPool`**: Constructor opens initial connections, validates credentials. Misconfigured pool fails immediately — not on first request at 2 AM.

---

## 3. Internal Working

### The Object Creation Pipeline

```
Step 1: Class Loading
  JVM checks if BankAccount.class is loaded. If not, loads it.
  Static fields initialized. Static initializers run. (Once per class)

Step 2: new BankAccount("Alice", 1000) → Memory Allocation
  JVM allocates heap memory. Size = object header + all instance fields.
  JVM zeroes out all memory:
    accountId = null
    owner     = null
    balance   = 0.0
  Returns raw reference.

Step 3: Constructor Chain Execution
  a. Before any code: implicit super() call → Object()
  b. Instance initializers run (e.g., "private int x = 5;")
  c. Constructor body executes
  d. Final fields are written

Step 4: Safe Publication
  JVM memory model guarantees:
  final fields written in constructor are visible to ALL threads
  once the constructor completes and reference is published.

Step 5: Reference Returned
  Reference to fully constructed object is assigned to variable.
```

### C++ Constructor Execution Order (More Explicit)
```cpp
class Dog : public Animal {
    string breed;
    int age;

public:
    // Initializer list: initializes BEFORE constructor body
    Dog(string name, string breed, int age)
        : Animal(name),           // 1. Parent constructor first
          breed(std::move(breed)), // 2. Member 'breed' initialized
          age(age)                 // 3. Member 'age' initialized
    {
        // 4. Constructor body runs last
        // By this point, all members are already initialized!
        // Initializer list is more efficient than assignment in body
        cout << "Dog constructed: " << name << endl;
    }
};
```

### Java Memory Model — `final` Fields and Safe Publication
```java
// CRITICAL JVM RULE: final fields set in constructor are
// safely published to all threads after constructor completes.

class ImmutablePoint {
    private final int x;    // final
    private final int y;    // final

    public ImmutablePoint(int x, int y) {
        this.x = x;
        this.y = y;
        // After constructor: JVM memory barrier ensures x, y visible to all threads
    }
}

class MutablePoint {
    private int x;    // NOT final
    private int y;    // NOT final

    public MutablePoint(int x, int y) {
        this.x = x;
        this.y = y;
        // NO memory barrier! Thread B might see x=5, y=0 (reordering!)
        // Use volatile or synchronized for safe publication of mutable state
    }
}
```

### C++ Initializer List vs Constructor Body
```cpp
class ExpensiveObject {
    string data;         // Has a constructor

public:
    // WITHOUT initializer list (INEFFICIENT):
    ExpensiveObject(string d) {
        // 1. string() default constructor called
        // 2. Assignment operator called (data = d)
        // Two operations on 'data'!
        data = d;
    }

    // WITH initializer list (EFFICIENT):
    ExpensiveObject(string d) : data(std::move(d)) {
        // 1. string(std::move(d)) — move constructor directly
        // One operation. Data moved, not copied.
    }
};
```

### Constructor Chaining — `this()` and `super()`
```java
public class HttpRequest {
    private final String url;
    private final String method;
    private final Map<String, String> headers;
    private final Duration timeout;

    // Primary constructor: all parameters
    public HttpRequest(String url, String method, Map<String, String> headers, Duration timeout) {
        if (url == null) throw new IllegalArgumentException("URL required");
        this.url = url;
        this.method = method;
        this.headers = Map.copyOf(headers);
        this.timeout = timeout;
    }

    // Convenience: GET with default timeout
    public HttpRequest(String url, Map<String, String> headers) {
        this(url, "GET", headers, Duration.ofSeconds(30)); // this() → primary constructor
        // Validation runs only ONCE, in primary constructor
    }

    // Minimal: GET, empty headers, default timeout
    public HttpRequest(String url) {
        this(url, Map.of()); // this() → 2-param constructor → primary constructor
    }
}
```

---

## 4. Syntax

### Java — All Constructor Types
```java
public class User {
    private final String id;
    private final String email;
    private String displayName;
    private final Instant createdAt;
    private List<String> roles;

    // 1. PRIMARY CONSTRUCTOR: Full validation, invariant enforcement
    public User(String id, String email, String displayName) {
        if (id == null || id.isBlank())
            throw new IllegalArgumentException("User ID cannot be blank");
        if (email == null || !email.contains("@"))
            throw new IllegalArgumentException("Invalid email: " + email);
        if (displayName == null || displayName.isBlank())
            throw new IllegalArgumentException("Display name required");

        this.id = id;
        this.email = email.toLowerCase().trim();    // Normalize
        this.displayName = displayName.trim();
        this.createdAt = Instant.now();
        this.roles = new ArrayList<>();
    }

    // 2. OVERLOADED CONSTRUCTOR: Convenience (auto-generate ID)
    public User(String email, String displayName) {
        this(UUID.randomUUID().toString(), email, displayName); // Chain to primary
    }

    // 3. COPY CONSTRUCTOR: Deep copy
    public User(User source) {
        this(UUID.randomUUID().toString(), source.email, source.displayName); // New ID!
        this.roles = new ArrayList<>(source.roles); // Deep copy of mutable list
    }

    // 4. STATIC FACTORY METHOD (over constructor): descriptive name
    public static User guestUser(String email) {
        User user = new User(email, "Guest");
        user.roles.add("GUEST");
        return user;
    }

    public static User adminUser(String email, String displayName) {
        User user = new User(email, displayName);
        user.roles.addAll(List.of("USER", "ADMIN", "SUPER_ADMIN"));
        return user;
    }
}
```

### C++ — Constructors, Initializer Lists, Rule of Five
```cpp
#include <string>
#include <vector>
#include <memory>
using namespace std;

class FileProcessor {
private:
    string filename;
    FILE* fileHandle;
    vector<string> processedLines;
    int lineCount;

public:
    // 1. PARAMETERIZED CONSTRUCTOR with initializer list
    explicit FileProcessor(const string& filename)
        : filename(filename),
          fileHandle(nullptr),  // Initialized to null
          lineCount(0)          // Initialized to 0
    {
        fileHandle = fopen(filename.c_str(), "r");
        if (!fileHandle)
            throw runtime_error("Cannot open file: " + filename);
    }

    // 2. COPY CONSTRUCTOR (deep copy)
    FileProcessor(const FileProcessor& other)
        : filename(other.filename),
          processedLines(other.processedLines),  // Copies vector
          lineCount(other.lineCount)
    {
        // Re-open the file independently
        fileHandle = fopen(filename.c_str(), "r");
        if (!fileHandle)
            throw runtime_error("Cannot re-open file for copy: " + filename);
    }

    // 3. MOVE CONSTRUCTOR (transfer ownership — no copy)
    FileProcessor(FileProcessor&& other) noexcept
        : filename(std::move(other.filename)),
          fileHandle(other.fileHandle),           // Transfer pointer
          processedLines(std::move(other.processedLines)),
          lineCount(other.lineCount)
    {
        other.fileHandle = nullptr;  // Source is now empty
        other.lineCount = 0;
    }

    // 4. DESTRUCTOR (RAII cleanup)
    ~FileProcessor() {
        if (fileHandle) {
            fclose(fileHandle);
            fileHandle = nullptr;
        }
    }

    // 5. DELETED constructors (prevent unintended usage)
    FileProcessor() = delete; // No default construction — filename is required
};
```

### Python — `__init__`, `__post_init__` (dataclass), Class Method Factories
```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import re

class Order:
    """Domain entity with full validation in constructor."""

    def __init__(self, order_id: str, customer_id: str, items: list) -> None:
        # Validation first (Fail-Fast)
        if not order_id or not order_id.strip():
            raise ValueError("Order ID cannot be empty")
        if not customer_id:
            raise ValueError("Customer ID required")
        if not items:
            raise ValueError("Order must have at least one item")

        # Assign (defensively copy mutable inputs)
        self._order_id = order_id
        self._customer_id = customer_id
        self._items = list(items)   # defensive copy
        self._status = "PENDING"
        self._created_at = datetime.utcnow()
        self._total = sum(item.price for item in items)

    # Class method factory — alternative named constructors
    @classmethod
    def from_dict(cls, data: dict) -> 'Order':
        """Create Order from dictionary (e.g., deserialized JSON)."""
        return cls(
            order_id=data['order_id'],
            customer_id=data['customer_id'],
            items=[OrderItem.from_dict(i) for i in data['items']]
        )

    @classmethod
    def test_order(cls, items: Optional[list] = None) -> 'Order':
        """Factory for testing — sensible defaults."""
        return cls(
            order_id="TEST-" + str(id(cls)),
            customer_id="test-customer",
            items=items or [OrderItem("test-item", 100.0)]
        )

    @property
    def order_id(self) -> str: return self._order_id
    @property
    def total(self) -> float: return self._total
    @property
    def status(self) -> str: return self._status


# Python dataclass with __post_init__ validation
@dataclass
class Money:
    amount_cents: int
    currency: str

    def __post_init__(self):
        # Validation after auto-generated __init__
        if self.amount_cents < 0:
            raise ValueError(f"Money amount cannot be negative: {self.amount_cents}")
        if self.currency not in ('USD', 'EUR', 'GBP', 'JPY'):
            raise ValueError(f"Unsupported currency: {self.currency}")
        # Normalize
        self.currency = self.currency.upper()
```

---

## 5. Visual Diagrams

### Object Creation Pipeline
```
Developer writes: User user = new User("alice@example.com", "Alice");

Step 1: Class Loading (if not already done)
  JVM loads User.class into Metaspace
  Static fields and static blocks run

Step 2: Memory Allocation (new keyword)
  JVM asks heap allocator for sizeof(User) bytes
  Thread Local Allocation Buffer (TLAB) → ~5ns bump pointer
  Memory ZEROED:
    id:          null
    email:       null
    displayName: null
    createdAt:   null
    roles:       null

Step 3: Constructor Execution
  a) super() → Object() → Object-level setup
  b) Instance initializers (if any outside constructor)
  c) Constructor body:
      id:          "f47ac10b-58cc-..." (generated UUID)
      email:       "alice@example.com"
      displayName: "Alice"
      createdAt:   Instant{2024-01-15T10:30:00Z}
      roles:       new ArrayList()

Step 4: JVM Memory Barrier (for final fields)
  Ensures all threads see: id, email, createdAt fully initialized

Step 5: Reference returned → stored in 'user'
  user → [User object on heap]
```

### Constructor Chaining Call Stack
```
User("alice@example.com", "Alice")  ← call
  → this(UUID.randomUUID().toString(), "alice@example.com", "Alice")
       ↓
  → User("abc-123", "alice@example.com", "Alice")  ← executes
       ↓
  → super() → Object()  ← runs first
       ↓
  ← Object() returns
       ↓
  → Constructor body validates, assigns all fields
       ↓
  ← User("abc-123", ...) returns
       ↓
← User("alice@example.com", "Alice") returns (no-op — delegated)

Rule: this() or super() MUST be first statement in constructor
```

### RAII (C++) — Constructor/Destructor Resource Lifecycle
```
void processFile(const string& path) {
    FileProcessor fp(path);  ← Constructor: opens file, allocates resources
    //
    // ... work with fp ...   ← File handle is open, ready to use
    //
}  ← } Destructor: automatically called when scope exits
       File closed, resources freed — even if exception thrown!

vs Java (manual):
void processFile(String path) {
    try (BufferedReader br = new BufferedReader(new FileReader(path))) {
        // try-with-resources = Java's RAII equivalent
        // close() called automatically on exit/exception
    }
}
```

---

## 6. Real World Analogy

### Building a House (Construction Analogy)
- **`new` keyword** = Buying the empty plot of land (allocating space)
- **Constructor** = Construction crew building the foundation, walls, roof
- **Constructor parameters** = Blueprint specifications (size, material, location)
- **Constructor validation** = Inspection: if land is unsuitable, construction halts immediately
- **Object after construction** = Move-in ready house — all rooms functional

You don't move into a house mid-construction (no partial objects). The constructor guarantees the house is livable before you get the keys.

### Employee Onboarding
- **`new Employee("Alice", "Engineering")`** = Hiring Alice
- **Constructor** = Onboarding process: assign employee ID, issue laptop, set up accounts, add to payroll
- **Constructor validation** = Background check: if it fails, Alice isn't hired (no Employee object created)
- **Fully constructed Employee** = First day, everything ready to work

If onboarding fails (no employee ID available), Alice is not an Employee object. She never enters the system in a broken state.

### Factory at the Assembly Line
- **Constructor** = Quality control at the end of the assembly line
- Items that don't pass inspection are thrown away (exception thrown)
- Only items passing all checks leave the factory (reach calling code)
- Once past QC, the item is guaranteed to be functional

---

## 7. Interview Explanation

### 30 Seconds
> "A constructor is a special initialization routine that runs when an object is created via `new`. It sets the object's initial state and enforces class invariants. It has the same name as the class, no return type, and if validation fails, it throws an exception to prevent an invalid object from ever being created."

### 1 Minute
> "Constructors are the Fail-Fast mechanism for object creation. The `new` keyword allocates memory and zeroes fields; the constructor then initializes them to valid values. This two-step process matters because Java's JVM zeroes memory before calling the constructor — unlike C++ where raw memory contains garbage values if you don't use the initializer list.
>
> Constructors can be chained: `this(args)` delegates to another constructor in the same class, and `super(args)` initializes the parent. This enforces a single validation path — write validation once in the primary constructor, call it from all overloads.
>
> Private constructors enforce patterns — Singleton prevents multiple instances, Factory Methods use named static methods for clarity."

### 3 Minutes
> "Let me go deep on three critical production aspects of constructors.
>
> First: JVM memory model and `final` fields. The Java Memory Model guarantees that `final` fields written inside a constructor are fully visible to ALL threads after the constructor completes, without any additional synchronization. This is called 'safe publication' — critical for thread-safe immutable objects. Non-final fields have NO such guarantee: another thread might see the object reference before all fields are written, seeing null/0 for fields that were already assigned (due to CPU instruction reordering). This is why immutable objects are the safest for concurrent sharing.
>
> Second: RAII in C++. C++ constructors acquire resources (open files, allocate memory, acquire locks), and the destructor releases them when the object goes out of scope — even if an exception is thrown. This eliminates resource leaks entirely because the C++ runtime guarantees destructors run during stack unwinding. Java's try-with-resources is the equivalent.
>
> Third: The 'Never call virtual methods from constructors' rule. In Java, if a parent constructor calls an overridable method, and a subclass overrides it, the override runs BEFORE the subclass constructor body — meaning the subclass's fields are still null/0. This is a common source of NullPointerExceptions. In C++, the behavior is different: calling a virtual method from a constructor calls the method as defined in the current class's vtable level, not the overriding subclass version."

---

## 8. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What is a constructor? | Special method that runs when object is created via `new`; initializes state; same name as class; no return type |
| Does Java provide a default constructor? | Yes — only if you write zero constructors. Once you write any constructor, the default is removed. |
| Can a constructor return a value? | No return type. `new` handles returning the object reference. |
| Can constructors be overloaded? | Yes — same name, different parameter lists. |
| What is constructor chaining? | `this(args)` calls another constructor in same class; `super(args)` calls parent constructor. Must be first line. |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| Why make a constructor private? | Singleton: prevent external instantiation; Static factory methods: control creation logic; Builder: force Builder usage |
| Can a constructor throw an exception? | Yes — constructor should throw `IllegalArgumentException`, `NullPointerException` if invariant can't be established |
| Copy constructor vs `clone()` in Java? | Copy constructor: explicit deep copy, type-safe, no `CloneNotSupportedException`; `clone()` is Cloneable marker interface, does shallow copy by default, error-prone |
| Can you call `this()` and `super()` both in a constructor? | No — both must be the first statement. Only one can be the first statement. |
| What is the difference between field initializer and constructor? | Field initializer (int x = 5) runs before constructor body; useful for simple defaults |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| Why can't constructors be virtual in C++? | Constructor builds the vtable; a virtual mechanism requires an existing vtable. Bootstrap problem: you need the vtable to call virtual methods, but the vtable is being built during construction. |
| What is the JVM Safe Publication guarantee for `final` fields? | JVM inserts a memory barrier after constructor writes to `final` fields. Any thread that sees the object reference is guaranteed to see correctly initialized `final` fields — no extra synchronization needed. |
| Why shouldn't you call virtual/overridable methods in a constructor? | In Java: subclass override runs before subclass constructor → subclass fields are null/0. In C++: virtual dispatch uses current class level — doesn't call subclass override at all. Both are confusing and error-prone. |
| What is `this` reference escape in constructor? | Publishing `this` (e.g., registering with a listener) before constructor completes → another thread may see partially initialized object. Dangerous pattern. |
| What happens if a constructor throws an exception? | Object creation is abandoned. Reference is never assigned. Partially constructed object becomes eligible for GC. In C++, destructors for successfully constructed members are called automatically. |

### 💼 Google Level
> *"You're designing a high-throughput event processor. Object creation runs in a tight loop creating 10M EventRecord objects per second. The constructor does validation (regex match on event type). Profiling shows constructors take 40% of CPU. How do you optimize?"*

Expected: (1) Pre-compile regex Pattern, share as static final. (2) Batch validate at boundary, not per-object. (3) Use object pooling (`ObjectPool<EventRecord>`) — reset fields instead of constructing new. (4) Consider Java records with compile-time constraints. (5) Use direct memory allocation + off-heap for extreme cases.

### 💼 Goldman Sachs Level
> *"Our `TradeOrder` constructor validates 15 fields including market rules. Validation changes every quarter (regulatory changes). How do you design the constructor to handle this without recompiling the entire class?"*

Expected: Extract validation into `OrderValidator` interface (Strategy pattern). Inject validator via constructor. `TradeOrder(OrderRequest req, OrderValidator validator)`. On regulatory change: deploy new `OrderValidator` implementation. `TradeOrder` constructor unchanged.

---

## 9. Coding Examples

### Basic — Invariant-Enforcing Constructor
```java
public final class EmailAddress {
    private static final Pattern EMAIL_PATTERN =
        Pattern.compile("^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$");

    private final String value;

    public EmailAddress(String email) {
        Objects.requireNonNull(email, "Email cannot be null");
        String normalized = email.trim().toLowerCase();
        if (!EMAIL_PATTERN.matcher(normalized).matches())
            throw new IllegalArgumentException("Invalid email format: " + email);
        this.value = normalized;
    }

    public String getValue() { return value; }

    @Override
    public boolean equals(Object o) {
        return o instanceof EmailAddress ea && ea.value.equals(this.value);
    }

    @Override public int hashCode() { return value.hashCode(); }
    @Override public String toString() { return value; }
}
```

### Intermediate — Constructor Chaining + Validation Consolidation
```java
public class DatabaseConnectionPool {
    private final String host;
    private final int port;
    private final String database;
    private final String username;
    private final String password;
    private final int minConnections;
    private final int maxConnections;
    private final Duration connectionTimeout;
    private final List<Connection> pool;

    // PRIMARY CONSTRUCTOR: Full validation — single source of truth
    public DatabaseConnectionPool(
            String host, int port, String database,
            String username, String password,
            int minConnections, int maxConnections,
            Duration connectionTimeout) {

        // Validate all inputs
        if (host == null || host.isBlank()) throw new IllegalArgumentException("Host required");
        if (port < 1 || port > 65535) throw new IllegalArgumentException("Invalid port: " + port);
        if (database == null || database.isBlank()) throw new IllegalArgumentException("Database required");
        if (username == null) throw new IllegalArgumentException("Username required");
        if (password == null) throw new IllegalArgumentException("Password required");
        if (minConnections < 0) throw new IllegalArgumentException("minConnections must be >= 0");
        if (maxConnections < minConnections) throw new IllegalArgumentException("maxConnections must be >= minConnections");
        if (connectionTimeout == null || connectionTimeout.isNegative())
            throw new IllegalArgumentException("Valid connection timeout required");

        this.host = host;
        this.port = port;
        this.database = database;
        this.username = username;
        this.password = password;
        this.minConnections = minConnections;
        this.maxConnections = maxConnections;
        this.connectionTimeout = connectionTimeout;
        this.pool = initializePool();
    }

    // OVERLOADED CONSTRUCTORS: Convenience, all delegate to primary
    public DatabaseConnectionPool(String host, int port, String database, String username, String password) {
        this(host, port, database, username, password, 2, 10, Duration.ofSeconds(30));
    }

    public DatabaseConnectionPool(String jdbcUrl, String username, String password) {
        this(parseHost(jdbcUrl), parsePort(jdbcUrl), parseDatabase(jdbcUrl),
             username, password, 2, 10, Duration.ofSeconds(30));
    }

    private List<Connection> initializePool() {
        List<Connection> connections = new ArrayList<>();
        for (int i = 0; i < minConnections; i++) {
            connections.add(openConnection());
        }
        return connections;
    }

    private Connection openConnection() {
        // Actually opens DB connection during construction — Fail-Fast!
        try {
            return DriverManager.getConnection(
                "jdbc:" + host + ":" + port + "/" + database, username, password);
        } catch (SQLException e) {
            throw new IllegalStateException("Failed to connect to database during pool initialization", e);
        }
    }

    private static String parseHost(String jdbcUrl) { /* parse */ return ""; }
    private static int parsePort(String jdbcUrl) { /* parse */ return 5432; }
    private static String parseDatabase(String jdbcUrl) { /* parse */ return ""; }
}
```

### Advanced — Copy Constructor with Deep Copy
```java
public class Portfolio {
    private final String portfolioId;
    private final String ownerId;
    private final Map<String, Position> positions;  // Mutable map of mutable objects!
    private final List<Transaction> transactions;   // Mutable list of immutable objects

    public Portfolio(String portfolioId, String ownerId) {
        this.portfolioId = Objects.requireNonNull(portfolioId);
        this.ownerId = Objects.requireNonNull(ownerId);
        this.positions = new HashMap<>();
        this.transactions = new ArrayList<>();
    }

    // DEEP COPY CONSTRUCTOR
    public Portfolio(Portfolio source) {
        this.portfolioId = UUID.randomUUID().toString(); // New ID for the copy!
        this.ownerId = source.ownerId;
        // Deep copy: each Position is itself copied
        this.positions = new HashMap<>();
        source.positions.forEach((symbol, pos) ->
            this.positions.put(symbol, new Position(pos))); // Position copy constructor
        // Transactions are immutable records — shallow copy sufficient
        this.transactions = new ArrayList<>(source.transactions);
    }

    // Controlled mutation
    public void addPosition(String symbol, int quantity, Money purchasePrice) {
        positions.merge(symbol,
            new Position(symbol, quantity, purchasePrice),
            Position::combine);
        transactions.add(new Transaction(symbol, quantity, purchasePrice, Instant.now()));
    }

    // Defensive return
    public Map<String, Position> getPositions() {
        return Collections.unmodifiableMap(positions);
    }
}
```

### Production — Builder Pattern (Telescoping Constructor Solution)
```java
// The problem: constructor with many parameters
// new HttpRequest(url, "POST", headers, body, Duration.ofSeconds(30), 3, auth);
// Which argument is which? Hard to read, easy to mix up order.

// SOLUTION: Builder Pattern
public final class HttpRequest {
    private final String url;
    private final String method;
    private final Map<String, String> headers;
    private final byte[] body;
    private final Duration timeout;
    private final int maxRetries;
    private final String authToken;

    // Private constructor — only Builder can call this
    private HttpRequest(Builder builder) {
        this.url = builder.url;
        this.method = builder.method;
        this.headers = Collections.unmodifiableMap(new HashMap<>(builder.headers));
        this.body = builder.body != null ? builder.body.clone() : null; // Defensive copy
        this.timeout = builder.timeout;
        this.maxRetries = builder.maxRetries;
        this.authToken = builder.authToken;
    }

    // Only getters — fully immutable
    public String getUrl() { return url; }
    public String getMethod() { return method; }
    public Map<String, String> getHeaders() { return headers; }
    public Optional<byte[]> getBody() { return Optional.ofNullable(body != null ? body.clone() : null); }
    public Duration getTimeout() { return timeout; }
    public int getMaxRetries() { return maxRetries; }

    public static Builder builder(String url) { return new Builder(url); }

    public static class Builder {
        private final String url;
        private String method = "GET";
        private final Map<String, String> headers = new HashMap<>();
        private byte[] body;
        private Duration timeout = Duration.ofSeconds(30);
        private int maxRetries = 3;
        private String authToken;

        private Builder(String url) {
            if (url == null || url.isBlank()) throw new IllegalArgumentException("URL required");
            this.url = url;
        }

        public Builder method(String method) { this.method = method; return this; }
        public Builder header(String key, String value) { this.headers.put(key, value); return this; }
        public Builder body(byte[] body) { this.body = body.clone(); return this; }
        public Builder timeout(Duration timeout) { this.timeout = timeout; return this; }
        public Builder maxRetries(int retries) { this.maxRetries = retries; return this; }
        public Builder authToken(String token) { this.authToken = token; return this; }

        public HttpRequest build() {
            // Final validation before constructing
            if ("POST".equals(method) && body == null)
                throw new IllegalStateException("POST request requires body");
            return new HttpRequest(this);
        }
    }
}

// Usage: readable, named parameters, immutable result
HttpRequest request = HttpRequest.builder("https://api.example.com/orders")
    .method("POST")
    .header("Content-Type", "application/json")
    .header("X-Request-ID", UUID.randomUUID().toString())
    .body(orderJson.getBytes())
    .timeout(Duration.ofSeconds(10))
    .authToken(bearerToken)
    .build();
```

---

## 10. Common Mistakes

### ⚠️ Mistake 1: Calling Overridable Methods in Constructor
```java
class Logger {
    private String prefix;

    public Logger() {
        // WRONG: initialize() is virtual — subclass override may run here!
        initialize();  // Dangerous!
    }

    protected void initialize() {
        this.prefix = "[DEFAULT]";
    }
}

class FileLogger extends Logger {
    private String logFile;

    public FileLogger(String logFile) {
        super(); // Calls Logger(), which calls initialize() override BELOW
        this.logFile = logFile; // TOO LATE — initialize() already ran with null logFile!
    }

    @Override
    protected void initialize() {
        // logFile is NULL here! Logger() constructor called this before FileLogger() set it!
        System.out.println("Logging to: " + logFile.toUpperCase()); // NullPointerException!
    }
}

// FIX: Don't call overridable methods in constructors
class Logger {
    protected String prefix;

    public Logger() {
        this.prefix = "[DEFAULT]"; // Direct assignment, no virtual call
    }
}
```

### ⚠️ Mistake 2: Missing Defensive Copy of Mutable Parameters
```java
// WRONG: Stores direct reference to caller's array
public class Config {
    private String[] allowedHosts;

    public Config(String[] allowedHosts) {
        this.allowedHosts = allowedHosts; // BUG: Caller still holds reference!
    }
}

String[] hosts = {"example.com", "api.example.com"};
Config config = new Config(hosts);
hosts[0] = "malicious.com"; // ← Config's allowedHosts is now corrupted!

// FIX: Defensive copy in constructor
public Config(String[] allowedHosts) {
    this.allowedHosts = Arrays.copyOf(allowedHosts, allowedHosts.length);
}
```

### ⚠️ Mistake 3: `this` Reference Escape
```java
// WRONG: Publishing 'this' before constructor completes
class EventProcessor implements EventListener {
    private String processorName;
    private EventBus eventBus;

    public EventProcessor(String name, EventBus bus) {
        this.processorName = name;
        bus.register(this);    // DANGER! 'this' escapes! Bus may fire events
        this.eventBus = bus;   // But we're not done constructing!
    }
    // Another thread could call onEvent() while constructor is still running!
}

// FIX: Static factory method handles registration after construction
public class EventProcessor implements EventListener {
    private String processorName;
    private EventBus eventBus;

    private EventProcessor(String name) { this.processorName = name; }

    public static EventProcessor create(String name, EventBus bus) {
        EventProcessor processor = new EventProcessor(name); // Fully constructed
        processor.eventBus = bus;
        bus.register(processor); // Safe to publish now
        return processor;
    }
}
```

### ⚠️ Mistake 4: No Validation = Delayed Failure
```java
// WRONG: Missing validation — fails later with cryptic errors
public class DatabaseConfig {
    public final String host;
    public final int port;
    public final String database;

    public DatabaseConfig(String host, int port, String database) {
        this.host = host;       // null? Empty string? No check!
        this.port = port;       // -1? 999999? No check!
        this.database = database; // null? No check!
    }
}
// Usage: new DatabaseConfig(null, -1, "") → object created!
// Fails later with: "Cannot connect to null:-1/"

// FIX: Validate eagerly
public DatabaseConfig(String host, int port, String database) {
    if (host == null || host.isBlank()) throw new IllegalArgumentException("Host required");
    if (port < 1 || port > 65535) throw new IllegalArgumentException("Invalid port: " + port);
    if (database == null || database.isBlank()) throw new IllegalArgumentException("Database required");
    this.host = host;
    this.port = port;
    this.database = database;
}
```

### ⚠️ Mistake 5: Heavy Work in Constructor (DB calls, Network calls)
```java
// WRONG: Constructor does network calls — hard to test, slow, potentially fails
public class UserProfileService {
    private List<User> allUsers;

    public UserProfileService(DatabaseRepository repo) {
        // Heavy operation in constructor!
        this.allUsers = repo.findAll(); // Could be 10M users, takes 10 seconds!
        // Testing requires real database with data!
    }
}

// FIX: Lazy initialization or delegate to factory/init method
public class UserProfileService {
    private final DatabaseRepository repo;
    private volatile List<User> userCache; // Lazy

    public UserProfileService(DatabaseRepository repo) {
        this.repo = Objects.requireNonNull(repo); // Light setup only
    }

    private List<User> getUserCache() {
        if (userCache == null) {
            synchronized (this) {
                if (userCache == null) userCache = repo.findAll(); // Lazy on first use
            }
        }
        return userCache;
    }
}
```

---

## 11. Best Practices

### Design Rules
- **Validate all parameters in the primary constructor** — fail fast, never allow invalid state
- **Use `Objects.requireNonNull()`** for null checks — concise and descriptive
- **Make fields `final` when possible** — immutability + JVM safe publication
- **Defensive copy mutable inputs** — arrays, lists, sets, maps
- **Chain to a primary constructor** — one validation path, less duplication
- **Don't call virtual/overridable methods** — subclass fields not initialized yet
- **Don't do heavy work** — DB calls, network calls belong in factory/init patterns

### When to Use Each Constructor Type
| Type | When to Use |
|------|-------------|
| Parameterized (primary) | Always — the main validation + initialization |
| Overloaded (convenience) | Provide defaults for common cases via `this()` chaining |
| Copy constructor | When you need a deep clone of an object |
| Private + static factory | When you need descriptive names or want to control instance creation |
| Builder | When > 4 parameters, many optional parameters |

---

## 12. Complexity

| Operation | Cost | Notes |
|-----------|------|-------|
| Object allocation (`new`) | ~5-10ns | TLAB bump pointer — extremely fast |
| Field initialization | O(1), <1ns per field | Simple assignments |
| Constructor validation | O(1) | Comparisons, null checks |
| Defensive copy (collection) | O(n) | Must copy all elements |
| Deep copy constructor | O(n) | Recursive copy of nested objects |
| Static block initialization | O(1) once | Runs once per class at first load |

---

## 13. Advantages

| Advantage | Concrete Benefit |
|-----------|-----------------|
| **Guaranteed initialization** | Object always fully initialized before use |
| **Invariant enforcement** | Invalid state impossible — constructor rejects it |
| **Immutability enablement** | `final` fields can only be set in constructor |
| **Fail-Fast** | Invalid config caught at startup, not at 2 AM in production |
| **Testability** | Constructor injection → easy to inject test doubles |
| **Thread safety** | `final` fields + constructor → JVM safe publication guarantee |

---

## 14. Disadvantages

| Disadvantage | Impact | Mitigation |
|-------------|--------|------------|
| **Telescoping constructors** | 5+ params is confusing | Builder pattern |
| **Cannot return different types** | Factory sometimes needed | Static factory methods |
| **Cannot be virtual (C++)** | Cannot override creation polymorphically | Factory Method pattern |
| **Heavy init slows startup** | DB/network in constructor = slow tests | Lazy init or factory |
| **Serialization bypass** | `ObjectInputStream` skips constructor (Java) | Custom `readObject()` |
| **`this` escape risk** | Publishing reference too early | Static factory method |

---

## 15. Comparison Table

### Constructor vs Method vs Static Factory

| Feature | Constructor | Instance Method | Static Factory |
|---------|-------------|----------------|----------------|
| **Name** | Same as class | Any name | Any name (descriptive!) |
| **Return type** | None (returns object) | Any type | Class type |
| **Can cache/reuse** | No (always new) | N/A | Yes (return cached instance) |
| **Can return subtype** | No | Yes | Yes |
| **Inheritance** | Not inherited | Yes | No (static) |
| **Overridable** | No | Yes | No |
| **When to use** | Standard creation | Operations | Named/controlled creation |

### `this()` vs `super()` in Constructor

| Aspect | `this()` | `super()` |
|--------|---------|---------|
| **Calls** | Another constructor in same class | Parent class constructor |
| **Position** | Must be first line | Must be first line |
| **Can combine** | Only one allowed (first line) | Mutually exclusive with this() |
| **Purpose** | Constructor chaining / reduce duplication | Initialize parent state |

---

## 16. Design Pattern Connection

| Pattern | Constructor Role |
|---------|----------------|
| **Singleton** | Private constructor prevents external creation; `getInstance()` controls single instance |
| **Builder** | Private constructor; Builder's `build()` calls it with validated parameters |
| **Factory Method** | Subclass factories override to create specific objects |
| **Abstract Factory** | Returns products; each product's constructor is called internally |
| **Prototype** | Copy constructor is the clone mechanism |
| **Dependency Injection** | Constructor injection is the preferred DI method (Spring `@Autowired`) |
| **Flyweight** | Private constructor; factory reuses existing instances |
| **Object Pool** | Constructors create initial pool; `acquire()`/`release()` manage lifecycle |

---

## 17. System Design Connection

### Dependency Injection (Spring)
```java
// Constructor injection: explicit dependencies, testable, final fields
@Service
public class OrderService {
    private final OrderRepository orderRepo;
    private final PaymentService paymentService;
    private final NotificationService notifications;

    // Spring calls this constructor, injects implementations
    @Autowired
    public OrderService(OrderRepository orderRepo,
                        PaymentService paymentService,
                        NotificationService notifications) {
        this.orderRepo = Objects.requireNonNull(orderRepo);
        this.paymentService = Objects.requireNonNull(paymentService);
        this.notifications = Objects.requireNonNull(notifications);
    }
    // Fields are final → immutable → thread-safe
}
```

### Connection Pooling
```java
// Constructor opens initial connections — Fail-Fast at startup
public class ConnectionPool {
    private final BlockingQueue<Connection> pool;

    public ConnectionPool(String url, String user, String pass, int size) {
        this.pool = new ArrayBlockingQueue<>(size);
        for (int i = 0; i < size; i++) {
            try {
                pool.offer(DriverManager.getConnection(url, user, pass));
            } catch (SQLException e) {
                throw new IllegalStateException("Pool init failed: " + e.getMessage(), e);
            }
        }
        // If any connection fails: startup fails. No broken pool enters service.
    }
}
```

---

## 18. Multithreading Connection

### `final` Fields and JVM Safe Publication
```java
// Thread A: creates ImmutableConfig and shares reference
class ImmutableConfig {
    private final String host;       // final!
    private final int port;          // final!
    private final List<String> tags; // final!

    public ImmutableConfig(String host, int port, List<String> tags) {
        this.host = host;
        this.port = port;
        this.tags = List.copyOf(tags); // Immutable copy
        // JVM inserts StoreStore barrier after writing final fields
    }
}

// Thread B: receives the reference
// JVM GUARANTEES: Thread B sees correct host, port, tags
// NO additional synchronization needed because fields are final!
```

```java
// Non-final fields: UNSAFE publication — Thread B might see stale values!
class MutableConfig {
    private String host;    // NOT final
    private int port;       // NOT final

    public MutableConfig(String host, int port) {
        this.host = host;
        this.port = port;
        // NO memory barrier — Thread B might see host=null, port=0
    }
}

// Fix: use volatile, synchronized, or make config immutable
class SafeMutableConfig {
    private volatile String host;
    private volatile int port;
    // ...
}
```

### Double-Checked Locking and Constructors
```java
public class Singleton {
    private static volatile Singleton instance; // volatile = memory barrier

    private Singleton() {
        // Private constructor: can't instantiate from outside
    }

    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton(); // Constructor runs here
                    // volatile ensures reference is fully visible after assignment
                }
            }
        }
        return instance;
    }
}
```

---

## 19. Company Interview Perspective

### Google
- "What's the JVM memory model guarantee for `final` fields set in a constructor?"
- Object allocation optimization: TLAB, escaping analysis (JIT can stack-allocate short-lived objects)
- Protobuf: why generated classes use builder pattern instead of constructors

### Goldman Sachs
- "Design a `TradeOrder` constructor that validates 15 fields including market hours and regulatory limits"
- Immutable value objects for financial amounts: `Money`, `Price`, `Quantity`
- `TradeOrder` construction from FIX protocol messages — validation at boundary

### Amazon
- "How does Spring Boot's constructor injection guarantee beans are ready at startup?"
- DynamoDB SDK client construction: validation of credentials and region at construction
- Lambda cold start: object construction as a hot path that must be fast

### Microsoft
- C++ constructor vs initializer list performance difference
- C# struct constructors vs class constructors — stack vs heap
- WPF `DependencyObject` construction and property system initialization

### Stripe (Payments Focus)
- "How do you design an idempotent constructor pattern for payment objects?"
- Using constructors to enforce payment validation (amount > 0, valid currency codes)
- Thread-safe `PaymentResult` as an immutable value object (final fields)

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | ⚠️ Does Java provide a default constructor if you have a parameterized one? | No! If you write any constructor, the default no-arg constructor is NOT provided. |
| 2 | Can constructors be abstract? | No — abstract means no implementation; a constructor IS the implementation for initialization. |
| 3 | ⚠️ Can a constructor call `this()` and `super()` both? | No — both must be first line. Only one can be first. Pick one or neither (implicit super()). |
| 4 | What happens when an exception is thrown in a constructor? | Object creation fails; reference never assigned. In C++, destructors of already-constructed members are called. |
| 5 | ⚠️ Why can't constructors be virtual (C++)? | Vtable is built DURING construction. No vtable exists yet to dispatch through. Chicken-and-egg problem. |
| 6 | What is the JVM `final` field safe publication rule? | `final` fields set in constructor are guaranteed visible to all threads after constructor completes. |
| 7 | ⚠️ What is `this` reference escape? | Publishing `this` (registering with observer) before constructor completes — other threads see partially initialized object. |
| 8 | Copy constructor vs `clone()` — which is better in Java? | Copy constructor: type-safe, explicit deep copy, no `CloneNotSupportedException`. `clone()`: shallow by default, marker interface, error-prone. Copy constructor wins. |
| 9 | ⚠️ Can `static` fields be initialized in a constructor? | Yes, but NOT recommended. Static fields are shared across instances. `static` initializers or static blocks are better. |
| 10 | What is C++'s initializer list and why is it more efficient? | Initializes members DIRECTLY using their constructors. Without it: default constructor + assignment operator. With it: single constructor call. |
| 11 | ⚠️ What is `readObject()` in Java serialization? | When deserializing, Java bypasses the constructor! `readObject()` lets you add validation that would have been in the constructor. |
| 12 | How can serialization break encapsulation? | `ObjectInputStream.readObject()` can create objects with arbitrary field values, bypassing constructor validation entirely. |
| 13 | ⚠️ What is the purpose of calling `super()` first in a constructor? | Parent's state must be initialized before child can safely reference `this`. Enforced by compiler. |
| 14 | Can an interface have a constructor? | No — interfaces hold no state; nothing to initialize. |
| 15 | What happens in C++ if you don't write a destructor for a class with a virtual function? | Memory leaks! If you delete a derived class through a base pointer, without `virtual ~BaseClass()`, only the base destructor runs — derived resources are never freed. |

---

## 21. Coding Problems

### Easy — Implement a Thread-Safe Singleton
```java
// Eager initialization (simplest, JVM class loading guarantees thread-safety)
public class DatabaseManager {
    private static final DatabaseManager INSTANCE = new DatabaseManager(); // Thread-safe!
    private final DataSource dataSource;

    private DatabaseManager() {
        // Private constructor
        this.dataSource = new HikariDataSource(buildConfig());
    }

    private HikariConfig buildConfig() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:postgresql://localhost:5432/mydb");
        config.setMaximumPoolSize(20);
        return config;
    }

    public static DatabaseManager getInstance() { return INSTANCE; }
    public DataSource getDataSource() { return dataSource; }
}
```

### Medium — Design an Immutable Money Class
```java
// Immutable value object — final fields, all set in constructor
public final class Money {
    private final long amountInMinorUnits; // Cents/Pence — avoid floating point!
    private final String currencyCode;

    public Money(long amountInMinorUnits, String currencyCode) {
        if (amountInMinorUnits < 0)
            throw new IllegalArgumentException("Amount cannot be negative: " + amountInMinorUnits);
        if (currencyCode == null || currencyCode.length() != 3)
            throw new IllegalArgumentException("Invalid currency code: " + currencyCode);
        this.amountInMinorUnits = amountInMinorUnits;
        this.currencyCode = currencyCode.toUpperCase();
    }

    // Static factories for readability
    public static Money ofUSD(long cents) { return new Money(cents, "USD"); }
    public static Money ofEUR(long cents) { return new Money(cents, "EUR"); }
    public static Money zero(String currency) { return new Money(0, currency); }

    // Operations return new instances (immutable)
    public Money add(Money other) {
        requireSameCurrency(other);
        return new Money(this.amountInMinorUnits + other.amountInMinorUnits, this.currencyCode);
    }

    public Money subtract(Money other) {
        requireSameCurrency(other);
        long result = this.amountInMinorUnits - other.amountInMinorUnits;
        if (result < 0) throw new ArithmeticException("Cannot subtract to negative amount");
        return new Money(result, this.currencyCode);
    }

    public Money multiply(int factor) {
        if (factor < 0) throw new IllegalArgumentException("Factor must be non-negative");
        return new Money(this.amountInMinorUnits * factor, this.currencyCode);
    }

    private void requireSameCurrency(Money other) {
        if (!this.currencyCode.equals(other.currencyCode))
            throw new IllegalArgumentException(
                "Currency mismatch: " + this.currencyCode + " vs " + other.currencyCode);
    }

    public long getAmountInMinorUnits() { return amountInMinorUnits; }
    public String getCurrencyCode() { return currencyCode; }
    public double getAmount() { return amountInMinorUnits / 100.0; }

    @Override public boolean equals(Object o) {
        return o instanceof Money m && amountInMinorUnits == m.amountInMinorUnits
            && currencyCode.equals(m.currencyCode);
    }

    @Override public int hashCode() { return Objects.hash(amountInMinorUnits, currencyCode); }
    @Override public String toString() { return currencyCode + " " + String.format("%.2f", getAmount()); }
}
```

### Hard — Builder for Complex HTTP Request
```java
// See earlier HttpRequest.Builder example — production-grade
// Test it:
@Test
void buildRequest_allFields_constructsCorrectly() {
    byte[] body = "{\"key\":\"value\"}".getBytes();
    HttpRequest request = HttpRequest.builder("https://api.example.com/orders")
        .method("POST")
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer token123")
        .body(body)
        .timeout(Duration.ofSeconds(15))
        .maxRetries(5)
        .build();

    assertThat(request.getUrl()).isEqualTo("https://api.example.com/orders");
    assertThat(request.getMethod()).isEqualTo("POST");
    assertThat(request.getHeaders()).containsKey("Content-Type");
    assertThat(request.getBody()).isPresent();
    assertThat(request.getTimeout()).isEqualTo(Duration.ofSeconds(15));
}

@Test
void buildRequest_postWithoutBody_throws() {
    assertThatThrownBy(() ->
        HttpRequest.builder("https://api.example.com")
            .method("POST")
            .build()
    ).isInstanceOf(IllegalStateException.class)
     .hasMessageContaining("body");
}
```

---

## 22. Revision Sheet

| Concept | Key Rule |
|---------|----------|
| Constructor purpose | Initialize state + enforce invariants |
| `new` vs constructor | `new` allocates memory; constructor initializes it |
| Default constructor | Provided by compiler ONLY if zero constructors written |
| Constructor chaining | `this()` = same class; `super()` = parent class; both must be first line |
| Private constructor | Singleton, Builder, Factory patterns |
| `final` fields | Must be set in constructor; JVM safe publication guarantee |
| Defensive copy | Copy mutable inputs in constructor and mutable returns in getters |
| `this` escape | Never publish `this` before constructor completes |
| No virtual method call | Subclass fields not initialized yet — undefined behavior |
| Copy constructor | Explicit deep copy; better than `clone()` |
| Builder pattern | For classes with > 4 parameters |
| Serialization bypass | `ObjectInputStream` skips constructor; use `readObject()` for validation |

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| What does `new` do vs constructor? | `new` allocates and zeroes memory; constructor initializes fields |
| When is the default constructor provided? | Only when you write zero constructors yourself |
| Can constructor return a value? | No return type. `new` returns the reference. |
| `this()` in constructor? | Calls another constructor in same class. Must be first line. |
| `super()` in constructor? | Calls parent constructor. Must be first line. Implicit if not written. |
| Why private constructor? | Singleton, Factory, Builder — control object creation |
| `final` field safe publication? | JVM guarantees all threads see final fields after constructor completes |
| Why no virtual methods in constructor? | Subclass fields not yet initialized when parent constructor calls override |
| Copy constructor vs `clone()`? | Copy constructor: type-safe, explicit, deep. clone(): shallow, error-prone. |
| What is `this` escape? | Publishing `this` before constructor completes — other threads see partial object |
| RAII in C++? | Resource Acquisition Is Initialization — constructor acquires, destructor releases |
| Defensive copy in constructor? | Copy mutable parameters before storing (arrays, lists, sets) |
| Serialization and constructor? | Java deserialization bypasses constructor; use readObject() for validation |
| Builder vs constructor? | Builder: readable named params, optional params, validation in build() |
| C++ initializer list benefit? | Initializes directly via copy constructor — more efficient than default + assignment |
| `static` field in constructor? | Avoid — static fields are shared; static blocks are better |
| Exception in constructor? | Object creation fails; reference never assigned; C++ destructs already-constructed members |
| Why validate in constructor? | Fail-Fast — invalid state never enters system; clearer error at creation time |
| TLAB allocation cost? | ~5-10ns — extremely fast bump pointer in Thread Local Allocation Buffer |
| Can abstract class have constructor? | Yes — called via subclass `super()` to initialize abstract class's shared fields |

---

## 24. Cheat Sheet

### Top 20 Facts
1. Constructor = same name as class, no return type, runs on `new`
2. `new` allocates + zeroes memory; constructor initializes that memory
3. Default constructor provided only if you write zero constructors
4. If you write ANY constructor, default no-arg is removed
5. `this()` must be first line; calls another constructor in same class
6. `super()` must be first line; calls parent constructor (implicit if not written)
7. `this()` and `super()` are mutually exclusive — only one as first line
8. Private constructor = Singleton, Builder, Factory patterns
9. `final` fields MUST be set in constructor; JVM safe publication guarantee
10. Never call virtual/overridable methods in constructor — subclass not initialized
11. Never escape `this` from constructor — other threads see partial object
12. Defensive copy mutable parameters before storing
13. Copy constructor: better than `clone()` — type-safe, explicit deep copy
14. Builder pattern: solution to telescoping constructors (4+ params)
15. Java deserialization bypasses constructor — use `readObject()` for validation
16. C++ initializer list: more efficient than assignment in body
17. RAII: C++ constructor acquires resource; destructor releases automatically
18. Validate ALL inputs in constructor — fail early with descriptive exception
19. Heavy work (DB/network) in constructor = Fail-Fast at startup (acceptable)
20. Static factory methods: named constructors, can return subtypes, can cache

---

## 25. Final Interview Summary

### Night-Before Revision
1. ⭐ Constructor = initialize state + enforce invariants; no return type
2. ⭐ `new` allocates + zeroes; constructor initializes
3. ⭐ Default constructor: only if you write zero constructors
4. ⭐ `this()` = same-class chain; `super()` = parent class; both must be first line
5. ⭐ Private constructor = Singleton, Builder, Factory
6. ⭐ `final` fields = JVM safe publication guarantee for all threads
7. ⭐ Never call overridable methods in constructor
8. ⭐ Never escape `this` before constructor completes
9. ⭐ Defensive copy mutable params (arrays, lists)
10. ⭐ Builder pattern for > 4 parameters (readable, safe, immutable)
