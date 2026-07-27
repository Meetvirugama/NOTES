# Encapsulation — Industry-Level Interview Notes

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
Encapsulation = **bundling data and methods together** + **hiding internal details** from the outside world. The object controls access to its own state.

### Technically
Encapsulation is the OOP principle of:
1. **Data hiding**: Restricting direct access to internal fields using access modifiers (`private`, `protected`, `public`)
2. **Data bundling**: Keeping related data (fields) and the operations on them (methods) within the same class
3. **Controlled access**: Providing `getter`/`setter` methods (or property accessors) that validate and control mutations

### From an Interviewer's Perspective
> "I'm looking for: access modifiers and their exact scope, why getters/setters aren't always the answer, immutability as the ultimate encapsulation, how encapsulation enables invariant enforcement, and the real difference between encapsulation and abstraction. Saying 'private fields with getters/setters' is the beginner answer."

⭐ **Key insight**: Encapsulation isn't just about making fields private. It's about **enforcing class invariants** and **controlling state transitions**.

⭐ **Encapsulation vs Abstraction**:
- **Encapsulation**: HOW data is hidden (mechanism — access modifiers)
- **Abstraction**: WHAT complexity is hidden (concept — interfaces/abstract classes)

---

## 2. Why It Exists

### Problem Without Encapsulation
```c
// No encapsulation: anyone can corrupt state
struct BankAccount {
    double balance;  // Public!
    char owner[50];
};

struct BankAccount alice = {"Alice", 1000.0};
alice.balance = -999999.0;   // Legal! Corrupts state!
alice.balance += 50.0;       // No validation — overdraft ignored!
```

### What Happens Without It
- **No invariant enforcement**: balance can go negative, age can be 999, email can be invalid
- **Hidden dependencies**: Other code depends on internal representation — can't change internals without breaking everything
- **Impossible to maintain**: To add validation later, must find every direct field access and update it

### Real Software Examples
- **Java's `ArrayList`**: Internal array, size, modCount are all private. You can't accidentally corrupt the array.
- **`java.time.LocalDate`**: Day, month, year stored privately — `LocalDate.of(2024, 2, 30)` throws exception rather than silently creating invalid date
- **Spring Security's `BCryptPasswordEncoder`**: Work factor, salt generation — all internal. You just call `encode(password)`.
- **Android's `SharedPreferences`**: Internal XML/database storage hidden. You just call `getString(key, default)`.

---

## 3. Internal Working

### Access Modifiers — Exact Scope

**Java:**
| Modifier | Same Class | Same Package | Subclass (diff pkg) | Other Classes |
|---------|-----------|--------------|--------------------|----|
| `private` | ✅ | ❌ | ❌ | ❌ |
| (default/package) | ✅ | ✅ | ❌ | ❌ |
| `protected` | ✅ | ✅ | ✅ | ❌ |
| `public` | ✅ | ✅ | ✅ | ✅ |

**C++:**
| Modifier | Same Class | Friend Class | Subclass | Other Classes |
|---------|-----------|--------------|----------|---|
| `private` | ✅ | ✅ | ❌ | ❌ |
| `protected` | ✅ | ✅ | ✅ | ❌ |
| `public` | ✅ | ✅ | ✅ | ✅ |

**Python** (convention-based, no enforcement):
| Convention | Visibility |
|-----------|-----------|
| `field` | Public |
| `_field` | Protected (convention — still accessible) |
| `__field` | "Private" — name mangled to `_ClassName__field` |

### How Name Mangling Works in Python
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Becomes _BankAccount__balance internally

acc = BankAccount(1000)
# acc.__balance  -> AttributeError
print(acc._BankAccount__balance)  # 1000 -- Python doesn't enforce!
```

### Memory Layout — Encapsulation at Machine Level
At the machine level, there are **no access modifiers**. They are a **compile-time construct** only.

```
BankAccount object in heap:
+-------------------+
| vptr              |  (if virtual methods)
+-------------------+
| balance (double)  |  <- labeled "private" in source
+-------------------+  <- CPU doesn't care about labels
| owner (String*)   |  <- same memory either way
+-------------------+
```

C++'s `private` means: **the compiler will reject code that accesses this field outside the class**. At the binary level, it's just a byte at an offset.

⚠️ **Interview trap**: "Private fields in Java can be accessed via reflection" — correct. Encapsulation is a compile-time contract, not a security mechanism.

### Getter/Setter Internals
Getters/setters are just regular methods. The benefit isn't syntax — it's:
1. **Validation on set**: `setAge(age)` can reject negative values
2. **Computation on get**: `getFullName()` can combine firstName + lastName
3. **Lazy initialization**: `getConnection()` creates connection on first call
4. **Change notification**: `setName()` can fire events (Observer pattern)
5. **Thread safety**: `synchronized` getter/setter for atomic access

---

## 4. Syntax

### C++
```cpp
#include <iostream>
#include <string>
#include <stdexcept>
using namespace std;

class BankAccount {
private:                        // Access modifier
    double balance;
    string owner;
    int transactionCount;

public:
    // Constructor — sets initial state with validation
    BankAccount(const string& owner, double initialBalance)
        : owner(owner), balance(0), transactionCount(0) {
        if (initialBalance < 0)
            throw invalid_argument("Initial balance cannot be negative");
        this->balance = initialBalance;
    }

    // Getter (accessor)
    double getBalance() const { return balance; } // const = doesn't modify state
    string getOwner() const { return owner; }

    // Controlled mutation — validates before modifying
    void deposit(double amount) {
        if (amount <= 0) throw invalid_argument("Deposit must be positive");
        balance += amount;
        ++transactionCount;
    }

    bool withdraw(double amount) {
        if (amount <= 0 || amount > balance) return false;
        balance -= amount;
        ++transactionCount;
        return true;
    }

    // No setter for balance! Controlled only through deposit/withdraw
    // No setter for transactionCount! Read-only internally computed
};

int main() {
    BankAccount acc("Alice", 1000.0);
    acc.deposit(500.0);
    acc.withdraw(200.0);
    // acc.balance = -99999; // Compile error! Private!
    cout << acc.getBalance() << endl; // 1300.0
}
```

### Java
```java
public class BankAccount {
    private final String accountId;  // Immutable after construction
    private final String owner;
    private double balance;
    private int transactionCount;

    public BankAccount(String owner, double initialBalance) {
        if (initialBalance < 0)
            throw new IllegalArgumentException("Initial balance cannot be negative");
        this.accountId = UUID.randomUUID().toString();
        this.owner = owner;
        this.balance = initialBalance;
        this.transactionCount = 0;
    }

    // Getters — no setters for immutable fields
    public String getAccountId() { return accountId; }
    public String getOwner() { return owner; }
    public double getBalance() { return balance; }

    // Business methods enforce invariants
    public void deposit(double amount) {
        if (amount <= 0) throw new IllegalArgumentException("Amount must be positive");
        this.balance += amount;
        this.transactionCount++;
    }

    public void withdraw(double amount) {
        if (amount <= 0) throw new IllegalArgumentException("Amount must be positive");
        if (amount > this.balance) throw new IllegalStateException("Insufficient funds");
        this.balance -= amount;
        this.transactionCount++;
    }

    // Computed property — no direct field for this
    public boolean isPremium() { return balance > 100_000; }

    @Override
    public String toString() {
        return String.format("BankAccount[id=%s, owner=%s, balance=%.2f]",
            accountId, owner, balance);
    }
}
```

### Python
```python
class BankAccount:
    def __init__(self, owner: str, initial_balance: float):
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self._owner = owner          # "Protected" convention
        self.__balance = initial_balance  # "Private" — name-mangled
        self.__transaction_count = 0

    # Property decorator — Pythonic getter
    @property
    def balance(self) -> float:
        return self.__balance

    @property
    def owner(self) -> str:
        return self._owner

    # Property setter with validation
    @balance.setter
    def balance(self, value: float) -> None:
        raise AttributeError("Cannot set balance directly. Use deposit/withdraw.")

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.__balance += amount
        self.__transaction_count += 1

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.__balance:
            raise ValueError("Insufficient funds")
        self.__balance -= amount
        self.__transaction_count += 1

    def __repr__(self) -> str:
        return f"BankAccount(owner='{self._owner}', balance={self.__balance:.2f})"

# Usage
acc = BankAccount("Alice", 1000.0)
acc.deposit(500.0)
print(acc.balance)        # 1500.0 (via property)
# acc.balance = 0         # Raises AttributeError!
# acc.__balance = 0       # Raises AttributeError!
```

---

## 5. Visual Diagrams

### Encapsulation as a Capsule
```
+=============================================+
|           BankAccount                       |
|  +-----------[PRIVATE]------------------+  |
|  |  balance: double                     |  |
|  |  owner: String                       |  |
|  |  transactionCount: int               |  |
|  +--------------------------------------+  |
|                                             |
|  +----------[PUBLIC API]----------------+  |
|  |  deposit(amount)     <- validates    |  |
|  |  withdraw(amount)    <- validates    |  |
|  |  getBalance()        <- read only    |  |
|  |  isPremium()         <- computed     |  |
|  +--------------------------------------+  |
+=============================================+

External code can ONLY interact through the PUBLIC API.
Internal state is protected by the class itself.
```

### Without Encapsulation vs With Encapsulation
```
WITHOUT Encapsulation:
External Code ─────────────────────────> balance field (direct write)
              ─────────────────────────> owner field (direct write)
              [No validation, no control, no invariants]

WITH Encapsulation:
                  +--[BankAccount]-------+
External Code --->| deposit(amount)      |---> validates --> balance
              --->| withdraw(amount)     |---> validates --> balance
              --->| getBalance()         |<--- read only <-- balance
                  +---------------------+
              [All paths controlled, invariants enforced]
```

### Access Modifier Scope (Java)
```
+--[com.bank]-------------------------------+
|  +--[BankAccount]---+  +--[AuditService]-+|
|  | private:         |  | package visible ||
|  |  __balance       |  |  (default accs) ||
|  | protected:       |  |                 ||
|  |  _txnCount       |  +-----------------+|
|  | public:          |                      |
|  |  getBalance()    |                      |
|  +------------------+                      |
+--------------------------------------------+
+--[com.fraud]-------------------------------+
|  +--[SavingsAccount extends BankAccount]--+|
|  | Can access: protected _txnCount         ||
|  | Cannot: private __balance               ||
|  +-----------------------------------------+|
+--------------------------------------------+
```

---

## 6. Real World Analogy

### Car (Control Panel = Public API, Engine = Private)
- Public interface: steering wheel, accelerator, brake, gear shift
- Private internals: fuel injection, engine timing, ECU logic, ABS algorithms
- You never touch the engine directly — you use the control interface which validates your actions (can't accelerate in reverse without braking first)

### Bank (Goldman Sachs)
- Account balance = private field
- Teller/app = public API (deposit, withdraw with validation)
- You can't reach into the database and change your balance directly — every operation goes through validated transaction logic

### Hospital
- Patient's medical record = private (HIPAA protected)
- Access via nurses/doctors with authentication = public API
- You can't directly edit your own diagnosis — it goes through validated medical workflows

### E-Commerce
- Inventory count = private field in `Product`
- `order.placeOrder()` = public API that atomically decrements inventory
- Direct inventory manipulation bypassed — race conditions and overselling prevented

### Operating System
- Memory pages = private to each process
- System calls = public API (`read()`, `write()`, `malloc()`)
- Process can't access another process's memory — kernel enforces via page tables (hardware encapsulation!)

### Company Organization
- Employee salary = private (HR only)
- `getCompensation()` = API (returns range or role level, not exact figure)
- Direct database access restricted to payroll system only

---

## 7. Interview Explanation

### 30 Seconds
> "Encapsulation is the bundling of data and methods within a class, with access modifiers controlling who can see what. Private fields hide state. Public methods are the validated interface for state mutation. It enforces class invariants — the class guarantees its internal state is always consistent."

### 1 Minute
> "Encapsulation serves two purposes: hiding how state is stored (so you can change implementation without breaking callers), and controlling how state is modified (so you can enforce invariants). A `BankAccount` with a private `balance` field and a `deposit()` method ensures balance never goes negative — you simply cannot bypass that validation without the public API. Making a field public removes that guarantee forever."

### 3 Minutes
> "Encapsulation is deeper than just 'make fields private'. Let me give you three levels.

> First, access modifiers. Java has four: private (class only), package-private (default — same package), protected (package + subclasses), public (everyone). C++ adds `friend` for fine-grained access grants. Python uses convention (`_`) and name mangling (`__`) — but has no compile-time enforcement.

> Second, invariant enforcement. This is the real purpose. A `LocalDate` class ensures day is between 1–31, month between 1–12, and that February 30th can never exist. These invariants are only possible if the fields are private and all mutations go through validated methods. With public fields, any invariant can be broken at any time.

> Third, implementation independence. If `balance` is private, you can change it from `double` to `BigDecimal` (for precision) without changing the public API. Callers don't know or care. If it's public, that change breaks every line of code that accesses the field directly.

> The key senior-level distinction: encapsulation ≠ security. In Java, reflection can access private fields. Access modifiers are a compile-time contract about intended usage, not a security barrier. And immutability is the strongest form of encapsulation — if state can't change, there's nothing to protect."

---

## 8. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What are the four access modifiers in Java? | private, default (package), protected, public |
| Why make fields private? | To control access, enforce invariants, allow internal refactoring |
| What is a getter/setter? | Methods that provide controlled read/write access to private fields |
| Is Python encapsulation enforced? | No — convention only (`_`, `__` with name mangling, not enforced) |
| Encapsulation vs Abstraction? | Encapsulation = data hiding mechanism; Abstraction = hiding complexity/implementation |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| When should you NOT have a setter? | When field is immutable after construction (final fields, IDs, timestamps) |
| What is the difference between `protected` and `private`? | Protected: accessible to subclasses; Private: strictly class-only |
| What is name mangling in Python? | `__field` becomes `_ClassName__field` — discourages accidental access |
| Can reflection bypass private access in Java? | Yes — `field.setAccessible(true)` bypasses private; encapsulation is compile-time only |
| What is an invariant? | A condition that must always be true about an object's state |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| Encapsulation enables "Tell Don't Ask" — explain | Instead of asking for data and deciding externally, tell the object what to do and let it handle internally |
| How does encapsulation relate to the Open-Closed Principle? | Encapsulation creates stable interfaces; internals can change (open for internal change) without breaking callers (closed externally) |
| What is the Law of Demeter? | A method should only call methods on its own fields, parameters, or objects it creates — not on return values of other methods |
| Why is immutability the strongest form of encapsulation? | Immutable state can't be corrupted — no need for defensive copies, no race conditions |
| What is the problem with returning mutable collections from getters? | Callers can modify the returned collection, bypassing encapsulation |

### Expert
| Question | Expected Answer |
|----------|-----------------|
| How does encapsulation relate to memory safety? | Private fields prevent external aliasing; reduces corruption risk; maps to Rust's ownership model |
| What is defensive copying and when is it needed? | Return `Collections.unmodifiableList(list)` or `new ArrayList<>(list)` to prevent external mutation of internal collections |
| How does Java's module system (Java 9+) extend encapsulation? | Module-level access control: packages can be hidden even from reflection unless exported |
| What is the difference between encapsulation and information hiding? | Information hiding = broader concept (hiding any implementation detail); encapsulation = specific OOP mechanism to achieve it |

### 💼 Google Level
> *"You're designing a distributed counter class that will be accessed by 10,000 threads. How does encapsulation help you evolve the implementation from a simple int to an AtomicLong to a distributed counter, without changing the public API?"*

### 💼 Goldman Sachs Level
> *"A `TradeOrder` object is accessed by risk engines, pricing engines, and reporting systems. How do you encapsulate its state to ensure different subsystems see only what they need?"*

---

## 9. Coding Examples

### Basic Example — Proper Encapsulation
```java
public class Temperature {
    private double celsius;  // Internal representation

    public Temperature(double celsius) {
        if (celsius < -273.15)
            throw new IllegalArgumentException("Temperature below absolute zero!");
        this.celsius = celsius;
    }

    // Multiple views of same data — no separate fields!
    public double getCelsius() { return celsius; }
    public double getFahrenheit() { return celsius * 9/5 + 32; }
    public double getKelvin() { return celsius + 273.15; }

    // Controlled mutation
    public void setCelsius(double celsius) {
        if (celsius < -273.15)
            throw new IllegalArgumentException("Temperature below absolute zero!");
        this.celsius = celsius;
    }
}
```

### Intermediate — Immutable Value Object
```java
// Immutable = ultimate encapsulation (no setters at all!)
public final class Money {
    private final long amountInCents;
    private final Currency currency;

    public Money(long amountInCents, Currency currency) {
        if (amountInCents < 0) throw new IllegalArgumentException("Amount cannot be negative");
        this.amountInCents = amountInCents;
        this.currency = Objects.requireNonNull(currency, "Currency required");
    }

    public long getAmountInCents() { return amountInCents; }
    public Currency getCurrency() { return currency; }

    // Operations return NEW instances — original unchanged
    public Money add(Money other) {
        requireSameCurrency(other);
        return new Money(this.amountInCents + other.amountInCents, this.currency);
    }

    public Money multiply(int factor) {
        if (factor < 0) throw new IllegalArgumentException("Factor must be non-negative");
        return new Money(this.amountInCents * factor, this.currency);
    }

    private void requireSameCurrency(Money other) {
        if (!this.currency.equals(other.currency))
            throw new IllegalArgumentException("Currency mismatch");
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Money m)) return false;
        return amountInCents == m.amountInCents && currency.equals(m.currency);
    }

    @Override public int hashCode() { return Objects.hash(amountInCents, currency); }
    @Override public String toString() { return currency + " " + amountInCents/100.0; }
}
```

### Advanced — Defensive Copying
```java
public class Portfolio {
    private final List<Stock> stocks;  // Mutable list!

    public Portfolio(List<Stock> stocks) {
        // Defensive copy on construction — don't trust caller's list
        this.stocks = new ArrayList<>(stocks);
    }

    // Return defensive copy — prevent external mutation
    public List<Stock> getStocks() {
        return Collections.unmodifiableList(stocks);  // Read-only view
    }

    // Controlled mutation
    public void addStock(Stock stock) {
        Objects.requireNonNull(stock);
        stocks.add(stock);
    }

    public boolean removeStock(String symbol) {
        return stocks.removeIf(s -> s.getSymbol().equals(symbol));
    }
}
```

### Production — Builder + Encapsulation
```java
// Builder pattern + deep encapsulation for complex objects
public final class HttpRequest {
    private final String url;
    private final String method;
    private final Map<String, String> headers;
    private final byte[] body;
    private final Duration timeout;

    private HttpRequest(Builder builder) {  // Private constructor!
        this.url = builder.url;
        this.method = builder.method;
        this.headers = Collections.unmodifiableMap(new HashMap<>(builder.headers));
        this.body = builder.body != null ? builder.body.clone() : null; // Defensive copy
        this.timeout = builder.timeout;
    }

    // Only getters — fully immutable after construction
    public String getUrl() { return url; }
    public String getMethod() { return method; }
    public Map<String, String> getHeaders() { return headers; } // Unmodifiable
    public Optional<byte[]> getBody() {
        return Optional.ofNullable(body != null ? body.clone() : null); // Return copy!
    }

    public static Builder builder(String url) { return new Builder(url); }

    public static class Builder {
        private final String url;
        private String method = "GET";
        private final Map<String, String> headers = new HashMap<>();
        private byte[] body;
        private Duration timeout = Duration.ofSeconds(30);

        private Builder(String url) { this.url = Objects.requireNonNull(url); }

        public Builder method(String method) { this.method = method; return this; }
        public Builder header(String key, String value) { headers.put(key, value); return this; }
        public Builder body(byte[] body) { this.body = body.clone(); return this; } // Defensive copy
        public Builder timeout(Duration timeout) { this.timeout = timeout; return this; }
        public HttpRequest build() { return new HttpRequest(this); }
    }
}
```

### Interview — Fix Broken Encapsulation
```java
// BROKEN: Find and fix all encapsulation issues
public class User {
    public String name;           // Issue 1: public field
    public List<String> roles = new ArrayList<>();  // Issue 2: mutable public
    private String password;
    public String getPassword() { return password; } // Issue 3: exposes secret!

    public void setAge(int age) { this.age = age; }  // Issue 4: no validation
    private int age;
}

// FIXED:
public class User {
    private String name;
    private final List<String> roles;
    private String passwordHash;  // Never store plaintext!
    private int age;

    public User(String name, int age, String password) {
        this.name = Objects.requireNonNull(name);
        setAge(age); // Reuse validation
        this.passwordHash = BCrypt.hashpw(password, BCrypt.gensalt());
        this.roles = new ArrayList<>();
    }

    public String getName() { return name; }

    public List<String> getRoles() {
        return Collections.unmodifiableList(roles); // Defensive!
    }

    public boolean verifyPassword(String plaintext) {
        return BCrypt.checkpw(plaintext, passwordHash); // Never expose hash!
    }

    public void setAge(int age) {
        if (age < 0 || age > 150)
            throw new IllegalArgumentException("Invalid age: " + age);
        this.age = age;
    }

    public int getAge() { return age; }
}
```

---

## 10. Common Mistakes

### ⚠️ Mistake 1: Getter Returns Mutable Internal Collection
```java
class Team {
    private List<Player> players = new ArrayList<>();

    public List<Player> getPlayers() { return players; } // WRONG! Caller can modify!

    // Fix: Return unmodifiable view
    public List<Player> getPlayers() {
        return Collections.unmodifiableList(players);
    }
}

// The bug:
team.getPlayers().clear(); // Clears internal list! Encapsulation broken!
```

### ⚠️ Mistake 2: Public Setter for Every Private Field
```java
// Anti-pattern: "JavaBean" setters without thought
class Order {
    private String status;
    private double amount;

    public void setStatus(String status) { this.status = status; } // Any string!
    public void setAmount(double amount) { this.amount = amount; }  // Negative allowed!
}

// Better: state machine via meaningful methods
class Order {
    private OrderStatus status = OrderStatus.PENDING;
    private final double amount;

    public void confirm() {
        if (status != OrderStatus.PENDING) throw new IllegalStateException();
        status = OrderStatus.CONFIRMED;
    }

    public void cancel() {
        if (status == OrderStatus.SHIPPED) throw new IllegalStateException("Cannot cancel shipped order");
        status = OrderStatus.CANCELLED;
    }
}
```

### ⚠️ Mistake 3: Constructor Doesn't Validate
```java
class EmailAddress {
    private String email;

    public EmailAddress(String email) {
        this.email = email; // No validation! "not_an_email" passes through!
    }

    // Fix: validate in constructor
    public EmailAddress(String email) {
        if (!email.matches("^[\\w._%+-]+@[\\w.-]+\\.[A-Z]{2,6}$"))
            throw new IllegalArgumentException("Invalid email: " + email);
        this.email = email;
    }
}
```

### ⚠️ Mistake 4: Exposing Internal Array (not copying)
```java
class Config {
    private int[] thresholds;

    public Config(int[] thresholds) {
        this.thresholds = thresholds; // WRONG: Caller still holds reference!
    }

    public int[] getThresholds() {
        return thresholds; // WRONG: Caller can modify array!
    }

    // Fix: defensive copies
    public Config(int[] thresholds) { this.thresholds = thresholds.clone(); }
    public int[] getThresholds() { return thresholds.clone(); }
}
```

### ⚠️ Mistake 5: Using `protected` When `private` Is Correct
```java
class BankAccount {
    protected double balance; // WRONG: any subclass can directly mutate balance!
    // Fix: private with protected helper methods if subclass needs controlled access
    private double balance;
    protected void adjustBalance(double delta) {
        if (balance + delta < 0) throw new IllegalStateException("Overdraft");
        balance += delta;
    }
}
```

---

## 11. Best Practices

### Design
- **Make fields private by default** — only promote to protected/public when needed
- **No setter without a reason** — prefer business-meaningful methods over generic setters
- **Immutable when possible** — `final` fields, no setters = strongest encapsulation
- **Return defensive copies** of mutable objects from getters
- **Validate in constructors** — fail fast before object enters invalid state
- **Tell Don't Ask** — ask the object to do something, don't ask for data and do it yourself

### Java Best Practices
- Use `final` fields for values that don't change after construction
- Use `Collections.unmodifiableList()` or `List.of()` for collection fields
- Use `Optional` for potentially-absent getters instead of returning null
- Use `record` (Java 16+) for pure data classes — auto-generated immutable encapsulation

### 🚀 Performance Notes
- Getters/setters are just method calls — JIT inlines them to zero overhead
- Immutable objects can be freely shared without synchronization (thread-safe)
- `final` fields enable JIT optimizations (constant folding)

---

## 12. Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Getter call | O(1) | JIT-inlined — equivalent to field access |
| Setter with validation | O(1) | Simple comparison + assignment |
| Defensive copy (collection) | O(n) | Must copy all elements |
| Name mangling lookup | O(1) | Python dict lookup |

### Memory Impact
- **No extra memory**: Encapsulation is purely a compile-time concept
- **Defensive copies**: Additional O(n) memory for each defensive copy
- **Immutable objects**: Can be cached/pooled freely (String pool example)

---

## 13. Advantages

| Advantage | Example |
|-----------|---------|
| **Invariant enforcement** | `BankAccount.balance` can never go negative |
| **Implementation independence** | Change `double` to `BigDecimal` internally — API unchanged |
| **Controlled mutation** | Only validated paths can modify state |
| **Maintainability** | Internal refactoring doesn't break callers |
| **Testability** | Mock setters/getters to test specific scenarios |
| **Thread safety** | Immutable encapsulated state = zero synchronization needed |
| **API stability** | Public API is the contract; private internals can change freely |

---

## 14. Disadvantages

| Disadvantage | When It Hurts |
|-------------|---------------|
| **Boilerplate** | Dozens of getters/setters for simple data classes |
| **Indirection** | Extra method call (though JIT eliminates this) |
| **Reflection bypass** | Encapsulation is not a security mechanism |
| **Serialization complexity** | Private fields need special handling in serialization |
| **Test access** | Testing private methods requires reflection or visibility changes |

---

## 15. Comparison Table

### Encapsulation vs Abstraction

| Aspect | Encapsulation | Abstraction |
|--------|--------------|-------------|
| Purpose | Data hiding + bundling | Complexity hiding |
| Mechanism | Access modifiers | Interfaces, abstract classes |
| What it hides | HOW data is stored | WHAT implementation does |
| Focus | Internal state protection | External interface design |
| Example | `private double balance` | `interface Payment { void pay(); }` |
| Question | "Who can access this?" | "What does this expose?" |

### Public Field vs Private + Getter

| Aspect | Public Field | Private + Getter |
|--------|-------------|-----------------|
| Validation | None | Yes |
| Change notification | None | Can add Observer |
| Lazy init | No | Yes (compute on first access) |
| Thread safety | Raw access | Can synchronize |
| Refactoring | Breaking change | Non-breaking |
| Performance | Slightly faster | Same after JIT |

### Setter vs Business Method

| Aspect | Setter (`setStatus`) | Business Method (`confirm()`) |
|--------|---------------------|-------------------------------|
| Validation | Generic/manual | Built-in to method |
| State machine | No | Yes |
| Intent clarity | Low | High |
| Error handling | External | Internal |
| Example | `order.setStatus("CONFIRMED")` | `order.confirm()` |

---

## 16. Design Pattern Connection

| Pattern | Encapsulation Role |
|---------|-------------------|
| **Builder** | Private constructor; state exposed only after build() |
| **Singleton** | Private constructor + private instance; public getInstance() |
| **Facade** | Hides complex subsystem behind simple public interface |
| **Proxy** | Wraps target; controls access to it |
| **Iterator** | Hides internal collection structure; exposes traversal only |
| **Observer** | Hides subscriber list; controlled via register/notify |
| **Memento** | State object is opaque to originator's caretaker |
| **State** | State transitions encapsulated within state objects |

---

## 17. System Design Connection

### Microservices
- **Service encapsulation**: Each microservice owns its database — no other service can directly access it
- **API contract**: Only REST/gRPC API is public; internal tables are private to service
- **Event publishing**: Internal state changes published as events — consumers see only events, not internals

### REST APIs
- **Response DTOs**: Don't expose JPA entities directly — use DTOs that expose only necessary fields
- **API versioning**: Encapsulated implementation can change; versioned API is the stable contract
- **Authentication**: Password hash never returned in API response — encapsulated from clients

### Databases
- **Stored procedures**: Encapsulate complex queries — application only sees the procedure interface
- **Views**: Encapsulate complex joins — expose simplified virtual table
- **Row-level security**: Database enforces data access control — encapsulation at data layer

### Cloud Systems
- **S3 bucket policies**: Object access controlled by bucket policy — internal replication hidden
- **IAM roles**: Encapsulate AWS permissions — services see only what they're allowed to

---

## 18. Multithreading Connection

### Encapsulation Enables Thread Safety
```java
public class ThreadSafeCounter {
    private volatile int count = 0;  // Private volatile for visibility
    private final Object lock = new Object();  // Private lock object

    public synchronized void increment() { count++; } // Controlled mutation
    public int getCount() { return count; }  // Safe read

    // External code cannot access count directly or the lock!
    // Only the controlled increment() path exists
}
```

### Immutability as Thread Safety
```java
public final class CacheKey {
    private final String region;   // final = thread-safe after construction
    private final String key;

    public CacheKey(String region, String key) {
        this.region = region;
        this.key = key;
    }

    // No setters = immutable = freely shareable across threads
    public String getRegion() { return region; }
    public String getKey() { return key; }
}
```

### ⚠️ Encapsulation Doesn't Guarantee Thread Safety
```java
class Counter {
    private int count = 0; // Private, but NOT thread-safe!

    public void increment() { count++; } // i++ is 3 operations: read, increment, write
    // Race condition still exists even with private field!
}
// Fix: AtomicInteger, synchronized, or ReentrantLock
```

---

## 19. Company Interview Perspective

### Google
- "Design a class that encapsulates a rate limiter — what state is private and what's the public API?"
- Questions about invariants in distributed systems (eventual consistency vs strong invariants)
- Builder pattern for complex request objects

### Goldman Sachs
- "Design a `TradeOrder` with risk controls — ensure position limits can't be bypassed"
- Immutability for financial values (Money, Price)
- Encapsulation in event sourcing (state only changed via events)

### Microsoft
- C++ `friend` class vs private access
- COM encapsulation via pure virtual interfaces
- .NET properties (`get; set;`) vs fields

### Amazon
- "How do you ensure your service's database is never directly accessed by other services?"
- Data encapsulation at service boundary level
- Spring Boot: `@Entity` never returned directly; `@ResponseBody` uses DTOs

### Meta
- Python dataclasses and `__slots__` for memory-efficient encapsulation
- Pydantic models for validated encapsulation in Python APIs
- GraphQL: field-level access control (encapsulation at API layer)

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Can private fields be accessed externally in Java? | Yes — via Reflection `field.setAccessible(true)`. Encapsulation is compile-time only. |
| 2 | ⚠️ Is `protected` better than `private` for subclass reuse? | Not always — `protected` exposes to all subclasses and breaks encapsulation more broadly |
| 3 | What is the difference between encapsulation and data hiding? | Data hiding = the concept; Encapsulation = the OOP mechanism to achieve it |
| 4 | ⚠️ Does `private` prevent access via the same class? | No — all instances of the same class can access each other's private fields |
| 5 | What is "Tell Don't Ask"? | Tell object to do something (deposit) rather than asking for data (getBalance) and acting |
| 6 | What is the Law of Demeter? | Don't call methods on objects returned by methods; only on direct collaborators |
| 7 | Why return `List.of()` instead of a regular getter? | `List.of()` is unmodifiable — prevents encapsulation break through returned reference |
| 8 | ⚠️ What is a "leaky abstraction"? | When internal implementation details "leak" through the public interface |
| 9 | Is `StringBuilder` encapsulated? | Yes — internal `char[]` is private; you only access via `append()`, `toString()` |
| 10 | What is immutability's relationship to encapsulation? | Immutability is the strongest form — no setters = no way to break invariants |
| 11 | Can encapsulation be violated through serialization? | Yes — Java's `ObjectInputStream` bypasses constructors, can create invalid state |
| 12 | ⚠️ What's wrong with returning `this.someList` directly? | External code can clear/modify the list, bypassing encapsulation |
| 13 | What is a "getter/setter anti-pattern"? | Mindless setter for every field: same as public field but with more boilerplate |
| 14 | How do Java records relate to encapsulation? | `record` = automatic private final fields + accessors + equals/hashCode — structured encapsulation |
| 15 | What is "package-private" access and when is it useful? | Default (no modifier): visible within package — for internal implementation hiding across files |
| 16 | ⚠️ Can you make a class field public but immutable? | Yes: `public final String name` — can't reassign, but only works for primitives/Strings |
| 17 | What is the difference between `final` and immutable? | `final` means reference can't change; the object it points to may still be mutable |
| 18 | What is C++'s `friend` keyword for? | Grant specific class/function access to private members — fine-grained encapsulation bypass |
| 19 | What is a "value object" and how does it use encapsulation? | Immutable object defined by its values; no setters; all validation in constructor |
| 20 | ⚠️ Reflection can access private fields — does this break encapsulation? | Technically yes, but: it requires explicit opt-in, Java module system restricts it, it's not "normal" access |

---

## 21. Coding Problems

### Easy — Fix Encapsulation
```java
// Identify and fix 3 encapsulation issues:
class Product {
    public String name;
    public double price;
    public int stock;

    public void setPrice(double price) { this.price = price; }
}

// Fixed:
class Product {
    private String name;
    private double price;
    private int stock;

    public Product(String name, double price, int stock) {
        if (price < 0) throw new IllegalArgumentException("Price cannot be negative");
        if (stock < 0) throw new IllegalArgumentException("Stock cannot be negative");
        this.name = Objects.requireNonNull(name);
        this.price = price;
        this.stock = stock;
    }

    public String getName() { return name; }
    public double getPrice() { return price; }
    public int getStock() { return stock; }

    public void updatePrice(double newPrice) {
        if (newPrice < 0) throw new IllegalArgumentException("Price cannot be negative");
        this.price = newPrice;
    }

    public boolean reduceStock(int quantity) {
        if (quantity > stock) return false;
        stock -= quantity;
        return true;
    }
}
```

### Medium — Design an Encapsulated Stack
```java
public class BoundedStack<T> {
    private final Object[] elements;
    private int size;
    private final int capacity;

    public BoundedStack(int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("Capacity must be positive");
        this.capacity = capacity;
        this.elements = new Object[capacity];
        this.size = 0;
    }

    public void push(T element) {
        if (size == capacity) throw new StackOverflowError("Stack is full");
        elements[size++] = element;
    }

    @SuppressWarnings("unchecked")
    public T pop() {
        if (isEmpty()) throw new NoSuchElementException("Stack is empty");
        T element = (T) elements[--size];
        elements[size] = null; // Prevent memory leak!
        return element;
    }

    @SuppressWarnings("unchecked")
    public T peek() {
        if (isEmpty()) throw new NoSuchElementException("Stack is empty");
        return (T) elements[size - 1];
    }

    public boolean isEmpty() { return size == 0; }
    public int size() { return size; }
    public int capacity() { return capacity; }
    // No direct access to elements array!
}
```

### Hard — Design a Thread-Safe Encapsulated Cache
```java
public class BoundedCache<K, V> {
    private final int maxSize;
    private final Map<K, V> store;
    private final ReadWriteLock lock = new ReentrantReadWriteLock();

    public BoundedCache(int maxSize) {
        this.maxSize = maxSize;
        this.store = new LinkedHashMap<>(maxSize, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K,V> e) { return size() > maxSize; }
        };
    }

    public Optional<V> get(K key) {
        lock.readLock().lock();
        try { return Optional.ofNullable(store.get(key)); }
        finally { lock.readLock().unlock(); }
    }

    public void put(K key, V value) {
        lock.writeLock().lock();
        try { store.put(key, Objects.requireNonNull(value)); }
        finally { lock.writeLock().unlock(); }
    }

    public int size() {
        lock.readLock().lock();
        try { return store.size(); }
        finally { lock.readLock().unlock(); }
    }
    // Internal store, lock, and eviction policy are completely hidden
}
```

---

## 22. Revision Sheet

| Concept | Key Point |
|---------|-----------|
| Encapsulation | Bundle data + methods; hide internal state |
| Access modifiers | private < default < protected < public |
| Invariant | Always-true condition about object state |
| Tell Don't Ask | Tell object to do something; don't ask for data and act externally |
| Defensive copy | Copy mutable objects before storing or returning |
| Immutability | Strongest encapsulation — no setters, final fields |
| Getter anti-pattern | Getter returning mutable reference — caller can bypass encapsulation |
| Setter anti-pattern | Setter with no validation — no better than public field |
| Law of Demeter | Only call methods on direct collaborators |
| Reflection | Can bypass private — encapsulation is compile-time, not security |
| Java `record` | Auto-encapsulated immutable data class (Java 16+) |
| Python `__field` | Name mangling — not enforcement |

### Common Pitfalls
- ⚠️ Returning mutable collection from getter
- ⚠️ Constructor without validation
- ⚠️ Protected when private is correct
- ⚠️ Defensive copy forgotten on construction
- ⚠️ `final` reference ≠ immutable object content
- ⚠️ Reflection bypass — not a security mechanism

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| What is encapsulation? | Bundle data + methods; hide internal state via access modifiers |
| 4 access modifiers in Java? | private, default (package), protected, public |
| Encapsulation vs Abstraction? | Encapsulation = HOW data is hidden; Abstraction = WHAT complexity is hidden |
| What is an invariant? | Always-true condition about object's state |
| Defensive copy? | Copy mutable before storing/returning to prevent external mutation |
| Immutability benefit? | No setters = no way to break invariants = thread-safe |
| Tell Don't Ask? | Tell object what to do; don't ask for data and decide externally |
| Law of Demeter? | Only talk to immediate friends; avoid chained method calls |
| Python `__field` enforcement? | No enforcement; just name mangling to `_ClassName__field` |
| Can reflection access private? | Yes — but requires explicit setAccessible(true) |
| `final` vs immutable? | final = reference can't change; object may still be mutable |
| Getter anti-pattern? | Returning mutable reference — caller can modify internal state |
| Setter anti-pattern? | Generic setter without validation = same as public field |
| Java `record`? | Auto-encapsulated immutable data class (Java 16+) |
| Business method vs setter? | Business method = intent + validation; setter = generic write |
| When to NOT have setter? | For immutable fields: ID, creation timestamp, core identity |
| Protected pitfall? | Exposes to all subclasses — breaks encapsulation wider than private |
| C++ `friend`? | Grants specific class access to private members |
| `Collections.unmodifiableList`? | Returns read-only view — prevents external mutation |
| Serialization bypass? | Java ObjectInputStream bypasses constructor — can create invalid state |
| `Optional` in getter? | Prefer over null for possibly-absent values |
| Encapsulation at service level? | Each microservice owns its DB — others access via API only |
| Package-private in Java? | Default access — visible within package; internal implementation |
| `__slots__` in Python? | Restricts instance attributes — enforces structure, saves memory |
| Java module system? | Module-level access control — stronger than package-private |

---

## 24. Cheat Sheet

### Top 20 Facts
1. Encapsulation = data hiding + bundling data with methods
2. Java: private < package-private < protected < public
3. Python: no enforcement — convention only (`_` protected, `__` name-mangled)
4. Access modifiers are compile-time only — reflection bypasses them
5. Immutability is the strongest form of encapsulation
6. `final` on reference ≠ immutable object
7. Defensive copy: clone on construction AND on return
8. Tell Don't Ask: let object act on its own data
9. Law of Demeter: only call methods on direct collaborators
10. Getter returning mutable reference = encapsulation violation
11. Generic setter = same as public field in terms of protection
12. Java `record` auto-encapsulates immutable data (Java 16+)
13. `Collections.unmodifiableList()` prevents external collection mutation
14. Constructor validation = fail-fast before invalid state enters system
15. Encapsulation enables API stability — internals can change freely
16. `protected` exposes to ALL subclasses — consider carefully
17. C++ `friend` = fine-grained private access exception
18. Java 9+ modules: package-level encapsulation even from reflection
19. Serialization can bypass constructors — use `readObject()` validation
20. Thread safety: private fields don't equal thread safety — need synchronization too

### Top 20 Keywords
`private`, `protected`, `public`, `encapsulation`, `invariant`, `defensive copy`, `immutable`, `final`, `getter`, `setter`, `Tell Don't Ask`, `Law of Demeter`, `reflection`, `name mangling`, `value object`, `record`, `unmodifiableList`, `Builder`, `Facade`, `module`

---

## 25. Final Interview Summary

### 5-Minute Revision
- Encapsulation = private fields + controlled public methods
- Java access: private < package < protected < public
- Access modifiers are compile-time — reflection bypasses them
- Immutability = strongest encapsulation (no setters, final fields)
- Defensive copy mutable objects on construction and return
- Validate in constructor — fail fast, never enter invalid state
- Tell Don't Ask — let object act, don't ask for data and act externally
- Getters returning mutable collections = encapsulation violation

### 15-Minute Revision
Add:
- Law of Demeter
- Builder pattern for complex object construction
- Java records (Java 16+)
- Python name mangling vs enforcement
- Thread safety: private ≠ thread-safe (need synchronization separately)
- Encapsulation at service level (microservices)
- Response DTOs vs entity exposure
- Serialization bypass and `readObject()` validation
- Java 9 module system
- Facade, Proxy, Iterator patterns

### Night-Before Interview Revision
1. ⭐ Encapsulation = hide state + control access
2. ⭐ private < package < protected < public (Java)
3. ⭐ Immutability = strongest form (final fields, no setters)
4. ⭐ Defensive copy: on construction + on getter return
5. ⭐ Validate in constructor — invariants from day 1
6. ⭐ Tell Don't Ask — behavioral not data-fetching API
7. ⭐ Getter returning mutable reference = encapsulation violation
8. ⭐ Encapsulation ≠ security (reflection bypasses)
9. ⭐ Java record = auto-encapsulated immutable data
10. ⭐ private field ≠ thread-safe (still need synchronization)
