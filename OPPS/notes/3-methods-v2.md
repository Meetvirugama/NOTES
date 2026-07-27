# Methods — Industry-Level Interview Notes

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
A method is a **named block of code** associated with a class or object that defines a specific behavior. Calling a method executes that block of code.

### Technically
A method is a function bound to a class (static method) or to an instance of a class (instance method). Instance methods implicitly receive a reference to the invoking object (the `this` pointer), giving them access to the object's state. Methods define the **behavioral contract** of a class — they are the API that external code can use to interact with an object.

### From an Interviewer's Perspective
> "Everyone knows a method is a function in a class. I want to hear: static dispatch vs dynamic dispatch, the hidden `this` pointer, JIT method inlining, pure functions vs side-effecting methods, the difference between method overloading (compile-time) and overriding (runtime), Java Memory Model implications for synchronized methods, and why `final` methods enable JIT optimization. Tell me about the `invokespecial` vs `invokevirtual` bytecode distinction."

⭐ **Key Insight**: Methods are not just "code in a class." They form the behavioral contract of an object. Designing good methods = designing good APIs.

---

## 2. Why It Exists

### Problem Without Methods

#### Global Functions on Shared Data (C procedural)
```c
// Pure procedural: functions and data are separate
double balance = 1000.0;

void deposit(double amount) { balance += amount; }    // No validation!
void withdraw(double amount) { balance -= amount; }   // Overdraft allowed!

// Problem: anyone can bypass these functions:
balance = -99999.0;   // Direct mutation — no control!
balance += 1e15;      // Overflow — no validation!

// Problem: doesn't scale — 100 accounts = 100 global variables
double balance1, balance2, ..., balance100;
// Functions need to know which balance to update — chaos!
```

#### What Methods Solve
```java
// OOP: methods bound to object, operate on its state
public class BankAccount {
    private double balance; // Private: ONLY accessible through methods

    public void deposit(double amount) {
        if (amount <= 0) throw new IllegalArgumentException("Amount must be positive");
        this.balance += amount;  // Controlled, validated mutation
    }

    public boolean withdraw(double amount) {
        if (amount <= 0 || amount > balance) return false;
        this.balance -= amount;
        return true;
    }

    public double getBalance() { return balance; }
}

BankAccount acc = new BankAccount("Alice", 1000.0);
acc.balance = -99999.0;  // COMPILE ERROR! Private field.
acc.deposit(-500.0);     // IllegalArgumentException! Method validates.
```

### Real Software Examples
- **`ArrayList.add(E e)`**: Encapsulates array resizing, bounds checking, size tracking — caller just says "add element"
- **`String.substring(int start, int end)`**: Validates start/end bounds, creates new String — hides internal `char[]` management
- **`Thread.start()`**: Registers with OS thread scheduler, allocates native stack — caller just says "start"
- **`HttpClient.send()`**: Handles DNS lookup, TCP handshake, SSL negotiation, HTTP framing — caller provides request, gets response

---

## 3. Internal Working

### The Hidden `this` Pointer
```
Java conceptual model (compiler transformation):

What you write:
    account.deposit(500.0);

What the compiler generates internally:
    BankAccount.deposit(account, 500.0);
    //          ↑ 'this' is a hidden first parameter!

Inside the method:
    void deposit(BankAccount this, double amount) {
        this.balance += amount;  // 'this' refers to 'account'
    }
```

### Static Dispatch vs Dynamic Dispatch

#### Static Dispatch (Early Binding) — Compile-Time
```
Used for: static methods, private methods, final methods, overloaded methods

Method overloading resolution:
    log("message");        → Logger.log(String)   ← compiler decides at compile time
    log(new IOException()) → Logger.log(Exception) ← based on DECLARED argument type

Cost: Zero at runtime — direct call to known address
Bytecode: invokestatic (static), invokespecial (private/final/super)
```

#### Dynamic Dispatch (Late Binding) — Runtime
```
Used for: overridden methods (virtual methods)

Account acc = new SavingsAccount();  // Declared as Account, actually SavingsAccount
acc.calculateInterest();             // Which one runs? Determined at RUNTIME!

Execution:
  1. acc.vptr → SavingsAccount vtable
  2. vtable[calculateInterest offset] → SavingsAccount::calculateInterest address
  3. Jump to SavingsAccount::calculateInterest

Cost: ~5-10ns for vtable lookup
Bytecode: invokevirtual (class methods), invokeinterface (interface methods)
```

### JVM Bytecode Instructions for Method Calls
```
invokestatic:      Static method call. No 'this'. Direct address at link time.
invokespecial:     Constructor calls (super/this()), private methods, super.method() calls.
invokevirtual:     Instance method calls on class types (virtual dispatch via vtable).
invokeinterface:   Instance method calls on interface types (search itable — slightly slower).
invokedynamic:     Lambda expressions, method references (Java 8+). Flexible linkage.
```

### JIT Compiler Method Optimization

#### Method Inlining
```
Interpreted (cold):
    int result = account.getBalance();
    → push frame for getBalance
    → access this.balance
    → pop frame
    → store result
    Cost: ~5ns

After JIT (hot method — called >10,000 times):
    int result = account.balance;   // Inlined! getBalance() body replaces call
    Cost: ~0.5ns

JIT inlining criteria:
  - Method body < ~35 bytecodes (by default)
  - Call site is monomorphic (one type seen)
  - Method is hot (called frequently)
```

#### JIT Devirtualization
```
Before JIT (polymorphic call):
    payment.process(order);
    // Must look up vtable each time → ~10ns

If JIT observes only StripeProcessor instances at this call site:
    // "Monomorphic call site detected"
    // Inline StripeProcessor.process() directly → ~1ns + null check
    // If new type appears: deoptimize, fall back to vtable dispatch

If 2 types seen (bimorphic):
    // JIT generates: if (type == StripeProcessor) inline1 else if (type == PayPalProcessor) inline2
    // Still faster than general vtable lookup

If 3+ types (megamorphic):
    // JIT gives up — uses general vtable dispatch
```

### Stack Frame Layout
```
Call stack during: order.addItem(product, 3)

+--[Current Frame: addItem]----------+
| Local Variables:                   |
|   this = 0x7FAB1234 (order ref)    |
|   product = 0x7FAB5678 (ref)       |
|   quantity = 3 (int)               |
|   result = null (uninitialized)    |
| Operand Stack (eval computation)   |
| Return Address (to caller)         |
+------------------------------------+
| Previous Frame: placeOrder         |
+------------------------------------+
```

---

## 4. Syntax

### Java — All Method Types
```java
public class PaymentService {
    private final PaymentRepository repo;
    private static final int MAX_RETRIES = 3;
    private static final AtomicLong requestCount = new AtomicLong(0);

    public PaymentService(PaymentRepository repo) { this.repo = repo; }

    // 1. INSTANCE METHOD: operates on 'this' object state
    public PaymentResult processPayment(PaymentRequest request) {
        requestCount.incrementAndGet();
        validateRequest(request);           // Private helper
        return executeWithRetry(request);   // Private helper
    }

    // 2. PRIVATE METHOD: helper, not part of public API
    private void validateRequest(PaymentRequest request) {
        Objects.requireNonNull(request, "Request cannot be null");
        if (request.getAmount().isNegativeOrZero())
            throw new IllegalArgumentException("Amount must be positive");
    }

    // 3. STATIC METHOD: no 'this', belongs to class, not instance
    public static boolean isSupportedCurrency(String currencyCode) {
        return Set.of("USD", "EUR", "GBP", "JPY", "INR").contains(currencyCode);
    }

    // 4. FINAL METHOD: cannot be overridden; enables JIT inlining
    public final String getServiceId() { return "payment-service-v2"; }

    // 5. SYNCHRONIZED METHOD: acquires 'this' lock before executing
    public synchronized void updateRateLimit(int requestsPerSecond) {
        this.rateLimitConfig.setRps(requestsPerSecond);
    }

    // 6. METHOD OVERLOADING: same name, different signatures
    public PaymentResult processPayment(String cardToken, Money amount) {
        return processPayment(new PaymentRequest.Builder()
            .cardToken(cardToken).amount(amount).build());
    }

    public PaymentResult processPayment(String cardToken, Money amount, String idempotencyKey) {
        return processPayment(new PaymentRequest.Builder()
            .cardToken(cardToken).amount(amount).idempotencyKey(idempotencyKey).build());
    }

    // 7. VARARGS METHOD: variable number of arguments
    public Money calculateTotal(Money... amounts) {
        return Arrays.stream(amounts)
            .reduce(Money.ZERO_USD, Money::add);
    }

    private PaymentResult executeWithRetry(PaymentRequest request) {
        for (int attempt = 0; attempt < MAX_RETRIES; attempt++) {
            try {
                return repo.execute(request);
            } catch (TransientException e) {
                if (attempt == MAX_RETRIES - 1) throw e;
                backoff(attempt);
            }
        }
        throw new IllegalStateException("Unreachable");
    }

    private void backoff(int attempt) {
        try { Thread.sleep((long) Math.pow(2, attempt) * 100); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
```

### C++ — const Methods, Pass-by-Reference, noexcept
```cpp
#include <string>
#include <vector>
#include <optional>
using namespace std;

class OrderBook {
private:
    vector<Order> bids;
    vector<Order> asks;
    string symbol;

public:
    explicit OrderBook(string symbol) : symbol(std::move(symbol)) {}

    // 1. CONST METHOD: promises not to modify object state
    //    Can be called on const OrderBook references
    double getBestBid() const {
        return bids.empty() ? 0.0 : bids.front().getPrice();
    }

    // 2. PASS BY CONST REFERENCE: read-only, no copy overhead
    void addBid(const Order& order) {
        bids.push_back(order); // Copies order into bids
    }

    // 3. PASS BY RVALUE REFERENCE (Move Semantics): transfer ownership, no copy
    void addBid(Order&& order) {
        bids.push_back(std::move(order)); // Moves, no copy
    }

    // 4. noexcept: guarantees no exception — enables compiler optimizations
    size_t getBidCount() const noexcept { return bids.size(); }

    // 5. STATIC METHOD: no 'this', utility function
    static double calculateSpread(double bidPrice, double askPrice) {
        return askPrice - bidPrice;
    }

    // 6. OPERATOR OVERLOADING (special method type)
    bool operator==(const OrderBook& other) const {
        return symbol == other.symbol;
    }

    // 7. CONST OVERLOAD: two versions — one for mutable, one for const access
    Order& at(size_t index) { return bids.at(index); }
    const Order& at(size_t index) const { return bids.at(index); }
};
```

### Python — Methods, Properties, Class/Static Methods
```python
from typing import Optional
from functools import lru_cache
import time

class Portfolio:
    _total_portfolios = 0  # Class variable (static)

    def __init__(self, owner_id: str):
        self._owner_id = owner_id
        self._positions: dict = {}
        Portfolio._total_portfolios += 1

    # 1. INSTANCE METHOD: operates on self
    def add_position(self, symbol: str, quantity: int, price: float) -> None:
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got: {quantity}")
        if price <= 0:
            raise ValueError(f"Price must be positive, got: {price}")
        self._positions[symbol] = self._positions.get(symbol, 0) + quantity

    # 2. PROPERTY (getter): access like a field, computed dynamically
    @property
    def total_value(self) -> float:
        return sum(
            qty * self._get_current_price(sym)
            for sym, qty in self._positions.items()
        )

    # 3. PROPERTY SETTER: controlled mutation
    @property
    def owner_id(self) -> str:
        return self._owner_id

    @owner_id.setter
    def owner_id(self, value: str) -> None:
        if not value or not value.strip():
            raise ValueError("Owner ID cannot be empty")
        self._owner_id = value

    # 4. CLASS METHOD: operates on class, not instance
    @classmethod
    def get_total_count(cls) -> int:
        return cls._total_portfolios

    # 5. STATIC METHOD: utility, no self or cls
    @staticmethod
    def is_valid_symbol(symbol: str) -> bool:
        return bool(symbol) and symbol.isupper() and len(symbol) <= 5

    # 6. CACHED METHOD (memoization): result cached for repeated calls
    @lru_cache(maxsize=128)
    def _get_current_price(self, symbol: str) -> float:
        # Expensive API call — cached per symbol
        return market_api.get_price(symbol)

    # 7. DUNDER/MAGIC METHOD: special behavior
    def __len__(self) -> int: return len(self._positions)
    def __repr__(self) -> str: return f"Portfolio(owner={self._owner_id}, positions={len(self._positions)})"
    def __contains__(self, symbol: str) -> bool: return symbol in self._positions
```

---

## 5. Visual Diagrams

### Instance Method — `this` Pointer Path
```
account1 = new BankAccount("Alice", 1000)  [Heap addr: 0xA100]
account2 = new BankAccount("Bob", 2000)    [Heap addr: 0xB200]

account1.deposit(500);
  Stack frame:
  +-----------+
  | this=0xA100 |  ← points to Alice's object
  | amount=500  |
  +-----------+
  this.balance += 500  → writes to 0xA100+offset(balance) = 1500

account2.deposit(300);
  Stack frame:
  +-----------+
  | this=0xB200 |  ← points to Bob's object
  | amount=300  |
  +-----------+
  this.balance += 300  → writes to 0xB200+offset(balance) = 2300

NOTE: Only ONE copy of deposit() code in Metaspace.
      'this' pointer distinguishes which object to operate on.
```

### Static Dispatch vs Dynamic Dispatch
```
STATIC DISPATCH (Overloading — Compile Time):

Programmer writes: logger.log("message")
Compiler sees:
  log(String)     → matches
  log(Exception)  → doesn't match
Decision made at COMPILE TIME. Binary has direct jump address.

DYNAMIC DISPATCH (Overriding — Runtime):
                                    HEAP
Animal a = new Dog();  a → [Dog object]
a.makeSound();                         → follows vptr
                              ↓
                    Dog vtable: makeSound → Dog::makeSound()
                    → executes "Woof"

If line 1 was: Animal a = new Cat();
                    Cat vtable: makeSound → Cat::makeSound()
                    → executes "Meow"

Same source code. Same bytecode. Different execution depending on object type.
```

### Method vs Field Storage in Memory
```
COMMON MISCONCEPTION: Each object has its own copy of methods.
REALITY: Only one copy of method code in Metaspace.

Metaspace (permanent, shared):
  BankAccount.deposit()   bytecode [0x4000]
  BankAccount.withdraw()  bytecode [0x4100]
  BankAccount.getBalance() bytecode [0x4200]

Heap (per object):
  account1: [header] [balance=1000] [owner="Alice"] [vptr → BankAccount vtable]
  account2: [header] [balance=2000] [owner="Bob"]   [vptr → BankAccount vtable]
  account3: [header] [balance=0]    [owner="Carol"]  [vptr → BankAccount vtable]

  All three objects SHARE the same method code.
  'this' pointer tells the method which object to work on.
```

---

## 6. Real World Analogy

### A Vending Machine
- The vending machine is the **object**
- Current inventory, money collected = **state (fields)**
- Buttons: `insertCoin()`, `selectItem()`, `cancelPurchase()` = **methods (public API)**
- Internal dispensing logic, change calculation = **private methods (hidden implementation)**
- The buttons validate input: `insertCoin(0)` → "Invalid coin" (method validates before acting)
- State is protected: you can't directly reach in and grab items (no direct field access)

### Employee and their Role
- Employee = **object**
- Salary, title, department = **fields**
- `promote()`, `assignProject()`, `requestLeave()` = **methods**
- HR validates all transitions via methods — direct salary changes require a `raiseSalary()` call with manager approval
- The same HR process (method) applies to ALL employees, regardless of which specific employee is calling

### Remote Control
- TV = **object**
- Current channel, volume, power state = **state**
- `volumeUp()`, `changeChannel(int)`, `powerToggle()` = **methods**
- Overloaded: `changeChannel(int number)` and `changeChannel(String name)` — different ways to call the same concept
- Private: internal signal encoding methods — caller doesn't see them

---

## 7. Interview Explanation

### 30 Seconds
> "A method is a named block of code bound to a class or instance. Instance methods implicitly receive a `this` pointer to the invoking object, giving access to its state. Methods define the behavioral contract — the API — of a class. They separate the 'what' (public interface) from the 'how' (implementation)."

### 1 Minute
> "Methods come in two resolution flavors. Overloading is resolved at compile-time — the compiler picks which method based on argument types declared in the call. Overriding is resolved at runtime — the JVM looks up the vtable of the actual object type, not the declared type. This is dynamic dispatch and is what makes polymorphism work.
>
> At the machine level, methods don't exist inside objects. There's ONE copy of each method in the code segment (Metaspace in Java). The `this` pointer, passed as a hidden first argument, is what connects a method call to the specific object's data."

### 3 Minutes
> "Let me go from method design down to machine-level execution.
>
> First: method design principles. A method should do ONE thing (SRP). It should be named with a verb that describes what it does ('processPayment', not 'doStuff'). Parameters should be minimal — more than 4 is a signal to group them into a parameter object. Return types should communicate intent: return `Optional<T>` instead of null, return empty collections instead of null.
>
> Second: dispatch mechanisms. At the JVM bytecode level, there are four dispatch instructions: `invokestatic` for static methods (fastest — direct address), `invokespecial` for constructors and private methods, `invokevirtual` for overridable instance methods (vtable lookup), and `invokeinterface` for interface calls (itable search — slightly slower than invokevirtual). `invokedynamic` is used for lambdas and method references.
>
> Third: JIT optimization. The JVM's JIT compiler monitors hot methods. Once a method is called enough times, it's compiled to native machine code. The key optimization is inlining: if `getBalance()` is a one-liner and is called millions of times, the JIT replaces every call site with the method body directly — zero method call overhead. For virtual calls, if only one type appears at a call site (monomorphic), the JIT devirtualizes and inlines. If multiple types appear (megamorphic: 3+), the JIT falls back to vtable dispatch.
>
> This is why premature micro-optimization based on abstraction overhead is usually wrong — the JIT closes the gap between abstraction cost and direct access for hot paths."

---

## 8. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What is a method signature? | Method name + parameter types (in order). Return type is NOT part of the signature. |
| What is the difference between a function and a method? | Method is bound to a class/object; has implicit `this` access. Function is standalone. |
| Can you overload a method by changing only the return type? | No — compile error. Compiler can't determine which to call at call site. |
| What is a static method? | Belongs to the class, not instances. No `this` pointer. Called via class name. |
| What is a private method? | Accessible only within the class. Resolved at compile time (invokespecial). Cannot be overridden. |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| What is method overloading? | Same name, different parameter signatures. Resolved at compile time (static dispatch). |
| What is method overriding? | Same name + same signature in subclass. Resolved at runtime (dynamic dispatch via vtable). |
| What is `this` in Java? | Reference to the current object instance. Hidden first parameter in every instance method. |
| What does `final` on a method mean? | Cannot be overridden by subclasses. JIT can inline it more aggressively. |
| Java is pass-by-value — explain for objects | Passes the VALUE of the object reference. Can mutate the object, but can't reassign the caller's variable. |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| `invokevirtual` vs `invokeinterface`? | `invokevirtual`: fixed vtable offset per class — fast. `invokeinterface`: itable search (variable offset) — slightly slower first time, cached. |
| What is JIT method inlining? | JIT replaces call with method body at hot call sites. Zero call overhead. Requires small method body + monomorphic call site. |
| What is a megamorphic call site and why is it a performance concern? | 3+ different types seen at a virtual call site. JIT gives up inlining — falls back to full vtable dispatch. |
| What is covariant return type in method overriding? | Overriding method can return a subtype of the base return type. `Animal create()` can be overridden as `Dog create()`. |
| Explain `synchronized` on a static method vs instance method | Instance: locks `this` object. Static: locks the `Class` object. Never conflict with each other. |

### 💼 Google Level
> *"You have a method `process(List<Event> events)` called 500K times/second. Each call does 3 virtual dispatch calls internally. Profiler shows 15% time in dispatch overhead. How do you analyze and optimize?"*

Expected: (1) async-profiler to confirm vtable dispatch hotspots. (2) Check if call sites are monomorphic (JIT should devirtualize automatically). (3) If megamorphic: refactor to concrete types in hot path using sealed classes/type-switching. (4) Batch processing: reduce call frequency. (5) C++ CRTP or template dispatch for zero overhead at compile time.

### 💼 Goldman Sachs Level
> *"A pricing method is called on 500 instrument types. We use an interface `PricingModel.price(MarketData)`. Profiler shows megamorphic dispatch on this call. How do you resolve the tradeoff between OOP design and performance?"*

Expected: Java sealed classes (Java 17) with pattern matching for exhaustive, JIT-inlinable dispatch. Switch on `PricingModel` subtypes — JIT can optimize closed type hierarchies better than open interface dispatch.

---

## 9. Coding Examples

### Pass-by-Value — The Tricky Object Case
```java
public class PassByValueDemo {

    // CASE 1: Primitive — original unchanged
    public static void doubleIt(int x) {
        x = x * 2;  // Local copy of x — original untouched
    }

    // CASE 2: Mutating the object works — same object in heap
    public static void depositInto(BankAccount account, double amount) {
        account.deposit(amount);  // Follows reference → mutates heap object
        // Original variable still points to same object. Mutation is visible.
    }

    // CASE 3: Reassigning the reference — does NOT affect caller
    public static void replaceAccount(BankAccount account) {
        account = new BankAccount("Eve", 5000); // Replaces LOCAL copy of reference
        // Caller's 'account' still points to original object!
    }

    public static void main(String[] args) {
        int num = 10;
        doubleIt(num);
        System.out.println(num);  // 10 — unchanged!

        BankAccount acc = new BankAccount("Alice", 1000.0);
        depositInto(acc, 500.0);
        System.out.println(acc.getBalance());  // 1500.0 — mutated!

        replaceAccount(acc);
        System.out.println(acc.getOwner());    // "Alice" — reference unchanged!
    }
}
```

### Pure vs Impure Methods
```java
public class PriceCalculator {

    // IMPURE METHOD: mutates state, has side effects, hard to test
    private double taxRate;
    private List<Double> calculationHistory;

    public double computePriceImpure(double basePrice) {
        double result = basePrice * (1 + taxRate); // Depends on mutable state!
        calculationHistory.add(result);             // Side effect!
        System.out.println("Computed: " + result); // Another side effect!
        return result;
    }
    // Testing: requires setting up taxRate, calculationHistory, capturing stdout

    // PURE METHOD: same inputs → same output, no side effects
    public static double computePriceWithTax(double basePrice, double taxRate) {
        if (basePrice < 0) throw new IllegalArgumentException("Price cannot be negative");
        if (taxRate < 0 || taxRate > 1) throw new IllegalArgumentException("Tax rate 0-1");
        return basePrice * (1 + taxRate);
    }
    // Testing: computePriceWithTax(100.0, 0.1) → always 110.0. No setup needed.
    // Thread-safe: no shared state. Can run in parallel freely.
}
```

### Method Chaining (Fluent Interface)
```java
// Builder / Fluent API: each method returns 'this' for chaining
public class QueryBuilder {
    private String tableName;
    private final List<String> conditions = new ArrayList<>();
    private final List<String> columns = new ArrayList<>();
    private Integer limit;
    private String orderByColumn;
    private boolean ascending = true;

    public QueryBuilder from(String table) {
        Objects.requireNonNull(table, "Table name required");
        this.tableName = table;
        return this;  // Return this for chaining
    }

    public QueryBuilder select(String... cols) {
        Collections.addAll(this.columns, cols);
        return this;
    }

    public QueryBuilder where(String condition) {
        Objects.requireNonNull(condition, "Condition required");
        this.conditions.add(condition);
        return this;
    }

    public QueryBuilder limit(int n) {
        if (n <= 0) throw new IllegalArgumentException("Limit must be positive");
        this.limit = n;
        return this;
    }

    public QueryBuilder orderBy(String column, boolean ascending) {
        this.orderByColumn = column;
        this.ascending = ascending;
        return this;
    }

    public String build() {
        if (tableName == null) throw new IllegalStateException("Table name required");
        StringBuilder sb = new StringBuilder("SELECT ");
        sb.append(columns.isEmpty() ? "*" : String.join(", ", columns));
        sb.append(" FROM ").append(tableName);
        if (!conditions.isEmpty()) sb.append(" WHERE ").append(String.join(" AND ", conditions));
        if (orderByColumn != null) sb.append(" ORDER BY ").append(orderByColumn)
                                     .append(ascending ? " ASC" : " DESC");
        if (limit != null) sb.append(" LIMIT ").append(limit);
        return sb.toString();
    }
}

// Readable, expressive usage:
String query = new QueryBuilder()
    .from("orders")
    .select("id", "customer_id", "total", "created_at")
    .where("status = 'PENDING'")
    .where("total > 100.00")
    .orderBy("created_at", false)  // Descending
    .limit(50)
    .build();
// → "SELECT id, customer_id, total, created_at FROM orders
//    WHERE status = 'PENDING' AND total > 100.00 ORDER BY created_at DESC LIMIT 50"
```

### Advanced — Method Design for High-Throughput System
```java
// Design: event processor method that must handle 1M events/second
// Key: minimize allocations, avoid virtual dispatch on hot path

public class EventProcessor {
    private static final int BATCH_SIZE = 1000;

    // METHOD DESIGN: batch processing to amortize overhead
    public void processBatch(EventRecord[] events, int count) {
        // Arrays.copyOf allocation avoided — passed in with count
        for (int i = 0; i < count; i++) {
            processEvent(events[i]); // Private (invokespecial — no vtable!)
        }
    }

    // Private + final: enables JIT inlining, no vtable lookup
    private void processEvent(EventRecord event) {
        // Type checking via sealed class instead of instanceof chain (Java 17+)
        switch (event.getType()) {
            case LOGIN -> handleLogin((LoginEvent) event);
            case PURCHASE -> handlePurchase((PurchaseEvent) event);
            case LOGOUT -> handleLogout((LogoutEvent) event);
        }
    }

    // @jdk.internal.vm.annotation.ForceInline equivalent behavior:
    // Keep these methods small so JIT inlines automatically
    private void handleLogin(LoginEvent e) {
        loginMetrics.increment();
        userSessions.activate(e.getUserId(), e.getSessionId());
    }

    private void handlePurchase(PurchaseEvent e) {
        purchaseMetrics.recordAmount(e.getAmount());
        inventory.decrementStock(e.getProductId(), e.getQuantity());
    }

    private void handleLogout(LogoutEvent e) {
        logoutMetrics.increment();
        userSessions.deactivate(e.getSessionId());
    }
}
```

---

## 10. Common Mistakes

### ⚠️ Mistake 1: God Method
```java
// WRONG: Method does too many things
public void processUserRegistration(String email, String password, String name) {
    // 1. Validate email
    if (!email.contains("@")) throw new RuntimeException("Bad email");

    // 2. Hash password
    String hash = BCrypt.hashpw(password, BCrypt.gensalt());

    // 3. Save to DB
    dbConnection.execute("INSERT INTO users VALUES(?, ?, ?)", email, hash, name);

    // 4. Send welcome email
    emailClient.send(email, "Welcome " + name, loadTemplate("welcome.html"));

    // 5. Add to marketing list
    mailchimpApi.addContact(email, name);

    // 6. Log analytics
    analyticsService.track("USER_REGISTERED", Map.of("email", email));

    // 6 things! Untestable, unreadable, inflexible.
}

// FIX: Each method does one thing; orchestrator composes them
public void registerUser(RegistrationRequest request) {
    validate(request);                                    // Single responsibility
    User user = createUser(request);                      // Single responsibility
    userRepository.save(user);                            // Single responsibility
    notificationService.sendWelcome(user);                // Single responsibility
    marketingService.enroll(user);                        // Single responsibility
    analyticsService.track(UserRegistered.of(user));      // Single responsibility
}
```

### ⚠️ Mistake 2: Returning Null
```java
// WRONG: Returning null for optional results
public User findUserByEmail(String email) {
    // Returns null if not found
    return db.queryForObject("SELECT * FROM users WHERE email = ?", email);
}

// Every caller must null-check — NullPointerException waiting to happen:
User user = userService.findUserByEmail("alice@example.com");
user.getName(); // NullPointerException if not found!

// FIX: Return Optional<T>
public Optional<User> findUserByEmail(String email) {
    return Optional.ofNullable(db.queryForObject("SELECT * FROM users WHERE email = ?", email));
}

// Caller forced to handle absence explicitly:
userService.findUserByEmail("alice@example.com")
    .map(User::getName)
    .orElse("Unknown User");
```

### ⚠️ Mistake 3: Overloading with Ambiguous Types
```java
// WRONG: Overloads with confusable types
public void remove(int index) { elements.remove(index); }         // Removes by index
public void remove(Integer value) { elements.remove(value); }     // Removes by value (boxed)

List<Integer> list = new ArrayList<>(List.of(1, 2, 3));
list.remove(1);           // Calls remove(int) — removes INDEX 1 → [1, 3]
list.remove(Integer.valueOf(1)); // Calls remove(Integer) — removes VALUE 1 → [2, 3]
// Subtle bug: autoboxing determines which overload!

// FIX: Use descriptive separate names
public void removeAtIndex(int index) { elements.remove(index); }
public void removeValue(Integer value) { elements.remove(value); }
```

### ⚠️ Mistake 4: Method Too Long / Too Nested
```java
// WRONG: Deep nesting, long method, multiple abstractions
public PaymentResult charge(Order order) {
    if (order != null) {
        if (order.getStatus() == OrderStatus.PENDING) {
            if (order.getAmount().isPositive()) {
                try {
                    String cardToken = order.getPaymentMethod().getCardToken();
                    if (cardToken != null && !cardToken.isEmpty()) {
                        // 50 more lines...
                    }
                } catch (Exception e) {
                    // handle...
                }
            }
        }
    }
    return null;
}

// FIX: Early returns, extract helpers, single level of abstraction
public PaymentResult charge(Order order) {
    validateChargeRequest(order);          // Throws if invalid
    String cardToken = extractCardToken(order);
    return executeCharge(cardToken, order.getAmount());
}

private void validateChargeRequest(Order order) {
    Objects.requireNonNull(order, "Order cannot be null");
    if (order.getStatus() != OrderStatus.PENDING)
        throw new IllegalStateException("Only PENDING orders can be charged");
    if (!order.getAmount().isPositive())
        throw new IllegalArgumentException("Order amount must be positive");
}
```

### ⚠️ Mistake 5: `synchronized` on Entire Method (Over-Locking)
```java
// WRONG: Entire method synchronized — blocks all callers even when not needed
public synchronized PaymentResult processPayment(PaymentRequest request) {
    validateRequest(request);         // No shared state — doesn't need lock!
    String token = tokenize(request); // No shared state — doesn't need lock!
    synchronized(rateTracker) {       // Only THIS part needs synchronization
        rateTracker.record(request);
    }
    return gateway.charge(request);   // No shared state — doesn't need lock!
}

// FIX: Narrow the critical section
public PaymentResult processPayment(PaymentRequest request) {
    validateRequest(request);         // Unsynchronized — safe
    String token = tokenize(request); // Unsynchronized — safe
    synchronized(rateTracker) {
        rateTracker.record(request);  // Only synchronize what needs it
    }
    return gateway.charge(request);   // Unsynchronized — safe
}
```

---

## 11. Best Practices

### Design Rules
- **One method = one thing**: If the name needs "And" or "Or", split it
- **Verb naming**: Methods are actions — `calculateTax()`, `validateRequest()`, `fetchUser()`
- **Return meaningful types**: `Optional<T>` not null, `List<T>` not null (empty list)
- **Small methods**: 5-15 lines is ideal. JIT inlines small methods → zero overhead.
- **Max 3-4 parameters**: Group more into a parameter object
- **Fail fast**: Validate at method start; throw early with descriptive exception messages
- **Tell, Don't Ask**: Don't get data, compute externally, then set. Let object do it.
- **Command-Query Separation**: Methods either query (return data) OR command (mutate state), not both

### Performance Rules
- **Private methods**: No vtable lookup (`invokespecial`). JIT inlines freely.
- **`final` methods**: JIT can devirtualize and inline. No override possible.
- **Static methods**: Fastest dispatch (`invokestatic`). No `this`, no vtable.
- **Avoid `synchronized` on full method**: Narrow to only the critical section.

---

## 12. Complexity

| Method Type | Dispatch Cost | After JIT | Notes |
|------------|--------------|-----------|-------|
| `static` | ~2ns | ~0.5ns | `invokestatic` — direct address |
| `private` / `final` | ~2-3ns | ~0-1ns | `invokespecial` — JIT inlines readily |
| `virtual` (monomorphic) | ~5ns | ~1ns | JIT devirtualizes + inlines |
| `virtual` (bimorphic) | ~5-8ns | ~2-3ns | JIT generates type guard |
| `interface` (megamorphic) | ~10-15ns | ~10-15ns | JIT can't optimize megamorphic |
| Synchronized method entry | ~20-50ns | — | Lock acquisition overhead |
| Stack frame allocation | ~2-5ns | 0 (inlined) | Managed by CPU stack pointer |

---

## 13. Advantages

| Advantage | Benefit |
|-----------|---------|
| **Encapsulation** | Hides implementation; callers interact through API only |
| **Code reuse** | Write once, call from many places |
| **Testability** | Each method unit-testable independently |
| **Polymorphism** | Overriding enables runtime behavior variation |
| **SRP enforcement** | Small focused methods = easier to maintain |
| **API contract** | Public methods define stable interface for callers |

---

## 14. Disadvantages

| Disadvantage | Context |
|-------------|---------|
| **Call overhead** | ~5-15ns per virtual call (mitigated by JIT inlining) |
| **Stack depth** | Deep call stacks risk StackOverflowError (especially with recursion) |
| **Over-engineering** | Too many tiny methods = harder to trace execution path |
| **Synchronized overhead** | Locking entire methods = bottleneck in high concurrency |
| **Verbosity** | Java's method signatures are verbose vs Python/JavaScript |

---

## 15. Comparison Table

### Overloading vs Overriding

| Feature | Overloading | Overriding |
|---------|-------------|------------|
| **Location** | Same class (or subclass) | Subclass only |
| **Name** | Same | Same |
| **Parameters** | MUST differ | MUST be identical |
| **Return type** | Can differ | Same (or covariant) |
| **Resolution** | Compile time | Runtime (vtable) |
| **Polymorphism type** | Ad-hoc / Compile-time | Subtype / Runtime |
| **Access modifier** | Can differ | Cannot be more restrictive |
| **Exceptions** | Can differ freely | Cannot add new checked exceptions |
| **`@Override`** | Not applicable | Use to verify correctness |

### Instance vs Static Method

| Feature | Instance Method | Static Method |
|---------|----------------|---------------|
| **`this` access** | Yes | No |
| **Dispatch** | Dynamic (vtable) | Static (compile-time) |
| **Can be overridden** | Yes | No (only hidden) |
| **Access to instance fields** | Yes | No |
| **When to use** | Object behavior | Utility, factory |
| **Thread safety** | Depends on state | Depends on static state |

---

## 16. Design Pattern Connection

| Pattern | Method's Role |
|---------|--------------|
| **Template Method** | Abstract class defines algorithm skeleton in one method; abstract methods are extension points |
| **Strategy** | `execute()` or `process()` method on interface — each Strategy implements it differently |
| **Command** | `execute()` method encapsulates an action as an object |
| **Observer** | `update()` / `notify()` method is the notification contract |
| **Decorator** | Overrides methods to add behavior before/after delegating to wrapped object |
| **Proxy** | Same method as real subject; adds cross-cutting concerns (logging, auth, caching) |
| **Builder** | `build()` final method; each property-setter method returns `this` for chaining |
| **Visitor** | `accept(Visitor v)` on elements; `visit(ElementType e)` on visitor |
| **Chain of Responsibility** | `handle(Request r)` method — each handler decides to process or forward |
| **State** | `handle()` method delegated to current State object; state-specific behavior |

---

## 17. System Design Connection

### RPC and Service Methods
```
Local method call:
    orderService.placeOrder(order);  → Function call, stack frame, ~10ns

gRPC method call (same signature, remote execution):
    orderServiceStub.placeOrder(order);  → Serialize → TCP → Remote JVM → Execute → Deserialize
    → Same method interface, completely different execution path
    → ~10ms latency

The abstraction: PlaceOrder(PlaceOrderRequest) → PlaceOrderResponse
  is the same interface locally and remotely.
  This is how microservice clients look like local method calls.
```

### Idempotent Methods for Distributed Systems
```java
// Problem: network retry → method called twice → duplicate charge!
// Solution: idempotent design

public PaymentResult charge(ChargeRequest request) {
    // Check if this idempotency key was already processed
    Optional<PaymentResult> cached = idempotencyStore.get(request.getIdempotencyKey());
    if (cached.isPresent()) {
        return cached.get(); // Return same result — no duplicate charge!
    }

    PaymentResult result = executeCharge(request);
    idempotencyStore.put(request.getIdempotencyKey(), result, Duration.ofDays(1));
    return result;
}
// POST /payments (idempotency key: K1) × 3 retries → only ONE charge
```

### Caching at Method Level (Read-Through)
```java
// Method-level caching: transparent to callers
@Service
public class ProductCatalogService {
    private final ProductRepository repo;
    private final Cache<String, Product> cache;

    public Product getProduct(String sku) {
        return cache.get(sku)
            .orElseGet(() -> {
                Product product = repo.findBySku(sku)
                    .orElseThrow(() -> new NotFoundException("Product not found: " + sku));
                cache.put(sku, product, Duration.ofMinutes(15));
                return product;
            });
    }
    // getProduct() looks like a simple lookup to callers.
    // Caching, expiry, cache-miss handling: all hidden.
}
```

---

## 18. Multithreading Connection

### Synchronized Methods
```java
class SafeCounter {
    private int count = 0;

    // Synchronized instance method: acquires lock on 'this'
    public synchronized void increment() { count++; }

    // Synchronized static method: acquires lock on SafeCounter.class
    public static synchronized void staticIncrement() { staticCount++; }
    // NOTE: Instance lock and class lock are different! No conflict.

    // BETTER: Atomic operations (no lock, CAS-based)
    private AtomicInteger atomicCount = new AtomicInteger(0);
    public void atomicIncrement() { atomicCount.incrementAndGet(); } // Lock-free!
}
```

### Narrowing the Critical Section
```java
// WRONG: Lock held during network call (long time!)
public synchronized List<Product> getAndCacheProducts(List<String> skus) {
    List<Product> products = remoteApi.fetch(skus);  // ~100ms! Lock held entire time!
    cache.put(skus, products);
    return products;
}

// RIGHT: Lock only the shared state mutation
public List<Product> getAndCacheProducts(List<String> skus) {
    List<Product> products = remoteApi.fetch(skus);  // Unsynchronized — no shared state
    synchronized (cache) {                            // Lock only for cache update (~1μs)
        cache.put(skus, products);
    }
    return products;
}
```

### Visibility with Volatile and Methods
```java
class StatusChecker {
    private volatile boolean running = true; // volatile: ensures visibility across threads

    // Reading thread sees 'running' updated by writing thread immediately
    public void processUntilStopped() {
        while (running) {    // Reads fresh value from main memory (volatile)
            process();
        }
    }

    // Writing thread can call this method from any thread
    public void stop() {
        running = false;     // Writes to main memory (volatile)
    }
}
```

---

## 19. Company Interview Perspective

### Google
- JIT method inlining thresholds and how to influence them
- `invokedynamic` for lambda implementation — how JVM generates and caches lambda classes
- Profiling megamorphic call sites with async-profiler

### Goldman Sachs
- Command-Query Separation (CQS) in trading systems: methods that query risk vs methods that execute trades
- Idempotent methods for order processing: `submitOrder(order, idempotencyKey)`
- Latency-sensitive method design: avoid allocations, synchronized blocks, logging in hot path

### Amazon
- REST API design: HTTP verbs map to method types (GET→query, POST→command)
- DynamoDB SDK: method design for atomic conditional writes (`putItem(request, condition)`)
- Lambda: cold-start impact of heavy static initializers and constructor work

### Microsoft
- C# extension methods: add methods to existing types without inheritance
- Expression-bodied methods (C#): syntactic sugar for one-line methods
- C++ `const` methods and their importance for const-correctness

### Meta/Facebook
- Python `@property` vs C++ getters: tradeoffs
- Method dispatch in Python: `__getattr__`, `__getattribute__` customization
- React class component methods vs functional component hooks (React transition rationale)

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | ⚠️ Can you override a `static` method? | No — static methods are Class-level, not instance-level. You can HIDE a static method in a subclass, but it's not polymorphic — resolved by declared type, not actual type. |
| 2 | What is a covariant return type? | Overriding method can return a subtype of the base return type. e.g., base: `Animal create()`, override: `Dog create()` — valid in Java. |
| 3 | ⚠️ Java is pass-by-value — but I can modify objects passed in. Contradiction? | No — the VALUE of the reference (memory address) is passed. You can follow that address and mutate the heap object. You can't reassign the caller's variable. |
| 4 | What is CQS (Command-Query Separation)? | Methods should either return data (query) OR cause side effects (command), not both. Prevents unexpected state changes when you thought you were reading. |
| 5 | ⚠️ Can an overriding method throw a new checked exception? | No — it can throw fewer or narrower checked exceptions, but not new ones that don't appear in the base method's throws clause. |
| 6 | What is an `invokedynamic` instruction used for? | Lambdas, method references, `switch` on strings in Java. Creates a dynamic call site whose target can be customized — enables flexibility for lambda capture. |
| 7 | ⚠️ Is `synchronized` on a method the same as `synchronized(this)` inside it? | Yes — a synchronized instance method is equivalent to a block `synchronized(this)` wrapping the entire body. |
| 8 | What does `@Override` annotation do at runtime? | Nothing at runtime — it's a compile-time check. Compiler verifies the method actually overrides a supertype method. Typos caught at compile time. |
| 9 | ⚠️ What is method hiding vs method overriding? | Override: dynamic dispatch — subclass method called at runtime. Hiding: static method with same name in subclass — called based on declared type (compile time). |
| 10 | What is the Call Stack Overflow and how do unbounded recursive methods cause it? | Each method call pushes a stack frame. Unlimited recursion = unlimited frames = StackOverflowError. JVM default stack is ~512KB. Tail-call optimization (Scala) eliminates this in some cases. |
| 11 | ⚠️ Can a method be both `static` and `abstract`? | No — abstract means "no implementation, must be overridden". Static methods cannot be overridden. Contradiction. |
| 12 | What is method reference syntax and how does it differ from a lambda? | `String::length` = method reference. `s -> s.length()` = lambda. Method references are more concise; lambdas can include additional logic. Both compile to `invokedynamic`. |
| 13 | ⚠️ What is the "Effective Java" rule about method overloading? | "Never export two overloadings with the same number of parameters" (Bloch, Item 52) — confusing autoboxing and varargs make it hard to predict which overload is called. |
| 14 | What is varargs (`...`) and what are its limitations? | `void log(String... messages)` — last parameter can take 0 or more args (passed as array). Limitation: only one varargs per method, must be last parameter. |
| 15 | What is a default method in Java interface and why was it added? | Java 8: interfaces can have concrete methods with `default` keyword. Added to allow backward-compatible API evolution (add new methods without breaking all implementors). |

---

## 21. Coding Problems

### Easy — Design a Method with Proper Error Handling
```java
// Problem: Design a method to safely transfer money between accounts
public TransferResult transfer(BankAccount from, BankAccount to, Money amount) {
    // Input validation (early return pattern)
    Objects.requireNonNull(from, "Source account required");
    Objects.requireNonNull(to, "Destination account required");
    Objects.requireNonNull(amount, "Transfer amount required");

    if (from.equals(to))
        return TransferResult.failed("Cannot transfer to same account");
    if (!amount.isPositive())
        return TransferResult.failed("Transfer amount must be positive");
    if (!from.hasSufficientFunds(amount))
        return TransferResult.failed("Insufficient funds: " + from.getBalance() + " available");

    // Execute transfer
    from.debit(amount);
    to.credit(amount);

    String txnId = UUID.randomUUID().toString();
    audit.record(txnId, from, to, amount);

    return TransferResult.success(txnId);
}
```

### Medium — Implement Retry Logic in a Method
```java
public <T> T executeWithRetry(
        Supplier<T> operation,
        int maxRetries,
        Duration initialDelay,
        Class<? extends Exception>... retryableExceptions) {

    Objects.requireNonNull(operation, "Operation required");
    if (maxRetries <= 0) throw new IllegalArgumentException("maxRetries must be > 0");

    Exception lastException = null;

    for (int attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            return operation.get();
        } catch (Exception e) {
            boolean isRetryable = Arrays.stream(retryableExceptions)
                .anyMatch(type -> type.isInstance(e));

            if (!isRetryable || attempt == maxRetries) {
                throw new RuntimeException("Operation failed after " + attempt + " attempts", e);
            }

            lastException = e;
            long delayMs = initialDelay.toMillis() * (long) Math.pow(2, attempt - 1);
            log.warn("Attempt {} failed, retrying in {}ms: {}", attempt, delayMs, e.getMessage());

            try { Thread.sleep(delayMs); }
            catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted during retry", ie);
            }
        }
    }
    throw new IllegalStateException("Unreachable", lastException);
}

// Usage:
T result = executeWithRetry(
    () -> externalApi.call(request),
    3,
    Duration.ofMillis(100),
    TransientException.class, NetworkTimeoutException.class
);
```

### Hard — Method Design for Concurrent Cache
```java
// Thread-safe cache with methods for concurrent access
public class ConcurrentLRUCache<K, V> {
    private final int maxSize;
    private final LinkedHashMap<K, V> cache;
    private final ReadWriteLock lock;

    public ConcurrentLRUCache(int maxSize) {
        if (maxSize <= 0) throw new IllegalArgumentException("maxSize must be positive");
        this.maxSize = maxSize;
        this.cache = new LinkedHashMap<>(maxSize, 0.75f, true) {
            @Override protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
                return size() > maxSize;
            }
        };
        this.lock = new ReentrantReadWriteLock();
    }

    // Query method: uses read lock for concurrent reads
    public Optional<V> get(K key) {
        Objects.requireNonNull(key);
        lock.readLock().lock();
        try {
            return Optional.ofNullable(cache.get(key));
        } finally {
            lock.readLock().unlock();
        }
    }

    // Command method: uses write lock for mutation
    public void put(K key, V value) {
        Objects.requireNonNull(key, "Key cannot be null");
        Objects.requireNonNull(value, "Value cannot be null");
        lock.writeLock().lock();
        try {
            cache.put(key, value);
        } finally {
            lock.writeLock().unlock();
        }
    }

    // Compute-if-absent: get or compute atomically (no double-computation)
    public V computeIfAbsent(K key, Function<K, V> computeFunc) {
        Objects.requireNonNull(key);
        Objects.requireNonNull(computeFunc);

        // First try with read lock (fast path)
        lock.readLock().lock();
        try {
            V existing = cache.get(key);
            if (existing != null) return existing;
        } finally {
            lock.readLock().unlock();
        }

        // Upgrade to write lock (slow path — compute + insert)
        lock.writeLock().lock();
        try {
            // Check again — another thread may have computed between read and write lock
            V existing = cache.get(key);
            if (existing != null) return existing;
            V computed = computeFunc.apply(key);
            cache.put(key, computed);
            return computed;
        } finally {
            lock.writeLock().unlock();
        }
    }

    // Query method
    public int size() {
        lock.readLock().lock();
        try { return cache.size(); }
        finally { lock.readLock().unlock(); }
    }

    // Command method
    public void invalidate(K key) {
        Objects.requireNonNull(key);
        lock.writeLock().lock();
        try { cache.remove(key); }
        finally { lock.writeLock().unlock(); }
    }
}
```

---

## 22. Revision Sheet

| Concept | Key Rule |
|---------|----------|
| Method | Named block of code bound to class/instance |
| Method signature | Name + parameter types (NOT return type) |
| `this` pointer | Hidden first arg in instance methods |
| Overloading | Same name, different params. Compile-time. |
| Overriding | Same name + params in subclass. Runtime. |
| Static dispatch | Resolved at compile time (static, private, final, overloaded) |
| Dynamic dispatch | Resolved at runtime via vtable (virtual/overridden methods) |
| `invokevirtual` | Fixed vtable offset — fast |
| `invokeinterface` | itable search — slightly slower, cached |
| JIT inlining | Small method body pasted at call site — zero overhead |
| Pure method | Same inputs → same output. No side effects. Thread-safe. |
| Command-Query | Method either returns data (query) OR mutates (command), not both |
| Pass-by-value | Java passes reference VALUE — can mutate object, can't reassign |
| Covariant return | Override can return subtype of base return type |
| Method hiding | Static method with same name in subclass — not polymorphic |

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| Method signature includes? | Name + parameter list. NOT return type. |
| `this` in instance method? | Hidden first argument — reference to the invoking object |
| Static dispatch used for? | Static, private, final methods; overloading |
| Dynamic dispatch used for? | Overridden methods (virtual methods) |
| `invokestatic` bytecode? | Static method call — fastest, no `this` |
| `invokespecial` bytecode? | Constructors, private methods, super calls |
| `invokevirtual` bytecode? | Instance method on class type — vtable lookup |
| `invokeinterface` bytecode? | Instance method on interface type — itable search |
| `invokedynamic` bytecode? | Lambdas, method references |
| JIT inlining? | Method body pasted at call site for zero overhead |
| Monomorphic call site? | One type seen — JIT devirtualizes and inlines |
| Megamorphic call site? | 3+ types seen — JIT gives up, falls back to vtable |
| Overloading vs Overriding? | Overloading = compile-time, same class. Overriding = runtime, subclass. |
| Java pass-by-value for objects? | Passes reference VALUE. Can mutate object; can't reassign caller's var. |
| Can override throw new checked exception? | No — can throw fewer/narrower, not new ones. |
| Synchronized instance method locks? | The `this` object |
| Synchronized static method locks? | The Class object |
| Covariant return type? | Override can return subtype of base return type |
| Method hiding? | Static method with same name in subclass — NOT polymorphic |
| Command-Query Separation? | Method returns data (query) OR mutates (command), not both |

---

## 24. Cheat Sheet

### Top 20 Facts
1. Method signature = name + parameter types (return type NOT included)
2. Instance methods implicitly receive `this` as a hidden first argument
3. Static methods: no `this`, no vtable, `invokestatic` — fastest dispatch
4. Private/final methods: `invokespecial` — JIT inlines aggressively
5. Virtual methods: `invokevirtual` — vtable lookup, ~5-10ns
6. Interface methods: `invokeinterface` — itable search, slightly slower
7. JIT inlining: replaces method call with body — ~0ns after JIT warmup
8. Overloading = compile-time (same class, different params)
9. Overriding = runtime (subclass, same signature, vtable dispatch)
10. Java is pass-by-value: objects pass reference value, not the object itself
11. Can mutate object through passed reference; can't reassign caller's variable
12. Covariant return: override can return subtype of base return type
13. Static method hiding ≠ overriding — resolved by declared type, not actual
14. `@Override` is compile-time check only; no runtime effect
15. Synchronized instance method = `synchronized(this)` wrapper
16. Synchronized static method = `synchronized(ClassName.class)`
17. Narrowing synchronized block beats synchronizing entire method
18. Pure methods: same inputs → same outputs, no side effects — thread-safe
19. Command-Query Separation: query returns data; command changes state
20. JIT devirtualization: monomorphic sites inlined; megamorphic (3+) not

---

## 25. Final Interview Summary

### Night-Before Revision
1. ⭐ Method = named code block bound to class; instance methods have `this`
2. ⭐ Signature = name + params. Return type NOT part of signature.
3. ⭐ Overloading: compile-time dispatch. Overriding: runtime vtable dispatch.
4. ⭐ `invokestatic` fastest; `invokevirtual` vtable; `invokeinterface` itable (slower first)
5. ⭐ JIT inlines small methods — zero call overhead on hot paths
6. ⭐ Java pass-by-value: can mutate objects, can't reassign caller's reference
7. ⭐ Pure methods = same inputs → same output, no side effects — thread-safe
8. ⭐ Command-Query Separation: return data OR mutate state, not both
9. ⭐ Synchronized method: instance = `this` lock; static = Class lock
10. ⭐ Narrow synchronization: lock only the critical section, not the entire method
