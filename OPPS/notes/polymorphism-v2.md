# Polymorphism — Industry-Level Interview Notes

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
Polymorphism means **"many forms"** — the same operation behaves differently depending on the object it acts upon.

### Technically
Polymorphism is the ability of a **single interface or method name** to represent **different underlying implementations**. It comes in two forms:
1. **Compile-time (Static) Polymorphism**: Resolved by compiler — method overloading, operator overloading, templates
2. **Runtime (Dynamic) Polymorphism**: Resolved at runtime via virtual dispatch — method overriding through inheritance/interfaces

### From an Interviewer's Perspective
> "I expect you to go beyond 'overloading vs overriding'. I want to hear about vtable mechanics, dynamic dispatch cost, how JIT eliminates it, type checking (instanceof), open-closed principle, and when polymorphism becomes a liability (performance, debugging complexity)."

⭐ **The power statement**: "Write code against an interface once — it works for all past and future implementations without change."

---

## 2. Why It Exists

### Problem It Solves
Without polymorphism, you must write separate code paths for every type — conditional explosion:

```java
// WITHOUT polymorphism — nightmare maintenance
void processPayment(Object payment) {
    if (payment instanceof CreditCard) {
        ((CreditCard) payment).chargeCreditCard();
    } else if (payment instanceof PayPal) {
        ((PayPal) payment).debitPayPal();
    } else if (payment instanceof UPI) {
        ((UPI) payment).executeUPI();
    }
    // Add Stripe? Add Apple Pay? Modify this method every time!
}

// WITH polymorphism — open-closed principle
void processPayment(Payment payment) {
    payment.pay(); // Works for CreditCard, PayPal, UPI, Stripe, ApplePay — forever
}
```

### Real Software Examples
- **Java Collections**: `Collections.sort()` works on `Comparable` — sorts strings, integers, custom objects
- **Java I/O**: `OutputStream.write()` works on `FileOutputStream`, `ByteArrayOutputStream`, `SocketOutputStream`
- **Spring**: `@Repository` interface — Spring injects MySQL/MongoDB/Redis implementation transparently
- **AWS SDK**: `S3Client.putObject()` — same call works across regions, mock clients in tests
- **Android**: `View.draw()` — Button, TextView, ImageView each draw differently via same interface

---

## 3. Internal Working

### Compile-Time Polymorphism (Overloading)

**Name Mangling**: The compiler generates unique internal names for each overloaded version:

```cpp
// Source:
void print(int x);
void print(double x);
void print(string x);

// Compiler internal names (mangled):
// _Z5printi   (print with int)
// _Z5printd   (print with double)
// _Z5printNSsE (print with string)
```

The call is resolved at **compile time** by matching argument types to mangled names. **Zero runtime cost.**

### Runtime Polymorphism — vtable Deep Dive

**Step 1: Compiler inserts vtable for each class with virtual methods**

```
Animal vtable (in read-only code segment):
Index  |  Function Pointer
  0    |  &Animal::speak
  1    |  &Animal::eat
  2    |  &Object::toString  (inherited)

Dog vtable:
Index  |  Function Pointer
  0    |  &Dog::speak    <- REPLACED (overridden)
  1    |  &Animal::eat   <- UNCHANGED (inherited)
  2    |  &Object::toString

Cat vtable:
Index  |  Function Pointer
  0    |  &Cat::speak    <- REPLACED
  1    |  &Animal::eat
  2    |  &Object::toString
```

**Step 2: Compiler inserts vptr in each object**

```
Animal object (heap):          Dog object (heap):
+------------------+           +------------------+
| vptr ------------|---------> Animal vtable      |
| name: "Animal"   |           | vptr ------------|---> Dog vtable
+------------------+           | name: "Rex"      |
                               | breed: "Lab"     |
                               +------------------+
```

**Step 3: Runtime dispatch**

```
call: animal->speak()

Assembly generated:
  mov rax, [animal]       ; Load object reference
  mov rax, [rax]          ; Load vptr (offset 0)
  mov rax, [rax + 0*8]    ; Load speak() function pointer (index 0)
  call rax                ; Jump to actual implementation
```

**Cost**: 2 extra memory reads vs direct call. Typically 3-5 ns on modern hardware. Can miss instruction cache.

### JVM Runtime Polymorphism

Java's JVM uses an **inline method table (itable/vtable)** similar to C++. Every class has an implicit vtable built by the ClassLoader.

**JIT Devirtualization** (critical for interviews):
```
// JIT observes that 90% of calls go to Dog::speak
// JIT generates optimized code:
if (animal.getClass() == Dog.class) {
    Dog.speak_inline();  // Direct call (zero overhead!)
} else {
    animal.speak();      // Fallback: vtable dispatch
}
// This is called "monomorphic/bimorphic inline cache"
```

### Python Dynamic Dispatch
Python uses a **dictionary-based dispatch** — slower but maximally flexible:

```
call: animal.speak()

1. Look up "speak" in animal.__dict__       (instance dict)
2. If not found: look in type(animal).__dict__ (class dict)
3. If not found: walk MRO chain
4. If not found: AttributeError
```

### Static vs Dynamic Binding Summary

```
STATIC BINDING (Compile-time):
Source Code -> Compiler -> Direct function address in binary
Cost: 0 (baked in at compile time)
Used for: private, static, final methods; overloaded methods

DYNAMIC BINDING (Runtime):
Source Code -> Compiler -> vtable slot index
Runtime: object.vptr -> vtable[index] -> function address
Cost: 2 pointer dereferences (~3-5ns)
Used for: virtual methods, interface methods
```

---

## 4. Syntax

### C++ — Compile-Time (Overloading)
```cpp
#include <iostream>
using namespace std;

class Calculator {
public:
    // Method overloading — compile-time polymorphism
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    string add(string a, string b) { return a + b; }  // Concatenation
    int add(int a, int b, int c) { return a + b + c;  } // Different arity
};

// Template — compile-time polymorphism without overloading
template<typename T>
T maximum(T a, T b) { return a > b ? a : b; }

// Operator overloading — compile-time polymorphism
class Vector2D {
public:
    double x, y;
    Vector2D operator+(const Vector2D& other) const {
        return {x + other.x, y + other.y};
    }
};
```

### C++ — Runtime (Overriding)
```cpp
class Payment {
public:
    virtual void pay(double amount) = 0;  // Pure virtual
    virtual string getName() const = 0;
    virtual ~Payment() {}
};

class CreditCard : public Payment {
public:
    void pay(double amount) override {
        cout << "Charging $" << amount << " to credit card" << endl;
    }
    string getName() const override { return "CreditCard"; }
};

class PayPal : public Payment {
public:
    void pay(double amount) override {
        cout << "Deducting $" << amount << " from PayPal" << endl;
    }
    string getName() const override { return "PayPal"; }
};

// Polymorphic usage
void processOrder(Payment& payment, double amount) {
    payment.pay(amount);  // Runtime dispatch — no if-else needed
}

int main() {
    CreditCard cc;
    PayPal pp;
    processOrder(cc, 100.0);  // "Charging $100 to credit card"
    processOrder(pp, 100.0);  // "Deducting $100 from PayPal"
}
```

### Java — Full Example
```java
// Runtime polymorphism
public interface Notification {
    void send(String message);
    default String getType() { return "Unknown"; } // default method (Java 8+)
}

public class EmailNotification implements Notification {
    private String email;

    public EmailNotification(String email) { this.email = email; }

    @Override
    public void send(String message) {
        System.out.println("Email to " + email + ": " + message);
    }

    @Override
    public String getType() { return "Email"; }
}

public class SMSNotification implements Notification {
    private String phone;

    public SMSNotification(String phone) { this.phone = phone; }

    @Override
    public void send(String message) {
        System.out.println("SMS to " + phone + ": " + message);
    }

    @Override
    public String getType() { return "SMS"; }
}

// Compile-time polymorphism — method overloading
public class NotificationService {
    private final List<Notification> channels = new ArrayList<>();

    public void addChannel(Notification n) { channels.add(n); }

    // Overloaded: different signatures
    public void broadcast(String message) {
        channels.forEach(n -> n.send(message));
    }

    public void broadcast(String message, String priority) {
        String tagged = "[" + priority + "] " + message;
        channels.forEach(n -> n.send(tagged));
    }
}
```

### Python — Full Example
```python
from abc import ABC, abstractmethod
from typing import List

# Runtime polymorphism
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

    def describe(self) -> str:
        return f"{type(self).__name__}: area={self.area():.2f}"


class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float: return 3.14159 * self.radius ** 2
    def perimeter(self) -> float: return 2 * 3.14159 * self.radius


class Rectangle(Shape):
    def __init__(self, w: float, h: float):
        self.width, self.height = w, h

    def area(self) -> float: return self.width * self.height
    def perimeter(self) -> float: return 2 * (self.width + self.height)


# Polymorphic usage — works for any Shape subclass
def total_area(shapes: List[Shape]) -> float:
    return sum(s.area() for s in shapes)


shapes = [Circle(5), Rectangle(4, 6), Circle(3)]
print(total_area(shapes))  # Works for any Shape


# Python "duck typing" — polymorphism without inheritance
class Duck:
    def speak(self): return "Quack!"

class Person:
    def speak(self): return "Hello!"

def make_sound(thing):  # No type requirement — duck typing
    return thing.speak()

print(make_sound(Duck()))    # "Quack!"
print(make_sound(Person()))  # "Hello!"
```

### Key Differences

| Aspect | C++ | Java | Python |
|--------|-----|------|--------|
| Compile-time | Overloading, templates, operator overloading | Overloading only | Not really (dynamic typing) |
| Runtime | `virtual` + override | All instance methods (auto) | All methods (auto) |
| Interface | Pure virtual abstract class | `interface` keyword | ABC / duck typing |
| Cost of vtable | Yes (explicit `virtual`) | Yes (always present) | Higher (dict lookup) |
| Duck typing | No (strict types) | No (strict types) | Yes |

---

## 5. Visual Diagrams

### Compile-Time vs Runtime Polymorphism
```
COMPILE-TIME POLYMORPHISM:

Source: add(5, 3)          Source: add(5.0, 3.0)
         |                           |
   Compiler resolves           Compiler resolves
         |                           |
 add_int(5, 3)               add_double(5.0, 3.0)
(baked into binary)          (baked into binary)

Zero runtime cost!

---

RUNTIME POLYMORPHISM:

Source: payment.pay(100)
         |
   Compiler: "look up pay() in vtable"
         |
   Runtime: payment.vptr -> vtable -> pay() address
         |                               |
     CreditCard.pay()           OR   PayPal.pay()
         (decided at runtime based on actual object)
```

### Method Resolution via vtable
```
+------------------+          vtable for Dog:
|  Dog Object      |          +---------------------+
|  vptr ---------->+--------->| [0] Dog::speak()    |
|  name: "Rex"     |          | [1] Animal::eat()   |
|  breed: "Lab"    |          | [2] Object::toString|
+------------------+          +---------------------+

Call: animal.speak()
  1. Load animal's vptr
  2. Jump to vtable
  3. Read slot [0] = Dog::speak()
  4. Call Dog::speak()
```

### Overloading Resolution Flow
```
Call: print(42)

Compiler checks:
  - print(int)?     YES -> bind to print(int)
  - print(double)?  Would work but int is better match
  - print(string)?  No implicit conversion

Resolution: print(int) -- DONE at compile time
```

### Polymorphic Collection Processing
```
List<Payment>:
  [0] CreditCard ──> vptr -> CreditCard vtable
  [1] PayPal     ──> vptr -> PayPal vtable
  [2] UPI        ──> vptr -> UPI vtable
  [3] Stripe     ──> vptr -> Stripe vtable

for payment in list:
  payment.pay(100)   <- Same call, different behavior each iteration
```

---

## 6. Real World Analogy

### Car (Accelerate = Polymorphic Operation)
- `car.accelerate()`:
  - Electric car: engages motor, draws battery
  - Gas car: opens throttle, injects fuel
  - Hybrid: decides based on battery level
- Driver uses same pedal (`accelerate()`) — behavior differs per car type

### Bank (Goldman Sachs)
- `instrument.calculatePnL()`:
  - Stock: `(currentPrice - avgCost) * quantity`
  - Bond: `(currentYield - purchaseYield) * notional`
  - Option: Black-Scholes model
- Risk engine calls same method — correct formula per instrument type

### Hospital
- `medicalStaff.treat(patient)`:
  - Doctor: prescribes medication
  - Surgeon: performs operation
  - Nurse: administers care
- Hospital system calls `treat()` — each staff member behaves appropriately

### E-Commerce (Amazon)
- `discount.apply(cart)`:
  - `PercentageDiscount`: 20% off
  - `FlatDiscount`: $10 off
  - `BuyOneGetOne`: free item
  - `LoyaltyDiscount`: points-based
- Cart system applies discount polymorphically — no if-else ladder

### Operating System
- `fileSystemNode.read()`:
  - `RegularFile`: reads from disk blocks
  - `SymbolicLink`: resolves target and reads
  - `/dev/random`: generates random bytes
  - `NetworkSocket`: reads from network buffer
- Same `read()` syscall — dramatically different behavior

### Game Development
- `enemy.attack(player)`:
  - Zombie: bites (melee)
  - Sniper: shoots from distance (ranged)
  - Wizard: casts spell (AoE)
  - Boss: multi-phase special attacks
- Game loop calls `attack()` on all enemies — each behaves correctly

---

## 7. Interview Explanation

### 30 Seconds
> "Polymorphism means one interface, many implementations. Compile-time polymorphism is method overloading — resolved by compiler based on argument types. Runtime polymorphism is method overriding via interfaces or inheritance — the correct implementation is selected at runtime through virtual dispatch, using vtables."

### 1 Minute
> "There are two forms. Compile-time: the compiler resolves overloaded method calls by matching argument signatures to function names — this has zero runtime cost. Runtime: the compiler doesn't know which method to call — it inserts vtable-based dispatch. Each polymorphic object holds a vptr that points to its class's vtable — a table of function pointers. At runtime, the call goes through vptr -> vtable -> actual method. This is how writing `payment.pay(100)` correctly calls `CreditCard.pay()` or `PayPal.pay()` depending on the object at runtime."

### 3 Minutes
> "Polymorphism is the mechanism that enables the open-closed principle: classes are open for extension, closed for modification. Without it, adding a new payment method means editing existing code — a violation that introduces bugs.

> At the machine level, virtual dispatch works like this: the compiler creates a vtable per class — a static read-only array in code memory where each slot is a function pointer to the most-derived override for that slot. Every object of a polymorphic class gets a hidden vptr field at byte offset 0, pointing to its class's vtable. When the compiler sees a virtual method call, instead of emitting a direct call instruction, it emits: load vptr, index into vtable, call through pointer. Two extra loads.

> The JVM's JIT compiler is clever about this. Through profile-guided optimization and Class Hierarchy Analysis, it identifies calls that always dispatch to the same concrete type and devirtualizes them — inlining the call directly. This eliminates the vtable overhead entirely for hot code paths.

> Python goes further into dynamic dispatch — it's dictionary lookup through the MRO chain. Far more flexible but significantly slower.

> For interviews at Goldman Sachs and Google, I'd also discuss duck typing vs structural typing: Python's duck typing means any object with `pay()` method works — no inheritance required. This is maximal flexibility but sacrifices static type safety."

---

## 8. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What are the two types of polymorphism? | Compile-time (overloading) and Runtime (overriding) |
| What is method overloading? | Same method name, different parameter types/count — resolved at compile time |
| What is method overriding? | Subclass provides its own implementation of inherited method — resolved at runtime |
| What is the `@Override` annotation? | Compiler check: ensures method actually overrides a parent method |
| Can you override a static method? | No — static methods are class-level, not virtual; they get hidden, not overridden |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| What is dynamic dispatch? | Runtime mechanism of selecting the correct method override via vtable |
| What is duck typing? | If an object has the required method, it can be used — no inheritance needed (Python) |
| Difference between overloading and overriding? | Overloading = compile-time, same class, different params; Overriding = runtime, different class, same signature |
| What is covariant return type? | Override can return subtype of parent's return type |
| Can private methods be overridden? | No — private methods are not visible to subclasses |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| How does vtable work internally? | Per-class table of function pointers; derived vtable replaces overridden slots; vptr in each object points to its vtable |
| What is vtable layout for multiple inheritance in C++? | Multiple vtables; object has multiple vptrs; thunks adjust `this` pointer |
| How does JIT devirtualize virtual calls? | Profile-guided optimization: if only one impl observed, inline direct call + guard |
| What is double dispatch? | Method resolution based on TWO object types — requires Visitor pattern or multiple dispatch |
| What is the Visitor pattern and why does it solve double dispatch? | Visitor separates algorithm from object; accept(visitor) + visitor.visit(this) achieves two-level dispatch |

### Expert
| Question | Expected Answer |
|----------|-----------------|
| What is CRTP and how does it achieve static polymorphism? | `Derived : public Base<Derived>` — base calls derived methods at compile time via template |
| Compare virtual dispatch cost vs direct call | 2 pointer loads extra; ~3-5ns; misses instruction cache; JIT can eliminate |
| What is a megamorphic call site? | JVM term: >2 implementations at same call site — JVM gives up on inline cache, falls back to vtable |
| Explain the inline method cache in HotSpot JVM | Monomorphic (1 type): direct call. Bimorphic (2 types): guarded inline. Megamorphic (3+): vtable |

### 💼 Google Level
> *"We have 10 million polymorphic method calls per second in our search serving layer. The vtable dispatch is adding 30ms latency per request. How do you fix this?"*
- CRTP for static polymorphism, JIT profiling and devirtualization, final classes to help JIT, data-oriented design (avoid virtual entirely), profile-guided optimization

### 💼 Goldman Sachs Level
> *"Design a polymorphic pricing engine for options, futures, swaps, and bonds. Each has a different pricing model. New instruments are added quarterly."*
- Abstract `PricingEngine` interface, concrete implementations, factory for instantiation, open-closed principle, no if-else in pricing loop

---

## 9. Coding Examples

### Basic Example
```java
abstract class Animal {
    abstract String speak();
}

class Dog extends Animal {
    @Override String speak() { return "Woof!"; }
}

class Cat extends Animal {
    @Override String speak() { return "Meow!"; }
}

// Polymorphic usage
List<Animal> animals = List.of(new Dog(), new Cat(), new Dog());
animals.forEach(a -> System.out.println(a.speak()));
// Woof! Meow! Woof!
```

### Intermediate — Strategy Pattern via Polymorphism
```java
// Strategy: interchangeable sorting algorithms
public interface SortStrategy {
    <T extends Comparable<T>> void sort(List<T> list);
}

public class QuickSort implements SortStrategy {
    @Override
    public <T extends Comparable<T>> void sort(List<T> list) {
        // QuickSort implementation
        Collections.sort(list); // simplified
    }
}

public class MergeSort implements SortStrategy {
    @Override
    public <T extends Comparable<T>> void sort(List<T> list) {
        // MergeSort implementation
        list.sort(Comparator.naturalOrder()); // simplified
    }
}

public class DataSorter {
    private SortStrategy strategy;

    public DataSorter(SortStrategy strategy) { this.strategy = strategy; }

    public void setStrategy(SortStrategy strategy) { this.strategy = strategy; }

    public <T extends Comparable<T>> void sort(List<T> data) {
        strategy.sort(data); // Polymorphic — delegates to strategy
    }
}

// Usage
DataSorter sorter = new DataSorter(new QuickSort());
sorter.sort(numbers);                     // Uses QuickSort
sorter.setStrategy(new MergeSort());
sorter.sort(numbers);                     // Now uses MergeSort — no code change!
```

### Advanced — Double Dispatch via Visitor Pattern
```java
// Problem: different operations (tax, discount) on different types (Food, Electronics)
interface Product { void accept(PriceVisitor visitor); }

class Food implements Product {
    double price;
    Food(double price) { this.price = price; }
    @Override public void accept(PriceVisitor visitor) { visitor.visit(this); }
}

class Electronics implements Product {
    double price;
    Electronics(double price) { this.price = price; }
    @Override public void accept(PriceVisitor visitor) { visitor.visit(this); }
}

interface PriceVisitor {
    void visit(Food food);
    void visit(Electronics electronics);
}

class TaxCalculator implements PriceVisitor {
    @Override public void visit(Food food) {
        System.out.printf("Food tax: $%.2f%n", food.price * 0.05);  // 5% GST
    }
    @Override public void visit(Electronics electronics) {
        System.out.printf("Electronics tax: $%.2f%n", electronics.price * 0.18); // 18% GST
    }
}

// Double dispatch: product.accept(visitor) => visitor.visit(specificProduct)
List<Product> cart = List.of(new Food(100), new Electronics(500), new Food(50));
PriceVisitor taxCalc = new TaxCalculator();
cart.forEach(p -> p.accept(taxCalc)); // Correct tax for each type!
```

### Production — Polymorphic Event Handler System
```java
// Production: extensible event processing pipeline
public interface EventHandler<T extends Event> {
    void handle(T event);
    Class<T> getEventType();
}

public class EventBus {
    private final Map<Class<?>, List<EventHandler<?>>> handlers = new ConcurrentHashMap<>();

    public <T extends Event> void register(EventHandler<T> handler) {
        handlers.computeIfAbsent(handler.getEventType(), k -> new CopyOnWriteArrayList<>())
                .add(handler);
    }

    @SuppressWarnings("unchecked")
    public <T extends Event> void publish(T event) {
        List<EventHandler<?>> eventHandlers = handlers.getOrDefault(event.getClass(), List.of());
        eventHandlers.forEach(h -> ((EventHandler<T>) h).handle(event));
    }
}

// Usage: new event types require zero changes to EventBus
public class OrderCreatedHandler implements EventHandler<OrderCreatedEvent> {
    @Override public void handle(OrderCreatedEvent event) { /* send confirmation email */ }
    @Override public Class<OrderCreatedEvent> getEventType() { return OrderCreatedEvent.class; }
}
```

### Competitive Coding — Polymorphism in LeetCode Design Problems
```java
// LeetCode #146-style: Design polymorphic cache
interface Cache<K, V> {
    V get(K key);
    void put(K key, V value);
    int size();
}

class LRUCache<K, V> implements Cache<K, V> {
    private final int capacity;
    private final LinkedHashMap<K, V> map;

    LRUCache(int capacity) {
        this.capacity = capacity;
        this.map = new LinkedHashMap<>(capacity, 0.75f, true) {
            protected boolean removeEldestEntry(Map.Entry<K,V> e) { return size() > capacity; }
        };
    }

    @Override public V get(K key) { return map.getOrDefault(key, null); }
    @Override public void put(K key, V value) { map.put(key, value); }
    @Override public int size() { return map.size(); }
}

class LFUCache<K, V> implements Cache<K, V> {
    // LFU implementation
    @Override public V get(K key) { /* ... */ return null; }
    @Override public void put(K key, V value) { /* ... */ }
    @Override public int size() { return 0; }
}

// Code using cache never changes — swap LRU for LFU transparently
Cache<String, User> userCache = new LRUCache<>(1000);
// Later: Cache<String, User> userCache = new LFUCache<>(1000);
```

---

## 10. Common Mistakes

### ⚠️ Mistake 1: Confusing Overloading and Overriding
```java
class Parent {
    void display(int x) { System.out.println("Parent: " + x); }
}

class Child extends Parent {
    // This is OVERLOADING, not OVERRIDING! Different parameter type
    void display(double x) { System.out.println("Child: " + x); }
    // To override: void display(int x) with @Override
}
```

### ⚠️ Mistake 2: Overloading Not Available at Runtime
```java
class Printer {
    void print(Object o) { System.out.println("Object: " + o); }
    void print(String s) { System.out.println("String: " + s); }
}

Object obj = "Hello"; // Static type is Object
printer.print(obj);   // Calls print(Object) -- NOT print(String)!
// Overloading resolved at COMPILE TIME based on STATIC type
```

### ⚠️ Mistake 3: instanceof Overuse (defeats polymorphism)
```java
// BAD: You wrote polymorphism but still using instanceof
void process(Animal a) {
    if (a instanceof Dog) ((Dog)a).bark();
    else if (a instanceof Cat) ((Cat)a).meow();
    // Every new Animal type requires modifying this method!
}

// GOOD: True polymorphism
void process(Animal a) {
    a.makeSound(); // Each animal implements makeSound() correctly
}
```

### ⚠️ Mistake 4: Not Using `@Override` in Java
```java
class Dog extends Animal {
    // Typo: "Speak" not "speak" -- silently creates new method, doesn't override!
    void Speak() { System.out.println("Woof"); }
}

// Fix: always use @Override
@Override
void speak() { System.out.println("Woof"); } // Compiler catches typos
```

### ⚠️ Mistake 5: Overloading with Null Arguments
```java
void process(String s) { System.out.println("String: " + s); }
void process(Object o) { System.out.println("Object: " + o); }

process(null); // AMBIGUOUS? No -- resolves to most specific: process(String)
// But if you add process(Integer i), it becomes ambiguous -> compile error!
```

---

## 11. Best Practices

### Design
- **Open-Closed Principle**: Design interfaces so adding new types doesn't require modifying existing code
- **No instanceof in business logic**: Replace with polymorphism
- **Interface segregation**: Small, focused interfaces over fat ones
- **Return types**: Return the most abstract type callers need

### 🚀 Performance
- Use `final` on leaf classes to help JIT devirtualize
- Avoid deep inheritance chains — megamorphic call sites kill JIT optimization
- For ultra-hot paths: consider CRTP (C++) or sealed classes (Java 17) for static dispatch
- Measure before optimizing vtable overhead — JIT usually handles it

### Naming
- Interfaces: noun (capability) → `Printable`, `Comparable`, `Runnable` or noun (`Payment`, `Repository`)
- Abstract classes: often `Abstract` prefix → `AbstractList`, `AbstractHandler`

---

## 12. Complexity

| Operation | Cost | Notes |
|-----------|------|-------|
| Overloaded method call | O(1) zero overhead | Compile-time resolved, direct call |
| Virtual method call (C++) | O(1) + ~3-5ns | 2 pointer dereferences |
| JIT-devirtualized call | O(1) zero overhead | After JIT warms up (after ~10k calls) |
| Python dict lookup | O(1) avg but slow | Dict traversal per call |
| instanceof check | O(1) or O(d) hierarchy depth | Modern JVM = O(1) via type check cache |

### Cache Impact
```
Virtual call cache behavior:
- Monomorphic site: vtable in L1 cache = fast
- Megamorphic site: thrashes instruction cache = slow
- JIT deoptimization: if assumption violated, recompile
```

---

## 13. Advantages

| Advantage | Example |
|-----------|---------|
| **Open-Closed Principle** | Add `ApplePay` without touching `processPayment()` |
| **Eliminates if-else chains** | Replace type-checking with dispatch |
| **Uniform interface** | Same code processes any `Shape`, `Payment`, `Animal` |
| **Testability** | Mock implementations replace real ones in tests |
| **Plugin architecture** | Load new implementations at runtime |
| **Code reduction** | One algorithm works on all types |

---

## 14. Disadvantages

| Disadvantage | When It Hurts |
|-------------|---------------|
| **vtable overhead** | 3-5ns per virtual call in hot loops |
| **Harder debugging** | Stack traces jump between implementations |
| **Megamorphic degradation** | Too many implementations kills JIT |
| **Duck typing risks** | Python: wrong type causes runtime errors |
| **Overuse** | Polymorphism everywhere makes code hard to follow |

---

## 15. Comparison Table

### Overloading vs Overriding

| Aspect | Overloading | Overriding |
|--------|-------------|------------|
| Resolved at | Compile time | Runtime |
| Class requirement | Same class | Different class (parent-child) |
| Method signature | DIFFERENT parameters | SAME signature |
| Return type | Can differ | Must be same or covariant |
| Access modifier | Can differ | Cannot be more restrictive |
| `static` allowed | Yes | No (method hiding, not overriding) |
| Polymorphism type | Compile-time | Runtime |

### Static vs Dynamic Polymorphism

| Aspect | Static | Dynamic |
|--------|--------|---------|
| When resolved | Compile time | Runtime |
| Mechanism | Name mangling, templates | vtable/vptr, interface dispatch |
| Cost | Zero | ~3-5ns per call |
| Flexibility | Fixed at compile | Runtime selection |
| Examples | Overloading, CRTP, templates | virtual, interface, abstract |

### Interface vs Abstract Class (Polymorphism angle)

| Aspect | Interface | Abstract Class |
|--------|-----------|---------------|
| Multiple "base types" | Yes (multiple implements) | No (single extends) |
| State sharing | No | Yes |
| Default implementation | Java 8+ default methods | Yes |
| When for polymorphism | Pure behavioral contract | Mix of contract + shared code |

---

## 16. Design Pattern Connection

| Pattern | How Polymorphism Enables It |
|---------|----------------------------|
| **Strategy** | Algorithm interface, multiple implementations, swap at runtime |
| **Observer** | Observer interface; update() called polymorphically on all subscribers |
| **Factory Method** | Base factory method overridden by subclasses to create specific types |
| **Command** | Command interface; execute() called polymorphically |
| **Visitor** | Double dispatch: polymorphism on both visitor and element |
| **Template Method** | Abstract steps filled in polymorphically by subclasses |
| **Decorator** | Decorator and component share same interface |
| **Composite** | Leaf and Composite share Component interface; treat uniformly |
| **State** | State interface; behavior changes as state object changes |
| **Adapter** | Adapter implements target interface; delegates to adaptee |

---

## 17. System Design Connection

### Microservices
- **gRPC**: Protobuf-generated interfaces; multiple language implementations
- **Service discovery**: `ServiceLocator` interface; Consul/Eureka implementations
- **Circuit breaker**: `CircuitBreaker` interface; Resilience4j, Hystrix implementations

### Message Queues
- `MessageConsumer` interface; Kafka, RabbitMQ, SQS implementations
- Same consumer code works with any queue system

### Databases
- Repository pattern: `UserRepository` interface; `JpaUserRepository`, `MongoUserRepository`
- Cache: `CacheProvider` interface; Redis, Memcached, Hazelcast implementations

### REST APIs
- `HttpClient` interface; `RestTemplate`, `WebClient`, `Feign` implementations
- `Serializer` interface; Jackson, Gson, Protobuf implementations

---

## 18. Multithreading Connection

### Thread-Safe Polymorphic Objects
```java
// Service objects are stateless (no fields) = naturally thread-safe
public class PaymentService {
    private final Payment payment; // Injected — could be any implementation

    public PaymentService(Payment payment) { this.payment = payment; }

    public void process(double amount) {
        payment.pay(amount); // Thread-safe if payment implementation is thread-safe
    }
}
```

### Polymorphism + Locks
```java
// Different implementations may have different sync needs
interface Counter {
    void increment();
    int get();
}

class UnsafeCounter implements Counter {
    private int count = 0;
    @Override public void increment() { count++; } // NOT thread-safe
    @Override public int get() { return count; }
}

class AtomicCounter implements Counter {
    private final AtomicInteger count = new AtomicInteger(0);
    @Override public void increment() { count.incrementAndGet(); } // Thread-safe
    @Override public int get() { return count.get(); }
}

// Swap implementations for thread safety without changing calling code
Counter counter = new AtomicCounter(); // Thread-safe polymorphic swap
```

---

## 19. Company Interview Perspective

### Google
- "Design a polymorphic ranking system for search results" (multiple ranking algorithms)
- vtable overhead questions for search serving
- JIT devirtualization and its limits
- CRTP for compile-time polymorphism in performance-critical code

### Goldman Sachs
- Polymorphic pricing engines for different financial instruments
- Visitor pattern for tax/risk calculations across instrument types
- Open-closed principle for regulatory reporting

### Microsoft
- C++: virtual dispatch in Windows kernel/COM components
- C#: generics + interfaces for polymorphic data structures
- .NET JIT devirtualization of sealed/final types

### Amazon
- "Design a notification system that supports Email, SMS, Push, Slack" — classic runtime polymorphism
- "How do you ensure adding a new notification type requires zero changes to existing code?"
- Repository pattern polymorphism (DynamoDB vs RDS)

### Meta
- Python duck typing: how does Facebook's Django ORM use it
- PyPy vs CPython: different vtable implementations, performance trade-offs
- C++: polymorphic plugin systems in Hack/HHVM

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | ⚠️ Can you overload based on return type only? | No — return type is not part of method signature |
| 2 | Can you override a final method? | No — `final` prevents overriding |
| 3 | ⚠️ Overloaded method with null argument? | Resolves to most specific type; ambiguous if multiple match |
| 4 | Is polymorphism possible without inheritance? | Yes — via interfaces (Java), duck typing (Python), CRTP (C++) |
| 5 | What is a vtable and how big is it? | Per-class table of function pointers; n pointers for n virtual methods |
| 6 | ⚠️ Why can't you overload on return type? | Call site can't determine expected return type; ambiguous |
| 7 | What is double dispatch? | Method resolution based on two types; requires Visitor pattern |
| 8 | How does Python achieve polymorphism without interfaces? | Duck typing — method name lookup in `__dict__` / MRO |
| 9 | ⚠️ What is method hiding vs overriding for static methods? | Static = compile-time resolution; hiding, not overriding |
| 10 | What is a megamorphic call site in JVM? | Site with 3+ implementations; JVM abandons inline cache, uses vtable |
| 11 | Can constructors be polymorphic? | No — object type is fixed at construction; vptr set in constructor |
| 12 | What is CRTP and how does it achieve zero-cost polymorphism? | Template tricks: `Base<Derived>` calls `static_cast<Derived*>(this)->method()` at compile time |
| 13 | ⚠️ Polymorphism vs Overloading: which is runtime? | Polymorphism (overriding) = runtime; Overloading = compile time |
| 14 | What is "type erasure" and how does it affect polymorphism? | Java generics are erased at runtime; `List<String>` and `List<Integer>` have same runtime type |
| 15 | Can you override `equals()` to violate symmetry? | Yes, but it breaks HashMap/HashSet — must maintain contract |
| 16 | What is covariant parameter (contravariance)? | Theoretically: override accepts supertype of parent's parameter; Java doesn't support (would be overloading) |
| 17 | ⚠️ In Java, is every method virtual? | Instance methods are virtual; static, private, final are not |
| 18 | What is an abstract method vs an interface method? | Abstract method: in abstract class, may have some implementation; interface: pure contract |
| 19 | What is open recursion? | `this` in a method refers to the actual runtime type — enables polymorphism within a class |
| 20 | ⚠️ Calling `super.method()` vs not calling it — when is each correct? | Call super when extending behavior; don't call when replacing behavior entirely |

---

## 21. Coding Problems

### Easy — Polymorphic Shape Area (LeetCode-style Design)
```java
interface Shape { double area(); }

class Circle implements Shape {
    Circle(double r) { this.r = r; }
    double r;
    public double area() { return Math.PI * r * r; }
}

class Square implements Shape {
    Square(double s) { this.s = s; }
    double s;
    public double area() { return s * s; }
}

// Polymorphic
double totalArea(List<Shape> shapes) {
    return shapes.stream().mapToDouble(Shape::area).sum();
}
```

### Medium — Design a Rate Limiter with Multiple Strategies
```java
interface RateLimiter {
    boolean allow(String userId);
}

class TokenBucketLimiter implements RateLimiter {
    // Token bucket implementation
    public boolean allow(String userId) { /* ... */ return true; }
}

class SlidingWindowLimiter implements RateLimiter {
    // Sliding window implementation
    public boolean allow(String userId) { /* ... */ return true; }
}

class FixedWindowLimiter implements RateLimiter {
    // Fixed window implementation
    public boolean allow(String userId) { /* ... */ return true; }
}

// API gateway uses polymorphic rate limiter — swap strategy without code change
class ApiGateway {
    private final RateLimiter limiter;
    ApiGateway(RateLimiter limiter) { this.limiter = limiter; }

    public Response handleRequest(Request req) {
        if (!limiter.allow(req.getUserId()))
            return Response.tooManyRequests();
        return processRequest(req);
    }
}
```

### Hard — Polymorphic Expression Evaluator (Interpreter Pattern)
```java
interface Expression {
    int evaluate();
}

class Number implements Expression {
    final int value;
    Number(int value) { this.value = value; }
    public int evaluate() { return value; }
}

class Add implements Expression {
    final Expression left, right;
    Add(Expression left, Expression right) { this.left = left; this.right = right; }
    public int evaluate() { return left.evaluate() + right.evaluate(); }
}

class Multiply implements Expression {
    final Expression left, right;
    Multiply(Expression left, Expression right) { this.left = left; this.right = right; }
    public int evaluate() { return left.evaluate() * right.evaluate(); }
}

// (2 + 3) * 4 = 20
Expression expr = new Multiply(new Add(new Number(2), new Number(3)), new Number(4));
System.out.println(expr.evaluate()); // 20
// Fully polymorphic: add Power, Subtract without changing evaluate loop
```

---

## 22. Revision Sheet

| Concept | Key Point |
|---------|-----------|
| Compile-time polymorphism | Overloading, templates — resolved by compiler, zero runtime cost |
| Runtime polymorphism | Overriding via virtual dispatch — vtable lookup |
| vtable | Per-class static table of function pointers |
| vptr | Per-object hidden pointer to vtable (8 bytes) |
| Dynamic dispatch | Runtime selection of override via vptr -> vtable |
| Duck typing | If it has the method, it works — no inheritance (Python) |
| Overloading | Same name, different params — compile time |
| Overriding | Same signature, different class — runtime |
| Open-Closed Principle | Add new types without changing existing code |
| JIT devirtualization | JVM optimizes single-implementation calls to direct calls |
| Megamorphic | 3+ implementations at one call site — JIT gives up |
| CRTP | Static polymorphism via templates — zero vtable cost |
| Double dispatch | Visitor pattern — dispatch on two types |
| @Override | Compile-time check that method actually overrides parent |

### Common Pitfalls
- ⚠️ Overloading resolved at compile time based on STATIC type
- ⚠️ Can't overload on return type only
- ⚠️ Static methods are hidden, not overridden
- ⚠️ `instanceof` overuse defeats polymorphism
- ⚠️ Not using `@Override` — silent bugs

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| Two types of polymorphism? | Compile-time (overloading) and Runtime (overriding) |
| Overloading resolved when? | Compile time, based on static type |
| Overriding resolved when? | Runtime, via vtable |
| What is vtable? | Per-class array of function pointers for virtual methods |
| What is vptr? | Per-object hidden pointer to vtable |
| Can static methods be overridden? | No — they are hidden, not overridden |
| Can final methods be overridden? | No |
| What is duck typing? | No inheritance needed — object just needs the method |
| What is double dispatch? | Method resolution on two types; Visitor pattern |
| What is CRTP? | Static polymorphism via template specialization |
| JIT devirtualization? | JVM inlines virtual calls if only one impl observed |
| Megamorphic call site? | 3+ impls; JVM uses vtable, no inline cache |
| Can you overload on return type? | No — not part of signature |
| What is open recursion? | `this` refers to actual runtime type |
| Type erasure impact? | Java generics erased; List<String> == List<Integer> at runtime |
| Observer pattern uses? | Polymorphic update() on all registered observers |
| Strategy pattern uses? | Polymorphic algorithm selection at runtime |
| What is covariant return? | Override can return subtype of parent return type |
| @Override annotation? | Compile-time check; prevents silent hiding bugs |
| Null arg overloading? | Resolves to most specific type; compile error if ambiguous |
| What is method hiding? | Static method in subclass with same name as static in parent |
| Can interfaces be polymorphic? | Yes — multiple implementations of same interface |
| What eliminates vtable cost? | Final classes, sealed types, JIT devirtualization |
| Python polymorphism mechanism? | Dict lookup through MRO chain |
| C++ polymorphism mechanism? | vptr -> vtable -> function pointer |

---

## 24. Cheat Sheet

### Top 20 Facts
1. Polymorphism = one interface, many implementations
2. Compile-time: overloading — zero runtime cost
3. Runtime: overriding via vtable — ~3-5ns cost
4. vtable: per-class, in code segment; vptr: per-object, 8 bytes
5. All Java instance methods are virtual by default
6. C++ requires `virtual` keyword for dynamic dispatch
7. Static methods are resolved at compile time (method hiding, not overriding)
8. `@Override` is compile-time safety net — always use it
9. Can't overload on return type only
10. Null arg resolves to most specific overloaded type
11. JIT devirtualizes monomorphic/bimorphic call sites automatically
12. Megamorphic site (3+ impls) defeats JIT optimization
13. CRTP enables zero-cost static polymorphism in C++
14. Duck typing (Python): no inheritance required, just the method
15. Open-Closed Principle: polymorphism enables extension without modification
16. Visitor pattern solves double dispatch problem
17. Type erasure: Java generics erased at runtime
18. `instanceof` overuse is a smell — replace with polymorphism
19. `final` class helps JIT devirtualize all calls to it
20. Strategy pattern = classic runtime polymorphism application

### Top 20 Keywords
`polymorphism`, `overloading`, `overriding`, `vtable`, `vptr`, `virtual`, `dynamic dispatch`, `static binding`, `duck typing`, `CRTP`, `monomorphic`, `megamorphic`, `JIT`, `devirtualization`, `Visitor`, `Strategy`, `Open-Closed`, `covariant`, `@Override`, `interface`

---

## 25. Final Interview Summary

### 5-Minute Revision
- Polymorphism = one interface, many forms
- Compile-time: overloading (zero cost, compile resolved)
- Runtime: overriding via vtable (vptr -> vtable -> function)
- Java: all instance methods virtual; C++: need `virtual`
- Static methods → method hiding, NOT overriding
- @Override: always use — compiler safety
- Open-Closed Principle: add new types, zero existing code change
- JIT devirtualizes monomorphic call sites

### 15-Minute Revision
Add:
- Double dispatch and Visitor pattern
- CRTP for static polymorphism
- Megamorphic call site degradation
- Duck typing (Python) vs structural typing
- Type erasure impact
- Strategy, Observer, Command, Composite patterns
- `instanceof` as code smell
- Sealed classes (Java 17) for exhaustive polymorphism
- Covariant return types

### Night-Before Interview Revision
1. ⭐ Overloading = compile-time; Overriding = runtime
2. ⭐ vtable = per-class function pointer array; vptr = per-object pointer
3. ⭐ @Override always! Catches silent overloading mistakes
4. ⭐ Static methods = method hiding, NOT overriding
5. ⭐ No overloading on return type
6. ⭐ Open-Closed: add implementations, never modify existing code
7. ⭐ JIT devirtualizes — don't over-optimize prematurely
8. ⭐ Visitor = double dispatch solution
9. ⭐ instanceof = design smell; replace with polymorphism
10. ⭐ CRTP = static polymorphism (C++), zero vtable cost
