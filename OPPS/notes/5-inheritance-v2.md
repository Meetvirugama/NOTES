# Inheritance — Industry-Level Interview Notes

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
Inheritance lets a class **acquire properties and behaviors** of another class. The child class IS-A type of the parent class.

### Technically
Inheritance is a mechanism by which a **derived class (child/subclass)** inherits **fields and methods** from a **base class (parent/superclass)**, while being able to extend or override them. It establishes an **IS-A relationship** and enables **code reuse** and **polymorphic substitution**.

### From an Interviewer's Perspective
> "When I ask about inheritance, I'm testing if you know: IS-A vs HAS-A, the diamond problem, fragile base class problem, when to prefer composition, method resolution order (MRO), vtable mechanics, and why Java chose single inheritance. Saying 'it's for code reuse' is the beginner answer."

⭐ **Key rule**: "Favor composition over inheritance" — GoF, Effective Java

---

## 2. Why It Exists

### Problem It Solves
Without inheritance, every class must implement common behavior independently — massive duplication.

```java
// Without inheritance: massive duplication
class Dog {
    String name;
    void eat() { System.out.println("Eating"); }
    void sleep() { System.out.println("Sleeping"); }
    void bark() { System.out.println("Barking"); }
}

class Cat {
    String name;
    void eat() { System.out.println("Eating"); }   // DUPLICATE
    void sleep() { System.out.println("Sleeping"); } // DUPLICATE
    void meow() { System.out.println("Meowing"); }
}
```

### What Happens Without It
- Code duplication → bugs in one copy don't get fixed in others
- No polymorphic substitution (can't write `Animal a = new Dog()`)
- No extensibility (adding a new animal requires rewriting common code)

### Real Software Examples
- **Java Collections**: `AbstractList` defines common list behavior; `ArrayList`, `LinkedList` inherit it
- **Android Views**: `View` is the base for `TextView`, `Button`, `ImageView` — all inherit layout logic
- **Spring Framework**: `AbstractApplicationContext` provides lifecycle management; specific contexts extend it
- **HTTP clients**: Base `HttpRequest` class; `GetRequest`, `PostRequest` extend with specific behavior

---

## 3. Internal Working

### Memory Layout with Inheritance

The derived class object **contains the base class sub-object** first (for vtable compatibility).

```
C++ Memory Layout:

Animal object:          Dog object (extends Animal):
+------------------+    +------------------+
| vptr             |    | vptr             |  <- Dog's vtable
+------------------+    +------------------+
| name (Animal)    |    | name (Animal)    |  <- Base part first
+------------------+    +------------------+
                        | breed (Dog)      |  <- Derived part appended
                        +------------------+
```

**Key insight**: Base class sub-object sits at offset 0 in derived object. This is why you can safely cast a `Dog*` to `Animal*` — the Animal sub-object IS at the start.

### vtable with Inheritance

```
Animal vtable:              Dog vtable:
+-------------------+       +-------------------+
| Animal::speak()   |       | Dog::speak()      |  <- Overridden!
| Animal::eat()     |       | Animal::eat()     |  <- Inherited (same ptr)
| Animal::toString()|       | Animal::toString()|
+-------------------+       +-------------------+

Dog object.vptr -----------> Dog vtable
                              (Dog::speak overrides Animal::speak)
```

### Method Resolution at Runtime
```
dog.speak()
  1. Load dog reference
  2. Follow vptr -> Dog vtable
  3. Read slot 0 (speak()) -> Dog::speak address
  4. CALL Dog::speak()
```

### JVM Internals — Method Resolution Order
Java uses a flat vtable per class. Each class's vtable is a copy of the parent's vtable with overridden slots replaced.

```
Object vtable:            Animal vtable:         Dog vtable:
+-------------------+     +-------------------+  +-------------------+
| Object::equals()  |  -> | Object::equals()  |->| Object::equals()  |
| Object::hashCode()|     | Object::hashCode()|  | Object::hashCode()|
| Object::toString()|     | Object::toString()|  | Object::toString()|
                          | Animal::speak()   |  | Dog::speak()      | <- overridden
                          | Animal::eat()     |  | Animal::eat()     |
```

### Python MRO (Method Resolution Order)
Python uses **C3 Linearization algorithm** for MRO — determines method lookup order in multiple inheritance.

```python
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
# C3: D -> B -> C -> A -> object (left-to-right, each appears once)
```

### Constructor Chain
```
new Dog("Rex", "Labrador")
  1. Allocate memory for Dog (size = Animal fields + Dog fields)
  2. Call Dog constructor
  3. Dog constructor calls super() (Animal constructor) FIRST
  4. Animal constructor initializes Animal fields (name)
  5. Dog constructor initializes Dog fields (breed)
  6. Object is ready
```

---

## 4. Syntax

### C++
```cpp
#include <iostream>
#include <string>
using namespace std;

class Animal {
protected:    // Accessible by subclasses
    string name;
    int age;

public:
    Animal(string name, int age) : name(name), age(age) {}

    virtual void speak() {    // virtual = allows overriding
        cout << name << " makes a sound" << endl;
    }

    virtual ~Animal() {}      // Virtual destructor — CRITICAL in C++!
};

class Dog : public Animal {   // public inheritance preserves access
    string breed;

public:
    Dog(string name, int age, string breed)
        : Animal(name, age), breed(breed) {}  // Call base constructor

    void speak() override {   // override keyword = compiler check
        cout << name << " barks!" << endl;
    }

    void fetch() { cout << name << " fetches the ball!" << endl; }
};

int main() {
    Animal* a = new Dog("Rex", 3, "Labrador");
    a->speak();   // "Rex barks!" - Dynamic dispatch
    delete a;     // Virtual destructor ensures Dog's destructor is called
}
```

### Java
```java
// Base class
public abstract class Animal {
    protected String name;
    protected int age;

    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void eat() { System.out.println(name + " is eating"); }

    // Template method pattern
    public abstract void speak(); // Forces subclasses to implement
}

// Single inheritance
public class Dog extends Animal {
    private String breed;

    public Dog(String name, int age, String breed) {
        super(name, age);   // Must call super() first in Java
        this.breed = breed;
    }

    @Override
    public void speak() { System.out.println(name + " barks!"); }

    public void fetch() { System.out.println(name + " fetches!"); }
}

// Multiple interface inheritance is allowed
public class GuideDog extends Dog implements ServiceAnimal, Trainable {
    public GuideDog(String name, int age, String breed) {
        super(name, age, breed);
    }

    @Override
    public void guide() { System.out.println(name + " guides their owner"); }
}
```

### Python
```python
class Animal:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def speak(self) -> str:
        raise NotImplementedError("Subclasses must implement speak()")

    def eat(self) -> None:
        print(f"{self.name} is eating")


class Dog(Animal):
    def __init__(self, name: str, age: int, breed: str):
        super().__init__(name, age)  # Call parent constructor
        self.breed = breed

    def speak(self) -> str:
        return f"{self.name} says: Woof!"

    def fetch(self) -> None:
        print(f"{self.name} fetches the ball!")


# Multiple inheritance (Python supports it)
class ServiceDog(Dog):
    def __init__(self, name: str, age: int, breed: str, service_type: str):
        super().__init__(name, age, breed)
        self.service_type = service_type

    def perform_service(self) -> None:
        print(f"{self.name} performs {self.service_type} service")


# Usage
dog = Dog("Rex", 3, "Labrador")
print(dog.speak())  # Rex says: Woof!
dog.eat()           # Rex is eating (inherited from Animal)
print(isinstance(dog, Animal))  # True — IS-A relationship
print(Dog.__mro__)  # Method Resolution Order
```

### Key Differences

| Feature | C++ | Java | Python |
|---------|-----|------|--------|
| Multiple class inheritance | Yes | No (single only) | Yes (with MRO) |
| Multiple interface | N/A (pure virtual) | Yes | N/A (mixins instead) |
| Virtual by default | No (need `virtual`) | Yes (all instance methods) | Yes (all methods) |
| Virtual destructor | Must declare explicitly | Handled by GC | GC handles it |
| `super` call | `BaseClass::method()` | `super.method()` | `super().method()` |
| Inheritance access | `public/protected/private` | `extends` (always public) | Direct parentheses |
| Abstract method | Pure virtual: `= 0` | `abstract` keyword | `raise NotImplementedError` |

---

## 5. Visual Diagrams

### Inheritance Tree
```
                    +----------+
                    |  Animal  |
                    | - name   |
                    | + eat()  |
                    | + speak()|
                    +----------+
                   /            \
          +-------+              +-------+
          |  Dog  |              |  Cat  |
          |+breed |              |+color |
          |+fetch()|             |+purr()|
          |+speak()|             |+speak()|
          +-------+              +-------+
              |
        +-----------+
        | GuideDog  |
        |+service   |
        |+guide()   |
        +-----------+
```

### Memory Layout (C++)
```
Dog Object in Memory:
+-------------------------+
| vptr                    |  ---> Dog's vtable
+-------------------------+
|  [Animal sub-object]    |
|  name: "Rex"            |
|  age: 3                 |
+-------------------------+
|  [Dog fields]           |
|  breed: "Labrador"      |
+-------------------------+

Dog's vtable:
+----------------------------+
| 0: Dog::speak()   address  |  <- Overrides Animal::speak
| 1: Animal::eat()  address  |  <- Inherited unchanged
| 2: Animal::~Animal address |  <- Virtual destructor
+----------------------------+
```

### Constructor Call Stack
```
Call: new GuideDog("Rex", 3, "Labrador", "Guide")

Call Stack:
+-------------------------------+
| GuideDog::GuideDog()          |  <- 4. GuideDog fields init
+-------------------------------+
| Dog::Dog()                    |  <- 3. Dog fields init
+-------------------------------+
| Animal::Animal()              |  <- 2. Animal fields init
+-------------------------------+
| Object::Object()              |  <- 1. Object init (Java)
+-------------------------------+

Order: bottom-up initialization, top-down destruction
```

### IS-A vs HAS-A
```
IS-A (Inheritance):             HAS-A (Composition):
Dog IS-A Animal                 Car HAS-A Engine

Animal                          Car
  ^                             +--------+
  |                             | engine |---> Engine
Dog                             | wheels |---> Wheel[]
                                +--------+
```

---

## 6. Real World Analogy

### Car Manufacturing
- **Base class**: `Vehicle` (has engine, wheels, can move)
- **Derived**: `Car` extends Vehicle + adds 4 doors, passenger capacity
- **Derived**: `Truck` extends Vehicle + adds cargo capacity, towing
- **Derived**: `ElectricCar` extends Car + overrides `fuelType()` to "Electric"

### Bank (Goldman Sachs)
- **Base**: `FinancialInstrument` (symbol, price, timestamp, execute())
- **Derived**: `Stock` extends with market_cap, dividend_yield
- **Derived**: `Bond` extends with maturity_date, coupon_rate
- **Derived**: `Option` extends with strike_price, expiry, put/call type

### Hospital
- **Base**: `MedicalStaff` (id, name, department, schedule())
- **Derived**: `Doctor` extends with specialization, prescribe()
- **Derived**: `Nurse` extends with ward, administer_medication()
- **Derived**: `Surgeon` extends Doctor with surgery_type, operate()

### E-Commerce (Amazon)
- **Base**: `User` (id, email, password, authenticate())
- **Derived**: `Customer` extends with cart, order history, placeOrder()
- **Derived**: `Seller` extends with inventory, listProduct()
- **Derived**: `Admin` extends with permissions, manageUsers()

### Operating System
- **Base**: `FileSystemNode` (path, permissions, timestamp, read(), write())
- **Derived**: `File` extends with content, size
- **Derived**: `Directory` extends with children list, addChild()
- **Derived**: `SymbolicLink` extends with target path, resolve()

### Game Development
- **Base**: `GameObject` (position, velocity, update(), render())
- **Derived**: `Enemy` extends with health, attackPlayer()
- **Derived**: `Player` extends with inventory, jump()
- **Derived**: `Boss` extends Enemy with specialAttack(), phaseChange()

---

## 7. Interview Explanation

### 30 Seconds
> "Inheritance is the IS-A mechanism where a derived class inherits fields and methods from a base class. The derived class can extend with new behavior or override inherited methods. It enables polymorphism — you can write code against the base type and have derived types work transparently."

### 1 Minute
> "Inheritance creates a class hierarchy where derived classes inherit the interface and optionally the implementation from the base. In memory, the base class sub-object sits at the top of the derived object's memory layout, which is why upcast (derived to base) is always safe. Method overriding works through virtual dispatch — the compiler installs a vtable pointer in each polymorphic object, and at runtime the call goes through the vtable to the most derived implementation.

> The critical distinction is IS-A vs HAS-A. Inheritance is appropriate only when the relationship is truly IS-A. A `Dog` IS-AN `Animal` — valid. A `Car` is NOT an `Engine` — that's a HAS-A, use composition."

### 3 Minutes
> "Let me go deep. Inheritance establishes three things simultaneously: code reuse (derived gets base's implementation), type substitutability (Liskov Substitution Principle — derived can replace base anywhere), and polymorphism enabling.

> Internally, when you have virtual methods in C++, the compiler inserts a hidden `vptr` field at offset 0 in every polymorphic class. This pointer references the class's vtable — a static array of function pointers. Each entry in the vtable corresponds to a virtual method. Derived classes get their own vtable where overridden methods have updated pointers.

> In Java, all instance methods are virtual by default. The JVM maintains an inline method table per class. The JIT compiler is smart enough to devirtualize calls when it can prove the runtime type is always the same — eliminating vtable overhead.

> The diamond problem is the canonical multi-inheritance issue: if B and C both extend A, and D extends B and C, which `A::method()` does D inherit? C++ solves this with virtual inheritance. Java sidesteps it by allowing only single class inheritance — but allows multiple interface inheritance. Python uses C3 Linearization to define a deterministic MRO.

> In practice at senior level, inheritance is often overused. The Fragile Base Class problem is real: changing a base class method can silently break subclasses. This is why 'favor composition over inheritance' is in both GoF and Effective Java."

---

## 8. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What is IS-A vs HAS-A? | IS-A = inheritance; HAS-A = composition (field of another type) |
| What is the difference between `extends` and `implements`? | `extends` = class inheritance; `implements` = interface implementation |
| Can you override a static method? | No — static methods are resolved at compile time (method hiding, not overriding) |
| What is `super`? | Keyword to call parent class constructor or method |
| Can a class extend multiple classes in Java? | No — only single class inheritance; multiple interface implementation allowed |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| What is the diamond problem? | Ambiguity when D inherits from B and C which both inherit from A |
| What is method hiding vs method overriding? | Override = virtual, runtime dispatch; Hide = static, compile-time resolution |
| What is the Liskov Substitution Principle? | Derived class must be substitutable for its base class without breaking correctness |
| What is constructor chaining? | Child constructor calls parent constructor (super()) to initialize inherited state |
| What is protected access? | Accessible within class and all subclasses (but not outside package in Java) |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| Why is `virtual ~Destructor()` critical in C++? | Without it, deleting derived via base pointer only calls base destructor — resource leak |
| What is the fragile base class problem? | Base class change silently breaks subclass behavior; reason to prefer composition |
| What is covariant return type? | Override can return subtype of parent's return type (e.g., return `Dog` where `Animal` expected) |
| How does Python MRO work? | C3 Linearization: left-to-right depth-first, removes duplicates except last occurrence |
| What is virtual inheritance in C++? | Ensures only one copy of base exists in diamond inheritance hierarchy |

### Expert
| Question | Expected Answer |
|----------|-----------------|
| Explain vtable layout for multiple inheritance in C++ | Multiple vtables; object contains multiple vptrs; thunks adjust `this` pointer for each base |
| Why can't you call virtual methods from constructors in C++ safely? | Derived part not yet initialized; vtable points to base at that moment |
| What is CRTP (Curiously Recurring Template Pattern)? | `class Derived : public Base<Derived>` — enables static polymorphism without vtable |
| How does Java JIT devirtualize virtual calls? | Class hierarchy analysis: if only one implementation observed, inline direct call |

### 💼 Google Level
> *"Design an inheritance hierarchy for a content delivery system serving images, videos, and documents. When would you switch to composition?"*

### 💼 Goldman Sachs Level
> *"We have a `FinancialInstrument` base class. An `Option` and a `Future` both inherit from it. Now we need a `FuturesOption` (a futures contract on an option). How do you design this without the diamond problem?"*

---

## 9. Coding Examples

### Basic Example
```java
public class Shape {
    protected String color;

    public Shape(String color) { this.color = color; }

    public double area() { return 0; }

    @Override
    public String toString() { return String.format("Shape[color=%s, area=%.2f]", color, area()); }
}

public class Rectangle extends Shape {
    private double width, height;

    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }

    @Override
    public double area() { return width * height; }
}

public class Circle extends Shape {
    private double radius;

    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }

    @Override
    public double area() { return Math.PI * radius * radius; }
}
```

### Intermediate — Template Method Pattern
```java
// Template Method Pattern: base class defines algorithm skeleton
public abstract class DataProcessor {
    // Template method — defines the algorithm
    public final void process() {
        readData();
        processData();
        writeData();
    }

    protected abstract void readData();
    protected abstract void processData();

    protected void writeData() {
        System.out.println("Writing processed data..."); // Default implementation
    }
}

public class CSVProcessor extends DataProcessor {
    @Override
    protected void readData() { System.out.println("Reading CSV file"); }

    @Override
    protected void processData() { System.out.println("Parsing CSV rows"); }
}

public class JSONProcessor extends DataProcessor {
    @Override
    protected void readData() { System.out.println("Reading JSON stream"); }

    @Override
    protected void processData() { System.out.println("Deserializing JSON objects"); }

    @Override
    protected void writeData() { System.out.println("Writing to NoSQL store"); }
}
```

### Advanced — Avoid Diamond via Interface Default Methods (Java 8+)
```java
interface Flyable {
    default void move() { System.out.println("Flying"); }
}

interface Swimmable {
    default void move() { System.out.println("Swimming"); }
}

// Compiler forces resolution of conflict
class Duck implements Flyable, Swimmable {
    @Override
    public void move() {
        Flyable.super.move(); // Explicit resolution
        Swimmable.super.move();
    }
}
```

### Production — Abstract Base Entity with Inheritance
```java
// Spring Boot: base entity with audit
@MappedSuperclass
public abstract class AuditableEntity {
    @Id @GeneratedValue private Long id;
    @CreatedDate private Instant createdAt;
    @CreatedBy private String createdBy;
    @LastModifiedDate private Instant updatedAt;
    @Version private Long version; // Optimistic locking

    // equals/hashCode by ID only
    @Override
    public final boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AuditableEntity e)) return false;
        return id != null && id.equals(e.id);
    }

    @Override
    public final int hashCode() { return getClass().hashCode(); }
}

@Entity
public class Order extends AuditableEntity {
    private String customerId;
    private BigDecimal totalAmount;
    @Enumerated private OrderStatus status;
    // Order-specific fields and methods
}

@Entity
public class Product extends AuditableEntity {
    private String sku;
    private BigDecimal price;
    private int stockQuantity;
}
```

### Interview Coding — LSP Violation Detection
```java
// LSP Violation: Square extends Rectangle
class Rectangle {
    protected int width, height;
    public void setWidth(int w) { this.width = w; }
    public void setHeight(int h) { this.height = h; }
    public int area() { return width * height; }
}

// VIOLATES LSP: Square breaks Rectangle's contract
class Square extends Rectangle {
    @Override
    public void setWidth(int w) { this.width = this.height = w; } // Breaks invariant!
    @Override
    public void setHeight(int h) { this.width = this.height = h; }
}

// Test that breaks LSP:
void test(Rectangle r) {
    r.setWidth(5);
    r.setHeight(4);
    assert r.area() == 20; // FAILS for Square!
}

// Fix: Don't use inheritance. Use separate classes or interface.
interface Shape { int area(); }
class Rectangle implements Shape { ... }
class Square implements Shape { ... }
```

---

## 10. Common Mistakes

### ⚠️ Mistake 1: Inheritance When Composition Is Correct
```java
// WRONG: Stack "is a" Vector? No! Stack USES a Vector
class Stack extends Vector { // Java's actual mistake! Exposes add(index, element)!
    public void push(E e) { addElement(e); }
}

// CORRECT: Composition
class Stack<E> {
    private final Deque<E> storage = new ArrayDeque<>();
    public void push(E e) { storage.push(e); }
    public E pop() { return storage.pop(); }
}
```

### ⚠️ Mistake 2: No Virtual Destructor in C++
```cpp
class Animal {
public:
    ~Animal() { cout << "Animal destroyed\n"; }  // NOT virtual!
};

class Dog : public Animal {
    int* data = new int[100];
public:
    ~Dog() { delete[] data; cout << "Dog destroyed\n"; }
};

Animal* a = new Dog();
delete a;  // Only Animal destructor called! Memory leak!

// Fix:
class Animal {
public:
    virtual ~Animal() {}  // Virtual destructor
};
```

### ⚠️ Mistake 3: Calling Virtual Methods in Constructor
```cpp
class Animal {
public:
    Animal() { speak(); } // BUG: calls Animal::speak, not Dog::speak
    virtual void speak() { cout << "Animal sound\n"; }
};

class Dog : public Animal {
public:
    void speak() override { cout << "Woof\n"; }
};

Dog d; // Prints "Animal sound", not "Woof"!
```

### ⚠️ Mistake 4: Breaking LSP
```java
// LSP violation: derived class throws where base doesn't
class Bird { public void fly() { /* flies */ } }
class Penguin extends Bird {
    @Override
    public void fly() { throw new UnsupportedOperationException("Penguins can't fly!"); }
    // Breaks LSP — can't substitute Penguin where Bird expected
}
```

### ⚠️ Mistake 5: Overriding vs Hiding Static Methods (Java)
```java
class Parent {
    public static void staticMethod() { System.out.println("Parent static"); }
    public void instanceMethod() { System.out.println("Parent instance"); }
}

class Child extends Parent {
    public static void staticMethod() { System.out.println("Child static"); }  // HIDES, not overrides
    @Override
    public void instanceMethod() { System.out.println("Child instance"); }     // OVERRIDES
}

Parent p = new Child();
p.staticMethod();    // "Parent static" — static is bound at compile time!
p.instanceMethod();  // "Child instance" — dynamic dispatch
```

---

## 11. Best Practices

### Design
- **IS-A test**: Before inheriting, ask "Is B truly a type of A?" If not, use composition
- **LSP compliance**: Derived class must fully honor the base class contract
- **Prefer shallow hierarchies**: Max 2-3 levels deep; beyond that, composition is cleaner
- **Program to base**: `Animal a = new Dog()` not `Dog d = new Dog()`
- **Seal leaf classes**: Mark final classes that should not be subclassed

### 🚀 Performance
- C++: Declare destructors `virtual` in base classes to avoid resource leaks
- Java: The JIT devirtualizes single-implementation calls automatically — don't over-abstract
- Python: `__slots__` in base class propagates to subclasses if properly declared

### Maintainability
- Document the class invariants in base classes
- Use `@Override` annotation always in Java — catches typos
- Avoid calling overridable methods in constructors
- Extract common behavior to protected helper methods rather than duplicating

---

## 12. Complexity

| Aspect | Complexity | Notes |
|--------|-----------|-------|
| Virtual method call | O(1) + constant overhead | 1-2 extra memory loads vs direct call |
| Constructor chain depth | O(d) where d = hierarchy depth | Each level's constructor runs once |
| Object size | Base size + derived fields + vptr | Base sub-object embedded in derived |
| vtable size | O(n) virtual methods | One pointer per virtual method |
| JIT devirtualization | O(1) after warm-up | JVM optimizes single-implementation |

### Memory Overhead
```
class Animal { vptr + name(8B) + age(4B) + padding(4B) } = 24B
class Dog extends Animal { Animal(24B) + breed(8B) } = 32B
class GuideDog extends Dog { Dog(32B) + serviceType(8B) } = 40B
```

---

## 13. Advantages

| Advantage | Example |
|-----------|---------|
| **Code reuse** | `Animal.eat()` implemented once, reused by Dog, Cat, Fish |
| **Polymorphism** | `List<Animal>` holds Dog, Cat, Bird — process uniformly |
| **Extensibility** | Add `Parrot extends Bird` without touching existing code |
| **Type safety** | Compiler ensures derived class has base class interface |
| **Framework hooks** | Spring/Android lifecycle methods (onCreate, onPause) |

---

## 14. Disadvantages

| Disadvantage | Impact |
|-------------|--------|
| **Fragile base class** | Base change can silently break all subclasses |
| **Deep hierarchy = tight coupling** | Change ripples through entire tree |
| **Inheritance breaks encapsulation** | Subclass can see protected internals of base |
| **Diamond problem** (C++/Python) | Ambiguity in multiple inheritance |
| **LSP violations** | Incorrect inheritance breaks substitutability |
| **Overuse** | Engineers inherit when composition is cleaner |

---

## 15. Comparison Table

### Inheritance vs Composition

| Aspect | Inheritance | Composition |
|--------|-------------|-------------|
| Relationship | IS-A | HAS-A |
| Coupling | Tight (base change affects derived) | Loose (dependency via interface) |
| Reusability | Share implementation directly | Reuse via delegation |
| Flexibility | Static hierarchy at compile time | Can swap implementations at runtime |
| Encapsulation | Broken (protected access) | Preserved |
| GoF Advice | Use sparingly | Prefer this |
| Example | `Dog extends Animal` | `Car has Engine` |

### Single vs Multiple Inheritance

| Aspect | Single (Java) | Multiple (C++/Python) |
|--------|--------------|----------------------|
| Diamond problem | Avoided | Exists; need virtual inheritance |
| Simplicity | High | Complex |
| Flexibility | Lower | Higher |
| Common in | Java, C# | C++, Python |
| Interface workaround | Multiple `implements` | N/A |

### Abstract Class vs Interface

| Aspect | Abstract Class | Interface |
|--------|---------------|-----------|
| Can have state | Yes (fields) | No (only static final in Java) |
| Constructor | Yes | No |
| Method implementations | Yes | Yes (default methods, Java 8+) |
| Multiple inheritance | No (Java) | Yes (multiple implements) |
| When to use | Shared state + IS-A | Pure contract / capability |
| Access modifiers | Any | Public only (Java) |

---

## 16. Design Pattern Connection

| Pattern | Inheritance Role |
|---------|-----------------|
| **Template Method** | Base defines skeleton algorithm; subclasses fill in steps |
| **Factory Method** | Base declares factory method; subclasses override to create specific product |
| **Strategy** | Interface (abstract base) for interchangeable algorithms |
| **Command** | Abstract `Command` base; concrete commands extend |
| **Observer** | Abstract `Observer` base; concrete observers extend |
| **Composite** | `Component` base; `Leaf` and `Composite` both extend |
| **Chain of Responsibility** | `Handler` base; concrete handlers extend and chain |

---

## 17. System Design Connection

### Microservices
- **Base service class**: Common HTTP handling, health checks, config loading — all services extend
- **Event base class**: `DomainEvent` base with timestamp, eventId — specific events extend

### REST APIs
- **Controller hierarchy**: `BaseController` with auth validation; specific controllers extend
- **Response hierarchy**: `ApiResponse<T>` base; `SuccessResponse`, `ErrorResponse` extend

### Databases
- **ORM entities**: JPA/Hibernate `@MappedSuperclass` for audit fields — all entities extend
- **Repository hierarchy**: Spring Data `JpaRepository` chain: `Repository <- CrudRepository <- JpaRepository`

### Cloud / Distributed Systems
- **Cloud provider abstraction**: `StorageClient` base; `S3Client`, `GCSClient`, `AzureClient` extend
- **Message consumer base**: `BaseConsumer` with retry/DLQ logic; specific consumers extend

---

## 18. Multithreading Connection

### Shared Inherited State
```java
class Counter {
    protected int count = 0;  // Protected = accessible by subclasses
}

class ThreadSafeCounter extends Counter {
    private final Object lock = new Object();

    public void increment() {
        synchronized (lock) { count++; }  // Protects inherited field
    }
}
```

### Thread-Local Inheritance Pattern
```java
class BaseWorker {
    private static final ThreadLocal<String> context = new ThreadLocal<>();

    protected void setContext(String ctx) { context.set(ctx); }
    protected String getContext() { return context.get(); }
}

class RequestWorker extends BaseWorker {
    public void process(String requestId) {
        setContext(requestId);  // Thread-safe: ThreadLocal
        // ... process ...
    }
}
```

### ⚠️ Inheritance + Synchronization Pitfall
```java
class Base {
    synchronized void method() { /* ... */ }
}

class Derived extends Base {
    @Override
    synchronized void method() {
        super.method(); // Acquires lock on `this` TWICE — reentrant OK in Java
        // But if locks are different objects, deadlock possible!
    }
}
```

---

## 19. Company Interview Perspective

### Google
- "Design a class hierarchy for [search, maps, YouTube]" — tests IS-A correctness
- Asks about vtable layout and virtual dispatch cost in performance-critical code
- LSP is frequently tested: "Can this substitution break anything?"

### Goldman Sachs
- Instrument hierarchy: Stock, Bond, Option, Future — when to use abstract base
- Event sourcing hierarchies: base `Event` with derived trade events
- Template Method for batch processing pipelines

### Microsoft
- C++ COM-style design: pure virtual interfaces, inheritance for implementation
- .NET: value types (struct) vs reference types (class) — when each is appropriate
- WPF/XAML: DependencyObject inheritance tree

### Amazon
- Leadership: "Design the class hierarchy for a warehouse inventory system"
- Service layer: abstract `BaseService` with common AWS SDK calls
- ORM entities: `AuditableEntity` base with createdAt/updatedAt

### Meta
- React component hierarchy (analogy to class hierarchy)
- Python metaclass and MRO for complex ORM relationships
- C++ multiple inheritance in game engine components (Meta Reality Labs)

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Can you instantiate an abstract class? | No — but you can have a reference of abstract type |
| 2 | ⚠️ What happens if you don't call `super()` in Java? | Compiler auto-inserts `super()` if parent has no-arg constructor; error if it doesn't |
| 3 | Can a final class be extended? | No — `final` prevents subclassing (String, Integer are final) |
| 4 | Can a constructor be inherited? | No — constructors are not inherited; `super()` must be called explicitly |
| 5 | ⚠️ Is method hiding the same as method overriding? | No — hiding = static compile-time; overriding = dynamic runtime dispatch |
| 6 | What is covariant return type? | Overriding method can return subtype of declared return type |
| 7 | Can an interface extend another interface? | Yes — interfaces can extend multiple interfaces |
| 8 | ⚠️ What is the fragile base class problem? | Adding a method to base breaks subclass if it has a method with the same signature |
| 9 | C++ private vs protected vs public inheritance? | Controls how inherited members appear in derived's public interface |
| 10 | What is protected inheritance in C++? | Base's public becomes protected in derived — restricts further inheritance |
| 11 | ⚠️ Can you have a virtual constructor? | No — constructors create objects, type not yet determined |
| 12 | What does `sealed` class do (Java 17+)? | Restricts which classes can extend — explicitly permits subclasses |
| 13 | ⚠️ Virtual destructor is not needed if...? | If class is not used polymorphically (no base ptr to derived) — but always safest |
| 14 | What is the Curiously Recurring Template Pattern? | `class D : public B<D>` — enables static polymorphism |
| 15 | Can you override `private` methods? | No — private methods are not visible to subclass, so not overridable |
| 16 | What is a mixin? | Class providing functionality to be inherited without being a primary base type |
| 17 | ⚠️ `super()` must be the first statement in constructor — why? | Base object must be initialized before derived can use inherited state |
| 18 | What is CRTP used for? | Policy-based design, static polymorphism, zero-cost abstraction |
| 19 | What is abstract class with all concrete methods? | Technically valid — prevents instantiation but doesn't force overriding |
| 20 | ⚠️ What does `protected` in Java actually mean? | Accessible within package + accessible in subclasses (even different package) |

---

## 21. Coding Problems

### Easy — Print Animal Hierarchy
```java
// Implement speak() for Animal, Dog, Cat and demonstrate polymorphism
Animal[] animals = {new Dog(), new Cat(), new Dog()};
for (Animal a : animals) a.speak(); // Polymorphic dispatch
```

### Medium — Design a Shape Hierarchy with Area and Perimeter
```java
abstract class Shape {
    abstract double area();
    abstract double perimeter();
    boolean isLargerThan(Shape other) { return this.area() > other.area(); }
}
// Implement Circle, Rectangle, Triangle extending Shape
// Sort shapes by area: Arrays.sort(shapes, Comparator.comparingDouble(Shape::area));
```

### Hard — Implement Template Method for Report Generation (LeetCode pattern)
```java
abstract class ReportGenerator {
    public final String generate() { // Template method
        return header() + "\n" + body() + "\n" + footer();
    }
    protected abstract String header();
    protected abstract String body();
    protected String footer() { return "--- End of Report ---"; }
}

class HTMLReport extends ReportGenerator {
    protected String header() { return "<html><body>"; }
    protected String body() { return "<p>Report Data</p>"; }
    protected String footer() { return "</body></html>"; }
}
```

### Interview — Detect LSP Violation
> "Given an inheritance hierarchy, write a test that would fail if LSP is violated."
```java
// LSP test for shapes
void assertLSP(Rectangle r) {
    r.setWidth(5);
    r.setHeight(4);
    assertEquals(20, r.area(), "LSP violated: area should be 20");
}
// If Square extends Rectangle, this test fails for Square
```

---

## 22. Revision Sheet

| Concept | Key Point |
|---------|-----------|
| IS-A | Inheritance — Dog IS-A Animal |
| HAS-A | Composition — Car HAS-A Engine |
| vtable | Per-class table of virtual method addresses |
| vptr | Per-object hidden pointer to vtable |
| Virtual destructor | Required in C++ base classes for safe deletion |
| LSP | Derived must be substitutable for base |
| Constructor chain | super() called before child initialization |
| Diamond problem | Multiple paths to same base — use virtual inheritance |
| MRO | Python's C3 linearization for method lookup order |
| Fragile base class | Base changes break subclasses — prefer composition |
| Template Method | Base defines algorithm; subclasses fill steps |
| Final class | Cannot be extended (Java: String, Integer) |

### Common Pitfalls
- ⚠️ No virtual destructor in C++ base
- ⚠️ Calling virtual methods in constructor
- ⚠️ Breaking LSP in derived classes
- ⚠️ Inheriting when composition is correct (Stack extends Vector)
- ⚠️ Static method hiding confused with overriding
- ⚠️ Forgetting `@Override` annotation

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| IS-A vs HAS-A | IS-A = inheritance; HAS-A = composition |
| What is vtable? | Per-class array of virtual method function pointers |
| What is vptr? | Hidden per-object pointer to its class's vtable |
| Diamond problem | Ambiguity from multiple inheritance to same base |
| MRO in Python | C3 linearization: left-right depth-first, dedup |
| Why virtual destructor? | Ensures derived destructor called via base pointer |
| LSP | Derived objects must work wherever base is expected |
| Fragile base class | Base method change breaks subclasses |
| Template Method pattern | Base defines algo skeleton; subclasses fill steps |
| `super()` rule (Java) | Must be first statement in constructor |
| Can constructor be inherited? | No — must explicitly call `super()` |
| Method hiding vs overriding | Hiding = static/compile; Overriding = virtual/runtime |
| Private inheritance C++ | Base public/protected become private in derived |
| `final` class | Cannot be extended |
| `sealed` class (Java 17) | Only permitted subclasses can extend |
| CRTP | `class D : public B<D>` — static polymorphism |
| Covariant return | Override can return subtype of parent return type |
| Can interface extend interface? | Yes — and multiple interfaces |
| Abstract class instantiation? | No — but can have reference of abstract type |
| Multiple inheritance in Java? | Classes: No. Interfaces: Yes (multiple implements) |
| `protected` in Java | Within package + accessible in subclasses |
| Stack extends Vector problem | Exposes inappropriate methods — LSP violation |
| Virtual method in constructor? | C++: calls base version (derived not yet built) |
| Max recommended hierarchy depth | 2-3 levels; beyond that, prefer composition |
| What is mixin? | Class providing capability via inheritance, not primary IS-A |

---

## 24. Cheat Sheet

### Top 20 Facts
1. Inheritance = IS-A relationship; establishes type hierarchy
2. `extends` in Java = single class inheritance only
3. `implements` = multiple interface implementation
4. C++ supports multiple class inheritance; Java does not
5. Python uses C3 MRO for multiple inheritance resolution
6. All instance methods in Java are virtual by default
7. C++ requires explicit `virtual` keyword
8. vtable = per-class array of virtual function pointers
9. vptr = per-object hidden pointer to vtable (8 bytes)
10. Virtual destructor is mandatory in C++ polymorphic base classes
11. Constructor chaining: `super()` must be first in constructor
12. Constructors are NOT inherited
13. LSP: derived must be substitutable for base without breaking behavior
14. Template Method = canonical inheritance design pattern
15. Fragile base class = hidden cost of deep inheritance
16. Favor composition over inheritance (GoF + Effective Java)
17. `final` class cannot be subclassed
18. Method hiding (static) ≠ method overriding (virtual)
19. Abstract class cannot be instantiated
20. Diamond problem: solved by virtual inheritance (C++) or avoided by Java design

### Top 20 Keywords
`extends`, `super`, `override`, `abstract`, `virtual`, `vtable`, `vptr`, `IS-A`, `HAS-A`, `LSP`, `MRO`, `diamond`, `fragile base class`, `template method`, `polymorphism`, `covariant`, `sealed`, `CRTP`, `mixin`, `composition`

---

## 25. Final Interview Summary

### 5-Minute Revision
- Inheritance = IS-A; Composition = HAS-A (prefer composition)
- vtable: per-class function pointer array; vptr: per-object pointer to vtable
- Java: single class inheritance, multiple interface; C++: multiple class; Python: C3 MRO
- Virtual destructor: mandatory in C++ polymorphic base classes
- LSP: derived must be fully substitutable for base
- Constructor chain: super() called before derived init
- Template Method = canonical inheritance pattern

### 15-Minute Revision
Add:
- Diamond problem and virtual inheritance (C++)
- Fragile base class problem
- CRTP for static polymorphism
- Covariant return types
- Method hiding vs overriding
- `sealed` classes (Java 17)
- Abstract class vs interface decision matrix
- Inheritance in Spring: MappedSuperclass, AbstractApplicationContext
- MRO in Python: C3 algorithm

### Night-Before Interview Revision
1. ⭐ IS-A = inheritance; HAS-A = composition. Prefer composition.
2. ⭐ vtable + vptr = mechanism for runtime polymorphism
3. ⭐ Virtual destructor mandatory in C++ base classes
4. ⭐ LSP: derived substitutable for base
5. ⭐ super() must be first statement in constructor
6. ⭐ Template Method: base defines algorithm, derived fills steps
7. ⭐ Diamond problem: virtual inheritance (C++) / avoided (Java)
8. ⭐ Java: single class inheritance, multiple interface
9. ⭐ Python MRO: C3 linearization
10. ⭐ Favor composition when IS-A is not clear
