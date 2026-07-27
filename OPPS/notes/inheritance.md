# Inheritance

## Definition

**Inheritance** is the mechanism by which a new class (Subclass/Derived) is derived from an existing class (Superclass/Base). It establishes an **IS-A** relationship (e.g., A Dog *is an* Animal). 

**Deep Dive:** In memory, inheritance is essentially structural concatenation. When a subclass is instantiated, the compiler allocates a single, contiguous block of memory that contains the base class's variables immediately followed by the subclass's variables. The subclass literally *contains* the physical memory layout of the base class.

## Why It Is Needed

Without inheritance, code duplication explodes. If `Car` and `Bike` share `speed`, `color`, and `start()`, rewriting them in both classes violates the **DRY (Don't Repeat Yourself)** principle. Inheritance allows you to abstract shared logic into a `Vehicle` class, ensuring that a bug fix in `Vehicle::start()` automatically propagates to both `Car` and `Bike`.

## Key Concepts

- **Subclass / Derived Class:** The class inheriting properties.
- **Superclass / Base Class:** The class being inherited from.
- **`super` (Java) / Base Initialization (C++):** Keywords used to invoke the parent class's constructor or methods.
- **Overriding:** Providing a new implementation for an inherited method.
- **Access Specifiers:** `protected` members are accessible inside subclasses but act as `private` to the outside world.

## How It Works Internally (Constructor Chaining)

When you instantiate a Subclass, the Superclass MUST be constructed first. 
1. `new Dog()` is called.
2. The `Dog` constructor implicitly (or explicitly) calls `super()`.
3. The `Animal` constructor executes, initializing the Animal memory segment.
4. The `Dog` constructor executes, initializing the Dog memory segment.

```mermaid
flowchart TD
    A[new Dog() Invoked] --> B[Implicit call to Animal() Constructor]
    B --> C[Animal State Initialized]
    C --> D[Dog State Initialized]
    
    subgraph Memory Layout
    C --- D
    end
```

## Syntax & Practical Example

Let's look at a Java example to see how `super` and `protected` work together.

### Java: Line-by-Line Execution Flow

```java
// Base Class
class Employee {
    protected int id; // Protected: Accessible to subclasses
    
    // Parent Constructor
    public Employee(int id) {
        this.id = id;
        System.out.println("Employee created with ID: " + this.id);
    }
}

// Derived Class
class Manager extends Employee {
    private int teamSize;
    
    // Child Constructor
    public Manager(int id, int teamSize) {
        super(id); // MUST be the very first line! Calls Employee(id).
        this.teamSize = teamSize;
        System.out.println("Manager created with Team Size: " + this.teamSize);
    }
}

public class Main {
    public static void main(String[] args) {
        Manager m = new Manager(101, 5);
    }
}
```

### Line-by-Line Breakdown:
1. `protected int id;`: Encapsulated from the public, but available to `Manager`.
2. `public Manager(int id, int teamSize)`: The child constructor takes arguments for both itself and its parent.
3. `super(id);`: The compiler forces you to initialize the parent state. If you don't write this, the compiler inserts `super()`, which would fail here because `Employee` lacks a default (no-arg) constructor!
4. The output will always print "Employee created..." before "Manager created...", proving constructor chaining.

## Edge Cases

### Edge Case 1: The Diamond Problem (Multiple Inheritance)
- **What happens:** If `Class B` and `Class C` both inherit from `Class A`, and `Class D` inherits from *both* `B` and `C`.
- **The Issue:** `D` inherits TWO physical copies of `A`'s variables. If `D` tries to access an inherited variable `A.x`, the compiler throws an ambiguity error (Which `x`? The one from `B`'s copy or `C`'s copy?).
- **C++ Fix:** **Virtual Inheritance** (`class B : virtual public A`). This tells the compiler to only lay out ONE instance of `A` in memory, shared by `B` and `C`.
- **Java Fix:** Java completely bans multiple inheritance of classes. You can only implement multiple Interfaces, which carry no state, thus sidestepping the memory layout issue entirely.

### Edge Case 2: Object Slicing (C++ Specific)
- **What happens:** 
  ```cpp
  Dog d;
  Animal a = d; // Passed by value!
  ```
- **Internal Behavior:** The C++ compiler copies the `Animal` portion of `Dog`'s memory into `a`. The `Dog`-specific memory is "sliced" off and lost forever. 
- **Prevention:** Always pass derived objects to base classes via pointers or references (`Animal& a = d;`).

## Tricky FAANG Interview Questions

### Question 1: Inheritance vs Composition
**Q:** *Why do modern software architectures (like Go or Rust) favor Composition over Inheritance, and when should you strictly use Inheritance?*
**Answer & Explanation:** Inheritance creates the tightest form of coupling in OOP. Changes in the base class heavily ripple down to subclasses (The Fragile Base Class Problem). Deep inheritance trees (e.g., `Mammal -> Dog -> Poodle`) become rigid. 
Composition (HAS-A) injects dependencies dynamically at runtime, making testing and swapping implementations vastly easier. 
*Rule of thumb:* Only use Inheritance when a strict, undeniable IS-A relationship exists AND you need polymorphic behavior (e.g., an array of `Shapes` calling `draw()`). Otherwise, use Composition.
**Why it's asked:** This separates junior coders (who abuse inheritance to share code) from senior architects (who use composition to build flexible systems).

### Question 2: Hiding vs Overriding
**Q:** *In Java, what happens if a parent has a variable `int x = 10` and the child has a variable `int x = 20`? Does the child override the variable?*
**Answer & Explanation:** No. **Variables cannot be overridden in Java; they are only hidden.** 
If you do `Parent p = new Child(); print(p.x);`, it prints `10`. Polymorphism (dynamic binding) applies *only* to instance methods, not variables. Variables are resolved at compile-time (Static Binding) based on the reference type.

## OA Tips

- **OOD Questions:** If asked to design a `ParkingLot` or `VendingMachine`, do NOT create deep inheritance trees (e.g., `Vehicle -> FourWheeler -> Car -> ElectricCar`). Keep it shallow. Use interfaces (`Chargeable`) and composition (`Engine`) to handle variations. 

## 2-Minute Revision

- **Inheritance (IS-A):** Subclass derives from Superclass.
- **Memory:** Subclass memory includes the superclass memory block.
- **Constructors:** Not inherited. Parent constructor executes *before* child constructor.
- **Diamond Problem:** Ambiguity in multiple inheritance. Solved by `virtual` inheritance (C++) or Interfaces (Java).
- **Golden Rule:** Favor Composition (HAS-A) over Inheritance (IS-A) to reduce tight coupling.


---
## 🚀 50 LPA Senior Engineer Deep Dive: Fragile Base Class & Thunks

At scale (monorepos with 10M+ lines of code), inheritance becomes a dangerous architectural liability known as the **Fragile Base Class Problem**.

### The Fragile Base Class
If Class `A` is inherited by 1,000 subclasses across a company's codebase, making a seemingly innocent change to `A` (like adding a virtual method, changing a protected variable's semantics, or altering the constructor order) can silently break hundreds of downstream systems. The tight coupling of inheritance means the base class cannot evolve safely. This is why Google and Facebook heavily enforce **Composition over Inheritance**.

### Multiple Inheritance Internals: Thunks
Java bans multiple inheritance to avoid the Diamond Problem. C++ allows it, but it requires brutal compiler gymnastics.
If `Class C` inherits from `Class A` and `Class B`:
```cpp
C* obj = new C();
B* bPtr = obj; // Upcasting
```
In memory, `C` contains the variables of `A`, then the variables of `B`. 
When you cast `C*` to `B*`, the compiler literally adds an offset (e.g., +8 bytes) to the memory address so the pointer correctly points to the `B` slice of the object!

If you call a virtual method on `bPtr` that was overridden by `C`, the `this` pointer passed to `C`'s method is currently pointing at the `B` offset. If it executes with that offset, memory corruption occurs. 
The compiler generates a invisible function called a **Thunk**. The v-table points to the Thunk, which subtracts the 8 bytes to fix the `this` pointer, and *then* jumps to the actual `C` method.

### 50 LPA FAANG Questions
**Q:** *Explain the memory layout of Virtual Inheritance in C++ and how it solves the Diamond Problem.*
**A:** Without `virtual` inheritance, a class `D` inheriting from `B` and `C` (which both inherit from `A`) will contain two physical copies of `A` in memory.
With `virtual` inheritance, the compiler extracts `A` and places it at the very end of `D`'s memory layout. `B` and `C` are given hidden pointers (**vbase_pointers**) that point to the shared `A` memory block. When `B` or `C` try to access an `A` variable, they must follow the pointer. This solves the ambiguity but adds a severe runtime pointer-indirection penalty.

```cpp
#include <iostream>

class A { int x; };

// Standard Inheritance (Diamond Problem -> 2 copies of A)
class B : public A {};
class C : public A {};
class D : public B, public C {};

// Virtual Inheritance (Solves Diamond Problem -> 1 copy of A)
class B_Virt : virtual public A {};
class C_Virt : virtual public A {};
class D_Virt : public B_Virt, public C_Virt {};

int main() {
    std::cout << "Standard D size: " << sizeof(D) << " bytes\n"; 
    // Usually 8 bytes (Two 4-byte ints)
    
    std::cout << "Virtual D size: " << sizeof(D_Virt) << " bytes\n"; 
    // Usually 24 bytes! (One 4-byte int + padding + TWO 8-byte vbase_pointers)
    
    // Virtual inheritance bloated the memory by 300% to fix the ambiguity!
    return 0;
}
```
