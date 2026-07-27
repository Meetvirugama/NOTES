# Polymorphism

## Definition

**Polymorphism** ("many forms") is the ability of a single interface or base reference to invoke different underlying implementations depending on the context or the exact object type at runtime. 

**Deep Dive:** Polymorphism decouples the *what* from the *how*. The caller only knows *what* operation to perform (e.g., `vehicle.drive()`), while the specific object decides *how* to execute it. This is the cornerstone of the Open/Closed Principle (software entities should be open for extension, but closed for modification).

## Why It Is Needed

Without polymorphism, code is littered with rigid `if-else` or `switch` statements checking object types:
```java
if (animal type is Dog) { bark(); }
else if (animal type is Cat) { meow(); }
```
Polymorphism eliminates this. You simply call `animal.makeSound()`, and the runtime engine automatically routes the call to the correct method. You can add a `Cow` class later without ever modifying the calling code.

## Types of Polymorphism

### 1. Compile-Time (Static Binding)
- **Method Overloading:** Same method name, different parameter signatures (types, number, order) in the same class.
- **How it works:** The compiler looks at the arguments passed during the method call, matches them against the signatures, and hardcodes the jump to the correct function in memory. *No runtime overhead.*

### 2. Run-Time (Dynamic Binding)
- **Method Overriding:** A subclass redefines a method present in its parent class with the exact same signature.
- **How it works:** Resolved at runtime by the Virtual Machine (JVM) or via a Virtual Table (v-table) in C++. 

## How It Works Internally: The V-Table (C++)

This is the most critical concept for senior interviews.
When a C++ class contains a `virtual` function, the compiler does two things:
1. **v-table Creation:** It creates a static array of function pointers (the v-table) for the class.
2. **v-ptr Insertion:** It secretly adds a pointer (`v-ptr`) to every object of that class. This `v-ptr` points to the class's v-table.

At runtime, when `animal->sound()` is called:
- The CPU dereferences `animal` to find its `v-ptr`.
- It follows the `v-ptr` to the correct `v-table` (e.g., `Dog`'s v-table).
- It looks up the address of `sound()` in that table.
- It executes the jump.

```mermaid
flowchart LR
    A[Animal Pointer] -->|Points to| B(Dog Object in Heap)
    B -->|Contains v-ptr| C[Dog V-Table]
    C -->|Function Pointer| D[Dog::sound() Implementation]
```

## Syntax & Practical Example

### Java: Run-Time Polymorphism (Overriding)

```java
class PaymentProcessor {
    // Base method
    public void processPayment(double amount) {
        System.out.println("Processing generic payment of $" + amount);
    }
}

class StripeProcessor extends PaymentProcessor {
    @Override // Compiler validation
    public void processPayment(double amount) {
        System.out.println("Routing $" + amount + " through Stripe API");
    }
}

class PayPalProcessor extends PaymentProcessor {
    @Override
    public void processPayment(double amount) {
        System.out.println("Routing $" + amount + " through PayPal API");
    }
}

public class Main {
    public static void main(String[] args) {
        // 1. Upcasting: Parent reference holds Child object
        PaymentProcessor p1 = new StripeProcessor();
        PaymentProcessor p2 = new PayPalProcessor();
        
        // 2. Dynamic Dispatch: JVM routes to the correct overridden method
        p1.processPayment(100.0); // Output: Routing through Stripe API
        p2.processPayment(50.0);  // Output: Routing through PayPal API
    }
}
```

### Line-by-Line Breakdown:
1. `PaymentProcessor p1 = new StripeProcessor();`: This is **Upcasting**. A broad reference holds a specific object. The compiler only allows methods defined in `PaymentProcessor` to be called.
2. `p1.processPayment(...)`: At compile-time, the compiler verifies `processPayment` exists in `PaymentProcessor`. At runtime, the JVM notices `p1` actually points to a `StripeProcessor` object and dynamically dispatches the call to the overridden method.

## Edge Cases

### Edge Case 1: Overriding Default Arguments (C++)
- **What happens:** In C++, default arguments are resolved at **compile-time** based on the reference type, while virtual functions are resolved at **runtime** based on the object type.
- **Expected Behavior:** If `Base::func(int x = 10)` is overridden by `Derived::func(int x = 20)`, calling `basePtr->func()` on a `Derived` object will execute the `Derived` logic but pass `10` as the argument!
- **Best Practice:** Never redefine default arguments in overridden virtual functions. It creates a horrific mismatch of static and dynamic binding.

### Edge Case 2: Covariant Return Types
- **What happens:** Normally, an overridden method must have the exact same return type. However, Java and C++ support Covariant Return Types, allowing the overridden method to return a narrower (subclass) type.
- **Example:** `Base::clone()` returns `Base*`. `Derived::clone()` can legally return `Derived*`.

## Tricky FAANG Interview Questions

### Question 1: Static Method Hiding
**Q:** *Can you override a `static` method in Java? What happens if a child class provides a static method with the exact same signature as the parent?*
**Answer & Explanation:** You **cannot** override static methods. Polymorphism depends on objects (runtime), but static methods belong to classes (compile-time). If a child class provides the same static method, it is called **Method Hiding**. The method executed depends entirely on the reference type declared at compile-time, ignoring the actual object created.

### Question 2: Downcasting Risks
**Q:** *What is downcasting, and why is it dangerous?*
**Answer & Explanation:** Upcasting (`Animal a = new Dog()`) is implicit and always safe. Downcasting (`Dog d = (Dog) a;`) is explicit and dangerous. If `a` is actually pointing to a `Cat` object, the JVM will throw a `ClassCastException` at runtime because a Cat cannot be forced into a Dog reference.
**Prevention:** Always use the `instanceof` operator in Java or `dynamic_cast` in C++ before attempting a downcast.

### Question 3: The Virtual Destructor Rule (C++)
**Q:** *Why must base class destructors be `virtual` in C++?*
**Answer & Explanation:** If you do `Base* b = new Derived();` and then `delete b;`, the compiler uses static binding to call `~Base()`. The `~Derived()` destructor is never called, meaning any heap memory allocated by the child class is leaked! Making the base destructor `virtual` forces the compiler to use the v-table, ensuring `~Derived()` executes first, followed by `~Base()`.

## OA Tips

- In OAs requiring parsing or multiple handlers (e.g., parsing XML vs JSON), use polymorphism. Create an interface `Parser`, implement `XmlParser` and `JsonParser`, and use a Factory to return the correct polymorphic reference. This keeps your main algorithm incredibly clean.

## 2-Minute Revision

- **Polymorphism:** Decouples interface from implementation.
- **Overloading (Static Binding):** Same name, different parameters. Resolved at compile-time. Fast.
- **Overriding (Dynamic Binding):** Same signature in subclass. Resolved at runtime. Requires indirection (v-tables).
- **Upcasting:** Parent reference holding a child object. Required for dynamic polymorphism.
- **C++ Specifics:** Requires `virtual` keyword. Always make base destructors `virtual` to prevent memory leaks.


---
## 🚀 50 LPA Senior Engineer Deep Dive: ABI & Virtual Table Internals

Polymorphism is not magic; it is implemented via hidden data structures that have profound implications for **ABI (Application Binary Interface)** compatibility in compiled libraries.

### ABI Breakage via V-Tables
Imagine you distribute a compiled `.so` or `.dll` library containing this class:
```cpp
class Parser {
public:
    virtual void parseXML();
};
```
The compiler generates a v-table with one entry at `Offset 0`.
A year later, you update the library to add JSON support:
```cpp
class Parser {
public:
    virtual void parseJSON();
    virtual void parseXML();
};
```
You compile the library and ship it. **Every client application that uses your library instantly crashes with a Segmentation Fault.**

**Why?** In the new v-table, `parseJSON` took `Offset 0`, pushing `parseXML` to `Offset 1`. The client's older compiled code is still looking at `Offset 0` to call XML, but now it executes JSON parsing logic (or garbage memory) because the v-table layout shifted. This is an **ABI Break**.
At FAANG, breaking ABI in core libraries takes down entire server fleets. The rule: *Never add virtual functions to the top/middle of an exported class. Always append them to the end.*

### 50 LPA FAANG Questions
**Q:** *How does Java achieve polymorphism differently than C++ under the hood?*
**A:** While C++ uses statically built v-tables per class, Java uses the JVM. The JVM maintains an **ITable** (Interface Table) and a **VTable** for classes. Because Java loads classes dynamically at runtime, it resolves the method offsets on the fly. Furthermore, the JVM's JIT (Just-In-Time) compiler performs **Monomorphic Inline Caching**. If the JVM notices that `animal.sound()` is actually a `Dog` 99% of the time, it will physically rewrite the assembly code at runtime to bypass the v-table entirely and directly inline the `Dog` code, falling back to a slow path only if a `Cat` appears. This makes Java's polymorphism often *faster* than C++'s static v-tables in long-running server processes.
