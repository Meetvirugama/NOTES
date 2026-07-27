# Methods

## Definition

A **Method** is a block of code defined inside a class that performs a specific task. It encapsulates the **behavior** of the objects instantiated from that class.

While the terms "function" and "method" are often used interchangeably, in strict OOP terminology, a *function* is an independent block of code (global), whereas a *method* is bound to an object or a class.

**Deep Dive:** At the compiler level, there is no physical difference between a method and a global function. A method is simply a global function that the compiler implicitly modifies to accept a hidden first parameter: the pointer to the object instance (the `this` pointer).

## Why It Is Needed

Methods provide modularity and reusability, adhering to the **DRY (Don't Repeat Yourself)** principle. They also enforce **Encapsulation**. Instead of external code directly manipulating an object's state (which could lead to invalid states), methods act as controlled gateways that validate operations before mutating the data.

## Key Concepts

- **Return Type:** What the method sends back to the caller (`void` if nothing).
- **Parameters / Arguments:** Data passed into the method.
- **Instance Methods:** Require an object instance to be invoked. They operate on specific object states.
- **Static Methods:** Belong to the class template itself. They cannot access instance variables and do not receive a `this` pointer.
- **`this` Pointer:** A hidden pointer passed to instance methods that holds the memory address of the calling object.

## How It Works Internally (The `this` Pointer)

When you write:
```java
obj.calculateSum(10, 20);
```
The compiler internally translates this to procedural C-style code:
```c
calculateSum(&obj, 10, 20);
```
The method receives `&obj` via a hidden parameter called `this`. That is how a single method sitting in the Code Segment of memory knows exactly which object's heap memory to mutate!

## Syntax & Code Example

### Java: Line-by-Line Execution Flow

```java
class Calculator {
    int total = 0; // Instance state

    // 1. Instance Method
    public void add(int amount) {
        // 'this' is implicitly used here: this.total += amount
        total += amount; 
    }

    // 2. Static Method
    public static void printInfo() {
        System.out.println("I am a Calculator class.");
        // total = 10; // ERROR! Cannot access instance variable from static context
    }
}

public class Main {
    public static void main(String[] args) {
        Calculator.printInfo(); // 3. Called via Class Name

        Calculator calc = new Calculator(); // 4. Instantiation
        calc.add(50); // 5. Called via Object Reference
    }
}
```

### Line-by-Line Breakdown:
1. `public void add...`: An instance method. It modifies the state (`total`) of the specific object that calls it.
2. `public static void printInfo...`: A class-level method. It resides in memory even if zero `Calculator` objects exist. It cannot touch `total` because it doesn't know *which* calculator's total to modify.
3. `Calculator.printInfo()`: Static methods are invoked using the Class Name, saving memory and processing time as no object instantiation is required.
4. `calc.add(50)`: The JVM passes the memory address of `calc` to the `add` method so it knows to update `calc`'s specific `total` variable.

## Edge Cases

### Edge Case 1: Method Hiding (Static Method Overriding)
- **What happens:** In Java, if a parent class has a `static` method, and a child class writes a `static` method with the exact same signature, it is NOT overriding. It is called **Method Hiding**.
- **Internal Behavior:** Static methods are resolved at **compile-time** (Static Binding) based on the *Reference Type*, not the *Object Type*.
- **Expected Output:**
  ```java
  class Parent { static void show() { print("Parent"); } }
  class Child extends Parent { static void show() { print("Child"); } }
  
  Parent p = new Child();
  p.show(); // Outputs: "Parent" (Resolved at compile-time!)
  ```
- **Best Practice:** Never call static methods using an object reference (`p.show()`). Always use the class name (`Parent.show()`) to avoid confusion.

### Edge Case 2: Covariant Return Types
- **What happens:** When overriding an instance method, the return type must match. However, you are allowed to return a *subclass* of the original return type.
- **Example:** If Parent returns `Animal`, Child can override it and return `Dog`. This is perfectly legal and heavily used in the **Factory Design Pattern**.

## Tricky FAANG Interview Questions

### Question 1: Memory Footprint of Methods
**Q:** *If I create 1 million `Calculator` objects, how many copies of the `add()` method are created in RAM?*
**Answer & Explanation:** Only **one**. Methods are instructions, not data. They are stored once in the **Text/Code Segment** of RAM. All 1 million objects share that exact same block of instructions. The only thing duplicated 1 million times is the instance variables (`total`) on the Heap.

### Question 2: Static Context Constraints
**Q:** *Why is the `main` method in Java declared as `static`?*
**Answer & Explanation:** The JVM needs an entry point to start the application *before* any objects exist. If `main` were an instance method, the JVM would have to guess how to instantiate your class (what constructor to use, what arguments to pass) just to run it. By making it `static`, the JVM can invoke it directly from the class blueprint immediately upon loading.

### Question 3: The Virtual Dispatch Cost (C++)
**Q:** *What is the performance difference between calling a regular instance method and a `virtual` method in C++?*
**Answer & Explanation:** A regular instance method call is resolved at compile-time (Static Binding). The compiler knows the exact memory address of the function and hardcodes a `JMP` instruction. 
A `virtual` method call is resolved at runtime (Dynamic Binding). The CPU must:
1. Fetch the object's `vptr` (Virtual Pointer).
2. Follow it to the class's `v-table` (Virtual Table).
3. Look up the function address for that specific class.
4. Execute the jump.
This extra pointer indirection causes a slight overhead and can break CPU instruction caching/pipelining.

## OA Tips

- **Helper Methods:** If a helper method doesn't use any instance variables, mark it `static`. It signals intent to the reader (and the compiler) that this is a pure function with no side effects on object state.
- **Method Length:** In interviews, keep methods short. If your `solve()` method in an OA exceeds 40 lines, extract chunks of logic into beautifully named private helper methods (e.g., `private boolean isValidNode(...)`).

## 2-Minute Revision

- **Method vs Function:** Methods are functions bound to a class.
- **Instance Methods:** Operate on object state, receive a hidden `this` pointer.
- **Static Methods:** Operate at the class level, no `this` pointer, resolved at compile-time.
- **Memory Layout:** Methods are stored once in the Code segment, not duplicated per object.
- **Method Hiding:** Static methods cannot be overridden, only hidden based on reference type.


---
## 🚀 50 LPA Senior Engineer Deep Dive: CPU Branch Prediction & V-Tables

At the Staff Engineer level, you must understand how calling a method translates into CPU instructions, and why certain methods are catastrophically slow in hot paths.

### Static Dispatch vs Dynamic Dispatch
When you call a normal instance method (`obj.compute()`), the compiler hardcodes the exact memory address of `compute()` into the assembly code (`CALL 0x4A3B2`). This is **Static Dispatch**. The CPU's Branch Predictor can easily pre-fetch the instructions, running them at lightning speed.

When you call a `virtual` method (Run-Time Polymorphism), the compiler cannot hardcode the address. It must use **Dynamic Dispatch**:
1. Fetch the object's hidden `vptr`.
2. Access the `v-table` array.
3. Fetch the function pointer.
4. Jump to the function.

### Branch Prediction Failures
The extra memory lookups are bad, but the real killer is **Branch Prediction Failure**. Modern CPUs execute instructions speculatively (ahead of time). If you loop over an array of polymorphic `Animal*` objects and call `makeSound()`, the CPU doesn't know if the next object is a `Dog` or a `Cat` until the very last nanosecond. It guesses wrong, flushes its instruction pipeline, and halts for 15-20 cycles.

### 50 LPA FAANG Questions
**Q:** *How do you eliminate the cost of virtual method calls in performance-critical C++ code without losing polymorphism?*
**A:** You use the **Curiously Recurring Template Pattern (CRTP)**. It achieves Static Polymorphism. By passing the Derived class as a template parameter to the Base class, the compiler resolves the method addresses at compile-time, completely eliminating the v-table and allowing the compiler to inline the methods.
```cpp
#include <iostream>

template <typename Derived>
class Base {
public:
    void compute() { 
        // Cast 'this' pointer to the Derived class at compile-time!
        static_cast<Derived*>(this)->computeImpl(); 
    }
};

class FastWorker : public Base<FastWorker> {
public:
    void computeImpl() { std::cout << "Fast static dispatch!\n"; }
};

int main() {
    FastWorker worker;
    // The compiler knows EXACTLY which function to call. 
    // No V-Table lookup required. 100% Branch Prediction accuracy.
    worker.compute(); 
    return 0;
}
```
