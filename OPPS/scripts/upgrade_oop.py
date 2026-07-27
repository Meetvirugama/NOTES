import os

# Content to append to oop-overview.md
oop_overview_append = """

---
## 🚀 50 LPA Senior Engineer Deep Dive: CPU Cache & DOD

At the Staff/Senior level (L5/L6), interviewers expect you to understand the hardware implications of Object-Oriented Programming, particularly in latency-sensitive environments like High-Frequency Trading (HFT) and Game Engines.

### The True Cost of OOP: Cache Misses
Modern CPUs are incredibly fast, but RAM is extremely slow. To bridge this gap, CPUs load data into L1/L2/L3 caches in chunks called **Cache Lines** (typically 64 bytes).

In traditional OOP, you often have an **Array of Objects** (Struct of Arrays / SoA vs Array of Structs / AoS).
```cpp
class Particle {
    double x, y, z;      // 24 bytes
    double velocity;     // 8 bytes
    bool isActive;       // 1 byte (padded to 8)
    // ... total 64 bytes
};
std::vector<Particle*> particles;
```
If you loop through `particles` to update just the `velocity`, the CPU must load the entire 64-byte `Particle` object into the L1 cache, update 8 bytes, and throw the rest away. Because the objects are allocated dynamically on the heap (pointers), they are scattered across RAM. Every pointer dereference causes an **L1 Cache Miss**, forcing the CPU to wait hundreds of cycles to fetch data from RAM.

### Data-Oriented Design (DOD) as the Solution
In HFT and AAA game engines, traditional OOP is abandoned in favor of DOD (Struct of Arrays).
```cpp
struct Particles {
    std::vector<double> x, y, z;
    std::vector<double> velocity;
    std::vector<bool> isActive;
};
```
Now, if you update velocity, you loop over a contiguous array of `double`s. The CPU loads 8 velocities into a single 64-byte cache line simultaneously. You achieve **100% cache utilization**, running orders of magnitude faster than the OOP approach.

### 50 LPA FAANG Questions
**Q:** *Why do trading systems at Optiver or Jane Street avoid traditional OOP design patterns?*
**A:** Because OOP relies heavily on heap allocation and polymorphism (v-tables). Heap allocation scatters data, ruining CPU cache spatial locality. Virtual functions ruin branch prediction and instruction caching. HFT requires deterministic latency, achieved by contiguous stack allocation and data-oriented design to maximize L1 cache hits.
"""

# Content to append to classes-and-objects.md
classes_objects_append = """

---
## 🚀 50 LPA Senior Engineer Deep Dive: Memory Alignment & Padding

When you create a class, the OS and compiler do not just pack variables together. They align them in memory to match the CPU's word boundaries (usually 4 or 8 bytes) to minimize CPU fetch cycles.

### The Padding Problem
Look at this C++ class:
```cpp
class Order {
    bool isBuy;      // 1 byte
    double price;    // 8 bytes
    int quantity;    // 4 bytes
};
```
You would expect `sizeof(Order)` to be `1 + 8 + 4 = 13 bytes`. 
Instead, the compiler outputs **24 bytes**.

**Why?** The CPU fetches memory in 8-byte chunks on a 64-bit system. 
1. `isBuy` takes 1 byte. The compiler adds **7 bytes of invisible padding** so the next variable (`price`) starts exactly at an 8-byte boundary. 
2. `price` takes 8 bytes.
3. `quantity` takes 4 bytes. The compiler adds **4 bytes of padding** at the end so the total object size is a multiple of 8 (24 bytes).

### The 50 LPA Solution
By simply reordering the variables from largest to smallest, you eliminate padding:
```cpp
class OptimizedOrder {
    double price;    // 8 bytes
    int quantity;    // 4 bytes
    bool isBuy;      // 1 byte
                     // 3 bytes of padding at the end
};
```
Now `sizeof(OptimizedOrder)` is **16 bytes**. 
If your application stores 100 Million orders in an in-memory cache, **you just saved 800 Megabytes of RAM** simply by reordering class variables!

### 50 LPA FAANG Questions
**Q:** *Explain how class field ordering affects Garbage Collection in Java.*
**A:** While Java developers cannot control exact memory layouts like C++ developers (the JVM optimizes field packing automatically), understanding object headers (Mark Word and Klass Pointer, taking 12-16 bytes) is critical. Creating millions of tiny objects (like `new Integer(5)`) causes immense GC pressure and memory bloat compared to using primitive arrays, because the object headers take up more RAM than the data itself.
"""

# Content to append to methods.md
methods_append = """

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
template <typename Derived>
class Base {
public:
    void compute() { static_cast<Derived*>(this)->computeImpl(); }
};
```
"""

# Content to append to constructors.md
constructors_append = """

---
## 🚀 50 LPA Senior Engineer Deep Dive: Thread-Safe Construction & Singletons

In distributed systems and highly concurrent backend services (like those at Amazon or Google), object construction can lead to severe race conditions.

### The Broken Double-Checked Locking Singleton
A classic interview question is to implement a Singleton (a class with a private constructor ensuring only one instance exists).
Juniors write this:
```java
public class Singleton {
    private static Singleton instance;
    private Singleton() {}
    
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton(); // BUG!
                }
            }
        }
        return instance;
    }
}
```
**Why this fails at 50 LPA:** The line `instance = new Singleton();` is NOT atomic. The JVM translates it to:
1. Allocate memory for Singleton.
2. Assign the memory reference to `instance`.
3. Call the constructor to initialize data.

Due to **Instruction Reordering** (Out-of-Order Execution) by the CPU or Compiler, Step 2 can happen before Step 3. 
Thread A allocates memory and sets `instance`. Thread B checks `if (instance == null)`, sees it's not null, and tries to use it. But Thread A hasn't run the constructor yet! Thread B crashes because it's interacting with an uninitialized object.

### The Solution
In Java, you must declare the variable as `volatile`:
`private static volatile Singleton instance;`
This inserts a **Memory Barrier** (fence) at the hardware level, preventing the CPU from reordering the initialization instructions.

Alternatively, use the **Initialization-on-Demand Holder Idiom** (using a static inner class), which relies on the JVM's class-loader guarantees to be 100% thread-safe without locks.

### 50 LPA FAANG Questions
**Q:** *What happens if an exception is thrown inside a C++ constructor? Does a memory leak occur?*
**A:** If a constructor throws, the object is considered "never fully constructed". Therefore, its **destructor is never called**. However, any fully constructed sub-objects (class members) *will* have their destructors called. If you dynamically allocated memory (`new int[100]`) before the exception was thrown, that memory is leaked. This is why you must use **Smart Pointers** (`std::unique_ptr`) inside classes, as their destructors will fire and clean up the heap even if the parent constructor aborts.
"""

# Content to append to inheritance.md
inheritance_append = """

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
"""

# Content to append to polymorphism.md
polymorphism_append = """

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
"""

# Content to append to encapsulation.md
encapsulation_append = """

---
## 🚀 50 LPA Senior Engineer Deep Dive: Monitors, Locks, & Encapsulation at Scale

In distributed and high-concurrency systems, Encapsulation is not just about hiding `age` or `balance`. It is about encapsulating **State Mutations and Locks** to prevent race conditions.

### The Monitor Pattern (Java Internals)
In Java, every single object (thanks to its Object Header) has an intrinsic lock (a **Monitor**).
If you want to encapsulate state safely in a multi-threaded environment, you can use the `synchronized` keyword:
```java
class BankAccount {
    private double balance; // Encapsulated state
    
    // Encapsulated lock (Monitor)
    public synchronized void deposit(double amount) {
        balance += amount;
    }
}
```
**How it works internally:** The JVM updates the **Mark Word** in the object's header memory to record the Thread ID that currently owns the lock. If another thread calls `deposit`, it sees the Mark Word is taken and parks itself at the OS level (using `futex` in Linux).

### The Danger of Encapsulating Locks
While the Monitor pattern is easy, it is dangerous at FAANG scale. Because the lock is intrinsic to the object, an external malicious or buggy thread can do this:
```java
BankAccount acc = new BankAccount();
synchronized(acc) {
    while(true) { Thread.sleep(1000); } // Locks the object forever!
}
```
Now, ANY other thread trying to call `acc.deposit()` will block permanently, causing a cascading system outage. 
**The 50 LPA Fix:** Never use intrinsic object locks for public APIs. Encapsulate a private, dedicated lock object inside the class.
```java
class SafeBankAccount {
    private double balance;
    private final Object lock = new Object(); // Hidden lock!
    
    public void deposit(double amount) {
        synchronized(lock) { balance += amount; }
    }
}
```

### 50 LPA FAANG Questions
**Q:** *Explain the concept of "Immutability" and why Functional Programming paradigms are invading OOP codebases.*
**A:** Immutability is the ultimate form of encapsulation. Instead of hiding data behind setters, you provide *no setters at all*. An object's state is set once in the constructor and never changes. If you need a change, you return a brand new object.
Why? Because immutable objects are **inherently thread-safe**. They require zero locks, zero monitors, and zero synchronization overhead. In massively parallel systems (like Apache Spark or trading engines), mutating state causes locking bottlenecks. Immutability completely bypasses this, which is why modern Java `Records` and C++ `const` correctness are so heavily emphasized.
"""

# Content to append to abstraction.md
abstraction_append = """

---
## 🚀 50 LPA Senior Engineer Deep Dive: API Contracts & Type Erasure

Abstraction at the Staff level is about defining contracts across Microservices, API boundaries, and Generic Programming.

### The Cost of Interfaces in C++ vs Java
In Java, an `interface` is a runtime abstraction. A class implements it, and the JVM resolves it dynamically using an **ITable**.
In C++, an interface is an Abstract Base Class with Pure Virtual Functions.
However, C++ also supports **Compile-Time Abstraction** via Templates (Concepts in C++20).
```cpp
template <typename T>
void process(T& object) {
    object.execute(); // T must have an execute() method.
}
```
Here, the abstraction is resolved completely at compile-time. No v-tables, no memory overhead. The compiler generates a unique version of the function for every type passed to it (Monomorphization). This results in blazing fast code but massive binary sizes (code bloat). 

### Java Generics: Type Erasure
Java chose a different path for generic abstractions. If you abstract a list `List<String>`, the JVM erases the `<String>` at runtime. Internally, it is just a `List` of `Object`s, and the compiler inserts hidden casts.
**The trade-off:** Java prevents binary bloat (only one List class exists in memory), but loses runtime type information (you cannot do `if (obj instanceof List<String>)`), forcing developers to pass `Class<T>` objects explicitly for reflection.

### 50 LPA FAANG Questions
**Q:** *You are designing a plugin architecture where third-party developers write C++ plugins (DLLs/SOs) for your core engine. Why must you use pure virtual abstract classes for your API instead of standard classes or templates?*
**A:** Because of the **Fragile Binary Interface (ABI) and Name Mangling**. C++ compilers (GCC, Clang, MSVC) mangle class and template names differently, meaning a plugin compiled with MSVC cannot link to an engine compiled with GCC. Furthermore, passing standard classes across DLL boundaries causes heap-corruption if the engine and plugin use different memory allocators (e.g., `std::string` memory freed by a different runtime library). 
By exposing a pure virtual interface (which is basically just a struct of function pointers at the assembly level) and a C-style `extern "C"` factory function to instantiate it, you create a perfect, stable, compiler-agnostic abstraction boundary that never breaks.
"""

# Mapping of file names to their respective append content
append_map = {
    "oop-overview.md": oop_overview_append,
    "classes-and-objects.md": classes_objects_append,
    "methods.md": methods_append,
    "constructors.md": constructors_append,
    "inheritance.md": inheritance_append,
    "polymorphism.md": polymorphism_append,
    "encapsulation.md": encapsulation_append,
    "abstraction.md": abstraction_append
}

base_path = "/Users/meetvirugama/Desktop/NOTES/OPPS/notes"

for filename, content in append_map.items():
    filepath = os.path.join(base_path, filename)
    if os.path.exists(filepath):
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Upgraded {filename}")
    else:
        print(f"❌ File not found: {filename}")

print("\n🚀 50 LPA Advanced Topics successfully injected into all notes!")
