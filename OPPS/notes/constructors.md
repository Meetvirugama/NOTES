# Constructors

## Definition

A **Constructor** is a specialized subroutine called to create an object. It prepares the new object for use, typically by initializing its instance variables. 

**Deep Dive:** A constructor acts as the gatekeeper of an object's lifecycle. It guarantees that an object cannot exist in memory in a "half-baked" or invalid state. In statically typed languages, the compiler guarantees that memory allocation (`new`) and initialization (the constructor) are treated as a single, atomic operation.

## Why It Is Needed

If you instantiate an object without a constructor, its internal state variables will contain garbage values (in C++) or zero/null values (in Java). If a `BankAccount` object is created with a `null` owner or a `-1` balance, the entire system is compromised. Constructors enforce invariants—rules that must be true for the object to be valid.

## Key Concepts

- **Name Binding:** Must match the class name exactly.
- **No Return Type:** Not even `void`. The implicit return is the memory address of the newly initialized object.
- **Default Constructor:** A no-argument constructor. Provided by the compiler ONLY if you write no other constructors.
- **Parameterized Constructor:** Allows passing custom initialization data.
- **Copy Constructor:** Creates a new object as a deep or shallow clone of an existing object.
- **Destructor (C++):** The reverse of a constructor. Cleans up heap allocations and releases locks when the object is destroyed.

## How It Works Internally

When you execute `Student* s = new Student(10);` in C++:
1. `operator new` is called to allocate `sizeof(Student)` bytes of raw, uninitialized memory on the heap.
2. The `Student(int)` constructor is invoked, and the `this` pointer is set to the address of that raw memory.
3. The constructor's Initializer List executes.
4. The constructor's body executes.
5. The memory address is returned and assigned to the pointer `s`.

## Syntax & Practical Example

Let's look at C++ to understand deep memory mechanics (Initializer Lists and Destructors).

### C++: Line-by-Line Execution Flow

```cpp
#include <iostream>
using namespace std;

class DatabaseConnection {
private:
    string url;
    int* cache; // Pointer to heap memory

public:
    // 1. Parameterized Constructor with Initializer List
    DatabaseConnection(string u) : url(u) {
        // 2. Constructor Body (Resource Allocation)
        cache = new int[100]; 
        cout << "Connected to " << url << " and allocated cache." << endl;
    }

    // 3. Copy Constructor (Deep Copy)
    DatabaseConnection(const DatabaseConnection& other) : url(other.url) {
        cache = new int[100]; // Allocate NEW memory
        for(int i=0; i<100; i++) cache[i] = other.cache[i]; // Copy contents
        cout << "Deep copied connection." << endl;
    }

    // 4. Destructor (Resource Deallocation)
    ~DatabaseConnection() {
        delete[] cache; 
        cout << "Disconnected from " << url << " and freed cache." << endl;
    }
};

int main() {
    DatabaseConnection db1("postgres://localhost"); // Calls Parameterized
    DatabaseConnection db2 = db1;                   // Calls Copy Constructor
    
    return 0; 
    // main ends. db2 destroyed first, then db1 (LIFO stack order).
}
```

### Line-by-Line Breakdown:
1. `DatabaseConnection(string u) : url(u)`: The **Initializer List** (`: url(u)`). This initializes the `url` variable directly into its memory slot *before* the constructor body runs. This is highly efficient and mandatory for `const` variables or references.
2. `cache = new int[100];`: The constructor acquires resources (memory, file handles, network sockets). This is the **RAII (Resource Acquisition Is Initialization)** principle.
3. `DatabaseConnection(const DatabaseConnection& other)`: The Copy Constructor. If we didn't write this, C++ would do a *shallow copy* (both `db1.cache` and `db2.cache` would point to the exact same array).
4. `~DatabaseConnection()`: The Destructor. Guarantees that when the object goes out of scope, the heap memory assigned to `cache` is returned to the OS.

## Edge Cases

### Edge Case 1: Infinite Recursion in Copy Constructors
- **What happens:** If you write a copy constructor like `Student(Student other)`, it will not compile. 
- **Why:** To pass `other` *by value* into the function, the compiler must make a copy of it... which requires calling the copy constructor... which requires passing by value... creating an infinite loop.
- **Fix:** Always pass by reference: `Student(const Student& other)`.

### Edge Case 2: Virtual Methods Inside Constructors
- **What happens:** Calling a `virtual` method from inside a constructor in C++ or Java is highly dangerous.
- **Java:** The child class's overridden method will execute, BUT the child class hasn't finished initializing yet! The overridden method might access variables that are still `null`.
- **C++:** The base class constructor temporarily sets the `v-table` to point to its own methods. It will NOT call the child's overridden method.
- **Best Practice:** Never call virtual/overridden methods from inside a constructor.

### Edge Case 3: Throwing Exceptions from Constructors
- **What happens:** If a constructor throws an exception half-way through, the object is considered "not fully constructed".
- **Internal Behavior (C++):** The destructor will **not** be called for that object. Any heap memory allocated before the exception was thrown will leak unless caught and freed within the constructor.

## Tricky FAANG Interview Questions

### Question 1: Private Constructors
**Q:** *Why would you ever make a constructor `private`? How would you create an object?*
**Answer & Explanation:** A private constructor prevents anyone outside the class from instantiating it. You create the object by providing a `public static` method (a Factory Method) inside the class that calls the private constructor. 
**Why it's asked:** To see if you understand the **Singleton Design Pattern** (ensuring only one instance of a class ever exists) or the **Factory Pattern**.

### Question 2: Initializer Lists vs Assignment
**Q:** *In C++, what is the performance difference between using an initializer list vs assigning values inside the constructor body?*
**Answer & Explanation:** If you use assignment inside the body, the member object is first constructed using its default constructor, and *then* the assignment operator is called. That's two operations. An initializer list bypasses the default constructor and directly constructs the member using the parameterized constructor. It is significantly faster and avoids double-initialization overhead.

### Question 3: Move Semantics (C++11)
**Q:** *What is a Move Constructor and why was it introduced?*
**Answer & Explanation:** Before C++11, returning a large object (like a massive vector) from a function by value resulted in a hugely expensive Copy Constructor call. A Move Constructor (`Student(Student&& other)`) "steals" the pointers/resources from a temporary (rvalue) object that is about to be destroyed anyway, leaving the temporary object in a valid but empty state. This turns an O(N) deep copy into an O(1) pointer swap.

## OA Tips

- If you define a class in Java for a custom data structure (e.g., a `TrieNode`), always explicitly write the constructor to initialize collections (`children = new HashMap<>()`). Forgetting this leads to instant `NullPointerException`s during BFS/DFS traversals.

## 2-Minute Revision

- **Constructor:** Gatekeeper of initialization. Enforces state invariants.
- **Types:** Default (no args), Parameterized, Copy (deep clones), Move (steals resources).
- **RAII:** Resource Acquisition Is Initialization. Bind resource lifecycles (memory, files) to object lifecycles using constructors and destructors.
- **Copy Constructor:** Must pass by reference to avoid infinite loops. Avoid shallow copies if the class contains pointers.
- **Warning:** Do not call virtual methods inside constructors.


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

Alternatively, use the **Initialization-on-Demand Holder Idiom** (using a static inner class), which relies on the JVM's class-loader guarantees to be 100% thread-safe without locks:
```java
public class SafeSingleton {
    private SafeSingleton() {}
    
    // The inner class is not loaded into memory until getInstance() is called.
    // The JVM guarantees that static class initialization is thread-safe.
    private static class Holder {
        private static final SafeSingleton INSTANCE = new SafeSingleton();
    }
    
    public static SafeSingleton getInstance() {
        return Holder.INSTANCE; // Zero locks, 100% safe!
    }
}
```

### 50 LPA FAANG Questions
**Q:** *What happens if an exception is thrown inside a C++ constructor? Does a memory leak occur?*
**A:** If a constructor throws, the object is considered "never fully constructed". Therefore, its **destructor is never called**. However, any fully constructed sub-objects (class members) *will* have their destructors called. If you dynamically allocated memory (`new int[100]`) before the exception was thrown, that memory is leaked. This is why you must use **Smart Pointers** (`std::unique_ptr`) inside classes, as their destructors will fire and clean up the heap even if the parent constructor aborts.
