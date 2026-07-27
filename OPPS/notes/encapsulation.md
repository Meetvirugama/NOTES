# Encapsulation

## Definition

**Encapsulation** is the mechanism of wrapping data (variables) and code (methods) together as a single unit. It restricts direct access to some of the object's components, which is a means of preventing accidental interference and misuse. It is often referred to as **Data Hiding**.

## Why It Is Needed

Without encapsulation, an object's state (its variables) could be directly modified from anywhere in the program, potentially setting it to an invalid state (e.g., setting a bank account balance to a negative number or age to -5). Encapsulation forces other classes to use a specific interface (methods) to interact with the data, allowing the class to validate the data before accepting it.

## Key Concepts

- **Access Modifiers:** Keywords used to set the visibility of classes, methods, and variables (`private`, `protected`, `public`, `default`).
- **Private Variables:** The core rule of encapsulation is that class variables should be `private`.
- **Getters (Accessors):** Public methods used to *read* the value of a private variable.
- **Setters (Mutators):** Public methods used to *update* or *modify* the value of a private variable.

## How It Works

```mermaid
flowchart LR
    A[Client Code \n e.g. obj.setId(101)] --> B[Public Methods \n + setId() \n + getId()]
    B -->|Controls Access to| C[Private Data \n - id \n - name]
    
    subgraph Student Class [Encapsulated Unit]
    B
    C
    end
```

The Client Code cannot access the Private Data directly. It must go through the Public Methods gate.

## Syntax & Code Examples

### Java Example

```java
class Student {
    // 1. Private data variables (Data Hiding)
    private int id;
    private String name;

    // 2. Public Getter for ID
    public int getId() {
        return id; // returns the private variable
    }

    // 3. Public Setter for ID with Validation
    public void setId(int id) {
        if (id > 0) { // Validation check
            this.id = id;
        } else {
            System.out.println("Invalid ID");
        }
    }

    // Getter for Name
    public String getName() {
        return name;
    }

    // Setter for Name
    public void setName(String name) {
        this.name = name;
    }
}

public class Main {
    public static void main(String[] args) {
        Student obj = new Student();
        
        // obj.id = 101; // ERROR: id has private access
        
        obj.setId(101); // Correct way
        System.out.println("ID is: " + obj.getId());
    }
}
```

### C++ Example

```cpp
#include <iostream>
using namespace std;

class Student {
private:
    int id;

public:
    // Getter
    int getId() {
        return id;
    }

    // Setter
    void setId(int i) {
        if (i > 0) {
            id = i;
        }
    }
};

int main() {
    Student s;
    s.setId(101);
    cout << "ID: " << s.getId() << endl;
    return 0;
}
```

## Real-World Example

Think of a **Medicine Capsule**.
- The medicine (data) is hidden and protected inside the capsule.
- You can't just touch the powder directly; you swallow the capsule (use the method), which safely delivers the medicine.

Alternatively, think of a **Bank ATM**. You cannot access the vault of cash directly (private data). You must use the ATM screen/buttons (public methods) to withdraw cash, which verifies your PIN and balance before dispensing.

## Advantages

- **Data Protection / Security:** Protects data from unauthorized or accidental modification.
- **Flexibility / Maintainability:** You can change the internal implementation of a class without breaking the code of others who use the class (as long as the public getter/setter signatures remain the same).
- **Control:** Allows controlled access (e.g., you can make a variable read-only by providing a getter but no setter).

## Disadvantages

- **Boilerplate Code:** Requires writing lots of getter and setter methods (though modern IDEs or tools like Lombok in Java generate these automatically).
- **Slight Overhead:** Calling a method to get/set a variable has a marginal overhead compared to direct access.

## Internal Working

Encapsulation relies heavily on the compiler enforcing **Access Modifiers**. 
- `private`: Visible only within the same class.
- `default` (package-private): Visible only within the same package/folder.
- `protected`: Visible within the same package AND in subclasses in other packages.
- `public`: Visible everywhere.

## Comparisons

### Encapsulation vs Abstraction

| Feature | Encapsulation | Abstraction |
| :--- | :--- | :--- |
| **Meaning** | Hiding the *data* to protect it. | Hiding the *implementation details* to reduce complexity. |
| **Mechanism** | Wrapping data and methods into a single unit (Classes & Access Modifiers). | Exposing only essential features (Interfaces & Abstract Classes). |
| **Focus** | How data is stored and protected. | What the object does instead of how it does it. |
| **Analogy** | A capsule protecting medicine. | The steering wheel of a car (you don't need to know how the engine works to turn). |

## Best Practices

- Make variables `private`.
- Provide `public` getter and setter methods.
- Access and modify data *only* through methods.
- Always include validation logic inside setter methods to maintain data integrity.

## Common Mistakes

- Making all variables `public` to save time typing getters and setters. This completely destroys encapsulation and is an immediate red flag in interviews.
- Returning direct references to mutable internal objects in getters (e.g., returning a `List` directly instead of a copy). This allows the caller to modify the internal list, breaking encapsulation.

## Interview Questions

### Beginner
1. What is encapsulation?
2. How do you achieve encapsulation in Java/C++?
3. What is a getter and a setter?

### Intermediate
1. ⭐ What is the difference between Encapsulation and Abstraction?
2. What are access modifiers? Explain the difference between `private` and `protected`.
3. How can you make a class completely "read-only"? (Answer: Provide private variables and only public getters, no setters).

### Advanced
1. ⭐ Explain how returning a mutable object from a getter breaks encapsulation, and how to fix it. (Answer: It allows external code to modify the internal object. Fix it by returning a defensive copy or an immutable view).
2. Is Encapsulation strictly about security? (Answer: No, it's more about preventing accidental corruption of state and decoupling the internal representation from the external API).

## OA Tips

- In OOD (Object-Oriented Design) OA questions, making your attributes `private` and using getters/setters is a core grading metric. If you leave variables `public`, you will likely lose points for poor design.

## 2-Minute Revision

- **Encapsulation:** Wrapping data and methods into one unit.
- **Goal:** Data hiding and protection.
- **How:** Use `private` for variables, `public` for getters/setters.
- **Benefit:** Controlled access, maintainability, prevents invalid state.
- **Difference from Abstraction:** Encapsulation hides data; Abstraction hides implementation complexity.

---

## Flashcards

**Q:** What is encapsulation?
**A:** The wrapping of data and methods into a single unit, restricting direct external access to the data.

**Q:** What is another term commonly used for Encapsulation?
**A:** Data Hiding.

**Q:** How do you achieve encapsulation?
**A:** By declaring class variables as `private` and providing `public` getter and setter methods.

**Q:** What is a Getter?
**A:** A method used to retrieve the value of a private variable.

**Q:** What is a Setter?
**A:** A method used to modify the value of a private variable, often containing validation logic.

**Q:** Encapsulation vs Abstraction?
**A:** Encapsulation hides data (security/protection). Abstraction hides implementation details (simplicity).

**Q:** Can encapsulation make a class read-only?
**A:** Yes, by providing only getters and no setters.

**Q:** Can encapsulation make a class write-only?
**A:** Yes, by providing only setters and no getters (rare, but possible).

**Q:** What is the scope of a `private` member?
**A:** It is accessible only within the class it is declared.

**Q:** What happens if a setter does not validate data?
**A:** It partially defeats the purpose of encapsulation, as invalid state can still be assigned.

**Q:** Does encapsulation improve maintainability?
**A:** Yes, because you can change the internal data structure without breaking external code that relies on the getters/setters.

**Q:** What is a defensive copy?
**A:** Returning a clone/copy of an internal object in a getter so the original private object cannot be modified.

**Q:** What is the default access modifier in Java?
**A:** Package-private (accessible within the same package).

**Q:** What is the default access modifier for a `class` in C++?
**A:** `private`.

**Q:** What is the default access modifier for a `struct` in C++?
**A:** `public`.

**Q:** Why is returning a private array directly from a getter bad?
**A:** Because arrays are mutable reference types; the caller can modify the array elements directly.

**Q:** How do you fix returning a private array?
**A:** Return a clone of the array (`return arr.clone();`).

**Q:** Are access modifiers checked at compile time or run time?
**A:** Compile time.

**Q:** Can Reflection break encapsulation in Java?
**A:** Yes, using `setAccessible(true)`, reflection can access private fields.

**Q:** Is encapsulation a feature of procedural programming?
**A:** No, it is a core pillar of Object-Oriented Programming.


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

public class Main {
    public static void main(String[] args) {
        SafeBankAccount acc = new SafeBankAccount();
        
        // Even if a malicious thread tries to lock the 'acc' object itself:
        synchronized(acc) {
            // It doesn't matter! The deposit() method uses a hidden 'lock' object.
            // Other threads can still deposit safely without being permanently blocked.
            acc.deposit(100); 
        }
    }
}
```

### 50 LPA FAANG Questions
**Q:** *Explain the concept of "Immutability" and why Functional Programming paradigms are invading OOP codebases.*
**A:** Immutability is the ultimate form of encapsulation. Instead of hiding data behind setters, you provide *no setters at all*. An object's state is set once in the constructor and never changes. If you need a change, you return a brand new object.
Why? Because immutable objects are **inherently thread-safe**. They require zero locks, zero monitors, and zero synchronization overhead. In massively parallel systems (like Apache Spark or trading engines), mutating state causes locking bottlenecks. Immutability completely bypasses this, which is why modern Java `Records` and C++ `const` correctness are so heavily emphasized.
