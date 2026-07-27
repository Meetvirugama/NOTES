# Object-Oriented Programming (OOP) Handbook

Welcome to the complete, production-grade OOP Handbook! This guide compiles all the essential, industry-level notes for technical interviews at top product-based companies (Google, Meta, Microsoft, Amazon, etc.), system design rounds, and deep conceptual revision.

## Table of Contents

1. [OOP Overview & The 4 Pillars](1-oop-overview-v2.md)
   - What is OOP and how it compares to Procedural/Functional Programming.
   - High-level overview of Encapsulation, Abstraction, Inheritance, and Polymorphism.
   - Composition over Inheritance & Data-Oriented Design.

2. [Classes and Objects](2-classes-and-objects-v2.md)
   - Blueprints and Instances.
   - Stack vs Heap memory allocation and Object Headers.
   - `new` keyword and JVM instantiation.

3. [Methods](3-methods-v2.md)
   - Object behavioral contracts.
   - Instance vs Static methods, and the hidden `this` pointer.
   - Dynamic dispatch, vtables, and JIT method inlining.

4. [Constructors](4-constructors-v2.md)
   - Initializing object state and invariant establishment.
   - Constructor chaining (`this()`, `super()`), Factory methods, and Copy constructors.
   - Resource Acquisition Is Initialization (RAII).

5. [Inheritance](5-inheritance-v2.md)
   - IS-A relationships and code reusability.
   - The Diamond Problem, memory layouts, and vptr/vtable internals.
   - The Fragile Base Class problem.

6. [Polymorphism](6-polymorphism-v2.md)
   - Compile-time (Method Overloading) vs Run-time (Method Overriding).
   - Dynamic Binding and JIT Devirtualization.
   - Strategy patterns and behavioral flexibility.

7. [Encapsulation](7-encapsulation-v2.md)
   - Data hiding, protection, and invariant enforcement.
   - Access modifiers across different languages.
   - Getters, Setters, Defensive copying, and Immutability.

8. [Abstraction](8-abstraction.md)
   - Hiding implementation complexity and API boundaries.
   - Abstract Classes vs Interfaces.
   - Dependency Inversion Principle (DIP).

9. [SOLID Principles](9-solid-principles.md)
   - The five foundational principles of OOP (SRP, OCP, LSP, ISP, DIP).
   - Code smells and how to refactor into robust OOP designs.

---

> **Tip for Interviews:** "Think Classes, Create Objects, Build Solutions!" But remember, at the senior level, focus on *why* OOP is used (decoupling, abstraction boundaries) and its costs (memory fragmentation, thread safety issues with mutable state). Always favor Composition over Inheritance!
