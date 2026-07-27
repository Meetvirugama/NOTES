# SOLID Principles — Industry-Level Interview Notes

> **Target:** SDE-1 → SDE-3 | Google · Goldman Sachs · Microsoft · Amazon · Meta · Apple · Uber · LinkedIn  
> **Level:** Production-grade, interview-ready, zero college fluff

---

## Table of Contents
1. [Definition](#1-definition)
2. [Why It Exists](#2-why-it-exists)
3. [S — Single Responsibility Principle (SRP)](#3-s--single-responsibility-principle)
4. [O — Open-Closed Principle (OCP)](#4-o--open-closed-principle)
5. [L — Liskov Substitution Principle (LSP)](#5-l--liskov-substitution-principle)
6. [I — Interface Segregation Principle (ISP)](#6-i--interface-segregation-principle)
7. [D — Dependency Inversion Principle (DIP)](#7-d--dependency-inversion-principle)
8. [Visual Diagrams](#8-visual-diagrams)
9. [Real World Analogy](#9-real-world-analogy)
10. [Interview Explanation](#10-interview-explanation)
11. [Interview Follow-up Questions](#11-interview-follow-up-questions)
12. [Coding Examples](#12-coding-examples)
13. [Common Mistakes](#13-common-mistakes)
14. [Best Practices](#14-best-practices)
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
SOLID is an acronym for **five design principles** that make object-oriented software easier to understand, extend, and maintain without causing regressions.

### Technically
SOLID principles, coined by Robert C. Martin ("Uncle Bob"), establish guidelines for arranging classes and modules to achieve:
- **Low coupling**: Changes in one module don't ripple into others
- **High cohesion**: Related code lives together
- **Testability**: Components can be tested in isolation
- **Extensibility**: New behavior added without modifying existing code

| Letter | Principle | Core Statement |
|--------|-----------|----------------|
| **S** | Single Responsibility | One class = one reason to change |
| **O** | Open-Closed | Open for extension; closed for modification |
| **L** | Liskov Substitution | Subtypes must be fully substitutable for base types |
| **I** | Interface Segregation | Clients shouldn't depend on interfaces they don't use |
| **D** | Dependency Inversion | Depend on abstractions, not concrete implementations |

### From an Interviewer's Perspective
> "Reciting the definitions is the floor. I want to hear: how SRP is about stakeholders, not lines of code. The formal LSP definition (preconditions, postconditions, invariants). How ISP emerged from fat interfaces causing distant code coupling. How DIP is different from Dependency Injection. And critically — when NOT to apply SOLID (YAGNI, startup code, throwaway scripts)."

⭐ **Key Insight**: SOLID principles are tools, not laws. Experienced engineers know when applying them is beneficial and when it creates over-engineering.

---

## 2. Why It Exists

### The Problem: "Big Ball of Mud" Codebases

**Without SOLID, software degrades into:**
```
3 months into development:
  - UserController has 2000 lines
  - Single `processOrder()` method touches DB, sends email, charges card, updates inventory
  - Adding a new payment method requires editing 8 files
  - Unit tests require setting up the entire application
  - Deployment: change one line → test entire system (because everything is coupled)
```

**The symptoms SOLID addresses:**

| Code Smell | Violated Principle |
|-----------|-------------------|
| God class (3000-line class) | SRP |
| `if-else` chain for every new type | OCP |
| `UnsupportedOperationException` in override | LSP |
| Class implements 10 methods, uses 3 | ISP |
| `new ConcreteClass()` inside business logic | DIP |
| Unit tests require starting real DB/server | DIP |

---

## 3. S — Single Responsibility Principle

### Definition
> "A class should have one, and only one, reason to change."
— Robert C. Martin

**Key**: "Reason to change" = a **stakeholder** who might request changes, not a feature or function.

### Why This Wording Matters
```
class UserManager {
    void createUser(User user) { /* DB team wants changes */ }
    void sendWelcomeEmail(User user) { /* Marketing team wants changes */ }
    void generateUserReport(User user) { /* Analytics team wants changes */ }
    void validateUserAge(User user) { /* Legal team wants changes */ }
}
```
Four stakeholders = four reasons to change = **SRP violated**.

### Violation vs. Correct

```java
// VIOLATION: One class, three stakeholders' concerns
public class OrderProcessor {
    public ProcessingResult processOrder(Order order) {
        // CONCERN 1: Validation (Business Analyst requests changes)
        if (order.getItems().isEmpty()) throw new IllegalArgumentException();
        if (order.getTotal().isNegative()) throw new IllegalArgumentException();

        // CONCERN 2: DB persistence (DBA requests changes)
        String sql = "INSERT INTO orders VALUES(?, ?, ?)";
        jdbcTemplate.update(sql, order.getId(), order.getStatus(), order.getTotal());

        // CONCERN 3: Email notification (Marketing requests changes)
        String template = "Dear " + order.getCustomerName() + ", your order #" + order.getId() + "...";
        emailClient.send(order.getCustomerEmail(), "Order Confirmed", template);

        return ProcessingResult.success();
    }
}
// PROBLEM: Marketing can't change email without testing DB logic
//          DBA can't change schema without testing email logic
//          All 3 tests need all 3 setups

// CORRECT: Each class has one stakeholder
public class OrderValidator {
    public void validate(Order order) {
        if (order.getItems().isEmpty())
            throw new ValidationException("Order must have items");
        if (order.getTotal().isNegative())
            throw new ValidationException("Order total cannot be negative");
    }
}

public class OrderRepository {
    public void save(Order order) {
        jdbcTemplate.update("INSERT INTO orders VALUES(?, ?, ?)",
            order.getId(), order.getStatus(), order.getTotal());
    }
}

public class OrderNotificationService {
    public void sendConfirmation(Order order) {
        String body = emailTemplateEngine.render("order-confirmation", order);
        emailClient.send(order.getCustomerEmail(), "Order Confirmed", body);
    }
}

// Orchestrator: SRP is about cohesion, not number of classes
public class PlaceOrderUseCase {
    private final OrderValidator validator;
    private final OrderRepository repository;
    private final OrderNotificationService notifications;

    public PlaceOrderUseCase(OrderValidator validator, OrderRepository repository,
                             OrderNotificationService notifications) {
        this.validator = validator;
        this.repository = repository;
        this.notifications = notifications;
    }

    public OrderId execute(PlaceOrderCommand command) {
        Order order = new Order(command);
        validator.validate(order);
        repository.save(order);
        notifications.sendConfirmation(order);
        return order.getId();
    }
}
```

### Key SRP Insight
- SRP is NOT "one method per class" — that leads to class explosion
- SRP is "one COHESIVE set of related responsibilities" belonging to the same stakeholder
- A `User` class with `getName()`, `getEmail()`, `getAge()` is fine — all identity concern

---

## 4. O — Open-Closed Principle

### Definition
> "Software entities should be open for extension but closed for modification."
— Bertrand Meyer, popularized by Robert C. Martin

**Translation**: Add new behavior by adding new code (extending), not by modifying existing, working code.

### Violation vs. Correct

```java
// VIOLATION: Adding a new shape requires modifying AreaCalculator
public class AreaCalculator {
    public double calculate(Object shape) {
        if (shape instanceof Circle c) {
            return Math.PI * c.getRadius() * c.getRadius();
        } else if (shape instanceof Rectangle r) {
            return r.getWidth() * r.getHeight();
        } else if (shape instanceof Triangle t) {    // Adding new type = MODIFY this class!
            return 0.5 * t.getBase() * t.getHeight();
        }
        throw new UnsupportedOperationException("Unknown shape: " + shape.getClass());
    }
}
// Adding Triangle requires: edit AreaCalculator, retest AreaCalculator, redeploy

// CORRECT: Each shape knows how to calculate its own area
public interface Shape {
    double area();           // Open for extension via new implementations
    double perimeter();
}

public class Circle implements Shape {
    private final double radius;
    public Circle(double radius) { this.radius = radius; }
    public double area() { return Math.PI * radius * radius; }
    public double perimeter() { return 2 * Math.PI * radius; }
}

public class Rectangle implements Shape {
    private final double width, height;
    public Rectangle(double width, double height) { this.width = width; this.height = height; }
    public double area() { return width * height; }
    public double perimeter() { return 2 * (width + height); }
}

// Adding Triangle: new class, zero modification to existing code!
public class Triangle implements Shape {
    private final double base, height, side1, side2;
    public Triangle(double base, double height, double side1, double side2) { /* ... */ }
    public double area() { return 0.5 * base * height; }
    public double perimeter() { return base + side1 + side2; }
}

// AreaCalculator never changes:
public class AreaCalculator {
    public double calculate(Shape shape) { return shape.area(); } // Closed for modification
    public double totalArea(List<Shape> shapes) {
        return shapes.stream().mapToDouble(Shape::area).sum();
    }
}
```

### OCP Mechanisms
- **Interfaces**: New behavior = new implementation class
- **Abstract class + Template Method**: New step = new subclass override
- **Strategy Pattern**: New algorithm = new strategy implementation
- **Decorator**: New behavior = new wrapper class
- **Plugin Architecture**: New feature = new plugin, zero core changes

---

## 5. L — Liskov Substitution Principle

### Definition
> "If S is a subtype of T, then objects of type T may be replaced with objects of type S without altering the correctness of the program."
— Barbara Liskov, 1987 (Turing Award recipient)

**Formal Rules for valid substitution:**
1. **Preconditions cannot be strengthened** in subtype (accept same or more)
2. **Postconditions cannot be weakened** in subtype (guarantee same or more)
3. **Invariants must be preserved** (subtype maintains base class invariants)
4. **No new exceptions** that aren't in the base method's exception spec

### Classic Violation: Square-Rectangle Problem

```java
// VIOLATION: Square "IS-A" Rectangle mathematically, but NOT behaviorally
class Rectangle {
    protected int width, height;

    public void setWidth(int w) { this.width = w; }
    public void setHeight(int h) { this.height = h; }
    public int area() { return width * height; }
}

class Square extends Rectangle {
    @Override public void setWidth(int w) { this.width = this.height = w; }  // Forces height = width!
    @Override public void setHeight(int h) { this.width = this.height = h; } // Forces width = height!
}

// Test that MUST pass for Rectangle → FAILS for Square
void assertRectangleInvariant(Rectangle r) {
    r.setWidth(5);
    r.setHeight(4);
    assert r.area() == 20; // Rectangle: 5 × 4 = 20 ✓
                           // Square: height=4 sets both to 4 → 4×4=16 ✗ FAILS!
}

// LSP violated: cannot use Square wherever Rectangle is expected
assertRectangleInvariant(new Rectangle()); // PASSES
assertRectangleInvariant(new Square());    // FAILS — broken!
```

### LSP-Compliant Redesign
```java
// FIX: Both shapes share a common abstraction, but no inheritance between them
public interface Shape { int area(); }

public final class Rectangle implements Shape {
    private int width, height;
    public Rectangle(int width, int height) { this.width = width; this.height = height; }
    public void setWidth(int w) { this.width = w; }   // Independent — no coupling to height
    public void setHeight(int h) { this.height = h; } // Independent — no coupling to width
    public int area() { return width * height; }
}

public final class Square implements Shape {
    private int side;
    public Square(int side) { this.side = side; }
    public void setSide(int s) { this.side = s; }    // Named correctly — not setWidth/setHeight
    public int area() { return side * side; }
}
// No LSP issue — neither can be substituted for the other's specific behaviors
```

### LSP Precondition/Postcondition Examples
```java
// PRECONDITION VIOLATION (strengthened input):
class FileProcessor {
    public void process(String filename) { /* accepts any filename */ }
}
class ImageProcessor extends FileProcessor {
    @Override
    public void process(String filename) {
        if (!filename.endsWith(".jpg")) throw new IllegalArgumentException(); // Stronger precondition!
        // Callers that pass "data.csv" to FileProcessor reference now fail!
    }
}

// POSTCONDITION VIOLATION (weakened guarantee):
class PositiveAdder {
    public int add(int a, int b) {
        // Postcondition: result is always positive
        return Math.abs(a + b);
    }
}
class BrokenAdder extends PositiveAdder {
    @Override
    public int add(int a, int b) {
        return a + b; // Might return negative! Weakened guarantee!
    }
}

// CORRECT: Subtype keeps same or stronger guarantees
class PositiveOnlyAdder extends PositiveAdder {
    @Override
    public int add(int a, int b) {
        int result = a + b;
        if (result <= 0) throw new ArithmeticException("Sum must be positive, got: " + result);
        return result; // Still positive — preserved and strengthened postcondition ✓
    }
}
```

---

## 6. I — Interface Segregation Principle

### Definition
> "Clients should not be forced to depend on interfaces they do not use."
— Robert C. Martin

**The problem**: Fat interfaces force implementing classes to stub methods they don't need — creating lying, broken implementations.

### Violation vs. Correct

```java
// VIOLATION: Fat interface forces unrelated implementations
public interface Worker {
    void work();
    void eat();
    void sleep();
    void attendMeeting();
    void submitTimesheet();
}

class HumanEmployee implements Worker {
    public void work() { /* works */ }
    public void eat() { /* eats */ }
    public void sleep() { /* sleeps */ }
    public void attendMeeting() { /* attends */ }
    public void submitTimesheet() { /* submits */ }
}

class Robot implements Worker {
    public void work() { /* works */ }
    public void eat() { throw new UnsupportedOperationException("Robots don't eat!"); } // LIE!
    public void sleep() { throw new UnsupportedOperationException("Robots don't sleep!"); } // LIE!
    public void attendMeeting() { /* can attend */ }
    public void submitTimesheet() { throw new UnsupportedOperationException("Robots don't timesheet!"); } // LIE!
}
// Robot is "a Worker" but implements Worker incorrectly. Any code calling worker.eat() on a Robot = crash!

// CORRECT: Segregated interfaces — each class implements only what applies
public interface Workable { void work(); }
public interface Feedable { void eat(); }
public interface Restable { void sleep(); }
public interface MeetingAttendee { void attendMeeting(); }
public interface TimesheetSubmitter { void submitTimesheet(); }

class HumanEmployee implements Workable, Feedable, Restable, MeetingAttendee, TimesheetSubmitter {
    // Implements all — correct!
}

class Robot implements Workable, MeetingAttendee {
    public void work() { /* works */ }
    public void attendMeeting() { /* attends digitally */ }
    // No eat(), sleep(), submitTimesheet() — doesn't pretend to support them!
}
```

### ISP in Java Standard Library
```
java.io.Closeable          → close()           (1 method)
java.lang.Runnable         → run()             (1 method)
java.lang.Comparable<T>    → compareTo()       (1 method)
java.lang.Iterable<T>      → iterator()        (1 method)
java.util.function.Function → apply()           (1 method)

These are ISP masterclasses: minimal, focused, composable.
CrudRepository → PagingAndSortingRepository → JpaRepository (graduated)
→ Take only what you need
```

---

## 7. D — Dependency Inversion Principle

### Definition
> "1. High-level modules should not depend on low-level modules. Both should depend on abstractions.
> 2. Abstractions should not depend on details. Details should depend on abstractions."
— Robert C. Martin

### Understanding the "Inversion"

```
BEFORE DIP (traditional dependency direction):
  [OrderService] ──depends on──> [MySQLOrderRepository] ──depends on──> [MySQL]
  High-level → Low-level → Infrastructure

AFTER DIP (inverted dependency direction):
  [OrderService] ──depends on──> [OrderRepository Interface]
                                         ↑ implements
                                 [MySQLOrderRepository]
  High-level → Abstraction ← Low-level
  The dependency direction from high-level to low-level is INVERTED!
```

### Violation vs. Correct

```java
// VIOLATION: High-level module coupled to concrete low-level module
public class ReportService {
    // Direct instantiation = tight coupling!
    private MySQLReportRepository repo = new MySQLReportRepository("jdbc:mysql://...", "user", "pass");
    private SMTPEmailSender emailSender = new SMTPEmailSender("smtp.gmail.com", 587);
    private PDFReportGenerator pdfGen = new PDFReportGenerator("/templates/");

    public void generateAndSendMonthlyReport(String recipientEmail, YearMonth month) {
        List<SalesData> data = repo.findByMonth(month);   // Coupled to MySQL
        Report report = pdfGen.generate(data, month);      // Coupled to PDF library
        emailSender.send(recipientEmail, report.getBytes()); // Coupled to SMTP
    }
    // Testing: requires MySQL + SMTP + PDF library configured!
    // Switching to MongoDB: rewrite ReportService!
    // Switching to SendGrid: rewrite ReportService!
}

// CORRECT: High-level depends on abstractions
public interface ReportRepository {
    List<SalesData> findByMonth(YearMonth month);
    List<SalesData> findByDateRange(LocalDate from, LocalDate to);
}

public interface ReportEmailSender {
    void send(String recipient, String subject, Report report);
}

public interface ReportGenerator {
    Report generate(List<SalesData> data, String title);
}

// High-level service: depends on abstractions ONLY
public class ReportService {
    private final ReportRepository repository;
    private final ReportEmailSender emailSender;
    private final ReportGenerator reportGenerator;

    // Dependencies INJECTED — not created!
    public ReportService(ReportRepository repository,
                         ReportEmailSender emailSender,
                         ReportGenerator reportGenerator) {
        this.repository = repository;
        this.emailSender = emailSender;
        this.reportGenerator = reportGenerator;
    }

    public void generateAndSendMonthlyReport(String recipientEmail, YearMonth month) {
        List<SalesData> data = repository.findByMonth(month);
        Report report = reportGenerator.generate(data, month + " Sales Report");
        emailSender.send(recipientEmail, "Monthly Report: " + month, report);
    }
}

// Production wiring:
ReportService prodService = new ReportService(
    new MySQLReportRepository(dbConfig),
    new SendGridEmailSender(sendGridConfig),
    new PDFReportGenerator(templatePath)
);

// Testing: inject fakes — zero infrastructure needed!
ReportService testService = new ReportService(
    new InMemoryReportRepository(testData),
    new FakeEmailSender(),              // Captures emails for assertion
    new SimpleTextReportGenerator()     // No PDF library needed in tests
);

testService.generateAndSendMonthlyReport("test@test.com", YearMonth.of(2024, 1));
assertThat(fakeEmailSender.getSentEmails()).hasSize(1);
```

### DIP vs Dependency Injection (Common Confusion)
| Concept | What It Is |
|---------|------------|
| **DIP** | Design principle: high-level modules depend on abstractions |
| **Dependency Injection (DI)** | Technique to achieve DIP: pass concrete implementations from outside |
| **IoC Container** | Framework automating DI (Spring, Guice, Dagger) |

---

## 8. Visual Diagrams

### SOLID Summary Diagram
```
S - Single Responsibility:
  BAD:  [UserService] ─→ [DB] + [Email] + [PDF] + [Validator]
  GOOD: [PlaceOrderUseCase] ─→ [UserRepo] + [EmailSvc] + [Validator]
        Each class = one reason to change

O - Open-Closed:
  BAD:  if shape == Circle → ... else if shape == Square → ...
  GOOD: [Shape] ←── [Circle] | [Square] | [Triangle]
        Adding shape = new class, zero changes to AreaCalculator

L - Liskov Substitution:
  BAD:  Square extends Rectangle → setWidth(5), setHeight(4) → area = 16, not 20
  GOOD: Circle implements Shape | Rectangle implements Shape
        No inheritance where behavioral contracts differ

I - Interface Segregation:
  BAD:  Worker { work() + eat() + sleep() + submitTimesheet() }
        Robot forced to implement eat() and throw!
  GOOD: Workable { work() } + Feedable { eat() } + Restable { sleep() }
        Robot: only implements Workable

D - Dependency Inversion:
  BAD:  OrderService ──────────→ MySQLRepo (concrete)
  GOOD: OrderService → OrderRepository (abstract) ← MySQLRepo
                                                   ← MongoRepo
                                                   ← InMemoryRepo
```

### DIP Dependency Arrow Inversion
```
Traditional (bad):
  [Controller] ──→ [Service] ──→ [Repository] ──→ [MySQL]
  All arrows point "down" → tight coupling top to bottom

DIP applied:
  [Controller] ──→ [IService]         ← [ServiceImpl]
  [ServiceImpl] ──→ [IRepository]     ← [MySQLRepository]
                                       ← [MongoRepository]
                                       ← [InMemoryRepository]

  High-level modules depend on abstractions (arrows toward interfaces).
  Low-level modules implement abstractions (arrows FROM implementations).
```

---

## 9. Real World Analogy

### S — Single Responsibility
- **Restaurant**: Chef cooks, waiter serves, cashier bills. One person doesn't do all three.
- **Company**: Engineer writes code, HR hires people, Finance manages budget. One team doesn't handle all.

### O — Open-Closed
- **Power outlet**: Accepts new devices (extension — new appliances plug in) without rewiring (modification — outlet doesn't change)
- **App stores**: New apps can be added (extension) without modifying the App Store platform (closed)

### L — Liskov Substitution
- **Vehicles at a car rental**: You booked "a Car". You should be able to drive any car (sedan, hatchback, SUV) placed in that spot. A non-drivable "car" violates LSP.
- **Employees in payroll**: Any Employee subtype (FullTime, PartTime, Contractor) should work in `processPayroll()` without special casing.

### I — Interface Segregation
- **Remote controls**: TV remote (TV functions), AC remote (AC functions), Sound Bar remote — not one massive remote with 200 buttons for all devices.
- **Work contracts**: Developer contract covers coding. HR contract covers hiring. Neither is forced to do the other's job.

### D — Dependency Inversion
- **Standard power socket**: Your device depends on a socket standard (abstraction), not on a specific power plant (concrete). Any power plant meeting the standard works.
- **JDBC**: Application depends on `Connection` interface (abstraction), not MySQL. Any JDBC driver works.

---

## 10. Interview Explanation

### 30 Seconds
> "SOLID is five design principles: Single Responsibility — one class, one reason to change. Open-Closed — add behavior by adding code, not changing existing code. Liskov Substitution — subtypes must be fully substitutable for base types without breaking correctness. Interface Segregation — don't force clients to depend on methods they don't use. Dependency Inversion — depend on abstractions, not concrete implementations."

### 1 Minute
> "SOLID principles emerge from the failure modes of poorly designed OOP. Without SRP, one class accumulates concerns from multiple stakeholders — any change risks breaking unrelated functionality. Without OCP, adding new types requires modifying existing working code — regression risk. Without LSP, inheritance hierarchies break the substitution that polymorphism relies on. Without ISP, fat interfaces force false implementations. Without DIP, business logic is tightly coupled to infrastructure — untestable and inflexible.
>
> Together they produce: low coupling, high cohesion, and testable code."

### 3 Minutes
> "Let me go principle by principle with the non-obvious insights.
>
> SRP's definition of 'reason to change' is about STAKEHOLDERS, not functions. A class that only developers will ever ask you to change has one reason to change. A class that the Marketing team and the Database team both request changes in has two reasons — that's the SRP signal.
>
> OCP is realized via interfaces and polymorphism. The classic mistake: solving it with inheritance instead of interfaces. Inheritance-based OCP creates fragile base class problems. Interface-based OCP — each new behavior = new class implementing the interface — is clean and safe.
>
> LSP is the most subtle. It's about BEHAVIORAL contracts, not just syntactic type compatibility. Java's type system ensures you can assign a Square to a Rectangle variable. LSP says: that's not enough — when callers use the Rectangle contract (independent width/height), the Square must still work correctly. The formal check: can you run all tests written against the base class against every subclass? If any test fails for a subclass, LSP is violated.
>
> ISP emerged from the "fat interface" problem in large systems. If Service A uses 2 of 15 methods in an interface, every change to the other 13 methods recompiles Service A — even though it doesn't use them. Narrow interfaces fix this.
>
> DIP is what makes Dependency Injection work. The principle is about the DIRECTION of dependency. High-level business logic should depend on stable abstractions, not volatile concrete implementations. The inversion is that instead of high-level modules knowing about low-level modules, both layers depend on an abstraction defined at the high level."

---

## 11. Interview Follow-up Questions

### Easy
| Question | Expected Answer |
|----------|-----------------|
| What does SOLID stand for? | Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, Dependency Inversion |
| Who coined SOLID? | Robert C. Martin ("Uncle Bob"). LSP: Barbara Liskov (1987). OCP: Bertrand Meyer (1988). |
| What does SRP mean? | A class should have one, and only one, reason to change |
| What does OCP mean? | Open for extension via new code; closed for modification of existing code |
| How does DIP differ from Dependency Injection? | DIP = design principle (direction of dependencies). DI = technique to achieve DIP. IoC = framework automating DI. |

### Medium
| Question | Expected Answer |
|----------|-----------------|
| What is "reason to change" in SRP? | A stakeholder who would request changes. Multiple stakeholders = SRP violated. |
| How does OCP prevent regressions? | New behavior = new class. Existing working code unchanged = no regression risk. |
| What is the Square-Rectangle LSP violation? | Square overrides setWidth/setHeight to keep them equal, breaking Rectangle's independent width/height invariant |
| What code smell indicates ISP violation? | Classes implementing interface but throwing `UnsupportedOperationException` for methods they don't support |
| What is the "Stable Dependencies Principle" and how does it relate to DIP? | Depend on things that change less often than you do. Abstractions are stable; implementations are volatile. DIP = depend on the stable layer. |

### Hard
| Question | Expected Answer |
|----------|-----------------|
| State LSP's formal precondition/postcondition rules | Preconditions can only be weakened (accept more), not strengthened. Postconditions can only be strengthened (guarantee more), not weakened. Invariants preserved. |
| Can SOLID principles conflict? | Yes. Strict SRP = many tiny classes = navigation overhead. Strict DIP = every class behind interface = over-engineering. Balance is contextual. |
| What is the "Reuse/Release Equivalence Principle"? | Granule of reuse = granule of release. Related to SRP: things released together should change together. |
| What is the "Acyclic Dependencies Principle"? | No dependency cycles between packages. DIP enables ADP: interface breaks cycle. |
| How does LSP relate to the Principle of Behavioral Subtyping? | They're equivalent. LSP = Barbara Liskov's formal definition of behavioral subtyping. |

### 💼 Google Level
> *"You're designing a distributed event processing system. How do you apply DIP so that event handlers can be tested without a real Kafka broker, and deployed to prod with a real Kafka broker?"*

Expected: `EventConsumer` interface (abstraction). `KafkaEventConsumer` for production. `InMemoryEventConsumer` for tests. `EventHandlerService` depends on `EventConsumer` interface. Test: inject `InMemoryEventConsumer` with synthetic events, assert handler behavior. No Kafka needed.

### 💼 Goldman Sachs Level
> *"Our RiskEngine directly creates MySQLPositionRepository. A requirement comes to support Redis-backed positions for low-latency trading. What SOLID principles are violated and how do you fix it with minimal disruption?"*

Expected: DIP violated — `RiskEngine` depends on concrete `MySQLPositionRepository`. Fix: introduce `PositionRepository` interface. Implement both `MySQLPositionRepository` and `RedisPositionRepository`. Wire `RiskEngine` with `PositionRepository`. OCP: adding Redis = new class, no change to `RiskEngine`. SRP: `RiskEngine` drops infrastructure concern. Test: inject `InMemoryPositionRepository`.

---

## 12. Coding Examples

### Full SOLID Refactor: Notification System
```java
// ============ BEFORE (All SOLID violations) ============

class NotificationService {
    // DIP violated: concrete dependencies created inline
    MySQLNotificationLog log = new MySQLNotificationLog();

    public void notify(String userId, String type, String message) {
        // SRP violated: type routing + sending + logging in one method
        // OCP violated: adding new type = modify this method
        if (type.equals("EMAIL")) {
            // ISP violated: EmailClient has 12 methods, we use 1
            EmailClient client = new EmailClient("smtp.gmail.com", 587);
            client.connect();
            client.authenticate("user", "pass");
            client.send(getUserEmail(userId), message);
            client.disconnect();
            log.insert(userId, type, message, "SENT");
        } else if (type.equals("SMS")) {
            TwilioClient twilio = new TwilioClient("SID", "TOKEN");
            twilio.sendSMS(getUserPhone(userId), message);
            log.insert(userId, type, message, "SENT");
        }
    }
}

// ============ AFTER (SOLID compliant) ============

// ISP: Small, focused interfaces
public interface MessageSender {
    boolean canHandle(NotificationChannel channel);
    void send(Recipient recipient, NotificationMessage message);
}

public interface NotificationLog {
    void record(NotificationEvent event);
    List<NotificationEvent> findByRecipient(String userId);
}

public interface RecipientResolver {
    Recipient resolve(String userId, NotificationChannel channel);
}

// OCP: New channel = new class, zero changes to NotificationService
@Component
public class EmailMessageSender implements MessageSender {
    private final JavaMailSender mailSender;
    private final TemplateEngine templates;

    public EmailMessageSender(JavaMailSender mailSender, TemplateEngine templates) {
        this.mailSender = mailSender;
        this.templates = templates;
    }

    @Override
    public boolean canHandle(NotificationChannel channel) {
        return channel == NotificationChannel.EMAIL;
    }

    @Override
    public void send(Recipient recipient, NotificationMessage message) {
        MimeMessage mimeMessage = mailSender.createMimeMessage();
        // ... email-specific sending logic
        mailSender.send(mimeMessage);
    }
}

@Component
public class SMSMessageSender implements MessageSender {
    private final TwilioClient twilioClient;

    @Override
    public boolean canHandle(NotificationChannel channel) {
        return channel == NotificationChannel.SMS;
    }

    @Override
    public void send(Recipient recipient, NotificationMessage message) {
        twilioClient.messages.create(recipient.getPhone(), message.getBody());
    }
}

// Adding WhatsApp: create WhatsAppMessageSender implements MessageSender
// → Zero changes to NotificationService (OCP)

// SRP: NotificationService only orchestrates
// DIP: Depends on interfaces, not concrete classes
@Service
public class NotificationService {
    private final List<MessageSender> senders;          // Abstraction
    private final NotificationLog notificationLog;      // Abstraction
    private final RecipientResolver recipientResolver;  // Abstraction

    public NotificationService(List<MessageSender> senders,
                               NotificationLog notificationLog,
                               RecipientResolver recipientResolver) {
        this.senders = senders;
        this.notificationLog = notificationLog;
        this.recipientResolver = recipientResolver;
    }

    public void notify(String userId, NotificationChannel channel, NotificationMessage message) {
        Recipient recipient = recipientResolver.resolve(userId, channel);

        MessageSender sender = senders.stream()
            .filter(s -> s.canHandle(channel))
            .findFirst()
            .orElseThrow(() -> new UnsupportedChannelException(channel));

        try {
            sender.send(recipient, message);
            notificationLog.record(NotificationEvent.success(userId, channel, message));
        } catch (Exception e) {
            notificationLog.record(NotificationEvent.failure(userId, channel, message, e.getMessage()));
            throw new NotificationException("Failed to send " + channel + " notification", e);
        }
    }
}

// Testing: ALL SOLID applied → trivially testable
@Test
void notify_emailChannel_sendsEmailAndLogs() {
    // Arrange
    FakeEmailSender fakeEmail = new FakeEmailSender();
    InMemoryNotificationLog fakeLog = new InMemoryNotificationLog();
    FakeRecipientResolver fakeResolver = new FakeRecipientResolver()
        .withEmail("user-1", "alice@example.com");

    NotificationService service = new NotificationService(
        List.of(fakeEmail), fakeLog, fakeResolver
    );

    // Act
    service.notify("user-1", NotificationChannel.EMAIL,
        NotificationMessage.of("Welcome!", "Your account is ready."));

    // Assert
    assertThat(fakeEmail.getSentMessages()).hasSize(1);
    assertThat(fakeLog.getEvents()).hasSize(1);
    assertThat(fakeLog.getEvents().get(0).getStatus()).isEqualTo(NotificationStatus.SENT);
}
```

### LSP — Correct Substitutable Hierarchy
```java
// LSP-compliant shape hierarchy
// Every shape can compute area and perimeter — behavioral contract is identical

public abstract class Shape {
    public abstract double area();
    public abstract double perimeter();

    // Concrete method that uses abstract — works for ALL shapes
    public double areaToPerimeterRatio() {
        return area() / perimeter();
    }

    public boolean isLargerThan(Shape other) {
        return this.area() > other.area();
    }
}

public class Circle extends Shape {
    private final double radius;

    public Circle(double radius) {
        if (radius <= 0) throw new IllegalArgumentException("Radius must be positive");
        this.radius = radius;
    }

    @Override public double area() { return Math.PI * radius * radius; }
    @Override public double perimeter() { return 2 * Math.PI * radius; }
}

public class Rectangle extends Shape {
    private final double width, height;

    public Rectangle(double width, double height) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("Dimensions must be positive");
        this.width = width;
        this.height = height;
    }

    @Override public double area() { return width * height; }
    @Override public double perimeter() { return 2 * (width + height); }
}

// LSP test: ALL methods in Shape must work correctly for ALL subtypes
@ParameterizedTest
@MethodSource("provideShapes")
void allShapes_arePositive(Shape shape) {
    assertThat(shape.area()).isGreaterThan(0);
    assertThat(shape.perimeter()).isGreaterThan(0);
    assertThat(shape.areaToPerimeterRatio()).isGreaterThan(0);
}

static Stream<Shape> provideShapes() {
    return Stream.of(
        new Circle(5),
        new Rectangle(4, 3),
        new Triangle(3, 4, 5, 6)  // Any new shape added here must pass same tests
    );
}
// Adding Triangle: if it passes provideShapes() test suite, LSP is respected.
```

---

## 13. Common Mistakes

### ⚠️ SRP: "One method per class" misinterpretation
```java
// WRONG: Taken too literally → class explosion
class UserNameGetter { String getName(User user) { return user.name; } }
class UserEmailGetter { String getEmail(User user) { return user.email; } }
class UserAgeGetter { int getAge(User user) { return user.age; } }
// No cohesion benefit — all are "user identity" concern!

// CORRECT: One COHESIVE responsibility
class User {
    String getName() { return name; }
    String getEmail() { return email; }
    int getAge() { return age; }
    // All: "user identity data" — one stakeholder (User domain team)
}
```

### ⚠️ OCP: Using Inheritance Instead of Interfaces
```java
// WRONG: OCP via inheritance — creates Fragile Base Class problem
abstract class Notifier { abstract void notify(User user, String message); }
class EmailNotifier extends Notifier { /* sends email */ }
class SMSNotifier extends Notifier { /* sends SMS */ }
// Problem: adding state to Notifier base class affects all subclasses!

// CORRECT: OCP via interface — no fragile base class
interface Notifier { void notify(Recipient recipient, Message message); }
class EmailNotifier implements Notifier { /* ... */ }
class SMSNotifier implements Notifier { /* ... */ }
// Adding new fields to EmailNotifier doesn't affect SMSNotifier
```

### ⚠️ LSP: Override Throws `UnsupportedOperationException`
```java
// IMMEDIATE LSP RED FLAG:
class ReadOnlyList<E> extends ArrayList<E> {
    @Override public boolean add(E e) {
        throw new UnsupportedOperationException("Read-only!"); // VIOLATION
    }
    @Override public E remove(int i) {
        throw new UnsupportedOperationException("Read-only!"); // VIOLATION
    }
}
// Callers with ArrayList reference: list.add(x) → crashes on ReadOnlyList!
// Fix: don't extend ArrayList. Implement List<E> selectively, or use composition.
```

### ⚠️ DIP: Creating Dependencies Inside the Class
```java
// VIOLATION: Cannot test without real Redis
public class UserCache {
    private RedisClient client = new RedisClient("redis://localhost:6379"); // Hardcoded!

    public Optional<User> getUser(String id) {
        return client.get("user:" + id).map(User::deserialize);
    }
}

// FIX: Accept abstraction via constructor injection
public class UserCache {
    private final CacheClient client; // Interface

    public UserCache(CacheClient client) { this.client = client; } // Injected!

    public Optional<User> getUser(String id) {
        return client.get("user:" + id).map(User::deserialize);
    }
}
// Test: new UserCache(new InMemoryCache())
```

### ⚠️ ISP: Returning to Fat Interface After Refactor
```java
// Common regression: consolidating for "simplicity" back into fat interface
interface DataAccess<T> {
    T findById(Long id);
    List<T> findAll();
    void save(T entity);
    void delete(Long id);
    List<T> search(SearchCriteria criteria);
    void bulkInsert(List<T> entities);          // Not all datastores support bulk
    void executeStoredProcedure(String name);    // Specific to relational DBs!
    void commitTransaction();                    // Not all datastores have transactions!
}
// All repositories now depend on transaction and stored procedure knowledge!

// Fix: Graduated interfaces (like Spring Data)
interface Findable<T, ID> { Optional<T> findById(ID id); }
interface Savable<T> { T save(T entity); }
interface Deletable<T, ID> { void deleteById(ID id); }
interface Searchable<T> { Page<T> search(SearchCriteria criteria, Pageable pageable); }
interface Transactional { void executeInTransaction(Runnable operation); }
```

---

## 14. Best Practices

### Application Guide

| Principle | Apply When | Don't Over-Apply When |
|-----------|-----------|----------------------|
| **SRP** | Class has multiple teams changing it | Simple utility class with cohesive code |
| **OCP** | Type hierarchy is expected to grow | Only ever 1-2 types (YAGNI) |
| **LSP** | Building IS-A hierarchies | Unrelated classes sharing interface (use composition) |
| **ISP** | Interface has 5+ methods with varied usage | Small 1-2 method interface |
| **DIP** | Class needs testing; implementations may vary | Throwaway script; no test needed |

### 🚀 Production Rules
- Apply SRP at multiple levels: method, class, package, microservice
- OCP through interfaces + Strategy/Observer/Factory patterns
- Test LSP by running base class tests against ALL subclasses (parameterized tests)
- ISP: start with small interfaces; let fat interfaces emerge from clear usage patterns, then split
- DIP: constructor injection in Spring (avoid field injection — can't be `final`, hides dependencies)

---

## 15. Comparison Table

### All Five Principles at a Glance

| Principle | Level | Key Mechanism | Violation Smell |
|-----------|-------|--------------|----------------|
| SRP | Class | Extract classes | God class, multiple stakeholders |
| OCP | Behavior | Interfaces, Strategy | `if-else` chain for types |
| LSP | Inheritance | Behavioral subtyping | `UnsupportedOperationException` in override |
| ISP | Interface | Interface splitting | Class stubs unused methods |
| DIP | Dependencies | Constructor injection | `new ConcreteClass()` in business logic |

### SRP vs OCP Interaction
- SRP says "split the class"
- OCP says "don't change the class — extend via new class"
- Together: small, focused classes (SRP) that accept extension via interface (OCP)

---

## 16. Design Pattern Connection

| Pattern | Primary SOLID Principle |
|---------|------------------------|
| Strategy | OCP: new algorithm = new strategy class |
| Observer | OCP: new observer = new implementation; ISP: Observer interface is minimal |
| Factory Method | OCP: new product = new factory; DIP: creator depends on Product abstraction |
| Abstract Factory | DIP: client depends on factory interface, not concrete factories |
| Template Method | OCP: new behavior = new subclass override |
| Decorator | OCP: new capability = new decorator wrapper |
| Composite | ISP: Component interface minimal; both Leaf and Composite implement it |
| Adapter | LSP: Adapter makes incompatible interface substitutable |
| Proxy | SRP: Proxy handles cross-cutting concern; Real Subject handles business logic |
| Repository | DIP: Domain depends on Repository abstraction; DB implements it |

---

## 17. System Design Connection

### SRP at Microservice Level
```
Monolith (SRP violated at service level):
  [UserService: handles Users + Payments + Notifications + Reports]

Microservices (SRP applied at service level):
  [UserService: users only]
  [PaymentService: payments only]
  [NotificationService: notifications only]
  [ReportService: reports only]
  
Each service: one business capability, one team, one deployment unit
```

### OCP at Plugin Architecture Level
```
Core Platform (closed for modification):
  [NotificationPlatform]
    → defines: NotificationChannel interface

Extension Points (open for extension):
  [EmailPlugin] implements NotificationChannel
  [SMSPlugin] implements NotificationChannel
  [PushPlugin] implements NotificationChannel

Adding WhatsApp: deploy new WhatsAppPlugin. Zero core changes.
Used by: IntelliJ plugins, Chrome extensions, WordPress plugins
```

### DIP at Infrastructure Level
```
Clean Architecture:
  [Domain Entities] ← [Use Cases] ← [Interface Adapters] ← [Frameworks/DB]

DIP applied: inner rings depend on INTERFACES defined at inner layers.
  Use Case has: OrderRepository (interface)
  Interface Adapter layer has: MySQLOrderRepository implements OrderRepository

Swap MySQL → PostgreSQL: change only Interface Adapter layer.
Domain and Use Case layers unchanged.
```

---

## 18. Multithreading Connection

### SRP and Thread Safety
```java
// SRP violation: class mixes state management + business logic
// Makes thread safety reasoning difficult
class OrderProcessor {
    private int processedCount = 0; // State

    public void process(Order order) {
        // Business logic
        calculate(order);
        processedCount++; // State mutation — needs synchronization?
    }
}

// SRP applied: separate concerns → clear thread safety reasoning
class OrderProcessingMetrics {
    private final AtomicInteger processedCount = new AtomicInteger(0);
    public void recordProcessed() { processedCount.incrementAndGet(); } // Thread-safe
    public int getCount() { return processedCount.get(); }
}

class OrderCalculator {
    public OrderResult calculate(Order order) { /* Pure function — thread-safe! */ }
}

// Immutable objects (SRP + DIP) → thread-safe by default:
// Each class has one concern → immutable state possible → no synchronization
```

### DIP and Testing Concurrent Code
```java
// DIP enables testing concurrent code without real infrastructure
public class OrderProcessor {
    private final OrderQueue queue;       // Interface — can be InMemory for tests
    private final ProcessingWorker worker; // Interface — can be synchronous for tests

    @Test
    void processOrder_concurrently_noDataLoss() {
        // Inject deterministic fake instead of real async queue
        SynchronousOrderQueue fakeQueue = new SynchronousOrderQueue();
        TrackingWorker fakeWorker = new TrackingWorker();
        OrderProcessor processor = new OrderProcessor(fakeQueue, fakeWorker);

        // Run 100 concurrent submissions
        IntStream.range(0, 100).parallel()
            .mapToObj(i -> new Order("order-" + i))
            .forEach(processor::submit);

        assertThat(fakeWorker.getProcessedCount()).isEqualTo(100);
    }
}
```

---

## 19. Company Interview Perspective

### Google
- "Design a system with OCP so adding a new machine learning model type requires zero changes to the serving infrastructure"
- LSP in distributed systems: service contracts (behavioral guarantees) vs protobuf schema (syntactic compatibility)
- DIP via gRPC: stub interfaces that can point to real service or test double

### Goldman Sachs
- "Our pricing engine has if-else for 20 product types. Restructure using OCP."
- LSP formal verification: behavioral testing for financial instrument pricing models
- DIP in trading: `RiskRepository`, `PositionRepository` interfaces — swap between real-time and historical data sources

### Amazon
- "How does SRP apply to Lambda functions?" (Each Lambda = one operation = SRP)
- DIP in AWS SDK: use client interfaces for unit testing without real AWS calls
- OCP for DynamoDB access patterns: new access pattern = new Query object, not new Repository method

### Microsoft
- C# extension methods and OCP: add methods to existing types without modification
- ISP in WCF/gRPC services: define narrow service contracts per client type
- SOLID in MVVM: ViewModel (SRP), Command pattern (OCP), ICommand (ISP)

### Meta
- Python ABCs and ISP: `Protocol` class for structural typing
- React component design: SRP (one component, one concern), OCP (extensible via props/slots)
- GraphQL schema as ISP: each client queries only the fields it needs

---

## 20. Tricky Interview Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | ⚠️ Is SRP about "one method per class"? | No — about one STAKEHOLDER or cohesive responsibility. A User class with getName/getEmail/getAge is fine (all user identity concern). |
| 2 | Can SOLID principles conflict with each other? | Yes. Strict SRP → many tiny classes (complex navigation). Strict DIP → interface for everything (over-engineering). Context drives balance. |
| 3 | ⚠️ Is every subclass an LSP violation waiting to happen? | No — subclass that only ADDS methods (no overrides that change contract) is always LSP-compliant. |
| 4 | What is the "Fragile Base Class" problem and which principle addresses it? | Parent class change unexpectedly breaks subclasses. Both SRP (overloaded base class) and LSP (behavioral contract violated) address this. |
| 5 | ⚠️ Is Java's Stack (extends Vector) an LSP violation? | Yes — `stack.add(0, element)` inserts at beginning (not stack behavior); `stack.remove(3)` removes arbitrary index. Stack should not be a Vector. |
| 6 | What is the difference between ISP and SRP? | SRP: class level — one reason to change. ISP: interface level — clients only depend on methods they use. Complementary. |
| 7 | ⚠️ Can a private constructor violate DIP? | Not directly. Private constructors (Singleton) can be abstracted behind an interface. But Singleton = global state = often an antipattern for DIP. |
| 8 | What is "Screaming Architecture" and which SOLID principle does it relate to? | Architecture should scream its intent (Robert C. Martin). SRP at package/module level: packages named by business concept, not technical role. |
| 9 | ⚠️ SOLID says depend on abstractions — does that mean every class needs an interface? | No — YAGNI. Create interface when: second implementation exists, testing requires mocking, or API must be stable across compilation boundaries. |
| 10 | What is the "Stable Abstractions Principle"? | As a package becomes more stable, it should become more abstract. Relates to DIP: stable packages hold interfaces; volatile packages hold implementations. |
| 11 | ⚠️ Can an abstract class violate LSP? | Yes — if it defines concrete methods that subclasses must override in incompatible ways. The abstract class sets the behavioral contract; subclasses must honor it. |
| 12 | What is "Shotgun Surgery" and which SOLID principle does it relate to? | One change requires editing many classes. SRP violation: the concern is spread across too many places. |
| 13 | ⚠️ Is a getter/setter for every field a SRP or ISP violation? | Getter-setter-only class = Anemic Domain Model (SRP: behavior separated from data). If the getters/setters are on an interface nobody uses: ISP. |
| 14 | What is the "Open for Extension — by whom?" question? | The client of the class. OCP says: client can extend behavior by passing a new strategy/impl — without modifying the class they use. |
| 15 | How do Java Sealed Classes (Java 17) relate to OCP? | Sealed classes are "closed for extension" — known subtypes. Allows compiler to exhaustively check switch expressions. Tradeoff: not OCP-open, but JIT-optimizable. |

---

## 21. Coding Problems

### Easy — Apply SRP to Invoice Generator
```java
// Problem: Extract concerns into proper SRP-compliant classes
// Original God class:
class InvoiceManager {
    public Invoice create(Order order) { /* DB logic */ }
    public String formatToHTML(Invoice inv) { /* HTML rendering */ }
    public void sendByEmail(Invoice inv, String email) { /* SMTP */ }
    public void archiveToDB(Invoice inv) { /* DB archiving */ }
}

// SRP-applied solution:
public class InvoiceFactory {
    private final InvoiceNumberGenerator numberGen;
    public Invoice create(Order order) {
        return new Invoice(numberGen.next(), order.getItems(), order.getTotal());
    }
}

public class InvoiceRenderer {
    private final TemplateEngine templates;
    public String toHTML(Invoice inv) { return templates.render("invoice.html", inv); }
    public String toPDF(Invoice inv) { return templates.render("invoice.pdf", inv); }
}

public class InvoiceEmailSender {
    private final EmailClient email;
    public void send(Invoice inv, String recipient) {
        email.send(recipient, "Invoice #" + inv.getNumber(), inv.toHTML());
    }
}

public class InvoiceRepository {
    private final DataSource ds;
    public void save(Invoice inv) { /* DB insert */ }
    public Optional<Invoice> findByNumber(String number) { /* DB query */ }
}

// Orchestrator
public class InvoiceService {
    // Constructor injection — DIP compliant
    public void generateAndSend(Order order, String recipientEmail) {
        Invoice invoice = invoiceFactory.create(order);
        invoiceRepository.save(invoice);
        invoiceEmailSender.send(invoice, recipientEmail);
    }
}
```

### Medium — Apply OCP + DIP: Report Exporter
```java
// OCP + DIP: Export to any format without changing ExportService
public interface ReportExporter {
    String getSupportedFormat();
    byte[] export(Report report);
}

public class CSVReportExporter implements ReportExporter {
    @Override public String getSupportedFormat() { return "CSV"; }
    @Override public byte[] export(Report report) {
        StringBuilder sb = new StringBuilder();
        sb.append("Date,Amount,Status\n");
        report.getRows().forEach(row ->
            sb.append(row.getDate()).append(",")
              .append(row.getAmount()).append(",")
              .append(row.getStatus()).append("\n"));
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }
}

public class JSONReportExporter implements ReportExporter {
    private final ObjectMapper mapper;
    @Override public String getSupportedFormat() { return "JSON"; }
    @Override public byte[] export(Report report) {
        try { return mapper.writeValueAsBytes(report); }
        catch (JsonProcessingException e) { throw new ExportException("JSON export failed", e); }
    }
}

// Adding Excel: just implement ReportExporter! Zero changes to ExportService.
public class ExcelReportExporter implements ReportExporter { /* ... */ }

// Service: OCP — closed for modification; DIP — depends on interface
public class ExportService {
    private final Map<String, ReportExporter> exporters;

    public ExportService(List<ReportExporter> exporters) {
        this.exporters = exporters.stream()
            .collect(Collectors.toMap(ReportExporter::getSupportedFormat, e -> e));
    }

    public byte[] export(Report report, String format) {
        return Optional.ofNullable(exporters.get(format.toUpperCase()))
            .map(exporter -> exporter.export(report))
            .orElseThrow(() -> new UnsupportedFormatException("Format not supported: " + format));
    }
}
```

### Hard — All 5 Principles: Event Processing Pipeline
```java
// S: Each class has one responsibility
// O: New event types = new handler, zero changes to pipeline
// L: All EventHandler implementations are substitutable
// I: EventHandler interface is minimal
// D: Pipeline depends on EventHandler interface, not concrete handlers

// ISP: Minimal interface — just what all handlers need
public interface EventHandler {
    boolean canHandle(Event event);
    EventHandlingResult handle(Event event);
}

// SRP: Each handler has one concern
public class OrderCreatedHandler implements EventHandler {
    private final InventoryService inventory;   // DIP: interface
    private final NotificationService notifier; // DIP: interface

    @Override public boolean canHandle(Event event) {
        return event instanceof OrderCreatedEvent;
    }

    @Override public EventHandlingResult handle(Event event) {
        OrderCreatedEvent e = (OrderCreatedEvent) event;
        inventory.reserveItems(e.getOrderItems());     // One concern: inventory
        notifier.notifyOrderCreated(e.getOrderId());   // One concern: notification
        return EventHandlingResult.success();
    }
}

public class PaymentProcessedHandler implements EventHandler {
    private final OrderRepository orders;    // DIP: interface
    private final ShippingService shipping; // DIP: interface

    @Override public boolean canHandle(Event event) {
        return event instanceof PaymentProcessedEvent;
    }

    @Override public EventHandlingResult handle(Event event) {
        PaymentProcessedEvent e = (PaymentProcessedEvent) event;
        orders.markPaymentComplete(e.getOrderId());
        shipping.scheduleShipment(e.getOrderId());
        return EventHandlingResult.success();
    }
}

// OCP: Add new event type → new Handler class. Pipeline unchanged.
public class RefundRequestedHandler implements EventHandler {
    // ...
}

// Pipeline: SRP (just dispatches), OCP (new handler = no pipeline change), DIP (depends on interface)
public class EventProcessingPipeline {
    private final List<EventHandler> handlers; // DIP: interface list
    private final DeadLetterQueue dlq;          // DIP: interface

    public void process(Event event) {
        handlers.stream()
            .filter(h -> h.canHandle(event))
            .findFirst()
            .map(h -> h.handle(event))
            .ifPresentOrElse(
                result -> { if (!result.isSuccess()) dlq.enqueue(event, result.getError()); },
                () -> dlq.enqueue(event, "No handler found for: " + event.getClass())
            );
    }
}

// Test: LSP verified — all handlers substitutable
@ParameterizedTest
@MethodSource("provideHandlersWithMatchingEvents")
void allHandlers_handleMatchingEvent_returnsSuccess(EventHandler handler, Event event) {
    assertThat(handler.canHandle(event)).isTrue();
    EventHandlingResult result = handler.handle(event);
    assertThat(result.isSuccess()).isTrue();
}
```

---

## 22. Revision Sheet

| Principle | One-Line Rule | Violation Smell | Fix |
|-----------|--------------|----------------|-----|
| SRP | One reason to change (one stakeholder) | God class | Extract into focused classes |
| OCP | New behavior = new code, not modified code | `if-else` chain for types | Strategy pattern / interfaces |
| LSP | All subtypes pass base class tests | `UnsupportedOperationException` | Fix hierarchy or use composition |
| ISP | Clients depend only on what they use | Stub methods in implementor | Split interface |
| DIP | Depend on interfaces, inject implementations | `new Concrete()` in business logic | Constructor injection |

### When Each Principle Applies
```
SRP → ask: "Which team/stakeholder would request a change here?"
OCP → ask: "Will I need to modify this file every time a new type is added?"
LSP → ask: "Can I run all tests on subclasses? Does substituting subclass for base break any?"
ISP → ask: "Does every implementor of this interface need ALL these methods?"
DIP → ask: "Can I test this without setting up the full infrastructure?"
```

---

## 23. Flashcards

| Question | Answer |
|----------|--------|
| SOLID acronym? | Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, Dependency Inversion |
| SRP "reason to change"? | A stakeholder who would request a change. Multiple stakeholders = SRP violated. |
| OCP mechanism? | Interfaces + polymorphism. New behavior = new class, not modified class. |
| LSP precondition rule? | Override cannot STRENGTHEN preconditions (cannot accept less) |
| LSP postcondition rule? | Override cannot WEAKEN postconditions (cannot guarantee less) |
| Classic LSP violation? | Square extends Rectangle — setWidth(5),setHeight(4) area = 16 not 20 |
| ISP violation smell? | Class implements interface but throws UnsupportedOperationException |
| DIP rule? | High-level depends on abstractions; low-level implements abstractions |
| DIP vs DI? | DIP = principle (direction). DI = technique (constructor injection). IoC = framework. |
| OCP violation smell? | if-else chain that grows with every new type |
| Which SOLID enables testing? | DIP — inject fakes via interfaces instead of real infrastructure |
| SRP violation smell? | God class. Shotgun surgery (one change, many files). |
| Stable Abstractions Principle? | More stable = more abstract. Volatile = concrete. Relate to DIP. |
| Acyclic Dependencies Principle? | No package dependency cycles. DIP breaks cycles via interface. |
| Can SOLID conflict? | Yes. Strict SRP = class explosion. Strict DIP = over-abstraction. Balance contextually. |
| Java sealed classes and OCP? | Sealed = closed for extension. Trade OCP for exhaustive switch + JIT optimization. |
| ISP in Java stdlib? | Runnable (1 method), Comparable (1 method), Iterable (1 method), Closeable (1 method) |
| Screaming Architecture (SRP)? | Package names reveal domain intent, not technical roles. |
| How to test LSP? | Parameterized tests: run base class test suite against every subclass implementation. |
| Design patterns for OCP? | Strategy, Decorator, Factory Method, Observer, Template Method |

---

## 24. Cheat Sheet

### Top 20 Facts
1. S = One class, one stakeholder, one reason to change
2. O = Add feature by adding class, not editing existing class
3. L = Every subtype passes ALL base class behavioral tests
4. I = Interface should have only what every client needs
5. D = High-level depends on interface; implementation depends on interface too
6. SRP violation: multiple teams ask you to change the same class
7. OCP mechanism: interfaces + Strategy/Factory/Decorator patterns
8. Classic LSP violation: Square extends Rectangle (width/height coupling)
9. LSP check: run base class test suite against all subtypes (parameterized tests)
10. ISP violation: `UnsupportedOperationException` in interface implementation
11. DIP enables testability: interface dependency → inject fake in tests
12. DIP ≠ DI. DI is the technique; DIP is the principle.
13. Spring IoC = automated DI (Dependency Injection) to achieve DIP
14. SOLID principles can conflict; apply contextually (YAGNI for small projects)
15. Anemic Domain Model: SRP violated (behavior separate from data)
16. Shotgun Surgery: SRP violated (one concern spread across many classes)
17. `instanceof` chain: OCP violated (should use polymorphism)
18. Fragile Base Class: LSP/SRP interaction — base class changes break subclasses
19. Java's Stack extends Vector = LSP violation (actual Java stdlib mistake)
20. SOLID at system level: each microservice = one business capability (SRP)

---

## 25. Final Interview Summary

### Night-Before Revision
1. ⭐ S: One class, one reason to change (one stakeholder)
2. ⭐ O: New behavior = new class (interface/Strategy), not modified existing
3. ⭐ L: Subtypes substitutable for base. Preconditions not strengthened. Postconditions not weakened.
4. ⭐ I: Interfaces only contain what every client needs. Split fat interfaces.
5. ⭐ D: High-level → interface ← low-level. Constructor inject implementations.
6. ⭐ Classic violations: God class (S), if-else chain (O), Square/Rectangle (L), UnsupportedOp (I), `new Concrete()` (D)
7. ⭐ DIP enables testing: interface + constructor injection → inject fake
8. ⭐ DIP ≠ DI: DIP is the principle; DI is the technique; IoC Container automates DI
9. ⭐ LSP test: parameterize base class tests over ALL subclasses
10. ⭐ SOLID is a guideline — apply contextually; YAGNI prevents over-engineering
