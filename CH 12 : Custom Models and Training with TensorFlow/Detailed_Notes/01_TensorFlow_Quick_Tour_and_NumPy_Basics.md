# 🛠️ Module 1: TensorFlow Quick Tour and NumPy-like Basics
> **Ch. 12 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [TensorFlow Architecture & API Structure](#api-structure)
3. [Tensors and Basic Operations](#tensor-basics)
4. [Tensors and NumPy Interoperability](#numpy-interop)
5. [Type Conversions and Strictness](#type-strictness)
6. [Variables: Managing Mutable State](#variables-state)
7. [Secondary Data Structures (Sparse, Ragged, String Tensors & TensorArrays)](#other-structures)
8. [Common Beginner Mistakes](#mistakes)
9. [Interview Q&A](#interview)
10. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** TensorFlow is a powerhouse library designed for numerical computation, specifically tailored for massive-scale deep learning. Its core API matches NumPy's multidimensional array operations but adds critical features: GPU/TPU acceleration, automatic gradient computation (Autodiff), and static compilation for production environments.

**The Real-World Analogy 🍕:**
Imagine a traditional restaurant kitchen where a single chef (CPU) manually reads recipes (Python code) and chops ingredients one by one. This works fine for small meals but falls apart when serving a banquet of thousands. 
TensorFlow is like an industrial kitchen: it has a conveyor belt system (data pipelines), multiple specialized prep stations (GPUs/TPUs executing mathematical kernels in parallel), and a chef coordinator who writes out an optimized prep list (Computation Graph) beforehand so no time is wasted communicating back and forth.

---

## 🏗️ 1. TensorFlow Architecture & API Structure {#api-structure}

TensorFlow has a layered design. At its lowest levels, it is written in highly optimized C++ code, which manages memory allocation, graph execution, and hardware kernels. The Python layer provides a clean wrapper for these operations.

![TensorFlow API Structure](../Visuals/01_tensorflow_api_structure.png)
> 📊 **Graph 01:** TensorFlow Python API and Execution Hierarchy, showing the progression from high-level tf.keras down to C++ execution kernels and hardware accelerators.

* **C++ Engine**: Executes symbolic computation graphs.
* **Kernels**: Hand-tuned C++/CUDA implementations for specific mathematical operations (e.g., matrix multiplication, convolutions) customized for CPU, GPU, or TPU.
* **Eager Execution**: Introduced in TensorFlow 2.x, operations are executed immediately upon call (just like Python or NumPy). This dramatically simplifies debugging and prototyping compared to the static graph engine of TF 1.x.

---

## 🔍 2. Tensors and Basic Operations {#tensor-basics}

The basic building block in TensorFlow is a **`tf.Tensor`**. A tensor is a multidimensional array (similar to a NumPy `ndarray`) but it is **immutable**; you cannot change its values once created.

### Mathematical Intuition
Tensors represent mathematical operations. A rank-2 tensor represents a matrix $\mathbf{M} \in \mathbb{R}^{m \times n}$. Since mathematical matrices are static values, modifying a tensor in place is mathematically equivalent to redefining the matrix, which requires creating a new matrix memory address.

```python
import tensorflow as tf

# 1. Creating Tensors
t = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("Shape:", t.shape)      # OUTPUT: Shape: (2, 3)
print("Dtype:", t.dtype)      # OUTPUT: Dtype: <dtype: 'float32'>

# 2. Basic Arithmetic (Creates new tensors under the hood)
print("Addition:\n", t + 10)  
# OUTPUT: Addition: tf.Tensor([[11. 12. 13.] [14. 15. 16.]], shape=(2, 3), dtype=float32)

print("Square:\n", tf.square(t))
# OUTPUT: Square: tf.Tensor([[ 1.  4.  9.] [16. 25. 36.]], shape=(2, 3), dtype=float32)

# 3. Matrix Multiplication (tf.matmul or '@' operator)
t_trans = tf.transpose(t)     # shape (3, 2)
product = t @ t_trans         # shape (2, 2)
print("Matrix Product:\n", product)
# OUTPUT: Matrix Product: tf.Tensor([[14. 32.] [32. 77.]], shape=(2, 2), dtype=float32)
```

---

## 🔄 3. Tensors and NumPy Interoperability {#numpy-interop}

TensorFlow play exceptionally well with NumPy.
* You can create tensors directly from NumPy arrays.
* You can pass tensors directly into NumPy functions.
* You can access the underlying NumPy array of a tensor using the `.numpy()` method.

> [!NOTE]
> When converting a NumPy array to a tensor, NumPy's default 64-bit precision (`float64` or `int64`) is preserved. Since deep learning typically runs much faster with 32-bit floats (`float32`), it is recommended to explicitly set the dtype or cast the arrays.

```python
import numpy as np

a = np.array([1., 2., 3.])
t = tf.constant(a) # Preserves float64
print("Tensor from NumPy type:", t.dtype) 
# OUTPUT: Tensor from NumPy type: <dtype: 'float64'>

# Force float32 conversion
t_32 = tf.constant(a, dtype=tf.float32)
print("Explicit float32 type:", t_32.dtype)
# OUTPUT: Explicit float32 type: <dtype: 'float32'>

# Converting back to NumPy
print("Back to NumPy array:", t_32.numpy())
# OUTPUT: Back to NumPy array: [1. 2. 3.]
```

---

## ⚠️ 4. Type Conversions and Strictness {#type-strictness}

Unlike NumPy, **TensorFlow does not perform any automatic type conversions (casting) or broadcasting across different data types.** 
For example, adding a float32 tensor to a float64 tensor, or an int32 tensor to a float32 tensor, will raise a `InvalidArgumentError`. This strictness prevents subtle precision losses or silent bugs during parallel hardware execution.

> [!IMPORTANT]
> To perform operations across different types, you must explicitly cast the tensors using `tf.cast()`.

```python
t1 = tf.constant(3.0, dtype=tf.float32)
t2 = tf.constant(4.0, dtype=tf.float64)

try:
    result = t1 + t2
except tf.errors.InvalidArgumentError as e:
    print("Caught expected type error:", str(e)[:50] + "...")
    # OUTPUT: Caught expected type error: cannot compute AddV2 as input #1(employee_id) was...

# Correct implementation using tf.cast
result = t1 + tf.cast(t2, tf.float32)
print("Result of cast addition:", result)
# OUTPUT: Result of cast addition: tf.Tensor(7.0, shape=(), dtype=float32)
```

---

## 📈 5. Variables: Managing Mutable State {#variables-state}

Because `tf.Tensor` objects are immutable, we cannot use them to represent weights, biases, or optimization states that must change at every training step. For this, TensorFlow provides `tf.Variable`.

![Tensor vs Variable](../Visuals/02_tensor_vs_variable.png)
> 📊 **Graph 02:** Data Mutability comparison. While modifying a `tf.Tensor` in-place fails, `tf.Variable` exposes state modifications like `assign()` and `assign_add()` in-place.

* A `tf.Variable` holds a mutable buffer of values.
* You modify the variable in-place using `.assign()`, `.assign_add()`, or `.assign_sub()`.
* You can also modify specific slices using scatter methods like `scatter_nd_update()`.

```python
v = tf.Variable([[1.0, 2.0], [3.0, 4.0]])

# 1. Modify the entire variable
v.assign(v * 2.0)
print("Assigned multiplication:\n", v.read_value())
# OUTPUT: Assigned multiplication: tf.Tensor([[2. 4.] [6. 8.]], shape=(2, 2), dtype=float32)

# 2. Modify a single cell in place
v[0, 0].assign(42.0)
print("Cell assignment:\n", v.read_value())
# OUTPUT: Cell assignment: tf.Tensor([[42.  4.] [ 6.  8.]], shape=(2, 2), dtype=float32)

# 3. Add in-place
v.assign_add([[1.0, 1.0], [1.0, 1.0]])
print("After assign_add:\n", v.read_value())
# OUTPUT: After assign_add: tf.Tensor([[43.  5.] [ 7.  9.]], shape=(2, 2), dtype=float32)
```

---

## 📦 6. Secondary Data Structures {#other-structures}

While dense tensors are the standard, TensorFlow supports several specialized data structures:

| Data Structure | Description | Common Use Case |
|---|---|---|
| **`tf.SparseTensor`** | Efficiently represents tensors containing mostly zeros. Saves memory and speed. | High-dimensional sparse categorical features (one-hot vectors, NLP bags-of-words). |
| **`tf.ragged.RaggedTensor`** | Tensors containing nested lists of variable lengths along one or more dimensions. | Sequential inputs of varying lengths (e.g. sentences with different word counts). |
| **String Tensors** | Tensors containing byte strings (dtype `tf.string`). | Text processing pipelines, file paths, serialised protocol buffers. |
| **`tf.TensorArray`** | Dynamic array of tensors. Must be initialized with a fixed or dynamic size, but elements must share the same type and shape. | Dynamic loops in Recurrent Neural Networks (RNNs). |

```python
# 1. Sparse Tensor: representing [[0, 10, 0], [0, 0, 20]]
s = tf.SparseTensor(indices=[[0, 1], [1, 2]], values=[10., 20.], dense_shape=[2, 3])
print("Sparse Tensor Dense View:\n", tf.sparse.to_dense(s))
# OUTPUT: Sparse Tensor Dense View: tf.Tensor([[ 0. 10.  0.] [ 0.  0. 20.]], shape=(2, 3), dtype=float32)

# 2. Ragged Tensor: varying length sequences
r = tf.ragged.constant([[1, 2], [3, 4, 5], [6]])
print("Ragged Shape:", r.shape)
# OUTPUT: Ragged Shape: (3, None)

# 3. String Tensor
str_t = tf.constant(["machine", "learning"])
print("String Tensor:", str_t)
# OUTPUT: String Tensor: tf.Tensor([b'machine' b'learning'], shape=(2,), dtype=string)
```

---

## ❌ Common Beginner Mistakes {#mistakes}

### 1. Modifying variables using standard assignment (`=`) ❌
Writing `v = v + 1` rebinds the Python variable name `v` to a **new immutable `tf.Tensor`** object. The original `tf.Variable` object is lost, and it will no longer track weights or be updated by optimizers.
> **Fix:** Use the mutable assignment APIs: `v.assign_add(1)`.

### 2. Creating variables inside a compiled loop or model call ❌
Instantiating variables inside a dynamic loop or inside a `@tf.function` compiled block creates a new node in memory at each step, causing massive memory leaks and compilation crashes.
> **Fix:** Always initialize `tf.Variable` weights at the model construction level (e.g., inside `build()` or `__init__()`), never during training execution.

---

## 🎤 Interview Q&A {#interview}

**Q1: What is the difference between a `tf.Tensor` and a `tf.Variable`?**
> **A:** The primary difference is mutability. A `tf.Tensor` is immutable; once initialized, its values cannot change in memory. Any math operations on it will allocate new tensors. Conversely, a `tf.Variable` is mutable and supports in-place modifications (via `.assign()`, `.assign_add()`, etc.). In models, weights and biases are wrapped in variables so they can be updated in-place by backpropagation optimizers, while data inputs are tensors.

**Q2: Why does TensorFlow raise an error when adding a Float32 and Float64 tensor, whereas NumPy handles this silently?**
> **A:** TensorFlow targets high-performance GPUs and TPUs. On these architectures, explicit data casting is necessary to avoid significant latency penalties (e.g., translating float64 to float32 is expensive). Silent conversions in NumPy can hide bugs and slow down computation graphs. TensorFlow enforces strict type conformity at compile time to guarantee consistent and optimal hardware execution speed.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║             MODULE 1 — TENSORFLOW BASICS FLASH CARD              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  KEY DATA TYPES & MUTABILITY:                                    ║
║  - tf.Tensor: Immutable, constant matrix. Cannot modify.         ║
║  - tf.Variable: Mutable. Modified in-place via:                  ║
║      v.assign(val), v.assign_add(val), v.assign_sub(val)         ║
║                                                                  ║
║  NUMPY INTEROP RULES:                                            ║
║  - tf.constant(np_array) -> Keeps precision (float64 by default) ║
║  - tensor.numpy() -> Casts tensor back to np.ndarray             ║
║  - NO AUTOMATIC TYPE CASTING. Use tf.cast(t, tf.float32)         ║
║                                                                  ║
║  SPECIALTY TENSORS:                                              ║
║  - Ragged: Varying sequences (shape [batch, None])               ║
║  - Sparse: Efficient memory for high-zero matrices               ║
║  - TensorArray: Dynamic write-once read-once loop array          ║
║                                                                  ║
║  COMMON PITFALL:                                                 ║
║  - v = v + 2  <-- DESTROYS tf.Variable, turns it into a Tensor!   ║
║  - Use v.assign_add(2) instead.                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Next Module →** [02_Custom_Losses_and_Components.md](02_Custom_Losses_and_Components.md)
