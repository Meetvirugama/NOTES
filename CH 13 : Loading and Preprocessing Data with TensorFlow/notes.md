# 📚 Chapter 13: Loading and Preprocessing Data with TensorFlow
### Complete Study Notes — Professor Level

> **All 31 pages analyzed. All concepts covered. Zero shortcuts.**

---

## 🖼️ Visual Gallery (Python-Generated Graphs)

> All visuals are in the [`Visuals/`](Visuals/) folder and are embedded in each module.
> Re-generate anytime: `python3 generate_visuals.py`

| # | Graph | Module | File |
|---|-------|--------|------|
| 01 | Dataset Transformations Chaining repeat(3).batch(7) | 1 | [01_dataset_chaining.png](Visuals/01_dataset_chaining.png) |
| 02 | Ingestion & Preprocessing Pipeline with tf.data | 1 | [02_ingestion_pipeline.png](Visuals/02_ingestion_pipeline.png) |
| 03 | Pipeline Execution Timeline with and without Prefetching | 1 | [03_prefetching_timeline.png](Visuals/03_prefetching_timeline.png) |
| 04 | Anatomy of a TFRecord Binary Record | 2 | [04_tfrecord_structure.png](Visuals/04_tfrecord_structure.png) |
| 05 | Hierarchical Structure of the Example Protobuf Schema | 2 | [05_example_protobuf_schema.png](Visuals/05_example_protobuf_schema.png) |
| 06 | SequenceExample Protobuf Schema (Lists of Lists) | 3 | [06_sequence_example_schema.png](Visuals/06_sequence_example_schema.png) |
| 07 | SparseTensor vs. Dense Tensor padded mapping | 2 | [07_sparse_to_dense_tensor.png](Visuals/07_sparse_to_dense_tensor.png) |
| 08 | Categorical Lookup Map & OOV Hashing Buckets | 4 | [08_lookup_table_oov_buckets.png](Visuals/08_lookup_table_oov_buckets.png) |
| 09 | Dense One-Hot Multiplier vs. Direct Index Retrieval | 4 | [09_embedding_lookup_efficiency.png](Visuals/09_embedding_lookup_efficiency.png) |
| 10 | TF Transform Architecture: Resolving Training / Serving Skew | 5 | [10_tf_transform_architecture.png](Visuals/10_tf_transform_architecture.png) |
| 11 | TFDS Loading Pipeline: Downloader to Keras Integration | 5 | [11_tfds_loading_pipeline.png](Visuals/11_tfds_loading_pipeline.png) |
| 12 | ⭐ Master Chapter Summary Dashboard | All | [12_summary_dashboard.png](Visuals/12_summary_dashboard.png) |

---

## 🗺️ Master Index

| Module | Topic | File | Pages Covered |
|--------|-------|------|---------------|
| 01 | Dataset creation (`from_tensor_slices`), chaining (`repeat`, `batch`, `map`), buffer shuffling, parallel interleaving, CSV parsing (`decode_csv`), prefetching and caching performance tuning. | [01_The_Data_API_and_Ingestion_Pipelines.md](Detailed_Notes/01_The_Data_API_and_Ingestion_Pipelines.md) | pp. 413–424 |
| 02 | TFRecord binary format, GZIP compression, Protocol Buffers concepts, `Example` protobuf schema (`BytesList`/`FloatList`/`Int64List`), `parse_single_example` parsing, and sparse-to-dense conversions. | [02_The_TFRecord_Format_and_Protobufs.md](Detailed_Notes/02_The_TFRecord_Format_and_Protobufs.md) | pp. 424–429 |
| 03 | SequenceExample protobuf schema (flat `context` + sequential `feature_lists`), `parse_single_sequence_example` parsing, and `RaggedTensor` mapping. | [03_SequenceExample_and_Nested_Data_Structures.md](Detailed_Notes/03_SequenceExample_and_Nested_Data_Structures.md) | pp. 429–430 |
| 04 | Lambda standardization, custom layer subclassing with `adapt()`, category lookup tables (`StaticVocabularyTable`), OOV bucket hashing, one-hot vectors, manual embedding lookups, and Keras `Embedding` layers. | [04_Preprocessing_Categorical_Features_and_Embeddings.md](Detailed_Notes/04_Preprocessing_Categorical_Features_and_Embeddings.md) | pp. 430–437 |
| 05 | Discretization layers, `PreprocessingStage` pipelines, `TF-IDF` calculation, TF Transform (`tft` pipelines and training/serving skew resolution), and TFDS loading pipelines. | [05_Advanced_Preprocessing_TFT_and_TFDS.md](Detailed_Notes/05_Advanced_Preprocessing_TFT_and_TFDS.md) | pp. 437–443 |

---

## ⚡ One-Page Chapter Summary

### The History of Data Loading in TensorFlow

```
2015: TensorFlow 1.x Queue Runners ────→ Multi-threaded reader queues in C++.
                                         Powerful but extremely complex and prone to deadlock.
2017: TensorFlow 1.4 tf.data API ───────→ Lazy, C++ optimized pipelines ("define-then-run").
                                         Simplifies multithreading, queuing, and prefetching.
2019: TensorFlow 2.x tf.data standard ──→ Replaces old queues entirely.
                                         Integrates with tf.keras, Eager execution, and @tf.function.
```

### Core Architecture: Step Pipeline of a Data Ingestion DAG

```
   DISK FILES (.csv / .tfrecord)
                │
                ▼
        list_files(shuffled)
                │
                ▼
     interleave(TextLineDataset)    <── Cycle & parallel read files
                │
                ▼
        map(preprocess)             <── Parse (decode_csv), stack, and normalize
                │
                ▼
            cache()                 <── Cache to RAM (placed before shuffle/batch)
                │
                ▼
       shuffle(buffer_size)         <── RAM buffer shuffling
                │
                ▼
          batch(size)               <── Group into mini-batch tensors
                │
                ▼
          prefetch(1)               <── CPU prepares batch N+1 while GPU trains on batch N
                │
                ▼
        GPU TRAINING STEP
```

### Core Code Snippet (Baseline Data API & Keras Preprocessing Layer)

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Custom Standardization Preprocessing Layer
class CustomStandardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)
    def call(self, inputs):
        eps = keras.backend.epsilon()
        return (inputs - self.means_) / (self.stds_ + eps)

# 2. Ingestion & Preprocessing Pipeline
X_mean = tf.constant([3.5, 25.0, 5.0, 1.0, 1400.0, 3.0, 35.0, -118.0])
X_std = tf.constant([1.1, 12.0, 1.5, 0.2, 1100.0, 1.0, 2.0, 2.0])

def preprocess(line):
    # Features default to float 0.0, target is required
    defs = [0.0] * 8 + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y

def csv_reader_dataset(filepaths, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=5, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(10000)
    return dataset.batch(batch_size).prefetch(1)

# 3. Model Architecture with Embedding Layer
vocab = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets=2)

regular_inputs = keras.layers.Input(shape=[8], name="num_inputs")
categories_input = keras.layers.Input(shape=[], dtype=tf.string, name="cat_inputs")

cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(categories_input)
cat_embed = keras.layers.Embedding(input_dim=len(vocab) + 2, output_dim=2)(cat_indices)

encoded_inputs = keras.layers.concatenate([regular_inputs, cat_embed])
outputs = keras.layers.Dense(1)(encoded_inputs)

model = keras.models.Model(inputs=[regular_inputs, categories_input], outputs=[outputs])
model.compile(loss="mse", optimizer="adam")
# OUTPUT: Compiled multi-input Keras model ready to take tf.data pipelines.
```

### Preprocessing Strategy Selection Guide

| Criterion | On-the-fly (`map()`) | Keras Layers | TF Transform (TFT) |
|---|---|---|---|
| **Execution Phase** | During training (CPU pipeline) | Inside model graph (CPU/GPU) | Preprocessed once before training |
| **Compute Overhead**| Multi-threaded, runs every epoch | Frozen during training | runs once (Apache Beam) |
| **Training/Serving Skew**| High (risk of client logic mismatch) | Zero (model carries logic) | Zero (model carries TFT Graph) |
| **Scale Capability**| Single-machine CPU | Single-machine CPU/GPU | Distributed clusters (Beam/Spark) |
| **Best Use Case** | Text parsing / image augmentations | Standardization, simple token lookup | Heavy computations, huge cloud datasets |

---

## 🏆 Top 5 Things to Remember

1. **Always Prefetch**: Always append `.prefetch(tf.data.AUTOTUNE)` at the end of your pipeline to overlap CPU data preparation and GPU model training.
2. **Optimize Caching Placement**: Place `.cache()` after expensive preprocessing (like `.map()`) but before shuffling or repeating. Caching after `.repeat()` can load infinite data loops into memory, causing OOM crashes.
3. **Use OOV Buckets**: When mapping categories to indices with `StaticVocabularyTable`, always include out-of-vocabulary buckets (`num_oov_buckets`) to prevent collisions and avoid throwing errors when the model encounters unseen categories in production.
4. **Convert Sparse Tensors from VarLenFeature**: Parsing variable-length records (using `VarLenFeature`) returns `SparseTensor` objects. Remember to run `tf.sparse.to_dense()` or convert them to `RaggedTensor` before passing them to neural network layers.
5. **Use TF Transform to Prevent Skew**: For expensive preprocessing, write the pipeline once in TF Transform. Running it on Apache Beam generates a TF Function with calculated stats embedded as constants, which can be prepended directly to the production model to eliminate training/serving skew.

---

## 🔗 Related Chapters

* **Chapter 12**: [Custom Models and Training with TensorFlow](../CH 12 : Custom Models and Training with TensorFlow/notes.md) - Explores custom loss functions, custom layers, tf.GradientTape, eagerness, and compiling graphs with @tf.function.
* **Chapter 14**: Deep Computer Vision Using Convolutional Neural Networks - Explores Convolutional Neural Networks (CNNs) and processing large image datasets.

---

*Notes created from 31 textbook pages covering pp. 413–443 of Hands-On ML with Scikit-Learn, Keras, and TensorFlow (2nd edition) by Aurélien Géron.*
