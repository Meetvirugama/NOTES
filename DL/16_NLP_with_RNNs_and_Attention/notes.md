# 📚 Chapter 16: Natural Language Processing with RNNs and Attention
> **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron)**

Welcome to Chapter 16. This chapter marks the transition from traditional sequence models to the modern era of Natural Language Processing. We explore text generation, translation networks, and the revolutionary Transformer architecture.

---

## 📝 Detailed Study Notes

Navigate through the detailed modules below. Each module contains deep theoretical explanations, mathematical derivations, Keras implementation details, and interview preparation flashcards.

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| **[Module 01](Detailed_Notes/01_Char_RNNs_and_Text_Generation.md)** | **Char-RNNs & Text Generation** | Stateless vs Stateful RNNs, `tf.data` windowing, Temperature scaling for softmax. |
| **[Module 02](Detailed_Notes/02_Sentiment_Analysis_and_Word_Embeddings.md)** | **Sentiment Analysis & Embeddings** | Word2Vec, GloVe, Tokenization, Masking padding (`mask_zero=True`), Transfer Learning. |
| **[Module 03](Detailed_Notes/03_Encoder_Decoder_and_Translation.md)** | **Encoder-Decoder & Translation** | Neural Machine Translation, Context Vectors, Teacher Forcing, Beam Search. |
| **[Module 04](Detailed_Notes/04_Attention_Mechanisms.md)** | **Attention Mechanisms** | Overcoming the Bottleneck, Bahdanau (Additive), Luong (Multiplicative), Alignment matrices. |
| **[Module 05](Detailed_Notes/05_The_Transformer_Architecture.md)** | **The Transformer Architecture** | Scaled Dot-Product Attention (Q,K,V), Multi-Head Attention, Positional Encoding, Look-Ahead Masking. |
| **[Module 06](Detailed_Notes/06_Recent_Innovations_in_Language_Models.md)** | **Modern Language Models** | Contextual Embeddings (ELMo), Decoder-only generation (GPT), Encoder-only Bidirectional representation (BERT). |

---

## 📈 Learning Objectives
By the end of this chapter, you should be able to:
1. Build and train a Character-RNN to generate synthetic text using temperature-scaled sampling.
2. Implement Word Embeddings and handle variable-length sequences using padding and masking.
3. Design an Encoder-Decoder network for Sequence-to-Sequence tasks like translation.
4. Explain the bottleneck problem and how Attention mechanisms dynamically weigh encoder states.
5. Deconstruct the entire Transformer architecture, including Multi-Head Attention and Positional Encoding.
6. Compare and contrast the architectural differences between GPT (autoregressive) and BERT (bidirectional).

---
*Created for deep-dive studying and interview preparation.*
