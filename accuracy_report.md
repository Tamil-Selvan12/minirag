# Mini RAG System Accuracy & Evaluation Benchmark Report

**Date/Time:** 2026-09-17 20:48:26  
**Document:** `Artificial Intelligence, Machine Learning, and Deep Learning.pdf`  
**Chunks in Index:** 1,308 chunks  
**Embedding Model:** `all-MiniLM-L6-v2` (384-dim)  
**LLM:** `llama3:latest` (via local Ollama)  
**Top K Chunks Retrieved:** 4  

---

## 1. Executive Summary

| Metric | Score | Target | Assessment |
|---|---|---|---|
| **Overall Accuracy / Pass Rate** | **100.0%** | ≥ 80.0% | ✅ Pass |
| **Retrieval Hit Rate @ 4** | **100.0%** | ≥ 85.0% | ✅ Excellent |
| **Mean Reciprocal Rank (MRR)** | **0.917** | ≥ 0.700 | ✅ Excellent |
| **Semantic Similarity (vs Reference)** | **88.4%** | ≥ 70.0% | ✅ Strong |
| **Key Factual Entity Recall** | **78.1%** | ≥ 75.0% | ✅ Good |
| **Faithfulness / Groundedness** | **87.0%** | ≥ 70.0% | ✅ Grounded |
| **Hallucination Resistance (Negative)** | **100.0%** | 100.0% | ✅ Robust |

---

## 2. Detailed Per-Question Results

| ID | Category | Question | Expected Page(s) | Retrieved Pages | Hit? | MRR | Sim | Term Cov | Refusal | Pass |
|---|---|---|---|---|---|---|---|---|---|---|
| `Q1` | AI Foundations | What is the Turing Test? | 24, 25 | 25, 24, 24, 24 | ✅ | 1.00 | 0.83 | 0.75 | ✅ | ✅ PASS |
| `Q2` | AI Foundations | What is the difference between Strong AI and Weak AI? | 23, 24 | 23, 24, 20, 8 | ✅ | 1.00 | 0.86 | 0.50 | ✅ | ✅ PASS |
| `Q3` | Machine Learning Classifiers | What is the kNN algorithm? | 86, 87 | 86, 86, 86, 46 | ✅ | 1.00 | 0.97 | 1.00 | ✅ | ✅ PASS |
| `Q4` | Deep Learning | What is a Perceptron and how is its function defined? | 122, 123 | 123, 129, 123, 122 | ✅ | 1.00 | 0.85 | 0.75 | ✅ | ✅ PASS |
| `Q5` | Deep Learning Architectures | What is BPTT in RNNs? | 149 | 149, 148, 208, 11 | ✅ | 1.00 | 0.85 | 0.75 | ✅ | ✅ PASS |
| `Q6` | Deep Learning Architectures | What is the role of the Max Pooling Layer in CNNs? | 134, 135 | 135, 134, 176, 133 | ✅ | 1.00 | 0.89 | 0.75 | ✅ | ✅ PASS |
| `Q7` | Machine Learning Preprocessing | What is data normalization and how is it calculated? | 54 | 54, 54, 9, 326 | ✅ | 1.00 | 0.96 | 1.00 | ✅ | ✅ PASS |
| `Q8` | Neural Networks | What are common activation functions used in neural networks? | 10, 80, 81, 82, 83, 84, 85, 86 | 99, 103, 82, 101 | ✅ | 0.33 | 0.86 | 0.75 | ✅ | ✅ PASS |
| `Q9` | Out-of-Scope (Hallucination Test) | What does the text say about quantum gravity and string theory? | N/A (OOS) | 265, 187, 3, 187 | ✅ | 1.00 | 0.72 | 1.00 | ✅ | ✅ PASS |
| `Q10` | Out-of-Scope (Hallucination Test) | Who won the FIFA World Cup in 2022? | N/A (OOS) | 328, 300, 29, 109 | ✅ | 1.00 | 0.68 | 1.00 | ✅ | ✅ PASS |

---

## 3. Sample Question & Answer Analysis

### Q1: What is the Turing Test?
- **Category:** AI Foundations
- **Retrieved Pages:** [25, 24, 24, 24]
- **Similarity to Ground Truth:** 82.6%
- **Status:** ✅ Passed

**Generated LLM Response:**
> According to the context, the Turing Test is described on page 24:

"The Turing Test
Alan Turing proposed two imitation games, in which one person or entity behaves as if he were another. In the first game, a person (called an interrogator) is in a room with a curtain that runs across the center of the room. On the other side of the curtain is a person, and the interrogator must determine whether it is a man or a woman. The interrogator (whose gender is irrelevant) accomplishes this task by asking a series of questions."

So, the answer to the question "What is the Turing Test?" is:

The Turing Test is an imitation game proposed by Alan Turing, where an interrogator asks a series of questions to determine whether the person or entity on the other side of a curtain is a man or a woman.

### Q2: What is the difference between Strong AI and Weak AI?
- **Category:** AI Foundations
- **Retrieved Pages:** [23, 24, 20, 8]
- **Similarity to Ground Truth:** 85.9%
- **Status:** ✅ Passed

**Generated LLM Response:**
> According to the context, the difference between Strong AI and Weak AI is that proponents of weak AI measure the success of systems based on their performance alone, regardless of how they are actually solved. On the other hand, proponents of strong AI are concerned with the structure of the systems they build, maintaining that computers can possess intelligence by possessing heuristics, algorithms, and knowledge.

(Cited pages: 23-24)

### Q3: What is the kNN algorithm?
- **Category:** Machine Learning Classifiers
- **Retrieved Pages:** [86, 86, 86, 46]
- **Similarity to Ground Truth:** 97.4%
- **Status:** ✅ Passed

**Generated LLM Response:**
> The kNN (k Nearest Neighbor) algorithm is a classification algorithm. In brief, data points that are near each other are classified as belonging to the same class. When a new point is introduced, it's added to the class of the majority of its nearest neighbor.

Source: Page 86

### Q4: What is a Perceptron and how is its function defined?
- **Category:** Deep Learning
- **Retrieved Pages:** [123, 129, 123, 122]
- **Similarity to Ground Truth:** 84.9%
- **Status:** ✅ Passed

**Generated LLM Response:**
> A Perceptron involves a function f(x) where the following holds:

f(x) = 1 if w*x + b > 0 (otherwise f(x) = 0)

In this definition, w is a vector of weights, x is an input vector, and b is an intercept.

(Cited from Page 123)

### Q9: What does the text say about quantum gravity and string theory?
- **Category:** Out-of-Scope (Hallucination Test)
- **Retrieved Pages:** [265, 187, 3, 187]
- **Similarity to Ground Truth:** 72.1%
- **Status:** ✅ Passed

**Generated LLM Response:**
> The text does not mention quantum gravity or string theory. The provided context is about Artificial Intelligence, Machine Learning, Deep Learning, TensorFlow, and working with strings, but it does not touch on these topics.

Cited pages: None (since the answer is "not mentioned" and no page numbers are relevant).

---

## 4. Strengths & Observations

1. **Dense Semantic Retrieval**: The `all-MiniLM-L6-v2` embedding model with FAISS `IndexFlatL2` achieves high recall for direct conceptual questions and definitions.
2. **Strict Grounding Enforcement**: The system prompt (*'If the answer isn't in the context, say so clearly instead of guessing.'*) effectively prevents hallucinations on out-of-scope queries.
3. **Page Citation Reliability**: Chunks preserve page metadata, allowing the LLM to cite verifiable page numbers.

## 5. Actionable Recommendations for Further Accuracy Improvements

- **Chunk Size Tuning**: Currently `CHUNK_SIZE = 500` characters (~80-100 words), which can occasionally split complex paragraphs or math formulas across chunk boundaries. Increasing to `800-1000` characters with `150` character overlap can preserve more surrounding context.
- **Hybrid Search (BM25 + FAISS)**: Dense vectors capture conceptual similarity, but sparse BM25 keyword matching helps significantly with exact acronyms (e.g. `BPTT`, `kNN`, `ELU`).
- **Reranking**: Adding a lightweight cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) on the top-10 retrieved chunks before passing the top-4 to the LLM can boost MRR and relevance by 10-15%.
- **Temperature Setting**: Explicitly setting temperature `0.1` or `0.0` in the Ollama API call ensures deterministic, strictly factual responses.
