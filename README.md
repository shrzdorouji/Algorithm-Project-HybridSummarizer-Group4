# 🧠 Hybrid Text Summarization System

> A hybrid framework integrating graph-based extractive ranking with transformer-based abstractive modeling.

**Course:** Algorithm Design  
**Academic Year:** 1404–1405  

---

## 📖 Overview

Automatic text summarization aims to condense a document while preserving semantic meaning and logical coherence.

This project implements a hybrid summarization framework composed of:

- TextRank — graph-based extractive summarization  
- Transformer-based LLM — abstractive language modeling  
- Hybrid fusion strategy — weighted integration of both approaches  

The objective is to combine algorithmic efficiency with semantic intelligence.

---

## 🎯 Problem Definition

Given a document:

```
D = { s1, s2, ..., sn }
```

Generate a summary S such that:

- Relevant information is maximized  
- Redundancy is minimized  
- Logical coherence is preserved  
- Computational complexity remains bounded  

The system is analyzed both theoretically (asymptotic complexity) and empirically (runtime behavior).

---

## 🏗 System Architecture

```
Raw Document
     │
     ├── TextRank (Extractive)
     │
     ├── LLM Module (Abstractive)
     │
     └── Hybrid Fusion
            │
       Final Summary
```

---

## ⚙ Algorithmic Components

### 1. TextRank (Extractive)

Pipeline:

1. Sentence segmentation  
2. Advanced preprocessing  
3. TF-IDF vectorization  
4. Sparse similarity graph construction  
5. PageRank ranking  
6. Redundancy filtering  

Complexity:

| Metric | Complexity |
|--------|------------|
| Time   | O(n²L)     |
| Space  | O(nL)      |

Where:

- n = number of sentences  
- L = average sentence length  

Optimizations include sparse graph representation, KNN pruning, and early convergence stopping.

---

### 2. Transformer-Based LLM (Abstractive)

Encoder-decoder architecture with self-attention.

Complexity:

| Metric | Complexity |
|--------|------------|
| Time   | O(n²)      |
| Space  | O(n²)      |

The quadratic behavior originates from the attention mechanism.

---

### 3. Hybrid Fusion Strategy

Final scoring formula:

```
Score = α * TextRank + β * LLM
```

This stage performs:

- Weighted score integration  
- Similarity-based redundancy filtering  
- Logical sentence reordering  

---

## 📊 Complexity Summary

| Component | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| TextRank  | O(n²L)        | O(nL)           |
| LLM       | O(n²)         | O(n²)           |
| Hybrid    | O(n²)         | O(n)            |

---

## 📂 Project Structure

```
Algorithm-Project-HybridSummarizer-Group4/
│
├── data/
│   ├── raw/
│   │   ├── sample_texts.md
│   │   └── .gitkeep
│   │
│   └── processed/
│       └── .gitkeep
│
├── docs/
│   ├── diagrams/
│   └── phase-1-report.md
│
├── src/
│   ├── textrank/
│   │   ├── textrank.py
│   │   ├── textrank_pseudocode.md
│   │   └── complexity_analysis.md
│   │
│   ├── llm/
│   │   ├── llm_integration.py
│   │   ├── llm_role.md
│   │   └── __init__.py
│   │
│   └── merge/
│       ├── merge_strategy.py
│       ├── merge_algorithm.md
│       ├── merge_algorithm_examples.md
│       ├── merge_analysis.md
│       └── __init__.py
│
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

```bash
git clone <repository-url>
cd Algorithm-Project-HybridSummarizer-Group4
pip install -r requirements.txt
```

---

## 🧪 Example Usage

```python
from src.textrank.textrank import TextRankSummarizer

document = "Your input text here..."
summarizer = TextRankSummarizer()
summary = summarizer.summarize(document, top_k=3)
print(summary)
```

---

## 🔮 Future Work

- Sparse attention mechanisms  
- Sentence-BERT similarity  
- GPU acceleration  
- ROUGE-based evaluation  
- Parallel similarity computation  

---

## 👥 Authors

Group 4  
Algorithm Design Course  
Academic Year 1404–1405  
