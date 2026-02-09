# Phase 1 Technical Report: Hybrid Summarization System
**Algorithm Design Course** | **Group 4** | **Semester 1404-1405**

---

## 1. Formal Problem Definition
The primary goal of this project is to develop an algorithmic engine that transforms a long input document into a concise and semantically accurate summary.

* **Input**: A raw text document $D$ consisting of $n$ sentences and a target summary length $L_{max}$.
* **Output**: A hybrid summary $S_h$ containing at most $L_{max}$ sentences, prioritized by significance and readability.
* **Example**: Converting a 1,000-word technical article into a 5-sentence paragraph that captures core conclusions.

---

## 2. Problem Analysis
* **Nature of the Problem**: This is a hybrid optimization task. It utilizes a **Graph-based** approach (TextRank) for structural centrality and **LLM-based** synthesis for abstractive richness.
* **Complexity**: The main challenge is the integration of diverse summary sources while enforcing a fixed length constraint and preventing semantic redundancy.
* **LLM Role (Oracle)**: The LLM serves as an **independent abstractive summarizer**. It generates a standalone summary that provides semantic depth which extractive methods cannot reach.

---

## 3. Algorithm Design
The system employs a Parallel Pipeline Architecture converging into a **Hybrid Merge Engine**.

### 3.1. Extractive Path: Optimized TextRank
1. **Preprocessing**: Tokenization and cleaning of document $D$.
2. **Representation**: Converting sentences into **Sparse TF-IDF** vectors to save memory.
3. **Ranking**: Iterative score calculation using a damping factor ($d=0.85$) until convergence.

### 3.2. Abstractive Path: LLM Oracle
* **Process**: The LLM receives raw text and a prompt, generating a fluent, coherent summary $S_{llm}$.
* **Independence**: The LLM summary is generated directly from the original document and is independent of the TextRank scores.

### 3.3. Decision Core: Hybrid Merge Algorithm
The algorithm merges candidates and ranks them using a weighted composite score:
$$Score(s) = \alpha \cdot TR\_Score(s) + \beta \cdot LLM\_Score(s)$$
* **Redundancy Management**: Pairwise similarity checks are performed in embedding space to exclude redundant sentences.

---

## 4. Analytical Complexity
Based on the **Hybrid Merge Algorithm Analysis**, the complexity breakdown is as follows:

| Phase     | Operation                                      | Time Complexity              | Space Complexity |
|:----------|:-----------------------------------------------|:-----------------------------|:-----------------|
| **Merge** | Simple concatenation                           | $O(n)$                       | $O(n)$           |
| **Score** | Compute sentence embeddings and similarity     | $O(n \cdot d)$               | $O(n \cdot d)$   |
| **Rank** | Sorting by final weights                       | $O(n \log n)$                 | $O(n)$           |
| **Generate**| Pairwise similarity checks (worst case)        | $O(L_{max}^2 \cdot d)$       | $O(L_{max})$     |

**Total Practical Runtime**: $O(n \cdot d + n \log n + L_{max}^2 \cdot d)$.

---

## 5. Manual Test Cases
The design was validated using five deterministic scenarios:

| Case | Scenario | Expected Outcome |
|:---|:---|:---|
| **Simple** | Unique sentences from TR and LLM | Merged and ranked by final weight |
| **Medium** | Balanced competition ($\alpha=0.6, \beta=0.4$) | Top scorers from both paths selected |
| **Hard** | Redundant $S_1$ and $L_1$ sentences | Redundancy filter skips $L_1$ via similarity check |
| **Edge 1** | Only LLM output available | System successfully generates a pure LLM summary |
| **Edge 2** | Strict length constraint ($L_{max} = 1$) | Only the single highest-priority sentence is returned |

---

## 6. System Diagram
The diagram below shows the flow from document ingestion to the final hybrid summary:
![System Architecture Diagram](/docs/diagrams/phase-1_diagram.png)

* **Classic Algorithm**: Path 1 (TextRank module).
* **LLM Role**: Independent Oracle providing semantic candidates.
* **Data Flow**: Parallel feature extraction followed by similarity-constrained ranking.