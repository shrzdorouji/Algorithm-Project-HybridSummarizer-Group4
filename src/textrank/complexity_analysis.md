# TextRank Complexity Analysis
# Complexity Analysis of Optimized TextRank

This document provides a **complete time and space complexity analysis**
of the **Optimized TextRank** algorithm with sparse vectors and KNN optimization,
including best, average, and worst-case scenarios.

---

## Notation

- `n` : number of sentences in the document  
- `m` : average number of tokens per sentence  
- `V` : vocabulary size (number of unique tokens)  
- `k` : number of candidate neighbors per sentence (k ≪ n in practice)  
- `E` : number of edges in the similarity graph  
- `Tmax` : maximum number of iterations in iterative ranking  
- `I` : actual number of iterations until convergence (≤ Tmax)  
- `avg_nonzero` : average number of non-zero elements per sentence vector  

> Sparse representation stores only **non-zero entries** in TF-IDF vectors, 
> e.g., `{"گربه": 0.85, "سفید": 0.62}` instead of a dense 1000-element array.

---

## Step 1: Preprocessing

- **Operations:** sentence splitting, tokenization, stop-word removal, lemmatization  
- **Time Complexity:** O(n·m) for all cases  
- **Space Complexity:** O(n·m)  

> Each word must be processed at least once.

---

## Step 2: Sentence Representation (TF-IDF)

- **Operations:** build vocabulary, construct TF-IDF vectors  

- **Dense (non-sparse) vectors:**  
  - Each sentence is represented by a vector of length |V|, storing all entries including zeros.  
  - **Time Complexity:** O(n · |V|)  
  - **Space Complexity:** O(n · |V|)  
  - Drawback: Most entries are zero → wasted memory and computation.

- **Sparse vectors:**  
  - Only non-zero entries are stored (words present in the sentence).  
  - Each sentence has approximately `avg_nonzero` non-zero entries.  
  - **Time Complexity:** O(n · avg_nonzero)  
  - **Space Complexity:** O(n · avg_nonzero)  
  - Advantage: Huge memory and computation savings, especially for large documents.  

> Practical example:  
> Dense: `[0, 0.85, 0, 0, 0.62, 0, ...]`  
> Sparse: `{"cat": 0.85, "white": 0.62}`  
> In this example, instead of storing 1000 entries, we only store 2 → 500× memory savings!
---
## Step 3: Sparse Similarity Graph Construction

- **Operations:** select candidate neighbors (KNN), compute cosine similarity, add edges  
- **Time Complexity:**  
  - Best: O(n · k · avg_nonzero) (sparse graph, high threshold θ)  
  - Average: O(n · k · avg_nonzero)  
  - Worst: O(n² · |V|) (dense graph, compare all pairs)  
- **Space Complexity:**  
  - Best / Average: O(n · k)  
  - Worst: O(n²)  

> Using KNN or blocking strategies significantly reduces comparisons.

---

## Step 4: Initialization

- **Operations:** initialize TR[i] = 1/n for each sentence  
- **Time Complexity:** O(n)  
- **Space Complexity:** O(n)

---

## Step 5: Iterative Ranking (Weighted PageRank)

- **Operations:** update TR scores using neighbors until convergence  
- **Time Complexity (per iteration):** O(E)  
- **Total Time Complexity:**  
  - Best: O(n · k) (fast convergence, sparse graph)  
  - Average: O(I · n · k)  
  - Worst: O(Tmax · n²) (dense graph)  
- **Space Complexity:** O(n) (TR and TR_new arrays)

---

## Step 6: Sentence Selection

- **Operations:** sort sentences by TR score, select top-k, restore original order  
- **Time Complexity:**  
  - Best: O(n) (top-k selection with heap)  
  - Average / Worst: O(n log n)  
- **Space Complexity:** O(n)

---

## Step 7: Output

- **Operations:** return top-k sentences  
- **Time/Space Complexity:** O(k) ≈ O(n) negligible

---

## Overall Complexity

**Time Complexity**

| Case | Complexity |
|------|-----------|
| Best | O(n·m + n·k) |
| Average | O(n·m + n·k·avg_nonzero + I·n·k + n log n) |
| Worst | O(n²·V + Tmax·n²) |

**Space Complexity**

| Case | Complexity |
|------|-----------|
| Best | O(n·avg_nonzero + n·k) |
| Average | O(n·avg_nonzero + n·k) |
| Worst | O(n·V + n²) |

---

## Key Observations

1. **Sparse TF-IDF vectors** → huge memory and time savings.  
2. **Graph construction** is the main bottleneck; KNN and θ reduce edges dramatically.  
3. Iterative ranking is efficient on sparse graphs.  
4. Dimensionality reduction (e.g., SVD) can further reduce |V|.  
5. Top-k selection can be optimized with a heap → O(n) best case.  

---

## Comparison with Classical TextRank

| Version | Graph Construction | Iterative Ranking | Notes |
|---------|-----------------|-----------------|-------|
| Classical | O(n²) | O(T · n²) | Dense graph, high memory |
| Optimized | O(n · k · avg_nonzero) | O(T · n · k) | Sparse vectors, memory-efficient |
