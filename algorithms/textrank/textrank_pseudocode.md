# TextRank Pseudocode

## Problem
Given a document D, select the top-k most important sentences using an
optimized graph-based ranking algorithm.

---

## Input
- Document D
- Summary size k
- Damping factor d = 0.85
- Similarity threshold θ
- Maximum iterations Tmax
- Convergence threshold ε

---

## Output
- Extractive summary of k sentences

---

## Algorithm

### Algorithm: Optimized-TextRank(D, k)

#### Step 1: Preprocessing
1. Split document D into sentences  
   S = {s₁, s₂, ..., sₙ}

2. For each sentence sᵢ:
   - Tokenize
   - Remove stop-words
   - Lemmatize (optional)

---

#### Step 2: Sentence Representation
3. Convert each sentence into a vector representation:
   - Use TF-IDF (recommended)

   V[i] = TFIDF(sᵢ)

---

#### Step 3: Sparse Similarity Graph Construction
4. Initialize a sparse weighted undirected graph G(V, E)

5. For each sentence i:
   - Select candidate neighbors (e.g., k-nearest neighbors)

6. For each candidate pair (i, j), i ≠ j:
   - Compute cosine similarity:
     
     wᵢⱼ = cosine(V[i], V[j])

   - If wᵢⱼ ≥ θ:
     - Add edge (i, j) with weight wᵢⱼ

---

#### Step 4: Initialization
7. Initialize TextRank scores:

   TR[i] = 1 / n

---

#### Step 5: Iterative Ranking (Weighted PageRank)
8. Repeat until convergence or Tmax iterations:
```text   
for i = 1 to n do
score = 0
for each j in Neighbors(i) do
score = score + ( w[j][i] / Σ w[j][*] ) × TR[j]
end for
TR_new[i] = (1 − d)/n + d × score
end for
```text
9. Compute convergence:

   diff = max |TR_new[i] − TR[i]|

10. Update scores:

   TR ← TR_new

11. Stop if:

   diff < ε

---

#### Step 6: Sentence Selection
12. Sort sentences by TR score in descending order

13. Select top-k sentences

14. Restore original sentence order

---

#### Step 7: Output
15. Return selected sentences as the extractive summary
