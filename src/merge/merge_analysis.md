# Hybrid Merge Algorithm — Analytical Overview

This document provides a full analytical explanation of the **Hybrid Merge Algorithm**
used in the summarization pipeline to integrate extractive (TextRank) and abstractive (LLM) summaries
into a unified hybrid summary.

---

## 1. Purpose and Context

This algorithm is the **decision-making core** of the summarization system.

It connects:
- **TextRank** → extractive, importance-based representation of key sentences.
- **LLM** → abstractive, semantically rich reformulation of the document.

The algorithm ensures controlled integration, where:
- TextRank provides structure (high-importance nodes)
- LLM contributes semantics and fluency
- The merge algorithm governs the combination logically and explainably.

---

## 2. Inputs and Outputs

| Symbol | Meaning | Type |
|:--|:--|:--|
| `S_textrank` | top-k extractive sentences from TextRank | list[str] |
| `S_llm` | abstractive sentences from LLM | list[str] |
| `L_max` | maximum allowed sentences in the final summary | int |
| `α`, `β` | weighting coefficients (α + β = 1) | float |
| `sim_threshold` | cosine similarity threshold for redundancy | float |
| **Output** | `S_hybrid` – selected hybrid summary sentences | list[str] |

---

## 3. Step-by-Step Analytical Explanation

### Step 1: Merge
Create a unified candidate set:

$$
C = S_{\text{textrank}} \cup S_{\text{llm}}
$$

No filtering yet → ensures fair competition between both sources.

---

### Step 2: Compare & Score
Each candidate sentence ($s \in C$) receives two potential scores:

| Score | Definition | Condition |
|:--|:--|:--|
| `TR_Score(s)` | Normalized TextRank score | if $s \in S_{textrank}$, else 0 |
| `LLM_Score(s)` | Semantic similarity to the original document (embedding-based cosine) | if $s \in S_{llm}$, else 0 |

---

Then compute the **weighted composite score**:

$$
\text{Final_weight}(s) = alpha \cdot \text{TR_Score}(s) + \beta \cdot \text{LLM_Score}(s)
$$

---

Default parameters ensure dominance of extraction structure:

- $\alpha = 0.6$
- $\beta = 0.4$

---

### Step 3: Ranking
Sort all candidates by `Final_weight(s)` descending.  
Higher score = higher inclusion priority.

Time complexity: $O(n \log n)$ for $n = |C|$.

---

### Step 4: Generation (Redundancy-Constrained Selection)

Iterate through the ranked candidates, choose sequentially until:

- Reached the maximum summary size $( L_{max} $)
- Exclude any sentence \( s \) whose embedding-level similarity  
  to any already selected sentence in $( S_{hybrid} $) exceeds `sim_threshold`.

Redundancy pruning uses cosine similarity in embedded vector space.

---

### Step 5: Post-processing (Optional)
- Reorder selected sentences according to their order in the original document  
- Apply light paraphrasing / surface smoothing

---

## 4. Analytical Complexity

Let:

- $( n_t $) = number of TextRank sentences  
- $( n_l $) = number of LLM sentences  
- $( n = n_t + n_l $)
- $( d $) = vector dimension for embeddings  


Then:

| Phase     | Operation                                      | Time Complexity              | Space Complexity |
|:----------|:-----------------------------------------------|:-----------------------------|:-----------------|
| Merge     | simple concatenation                           | O(n)                         | O(n)             |
| Score     | compute sentence embeddings and cosine similarity | O(n · d)                     | O(n · d)         |
| Rank      | sorting by final weights                       | O(n log n)                   | O(n)             |
| Generate  | pairwise similarity checks (filtered)          | O(L_max² · d) (worst case)   | O(L_max)         |

**Total (Expected Practical Runtime):**

$$
O(n \cdot d + n \log n + L_{max}^2 \cdot d)
$$

Given that $L_{max} \ll n$, the dominant term is $O(n \cdot d)$.




---