# Manual Test Suite — Hybrid Merge Algorithm (Phase‑1)

This document provides **manual, deterministic test cases** for validating the design of the **Hybrid Merge Algorithm** in Phase‑1.

These tests are intended for **design validation**, not executable unit testing.

---

## Assumptions and Global Parameters

Unless otherwise stated, the following parameters are used:
- `alpha = 0.6`
- `beta = 0.4`
- `sim_threshold = 0.7`

Final sentence score is computed as:

$$Score(s) = \alpha \cdot TR\_Score(s) + \beta \cdot LLM\_Score(s)$$

---

## Test Case 1 — Simple Case

### Purpose
Validate basic merge, scoring, and ranking behavior without redundancy.

### Inputs
**S_textrank:**
- S1 = "TextRank is a graph-based ranking algorithm."
- S2 = "It is used for extractive summarization."

**TR_scores:**
- S1: 0.9
- S2: 0.7

**S_llm:**
- L1 = "TextRank ranks sentences using graph structure."

**LLM_scores:**
- L1: 0.8

**L_max = 2**

### Step 1: Merge
Candidates = {S1, S2, L1}

### Step 2: Compute Final Scores
| Sentence | TR | LLM | Final Score |
| :--- | :--- | :--- | :--- |
| S1 | 0.9 | 0.0 | 0.54 |
| S2 | 0.7 | 0.0 | 0.42 |
| L1 | 0.0 | 0.8 | 0.32 |

### Step 3: Ranking
[S1, S2, L1]

### Step 4: Summary Generation
1. Select S1
2. Select S2 (L_max reached)

### Output
**S_hybrid =**
- "TextRank is a graph-based ranking algorithm."
- "It is used for extractive summarization."

---

## Test Case 2 — Medium Case (Balanced Competition)

### Purpose
Evaluate balanced competition between TextRank and LLM outputs.

### Inputs
**S_textrank:**
- S1 = "Hybrid summarization combines extractive methods."
- S2 = "TextRank identifies important sentences."

**TR_scores:**
- S1: 0.8
- S2: 0.6

**S_llm:**
- L1 = "Hybrid summarization improves readability."
- L2 = "TextRank finds key sentences in documents."

**LLM_scores:**
- L1: 0.9
- L2: 0.7

**L_max = 3**

### Final Scores
| Sentence | Final Score |
| :--- | :--- |
| S1 | 0.48 |
| S2 | 0.36 |
| L1 | 0.36 |
| L2 | 0.28 |

### Ranking
[S1, S2, L1, L2] (Note: S2 ranks above L1 if original order is preserved during ties)

### Output
**S_hybrid =**
1. "Hybrid summarization combines extractive methods."
2. "TextRank identifies important sentences."
3. "Hybrid summarization improves readability."

---

## Test Case 3 — Hard Case (Redundancy Handling)

### Purpose
Ensure semantically redundant sentences are filtered correctly.

### Inputs
**S_textrank:**
- S1 = "TextRank ranks sentences based on graph centrality."
- S2 = "It is widely used in extractive summarization."

**TR_scores:**
- S1: 0.85
- S2: 0.65

**S_llm:**
- L1 = "TextRank uses a graph-based ranking approach."
- L2 = "LLMs generate fluent abstractive summaries."

**LLM_scores:**
- L1: 0.9
- L2: 0.8

**L_max = 3**

### Assumed Similarity
`similarity(S1, L1) = 0.85 > sim_threshold`

### Final Scores
| Sentence | Final Score |
| :--- | :--- |
| S1 | 0.51 |
| S2 | 0.39 |
| L1 | 0.36 |
| L2 | 0.32 |

### Selection Process
1. S1 → selected
2. S2 → selected
3. L1 → skipped (redundant with S1)
4. L2 → selected

### Output
**S_hybrid =**
1. "TextRank ranks sentences based on graph centrality."
2. "It is widely used in extractive summarization."
3. "LLMs generate fluent abstractive summaries."

---

## Test Case 4 — Edge Case: Only LLM Output

### Purpose
Validate algorithm behavior when TextRank returns no sentences.

### Inputs
**S_textrank = []**

**S_llm:**
- L1 = "Large language models generate summaries."
- L2 = "They can paraphrase information."

**LLM_scores:**
- L1: 0.9
- L2: 0.7

**L_max = 2**

### Output
**S_hybrid =**
1. "Large language models generate summaries."
2. "They can paraphrase information."

---

## Test Case 5 — Edge Case: Strict Length Constraint (L_max = 1)

### Purpose
Verify strict enforcement of summary length.

### Inputs
**S_textrank:**
- S1 = "TextRank selects key sentences."

**TR_scores:**
- S1: 0.8

**S_llm:**
- L1 = "TextRank is used for extractive summarization."

**LLM_scores:**
- L1: 0.95

**L_max = 1**

### Final Scores
| Sentence | Final Score |
| :--- | :--- |
| S1 | 0.48 |
| L1 | 0.38 |

### Output
**S_hybrid =**
1. "TextRank selects key sentences."

---

## Summary
These five manual test cases validate that the Hybrid Merge Algorithm:
* Correctly merges extractive and abstractive candidates.
* Applies weighted scoring consistently.
* Handles redundancy using similarity thresholds.
* Enforces summary length constraints.