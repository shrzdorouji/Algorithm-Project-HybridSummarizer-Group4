# 🧩 Hybrid Summarizer Project

## 📘 Overview
This project is a **hybrid text summarizer** that combines both **extractive** and **abstractive** summarization methods.
It aims to design, analyze, and implement a pipeline that merges classical algorithms (e.g., TextRank) with modern LLM-based summarization.

---

## 📂 Project Structure
```
Algorithm-Project-HybridSummarizer-Group4/
│
├── data/                     # Input and processed text data
│   ├── raw/                  # Raw input texts
│   │   ├── sample_texts.md
│   │   └── .gitkeep
│   └── processed/            # Cleaned / preprocessed data
│       └── .gitkeep
│
├── docs/                     # Documentation files
│   ├── diagrams/             # Diagrams and visual materials
│   ├── README.md             # Documentation index
│   └── phase-1-report.md     # Phase 1 design/report draft
│
├── src/                      # Source code for modules
│   ├── textrank/             # Extractive summarization module
│   │   ├── textrank.py
│   │   ├── textrank_pseudocode.md
│   │   └── complexity_analysis.md
│   │
│   ├── llm/                  # Abstractive summarization (LLM)
│   │   ├── llm_integration.py
│   │   ├── llm_role.md
│   │   └── __init__.py
│   │
│   └── merge/                # Hybrid merge algorithm module
│       ├── merge_algorithm.md
│       ├── merge_algorithm_examples.md
│       ├── merge_analysis.md
│       ├── merge_strategy.py
│       └── __init__.py
│
├── requirements.txt          # Required dependencies
├── .gitignore
└── README.md                 # Project entry point
```
---
## ⚙️ High-Level Workflow
1. Load and preprocess text data from `data/raw/`
2. Generate an extractive summary using **TextRank**
3. Generate an abstractive summary using an **LLM**
4. Merge both summaries using a **hybrid strategy**
5. Produce the final summary output

---

## 🛠 Installation & Setup
Clone the repository and install dependencies:
```
git clone <repository-url>
cd Algorithm-Project-HybridSummarizer-Group4
pip install -r requirements.txt
```
---
## 🧑‍💻 Current Status
- ✅ Project structure initialized  
- ✅ Phase 1: Algorithm design and documentation  
- ⬜ Phase 2: Implementation    

---

## 🎯 Future Work
- Complete implementation of all modules  
- Add automated tests  
- Evaluate results using standard metrics  
- Prepare final report and presentation  

---

## 👥 Team
Group 4 – Algorithm Design Course  
Semester 1404–1405
