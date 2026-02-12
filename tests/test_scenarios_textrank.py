import sys
import os
import pytest
import nltk

# اضافه کردن مسیر پروژه برای شناسایی ماژول textrank
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.textrank.textrank import TextRankSummarizer, sentence_segmentation


def test_summarize_deep_trace():
    """
    تست مرحله‌به‌مرحله برای مشاهده خروجی‌های میانی الگوریتم
    """
    # تنظیم پارامترها (k کوچک برای گراف خلوت‌تر در تست)
    summarizer = TextRankSummarizer(similarity_threshold=0.01, knn=2)

    document = (
        "Artificial Intelligence is a transformative technology. "
        "AI models can solve complex problems efficiently. "
        "Machine learning is a subset of artificial intelligence. "
        "The sun rises in the east every morning. "
        "Future AI systems will change how we work."
    )

    print("\n" + "=" * 50)
    print("🔍 شروع ردیابی مرحله‌به‌مرحله (Deep Trace)")
    print("=" * 50)

    # --- Step 1: Segmentation ---
    raw_sents = sentence_segmentation(document)
    print(f"\n[Step 1] Segmentation:")
    print(f"   - Total sentences: {len(raw_sents)}")
    for i, s in enumerate(raw_sents):
        print(f"   {i}: {s}")

    # --- Step 1.5: Advanced Preprocessing ---
    cleaned_sents = summarizer.advanced_preprocess(raw_sents)
    print(f"\n[Step 1.5] Preprocessing (Cleaned & Stemmed):")
    for i, s in enumerate(cleaned_sents):
        print(f"   {i}: {s}")

    # --- Step 2: Sentence Representation ---
    vectors = summarizer.sentence_representation(cleaned_sents)
    print(f"\n[Step 2] Representation (Sample Terms):")
    if vectors:
        print(f"   - Sent 0 keywords: {list(vectors[0].keys())}")
        print(f"   - Sent 2 keywords: {list(vectors[2].keys())}")

    # --- Step 3: Graph Construction ---
    graph = summarizer.build_similarity_graph(vectors)
    print(f"\n[Step 3] Similarity Graph (Edges):")
    for node, neighbors in graph.items():
        print(f"   - Sentence {node} connected to: {list(neighbors.keys())}")

    # --- Step 4 & 5: Ranking ---
    scores = summarizer.rank_sentences(graph)
    print(f"\n[Step 4 & 5] Final Scores (PageRank):")
    for i, score in enumerate(scores):
        bar = "█" * int(score * 200)  # نمایش بصری امتیاز
        print(f"   Sent {i}: {score:.4f} {bar}")

    # --- Step 6 & 7: Final Summary ---
    result = summarizer.summarize(document, top_k=2)
    print(f"\n[Step 6 & 7] Final Summary (Top 2):")
    print(f"   >>> {result}")

    # Assertions
    assert len(raw_sents) == 5
    assert scores[3] < max(scores), "جمله نویز نباید بالاترین امتیاز را بگیرد"