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
        "The phenomenon of global climate change, driven primarily by anthropogenic greenhouse gas emissions, represents one of the most formidable challenges to contemporary civilization and the delicate equilibrium of Earth's ecosystems. Over the past century, industrial activities—including the combustion of fossil fuels like coal, oil, and natural gas—have significantly elevated atmospheric concentrations of carbon dioxide and methane. These gases trap thermal energy within the troposphere, leading to a consistent rise in global mean temperatures, a process commonly referred to as global warming. "
        "The ramifications of this temperature increase are multi-faceted and catastrophic. Arctic sea ice is retreating at unprecedented rates, contributing to eustatic sea-level rise that threatens low-lying coastal regions and island nations such as the Maldives and Kiribati. Furthermore, the intensification of the hydrological cycle has resulted in more frequent and severe meteorological events, including category-five hurricanes, prolonged droughts in sub-Saharan Africa, and devastating wildfires in temperate forest biomes. "
        "Concurrently, we are witnessing a global biodiversity crisis often characterized as the sixth mass extinction. The loss of habitat due to agricultural expansion, combined with the shifting climatic envelopes that species must navigate, has pushed thousands of organisms to the brink of extinction. Coral reefs, which serve as vital nurseries for marine life, are undergoing massive bleaching events as oceanic acidity increases due to carbon sequestration by the seas. "
        "Mitigation strategies, such as the transition to renewable energy sources like solar, wind, and geothermal power, are essential but require global political cooperation and massive capital investment. The Paris Agreement aimed to limit the temperature increase to well below 2 degrees Celsius, yet current emission trajectories suggest that more radical systemic changes are necessary to avoid the most dire ecological tipping points. Ultimately, the survival of diverse life forms on this planet depends on our collective ability to restructure our relationship with the environment and move toward a sustainable, circular economy that prioritizes ecological health over indefinite industrial growth. "

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

def test_textrank_deep_trace_similar_sentences():

        summarizer = TextRankSummarizer(similarity_threshold=0.05, knn=3)

        document = (
            "AI improves healthcare outcomes. "  
            "AI improves healthcare significantly. "
            "AI improves healthcare in modern hospitals. "
            "Football is played worldwide."
        )

        print("\n" + "=" * 60)
        print("🧪 TEST 2 — Dense Similar Sentences")
        print("=" * 60)

        raw_sents = sentence_segmentation(document)
        print("\n[Step 1] Segmentation:")
        for i, s in enumerate(raw_sents):
            print(f"{i}: {s}")

        cleaned = summarizer.advanced_preprocess(raw_sents)
        print("\n[Step 1.5] Preprocessing:")
        for i, s in enumerate(cleaned):
            print(f"{i}: {s}")

        vectors = summarizer.sentence_representation(cleaned)
        print("\n[Step 2] Vectors:")
        for i, v in enumerate(vectors):
            print(f"{i}: {list(v.keys())}")

        graph = summarizer.build_similarity_graph(vectors)
        print("\n[Step 3] Graph:")
        for i, edges in graph.items():
            print(f"{i} -> {edges}")

        scores = summarizer.rank_sentences(graph)
        print("\n[Step 4] Scores:")
        for i, s in enumerate(scores):
            print(f"{i}: {s:.4f}")

        summary = summarizer.summarize(document, top_k=2)
        print("\n[Final Summary]")
        print(summary)

def test_textrank_deep_trace_order_preservation_forced():
    summarizer = TextRankSummarizer(similarity_threshold=0.01, knn=2)

    document = (
        "Artificial Intelligence is rapidly transforming the healthcare industry by enabling faster and more accurate diagnoses. "
        "Medical professionals are now using deep learning algorithms to detect diseases like cancer from radiological images with higher precision than ever before. "
        "However, the integration of AI in clinics raises significant ethical questions regarding patient privacy and the black box nature of algorithmic decision-making. "
        "To address these concerns, global health organizations are drafting new frameworks to ensure transparency and accountability in medical AI. "
        "Despite these challenges, the potential to save millions of lives through early intervention makes AI an indispensable tool for future medicine. "
    )

    print("\n" + "="*70)
    print("🧪 TEST 3 — Order Preservation (FORCED DEEP TRACE)")
    print("="*70)

    # ------------------------------------------------------------------
    # Step 1: Segmentation
    # ------------------------------------------------------------------
    raw_sents = sentence_segmentation(document)
    print("\n[Step 1] Segmentation (Original Order):")
    for i, s in enumerate(raw_sents):
        print(f"{i}: {s}")

    # ------------------------------------------------------------------
    # Step 2: Preprocessing
    # ------------------------------------------------------------------
    cleaned = summarizer.advanced_preprocess(raw_sents)
    print("\n[Step 2] Preprocessing:")
    for i, s in enumerate(cleaned):
        print(f"{i}: {s}")

    # ------------------------------------------------------------------
    # Step 3: Sentence Representation
    # ------------------------------------------------------------------
    vectors = summarizer.sentence_representation(cleaned)
    print("\n[Step 3] Sentence Vectors:")
    for i, v in enumerate(vectors):
        print(f"{i}: {list(v.keys())}")

    # ------------------------------------------------------------------
    # Step 4: Similarity Graph
    # ------------------------------------------------------------------
    graph = summarizer.build_similarity_graph(vectors)
    print("\n[Step 4] Similarity Graph:")
    for i, edges in graph.items():
        print(f"{i} -> {edges}")

    # ------------------------------------------------------------------
    # Step 5: PageRank
    # ------------------------------------------------------------------
    scores = summarizer.rank_sentences(graph)
    print("\n[Step 5] PageRank Scores:")
    for i, score in enumerate(scores):
        print(f"{i}: {score:.6f}")

    # ------------------------------------------------------------------
    # Step 6: Top-k Selection (BEFORE order fix)
    # ------------------------------------------------------------------
    top_k = 3
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    top_indices = ranked_indices[:top_k]

    print("\n[Step 6] Top-k by SCORE (Order-Breaking Stage):")
    print("Ranked indices by score:", ranked_indices)
    print("Selected top-k indices:", top_indices)

    # ------------------------------------------------------------------
    # Step 7: Order Preservation (CRITICAL STEP)
    # ------------------------------------------------------------------
    top_indices_sorted = sorted(top_indices)

    print("\n[Step 7] Top-k AFTER order restoration:")
    print("Sorted indices (original order):", top_indices_sorted)

    # ------------------------------------------------------------------
    # Step 8: Final Summary Construction
    # ------------------------------------------------------------------
    final_summary = " ".join(raw_sents[i] for i in top_indices_sorted)

    print("\n[Step 8] Final Summary:")
    print(final_summary)

    # Sanity only
    assert len(final_summary) > 0

def test_textrank_deep_trace_sparse_graph():

                summarizer = TextRankSummarizer(similarity_threshold=0.2, knn=1)

                document = (
                    "Cats are small animals. "
                    "Quantum physics studies particles. "
                    "Cooking requires ingredients. "
                    "Space exploration is expensive."
                )

                print("\n" + "=" * 60)
                print("🧪 TEST 4 — Sparse Similarity Graph")
                print("=" * 60)

                raw_sents = sentence_segmentation(document)
                print("\n[Step 1] Sentences:")
                for i, s in enumerate(raw_sents):
                    print(f"{i}: {s}")

                cleaned = summarizer.advanced_preprocess(raw_sents)
                vectors = summarizer.sentence_representation(cleaned)
                graph = summarizer.build_similarity_graph(vectors)

                print("\n[Step 3] Graph:")
                for i, edges in graph.items():
                    print(f"{i} -> {edges}")

                scores = summarizer.rank_sentences(graph)
                print("\n[Step 4] Scores:")
                for i, s in enumerate(scores):
                    print(f"{i}: {s:.4f}")

                summary = summarizer.summarize(document, top_k=2)
                print("\n[Final Summary]")
                print(summary)


