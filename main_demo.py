import sys
import os
import numpy as np

# ۱. تنظیم مسیرها و ایمپورت‌ها
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.textrank.textrank import TextRankSummarizer, sentence_segmentation
    from src.llm.llm_integration import LLMAbstractiveSummarizer
    from src.merge.merge_strategy import HybridMergeSummarizer
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)


def run_demo():
    print("====================================================")
    print("✨ HYBRID SUMMARIZER: INTERACTIVE DEMO (GROUP 4) ✨")
    print("====================================================")

    # ۲. بارگذاری اولیه مدل‌ها با پارامترهای دقیق فایل تست
    print("\n🔄 Initializing Engines (Strict Sync with Test Scripts)...")
    try:
        # پارامترها دقیقاً مطابق با تست: similarity_threshold=0.01 و knn=2
        tr_model = TextRankSummarizer(similarity_threshold=0.01, knn=2)
        llm_model = LLMAbstractiveSummarizer(model_path="./my_pegasus")
        merger = HybridMergeSummarizer(alpha=0.5, beta=0.5, sim_threshold=0.7)
        print("✅ Systems Synced and Ready!")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    while True:
        print("\n" + "=" * 52)
        print("📝 PASTE YOUR TEXT (Press 'Enter' twice to process):")
        print("   (Type 'exit' to quit)")
        print("-" * 52)

        lines = []
        while True:
            line = input()
            if line.lower() == 'exit':
                print("\n👋 Goodbye!")
                return
            if line == "": break
            lines.append(line)

        input_text = " ".join(lines).strip()
        if not input_text: continue

        print("\n🚀 Processing (Following Test Scenario Steps)...")

        try:
            # --- مرحله ۱: Segmentation ---
            raw_sents = sentence_segmentation(input_text)

            # --- مرحله ۲: Advanced Preprocessing (ریشه‌یابی و حذف استاپ‌ورد استاندارد) ---
            tr_processed = tr_model.advanced_preprocess(raw_sents)

            # --- مرحله ۳: Representation (TF-IDF) ---
            tr_vectors = tr_model.sentence_representation(tr_processed)

            # --- مرحله ۴: Similarity Graph (KNN + Cosine) ---
            tr_graph = tr_model.build_similarity_graph(tr_vectors)

            # --- مرحله ۵: Ranking (PageRank) ---
            tr_scores_list = tr_model.rank_sentences(tr_graph)

            # --- مرحله ۶: Top-k Selection (دقیقاً مشابه منطق تست) ---
            top_k = 3
            # پیدا کردن اندیس‌ها به ترتیب امتیاز (نزولی)
            ranked_indices = np.argsort(tr_scores_list)[::-1]
            top_indices = ranked_indices[:top_k].tolist()

            # --- مرحله ۷: Order Restoration (بازگرداندن به ترتیب متن اصلی) ---
            top_indices_sorted = sorted(top_indices)

            # استخراج جملات نهایی TextRank
            s_textrank = [raw_sents[i] for i in top_indices_sorted]

            # آماده‌سازی برای Merger
            tr_scores_dict = {raw_sents[i]: tr_scores_list[i] for i in range(len(raw_sents))}

            # --- فرآیند Pegasus (Abstractive) ---
            s_llm_raw = llm_model.summarize(input_text)
            s_llm = [s.strip() + "." for s in s_llm_raw.split('.') if len(s.strip()) > 5]
            llm_scores_dict = {s: 0.9 for s in s_llm}

            # --- ادغام نهایی (Hybrid Merge) ---
            final_result = merger.merge(s_textrank, s_llm, tr_scores_dict, llm_scores_dict, input_text)

            # ۴. نمایش گزارش نهایی
            print("\n" + "============================================================")
            print("📊 FINAL HYBRID REPORT (Synced with Test Results)")
            print("============================================================")

            print("\n[1] EXTRACTIVE (TextRank - Preserved Order):")
            for i, sent in enumerate(s_textrank, 1):
                print(f"  {i}. {sent}")

            print("\n[2] ABSTRACTIVE (Pegasus LLM):")
            for i, sent in enumerate(s_llm, 1):
                print(f"  {i}. {sent}")

            print("\n[3] HYBRID FINAL SUMMARY (Weighted & Cleaned):")
            print("------------------------------------------------------------")
            for sent in final_result:
                print(f" ✨ {sent}")
            print("------------------------------------------------------------")

            print(f"📌 Total Summary Sentences: {len(final_result)}")
            print("============================================================\n")

        except Exception as e:
            print(f"❌ Error during execution: {e}")


if __name__ == "__main__":
    run_demo()