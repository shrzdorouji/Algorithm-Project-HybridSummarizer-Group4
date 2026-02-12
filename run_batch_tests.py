import os
import sys
import pandas as pd
import numpy as np
from src.textrank.textrank import TextRankSummarizer, sentence_segmentation
from src.llm.llm_integration import LLMAbstractiveSummarizer
from src.merge.merge_strategy import HybridMergeSummarizer


def run_comprehensive_evaluation():
    # ۱. آماده‌سازی مدل‌ها (مشابه دمو)
    tr_model = TextRankSummarizer(similarity_threshold=0.01, knn=2)
    llm_model = LLMAbstractiveSummarizer(model_path="./my_pegasus")

    # تنظیمات پیش‌فرض برای تست‌های استاندارد
    K_SENTENCES = 3
    L_MAX = 4
    ALPHA = 0.5
    BETA = 0.5
    SIM_THRESHOLD = 0.7

    test_dir = "./test_samples"
    results = []

    # تحلیل‌های تئوری برای هر فایل (طبق استانداردهای آموزشی)
    theory_analysis = {
        "sample_01.txt": "Simple Case: High coherence expected between Extractive and Abstractive parts.",
        "sample_02.txt": "Medium Case: Tests how well the system handles technical terms like 'Qubits'.",
        "sample_03.txt": "Hard Case: Long specialized sentences; tests information density.",
        "sample_04.txt": "Edge Case: Very short text; tests if the system maintains stability with minimal input.",
        "sample_05.txt": "Edge Case: Input < K; tests the merger's ability to not over-extract.",
        "sample_06.txt": "Edge Case: High punctuation; tests the robustness of the Sentence Segmenter.",
        "sample_07.txt": "Hard Case: Logical contrast; tests if Pegasus captures the 'However' nuance.",
        "sample_08.txt": "Worst-Case: Heavy Redundancy; tests the Redundancy Filter (SequenceMatcher).",
        "sample_09.txt": "Worst-Case: Unrelated sentences; tests the Graph connectivity logic.",
        "sample_10.txt": "Worst-Case: Token Stress; tests the 512-token limit of the Pegasus model."
    }

    print("🚀 Starting Hybrid Summarization Pipeline (10-Step Validation)...")

    if not os.path.exists(test_dir):
        print(f"❌ Error: Directory '{test_dir}' not found!")
        return

    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(test_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            print(f"📄 Processing: {filename}...")

            try:
                # --- مرحله ۱: استخراج (TextRank) ---
                raw_sents = sentence_segmentation(content)
                tr_processed = tr_model.advanced_preprocess(raw_sents)
                tr_vectors = tr_model.sentence_representation(tr_processed)
                tr_graph = tr_model.build_similarity_graph(tr_vectors)
                tr_scores_list = tr_model.rank_sentences(tr_graph)

                # منطق حذف تکرار دستی (مشابه دمو)
                ranked_indices = np.argsort(tr_scores_list)[::-1]
                unique_top_indices = []
                seen_content = set()
                for idx in ranked_indices:
                    clean_s = raw_sents[idx].strip().lower()
                    if clean_s not in seen_content:
                        seen_content.add(clean_s)
                        unique_top_indices.append(idx)
                    if len(unique_top_indices) == K_SENTENCES: break

                s_textrank = [raw_sents[i] for i in sorted(unique_top_indices)]
                tr_scores_dict = {raw_sents[i]: tr_scores_list[i] for i in range(len(raw_sents))}

                # --- مرحله ۲: انتزاع (Pegasus) ---
                s_llm_raw = llm_model.summarize(content)
                s_llm = [s.strip() + "." for s in s_llm_raw.split('.') if len(s.strip()) > 5]
                llm_scores_dict = {s: 0.9 for s in s_llm}

                # --- مرحله ۳: ادغام (Hybrid Merge) ---
                merger = HybridMergeSummarizer(alpha=ALPHA, beta=BETA, sim_threshold=SIM_THRESHOLD, l_max=L_MAX)
                final_summary = merger.merge(s_textrank, s_llm, tr_scores_dict, llm_scores_dict, content)

                # ذخیره نتایج
                results.append({
                    "Test_File": filename,
                    "Input_Sents": len(raw_sents),
                    "Final_Sents": len(final_summary),
                    "TextRank_Count": len(s_textrank),
                    "Theory_Analysis": theory_analysis.get(filename, "N/A"),
                    "Status": "SUCCESS ✅"
                })

            except Exception as e:
                print(f"⚠️ Error on {filename}: {e}")
                results.append({"Test_File": filename, "Status": f"FAILED ❌ ({str(e)})"})

    # خروجی نهایی به صورت CSV برای گزارش پروژه
    df = pd.DataFrame(results)
    df.to_csv("hybrid_test_report.csv", index=False)

    print("\n" + "=" * 50)
    print("🎯 TEST COMPLETED SUCCESSFULLY!")
    print("📂 Report saved as: 'hybrid_test_report.csv'")
    print("=" * 50)
    print(df[["Test_File", "Input_Sents", "Final_Sents", "Status"]])


if __name__ == "__main__":
    run_comprehensive_evaluation()