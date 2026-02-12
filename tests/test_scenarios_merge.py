import sys
import os

# پیدا کردن مسیر اصلی پروژه و اضافه کردن آن به لیست مسیرهای پایتون
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.textrank.textrank import TextRankSummarizer, sentence_segmentation
from src.llm.llm_integration import LLMAbstractiveSummarizer
from src.merge.merge_strategy import HybridMergeSummarizer
import numpy as np

# ۱. متن تست طولانی و استاندارد
text = """
Climate change remains the most significant threat to global stability, driving extreme weather events and rising sea levels. To combat this crisis, the transition to renewable energy sources like solar and wind power is no longer optional but a necessity for survival. International agreements, such as the Paris Accord, aim to limit global warming to below 2 degrees Celsius compared to pre-industrial levels. However, many developing nations struggle with the high initial costs of green technology, requiring financial support from wealthier countries to achieve their climate goals. Experts suggest that a combination of government policy and private innovation will be crucial in accelerating this global transition toward a carbon-neutral future.
"""

print("🚀 Starting Hybrid Summarization Pipeline...\n")

# ۲. اجرای TextRank (استخراجی)
tr_model = TextRankSummarizer()
raw_sents = sentence_segmentation(text)
tr_processed = tr_model.advanced_preprocess(raw_sents)
tr_vectors = tr_model.sentence_representation(tr_processed)
tr_graph = tr_model.build_similarity_graph(tr_vectors)
tr_scores_list = tr_model.rank_sentences(tr_graph)
tr_scores_dict = {raw_sents[i]: tr_scores_list[i] for i in range(len(raw_sents))}
s_textrank = [raw_sents[i] for i in np.argsort(tr_scores_list)[-3:]]

# ۳. اجرای Pegasus (انتزاعی)
llm_model = LLMAbstractiveSummarizer(model_path="./my_pegasus")
s_llm_raw = llm_model.summarize(text)
s_llm = [s.strip() + "." for s in s_llm_raw.split('.') if len(s.strip()) > 5]
llm_scores_dict = {s: 0.9 for s in s_llm}

# ۴. ادغام نهایی (با حفظ ترتیب)
merger = HybridMergeSummarizer(alpha=0.5, beta=0.5, sim_threshold=0.7)
final_result = merger.merge(s_textrank, s_llm, tr_scores_dict, llm_scores_dict, text)

print("\n" + "="*60)
print("📊 FINAL HYBRID REPORT")
print("="*60)
print(f"\n[1] EXTRACTIVE (TextRank):\n--- {' '.join(s_textrank)}")
print(f"\n[2] ABSTRACTIVE (Pegasus LLM):\n--- {' '.join(s_llm)}")
print(f"\n[3] HYBRID FINAL SUMMARY (Ordered):\n✨ {' '.join(final_result)}")
print("\n" + "="*60)