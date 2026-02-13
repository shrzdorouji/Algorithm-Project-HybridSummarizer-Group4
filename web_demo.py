import streamlit as st
import numpy as np
import sys
import os

# ۱. تنظیم مسیرها
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.textrank.textrank import TextRankSummarizer, sentence_segmentation
    from src.llm.llm_integration import LLMAbstractiveSummarizer
    from src.merge.merge_strategy import HybridMergeSummarizer
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")

# تنظیمات ظاهری صفحه
st.set_page_config(page_title="Hybrid Summarizer - Group 4", page_icon="✨", layout="wide")


# بارگذاری مدل‌های سنگین با قابلیت Cache
@st.cache_resource
def load_heavy_models():
    tr_model = TextRankSummarizer(similarity_threshold=0.01, knn=2)
    llm_model = LLMAbstractiveSummarizer(model_path="./my_pegasus")
    return tr_model, llm_model


st.title("🚀 Hybrid Text Summarization System")
st.markdown("##### AI-Powered Extraction & Abstraction (Dynamic Control)")
st.divider()

# بارگذاری مدل‌های سنگین
tr_model, llm_model = load_heavy_models()

# --- منوی کناری (Sidebar) برای تنظیمات داینامیک ---
st.sidebar.title("🛠️ Summary Configuration")
st.sidebar.markdown("تنظیم وزن مدل‌ها:")

# اسلایدر برای Alpha و Beta
# این مقادیر مستقیماً به کلاس HybridMergeSummarizer تزریق می‌شوند
alpha = st.sidebar.slider("Alpha (Extractive Weight):", 0.0, 1.0, 0.5, 0.05)
beta = st.sidebar.slider("Beta (Abstractive Weight):", 0.0, 1.0, 0.5, 0.05)

st.sidebar.divider()
# اسلایدر برای تعداد جملات استخراجی (K) و حداکثر طول خلاصه (L-max)
top_k = st.sidebar.number_input("Sentences for TextRank (K):", min_value=1, max_value=10, value=3)
l_max = st.sidebar.number_input("Max Final Sentences (L-max):", min_value=1, max_value=10, value=4)

# آستانه شباهت برای حذف تکرار
sim_threshold = st.sidebar.slider("Similarity Threshold:", 0.4, 0.9, 0.7, 0.05)

# ورودی متن
input_text = st.text_area("📄 Paste your long text here:", height=200, placeholder="Enter text to summarize...")

if st.button("✨ Generate Hybrid Summary"):
    if input_text.strip():
        with st.spinner("Analyzing text and generating summary..."):
            try:
                # مقداردهی به Merger با پارامترهای اسلایدر
                # این کار باعث می‌شود مقادیر پیش‌فرض کلاس (0.6 و 0.4) نادیده گرفته شوند
                merger = HybridMergeSummarizer(
                    alpha=alpha,
                    beta=beta,
                    sim_threshold=sim_threshold,
                    l_max=l_max
                )

                # --- مرحله ۱: پردازش TextRank ---
                raw_sents = sentence_segmentation(input_text)
                tr_processed = tr_model.advanced_preprocess(raw_sents)
                tr_vectors = tr_model.sentence_representation(tr_processed)
                tr_graph = tr_model.build_similarity_graph(tr_vectors)
                tr_scores_list = tr_model.rank_sentences(tr_graph)

                # استفاده از مقدار top_k از اینپوت سایدبار
                ranked_indices = sorted(range(len(tr_scores_list)),
                                        key=lambda i: (tr_scores_list[i], -i),
                                        reverse=True)

                seen = set()
                unique_indices = []

                for idx in ranked_indices:
                    sent = raw_sents[idx].strip()
                    if sent not in seen:
                        seen.add(sent)
                        unique_indices.append(idx)
                    if len(unique_indices) == top_k:
                        break

                # حفظ ترتیب متن اصلی
                unique_indices = sorted(unique_indices)
                s_textrank = [raw_sents[i] for i in unique_indices]

                tr_scores_dict = {raw_sents[i]: tr_scores_list[i] for i in range(len(raw_sents))}

                # --- مرحله ۲: پردازش Pegasus ---
                s_llm_raw = llm_model.summarize(input_text)
                s_llm = [s.strip() + "." for s in s_llm_raw.split('.') if len(s.strip()) > 5]
                llm_scores_dict = {s: 0.9 for s in s_llm}

                # --- مرحله ۳: ادغام هیبریدی با پارامترهای جدید ---
                hybrid_results = merger.merge(s_textrank, s_llm, tr_scores_dict, llm_scores_dict, input_text)


                # --- مرحله ۴: اصلاح ترتیب نهایی ---
                def get_original_position(sentence):
                    pos = input_text.find(sentence[:30])
                    return pos if pos != -1 else 999999


                final_ordered_summary = sorted(hybrid_results, key=get_original_position)

                # --- نمایش خروجی ---
                tab1, tab2, tab3 = st.tabs(["🎯 Combined Summary", "🔍 TextRank (Extractive)", "🤖 Pegasus (Abstractive)"])

                with tab1:
                    st.success("### Final Hybrid Summary")
                    for sent in final_ordered_summary:
                        st.markdown(f"- {sent}")

                    st.divider()
                    st.caption(f"Settings used: Alpha={alpha}, Beta={beta}, K={top_k}, L-max={l_max}")

                with tab2:
                    st.write(f"Top {top_k} sentences chosen by TextRank:")
                    for i, sent in enumerate(s_textrank, 1):
                        st.info(f"{i}. {sent}")

                with tab3:
                    st.write("Summary generated by Pegasus LLM:")
                    for i, sent in enumerate(s_llm, 1):
                        st.warning(f"{i}. {sent}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please paste some text first!")

# اطلاعات سایدبار پایینی
st.sidebar.divider()
st.sidebar.markdown("Created by **Group 4**")