import pytest
from src.textrank.textrank import TextRankSummarizer, sentence_segmentation
from src.merge.merge_strategy import HybridMergeSummarizer


# ------------------------------------------------------------------
# ۱. تست‌های واحد بخش TextRank
# ------------------------------------------------------------------

def test_sentence_segmentation_logic():
    """تست صحت عملکرد قطعه‌بندی جملات"""
    text = "AI is the future. It changes everything! Do you agree?"
    sentences = sentence_segmentation(text)
    assert len(sentences) == 3
    assert "future" in sentences[0]
    assert "agree" in sentences[2]


def test_textrank_preprocessing():
    """تست پیش‌پردازش متنی (حذف استاپ‌وردها و ریشه‌یابی)"""
    summarizer = TextRankSummarizer()
    raw_sents = ["The cats are running in the garden."]
    processed = summarizer.advanced_preprocess(raw_sents)
    # انتظار داریم 'the', 'are', 'in' حذف شوند و 'running' ریشه‌یابی شود
    assert "the" not in processed[0].lower()
    assert "cat" in processed[0].lower()


def test_cosine_similarity_sparse():
    """تست محاسبه شباهت کسینوسی برای بردارهای پراکنده"""
    summarizer = TextRankSummarizer()
    v1 = {"ai": 0.5, "future": 0.8}
    v2 = {"ai": 0.5, "past": 0.2}
    # فقط کلمه ai مشترک است: 0.5 * 0.5 = 0.25
    graph = summarizer.build_similarity_graph([v1, v2])
    assert graph[0][1] == pytest.approx(0.25)


# ------------------------------------------------------------------
# ۲. تست‌های واحد بخش Hybrid Merger (قلب پروژه)
# ------------------------------------------------------------------

def test_merger_weight_calculation():
    """تست فرمول ریاضی ترکیب امتیازهای استخراجی و انتزاعی"""
    # تنظیم وزن‌های مساوی
    merger = HybridMergeSummarizer(alpha=0.5, beta=0.5)

    candidates = ["The earth is round."]
    s_llm_list = ["The earth is round."]  # جمله در هر دو لیست است
    tr_scores = {"The earth is round.": 1.0}  # امتیاز کامل از گراف

    weights = merger._compute_final_weights(candidates, s_llm_list, tr_scores)

    # فرمول: (0.5 * 1.0) + (0.5 * 0.9) = 0.95
    assert weights["The earth is round."] == pytest.approx(0.95)


def test_redundancy_filter_threshold():
    """تست حذف جملات تکراری بر اساس آستانه شباهت"""
    # آستانه را کمی پایین می‌آوریم یا جملات را شبیه‌تر می‌کنیم
    merger = HybridMergeSummarizer(sim_threshold=0.6, l_max=2)
    ranked_candidates = [
        "The climate change is a global threat to humanity.",
        "The climate change is a worldwide threat to humans.", # بسیار شبیه به اولی
        "The weather today is sunny."
    ]

    selected = merger._generate_summary_algorithmic(ranked_candidates)

    # جمله دوم باید به دلیل شباهت بالا حذف شود
    assert len(selected) == 2
    assert "climate change" in selected[0]
    assert "weather" in selected[1] # حالا باید جمله دوم لیست، weather باشد

# ------------------------------------------------------------------
# ۳. تست‌های واحد بخش LLM (Logic Only)
# ------------------------------------------------------------------

def test_llm_prompt_building():
    """تست ساخت پرامپت برای مدل پگاسوس بدون لود کردن مدل"""
    from src.llm.llm_integration import LLMAbstractiveSummarizer
    # لود نکردن مدل با Mock کردن یا تست متد مستقل
    summarizer = LLMAbstractiveSummarizer(model_path="dummy")  # مسیر فرضی
    doc = "  This is a messy    text.  "
    processed = summarizer.preprocess(doc)
    prompt = summarizer.build_prompt(processed)

    assert processed == "This is a messy text."
    assert "Summarize and paraphrase" in prompt