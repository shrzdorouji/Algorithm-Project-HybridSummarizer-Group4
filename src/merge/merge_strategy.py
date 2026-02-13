from typing import List, Dict
from difflib import SequenceMatcher


class HybridMergeSummarizer:
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, sim_threshold: float = 0.6, l_max: int = 4):
        self.alpha = alpha
        self.beta = beta
        self.sim_threshold = sim_threshold
        self.l_max = l_max
        print(f"✅ Hybrid Merger initialized (Alpha={alpha}, Beta={beta})")

    def merge(self, s_textrank: List[str], s_llm: List[str], tr_scores: Dict[str, float], llm_scores: Dict[str, float],
              original_text: str) -> List[str]:

        # ۱. فیلتر کردن جملات ناقص Pegasus
        valid_llm_sents = []
        for sent in s_llm:
            s_strip = sent.strip()
            if len(s_strip.split()) < 5:
                continue
            # پذیرش جملاتی که با نقطه، سوال یا علامت تعجب تمام می‌شوند
            if not any(s_strip.endswith(p) for p in ['.', '!', '?']):
                continue
            valid_llm_sents.append(s_strip)

        # ۲. ترکیب کاندیداها (حذف تکرار دقیق)
        candidates = list(dict.fromkeys(s_textrank + valid_llm_sents))
        if not candidates: return []

        # ۳. وزن‌دهی هوشمند با قابلیت تطابق رشته‌ای (String Normalization)
        weighted_scores = self._compute_final_weights(candidates, valid_llm_sents, tr_scores)

        # ۴. رتبه‌بندی بر اساس امتیاز نهایی
        ranked_candidates = sorted(candidates, key=lambda s: weighted_scores.get(s, 0.0), reverse=True)

        # ۵. انتخاب با حذف جملات مشابه (Redundancy Filter)
        selected = self._generate_summary_algorithmic(ranked_candidates)

        # ۶. مرتب‌سازی بر اساس ظهور در متن اصلی برای حفظ جریان منطقی
        def get_sentence_position(sent):
            # جستجوی ۲۰ کاراکتر اول برای یافتن جایگاه در متن اصلی
            pos = original_text.find(sent[:20])
            return pos if pos != -1 else 9999

        return sorted(selected, key=get_sentence_position)

    def _compute_final_weights(self, candidates, s_llm_list, tr_scores):
        final_weights = {}

        # نرمال‌سازی لیست LLM برای جستجوی دقیق‌تر
        # حذف نقطه و فاصله‌های اضافی و کوچک کردن حروف
        clean_llm = [s.strip().lower().rstrip('.') for s in s_llm_list]

        max_tr = max(tr_scores.values()) if tr_scores else 1.0

        for s in candidates:
            s_clean = s.strip().lower().rstrip('.')

            # امتیاز بخش استخراجی (نرمال شده بین 0 و 1)
            raw_tr = tr_scores.get(s, 0.0)
            score_tr = raw_tr / max_tr if max_tr > 0 else 0.0

            # امتیاز بخش انتزاعی
            # اگر جمله (بدون در نظر گرفتن نقطه آخر) در لیست Pegasus باشد، امتیاز می‌گیرد
            score_llm = 0.9 if s_clean in clean_llm else 0.0

            # فرمول ترکیب امتیازها
            final_weights[s] = (self.alpha * score_tr) + (self.beta * score_llm)

        return final_weights

    def _generate_summary_algorithmic(self, ranked_candidates: List[str]) -> List[str]:
        selected = []
        for sentence in ranked_candidates:
            if len(selected) >= self.l_max:
                break

            is_duplicate = False
            for s_selected in selected:
                # استفاده از SequenceMatcher برای تشخیص جملات هم‌معنی اما با کلمات متفاوت
                similarity = SequenceMatcher(None, sentence, s_selected).ratio()
                if similarity > self.sim_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(sentence)
        return selected