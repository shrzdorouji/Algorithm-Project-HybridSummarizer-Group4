from typing import List, Dict
from difflib import SequenceMatcher


class HybridMergeSummarizer:
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, sim_threshold: float = 0.6, l_max: int = 4):
        self.alpha = alpha
        self.beta = beta
        self.sim_threshold = sim_threshold  # آستانه شباهت (مثلا ۶۰ درصد)
        self.l_max = l_max
        print("✅ Hybrid Merger initialized (Algorithmic Mode).")

    def merge(self, s_textrank: List[str], s_llm: List[str], tr_scores: Dict[str, float], llm_scores: Dict[str, float],
              original_text: str) -> List[str]:
        # ۱. ترکیب کاندیداها
        candidates = list(dict.fromkeys(s_textrank + s_llm))
        if not candidates: return []

        # ۲. وزن‌دهی
        weighted_scores = self._compute_final_weights(candidates, s_llm, tr_scores, llm_scores)

        # ۳. رتبه‌بندی بر اساس امتیاز
        ranked_candidates = sorted(candidates, key=lambda s: weighted_scores.get(s, 0.0), reverse=True)

        # ۴. انتخاب با حذف تکرار با الگوریتم ریاضی (بدون مدل)
        selected = self._generate_summary_algorithmic(ranked_candidates)

        # ۵. مرتب‌سازی بر اساس ظهور در متن اصلی
        def get_sentence_position(sent):
            pos = original_text.find(sent[:20])
            return pos if pos != -1 else 9999

        return sorted(selected, key=get_sentence_position)

    def _compute_final_weights(self, candidates, s_llm_list, tr_scores, llm_scores):
        final_weights = {}
        max_tr = max(tr_scores.values()) if tr_scores else 1.0
        for s in candidates:
            score_tr = tr_scores.get(s, 0.0) / (max_tr if max_tr > 0 else 1.0)
            score_llm = 0.9 if s in s_llm_list else 0.0
            final_weights[s] = (self.alpha * score_tr) + (self.beta * score_llm)
        return final_weights

    def _generate_summary_algorithmic(self, ranked_candidates: List[str]) -> List[str]:
        selected = []
        for sentence in ranked_candidates:
            if len(selected) >= self.l_max: break

            is_duplicate = False
            for s_selected in selected:
                # محاسبه شباهت با الگوریتم SequenceMatcher
                similarity = SequenceMatcher(None, sentence, s_selected).ratio()
                if similarity > self.sim_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(sentence)
        return selected