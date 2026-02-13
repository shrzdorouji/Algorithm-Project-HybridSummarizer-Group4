"""
Optimized TextRank Extractive Summarizer
---------------------------------------
This module provides a skeleton implementation of the Optimized TextRank
algorithm for extractive text summarization.

The implementation follows exactly the steps described in:
- textrank_pseudocode.md
- complexity_analysis.md

NOTE:
This is an initial skeleton for Phase-1.
Actual implementations will be completed in later phases.
"""
import re
from typing import List, Dict

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

def _ensure_nltk_resources():
    """
    Ensure required NLTK resources are available (CI-safe).
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab/english",
        "stopwords": "corpora/stopwords",
    }

    for res, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res)

# ✅ Run once at import time
_ensure_nltk_resources()




def sentence_segmentation(document: str) -> List[str]:
    """
    Split the document into raw sentences using NLTK's tokenizer.
    """
    return sent_tokenize(document)


class TextRankSummarizer:
    """
    Optimized TextRank Summarizer (Extractive)
    """

    def __init__(
            self,
            damping_factor: float = 0.85,
            similarity_threshold: float = 0.1,
            max_iterations: int = 100,
            convergence_threshold: float = 1e-4,
            knn: int = 10,
    ):
        """
        Initialize TextRank hyperparameters.

        Parameters
        ----------
        damping_factor : float
            PageRank damping factor (d).
        similarity_threshold : float
            Minimum cosine similarity (θ) to create an edge.
        max_iterations : int
            Maximum number of PageRank iterations (Tmax).
        convergence_threshold : float
            Convergence threshold (ε).
        knn : int
            Number of candidate neighbors per sentence.
        """
        self.d = damping_factor
        self.theta = similarity_threshold
        self.Tmax = max_iterations
        self.epsilon = convergence_threshold
        self.knn = knn

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, document: str) -> List[str]:
        # 1. Sentence splitting (simple regex-based)
        sentences = re.split(r"[.!?]\s+", document.strip())

        # Basic stop-word list (can be expanded later)
        stop_words = {
            "the", "is", "a", "an", "and", "or", "to", "of", "in", "on",
            "for", "with", "as", "by", "at", "from", "that", "this", "it"
        }

        processed_sentences = []

        for sentence in sentences:
            # 2. Lowercase
            sentence = sentence.lower()

            # 3. Remove non-alphabetic characters
            sentence = re.sub(r"[^a-z\s]", "", sentence)

            # 4. Tokenization
            tokens = sentence.split()

            # 5. Stop-word removal
            tokens = [t for t in tokens if t not in stop_words]

            if tokens:
                processed_sentences.append(" ".join(tokens))

        return processed_sentences

    # ------------------------------------------------------------------
    # Step 1: Advanced_Preprocessing
    # -----------------------------------------------------------------

    def advanced_preprocess(self, raw_sentences: List[str]) -> List[str]:
        """
        Advanced preprocessing for Phase 2.
        Input is now a List of raw sentences to ensure 1-to-1 alignment.
        """
        # ۱. بارگذاری منابع (فقط یک‌بار)
        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("punkt")
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))

        stemmer = PorterStemmer()
        processed_sentences: List[str] = []

        # ۲. پردازش روی تک‌تک جملاتی که از قبل جدا شده‌اند
        for sent in raw_sentences:
            # توکن‌بندی کلمات
            tokens = word_tokenize(sent.lower())

            # حذف استاپ‌وردها، ریشه‌یابی و فیلتر کردن کلمات غیرالفبایی
            filtered_tokens = [
                stemmer.stem(t)
                for t in tokens
                if t.isalpha() and t not in stop_words
            ]

            # بازسازی جمله پردازش‌شده
            # نکته مهم: حتی اگر جمله خالی شد، یک رشته خالی اضافه می‌کنیم تا هم‌ترازی لیست به‌هم نخورد
            processed_sentences.append(" ".join(filtered_tokens))

        return processed_sentences

    # ------------------------------------------------------------------
    # Step 2: Sentence Representation (TF-IDF)
    # ------------------------------------------------------------------
    def sentence_representation(
            self, sentences: List[str]
    ) -> List[Dict[str, float]]:
        """
        Convert sentences into sparse TF-IDF vectors.

        Parameters
        ----------
        sentences : List[str]
            Preprocessed sentences.

        Returns
        -------
        List[Dict[str, float]]
            Sparse TF-IDF vectors for each sentence.
        """
        if not sentences:
            return []

        # Initialize TF-IDF Vectorizer (standard, sparse, normalized)
        vectorizer = TfidfVectorizer()

        # Fit and transform sentences into a sparse TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Get vocabulary terms aligned with matrix columns
        feature_names = vectorizer.get_feature_names_out()

        vectors = []
        # Convert each sparse row into a dictionary (non-zero entries only)
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(i)
            vector = {
                feature_names[idx]: float(value)
                for idx, value in zip(row.indices, row.data)
            }
            vectors.append(vector)

        return vectors

    # ------------------------------------------------------------------
    # Step 3: Sparse Similarity Graph Construction
    # ------------------------------------------------------------------
    def build_similarity_graph(
            self, vectors: List[Dict[str, float]]
    ) -> Dict[int, Dict[int, float]]:
        """
        Build a sparse weighted similarity graph using cosine similarity
        and KNN candidate selection using a Min-Heap.
        """
        import heapq
        n = len(vectors)

        if n <= 1:
            return {i: {} for i in range(n)}

        # Syncing with your __init__ parameters
        theta = self.theta
        k = self.knn

        graph: Dict[int, Dict[int, float]] = {i: {} for i in range(n)}

        # Optimized cosine similarity for sparse dicts
        def cosine_sparse(v1: Dict[str, float], v2: Dict[str, float]) -> float:
            if not v1 or not v2:
                return 0.0
            # Iterate over the smaller vector for efficiency
            if len(v1) > len(v2):
                v1, v2 = v2, v1
            return sum(v1[t] * v2[t] for t in v1 if t in v2)

        # Build graph using Step 5 & 6 of pseudocode
        for i in range(n):
            heap = []  # min-heap to keep track of top-k neighbors

            for j in range(n):
                if i == j:
                    continue

                sim = cosine_sparse(vectors[i], vectors[j])

                if sim < theta:
                    continue

                # KNN Optimization using Heap: O(N log K)
                if len(heap) < k:
                    heapq.heappush(heap, (sim, j))
                else:
                    heapq.heappushpop(heap, (sim, j))

            # Add edges to the sparse adjacency list
            for sim, j in heap:
                graph[i][j] = sim
                graph[j][i] = sim

        return graph

    # ------------------------------------------------------------------
    # Step 4 & 5: Initialization and Iterative Ranking (PageRank)
    # ------------------------------------------------------------------
    def rank_sentences(self, graph: Dict[int, Dict[int, float]]) -> List[float]:
        import numpy as np
        n = len(graph)
        if n == 0: return []

        # مقداردهی اولیه با نایمپای (سریع و تمیز)
        TR = np.array([1.0 / n] * n)

        # پیش‌محاسبه مجموع وزن‌ها
        sum_of_weights = np.array([sum(graph[i].values()) for i in range(n)])

        for _ in range(self.Tmax):
            TR_new = np.zeros(n)

            for i in range(n):
                score_sum = 0.0
                # بهینه‌سازی: فقط روی همسایه‌های واقعی جمله i می‌چرخیم (نه کل n جمله)
                for j, weight_ji in graph[i].items():
                    if sum_of_weights[j] > 0:
                        score_sum += (weight_ji / sum_of_weights[j]) * TR[j]

                TR_new[i] = (1.0 - self.d) / n + self.d * score_sum

            # بررسی همگرایی با قدرت نایمپای
            if np.max(np.abs(TR_new - TR)) < self.epsilon:
                TR = TR_new
                break
            TR = TR_new

        return TR.tolist()

    # ------------------------------------------------------------------
    # Step 6 & 7: Sentence Selection and Output
    # ------------------------------------------------------------------

    def summarize(self, document: str, top_k: int) -> str:
        """
        تولید خلاصه با رعایت دو شرط:
        1. حذف جملات تکراری (حتی اگر خروجی کمتر از top_k شود).
        2. حفظ ترتیب اصلی جملات.
        """
        from difflib import SequenceMatcher

        # ۱. قطعه‌بندی جملات
        raw_sentences = sentence_segmentation(document)
        if not raw_sentences:
            return ""

        # ۲. پیش‌پردازش پیشرفته
        cleaned_sentences = self.advanced_preprocess(raw_sentences)

        # ۳. بازنمایی جملات (TF-IDF)
        vectors = self.sentence_representation(cleaned_sentences)

        # ۴. ساخت گراف شباهت
        graph = self.build_similarity_graph(vectors)

        # ۵. رتبه‌بندی (PageRank)
        scores = self.rank_sentences(graph)

        # ۶. انتخاب جملات با فیلتر حذف تکرار (Redundancy Filter)
        indexed_scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

        top_indices = []
        selected_content = []

        for idx, score in indexed_scores:
            # متن جمله برای مقایسه (حذف فضاها و کوچک کردن حروف)
            current_sent = raw_sentences[idx].strip().lower()

            # چک کردن شباهت با جملات قبلاً انتخاب شده
            is_duplicate = False
            for prev_sent in selected_content:
                # اگر شباهت بالای 80% بود (قابل تنظیم)
                if SequenceMatcher(None, current_sent, prev_sent).ratio() > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                top_indices.append(idx)
                selected_content.append(current_sent)

            # توقف در صورت رسیدن به سقف درخواستی
            if len(top_indices) == top_k:
                break

        # ۷. سورت مجدد اندیس‌ها برای حفظ جریان منطقی متن
        top_indices.sort()
        selected_sentences = [raw_sentences[i] for i in top_indices]

        # ۸. خروجی نهایی به صورت رشته متنی
        return " ".join(selected_sentences)