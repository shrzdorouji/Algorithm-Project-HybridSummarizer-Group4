"""
LLM-Based Abstractive Summarization Module
------------------------------------------
This module defines the role of a Large Language Model (LLM)
as an independent abstractive summarizer.
"""

from typing import Optional
import re


class LLMAbstractiveSummarizer:
    """
    Independent LLM-based abstractive summarizer.
    """

    def __init__(
            self,
            model_name: str = "facebook/bart-large-cnn",  # مدل اینجا تعریف شده
            max_length: int = 150,
            prompt_template: Optional[str] = None,
    ):
        """
        Initialize LLM summarization parameters.
        """
        self.max_length = max_length
        # اضافه کردن {document} برای اینکه متد format کار کند
        self.prompt_template = prompt_template or (
            "Summarize the following document in a concise and coherent manner:\n\n{document}"
        )

        # بارگذاری مدل به صورت محلی
        try:
            from transformers import pipeline
            # حالا model_name دقیقاً از ورودی بالا خوانده می‌شود
            print(f"🔄 Loading LLM model ({model_name}). Please wait...")
            self.summarizer = pipeline("summarization", model=model_name)
            print("✅ LLM Model loaded successfully.")
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, document: str) -> str:
        if document is None or not document.strip():
            return ""
        text = document.strip()
        lines = text.splitlines()
        clean_lines = [line.strip() for line in lines if line.strip()]
        normalized_text = " ".join(clean_lines)
        normalized_text = re.sub(r'\s+', ' ', normalized_text)
        return normalized_text

    # ------------------------------------------------------------------
    # Step 2: Prompt Construction
    # ------------------------------------------------------------------
    def build_prompt(self, document: str) -> str:
        if not document:
            return self.prompt_template
        try:
            # استفاده از تمپلیتی که شامل {document} است
            return self.prompt_template.format(document=document)
        except KeyError:
            return f"{self.prompt_template}\n\n{document}"

    # ------------------------------------------------------------------
    # Step 3: LLM Generation
    # ------------------------------------------------------------------
    def generate_summary(self, prompt: str) -> str:
        if not prompt:
            return ""

        min_len = min(40, self.max_length // 2)

        try:
            outputs = self.summarizer(
                prompt,
                max_length=self.max_length,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            return outputs[0]['summary_text']
        except Exception as e:
            print(f"⚠️ LLM Error: {e}")
            words = prompt.split()
            return " ".join(words[:self.max_length])

    # ------------------------------------------------------------------
    # Main Interface
    # ------------------------------------------------------------------
    def summarize(self, document: str) -> str:
        processed_doc = self.preprocess(document)
        prompt = self.build_prompt(processed_doc)
        summary = self.generate_summary(prompt)
        return summary