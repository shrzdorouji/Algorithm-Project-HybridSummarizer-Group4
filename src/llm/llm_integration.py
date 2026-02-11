"""
LLM-Based Abstractive Summarization Module
------------------------------------------
Model: Pegasus (Distilled) - Fully Offline Version
"""

from typing import Optional
import re
import os


class LLMAbstractiveSummarizer:
    def __init__(
            self,
            # آدرس پوشه‌ای که فایل‌های مدل در آن قرار دارند
            model_path: str = "./my_pegasus",
            max_length: int = 150,
            prompt_template: Optional[str] = None,
    ):
        self.max_length = max_length
        self.prompt_template = prompt_template or "{document}"

        try:
            # وارد کردن مستقیم کلاس‌های مخصوص پگاسوس برای پایداری بیشتر
            from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration

            print(f"🔄 Loading Pegasus from local directory: {model_path}")

            # بارگذاری توکنایزر و مدل مستقیماً از پوشه ساخته شده توسط شما
            tokenizer = PegasusTokenizer.from_pretrained(model_path)
            model = PegasusForConditionalGeneration.from_pretrained(model_path)

            # ایجاد خط لوله خلاصه‌سازی با استفاده از منابع محلی
            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=-1  # استفاده از CPU برای اطمینان از عدم تداخل با کارت گرافیک
            )
            print("✅ Pegasus Engine is fully loaded and ready!")

        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            print("💡 Tip: Ensure all 5 files (including pytorch_model.bin) are in 'my_pegasus' folder.")

    def preprocess(self, document: str) -> str:
        if not document or not document.strip():
            return ""
        # پاکسازی فواصل اضافی برای درک بهتر مدل
        text = document.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def build_prompt(self, document: str) -> str:
        """
        Construct the final LLM input by combining
        the fixed prompt and the document text.

        Parameters
        ----------
        document : str
            Preprocessed document.

        Returns
        -------
        str
            Final input prompt for the LLM.
        """
        pass

    # ------------------------------------------------------------------
    # Step 3: LLM Generation
    # ------------------------------------------------------------------
    def generate_summary(self, prompt: str) -> str:
        """
        Generate an abstractive summary using an external LLM.

        NOTE:
        - The LLM is treated as a black-box oracle.
        - No internal states, scores, or explanations are exposed.

        Parameters
        ----------
        prompt : str
            Input prompt for the LLM.

        Returns
        -------
        str
            Abstractive summary S_llm.
        """
        pass

    # ------------------------------------------------------------------
    # Main Interface
    # ------------------------------------------------------------------
    def summarize(self, document: str) -> str:
        """
        Generate a standalone abstractive summary from the original document.

        Algorithm:
        1. Preprocess document D
        2. Construct prompt I = P + D
        3. Generate summary S_llm = LLM.generate(I)
        4. Return S_llm

        Parameters
        ----------
        document : str
            Original input document.

        Returns
        -------
        str
            Abstractive LLM-generated summary.
        """
        processed_doc = self.preprocess(document)
        prompt = self.build_prompt(processed_doc)
        summary = self.generate_summary(prompt)
        return summary
