"""
LLM-Based Abstractive Summarization Module
------------------------------------------
Model: Pegasus (Distilled) - Fully Offline Version
"""

from typing import Optional
import re
import os

from nltk.translate.lepor import length_penalty


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
        if not document:
            return ""
        return self.prompt_template.format(document=document)

    def generate_summary(self, prompt: str) -> str:
        if not prompt:
            return ""

        try:
            # تغییرات استراتژیک برای بازنویسی خلاقانه
            outputs = self.summarizer(
                prompt,
                max_length=60,
                min_length=30,

                # --- تغییرات اصلی اینجاست ---
                do_sample=True,  # فعال کردن نمونه‌برداری برای خلاقیت بیشتر
                top_k=50,  # انتخاب از بین ۵۰ کلمه برتر
                top_p=0.95,  # استفاده از تکنیک Nucleus Sampling
                temperature=1.2,  # کنترل میزان خلاقیت (عدد بالاتر = بازنویسی بیشتر)

                no_repeat_ngram_size=2,  # جلوگیری از تکرار عبارات ۳ کلمه‌ای متن اصلی
                repetition_penalty=10.0,  # جریمه سنگین برای کپی کردن کلمات
                # --------------------------
                length_penalty=1.5,
                truncation=True
            )

            res = outputs[0]['summary_text'].strip()
            # پاکسازی خروجی
            return res.replace("<n>", " ").strip()

        except Exception as e:
            print(f"⚠️ LLM Generation Error: {e}")
            return " ".join(prompt.split()[:25]) + "..."

    def summarize(self, document: str) -> str:
        processed_doc = self.preprocess(document)
        input_text = self.build_prompt(processed_doc)
        return self.generate_summary(input_text)