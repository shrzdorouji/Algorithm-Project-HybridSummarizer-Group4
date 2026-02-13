from typing import Optional
import re
import os

class LLMAbstractiveSummarizer:
    def __init__(
            self,
            model_path: str = "./my_pegasus",
            max_length: int = 150,
            prompt_template: Optional[str] = None,
    ):
        self.max_length = max_length
        self.prompt_template = prompt_template or "{document}"

        try:
            from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration

            print(f"🔄 Loading Pegasus from local directory: {model_path}")
            tokenizer = PegasusTokenizer.from_pretrained(model_path)
            model = PegasusForConditionalGeneration.from_pretrained(model_path)

            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=-1  # CPU
            )
            print("✅ Pegasus Engine is ready with advanced sampling!")

        except Exception as e:
            print(f"❌ Error loading local model: {e}")

    def preprocess(self, document: str) -> str:
        if not document or not document.strip():
            return ""
        text = document.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def build_prompt(self, document: str) -> str:
        if not document:
            return ""
        # طبق نظر استاد برای زیروشات بهتر، دستور صریح‌تر می‌دهیم
        return f"Summarize and paraphrase the following: {document}"

    def generate_summary(self, prompt: str) -> str:
        if not prompt: return ""
        try:
            # محاسبه طول ورودی به کلمه
            input_token_len = len(prompt.split())

            # تعیین سقف خروجی: یا 60% طول ورودی، یا حداکثر 80 کلمه (هر کدام کمتر بود)
            dynamic_max = min(150, int(input_token_len * 0.6))
            dynamic_min = min(5, int(input_token_len * 0.5)) # برای متن‌های کوتاه، حداقل را روی ۵ بگذار

            outputs = self.summarizer(
                prompt,
                max_length=dynamic_max,
                min_length=dynamic_min,
                do_sample=True,
                top_k=40,
                top_p=0.90,
                temperature=0.8,  # دمای متعادل برای کاهش توهم (Hallucination)
                repetition_penalty=3.5,
                no_repeat_ngram_size=2,
                num_beams=1,
                length_penalty=1.0,  # خنثی کردن برای حذف Warning
                early_stopping=False,  # خنثی کردن برای حذف Warning
                truncation=True
            )

            res = outputs[0]['summary_text'].strip()
            # تمیزکاری نهایی برای حذف کاراکترهای اضافه پگاسوس
            res = res.replace("<n>", " ").replace(" .", ".").strip()
            return res

        except Exception as e:
            print(f"⚠️ Generation Error: {e}")
            return " ".join(prompt.split()[:dynamic_max]) if 'dynamic_max' in locals() else ""

    def summarize(self, document: str) -> str:
        processed_doc = self.preprocess(document)
        input_text = self.build_prompt(processed_doc)
        return self.generate_summary(input_text)