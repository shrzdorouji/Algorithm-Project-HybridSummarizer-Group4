import sys
import os
import time

# اضافه کردن مسیر پروژه به پایتون
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.llm_integration import LLMAbstractiveSummarizer


def run_final_test():
    # مقداردهی اولیه مدل (BART-Large)
    summarizer = LLMAbstractiveSummarizer()

    # یک متن تست چالش‌برانگیز (درباره تغییرات اقلیمی)
    # این متن جملات بلندی دارد تا ببینیم مدل چطور آن‌ها را کوتاه و بازنویسی می‌کند
    document = """
The development of mRNA vaccines has marked a new era in modern medicine, offering a faster and more flexible approach to preventing infectious diseases. Unlike traditional vaccines, which use weakened or inactivated viruses, mRNA technology teaches cells how to make a protein that triggers an immune response. Researchers are now exploring the potential of this technology to treat other conditions, including cancer and rare genetic disorders, potentially saving millions of lives in the coming decades.
    """

    print("\n" + "=" * 60)
    print("📄 متن اصلی ورودی (Original Document):")
    print("-" * 60)
    print(document.strip())
    print("=" * 60)

    print("\n🤖 در حال تولید خلاصه انتزاعی با مدل BART-Large...")
    print("⚠️ (این فرآیند روی CPU ممکن است ۱۵ تا ۳۰ ثانیه زمان ببرد)")

    start_time = time.time()

    # اجرای متد اصلی
    summary = summarizer.summarize(document)

    end_time = time.time()

    print("\n" + "✨" * 15)
    print("🎯 خروجی نهایی بازنویسی شده (S_llm):")
    print("✨" * 15)
    print(f"\n{summary}")
    print("\n" + "-" * 30)
    print(f"⏱️ زمان صرف شده: {end_time - start_time:.2f} ثانیه")
    print("=" * 60)


if __name__ == "__main__":
    run_final_test()