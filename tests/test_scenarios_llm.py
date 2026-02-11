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
    Climate change is one of the most pressing challenges facing the global community in the 21st century. 
    The rising levels of carbon dioxide in the atmosphere, primarily caused by industrial activities and 
    the burning of fossil fuels, have led to an increase in global temperatures. This phenomenon, 
    often referred to as global warming, results in the melting of polar ice caps and a significant 
    rise in sea levels, which threatens coastal cities around the world. Environmental scientists 
    urgently advocate for a transition to renewable energy sources, such as solar and wind power, 
    to mitigate the long-term effects of this environmental crisis and ensure a sustainable future 
    for coming generations.
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