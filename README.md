# NIK Executive Management OS — V0.9

نسخه V0.9 بر پایه V0.8 ساخته شده و موتورهای قبلی Data Science، Media Intelligence، Task Engine و Automation Prototype را حفظ می‌کند.

## فلسفه V0.9
مدیرعامل نباید در صفحه اول با انبوه نمودار مواجه شود. صفحه «میز مدیرعامل» فقط موارد نیازمند توجه، تصمیم و اقدام را نشان می‌دهد. جزئیات هر شاخص در واحد صاحب آن شاخص قرار دارد.

## ساختار سازمانی
- میز مدیرعامل: مرکز فرمان، تسک/گزارش، نبض سازمان
- واحدهای شرکت: مارکتینگ، فروش، حسابداری، IT، تولید/QC، پشتیبانی، منابع انسانی
- رشد و تصمیم: درآمد، جشنواره، شبیه‌ساز
- هوشمندی سازمان: مشتری، نیک‌پوز، پیامک، ناهنجاری، پیش‌بینی، بینش
- سیستم و اتوماسیون: n8n/API، دسترسی، داده، Pipeline، تنظیمات

## مارکتینگ
تحلیل محتوا و Media Intelligence دوباره زیر مارکتینگ قرار گرفته‌اند. Timeline ویدیو، گالری، آمار ۱۰ ریلز، اقتصاد مارکتینگ و روند عملکرد در زیرمنوی همان واحد هستند.

## تولید و ارز
صفحه «ارز، تأمین و بهای ساخت» برای ورود نرخ CNY/Toman و AED/Toman، هزینه قطعات، Batch تولید و جریان شرکت دبی اضافه شده است. نرخ‌ها فعلاً دستی‌اند و Live معرفی نمی‌شوند. اتصال FX API/n8n برای نسخه واقعی آماده طراحی است.

## Integration
بالای Workspaceهای سازمانی وضعیت API و n8n دیده می‌شود. این دکمه‌ها فعلاً Placeholder امن هستند و هیچ Credential ذخیره نمی‌شود.

## Fonts
IRANSansX به صورت Optional Local Font پشتیبانی می‌شود. فونت‌ها در ZIP بازتوزیع نشده‌اند. فایل‌های دارای مجوز خودتان را طبق FONT_SETUP_FA.txt در assets/fonts قرار دهید.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Safety / data trust
Prototype using baseline/demo/synthetic data. Not connected to NIK internal systems.
Forecasts and ML outputs are experimental and should not be used for production decisions.

---

# V0.10 — Customer Growth Edition

V0.10 یک حوزه اصلی مستقل با عنوان **مشتریان کسب‌وکار** اضافه می‌کند. این حوزه نمونه محصولی است که می‌تواند در آینده به‌صورت Add-on داخل پنل NIKSMS/NIKPOS به صاحبان کسب‌وکار ارائه شود.

منطق آن: **داده مشتری → سگمنت → فرصت → اقدام → کمپین/اتوماسیون → نتیجه → ROI**.

تمام اعداد این حوزه Synthetic/Demo هستند و از پنل مدیریتی داخلی NIK جدا نگه داشته شده‌اند. برای جزئیات محصول، `V10_CUSTOMER_PRODUCT.md` را ببینید.

---

## V0.10.1 — Visual Polish & Readability
این Patch منطق برنامه را تغییر نمی‌دهد. یک لایه نهایی UI بعد از CSSهای Legacy اضافه شده تا تمام بخش‌های قدیمی و جدید از یک Design System روشن و خوانا استفاده کنند. رنگ متن، آیکون، فرم‌ها، منوها، کارت‌های Glass، Status Chipها و بعضی Annotationهای نمودارها برای کنتراست بالاتر اصلاح شده‌اند.
