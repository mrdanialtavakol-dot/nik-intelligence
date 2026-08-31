# NIK Executive Management OS V0.9

## Product principle
The CEO home is not a data dump. It shows only: exception, decision, action, and confidence.
Detailed analytics live inside the department that owns them.

## Navigation
- میز مدیرعامل
  - مرکز فرمان
  - تسک‌ها و گزارش‌ها
  - نبض سازمان
- واحدهای شرکت
  - مارکتینگ
    - نمای کلی مارکتینگ
    - تحلیل محتوا و اینستاگرام
    - آزمایشگاه تحلیل محتوا
    - اقتصاد مارکتینگ
    - روند و عملکرد
  - فروش
    - مرکز لید و پیگیری
    - تحلیل فروش
    - پخش لید و عملکرد کارشناسان
  - حسابداری
    - نمای مالی مدیریتی
    - اتوماسیون حسابداری
  - فناوری اطلاعات
    - اتاق عملیات IT
    - آپدیت، تسک و تحویل
  - تولید و QC
    - تولید و کنترل کیفیت
    - ارز، تأمین و بهای ساخت
  - پشتیبانی
  - منابع انسانی
- رشد و تصمیم
- هوشمندی سازمان
- سیستم و اتوماسیون

## V0.9 additions
- Lighter Apple-like liquid-glass visual system based on #ADCBFF.
- Department-first navigation.
- API + n8n controls at department level.
- Marketing economics: average salary input, team cost, online revenue model, tracked attribution ROI.
- Marketing timeline and media lab restored under Marketing.
- Sales lead routing separated from executive overview.
- Finance CEO view separated from accounting automation.
- FX & supply center for CNY/Toman and AED/Toman with manual rates until an FX API is connected.
- Currency sensitivity model for device batch cost.
- CEO home reduced to a short executive command surface.
- Optional local IRANSansX loader; app falls back safely when fonts are absent.

## Data trust
Manual rates are never presented as live rates.
Marketing online revenue is not automatically called marketing-attributed revenue.
Synthetic and demo metrics remain explicitly labelled.
