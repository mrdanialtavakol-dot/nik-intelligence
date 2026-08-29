# NIK INTELLIGENCE V0.4

نسخه مدیریتی آزمایشی سامانه «نیک اس‌ام‌اس | تحلیل داده» برای NIKSMS / NIKPOS.

> این نسخه با داده‌های مبنای Aggregate و داده‌های Synthetic / Demo ساخته شده و هنوز به سیستم‌های داخلی NIK متصل نیست.
>
> Forecast، Churn و Timeline ویدیو خروجی آزمایشی هستند و نباید به‌عنوان واقعیت Production یا داده قطعی اینستاگرام ارائه شوند.

## مهم‌ترین تغییرات V0.4

- رفع خطاهای `AttributeError` مربوط به `instagram_followers` و `reels_per_day`.
- اضافه‌شدن لایه Compatibility بین `app.py`، `Scenario` و Analytics تا اختلاف موقت نسخه فایل‌ها هنگام Deploy باعث Crash نشود.
- اضافه‌شدن fallback امن برای KPIهای محتوا و Instagram Snapshot.
- فارسی‌سازی گسترده رابط کاربری، جدول‌ها، عنوان نمودارها، KPIها و توضیحات مدیریتی.
- بازطراحی Visual System با محور رنگ `#ADCBFF`، گرادینت آبی تیره، Glassmorphism و نورپردازی کنترل‌شده.
- استفاده از لوگوی NIKSMS در Header و Sidebar.
- حفظ صفحه اول CEO با 8 KPI تصمیم‌ساز و سیگنال‌های مدیریتی.
- حفظ Media Intelligence Lab شامل ویدیوها، تصاویر، Timeline آزمایشی و Event Markerها.
- دکمه‌های API و n8n همچنان Placeholder هستند و اتصال واقعی ندارند.

## نکته بسیار مهم برای Deploy

برای جلوگیری از تکرار خطاهای نسخه‌های قبلی، فقط `app.py` را جایگزین نکنید.

**محتویات کامل ZIP را Extract کنید و همه فایل‌ها و پوشه‌ها را در Root همان Repository جایگزین کنید.**

ساختار Repository باید به شکل زیر باشد:

```text
app.py
analytics_engine.py
business_data.py
data_generator.py
insight_engine.py
media_data.py
ml_engine.py
requirements.txt
README.md
VERSION.txt
DEPLOY_GUIDE_FA.txt
assets/
  images/
  videos/
```

بعد از Commit در GitHub:

1. وارد Streamlit Community Cloud شوید.
2. روی سه‌نقطه کنار App بزنید.
3. `Reboot` را انتخاب کنید.
4. بعد از اتمام Build، صفحه را Hard Refresh کنید.

Main file path باید همچنان این باشد:

```text
app.py
```

## Run محلی

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## تست‌های V0.4

- Syntax همه فایل‌های Python بررسی شده است.
- تمام صفحات اصلی اپ در تست Runtime عبور کرده‌اند.
- حالت Mixed Revision نیز تست شده؛ یعنی حتی اگر `data_generator.py` قدیمی و فاقد فیلدهای `instagram_followers` و `reels_per_day` باشد، صفحات اصلی دیگر با AttributeError از کار نمی‌افتند.

## وضعیت اتصال

فعلاً:

```text
Demo / Synthetic Data
API: Not connected
Database: Not connected
n8n: Not connected
```

معماری آینده می‌تواند به شکل زیر باشد:

```text
NIK Database / Instagram / CRM / Tracked Links
        ↓
Read-only API / Webhook
        ↓
n8n / Data Pipeline
        ↓
Validation + Warehouse
        ↓
NIK Intelligence
        ↓
Dashboard / Alerts / AI Assistant
```
