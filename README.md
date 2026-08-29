# NIK INTELLIGENCE V0.5

نسخه مدیریتی آزمایشی سامانه «نیک اس‌ام‌اس | تحلیل داده» برای NIKSMS / NIKPOS.

> این نسخه با داده‌های مبنای Aggregate و داده‌های Synthetic / Demo ساخته شده و هنوز به سیستم‌های داخلی NIK متصل نیست.
>
> پیش‌بینی، Churn، Anomaly Detection و Timeline ویدیو آزمایشی هستند و نباید به‌عنوان واقعیت عملیاتی شرکت ارائه شوند.

## تغییر اصلی V0.5 — مرکز فرمان مدیرعامل

صفحه اول از یک Dashboard عمومی به **Executive Command Center** تبدیل شده است و چهار سؤال مدیریتی را پاسخ می‌دهد:

1. الان وضعیت چیست؟
2. چه چیزی تغییر کرده یا نیازمند توجه است؟
3. پول کجا ساخته می‌شود؟
4. مدیر باید امروز روی چه چیزی تمرکز کند؟

### اجزای جدید صفحه اول

- وضعیت مدیریتی قاعده‌محور با توضیح شفاف منبع داده.
- ۶ KPI اصلی در نمای ۵ ثانیه‌ای.
- صف تصمیم با سه اولویت پویا.
- نقشه پول بر اساس Channel و Plan.
- نمایش سهم درآمد Plan B و سهم کانال تلفنی/آنلاین.
- تحلیل حساسیت و «اهرم رشد»:
  - اثر +۱ فروش تلفنی در روز.
  - اثر +۵ فروش تلفنی در روز.
  - اثر +۱۰ فروش آنلاین در ماه.
  - اثر تغییر ۱۰ واحد درصدی Mix از A به B.
- بخش «شکاف داده» برای مشخص‌کردن چهار اتصال حیاتی بعدی.
- رادار آزمایشی Anomaly و Churn جدا از وضعیت واقعی کسب‌وکار.
- Forecast و Funnel به بخش جمع‌شونده منتقل شده‌اند تا صفحه اول فنی و شلوغ نشود.

## اصل مهم V0.5

هشدارهای Synthetic و مدل Churn **وضعیت واقعی شرکت را قرمز نمی‌کنند**. وضعیت صفحه اول فقط از داده‌های مبنا و محاسبات قابل توضیح ساخته می‌شود. خروجی‌های ML/Anomaly با برچسب آزمایشی جدا نمایش داده می‌شوند.

## پایداری

V0.5 بر پایه Compatibility Layer نسخه V0.4 ساخته شده و موتورهای قبلی دستکاری اساسی نشده‌اند. Media Intelligence Lab، Sales Analytics، Customer Intelligence، SMS، Forecast و سایر صفحات حفظ شده‌اند.

## Deploy

برای کمترین ریسک، ZIP کامل را Extract و تمام فایل‌ها را در Root همان Repository جایگزین کنید.

ساختار باید شامل این موارد باشد:

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

سپس:

1. Commit changes در GitHub.
2. Streamlit Community Cloud → سه‌نقطه کنار App → Reboot.
3. بعد از Build یک Hard Refresh انجام دهید.

Main file path:

```text
app.py
```

## تست V0.5

- `app.py` با `py_compile` بررسی شده است.
- Command Center با داده واقعیِ مبنا + داده Synthetic موتور فعلی اجرا شده است.
- سناریوی پیش‌فرض همچنان:
  - Monthly Units = 320
  - ASP = 22.5M Toman
  - Monthly Revenue = 7.2B Toman
- Priority Engine در سناریوی پیش‌فرض سه موضوع زیر را شناسایی می‌کند:
  - تمرکز بالای کانال تلفنی.
  - تمرکز عملکرد محتوا روی چند محتوای پربازدید.
  - نیاز به داده Call Center برای تفسیر علمی Lead Backlog.

## وضعیت اتصال

```text
Demo / Synthetic Data
API: Not connected
Database: Not connected
n8n: Not connected
```

معماری آینده:

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
Executive Command Center / Alerts / AI Assistant
```
