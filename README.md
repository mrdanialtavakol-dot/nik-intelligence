# NIK MANAGEMENT OS — V0.8

پنل مدیریت و اتوماسیون اختصاصی **کیوان میرزایی**، مدیرعامل NIK.

V0.8 روی هسته پایدار V0.7 ساخته شده و Data Science / ML / Media Intelligence قبلی را حذف یا بازنویسی نمی‌کند. قابلیت‌های جدید در فایل مستقل `ceo_ops_engine.py` قرار گرفته‌اند تا تغییرات عملیاتی باعث شکستن موتور Analytics نشوند.

## تغییر جهت محصول

پروژه از یک Data Intelligence Prototype به یک **CEO Management & Automation OS** حرکت کرده است.

هدف V0.8:

- مدیرعامل هر واحد را جدا ببیند.
- تسک ایجاد کند و Owner / Deadline / KPI داشته باشد.
- گزارش‌های عقب‌افتاده به شکل Exception مشخص شوند.
- ربات Rule-based تسک پیشنهاد دهد.
- پیشنهاد با یک کلیک وارد Task Engine شود.
- در آینده n8n / API همان تسک و Reminder را به ابزار واقعی شرکت مخابره کند.
- حسابداری و فروش تلفنی به سمت Straight-Through Processing حرکت کنند.

## تیم‌های ثبت‌شده در V0.8

### IT
- حمید تهرانی
- مسعود طاهری
- سروش کلانتریان
- پوریا سلیمانی
- کوروش عذت پور

### حسابداری
- حسین جودکی — مدیر حسابداری
- مشتبا بیان — کارمند حسابداری

### منابع انسانی
- خانم مقصودی — مسئول منابع انسانی

### پشتیبانی
- خانم ملیکا جمع دار — سرپرست پشتیبانی
- خانم آزاد — نیروی پشتیبانی

### مارکتینگ
- امیر عباس حبیبی — سرپرست مارکتینگ
- دانیال توکل
- سحر نور محمدی
- داریوش مشتاقی
- امین عین آبادی

نام کارشناسان فروش هنوز در ورودی فعلی ثبت نشده است؛ V0.8 از Placeholder قابل ویرایش استفاده می‌کند.

## Baseline جدید

`Lead Backlog = 5,490`

این عدد Baseline فعلی است. داده‌های Performance فروشندگان، تراکنش‌های حسابداری، SLA پشتیبانی و برخی KPIهای عملیاتی همچنان Demo هستند تا منبع واقعی متصل شود.

## صفحات جدید V0.8

### مرکز کیوان
- مرکز فرمان مدیرعامل
- کارها و گزارش‌های کیوان
- نبض سازمان

### واحدهای شرکت
- اتاق عملیات IT
- اتوماسیون حسابداری
- مرکز پخش و پیگیری لید
- پشتیبانی
- منابع انسانی
- تولید و QC
- مارکتینگ

### رشد و تصمیم
- هوشمندی درآمد
- برنامه‌ریز جشنواره
- شبیه‌ساز تصمیم

### هوشمندی داده
Analytics قبلی V0.7/V0.6 بدون حذف باقی مانده‌اند.

### سیستم و اتوماسیون
- Automation Center
- Connections
- Access Control
- Data Center
- Pipeline
- Settings

## Task & Reporting Engine

Task schema در Prototype شامل:

`task_id, department, title, assignee, creator, source, priority, status, KPI, created_at, due_at, followup_hours, last_report, report_required, note`

در نسخه واقعی:

`Task → Reminder → Report → Verify KPI → Escalate if needed → Close → Audit`

## Accounting Automation

V0.8 یک Prototype برای مسیر زیر دارد:

`Bank/Gateway → Idempotency → Invoice Match → Coding → Double Entry → Reconciliation → Auto-post OR Exception Queue → Audit`

اصل طراحی:
- تراکنش مطمئن = Straight-Through Processing
- تراکنش مبهم/تکراری/بدون سند/مبلغ بالا = Human-on-exception

**کدینگ حساب‌ها در V0.8 Demo است و باید قبل از استفاده واقعی با کدینگ رسمی حسابداری NIK و قواعد مالیاتی جایگزین شود.**

## Lead Routing

صف 5,490 لید در Prototype با منطق زیر مدیریت می‌شود:

`Lead In → Validate/Dedupe → Score → Route → First-touch SLA → Outcome → Follow-up / Recycle`

Routing پیشنهادی بر اساس:

`Capacity × Performance × Fairness`

Performance کارشناسان در V0.8 Synthetic است تا CRM/Call Center واقعی متصل شود.

## Security / Persistence

- Authentication واقعی هنوز فعال نیست.
- Taskها و Audit در Prototype در Session نگه داشته می‌شوند.
- هیچ Credential بانکی، n8n، API یا Database داخل کد قرار ندارد.
- برای داده واقعی: Database + Authentication + RBAC + Audit + Secret Management لازم است.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
