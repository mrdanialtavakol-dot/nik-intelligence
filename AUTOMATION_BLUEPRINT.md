# Automation Blueprint — n8n / API

هر اتوماسیون واقعی باید این اجزا را داشته باشد:

1. Schedule: مثلا هر ۲ ساعت، روزانه یا هفتگی
2. Source: منبع داده معتبر
3. Validation: بررسی کیفیت/تازگی داده
4. Condition: شرط قابل توضیح
5. Action: Task / Alert / Approval / Workflow
6. Owner: مسئول پیگیری
7. Approval: در اقدامات مالی/جشنواره/تولید
8. Retry Policy: در صورت خطای API
9. Audit Log: ثبت اجرای Rule و نتیجه
10. Verification: بررسی اینکه اقدام نتیجه مورد انتظار را داشته یا نه

نمونه:

Revenue < 90% Monthly Target
→ Validate accounting feed
→ Open Campaign Planner scenario
→ Check margin + QC inventory + production capacity
→ Create approval task for CEO/Sales/Accounting
→ If approved, send workflow to n8n
→ Track campaign result
