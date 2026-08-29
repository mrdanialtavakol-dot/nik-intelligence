from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from data_generator import Scenario


def _fa(value) -> str:
    return str(value).translate(str.maketrans("0123456789,.%", "۰۱۲۳۴۵۶۷۸۹٬٫٪"))


def _pct(value: float) -> str:
    return _fa(f"{value:.1%}")


def generate_insights(
    scenario: Scenario,
    kpis: Dict[str, float],
    monthly_sales: pd.DataFrame,
    risk_stats: Dict[str, float],
    sales_anomalies: pd.DataFrame,
    sms_anomalies: pd.DataFrame,
) -> List[Dict[str, str]]:
    insights: List[Dict[str, str]] = []

    if len(monthly_sales) >= 2:
        current = monthly_sales.iloc[-1]["revenue"]
        previous = monthly_sales.iloc[-2]["revenue"]
        change = (current / previous - 1) if previous else 0
        direction = "افزایش یافته" if change >= 0 else "کاهش یافته"
        insights.append(
            {
                "type": "روند",
                "title": "تغییر درآمد",
                "text": f"درآمد تاریخی آزمایشی نسبت به ماه قبل {_pct(abs(change))} {direction} است.",
            }
        )

    insights.append(
        {
            "type": "ترکیب کانال",
            "title": "تمرکز فروش تلفنی",
            "text": f"فروش تلفنی حدود {_pct(kpis['phone_share'])} از تعداد فروش سناریوی فعلی را تشکیل می‌دهد.",
        }
    )

    days_to_process = scenario.lead_backlog / scenario.daily_phone_sales if scenario.daily_phone_sales > 0 else np.inf
    if np.isfinite(days_to_process):
        severity = "زیاد" if days_to_process > 90 else "متوسط" if days_to_process > 30 else "قابل مدیریت"
        insights.append(
            {
                "type": "شاخص ظرفیت",
                "title": "صف سرنخ‌ها در برابر ظرفیت فروش",
                "text": (
                    f"نسبت تعداد سرنخ‌های در صف به فروش تلفنی روزانه حدود {_fa(f'{days_to_process:.0f}')} روز است و در این مدل «{severity}» ارزیابی می‌شود. "
                    "این عدد فقط یک شاخص تقریبی ظرفیت است و زمان واقعی پردازش سرنخ‌ها محسوب نمی‌شود."
                ),
            }
        )

    insights.append(
        {
            "type": "نرخ تبدیل",
            "title": "تبدیل سرنخ به خرید",
            "text": f"نرخ تبدیل سرنخ به خرید در سناریوی فعلی حدود {_pct(kpis['lead_purchase_conversion'])} است.",
        }
    )

    insights.append(
        {
            "type": "محتوا",
            "title": "انتساب تخمینی / آزمایشی",
            "text": (
                f"فروش منتسب به محتوا در این مدل حدود {_fa(f'{kpis['content_monthly_sales']:.0f}')} دستگاه در ماه برآورد می‌شود. "
                "برای Attribution واقعی باید منبع ورودی و مدل انتساب مشخص وجود داشته باشد."
            ),
        }
    )

    insights.append(
        {
            "type": "ریسک مشتری",
            "title": "ریسک آزمایشی ریزش",
            "text": f"مشتریان با ریسک زیاد یا بسیار زیاد حدود {_pct(risk_stats['high_or_very_high_share'])} از مشتریان مصنوعی را تشکیل می‌دهند.",
        }
    )

    sales_anomaly_count = int(sales_anomalies["is_anomaly"].sum()) if "is_anomaly" in sales_anomalies else 0
    sms_anomaly_count = int(sms_anomalies["is_anomaly"].sum()) if "is_anomaly" in sms_anomalies else 0
    if sales_anomaly_count:
        insights.append(
            {
                "type": "هشدار",
                "title": "ناهنجاری در فروش شناسایی شد",
                "text": f"روش Rolling Z-score تعداد {_fa(sales_anomaly_count)} مشاهده غیرعادی در درآمد فروش آزمایشی شناسایی کرده است.",
            }
        )
    if sms_anomaly_count:
        insights.append(
            {
                "type": "هشدار",
                "title": "ناهنجاری در تحویل پیامک شناسایی شد",
                "text": f"روش Rolling Z-score تعداد {_fa(sms_anomaly_count)} مشاهده غیرعادی در نرخ تحویل پیامک شناسایی کرده است.",
            }
        )

    return insights
