from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from business_data import reel_snapshot_metrics
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
    reel = reel_snapshot_metrics()

    insights.append(
        {
            "type": "اقتصاد فروش",
            "title": "درآمد ماهانه محاسبه‌شده",
            "text": (
                f"با فرض {_fa(scenario.sales_days_per_month)} روز فروش، {_fa(scenario.daily_phone_sales)} فروش تلفنی در روز، "
                f"{_fa(scenario.monthly_online_sales)} فروش آنلاین در ماه و ترکیب فعلی طرح‌ها، فروش ماهانه {_fa(f'{kpis['monthly_units']:.0f}')} دستگاه می‌شود. "
                "این عدد محاسبه‌شده است و فروش ثبت‌شده حسابداری نیست."
            ),
        }
    )

    insights.append(
        {
            "type": "ترکیب کانال",
            "title": "وابستگی شدید حجم فروش به تلفن",
            "text": f"در سناریوی فعلی حدود {_pct(kpis['phone_share'])} از تعداد فروش ماهانه از کانال تلفنی می‌آید.",
        }
    )

    if np.isfinite(kpis["backlog_months_of_sales"]):
        insights.append(
            {
                "type": "صف فروش",
                "title": "اندازه صف در برابر جریان فروش",
                "text": (
                    f"صف {_fa(scenario.lead_backlog)} لید، از نظر حجم، معادل حدود {_fa(f'{kpis['backlog_months_of_sales']:.1f}')} برابر فروش ماهانه سناریوی فعلی است. "
                    "این شاخص زمان تخلیه صف یا نرخ تبدیل نیست؛ برای آن به تعداد تماس، تماس موفق، لید واجد شرایط و زمان پردازش واقعی نیاز داریم."
                ),
            }
        )

    insights.append(
        {
            "type": "محتوا",
            "title": "عملکرد محتوا به چند ریلز پربازدید وابسته است",
            "text": (
                f"در نمای داده ده ریلز، میانگین بازدید {_fa(f'{reel['average_views']:.0f}')} ولی میانه فقط {_fa(f'{reel['median_views']:.0f}')} است؛ "
                f"همچنین سه ریلز برتر {_pct(reel['top3_view_share'])} کل ویوها را ساخته‌اند. برای ارزیابی تیم، میانه و توزیع عملکرد مهم‌تر از میانگین تنهاست."
            ),
        }
    )

    insights.append(
        {
            "type": "تعامل محتوا",
            "title": "کامنت + اشتراک‌گذاری",
            "text": (
                f"در همان نمای داده، مجموع کامنت و اشتراک‌گذاری {_fa(f'{reel['total_interactions']:.0f}')} و نسبت آن به بازدید حدود {_pct(reel['interaction_rate'])} است. "
                "این شاخص تقریبی تعامل است و جایگزین ریچ، ذخیره یا زمان تماشا نیست."
            ),
        }
    )

    insights.append(
        {
            "type": "انتساب فروش",
            "title": "فروش منتسب به محتوا دوباره شماری نمی‌شود",
            "text": (
                f"فروش منتسب به محتوا در مدل حدود {_fa(f'{kpis['content_monthly_sales_estimated']:.0f}')} دستگاه در ماه برآورد می‌شود، "
                "اما به فروش کل اضافه نشده؛ چون ممکن است همان فروش تلفنی یا آنلاین باشد که منبع اولیه‌اش محتوا بوده است."
            ),
        }
    )

    insights.append(
        {
            "type": "ریسک مشتری",
            "title": "مدل آزمایشی ریزش",
            "text": f"در دیتای مصنوعی، سهم مشتریان با ریسک زیاد یا بسیار زیاد حدود {_pct(risk_stats['high_or_very_high_share'])} است.",
        }
    )

    if len(monthly_sales) >= 2:
        current = monthly_sales.iloc[-1]["revenue"]
        previous = monthly_sales.iloc[-2]["revenue"]
        change = (current / previous - 1) if previous else 0
        direction = "افزایش" if change >= 0 else "کاهش"
        insights.append(
            {
                "type": "روند مصنوعی",
                "title": "تغییر ماه‌به‌ماه داده دمو",
                "text": f"در داده تاریخی مصنوعی، درآمد نسبت به ماه قبل {_pct(abs(change))} {direction} داشته است.",
            }
        )

    sales_anomaly_count = int(sales_anomalies["is_anomaly"].sum()) if "is_anomaly" in sales_anomalies else 0
    sms_anomaly_count = int(sms_anomalies["is_anomaly"].sum()) if "is_anomaly" in sms_anomalies else 0
    if sales_anomaly_count or sms_anomaly_count:
        insights.append(
            {
                "type": "هشدار دمو",
                "title": "سیگنال ناهنجاری",
                "text": (
                    f"امتیاز Z متحرک در دیتای مصنوعی {_fa(sales_anomaly_count)} ناهنجاری فروش و "
                    f"{_fa(sms_anomaly_count)} ناهنجاری تحویل پیامک علامت‌گذاری کرده است."
                ),
            }
        )

    return insights
