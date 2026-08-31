from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd

V09_VERSION = "V0.9"
CEO_PAGES = {
    "Executive Overview": "مرکز فرمان",
    "CEO Task Center": "تسک‌ها و گزارش‌ها",
    "Organization Pulse": "نبض سازمان",
}
DEPARTMENT_PAGES = {
    "مارکتینگ": {
        "Marketing Workspace": "نمای کلی مارکتینگ",
        "Content Analytics": "تحلیل محتوا و اینستاگرام",
        "Media Intelligence": "آزمایشگاه تحلیل محتوا",
        "Marketing Economics": "اقتصاد مارکتینگ",
        "Marketing Trend": "روند و عملکرد",
    },
    "فروش": {
        "Sales Lead Center": "مرکز لید و پیگیری",
        "Sales Analytics": "تحلیل فروش",
        "Sales Funnel Ops": "پخش لید و عملکرد کارشناسان",
    },
    "حسابداری": {
        "Finance Dashboard": "نمای مالی مدیریتی",
        "Accounting Automation": "اتوماسیون حسابداری",
    },
    "فناوری اطلاعات": {
        "IT Workspace": "اتاق عملیات IT",
        "IT Delivery": "آپدیت، تسک و تحویل",
    },
    "تولید و QC": {
        "Production & QC": "تولید و کنترل کیفیت",
        "FX & Supply": "ارز، تأمین و بهای ساخت",
    },
    "پشتیبانی": {"Support Workspace": "عملکرد پشتیبانی"},
    "منابع انسانی": {"HR Workspace": "منابع انسانی"},
}
GROWTH_PAGES = {
    "Revenue Intelligence": "هوشمندی درآمد",
    "Campaign Planner": "برنامه‌ریز جشنواره",
    "Scenario Simulator": "شبیه‌ساز تصمیم",
}
INTELLIGENCE_PAGES = {
    "Customer Intelligence": "رفتار و ارزش مشتری",
    "NIKPOS Analytics": "هوشمندی نیک‌پوز",
    "SMS Analytics": "هوشمندی پیامک",
    "Anomaly Detection": "تغییرات غیرعادی",
    "Predictions": "پیش‌بینی",
    "Automated Insights": "بینش‌های خودکار",
}
SYSTEM_PAGES = {
    "Automation Center": "مرکز اتوماسیون",
    "Connections": "اتصال‌ها",
    "Access Control": "دسترسی و نقش‌ها",
    "Data Center": "مرکز داده",
    "Analysis Pipeline": "جریان پردازش داده",
    "Settings / Scenario Controls": "تنظیمات",
}

def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)

def fx_cost_snapshot(cny_toman: float, aed_toman: float, component_cost_cny: float, batch_units: int,
                     freight_toman: float = 0.0, local_cost_per_unit_toman: float = 0.0,
                     dubai_inflow_aed: float = 0.0, dubai_outflow_aed: float = 0.0) -> Dict[str, float]:
    cny_toman, aed_toman = max(_f(cny_toman),0), max(_f(aed_toman),0)
    component_cost_cny = max(_f(component_cost_cny),0); batch_units=max(int(batch_units or 0),0)
    freight_toman=max(_f(freight_toman),0); local=max(_f(local_cost_per_unit_toman),0)
    din=max(_f(dubai_inflow_aed),0); dout=max(_f(dubai_outflow_aed),0)
    component_unit=component_cost_cny*cny_toman
    component_batch=component_unit*batch_units
    total=component_batch+local*batch_units+freight_toman
    return {
        "component_per_unit_toman": component_unit,
        "component_batch_toman": component_batch,
        "batch_total_toman": total,
        "unit_total_toman": total/batch_units if batch_units else 0.0,
        "dubai_net_aed": din-dout,
        "dubai_net_toman": (din-dout)*aed_toman,
        "batch_if_cny_plus_5": total+component_batch*.05,
        "batch_if_cny_minus_5": max(0.0,total-component_batch*.05),
        "cny_5pct_impact": component_batch*.05,
    }

def marketing_economics_snapshot(online_units: float, average_selling_price: float, attributed_sales_per_day: float,
                                 sales_days: int, monthly_team_cost: float = 0.0,
                                 tracked_marketing_revenue: float = 0.0) -> Dict[str, float]:
    online_units=max(_f(online_units),0); asp=max(_f(average_selling_price),0); att=max(_f(attributed_sales_per_day),0)
    days=max(int(sales_days or 0),0); cost=max(_f(monthly_team_cost),0); rev=max(_f(tracked_marketing_revenue),0)
    return {
        "online_revenue_model": online_units*asp,
        "estimated_attributed_units": att*days,
        "estimated_attributed_revenue": att*days*asp,
        "monthly_team_cost": cost,
        "tracked_marketing_revenue": rev,
        "verified_roi": rev/cost if cost>0 and rev>0 else 0.0,
    }

def department_kpi_catalog() -> pd.DataFrame:
    rows = [
        ("مارکتینگ","خروجی محتوا / Median / Share / Lead / Sale","روزانه/هفتگی","Instagram + CRM"),
        ("مارکتینگ","هزینه تیم / Revenue منتسب / ROI","ماهانه","HR + Accounting + Attribution"),
        ("فروش","Lead Assigned / Contacted / Won / Lost / SLA","روزانه","CRM / Call Center"),
        ("حسابداری","Cash In / Cash Out / Reconciliation / Exception","روزانه","Bank / Accounting"),
        ("فناوری اطلاعات","Sprint / Release / Blocker / Bug","روزانه","Task Tracker / Git"),
        ("تولید و QC","Production / QC / Reject / CNY Exposure","روزانه","Production + FX API"),
        ("پشتیبانی","Ticket / SLA / Escalation / Resolution","روزانه","Support System"),
        ("منابع انسانی","Headcount / Attendance / Hiring / Cost","هفتگی/ماهانه","HRIS / Payroll"),
    ]
    return pd.DataFrame(rows, columns=["بخش","KPI","بازه","منبع آینده"])
