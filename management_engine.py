from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


DEPARTMENTS = [
    {"key": "accounting", "name": "حسابداری", "icon": "◫"},
    {"key": "sales", "name": "فروش", "icon": "↗"},
    {"key": "development", "name": "برنامه‌نویسی", "icon": "⌘"},
    {"key": "hr", "name": "منابع انسانی", "icon": "◎"},
    {"key": "qc", "name": "QC دستگاه", "icon": "✓"},
    {"key": "marketing", "name": "مارکتینگ", "icon": "▶"},
]

DEPARTMENT_NAMES = {item["key"]: item["name"] for item in DEPARTMENTS}

# These values exist only to make the management prototype interactive.
# Every value is explicitly labeled demo/synthetic in the UI until a real source is connected.
MANAGEMENT_DEMO_DEFAULTS: Dict[str, float] = {
    "revenue_target": 8_000_000_000,
    "monthly_expenses": 2_250_000_000,
    "cash_collection_rate": 0.86,
    "overdue_receivables": 420_000_000,
    "monthly_units_target": 360,
    "calls_per_day": 180,
    "lead_contact_rate": 0.64,
    "sprint_completion": 0.78,
    "open_bugs": 14,
    "critical_bugs": 2,
    "release_readiness": 0.86,
    "headcount": 24,
    "open_positions": 2,
    "attendance_rate": 0.94,
    "monthly_turnover_rate": 0.02,
    "qc_ready": 78,
    "qc_pending": 31,
    "qc_rejected": 6,
    "qc_rework": 9,
    "qc_pass_rate": 0.93,
    "production_daily_capacity": 18,
    "raw_inventory": 120,
    "campaign_leads": 350,
    "marketing_budget": 180_000_000,
    "campaign_roas": 3.2,
    "task_on_time_rate": 0.79,
}


ROLE_MATRIX = pd.DataFrame(
    [
        ("مدیرعامل / مدیر سیستم", "کامل", "کامل", "کامل", "کامل", "کامل", "کامل", "کامل"),
        ("سرپرست حسابداری", "مشاهده", "ویرایش", "مشاهده", "—", "—", "—", "مشاهده"),
        ("سرپرست فروش", "مشاهده", "مشاهده", "ویرایش", "—", "—", "مشاهده", "مشاهده"),
        ("سرپرست برنامه‌نویسی", "مشاهده", "—", "—", "ویرایش", "—", "—", "مشاهده"),
        ("سرپرست منابع انسانی", "مشاهده", "—", "—", "—", "ویرایش", "—", "مشاهده"),
        ("سرپرست QC", "مشاهده", "—", "مشاهده", "—", "—", "ویرایش", "مشاهده"),
        ("سرپرست مارکتینگ", "مشاهده", "—", "مشاهده", "—", "—", "—", "ویرایش"),
    ],
    columns=["نقش", "مرکز فرمان", "حسابداری", "فروش", "برنامه‌نویسی", "منابع انسانی", "QC", "مارکتینگ"],
)


DEFAULT_TASKS = pd.DataFrame(
    [
        ("فروش", "بازبینی ۴۰۰۰ لید و تعریف Stage واقعی", "سرپرست فروش", "در حال انجام", "بالا", "نرخ تبدیل لید به خرید", 0.10, 0.08),
        ("حسابداری", "تعریف Revenue و وصول واقعی روزانه برای اتصال به پنل", "سرپرست حسابداری", "برنامه‌ریزی", "بالا", "نرخ وصول", 0.92, 0.86),
        ("QC دستگاه", "کاهش صف QC و تعیین ظرفیت روزانه", "سرپرست QC", "در حال انجام", "بالا", "نرخ قبولی QC", 0.95, 0.93),
        ("مارکتینگ", "ساخت Tracking برای Content → Lead → Sale", "سرپرست مارکتینگ", "برنامه‌ریزی", "بالا", "درصد Attribution قابل ردیابی", 0.70, 0.15),
        ("برنامه‌نویسی", "تعریف API فقط‌خواندنی برای داده فروش", "سرپرست برنامه‌نویسی", "Backlog", "متوسط", "آمادگی Release", 0.95, 0.86),
        ("منابع انسانی", "تعریف KPI مشترک برای سرپرستان واحدها", "سرپرست منابع انسانی", "Backlog", "متوسط", "تکمیل KPI واحدها", 1.00, 0.65),
    ],
    columns=["بخش", "تسک", "مسئول", "وضعیت", "اولویت", "KPI", "هدف", "عملکرد"],
)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(fallback)


def _scenario_value(scenario: Any, names: Iterable[str], fallback: float = 0.0) -> float:
    for name in names:
        if hasattr(scenario, name):
            value = getattr(scenario, name)
            if value is not None:
                return _safe_float(value, fallback)
    return float(fallback)


def _kpi_value(kpis: Dict[str, Any], key: str, fallback: float = 0.0) -> float:
    return _safe_float(kpis.get(key, fallback), fallback)


def _score_higher(actual: float, target: float) -> float:
    if target <= 0:
        return 100.0
    return float(np.clip(actual / target * 100.0, 0.0, 120.0))


def _score_lower(actual: float, target: float) -> float:
    if actual <= 0:
        return 100.0
    if target <= 0:
        return 0.0
    return float(np.clip(target / actual * 100.0, 0.0, 120.0))


def _status_from_score(score: float) -> str:
    if score >= 95:
        return "عادی"
    if score >= 75:
        return "نیازمند توجه"
    return "اقدام پیشنهادی"


def management_defaults(overrides: Dict[str, Any] | None = None) -> Dict[str, float]:
    values = dict(MANAGEMENT_DEMO_DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if key in values:
                values[key] = _safe_float(value, values[key])
    return values


def department_kpis(scenario: Any, kpis: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> pd.DataFrame:
    d = management_defaults(overrides)
    monthly_revenue = _kpi_value(kpis, "monthly_revenue")
    monthly_units = _kpi_value(kpis, "monthly_units")
    phone_daily = _scenario_value(scenario, ["daily_phone_sales"], 10)
    online_month = _scenario_value(scenario, ["monthly_online_sales"], 20)
    lead_backlog = _scenario_value(scenario, ["lead_backlog"], 4000)
    stories = _scenario_value(scenario, ["stories_per_day"], 9)
    reels = _scenario_value(scenario, ["reels_per_day"], 1)
    content_sales = _scenario_value(scenario, ["content_sales_per_day", "estimated_content_sales_per_day"], 2)
    followers = _scenario_value(scenario, ["instagram_followers"], 207_000)

    expense_ratio = d["monthly_expenses"] / monthly_revenue if monthly_revenue > 0 else 1.0
    lead_capacity_proxy = max(phone_daily, 1)
    backlog_pressure = lead_backlog / lead_capacity_proxy

    rows: List[Dict[str, Any]] = []

    def add(dept: str, metric: str, actual: float, target: float, direction: str, unit: str, source: str, note: str = ""):
        score = _score_higher(actual, target) if direction == "higher" else _score_lower(actual, target)
        rows.append({
            "department": dept,
            "department_name": DEPARTMENT_NAMES[dept],
            "metric": metric,
            "actual": float(actual),
            "target": float(target),
            "direction": direction,
            "unit": unit,
            "source": source,
            "score": score,
            "status": _status_from_score(score),
            "note": note,
        })

    add("accounting", "درآمد ماهانه", monthly_revenue, d["revenue_target"], "higher", "تومان", "محاسبه‌شده", "تا اتصال حسابداری Revenue مدل است.")
    add("accounting", "نرخ وصول", d["cash_collection_rate"], 0.92, "higher", "percent", "آزمایشی")
    add("accounting", "نسبت هزینه به درآمد", expense_ratio, 0.35, "lower", "percent", "آزمایشی", "هزینه در نسخه Demo است.")
    add("accounting", "مطالبات سررسیدگذشته", d["overdue_receivables"], 250_000_000, "lower", "تومان", "آزمایشی")

    add("sales", "فروش ماهانه", monthly_units, d["monthly_units_target"], "higher", "دستگاه", "محاسبه‌شده")
    add("sales", "فروش تلفنی روزانه", phone_daily, 12, "higher", "دستگاه", "مبنای فعلی")
    add("sales", "فروش آنلاین ماهانه", online_month, 40, "higher", "دستگاه", "مبنای فعلی")
    add("sales", "فشار صف لید", backlog_pressure, 250, "lower", "شاخص", "Proxy", "فروش روزانه ظرفیت پردازش لید نیست؛ فقط شاخص فشار است.")

    add("development", "تکمیل Sprint", d["sprint_completion"], 0.90, "higher", "percent", "آزمایشی")
    add("development", "باگ باز", d["open_bugs"], 8, "lower", "عدد", "آزمایشی")
    add("development", "باگ Critical", d["critical_bugs"], 0.5, "lower", "عدد", "آزمایشی")
    add("development", "آمادگی Release", d["release_readiness"], 0.95, "higher", "percent", "آزمایشی")

    add("hr", "نرخ حضور", d["attendance_rate"], 0.95, "higher", "percent", "آزمایشی")
    add("hr", "موقعیت شغلی باز", d["open_positions"], 1, "lower", "عدد", "آزمایشی")
    add("hr", "نرخ خروج ماهانه", d["monthly_turnover_rate"], 0.03, "lower", "percent", "آزمایشی")
    add("hr", "نرخ انجام به‌موقع تسک", d["task_on_time_rate"], 0.90, "higher", "percent", "آزمایشی")

    total_qc = max(d["qc_ready"] + d["qc_pending"] + d["qc_rejected"] + d["qc_rework"], 1)
    reject_rate = d["qc_rejected"] / total_qc
    add("qc", "دستگاه آماده خروج", d["qc_ready"], 100, "higher", "دستگاه", "آزمایشی")
    add("qc", "صف QC", d["qc_pending"], 20, "lower", "دستگاه", "آزمایشی")
    add("qc", "نرخ قبولی QC", d["qc_pass_rate"], 0.95, "higher", "percent", "آزمایشی")
    add("qc", "نرخ Reject", reject_rate, 0.03, "lower", "percent", "آزمایشی")

    add("marketing", "استوری روزانه", stories, 9, "higher", "عدد", "مبنای فعلی")
    add("marketing", "ریلز روزانه", reels, 1, "higher", "عدد", "مبنای فعلی")
    add("marketing", "فروش منتسب به محتوا / روز", content_sales, 3, "higher", "دستگاه", "تخمینی")
    add("marketing", "فالوئر اینستاگرام", followers, max(followers, 1), "higher", "نفر", "Snapshot")

    return pd.DataFrame(rows)


def department_summary(scenario: Any, kpis: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> pd.DataFrame:
    frame = department_kpis(scenario, kpis, overrides)
    summaries = []
    for item in DEPARTMENTS:
        dept = frame[frame["department"] == item["key"]]
        score = float(dept["score"].mean()) if not dept.empty else 0.0
        attention = int((dept["status"] != "عادی").sum()) if not dept.empty else 0
        summaries.append({
            "department": item["key"],
            "department_name": item["name"],
            "icon": item["icon"],
            "score": round(score, 1),
            "status": _status_from_score(score),
            "attention_count": attention,
            "kpi_count": int(len(dept)),
        })
    return pd.DataFrame(summaries)


def organization_score(scenario: Any, kpis: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    summary = department_summary(scenario, kpis, overrides)
    score = float(summary["score"].mean()) if not summary.empty else 0.0
    return {
        "score": round(score, 1),
        "status": _status_from_score(score),
        "departments_needing_attention": int((summary["status"] != "عادی").sum()),
        "department_count": int(len(summary)),
    }


def recommended_tasks(scenario: Any, kpis: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> pd.DataFrame:
    frame = department_kpis(scenario, kpis, overrides)
    actions = []
    action_map = {
        "درآمد ماهانه": "بررسی شکاف درآمد و تصمیم درباره کمپین / ظرفیت فروش",
        "نرخ وصول": "بررسی مطالبات و برنامه وصول هفتگی",
        "فروش ماهانه": "بررسی ظرفیت تیم فروش و کیفیت لید",
        "فشار صف لید": "تعریف Lead Stage و ظرفیت واقعی تماس روزانه",
        "تکمیل Sprint": "بازبینی Scope و Blockerهای Sprint",
        "باگ Critical": "اولویت‌بندی فوری باگ‌های Critical",
        "نرخ حضور": "بررسی علت غیبت و ظرفیت نیروی انسانی",
        "نرخ انجام به‌موقع تسک": "بازبینی بار کاری و تعریف SLA تسک",
        "صف QC": "افزایش ظرفیت QC یا اصلاح برنامه تولید",
        "نرخ قبولی QC": "تحلیل علت Rework / Reject قبل از افزایش تولید",
        "فروش منتسب به محتوا / روز": "راه‌اندازی Attribution واقعی Content → Lead → Sale",
    }
    for _, row in frame.iterrows():
        if row["status"] == "عادی":
            continue
        task = action_map.get(row["metric"], f"بررسی KPI «{row['metric']}» و تعیین اقدام اصلاحی")
        actions.append({
            "بخش": row["department_name"],
            "KPI": row["metric"],
            "وضعیت": row["status"],
            "تسک پیشنهادی": task,
            "منبع": row["source"],
        })
    return pd.DataFrame(actions)


def automation_checks(
    scenario: Any,
    kpis: Dict[str, Any],
    overrides: Dict[str, Any] | None = None,
    revenue_trigger_ratio: float = 0.90,
) -> pd.DataFrame:
    d = management_defaults(overrides)
    monthly_revenue = _kpi_value(kpis, "monthly_revenue")
    monthly_units = _kpi_value(kpis, "monthly_units")
    lead_backlog = _scenario_value(scenario, ["lead_backlog"], 4000)
    rows: List[Dict[str, Any]] = []

    def trigger(name: str, dept: str, active: bool, severity: str, condition: str, action: str, source: str):
        rows.append({
            "قانون": name,
            "بخش": dept,
            "فعال شده": bool(active),
            "شدت": severity if active else "—",
            "شرط": condition,
            "اقدام پیشنهادی": action,
            "منبع": source,
        })

    revenue_floor = d["revenue_target"] * float(np.clip(revenue_trigger_ratio, 0.1, 1.5))
    trigger(
        "افت درآمد",
        "مدیریت / فروش / مارکتینگ",
        monthly_revenue < revenue_floor,
        "بالا",
        f"Revenue < {revenue_trigger_ratio:.0%} Target",
        "باز کردن Campaign Planner و ساخت سناریوی جشنواره با کنترل حاشیه سود و موجودی",
        "Revenue مدل + Target آزمایشی",
    )
    trigger(
        "فروش زیر هدف",
        "فروش",
        monthly_units < d["monthly_units_target"] * 0.9,
        "متوسط",
        "Units < 90% Target",
        "بررسی کیفیت لید، ظرفیت تماس و Online Sales",
        "محاسبه‌شده + Target آزمایشی",
    )
    trigger(
        "صف لید بالا",
        "فروش",
        lead_backlog >= 3500,
        "متوسط",
        "Lead Backlog ≥ 3500",
        "اندازه‌گیری Calls/Answered/Qualified و ساخت SLA پردازش لید",
        "مبنای فعلی",
    )
    trigger(
        "صف QC بالا",
        "QC دستگاه",
        d["qc_pending"] > 25,
        "بالا",
        "QC Pending > 25",
        "بازبینی ظرفیت QC قبل از افزایش Production Order",
        "آزمایشی",
    )
    trigger(
        "Reject QC بالا",
        "QC دستگاه / تولید",
        d["qc_rejected"] > 5,
        "متوسط",
        "QC Reject > 5",
        "Root Cause روی ایرادهای پرتکرار و کنترل Rework",
        "آزمایشی",
    )
    trigger(
        "وصول پایین",
        "حسابداری",
        d["cash_collection_rate"] < 0.90,
        "متوسط",
        "Collection Rate < 90%",
        "ساخت لیست وصول و گزارش مطالبات سررسیدگذشته",
        "آزمایشی",
    )
    return pd.DataFrame(rows)


def production_plan(
    target_units: int,
    ready_inventory: int,
    qc_pending: int,
    qc_pass_rate: float,
    daily_capacity: int,
    horizon_days: int,
    safety_stock_pct: float = 0.10,
) -> Dict[str, Any]:
    target_units = max(int(target_units), 0)
    ready_inventory = max(int(ready_inventory), 0)
    qc_pending = max(int(qc_pending), 0)
    daily_capacity = max(int(daily_capacity), 0)
    horizon_days = max(int(horizon_days), 1)
    qc_pass_rate = float(np.clip(qc_pass_rate, 0.0, 1.0))
    safety_stock_pct = float(np.clip(safety_stock_pct, 0.0, 1.0))

    expected_qc_release = int(round(qc_pending * qc_pass_rate))
    available_after_qc = ready_inventory + expected_qc_release
    protected_target = int(ceil(target_units * (1.0 + safety_stock_pct)))
    production_order = max(protected_target - available_after_qc, 0)
    max_production = daily_capacity * horizon_days
    capacity_gap = max(production_order - max_production, 0)

    if production_order == 0:
        status = "موجودی کافی"
    elif capacity_gap == 0:
        status = "قابل تولید در بازه"
    else:
        status = "نیازمند افزایش ظرفیت / زمان"

    return {
        "target_units": target_units,
        "protected_target": protected_target,
        "expected_qc_release": expected_qc_release,
        "available_after_qc": available_after_qc,
        "production_order": production_order,
        "max_production_in_horizon": max_production,
        "capacity_gap": capacity_gap,
        "status": status,
    }


def campaign_plan(
    current_revenue: float,
    revenue_target: float,
    regular_price: float,
    campaign_price: float,
    unit_cost: float,
    campaign_days: int,
    daily_sales_target: int,
    ready_inventory: int,
    qc_pending: int,
    qc_pass_rate: float,
    production_daily_capacity: int,
    min_margin_pct: float,
    safety_stock_pct: float = 0.10,
) -> Dict[str, Any]:
    current_revenue = max(_safe_float(current_revenue), 0.0)
    revenue_target = max(_safe_float(revenue_target), 0.0)
    regular_price = max(_safe_float(regular_price), 1.0)
    campaign_price = max(_safe_float(campaign_price), 1.0)
    unit_cost = max(_safe_float(unit_cost), 0.0)
    campaign_days = max(int(campaign_days), 1)
    daily_sales_target = max(int(daily_sales_target), 0)
    min_margin_pct = float(np.clip(min_margin_pct, 0.0, 0.95))

    revenue_gap = max(revenue_target - current_revenue, 0.0)
    gap_ratio = revenue_gap / revenue_target if revenue_target > 0 else 0.0
    margin = (campaign_price - unit_cost) / campaign_price if campaign_price > 0 else -1.0
    gross_profit_per_unit = campaign_price - unit_cost
    min_allowed_price = unit_cost / max(1.0 - min_margin_pct, 0.01)
    discount_pct = max((regular_price - campaign_price) / regular_price, 0.0)
    max_discount_pct = max(1.0 - min_allowed_price / regular_price, 0.0)

    units_for_revenue_gap = int(ceil(revenue_gap / campaign_price)) if revenue_gap > 0 else 0
    units_from_sales_target = campaign_days * daily_sales_target
    target_units = max(units_for_revenue_gap, units_from_sales_target)

    prod = production_plan(
        target_units=target_units,
        ready_inventory=ready_inventory,
        qc_pending=qc_pending,
        qc_pass_rate=qc_pass_rate,
        daily_capacity=production_daily_capacity,
        horizon_days=campaign_days,
        safety_stock_pct=safety_stock_pct,
    )

    campaign_revenue = target_units * campaign_price
    gross_profit = target_units * gross_profit_per_unit
    margin_ok = margin >= min_margin_pct and gross_profit_per_unit > 0
    inventory_ok = prod["capacity_gap"] == 0

    if revenue_gap <= 0:
        recommendation = "فشار فوری برای جشنواره از سمت Revenue دیده نمی‌شود. کمپین فقط با هدف استراتژیک اجرا شود."
        readiness = "اختیاری"
    elif not margin_ok:
        recommendation = "قیمت پیشنهادی کف سود را نقض می‌کند؛ قبل از اجرا قیمت یا هزینه باید اصلاح شود."
        readiness = "غیرقابل اجرا با قیمت فعلی"
    elif not inventory_ok:
        recommendation = "فاصله درآمد وجود دارد اما ظرفیت تولید/موجودی برای Target کافی نیست؛ ابتدا Production Order اصلاح شود."
        readiness = "نیازمند آماده‌سازی"
    else:
        recommendation = "با فرض‌های فعلی جشنواره قابل سناریوسازی است؛ قبل از اجرا Demand و Attribution واقعی تأیید شود."
        readiness = "آماده سناریو"

    if gap_ratio >= 0.20:
        names = ["شتاب فروش", "پالس فروش", "فرصت نیک"]
    elif gap_ratio >= 0.08:
        names = ["فرصت نیک", "موج فروش", "جهش نیک"]
    else:
        names = ["پیشنهاد نیک", "فرصت محدود", "روز نیک"]

    return {
        "revenue_gap": revenue_gap,
        "gap_ratio": gap_ratio,
        "target_units": target_units,
        "units_for_revenue_gap": units_for_revenue_gap,
        "campaign_revenue": campaign_revenue,
        "gross_profit": gross_profit,
        "gross_profit_per_unit": gross_profit_per_unit,
        "margin": margin,
        "margin_ok": margin_ok,
        "min_allowed_price": min_allowed_price,
        "discount_pct": discount_pct,
        "max_discount_pct": max_discount_pct,
        "readiness": readiness,
        "recommendation": recommendation,
        "name_suggestions": names,
        **prod,
    }
