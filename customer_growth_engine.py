from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import zlib

V10_VERSION = "V0.10"
CUSTOMER_BUSINESS_PAGES = {
    "Customer Growth Home": "مرکز رشد کسب‌وکار",
    "Customer Segments Portal": "مشتریان و سگمنت‌ها",
    "Smart Campaigns Portal": "کمپین‌های هوشمند",
    "Business Automations Portal": "اتوماسیون رشد",
    "Business ROI Portal": "نتیجه و بازگشت سرمایه",
    "Subscription Plans Portal": "پلن‌های اشتراک",
}

VERTICAL_PROFILES: Dict[str, Dict[str, float]] = {
    "پوشاک": {"customers": 1842, "monthly_sales": 430_000_000, "aov": 1_350_000, "capture_30d": 318, "dormant_45": 273, "repeat_dormant": 82, "vip": 146, "new_30d": 214, "due_return": 96},
    "سالن زیبایی": {"customers": 1260, "monthly_sales": 315_000_000, "aov": 980_000, "capture_30d": 247, "dormant_45": 188, "repeat_dormant": 71, "vip": 112, "new_30d": 154, "due_return": 134},
    "کافه و رستوران": {"customers": 3260, "monthly_sales": 680_000_000, "aov": 520_000, "capture_30d": 486, "dormant_45": 412, "repeat_dormant": 158, "vip": 236, "new_30d": 391, "due_return": 176},
    "کلینیک": {"customers": 980, "monthly_sales": 540_000_000, "aov": 2_450_000, "capture_30d": 126, "dormant_45": 144, "repeat_dormant": 49, "vip": 78, "new_30d": 83, "due_return": 91},
}

PLAN_CATALOG = pd.DataFrame([
    {
        "plan": "هوشمندی پایه",
        "price_monthly": 490_000,
        "best_for": "کسب‌وکار در شروع باشگاه مشتریان",
        "features": ["داشبورد مشتری", "سگمنت‌های آماده", "۳ پیشنهاد رشد در ماه", "گزارش رشد شماره‌ها", "پیشنهاد کمپین"],
        "cta": "شروع پایه",
    },
    {
        "plan": "رشد حرفه‌ای",
        "price_monthly": 990_000,
        "best_for": "کسب‌وکارهای دارای خرید تکرارشونده",
        "features": ["همه امکانات پایه", "کمپین هوشمند", "تحلیل مشتری خوابیده و VIP", "گزارش ROI", "پیشنهاد زمان ارسال", "۵ اتوماسیون فعال"],
        "cta": "انتخاب رشد",
    },
    {
        "plan": "اتوماسیون پرو",
        "price_monthly": 1_990_000,
        "best_for": "کسب‌وکارهای جدی، چندشعبه‌ای و پرتراکنش",
        "features": ["همه امکانات رشد", "اتوماسیون نامحدود", "API / n8n", "Branch Analytics", "قواعد سفارشی", "گزارش اجرایی", "پشتیبانی اولویت‌دار"],
        "cta": "فعال‌سازی پرو",
    },
])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(round(float(value))))
    except Exception:
        return max(0, int(default))


def business_snapshot(vertical: str = "پوشاک", sms_unit_cost: float = 250.0) -> Dict[str, float | str]:
    profile = VERTICAL_PROFILES.get(vertical, VERTICAL_PROFILES["پوشاک"]).copy()
    customers = _safe_int(profile["customers"])
    dormant = _safe_int(profile["dormant_45"])
    repeat_dormant = min(_safe_int(profile["repeat_dormant"]), dormant)
    aov = max(_safe_float(profile["aov"]), 0.0)
    sms_cost = max(_safe_float(sms_unit_cost, 250.0), 0.0)

    # Demo outcome. These are deterministic synthetic business assumptions, not internal NIK facts.
    returners = min(repeat_dormant, max(1, int(round(repeat_dormant * 0.378)))) if repeat_dormant else 0
    purchasers = min(returners, max(1, int(round(returners * 0.58)))) if returners else 0
    campaign_cost = repeat_dormant * sms_cost
    returned_revenue = purchasers * aov
    capture_30d = _safe_int(profile["capture_30d"])
    health_score = min(96, max(42, round(
        58
        + min(capture_30d / max(customers, 1), 0.25) * 80
        - min(dormant / max(customers, 1), 0.35) * 42
        + min(_safe_int(profile["vip"]) / max(customers, 1), 0.20) * 28
    )))

    return {
        "vertical": vertical,
        "total_customers": customers,
        "monthly_sales": _safe_float(profile["monthly_sales"]),
        "aov": aov,
        "captured_30d": capture_30d,
        "dormant_45": dormant,
        "repeat_dormant": repeat_dormant,
        "vip": _safe_int(profile["vip"]),
        "new_30d": _safe_int(profile["new_30d"]),
        "due_return": _safe_int(profile["due_return"]),
        "health_score": health_score,
        "sms_unit_cost": sms_cost,
        "suggested_campaign_target": repeat_dormant,
        "suggested_campaign_cost": campaign_cost,
        "demo_returners": returners,
        "demo_purchasers": purchasers,
        "demo_returned_revenue": returned_revenue,
        "demo_campaign_roi": ((returned_revenue - campaign_cost) / campaign_cost) if campaign_cost > 0 else 0.0,
    }


def segment_table(vertical: str = "پوشاک") -> pd.DataFrame:
    s = business_snapshot(vertical)
    total = int(s["total_customers"])
    vip = int(s["vip"])
    dormant = int(s["dormant_45"])
    new = int(s["new_30d"])
    due = int(s["due_return"])
    regular = max(0, total - vip - dormant - new)
    rows = [
        ("VIP", vip, "ارزش خرید بالا / تعامل مکرر", "دسترسی زودتر به محصول جدید"),
        ("موعد خرید مجدد", due, "الگوی زمانی خرید نشان می‌دهد زمان بازگشت رسیده", "یادآوری شخصی‌سازی‌شده"),
        ("مشتری جدید", new, "در ۳۰ روز اخیر ثبت شده", "Welcome + پیشنهاد خرید دوم"),
        ("خوابیده ۴۵+ روز", dormant, "بیش از ۴۵ روز بدون بازگشت", "کمپین بازگشت"),
        ("عادی", regular, "رفتار عادی و بدون سیگنال فوری", "پرورش تدریجی"),
    ]
    df = pd.DataFrame(rows, columns=["segment", "customers", "definition", "recommended_action"])
    df["share"] = df["customers"] / max(total, 1)
    return df


def campaign_opportunities(vertical: str = "پوشاک") -> pd.DataFrame:
    s = business_snapshot(vertical)
    aov = float(s["aov"])
    unit_cost = float(s["sms_unit_cost"])
    opportunities = [
        {
            "campaign_id": "winback_repeat",
            "title": "بازگشت مشتریان ارزشمند خوابیده",
            "audience": int(s["repeat_dormant"]),
            "reason": "بیش از ۴۵ روز برنگشته‌اند و قبلاً بیش از دو بار خرید داشته‌اند.",
            "message": "یک پیام بازگشت با پیشنهاد محدود برای خرید بعدی ارسال شود.",
            "cost": int(s["repeat_dormant"]) * unit_cost,
            "expected_purchases": int(s["demo_purchasers"]),
            "expected_value": int(s["demo_purchasers"]) * aov,
            "priority": "خیلی بالا",
        },
        {
            "campaign_id": "repeat_due",
            "title": "یادآوری زمان خرید مجدد",
            "audience": int(s["due_return"]),
            "reason": "بر اساس الگوی رفتاری، موعد منطقی خرید بعدی رسیده است.",
            "message": "یادآوری بدون تخفیف یا با مزیت کوچک برای بازگشت ارسال شود.",
            "cost": int(s["due_return"]) * unit_cost,
            "expected_purchases": max(1, int(round(int(s["due_return"]) * 0.12))),
            "expected_value": max(1, int(round(int(s["due_return"]) * 0.12))) * aov,
            "priority": "بالا",
        },
        {
            "campaign_id": "vip_drop",
            "title": "دسترسی زودهنگام برای VIP",
            "audience": int(s["vip"]),
            "reason": "مشتریان باارزش احتمال پاسخ بیشتری به موجودی/خدمت ویژه دارند.",
            "message": "اطلاع‌رسانی اختصاصی قبل از انتشار عمومی.",
            "cost": int(s["vip"]) * unit_cost,
            "expected_purchases": max(1, int(round(int(s["vip"]) * 0.16))),
            "expected_value": max(1, int(round(int(s["vip"]) * 0.16))) * aov,
            "priority": "متوسط",
        },
        {
            "campaign_id": "second_purchase",
            "title": "تبدیل مشتری جدید به خرید دوم",
            "audience": int(s["new_30d"]),
            "reason": "پنجره ۳۰ روز اول برای ساخت عادت خرید مجدد ارزشمند است.",
            "message": "پیام خوشامد/مزیت خرید دوم به مشتری جدید ارسال شود.",
            "cost": int(s["new_30d"]) * unit_cost,
            "expected_purchases": max(1, int(round(int(s["new_30d"]) * 0.10))),
            "expected_value": max(1, int(round(int(s["new_30d"]) * 0.10))) * aov,
            "priority": "متوسط",
        },
    ]
    df = pd.DataFrame(opportunities)
    df["roi_model"] = np.where(df["cost"] > 0, (df["expected_value"] - df["cost"]) / df["cost"], 0.0)
    return df


def automation_catalog(vertical: str = "پوشاک") -> pd.DataFrame:
    vertical_rule = {
        "پوشاک": "رسیدن کالکشن جدید → VIPهای مرتبط → پیام دسترسی زودهنگام",
        "سالن زیبایی": "گذشت دوره مراجعه → مشتری بدون رزرو جدید → یادآوری وقت",
        "کافه و رستوران": "۳۰ روز عدم مراجعه → مشتری پرتکرار → پیشنهاد بازگشت",
        "کلینیک": "رسیدن موعد پیگیری → مراجعه‌کننده واجد شرایط → یادآوری",
    }.get(vertical, "سیگنال رفتاری → سگمنت → پیام مناسب")
    rows = [
        ("بازگشت خودکار", "اگر ۴۵ روز از آخرین مراجعه گذشت", "ورود به سگمنت خوابیده → پیام بازگشت", "آماده"),
        ("خرید دوم", "مشتری جدید + ۷ روز بدون خرید دوم", "پیام مزیت خرید دوم", "آماده"),
        ("VIP", "مشتری وارد سطح VIP شد", "Tag VIP → پیام اختصاصی / دسترسی زودهنگام", "آماده"),
        ("تولد", "۷ روز تا تولد مشتری", "پیام تولد + پیشنهاد قابل تنظیم", "آماده"),
        ("رفتار تخصصی", vertical_rule.split(" → ")[0], vertical_rule, "پیشنهادی"),
        ("گزارش هفتگی", "هر شنبه صبح", "خلاصه رشد مشتری / کمپین / فروش برای صاحب کسب‌وکار", "آماده"),
    ]
    return pd.DataFrame(rows, columns=["automation", "trigger", "action", "status"])


def growth_trend(vertical: str = "پوشاک", months: int = 8) -> pd.DataFrame:
    s = business_snapshot(vertical)
    months = max(4, min(int(months), 18))
    rng = np.random.default_rng(zlib.crc32(vertical.encode("utf-8")))
    dates = pd.date_range(end=pd.Timestamp("2026-08-01"), periods=months, freq="MS")
    base_customers = max(300, int(s["total_customers"]) - int(s["captured_30d"]) * (months - 1))
    customer_growth = np.maximum.accumulate(base_customers + np.cumsum(np.maximum(0, rng.normal(int(s["captured_30d"]) * 0.92, 24, months))).astype(int))
    campaign_revenue = np.maximum(0, rng.normal(float(s["demo_returned_revenue"]) * 0.82, max(float(s["demo_returned_revenue"]) * 0.15, 1), months))
    repeat_rate = np.clip(np.linspace(0.18, 0.27, months) + rng.normal(0, 0.012, months), 0.10, 0.42)
    return pd.DataFrame({"month": dates, "customer_base": customer_growth, "campaign_revenue": campaign_revenue, "repeat_rate": repeat_rate})


def business_health_insights(vertical: str = "پوشاک") -> list[Dict[str, str]]:
    s = business_snapshot(vertical)
    return [
        {"title": "دارایی مشتری در حال رشد است", "detail": f"در ۳۰ روز اخیر {int(s['captured_30d']):,} شماره جدید به دارایی مشتری اضافه شده است.", "level": "good"},
        {"title": "فرصت بازگشت فوری", "detail": f"{int(s['dormant_45']):,} مشتری بیش از ۴۵ روز برنگشته‌اند؛ {int(s['repeat_dormant']):,} نفر از آن‌ها سابقه بیش از دو خرید دارند.", "level": "attention"},
        {"title": "داده برای اقدام آماده است", "detail": "مشتری VIP، موعد خرید مجدد، مشتری جدید و مشتری خوابیده از هم تفکیک شده‌اند؛ گام بعد انتخاب Action است.", "level": "good"},
    ]
