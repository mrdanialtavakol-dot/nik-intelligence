from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from business_data import BUSINESS_BASELINE, reel_snapshot_metrics
from data_generator import Scenario


# -----------------------------------------------------------------------------
# Compatibility helpers
# -----------------------------------------------------------------------------
# Streamlit Cloud can briefly run mixed revisions while a multi-file GitHub
# commit is being rebuilt. These accessors make analytics tolerant of an older
# Scenario class so the UI does not crash when newer optional fields are absent.


def _baseline(key: str, fallback):
    try:
        return BUSINESS_BASELINE.get(key, fallback)
    except Exception:
        return fallback


def _sv(scenario: Scenario, key: str, fallback):
    return getattr(scenario, key, fallback)


def _plan_b_share(scenario: Scenario) -> float:
    return float(getattr(scenario, "plan_b_share", 1.0 - float(_sv(scenario, "plan_a_share", 0.50))))


def _asp(scenario: Scenario) -> float:
    if hasattr(scenario, "average_selling_price"):
        return float(scenario.average_selling_price)
    a = float(_sv(scenario, "price_plan_a", _baseline("plan_a_price", 15_000_000)))
    b = float(_sv(scenario, "price_plan_b", _baseline("plan_b_price", 30_000_000)))
    share_a = float(_sv(scenario, "plan_a_share", _baseline("plan_a_share", 0.50)))
    return a * share_a + b * (1.0 - share_a)


def _monthly_phone_units(scenario: Scenario) -> float:
    if hasattr(scenario, "monthly_phone_units"):
        return float(scenario.monthly_phone_units)
    return float(_sv(scenario, "daily_phone_sales", _baseline("daily_phone_sales", 10))) * float(
        _sv(scenario, "sales_days_per_month", _baseline("sales_days_per_month", 30))
    )


def _monthly_units(scenario: Scenario) -> float:
    if hasattr(scenario, "monthly_units"):
        return float(scenario.monthly_units)
    return _monthly_phone_units(scenario) + float(
        _sv(scenario, "monthly_online_sales", _baseline("monthly_online_sales", 20))
    )


def _total_content_per_day(scenario: Scenario) -> float:
    if hasattr(scenario, "total_content_per_day"):
        return float(scenario.total_content_per_day)
    return float(_sv(scenario, "stories_per_day", _baseline("stories_per_day", 9))) + float(
        _sv(scenario, "reels_per_day", _baseline("reels_per_day", 1))
    )


def current_kpis(scenario: Scenario, customers: pd.DataFrame) -> Dict[str, float]:
    monthly_phone = _monthly_phone_units(scenario)
    monthly_online = float(_sv(scenario, "monthly_online_sales", _baseline("monthly_online_sales", 20)))
    monthly_units = monthly_phone + monthly_online
    asp = _asp(scenario)
    monthly_revenue = monthly_units * asp
    active_customers = int((customers["customer_status"] == "Active").sum()) if "customer_status" in customers else 0
    backlog = float(_sv(scenario, "lead_backlog", _baseline("lead_backlog", 4_000)))
    sales_days = float(_sv(scenario, "sales_days_per_month", _baseline("sales_days_per_month", 30)))
    content_sales = float(
        _sv(scenario, "content_sales_per_day", _baseline("estimated_content_sales_per_day", 2.0))
    )
    stories = float(_sv(scenario, "stories_per_day", _baseline("stories_per_day", 9)))
    reels = float(_sv(scenario, "reels_per_day", _baseline("reels_per_day", 1)))

    # Deliberately NOT called conversion: backlog is a stock and sales are a flow.
    backlog_sales_volume_ratio = monthly_units / backlog if backlog > 0 else 0.0
    backlog_months_of_sales = backlog / monthly_units if monthly_units > 0 else np.inf

    return {
        "monthly_revenue": monthly_revenue,
        "monthly_units": monthly_units,
        "active_customers": active_customers,
        "lead_pool": backlog,
        "average_selling_price": asp,
        "monthly_phone_units": monthly_phone,
        "monthly_online_units": monthly_online,
        "phone_share": monthly_phone / monthly_units if monthly_units else 0.0,
        "online_share": monthly_online / monthly_units if monthly_units else 0.0,
        "content_monthly_sales_estimated": content_sales * sales_days,
        "stories_month": stories * sales_days,
        "reels_month": reels * sales_days,
        "total_content_month": (stories + reels) * sales_days,
        "backlog_sales_volume_ratio": backlog_sales_volume_ratio,
        "backlog_months_of_sales": backlog_months_of_sales,
    }


def sales_daily(sales: pd.DataFrame) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=["date", "units", "revenue"])
    return sales.groupby("date", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))


def sales_monthly(sales: pd.DataFrame) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=["month", "units", "revenue"])
    df = sales.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    out = df.groupby("month", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))
    out["mom_growth"] = out["revenue"].pct_change()
    return out


def plan_performance(scenario: Scenario) -> pd.DataFrame:
    total = _monthly_units(scenario)
    share_a = float(_sv(scenario, "plan_a_share", _baseline("plan_a_share", 0.50)))
    share_b = 1.0 - share_a
    a_units = total * share_a
    b_units = total - a_units
    price_a = float(_sv(scenario, "price_plan_a", _baseline("plan_a_price", 15_000_000)))
    price_b = float(_sv(scenario, "price_plan_b", _baseline("plan_b_price", 30_000_000)))
    return pd.DataFrame(
        {
            "plan": ["Plan A", "Plan B"],
            "units": [a_units, b_units],
            "share": [share_a, share_b],
            "unit_price": [price_a, price_b],
            "revenue": [a_units * price_a, b_units * price_b],
        }
    )


def lead_funnel(scenario: Scenario) -> pd.DataFrame:
    # Synthetic funnel for demonstration only. Real conversion needs matched lead-stage data.
    new_leads = int(max(0, float(_sv(scenario, "lead_backlog", _baseline("lead_backlog", 4_000)))))
    contacted = int(new_leads * 0.86)
    qualified = int(contacted * 0.62)
    interested = int(qualified * 0.58)
    purchased = int(min(round(_monthly_units(scenario)), interested))
    active = int(purchased * 0.95)
    stages = ["New Leads", "Contacted", "Qualified", "Interested", "Purchased", "Active Customer"]
    values = [new_leads, contacted, qualified, interested, purchased, active]
    conversion_from_previous = [1.0]
    for prev, cur in zip(values[:-1], values[1:]):
        conversion_from_previous.append(cur / prev if prev else 0.0)
    return pd.DataFrame(
        {
            "stage": stages,
            "count": values,
            "step_conversion": conversion_from_previous,
            "overall_conversion": [v / new_leads if new_leads else 0.0 for v in values],
        }
    )


def backlog_capacity(scenario: Scenario) -> Dict[str, float]:
    monthly_sales = max(float(_monthly_units(scenario)), 0.0)
    backlog = float(_sv(scenario, "lead_backlog", _baseline("lead_backlog", 4_000)))
    ratio = backlog / monthly_sales if monthly_sales > 0 else np.inf
    return {
        "backlog": backlog,
        "monthly_sales_flow": monthly_sales,
        "sales_volume_equivalent_months": ratio,
    }


def data_quality(df: pd.DataFrame, required_columns: Iterable[str] | None = None) -> Dict[str, object]:
    record_count = len(df)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    invalid = 0

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        invalid += int((numeric.replace([np.inf, -np.inf], np.nan).isna().sum().sum()))

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        invalid += len(missing_columns) * max(record_count, 1)
    else:
        missing_columns = []

    cells = max(1, df.shape[0] * max(1, df.shape[1]))
    penalty = (missing + duplicates + invalid) / cells
    score = float(np.clip(1 - penalty, 0, 1))
    status = "Healthy" if score >= 0.97 else "Watch" if score >= 0.90 else "Needs Review"
    return {
        "record_count": record_count,
        "missing_values": missing,
        "duplicate_records": duplicates,
        "invalid_values": invalid,
        "quality_score": score,
        "status": status,
        "missing_columns": missing_columns,
    }


def quality_table(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        q = data_quality(df)
        rows.append(
            {
                "Dataset": name.title(),
                "Record Count": q["record_count"],
                "Missing Values": q["missing_values"],
                "Duplicate Records": q["duplicate_records"],
                "Invalid Values": q["invalid_values"],
                "Quality Score": q["quality_score"],
                "Status": q["status"],
            }
        )
    return pd.DataFrame(rows)


def content_metrics(scenario: Scenario) -> Dict[str, float]:
    snapshot = reel_snapshot_metrics()
    stories = float(_sv(scenario, "stories_per_day", _baseline("stories_per_day", 9)))
    reels = float(_sv(scenario, "reels_per_day", _baseline("reels_per_day", 1)))
    sales_days = float(_sv(scenario, "sales_days_per_month", _baseline("sales_days_per_month", 30)))
    estimated_sales = float(
        _sv(scenario, "content_sales_per_day", _baseline("estimated_content_sales_per_day", 2.0))
    )
    followers = float(_sv(scenario, "instagram_followers", _baseline("instagram_followers", 207_000)))
    team_size = float(_sv(scenario, "content_team_size", _baseline("content_team_size", 5)))
    total_daily = stories + reels
    return {
        "stories_per_day": stories,
        "reels_per_day": reels,
        "total_content_per_day": total_daily,
        "stories_per_month": stories * sales_days,
        "reels_per_month": reels * sales_days,
        "total_content_per_month": total_daily * sales_days,
        "estimated_sales_per_day": estimated_sales,
        "estimated_sales_per_month": estimated_sales * sales_days,
        "instagram_followers": followers,
        "content_team_size": team_size,
        **snapshot,
    }
