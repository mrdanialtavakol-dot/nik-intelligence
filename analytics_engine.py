from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from data_generator import Scenario


def current_kpis(scenario: Scenario, customers: pd.DataFrame) -> Dict[str, float]:
    monthly_phone = scenario.monthly_phone_units
    monthly_online = float(scenario.monthly_online_sales)
    monthly_units = monthly_phone + monthly_online
    asp = scenario.average_selling_price
    monthly_revenue = monthly_units * asp
    conversion = monthly_units / scenario.lead_backlog if scenario.lead_backlog > 0 else 0.0
    active_customers = int((customers["customer_status"] == "Active").sum()) if "customer_status" in customers else 0
    return {
        "monthly_revenue": monthly_revenue,
        "monthly_units": monthly_units,
        "active_customers": active_customers,
        "lead_pool": float(scenario.lead_backlog),
        "average_selling_price": asp,
        "lead_purchase_conversion": conversion,
        "monthly_phone_units": monthly_phone,
        "monthly_online_units": monthly_online,
        "phone_share": monthly_phone / monthly_units if monthly_units else 0.0,
        "online_share": monthly_online / monthly_units if monthly_units else 0.0,
        "content_monthly_sales": scenario.content_sales_per_day * scenario.sales_days_per_month,
        "stories_month": scenario.stories_per_day * scenario.sales_days_per_month,
        "content_sales_rate": (
            scenario.content_sales_per_day / scenario.stories_per_day if scenario.stories_per_day else 0.0
        ),
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
    total = scenario.monthly_units
    a_units = total * scenario.plan_a_share
    b_units = total - a_units
    return pd.DataFrame(
        {
            "plan": ["Plan A", "Plan B"],
            "units": [a_units, b_units],
            "share": [scenario.plan_a_share, scenario.plan_b_share],
            "unit_price": [scenario.price_plan_a, scenario.price_plan_b],
            "revenue": [a_units * scenario.price_plan_a, b_units * scenario.price_plan_b],
        }
    )


def lead_funnel(scenario: Scenario) -> pd.DataFrame:
    new_leads = int(max(0, scenario.lead_backlog))
    contacted = int(new_leads * 0.86)
    qualified = int(contacted * 0.62)
    interested = int(qualified * 0.58)
    purchased = int(min(round(scenario.monthly_units), interested))
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
    # This is a capacity proxy, not a true processing-time estimate: sales are used as a stand-in for throughput.
    daily_capacity_proxy = max(float(scenario.daily_phone_sales), 0.0)
    days = scenario.lead_backlog / daily_capacity_proxy if daily_capacity_proxy > 0 else np.inf
    return {
        "backlog": float(scenario.lead_backlog),
        "daily_capacity_proxy": daily_capacity_proxy,
        "estimated_days": days,
        "estimated_weeks": days / 7 if np.isfinite(days) else np.inf,
        "estimated_months": days / 30 if np.isfinite(days) else np.inf,
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
    return {
        "stories_per_day": float(scenario.stories_per_day),
        "stories_per_month": float(scenario.stories_per_day * scenario.sales_days_per_month),
        "estimated_sales_per_day": float(scenario.content_sales_per_day),
        "estimated_sales_per_month": float(scenario.content_sales_per_day * scenario.sales_days_per_month),
        "sales_per_story": (
            float(scenario.content_sales_per_day / scenario.stories_per_day)
            if scenario.stories_per_day
            else 0.0
        ),
    }
