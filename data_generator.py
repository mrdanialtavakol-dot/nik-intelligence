from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from business_data import BUSINESS_BASELINE, REEL_SNAPSHOT


@dataclass(frozen=True)
class Scenario:
    price_plan_a: float = float(BUSINESS_BASELINE["plan_a_price"])
    price_plan_b: float = float(BUSINESS_BASELINE["plan_b_price"])
    plan_a_share: float = float(BUSINESS_BASELINE["plan_a_share"])
    daily_phone_sales: int = int(BUSINESS_BASELINE["daily_phone_sales"])
    monthly_online_sales: int = int(BUSINESS_BASELINE["monthly_online_sales"])
    lead_backlog: int = int(BUSINESS_BASELINE["lead_backlog"])
    stories_per_day: int = int(BUSINESS_BASELINE["stories_per_day"])
    reels_per_day: int = int(BUSINESS_BASELINE["reels_per_day"])
    content_sales_per_day: float = float(BUSINESS_BASELINE["estimated_content_sales_per_day"])
    instagram_followers: int = int(BUSINESS_BASELINE["instagram_followers"])
    content_team_size: int = int(BUSINESS_BASELINE["content_team_size"])
    synthetic_customer_count: int = 5_000
    history_months: int = 12
    sales_days_per_month: int = int(BUSINESS_BASELINE["sales_days_per_month"])
    seed: int = 42

    @property
    def plan_b_share(self) -> float:
        return 1.0 - self.plan_a_share

    @property
    def average_selling_price(self) -> float:
        return self.price_plan_a * self.plan_a_share + self.price_plan_b * self.plan_b_share

    @property
    def monthly_phone_units(self) -> float:
        return float(self.daily_phone_sales * self.sales_days_per_month)

    @property
    def monthly_units(self) -> float:
        return self.monthly_phone_units + float(self.monthly_online_sales)

    @property
    def monthly_revenue(self) -> float:
        return self.monthly_units * self.average_selling_price

    @property
    def total_content_per_day(self) -> float:
        return float(self.stories_per_day + self.reels_per_day)


CITIES = [
    "Tehran", "Mashhad", "Isfahan", "Shiraz", "Tabriz", "Karaj",
    "Qom", "Ahvaz", "Rasht", "Kerman", "Yazd", "Urmia",
]

INDUSTRIES = [
    "Retail", "Fashion", "Cafe & Restaurant", "Beauty", "Clinic",
    "Real Estate", "Wholesale", "Automotive", "Education", "Services",
]

LEAD_SOURCES = ["Instagram", "Phone", "Website", "Referral", "SMS", "Organic"]


def _date_window(months: int) -> Tuple[pd.Timestamp, pd.DatetimeIndex]:
    end = pd.Timestamp.today().normalize()
    days = max(90, int(months * 30.5))
    dates = pd.date_range(end=end, periods=days, freq="D")
    return end, dates


def generate_sales_data(scenario: Scenario) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed)
    _, dates = _date_window(scenario.history_months)
    n = len(dates)

    trend = np.linspace(0.82, 1.0, n)
    weekly = np.array([0.92 if d.weekday() == 4 else 1.05 if d.weekday() in (0, 1) else 1.0 for d in dates])
    phone_lambda = np.clip(scenario.daily_phone_sales * trend * weekly, 0.05, None)
    online_lambda = np.clip((scenario.monthly_online_sales / max(scenario.sales_days_per_month, 1)) * trend, 0.02, None)

    phone_units = rng.poisson(phone_lambda)
    online_units = rng.poisson(online_lambda)

    if n >= 45:
        anomaly_idx = rng.choice(np.arange(10, n - 5), size=min(3, max(1, n // 120)), replace=False)
        phone_units[anomaly_idx] = np.floor(phone_units[anomaly_idx] * rng.uniform(0.25, 0.55, len(anomaly_idx))).astype(int)

    rows = []
    for i, date in enumerate(dates):
        for channel, units in (("Phone", int(phone_units[i])), ("Online", int(online_units[i]))):
            if units <= 0:
                continue
            plan_a_units = int(rng.binomial(units, scenario.plan_a_share))
            plan_b_units = units - plan_a_units
            if plan_a_units:
                rows.append(
                    {
                        "date": date,
                        "channel": channel,
                        "plan": "Plan A",
                        "units": plan_a_units,
                        "unit_price": scenario.price_plan_a,
                        "revenue": plan_a_units * scenario.price_plan_a,
                    }
                )
            if plan_b_units:
                rows.append(
                    {
                        "date": date,
                        "channel": channel,
                        "plan": "Plan B",
                        "units": plan_b_units,
                        "unit_price": scenario.price_plan_b,
                        "revenue": plan_b_units * scenario.price_plan_b,
                    }
                )

    sales = pd.DataFrame(rows)
    if sales.empty:
        sales = pd.DataFrame(columns=["date", "channel", "plan", "units", "unit_price", "revenue"])
    return sales


def generate_customer_data(scenario: Scenario) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + 11)
    n = max(500, int(scenario.synthetic_customer_count))
    today = pd.Timestamp.today().normalize()

    customer_ids = [f"C{idx:06d}" for idx in range(1, n + 1)]
    signup_days_ago = rng.integers(7, max(365, scenario.history_months * 31 + 180), size=n)
    signup_date = today - pd.to_timedelta(signup_days_ago, unit="D")

    plan_a = rng.random(n) < scenario.plan_a_share
    plan = np.where(plan_a, "Plan A", "Plan B")
    plan_price = np.where(plan_a, scenario.price_plan_a, scenario.price_plan_b)

    latent_value = rng.gamma(shape=2.0, scale=1.2, size=n)
    purchase_count = np.maximum(1, rng.poisson(1.2 + latent_value * 0.75))
    recency = np.clip((rng.gamma(2.2, 32, size=n) / np.maximum(0.65, latent_value)).astype(int), 0, 365)
    last_activity = today - pd.to_timedelta(recency, unit="D")

    sms_usage = np.maximum(0, rng.poisson(120 + latent_value * 180)).astype(int)
    nikpos_usage = np.maximum(0, rng.poisson(10 + latent_value * 18)).astype(int)
    service_multiplier = rng.uniform(0.92, 1.22, size=n)
    revenue = purchase_count * plan_price * service_multiplier

    frequency = purchase_count.astype(float)
    monetary = revenue.astype(float)
    activity_score = (
        0.45 * (1 / (1 + recency / 30))
        + 0.25 * np.clip(frequency / np.percentile(frequency, 95), 0, 1)
        + 0.15 * np.clip(sms_usage / np.percentile(sms_usage, 95), 0, 1)
        + 0.15 * np.clip(nikpos_usage / np.percentile(nikpos_usage, 95), 0, 1)
    )

    status = np.select(
        [recency <= 30, recency <= 90, recency <= 180],
        ["Active", "Cooling", "At Risk"],
        default="Inactive",
    )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": signup_date,
            "city": rng.choice(CITIES, size=n),
            "industry": rng.choice(INDUSTRIES, size=n),
            "plan": plan,
            "last_activity": last_activity,
            "purchase_count": purchase_count,
            "revenue": np.round(revenue, 0),
            "sms_usage": sms_usage,
            "nikpos_usage": nikpos_usage,
            "recency": recency,
            "frequency": frequency,
            "monetary_value": np.round(monetary, 0),
            "lead_source": rng.choice(LEAD_SOURCES, size=n, p=[0.30, 0.18, 0.20, 0.10, 0.12, 0.10]),
            "customer_status": status,
            "activity_score": np.round(activity_score, 4),
        }
    )

    if n >= 1000:
        miss_idx = rng.choice(n, size=max(2, n // 500), replace=False)
        customers.loc[miss_idx, "city"] = np.nan
        dup_rows = customers.sample(n=max(1, n // 1000), random_state=scenario.seed)
        customers = pd.concat([customers, dup_rows], ignore_index=True)

    return customers


def generate_sms_data(scenario: Scenario, customers: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + 22)
    _, dates = _date_window(scenario.history_months)
    customer_factor = max(1.0, len(customers) / 5_000)
    baseline_sent = 5_000 * customer_factor
    seasonal = 1 + 0.08 * np.sin(np.arange(len(dates)) / 10)
    sent = np.maximum(100, rng.normal(baseline_sent * seasonal, baseline_sent * 0.08)).astype(int)
    delivery_rate = np.clip(rng.normal(0.975, 0.008, len(dates)), 0.85, 0.995)

    if len(dates) >= 60:
        dip_idx = rng.choice(np.arange(20, len(dates) - 5), size=min(2, max(1, len(dates) // 180)), replace=False)
        delivery_rate[dip_idx] = rng.uniform(0.82, 0.90, len(dip_idx))

    delivered = np.floor(sent * delivery_rate).astype(int)
    clicks = np.floor(delivered * np.clip(rng.normal(0.12, 0.025, len(dates)), 0.03, 0.25)).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "sent": sent,
            "delivered": delivered,
            "delivery_rate": np.round(delivered / sent, 4),
            "clicks": clicks,
            "click_rate": np.round(clicks / np.maximum(delivered, 1), 4),
        }
    )


def generate_nikpos_data(scenario: Scenario, customers: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + 33)
    sample_n = min(len(customers), max(800, int(len(customers) * 0.45)))
    sample = customers.sample(sample_n, random_state=scenario.seed).copy()
    sample["device_id"] = [f"NP{idx:06d}" for idx in range(1, sample_n + 1)]
    sample["activation_date"] = sample["signup_date"] + pd.to_timedelta(rng.integers(0, 14, sample_n), unit="D")
    sample["captures_30d"] = np.maximum(0, rng.poisson(35 + sample["activity_score"].to_numpy() * 90)).astype(int)
    sample["sms_actions_30d"] = np.maximum(0, rng.poisson(20 + sample["activity_score"].to_numpy() * 55)).astype(int)
    sample["active_device"] = sample["recency"] <= 120
    return sample[
        [
            "device_id", "customer_id", "activation_date", "plan", "city", "industry",
            "captures_30d", "sms_actions_30d", "active_device",
        ]
    ].reset_index(drop=True)


def generate_subscriptions(customers: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 44)
    n = len(customers)
    return pd.DataFrame(
        {
            "customer_id": customers["customer_id"].astype(str).to_numpy(),
            "subscription_status": np.where(customers["recency"].to_numpy() <= 150, "Active", "Expired"),
            "months_active": np.maximum(1, rng.integers(1, 25, n)),
            "auto_renew": rng.random(n) < 0.58,
        }
    )


def generate_leads_data(scenario: Scenario) -> pd.DataFrame:
    rng = np.random.default_rng(scenario.seed + 55)
    n = int(max(100, scenario.lead_backlog))
    created = pd.Timestamp.today().normalize() - pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    scores = rng.beta(2.2, 2.8, n)
    stages = np.select(
        [scores > 0.80, scores > 0.62, scores > 0.42, scores > 0.20],
        ["Purchased", "Interested", "Qualified", "Contacted"],
        default="New Lead",
    )
    return pd.DataFrame(
        {
            "lead_id": [f"L{idx:06d}" for idx in range(1, n + 1)],
            "created_date": created,
            "lead_source": rng.choice(LEAD_SOURCES, n),
            "lead_score": np.round(scores, 4),
            "stage": stages,
        }
    )


def generate_all(scenario: Scenario) -> Dict[str, pd.DataFrame]:
    customers = generate_customer_data(scenario)
    return {
        "sales": generate_sales_data(scenario),
        "customers": customers,
        "leads": generate_leads_data(scenario),
        "sms": generate_sms_data(scenario, customers),
        "nikpos": generate_nikpos_data(scenario, customers),
        "subscriptions": generate_subscriptions(customers, scenario.seed),
        "content_snapshot": REEL_SNAPSHOT.copy(),
    }
