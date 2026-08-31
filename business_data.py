from __future__ import annotations

import pandas as pd


BUSINESS_BASELINE = {
    "snapshot_date": "2026-08-31",
    "plan_a_price": 15_000_000,
    "plan_b_price": 30_000_000,
    "plan_a_share": 0.50,
    "daily_phone_sales": 10,
    "monthly_online_sales": 20,
    "lead_backlog": 5_490,
    "stories_per_day": 9,
    "reels_per_day": 1,
    "estimated_content_sales_per_day": 2.0,
    "instagram_followers": 207_000,
    "content_team_size": 5,
    "sales_days_per_month": 30,
}

REEL_SNAPSHOT = pd.DataFrame(
    [
        {"reel": "ریلز ۱", "views": 69_700, "comments": 694, "shares": 1_953},
        {"reel": "ریلز ۲", "views": 4_928, "comments": 24, "shares": 8},
        {"reel": "ریلز ۳", "views": 7_014, "comments": 28, "shares": 14},
        {"reel": "ریلز ۴", "views": 7_865, "comments": 48, "shares": 51},
        {"reel": "ریلز ۵", "views": 46_900, "comments": 1_963, "shares": 782},
        {"reel": "ریلز ۶", "views": 46_100, "comments": 86, "shares": 147},
        {"reel": "ریلز ۷", "views": 5_862, "comments": 14, "shares": 28},
        {"reel": "ریلز ۸", "views": 11_000, "comments": 213, "shares": 128},
        {"reel": "ریلز ۹", "views": 189_000, "comments": 3_467, "shares": 7_225},
        {"reel": "ریلز ۱۰", "views": 9_040, "comments": 110, "shares": 99},
    ]
)
REEL_SNAPSHOT["interactions"] = REEL_SNAPSHOT["comments"] + REEL_SNAPSHOT["shares"]
REEL_SNAPSHOT["interaction_rate"] = REEL_SNAPSHOT["interactions"] / REEL_SNAPSHOT["views"]
REEL_SNAPSHOT["share_rate"] = REEL_SNAPSHOT["shares"] / REEL_SNAPSHOT["views"]
REEL_SNAPSHOT["comment_rate"] = REEL_SNAPSHOT["comments"] / REEL_SNAPSHOT["views"]


PRICE_HISTORY = pd.DataFrame(
    [
        {"period": "اوایل پروژه ۲۰۲۵", "list_price": 8_000_000, "campaign_price": 8_000_000, "type": "تاریخی"},
        {"period": "مرحله بعد", "list_price": 10_000_000, "campaign_price": 10_000_000, "type": "تاریخی"},
        {"period": "مرحله بعد", "list_price": 15_000_000, "campaign_price": 15_000_000, "type": "تاریخی"},
        {"period": "قیمت پایه کمپین‌ها", "list_price": 20_000_000, "campaign_price": 20_000_000, "type": "تاریخی"},
        {"period": "جشنواره ۱۴۰۵", "list_price": 20_000_000, "campaign_price": 12_000_000, "type": "کمپین"},
        {"period": "نجات فروش", "list_price": 20_000_000, "campaign_price": 13_000_000, "type": "کمپین"},
        {"period": "Baseline فعلی A", "list_price": 15_000_000, "campaign_price": 15_000_000, "type": "فعلی"},
        {"period": "Baseline فعلی B", "list_price": 30_000_000, "campaign_price": 30_000_000, "type": "فعلی"},
    ]
)


BACKLOG_SNAPSHOTS = pd.DataFrame(
    [
        {"snapshot": "کمپین", "lead_backlog": 2_644},
        {"snapshot": "کمپین", "lead_backlog": 3_800},
        {"snapshot": "Baseline قبلی", "lead_backlog": 4_000},
        {"snapshot": "Baseline فعلی 2026-08-31", "lead_backlog": 5_490},
        {"snapshot": "کمپین", "lead_backlog": 4_500},
    ]
)


def reel_snapshot_metrics() -> dict[str, float]:
    df = REEL_SNAPSHOT
    top3_views = float(df.nlargest(3, "views")["views"].sum())
    total_views = float(df["views"].sum())
    return {
        "reel_count": float(len(df)),
        "total_views": total_views,
        "average_views": float(df["views"].mean()),
        "median_views": float(df["views"].median()),
        "min_views": float(df["views"].min()),
        "max_views": float(df["views"].max()),
        "total_comments": float(df["comments"].sum()),
        "total_shares": float(df["shares"].sum()),
        "total_interactions": float(df["interactions"].sum()),
        "interaction_rate": float(df["interactions"].sum() / total_views) if total_views else 0.0,
        "top3_view_share": float(top3_views / total_views) if total_views else 0.0,
        "best_share_rate": float(df["share_rate"].max()),
        "best_comment_rate": float(df["comment_rate"].max()),
    }
