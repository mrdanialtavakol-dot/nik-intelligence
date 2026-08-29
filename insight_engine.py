from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from data_generator import Scenario


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
        direction = "increased" if change >= 0 else "decreased"
        insights.append(
            {
                "type": "Trend",
                "title": "Revenue movement",
                "text": f"Synthetic historical revenue {direction} {abs(change):.1%} versus the previous month.",
            }
        )

    insights.append(
        {
            "type": "Channel Mix",
            "title": "Phone channel concentration",
            "text": f"Phone sales represent approximately {kpis['phone_share']:.1%} of current scenario units.",
        }
    )

    days_to_process = scenario.lead_backlog / scenario.daily_phone_sales if scenario.daily_phone_sales > 0 else np.inf
    if np.isfinite(days_to_process):
        severity = "high" if days_to_process > 90 else "moderate" if days_to_process > 30 else "manageable"
        insights.append(
            {
                "type": "Capacity Proxy",
                "title": "Lead backlog vs sales capacity",
                "text": (
                    f"The backlog-to-daily-phone-sales ratio is {days_to_process:.0f} days, which appears {severity}. "
                    "This is a capacity proxy, not a true lead-processing SLA."
                ),
            }
        )

    insights.append(
        {
            "type": "Conversion",
            "title": "Lead to purchase scenario rate",
            "text": f"Current scenario lead-to-purchase conversion is approximately {kpis['lead_purchase_conversion']:.1%}.",
        }
    )

    insights.append(
        {
            "type": "Content",
            "title": "Estimated / Synthetic Attribution",
            "text": (
                f"Content-attributed sales are modeled at about {kpis['content_monthly_sales']:.0f} units per month. "
                "Real attribution requires source tracking and a defined attribution model."
            ),
        }
    )

    insights.append(
        {
            "type": "Customer Risk",
            "title": "Prototype churn risk",
            "text": f"High or very-high risk customers represent {risk_stats['high_or_very_high_share']:.1%} of the synthetic customer base.",
        }
    )

    sales_anomaly_count = int(sales_anomalies["is_anomaly"].sum()) if "is_anomaly" in sales_anomalies else 0
    sms_anomaly_count = int(sms_anomalies["is_anomaly"].sum()) if "is_anomaly" in sms_anomalies else 0
    if sales_anomaly_count:
        insights.append(
            {
                "type": "Alert",
                "title": "Sales anomaly detected",
                "text": f"Rolling Z-score detection flagged {sales_anomaly_count} unusual sales-revenue observations in the synthetic history.",
            }
        )
    if sms_anomaly_count:
        insights.append(
            {
                "type": "Alert",
                "title": "SMS delivery anomaly detected",
                "text": f"Rolling Z-score detection flagged {sms_anomaly_count} unusual SMS delivery-rate observations.",
            }
        )

    return insights
