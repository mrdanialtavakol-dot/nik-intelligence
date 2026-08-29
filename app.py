from __future__ import annotations

import io
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics_engine import (
    backlog_capacity,
    content_metrics,
    current_kpis,
    lead_funnel,
    plan_performance,
    quality_table,
    sales_daily,
    sales_monthly,
)
from data_generator import Scenario, generate_all
from insight_engine import generate_insights
from ml_engine import churn_risk_model, detect_anomalies, revenue_forecast, segment_customers


st.set_page_config(
    page_title="NIK INTELLIGENCE V0.1",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


GOLD = "#D6B86A"
MUTED = "#8B8F98"
BG = "#090A0C"
CARD = "#111318"
BORDER = "#262A31"


st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; color: #F5F5F5; }}
    [data-testid="stSidebar"] {{ background: #0D0F13; border-right: 1px solid {BORDER}; }}
    [data-testid="stMetric"] {{
        background: linear-gradient(180deg, #14171D 0%, #101216 100%);
        border: 1px solid {BORDER};
        padding: 16px 18px;
        border-radius: 14px;
    }}
    [data-testid="stMetricLabel"] {{ color: #A8ADB7; }}
    [data-testid="stMetricValue"] {{ color: #FFFFFF; }}
    .nik-title {{ font-size: 2.1rem; font-weight: 760; letter-spacing: 0.03em; margin-bottom: 0; }}
    .nik-subtitle {{ color: #A7ABB3; margin-top: 2px; }}
    .demo-banner {{
        border: 1px solid rgba(214,184,106,0.38);
        background: rgba(214,184,106,0.08);
        color: #E7D49A;
        padding: 10px 14px;
        border-radius: 10px;
        font-weight: 650;
        margin: 8px 0 18px 0;
    }}
    .section-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }}
    .insight-card {{
        background: #111318;
        border: 1px solid #2A2E36;
        border-left: 3px solid {GOLD};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }}
    .insight-type {{ color: {GOLD}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }}
    .pipeline-step {{
        background: #111318;
        border: 1px solid #2A2E36;
        padding: 12px 14px;
        border-radius: 10px;
        text-align: center;
        font-weight: 650;
        min-height: 48px;
    }}
    .small-muted {{ color: #90959F; font-size: .86rem; }}
    hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def toman(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B T"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M T"
    return f"{value:,.0f} T"


def pct(value: float) -> str:
    return f"{value:.1%}"


def style_fig(fig, height: int = 380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D7DAE0"),
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#22262D")
    fig.update_yaxes(gridcolor="#22262D")
    return fig


@st.cache_data(show_spinner=False)
def build_synthetic_data(scenario: Scenario) -> Dict[str, pd.DataFrame]:
    return generate_all(scenario)


@st.cache_data(show_spinner=False)
def build_models(customers: pd.DataFrame, seed: int):
    segmented, segment_profile = segment_customers(customers, seed)
    risk_df, risk_stats = churn_risk_model(segmented, seed)
    return risk_df, segment_profile, risk_stats


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw))


def validate_upload(name: str, df: pd.DataFrame) -> Tuple[bool, str]:
    schemas = {
        "sales": {"date", "channel", "plan", "units", "revenue"},
        "customers": {
            "customer_id", "signup_date", "city", "industry", "plan", "last_activity",
            "purchase_count", "revenue", "sms_usage", "nikpos_usage", "recency",
            "frequency", "monetary_value", "lead_source", "customer_status",
        },
        "leads": {"lead_id", "created_date", "lead_source", "stage"},
        "sms": {"date", "sent", "delivered", "delivery_rate"},
        "nikpos": {"device_id", "customer_id", "activation_date", "plan"},
    }
    missing = schemas[name] - set(df.columns)
    if missing:
        return False, "Missing expected columns: " + ", ".join(sorted(missing))
    return True, "Schema accepted"


def scenario_sidebar() -> Tuple[Scenario, str]:
    st.sidebar.markdown("## NIK INTELLIGENCE")
    st.sidebar.caption("Automated Data Intelligence Platform | V0.1")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Overview",
            "Data Center",
            "Sales Analytics",
            "Customer Intelligence",
            "NIKPOS Analytics",
            "Content Analytics",
            "SMS Analytics",
            "Anomaly Detection",
            "Predictions",
            "Automated Insights",
            "Analysis Pipeline",
            "Settings / Scenario Controls",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Scenario Controls")
    price_a = st.sidebar.number_input("Price Plan A (Toman)", 1_000_000, 200_000_000, 15_000_000, 1_000_000)
    price_b = st.sidebar.number_input("Price Plan B (Toman)", 1_000_000, 300_000_000, 30_000_000, 1_000_000)
    share_a_pct = st.sidebar.slider("Plan A Share", 0, 100, 50, 5)
    phone = st.sidebar.number_input("Daily Phone Sales", 0, 500, 10, 1)
    online = st.sidebar.number_input("Monthly Online Sales", 0, 5_000, 20, 5)
    backlog = st.sidebar.number_input("Lead Backlog", 0, 500_000, 4_000, 100)
    stories = st.sidebar.number_input("Stories per Day", 0, 100, 9, 1)
    content_sales = st.sidebar.number_input("Content-attributed Sales / Day", 0.0, 100.0, 2.0, 0.5)
    customers = st.sidebar.number_input("Synthetic Customer Count", 500, 50_000, 5_000, 500)
    months = st.sidebar.slider("Months of Historical Data", 3, 36, 12, 1)
    sales_days = st.sidebar.slider("Sales Days / Month", 20, 31, 30, 1)
    seed = st.sidebar.number_input("Seed", 1, 999_999, 42, 1)

    scenario = Scenario(
        price_plan_a=float(price_a),
        price_plan_b=float(price_b),
        plan_a_share=float(share_a_pct) / 100,
        daily_phone_sales=int(phone),
        monthly_online_sales=int(online),
        lead_backlog=int(backlog),
        stories_per_day=int(stories),
        content_sales_per_day=float(content_sales),
        synthetic_customer_count=int(customers),
        history_months=int(months),
        sales_days_per_month=int(sales_days),
        seed=int(seed),
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("RUN ANALYSIS", use_container_width=True, type="primary"):
        stages = [
            "Loading Data",
            "Validating Data",
            "Cleaning Data",
            "Calculating KPIs",
            "Analyzing Trends",
            "Detecting Anomalies",
            "Running Models",
            "Generating Insights",
            "Analysis Complete",
        ]
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()
        for i, stage in enumerate(stages, start=1):
            status.caption(stage)
            progress.progress(int(i / len(stages) * 100))
            time.sleep(0.04)
        st.session_state["analysis_complete"] = True
        status.success("Analysis Complete")

    st.sidebar.caption("Prototype using synthetic/demo data. Not connected to NIK internal systems.")
    return scenario, page


def page_header(title: str, subtitle: str = ""):
    st.markdown('<div class="nik-title">NIK INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="nik-subtitle">Automated Data Intelligence Platform | V0.1 | Prototype / Proof of Concept</div>', unsafe_allow_html=True)
    st.markdown('<div class="demo-banner">DEMO / SYNTHETIC DATA — No NIK internal system connection</div>', unsafe_allow_html=True)
    st.header(title)
    if subtitle:
        st.caption(subtitle)


def executive_overview(scenario, data, customers_model, kpis, monthly, funnel, forecast, insights):
    page_header("Executive Overview", "Designed for a five-second management read.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly Revenue", toman(kpis["monthly_revenue"]))
    c2.metric("Monthly Units", f"{kpis['monthly_units']:,.0f}")
    c3.metric("Active Customers", f"{kpis['active_customers']:,}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Lead Pool", f"{kpis['lead_pool']:,.0f}")
    c5.metric("Average Selling Price", toman(kpis["average_selling_price"]))
    c6.metric("Lead -> Purchase", pct(kpis["lead_purchase_conversion"]))

    left, right = st.columns([1.45, 1])
    with left:
        fig = px.line(monthly, x="month", y="revenue", markers=True, title="Synthetic Revenue Trend")
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        channel = pd.DataFrame(
            {
                "Channel": ["Phone", "Online"],
                "Units": [kpis["monthly_phone_units"], kpis["monthly_online_units"]],
            }
        )
        fig = px.pie(channel, names="Channel", values="Units", hole=0.58, title="Current Channel Mix")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    f1, f2 = st.columns([1, 1.25])
    with f1:
        fig = go.Figure(go.Funnel(y=funnel["stage"], x=funnel["count"], textinfo="value+percent initial"))
        fig.update_layout(title="Synthetic Lead Funnel")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    with f2:
        fig = px.line(forecast, x="month", y="revenue", color="series", markers=True, title="Revenue History + 3-Month Forecast")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.subheader("Management Signals")
    for item in insights[:5]:
        st.markdown(
            f'<div class="insight-card"><div class="insight-type">{item["type"]}</div><b>{item["title"]}</b><br>{item["text"]}</div>',
            unsafe_allow_html=True,
        )

    st.caption("Forecasts and ML outputs are experimental and should not be used for production decisions.")


def data_center_page(data: Dict[str, pd.DataFrame]):
    page_header("Data Center", "Synthetic datasets are active by default. Manual CSV upload is local-only for this prototype.")
    st.info("CSV uploads are not sent to any NIK API or database. V0.1 replaces a synthetic dataset only when the uploaded schema matches the expected prototype schema.")

    names = ["sales", "customers", "leads", "sms", "nikpos"]
    uploaded = {}
    cols = st.columns(5)
    for col, name in zip(cols, names):
        with col:
            uploaded[name] = st.file_uploader(f"Upload {name.title()} CSV", type=["csv"], key=f"up_{name}")

    active = data.copy()
    for name, up in uploaded.items():
        if up is not None:
            try:
                df = load_uploaded_csv(up)
                valid, msg = validate_upload(name, df)
                if valid:
                    active[name] = df
                    st.success(f"{name.title()}: {msg}. Uploaded dataset active in Data Center.")
                else:
                    st.warning(f"{name.title()}: {msg}. Synthetic dataset remains active.")
            except Exception as exc:
                st.error(f"Could not read {name} CSV: {exc}")

    q = quality_table(active)
    q_show = q.copy()
    q_show["Quality Score"] = q_show["Quality Score"].map(lambda x: f"{x:.1%}")
    st.subheader("Data Quality")
    st.dataframe(q_show, use_container_width=True, hide_index=True)

    st.subheader("Dataset Explorer")
    selected = st.selectbox("Dataset", list(active.keys()))
    st.dataframe(active[selected].head(200), use_container_width=True, hide_index=True)
    st.caption(f"Showing first 200 rows of {len(active[selected]):,} records.")


def sales_analytics_page(scenario, data, kpis, daily, monthly):
    page_header("Sales Analytics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily Phone Sales", f"{scenario.daily_phone_sales}")
    c2.metric("Monthly Phone Units", f"{kpis['monthly_phone_units']:.0f}")
    c3.metric("Monthly Online Units", f"{kpis['monthly_online_units']:.0f}")
    c4.metric("Current Scenario Revenue", toman(kpis["monthly_revenue"]))

    if len(monthly) >= 2:
        mom = monthly.iloc[-1]["revenue"] / monthly.iloc[-2]["revenue"] - 1 if monthly.iloc[-2]["revenue"] else 0
        st.metric("Synthetic Month-over-Month Revenue Change", pct(mom))

    left, right = st.columns(2)
    with left:
        fig = px.line(daily.tail(90), x="date", y="units", title="Daily Units — Last 90 Days")
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        fig = px.line(monthly, x="month", y="revenue", markers=True, title="Monthly Revenue")
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    sales = data["sales"].copy()
    if not sales.empty:
        by_channel = sales.groupby("channel", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))
        by_plan = sales.groupby("plan", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))
        l, r = st.columns(2)
        with l:
            fig = px.bar(by_channel, x="channel", y="units", title="Historical Units by Channel")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with r:
            fig = px.bar(by_plan, x="plan", y="revenue", title="Historical Revenue by Plan")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Current Plan Economics")
    pp = plan_performance(scenario).copy()
    pp["share"] = pp["share"].map(lambda x: f"{x:.1%}")
    pp["unit_price"] = pp["unit_price"].map(toman)
    pp["revenue"] = pp["revenue"].map(toman)
    st.dataframe(pp, use_container_width=True, hide_index=True)


def customer_intelligence_page(customers_model, segment_profile, risk_stats):
    page_header("Customer Intelligence", "RFM-style segmentation plus an experimental churn-risk prototype.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Synthetic Customers", f"{len(customers_model):,}")
    c2.metric("High / Very High Risk", pct(risk_stats["high_or_very_high_share"]))
    c3.metric("Synthetic Holdout AUC", f"{risk_stats['synthetic_holdout_auc']:.3f}")

    left, right = st.columns(2)
    with left:
        seg = customers_model["segment"].value_counts().rename_axis("segment").reset_index(name="customers")
        fig = px.bar(seg, x="segment", y="customers", title="Prototype RFM Segments")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        risk = customers_model["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="customers")
        fig = px.pie(risk, names="risk_level", values="customers", hole=0.55, title="Prototype Churn Risk")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Segment Profiles")
    profile = segment_profile[["segment", "customers", "recency", "frequency", "monetary_value"]].copy()
    profile["recency"] = profile["recency"].round(1)
    profile["frequency"] = profile["frequency"].round(2)
    profile["monetary_value"] = profile["monetary_value"].map(toman)
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("Customers Requiring Attention")
    cols = ["customer_id", "city", "industry", "segment", "recency", "frequency", "monetary_value", "risk_score", "risk_level"]
    risk_table = customers_model.sort_values("risk_score", ascending=False)[cols].head(100).copy()
    risk_table["monetary_value"] = risk_table["monetary_value"].map(toman)
    st.dataframe(risk_table, use_container_width=True, hide_index=True)
    st.warning("Prototype Churn Risk Model: Logistic Regression trained on a synthetic risk label. The AUC above is illustrative only and is not production validation.")


def nikpos_page(scenario, data):
    page_header("NIKPOS Analytics")
    plans = plan_performance(scenario)
    devices = data["nikpos"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plan A Units", f"{plans.loc[plans.plan == 'Plan A', 'units'].iloc[0]:.0f}")
    c2.metric("Plan B Units", f"{plans.loc[plans.plan == 'Plan B', 'units'].iloc[0]:.0f}")
    c3.metric("Average Selling Price", toman(scenario.average_selling_price))
    c4.metric("Active Synthetic Devices", pct(devices["active_device"].mean()))

    l, r = st.columns(2)
    with l:
        fig = px.bar(plans, x="plan", y="revenue", title="Current Scenario Revenue by Plan")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with r:
        city = devices.groupby("city", as_index=False).agg(devices=("device_id", "count"), captures_30d=("captures_30d", "sum"))
        city = city.sort_values("captures_30d", ascending=False).head(10)
        fig = px.bar(city, x="city", y="captures_30d", title="Synthetic 30-Day Captures by City")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    monthly = data["sales"].copy()
    if not monthly.empty:
        monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").dt.to_timestamp()
        trend = monthly.groupby(["month", "plan"], as_index=False).agg(units=("units", "sum"))
        fig = px.line(trend, x="month", y="units", color="plan", markers=True, title="NIKPOS Monthly Unit Trend by Plan")
        st.plotly_chart(style_fig(fig), use_container_width=True)


def content_page(scenario):
    page_header("Content Analytics", "Attribution is intentionally labeled as estimated/synthetic.")
    m = content_metrics(scenario)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stories / Day", f"{m['stories_per_day']:.0f}")
    c2.metric("Stories / Month", f"{m['stories_per_month']:.0f}")
    c3.metric("Estimated Sales / Day", f"{m['estimated_sales_per_day']:.1f}")
    c4.metric("Estimated Sales / Month", f"{m['estimated_sales_per_month']:.0f}")

    st.markdown(
        f'<div class="insight-card"><div class="insight-type">Estimated / Synthetic Attribution</div>'
        f'<b>Content Sales Rate proxy</b><br>Estimated sales per story: {m["sales_per_story"]:.2f}. '
        'This is not a marketing conversion rate because story reach, exposures, source IDs, and attribution windows are not available.</div>',
        unsafe_allow_html=True,
    )

    days = pd.DataFrame({"day": np.arange(1, scenario.sales_days_per_month + 1)})
    days["stories"] = scenario.stories_per_day
    days["estimated_sales"] = scenario.content_sales_per_day
    fig = px.line(days, x="day", y=["stories", "estimated_sales"], title="Scenario Content Activity — 30-day-style Assumption")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Real content attribution requires source tracking, campaign IDs, channel tagging, and an explicit attribution model.")


def sms_page(data):
    page_header("SMS Analytics")
    sms = data["sms"].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Synthetic Messages Sent", f"{sms['sent'].sum():,}")
    c2.metric("Synthetic Delivery Rate", pct(sms["delivered"].sum() / sms["sent"].sum()))
    c3.metric("Synthetic Click Rate", pct(sms["clicks"].sum() / max(sms["delivered"].sum(), 1)))

    l, r = st.columns(2)
    with l:
        fig = px.line(sms.tail(90), x="date", y="delivery_rate", title="SMS Delivery Rate — Last 90 Days")
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with r:
        fig = px.line(sms.tail(90), x="date", y="sent", title="SMS Volume — Last 90 Days")
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)


def anomaly_page(sales_anomalies, sms_anomalies):
    page_header("Anomaly Detection", "Explainable Rolling Mean + Z-score method; threshold = |Z| >= 2.5.")
    c1, c2 = st.columns(2)
    c1.metric("Sales Revenue Anomalies", int(sales_anomalies["is_anomaly"].sum()))
    c2.metric("SMS Delivery Anomalies", int(sms_anomalies["is_anomaly"].sum()))

    l, r = st.columns(2)
    with l:
        fig = px.line(sales_anomalies, x="date", y="revenue", title="Daily Revenue with Flagged Points")
        flagged = sales_anomalies[sales_anomalies["is_anomaly"]]
        fig.add_scatter(x=flagged["date"], y=flagged["revenue"], mode="markers", name="Anomaly", marker=dict(size=10, color=GOLD))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        fig = px.line(sms_anomalies, x="date", y="delivery_rate", title="SMS Delivery Rate with Flagged Points")
        flagged = sms_anomalies[sms_anomalies["is_anomaly"]]
        fig.add_scatter(x=flagged["date"], y=flagged["delivery_rate"], mode="markers", name="Anomaly", marker=dict(size=10, color=GOLD))
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Flagged Observations")
    tab1, tab2 = st.tabs(["Sales", "SMS"])
    with tab1:
        st.dataframe(sales_anomalies[sales_anomalies["is_anomaly"]].sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(sms_anomalies[sms_anomalies["is_anomaly"]].sort_values("date", ascending=False), use_container_width=True, hide_index=True)


def predictions_page(forecast, forecast_stats, customers_model, risk_stats):
    page_header("Predictions", "Simple, explainable prototype models only.")
    c1, c2 = st.columns(2)
    c1.metric("Revenue Trend Model R2", f"{forecast_stats['r2']:.3f}")
    c2.metric("Synthetic Churn Holdout AUC", f"{risk_stats['synthetic_holdout_auc']:.3f}")

    fig = px.line(forecast, x="month", y="revenue", color="series", markers=True, title="Historical Revenue + Next 3 Months")
    st.plotly_chart(style_fig(fig, 470), use_container_width=True)

    st.subheader("Prototype Churn Risk Model")
    top = customers_model.sort_values("risk_score", ascending=False)[
        ["customer_id", "recency", "frequency", "monetary_value", "sms_usage", "nikpos_usage", "risk_score", "risk_level"]
    ].head(50).copy()
    top["monetary_value"] = top["monetary_value"].map(toman)
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.warning("Forecasts and ML outputs are experimental and should not be used for production decisions.")


def insights_page(insights):
    page_header("Automated Insights", "Automated Rule-based Insights generated from the current scenario and synthetic analysis outputs.")
    for item in insights:
        st.markdown(
            f'<div class="insight-card"><div class="insight-type">{item["type"]}</div><b>{item["title"]}</b><br>{item["text"]}</div>',
            unsafe_allow_html=True,
        )


def pipeline_page():
    page_header("Analysis Pipeline")
    stages = [
        "Data Import",
        "Data Validation",
        "Data Cleaning",
        "KPI Calculation",
        "Trend Analysis",
        "Anomaly Detection",
        "Customer Segmentation",
        "Risk Scoring",
        "Forecast",
        "Insight Generation",
        "Dashboard Update",
    ]
    for i in range(0, len(stages), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(stages):
                with col:
                    st.markdown(f'<div class="pipeline-step">{idx + 1}. {stages[idx]}</div>', unsafe_allow_html=True)
        if i + 3 < len(stages):
            st.markdown("<div style='text-align:center;color:#D6B86A;font-size:1.3rem;'>↓</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.code(
        "INPUT -> DATA -> ANALYSIS -> KPI -> CHART -> MODEL -> INSIGHT -> DASHBOARD UPDATE",
        language="text",
    )
    st.caption("RUN ANALYSIS in the sidebar demonstrates the pipeline sequence. Inputs also trigger Streamlit's live rerun behavior immediately.")


def settings_page(scenario, kpis):
    page_header("Settings / Scenario Controls", "All core values below are derived from the editable sidebar scenario.")
    rows = [
        ("Price Plan A", toman(scenario.price_plan_a)),
        ("Price Plan B", toman(scenario.price_plan_b)),
        ("Plan A Share", pct(scenario.plan_a_share)),
        ("Plan B Share", pct(scenario.plan_b_share)),
        ("Daily Phone Sales", f"{scenario.daily_phone_sales}"),
        ("Monthly Online Sales", f"{scenario.monthly_online_sales}"),
        ("Lead Backlog", f"{scenario.lead_backlog:,}"),
        ("Stories / Day", f"{scenario.stories_per_day}"),
        ("Content-attributed Sales / Day", f"{scenario.content_sales_per_day:.1f}"),
        ("Synthetic Customers", f"{scenario.synthetic_customer_count:,}"),
        ("Historical Months", f"{scenario.history_months}"),
        ("Sales Days / Month", f"{scenario.sales_days_per_month}"),
        ("Seed", f"{scenario.seed}"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Setting", "Current Value"]), use_container_width=True, hide_index=True)

    st.subheader("Live Dependency Check")
    st.code(
        f"Daily Phone Sales = {scenario.daily_phone_sales}\n"
        f"Monthly Phone Units = {kpis['monthly_phone_units']:.0f}\n"
        f"Monthly Total Units = {kpis['monthly_units']:.0f}\n"
        f"Average Selling Price = {scenario.average_selling_price:,.0f} Toman\n"
        f"Monthly Revenue = {kpis['monthly_revenue']:,.0f} Toman\n"
        f"Lead -> Purchase = {kpis['lead_purchase_conversion']:.2%}",
        language="text",
    )
    st.success("Demo test: change Daily Phone Sales from 10 to 15 in the sidebar. Revenue, units, funnel, historical trend, forecast, channel mix, and automated insights recalculate on rerun.")


def main():
    scenario, page = scenario_sidebar()
    data = build_synthetic_data(scenario)

    kpis = current_kpis(scenario, data["customers"])
    daily = sales_daily(data["sales"])
    monthly = sales_monthly(data["sales"])
    funnel = lead_funnel(scenario)
    capacity = backlog_capacity(scenario)

    customers_model, segment_profile, risk_stats = build_models(data["customers"], scenario.seed)
    forecast, forecast_stats = revenue_forecast(monthly, 3)
    sales_anomalies = detect_anomalies(daily, "revenue", "date", 14)
    sms_anomalies = detect_anomalies(data["sms"], "delivery_rate", "date", 14)
    insights = generate_insights(scenario, kpis, monthly, risk_stats, sales_anomalies, sms_anomalies)

    if page == "Executive Overview":
        executive_overview(scenario, data, customers_model, kpis, monthly, funnel, forecast, insights)
    elif page == "Data Center":
        data_center_page(data)
    elif page == "Sales Analytics":
        sales_analytics_page(scenario, data, kpis, daily, monthly)
    elif page == "Customer Intelligence":
        customer_intelligence_page(customers_model, segment_profile, risk_stats)
    elif page == "NIKPOS Analytics":
        nikpos_page(scenario, data)
    elif page == "Content Analytics":
        content_page(scenario)
    elif page == "SMS Analytics":
        sms_page(data)
    elif page == "Anomaly Detection":
        anomaly_page(sales_anomalies, sms_anomalies)
    elif page == "Predictions":
        predictions_page(forecast, forecast_stats, customers_model, risk_stats)
    elif page == "Automated Insights":
        insights_page(insights)
    elif page == "Analysis Pipeline":
        pipeline_page()
    elif page == "Settings / Scenario Controls":
        settings_page(scenario, kpis)

    st.markdown("---")
    st.caption("Prototype using synthetic/demo data. Not connected to NIK internal systems.")
    st.caption("Forecasts and ML outputs are experimental and should not be used for production decisions.")


if __name__ == "__main__":
    main()
