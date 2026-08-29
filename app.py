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
from business_data import (
    BACKLOG_SNAPSHOTS,
    BUSINESS_BASELINE,
    PRICE_HISTORY,
    REEL_SNAPSHOT,
    reel_snapshot_metrics,
)
from data_generator import Scenario, generate_all
from insight_engine import generate_insights
from ml_engine import churn_risk_model, detect_anomalies, revenue_forecast, segment_customers


st.set_page_config(
    page_title="نیک اس‌ام‌اس | تحلیل داده",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Visual system ----------
ACCENT = "#1DCBFF"
ACCENT_2 = "#006CFF"
TEXT = "#F6F9FF"
MUTED = "#94A3B8"
BG = "#060A10"
BORDER = "rgba(255,255,255,.10)"

PAGE_LABELS = {
    "Executive Overview": "نمای مدیریتی",
    "Data Center": "مرکز داده",
    "Sales Analytics": "تحلیل فروش",
    "Customer Intelligence": "هوشمندی مشتریان",
    "NIKPOS Analytics": "تحلیل نیک‌پوز",
    "Content Analytics": "تحلیل محتوا و اینستاگرام",
    "SMS Analytics": "تحلیل پیامک",
    "Anomaly Detection": "تشخیص ناهنجاری",
    "Predictions": "پیش‌بینی‌ها",
    "Automated Insights": "بینش‌های خودکار",
    "Analysis Pipeline": "خط لوله تحلیل",
    "Settings / Scenario Controls": "تنظیمات و سناریو",
}
PAGE_ICONS = {
    "Executive Overview": "◈",
    "Data Center": "▦",
    "Sales Analytics": "↗",
    "Customer Intelligence": "◎",
    "NIKPOS Analytics": "▣",
    "Content Analytics": "▶",
    "SMS Analytics": "✉",
    "Anomaly Detection": "⚡",
    "Predictions": "⌁",
    "Automated Insights": "✦",
    "Analysis Pipeline": "⇄",
    "Settings / Scenario Controls": "⚙",
}
SEGMENT_FA = {
    "High Value": "باارزش",
    "Growth": "در حال رشد",
    "Regular": "عادی",
    "At Risk": "در معرض ریزش",
    "Inactive": "غیرفعال",
}
RISK_FA = {"Low": "کم", "Medium": "متوسط", "High": "زیاد", "Very High": "بسیار زیاد"}
PLAN_FA = {"Plan A": "طرح A", "Plan B": "طرح B"}
CHANNEL_FA = {"Phone": "فروش تلفنی", "Online": "فروش آنلاین", "phone": "فروش تلفنی", "online": "فروش آنلاین"}
FUNNEL_FA = {
    "New Leads": "سرنخ‌های جدید",
    "Contacted": "تماس گرفته‌شده",
    "Qualified": "واجد شرایط",
    "Interested": "علاقه‌مند",
    "Purchased": "خرید کرده",
    "Active Customer": "مشتری فعال",
}
QUALITY_STATUS_FA = {"Healthy": "سالم", "Watch": "نیازمند پایش", "Needs Review": "نیازمند بررسی"}
DATASET_FA = {
    "Sales": "فروش",
    "Customers": "مشتریان",
    "Leads": "سرنخ‌ها",
    "Sms": "پیامک",
    "Nikpos": "نیک‌پوز",
    "Subscriptions": "اشتراک‌ها",
    "Content_Snapshot": "محتوا",
}
FORECAST_SERIES_FA = {"Historical": "تاریخی مصنوعی", "Forecast": "پیش‌بینی آزمایشی"}
SOURCE_LABELS = {
    "real": "Baseline واقعی",
    "derived": "محاسبه‌شده",
    "estimated": "تخمینی",
    "synthetic": "مصنوعی",
}
SOURCE_CLASSES = {
    "real": "src-real",
    "derived": "src-derived",
    "estimated": "src-estimated",
    "synthetic": "src-synthetic",
}
DATAFRAME_COL_FA = {
    "date": "تاریخ",
    "month": "ماه",
    "channel": "کانال",
    "plan": "طرح",
    "units": "تعداد",
    "revenue": "درآمد",
    "customer_id": "شناسه مشتری",
    "signup_date": "تاریخ عضویت",
    "city": "شهر",
    "industry": "صنعت",
    "last_activity": "آخرین فعالیت",
    "purchase_count": "تعداد خرید",
    "sms_usage": "مصرف پیامک",
    "nikpos_usage": "استفاده از نیک‌پوز",
    "recency": "روز از آخرین فعالیت",
    "frequency": "تکرار خرید",
    "monetary_value": "ارزش مالی",
    "lead_source": "منبع سرنخ",
    "customer_status": "وضعیت مشتری",
    "lead_id": "شناسه سرنخ",
    "created_date": "تاریخ ایجاد",
    "stage": "مرحله",
    "sent": "ارسال‌شده",
    "delivered": "تحویل‌شده",
    "delivery_rate": "نرخ تحویل",
    "clicks": "کلیک",
    "device_id": "شناسه دستگاه",
    "activation_date": "تاریخ فعال‌سازی",
    "active_device": "دستگاه فعال",
    "captures_30d": "ثبت شماره ۳۰ روزه",
    "z_score": "امتیاز Z",
    "is_anomaly": "ناهنجاری",
    "rolling_mean": "میانگین متحرک",
    "series": "نوع داده",
    "risk_score": "امتیاز ریسک",
    "risk_level": "سطح ریسک",
}

SVG_ICONS = {
    "revenue": """<svg viewBox="0 0 24 24"><path d="M4 17l5-5 4 4 7-8"/><path d="M15 8h5v5"/></svg>""",
    "sales": """<svg viewBox="0 0 24 24"><path d="M4 19V9"/><path d="M10 19V5"/><path d="M16 19v-7"/><path d="M22 19H2"/></svg>""",
    "leads": """<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3"/><path d="M3 19c.7-4 2.8-6 6-6s5.3 2 6 6"/><path d="M17 7h4M19 5v4"/></svg>""",
    "phone": """<svg viewBox="0 0 24 24"><path d="M7 3h3l2 5-2 2c1.3 2.7 3.3 4.7 6 6l2-2 5 2v3c0 1.1-.9 2-2 2C11 21 3 13 3 3c0-1.1.9-2 2-2h2"/></svg>""",
    "price": """<svg viewBox="0 0 24 24"><path d="M3 7l8-4 10 5-8 5-10-6z"/><path d="M3 12l10 6 8-5"/><path d="M3 17l10 6 8-5"/></svg>""",
    "online": """<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3c3 3 3 15 0 18"/><path d="M12 3c-3 3-3 15 0 18"/></svg>""",
    "followers": """<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3"/><circle cx="18" cy="9" r="2"/><path d="M3 20c.7-4 2.8-6 6-6s5.3 2 6 6"/><path d="M15 15c3 0 5 1.7 6 5"/></svg>""",
    "content": """<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M10 9l6 3-6 3V9z"/></svg>""",
}


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --nik-blue: {ACCENT};
        --nik-blue-2: {ACCENT_2};
        --nik-bg: {BG};
        --nik-text: {TEXT};
        --nik-muted: {MUTED};
    }}

    html, body, [class*="css"] {{
        font-family: "Vazirmatn", "Segoe UI", Tahoma, sans-serif;
    }}
    .stApp {{
        direction: rtl;
        color: var(--nik-text);
        background:
            radial-gradient(circle at 80% -10%, rgba(29,203,255,.16), transparent 30%),
            radial-gradient(circle at 15% 20%, rgba(0,108,255,.12), transparent 27%),
            radial-gradient(circle at 60% 80%, rgba(98,53,255,.08), transparent 25%),
            #060A10;
    }}
    .block-container {{
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(10,17,26,.94), rgba(5,10,17,.96));
        border-left: 1px solid rgba(255,255,255,.08);
        border-right: 0;
        backdrop-filter: blur(24px);
    }}
    [data-testid="stSidebar"] * {{
        text-align: right;
    }}
    [data-testid="stSidebarNav"] {{
        display: none;
    }}
    .stMarkdown, .stCaption, .stAlert, .stHeader, .stSubheader {{
        direction: rtl;
        text-align: right;
    }}

    /* Native Streamlit surfaces */
    [data-testid="stMetric"] {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, rgba(255,255,255,.075), rgba(255,255,255,.025));
        border: 1px solid rgba(255,255,255,.11);
        padding: 17px 18px;
        border-radius: 20px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 18px 55px rgba(0,0,0,.18);
        backdrop-filter: blur(22px) saturate(130%);
    }}
    [data-testid="stMetric"]::before {{
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        left: -35px;
        top: -45px;
        background: radial-gradient(circle, rgba(29,203,255,.18), transparent 70%);
        pointer-events: none;
    }}
    [data-testid="stMetricLabel"] {{
        color: #AAB8C8;
    }}
    [data-testid="stMetricValue"] {{
        color: #FFFFFF;
    }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        overflow: hidden;
    }}
    [data-testid="stFileUploader"] {{
        border-radius: 18px;
    }}
    div[data-testid="stExpander"] {{
        background: rgba(255,255,255,.035);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px;
    }}
    div.stButton > button {{
        border-radius: 14px;
        border: 1px solid rgba(29,203,255,.23);
        background: linear-gradient(145deg, rgba(29,203,255,.13), rgba(0,108,255,.08));
        color: #EAF9FF;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
        transition: all .18s ease;
    }}
    div.stButton > button:hover {{
        border-color: rgba(29,203,255,.58);
        background: linear-gradient(145deg, rgba(29,203,255,.22), rgba(0,108,255,.14));
        transform: translateY(-1px);
    }}

    /* Hero */
    .hero-glass {{
        position: relative;
        overflow: hidden;
        padding: 25px 28px;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(255,255,255,.085), rgba(255,255,255,.025)),
            linear-gradient(90deg, rgba(29,203,255,.035), rgba(0,108,255,.02));
        box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 28px 80px rgba(0,0,0,.22);
        backdrop-filter: blur(28px) saturate(140%);
    }}
    .hero-glass::before {{
        content: "";
        position: absolute;
        width: 360px;
        height: 180px;
        left: -90px;
        top: -100px;
        background: radial-gradient(circle, rgba(29,203,255,.19), transparent 68%);
        filter: blur(8px);
        pointer-events: none;
    }}
    .hero-glass::after {{
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: -100px;
        bottom: -170px;
        background: radial-gradient(circle, rgba(0,108,255,.14), transparent 67%);
        pointer-events: none;
    }}
    .eyebrow {{
        display: inline-flex;
        gap: 8px;
        align-items: center;
        color: #8DDFFF;
        font-size: .79rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .hero-title {{
        font-size: clamp(1.9rem, 3.4vw, 3.35rem);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -.04em;
        color: #F7FBFF;
        margin: 0 0 8px 0;
    }}
    .hero-title .accent {{
        background: linear-gradient(90deg, #BDEFFF, #1DCBFF 55%, #75A9FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-subtitle {{
        max-width: 820px;
        color: #9EB0C2;
        font-size: .98rem;
        line-height: 1.9;
    }}
    .hero-meta {{
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin-top:16px;
    }}
    .meta-pill {{
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding:6px 10px;
        border-radius:999px;
        background:rgba(255,255,255,.045);
        border:1px solid rgba(255,255,255,.08);
        color:#B7C6D5;
        font-size:.77rem;
    }}
    .meta-dot {{
        width:7px;
        height:7px;
        border-radius:50%;
        background:#1DCBFF;
        box-shadow:0 0 12px rgba(29,203,255,.7);
    }}

    /* KPI cards */
    .glass-kpi {{
        position: relative;
        min-height: 165px;
        overflow:hidden;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,.11);
        background: linear-gradient(150deg, rgba(255,255,255,.082), rgba(255,255,255,.026));
        box-shadow: inset 0 1px 0 rgba(255,255,255,.09), 0 22px 70px rgba(0,0,0,.20);
        backdrop-filter: blur(24px) saturate(135%);
        padding: 18px 18px 16px;
    }}
    .glass-kpi::after {{
        content:"";
        position:absolute;
        width:115px;height:115px;
        left:-45px;bottom:-70px;
        border-radius:50%;
        background:radial-gradient(circle, rgba(29,203,255,.16), transparent 70%);
        pointer-events:none;
    }}
    .kpi-top {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:10px;
    }}
    .kpi-icon {{
        width:38px;height:38px;
        display:flex;align-items:center;justify-content:center;
        border-radius:13px;
        background:linear-gradient(145deg,rgba(29,203,255,.15),rgba(0,108,255,.07));
        border:1px solid rgba(29,203,255,.22);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.08);
    }}
    .kpi-icon svg {{
        width:20px;height:20px;
        stroke:#86E3FF;
        fill:none;
        stroke-width:1.7;
        stroke-linecap:round;
        stroke-linejoin:round;
    }}
    .source-tag {{
        font-size:.67rem;
        font-weight:700;
        border-radius:999px;
        padding:4px 8px;
        white-space:nowrap;
    }}
    .src-real {{ color:#8FF6C1;background:rgba(33,210,133,.09);border:1px solid rgba(33,210,133,.18); }}
    .src-derived {{ color:#89E5FF;background:rgba(29,203,255,.09);border:1px solid rgba(29,203,255,.18); }}
    .src-estimated {{ color:#FFD889;background:rgba(255,188,76,.09);border:1px solid rgba(255,188,76,.18); }}
    .src-synthetic {{ color:#C4AEFF;background:rgba(141,102,255,.09);border:1px solid rgba(141,102,255,.18); }}
    .kpi-label {{ color:#9AACBE;font-size:.80rem;margin-top:16px; }}
    .kpi-value {{ color:#F9FCFF;font-size:1.68rem;font-weight:800;letter-spacing:-.035em;margin-top:4px; }}
    .kpi-note {{ color:#718298;font-size:.69rem;margin-top:4px;line-height:1.6; }}

    /* Reusable cards */
    .glass-panel {{
        border:1px solid rgba(255,255,255,.09);
        border-radius:22px;
        padding:18px 20px;
        background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.018));
        box-shadow:inset 0 1px 0 rgba(255,255,255,.07);
        backdrop-filter:blur(20px);
        margin-bottom:12px;
    }}
    .insight-card {{
        position:relative;
        overflow:hidden;
        background:linear-gradient(145deg,rgba(255,255,255,.057),rgba(255,255,255,.018));
        border:1px solid rgba(255,255,255,.09);
        border-radius:18px;
        padding:15px 17px;
        margin-bottom:10px;
        box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
    }}
    .insight-card::before {{
        content:"";
        position:absolute;
        right:0;top:0;bottom:0;width:3px;
        background:linear-gradient(#1DCBFF,#006CFF);
    }}
    .insight-type {{
        color:#73DCFF;
        font-size:.68rem;
        font-weight:800;
        letter-spacing:.06em;
        margin-bottom:4px;
    }}
    .insight-title {{
        font-size:.96rem;
        font-weight:750;
        color:#F5FAFF;
        margin-bottom:4px;
    }}
    .insight-text {{
        color:#A4B3C3;
        font-size:.82rem;
        line-height:1.85;
    }}
    .section-kicker {{
        color:#65D8FF;
        font-size:.73rem;
        font-weight:800;
        letter-spacing:.05em;
        margin-bottom:3px;
    }}
    .section-title {{
        color:#F5FAFF;
        font-size:1.32rem;
        font-weight:750;
        margin-bottom:3px;
    }}
    .section-subtitle {{
        color:#8293A7;
        font-size:.82rem;
        margin-bottom:13px;
    }}
    .pipeline-step {{
        min-height:72px;
        display:flex;
        align-items:center;
        justify-content:center;
        text-align:center;
        border-radius:18px;
        border:1px solid rgba(255,255,255,.09);
        background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.018));
        color:#DDE8F2;
        font-weight:650;
        box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
    }}
    .connector-status {{
        padding:9px 11px;
        border-radius:14px;
        background:rgba(29,203,255,.055);
        border:1px solid rgba(29,203,255,.13);
        color:#9FCFE0;
        font-size:.76rem;
        margin-top:8px;
    }}
    hr {{ border-color: rgba(255,255,255,.075); }}
    </style>
    """,
    unsafe_allow_html=True,
)


def fa_digits(value) -> str:
    return str(value).translate(str.maketrans("0123456789,.%", "۰۱۲۳۴۵۶۷۸۹٬٫٪"))


def fa_num(value: float, decimals: int = 0) -> str:
    return fa_digits(f"{value:,.{decimals}f}")


def toman(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{fa_num(value / 1_000_000_000, 2)} میلیارد"
    if abs(value) >= 1_000_000:
        return f"{fa_num(value / 1_000_000, 1)} میلیون"
    return fa_num(value)


def pct(value: float) -> str:
    return fa_digits(f"{value:.1%}")


def style_fig(fig, height: int = 380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C4D0DC", family="Vazirmatn, Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        title_font=dict(size=15, color="#EFF7FF"),
        hoverlabel=dict(bgcolor="#0C1520", font_color="#F5FAFF"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.055)", zerolinecolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.055)", zerolinecolor="rgba(255,255,255,.08)")
    return fig


def kpi_card(label: str, value: str, source: str, icon: str, note: str = ""):
    src_class = SOURCE_CLASSES[source]
    src_label = SOURCE_LABELS[source]
    svg = SVG_ICONS.get(icon, SVG_ICONS["sales"])
    st.markdown(
        f"""
        <div class="glass-kpi">
            <div class="kpi-top">
                <div class="kpi-icon">{svg}</div>
                <div class="source-tag {src_class}">{src_label}</div>
            </div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(kicker: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def source_legend():
    st.markdown(
        """
        <div class="glass-panel">
            <div style="font-weight:700;margin-bottom:8px">راهنمای اعتبار داده</div>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
                <span class="source-tag src-real">Baseline واقعی</span>
                <span class="source-tag src-derived">محاسبه‌شده از Baseline</span>
                <span class="source-tag src-estimated">تخمینی / Attribution</span>
                <span class="source-tag src-synthetic">مصنوعی / مدل آزمایشی</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(page_title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="hero-glass">
            <div class="eyebrow"><span>◈</span> NIK INTELLIGENCE / V0.2</div>
            <div class="hero-title">نیک اس‌ام‌اس <span class="accent">| تحلیل داده</span></div>
            <div class="hero-subtitle">{subtitle or "سامانه هوشمندی داده و تصمیم‌سازی مدیریتی؛ ترکیب Baseline واقعی، محاسبات قابل توضیح و مدل‌های آزمایشی."}</div>
            <div class="hero-meta">
                <span class="meta-pill"><span class="meta-dot"></span> {page_title}</span>
                <span class="meta-pill">Snapshot: ۲۹ اوت ۲۰۲۶</span>
                <span class="meta-pill">Prototype / Proof of Concept</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_col, n8n_col, status_col = st.columns([1, 1, 4])
    with api_col:
        if st.button("اتصال API", use_container_width=True, key=f"api_{page_title}"):
            st.toast("دکمه API برای نسخه اتصال واقعی آماده شده؛ در V0.2 هنوز هیچ اتصال خارجی ایجاد نمی‌شود.")
    with n8n_col:
        if st.button("اتصال n8n", use_container_width=True, key=f"n8n_{page_title}"):
            st.toast("دکمه n8n فقط Placeholder است؛ Workflow اتصال در نسخه بعدی ساخته می‌شود.")
    with status_col:
        st.markdown(
            '<div class="connector-status">وضعیت اتصال: <b>Demo Mode</b> — API / Database / n8n هنوز متصل نیست.</div>',
            unsafe_allow_html=True,
        )


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
            "customer_id",
            "signup_date",
            "city",
            "industry",
            "plan",
            "last_activity",
            "purchase_count",
            "revenue",
            "sms_usage",
            "nikpos_usage",
            "recency",
            "frequency",
            "monetary_value",
            "lead_source",
            "customer_status",
        },
        "leads": {"lead_id", "created_date", "lead_source", "stage"},
        "sms": {"date", "sent", "delivered", "delivery_rate"},
        "nikpos": {"device_id", "customer_id", "activation_date", "plan"},
        "content_snapshot": {"reel", "views", "comments", "shares"},
    }
    missing = schemas[name] - set(df.columns)
    if missing:
        return False, "ستون‌های مورد انتظار وجود ندارند: " + ", ".join(sorted(missing))
    return True, "ساختار فایل تأیید شد"


def scenario_sidebar() -> Tuple[Scenario, str]:
    st.sidebar.markdown(
        """
        <div style="padding:12px 4px 4px">
            <div style="font-size:.72rem;color:#67D8FF;font-weight:800;letter-spacing:.08em">NIK INTELLIGENCE</div>
            <div style="font-size:1.25rem;color:#F5FAFF;font-weight:800;margin-top:3px">تحلیل داده نیک اس‌ام‌اس</div>
            <div style="font-size:.74rem;color:#7E92A7;margin-top:4px">Executive Intelligence Prototype</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    pages = list(PAGE_LABELS.keys())
    page = st.sidebar.radio(
        "منوی اصلی",
        pages,
        format_func=lambda x: f"{PAGE_ICONS[x]}   {PAGE_LABELS[x]}",
    )

    with st.sidebar.expander("کنترل سناریوی مدیریتی", expanded=True):
        price_a = st.number_input("قیمت طرح A (تومان)", 1_000_000, 200_000_000, int(BUSINESS_BASELINE["plan_a_price"]), 1_000_000)
        price_b = st.number_input("قیمت طرح B (تومان)", 1_000_000, 300_000_000, int(BUSINESS_BASELINE["plan_b_price"]), 1_000_000)
        share_a_pct = st.slider("سهم طرح A", 0, 100, int(BUSINESS_BASELINE["plan_a_share"] * 100), 5)
        phone = st.number_input("فروش تلفنی روزانه", 0, 500, int(BUSINESS_BASELINE["daily_phone_sales"]), 1)
        online = st.number_input("فروش آنلاین ماهانه", 0, 5_000, int(BUSINESS_BASELINE["monthly_online_sales"]), 5)
        backlog = st.number_input("صف فعلی لید", 0, 500_000, int(BUSINESS_BASELINE["lead_backlog"]), 100)
        sales_days = st.slider("فرض روز فروش در ماه", 20, 31, int(BUSINESS_BASELINE["sales_days_per_month"]), 1)

    with st.sidebar.expander("کنترل محتوا"):
        stories = st.number_input("استوری روزانه", 0, 100, int(BUSINESS_BASELINE["stories_per_day"]), 1)
        reels = st.number_input("ریلز روزانه", 0, 20, int(BUSINESS_BASELINE["reels_per_day"]), 1)
        content_sales = st.number_input(
            "فروش منتسب به محتوا / روز",
            0.0,
            100.0,
            float(BUSINESS_BASELINE["estimated_content_sales_per_day"]),
            0.5,
            help="این عدد Estimated Attribution است و به فروش کل اضافه نمی‌شود.",
        )
        followers = st.number_input("فالوئر اینستاگرام", 0, 10_000_000, int(BUSINESS_BASELINE["instagram_followers"]), 1_000)
        team_size = st.number_input("اندازه تیم محتوا", 1, 100, int(BUSINESS_BASELINE["content_team_size"]), 1)

    with st.sidebar.expander("تنظیمات مدل آزمایشی"):
        customers = st.number_input("مشتری مصنوعی", 500, 50_000, 5_000, 500)
        months = st.slider("ماه‌های داده تاریخی مصنوعی", 3, 36, 12, 1)
        seed = st.number_input("Seed", 1, 999_999, 42, 1)

    scenario = Scenario(
        price_plan_a=float(price_a),
        price_plan_b=float(price_b),
        plan_a_share=float(share_a_pct) / 100,
        daily_phone_sales=int(phone),
        monthly_online_sales=int(online),
        lead_backlog=int(backlog),
        stories_per_day=int(stories),
        reels_per_day=int(reels),
        content_sales_per_day=float(content_sales),
        instagram_followers=int(followers),
        content_team_size=int(team_size),
        synthetic_customer_count=int(customers),
        history_months=int(months),
        sales_days_per_month=int(sales_days),
        seed=int(seed),
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("اجرای تحلیل کامل", use_container_width=True, type="primary"):
        stages = [
            "بارگذاری داده",
            "اعتبارسنجی داده",
            "پاک‌سازی",
            "محاسبه KPI",
            "تحلیل روند",
            "تشخیص ناهنجاری",
            "اجرای مدل‌ها",
            "تولید Insight",
            "تحلیل کامل شد",
        ]
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()
        for i, stage in enumerate(stages, start=1):
            status.caption(stage)
            progress.progress(int(i / len(stages) * 100))
            time.sleep(0.035)
        status.success("تحلیل کامل شد")

    st.sidebar.caption("V0.2 — Baseline واقعی تجمیعی + داده مصنوعی. بدون اتصال به سیستم داخلی NIK.")
    return scenario, page


def executive_overview(scenario, data, kpis, monthly, funnel, forecast, insights):
    page_header(
        "نمای مدیریتی",
        "Snapshot مدیریتی برای درک وضعیت فروش، صف لید، محتوا و ظرفیت فعلی در چند ثانیه. منبع اعتماد هر عدد روی همان کارت مشخص شده است.",
    )

    section_heading("EXECUTIVE SNAPSHOT", "وضعیت فعلی کسب‌وکار", "اعداد واقعی از Baseline از محاسبات Derived جدا شده‌اند.")

    row1 = st.columns(4)
    with row1[0]:
        kpi_card("درآمد ماهانه", f"{toman(kpis['monthly_revenue'])} تومان", "derived", "revenue", f"فرض {fa_num(scenario.sales_days_per_month)} روز فروش")
    with row1[1]:
        kpi_card("فروش ماهانه", f"{fa_num(kpis['monthly_units'])} دستگاه", "derived", "sales", "تلفنی + آنلاین؛ بر اساس سناریوی فعلی")
    with row1[2]:
        kpi_card("صف فعلی لید", fa_num(scenario.lead_backlog), "real", "leads", "Stock فعلی؛ Conversion واقعی نیست")
    with row1[3]:
        kpi_card("فروش تلفنی روزانه", fa_num(scenario.daily_phone_sales), "real", "phone", "Baseline عملیاتی فعلی")

    st.write("")
    row2 = st.columns(4)
    with row2[0]:
        kpi_card("میانگین قیمت فروش", f"{toman(scenario.average_selling_price)} تومان", "derived", "price", f"Mix: {fa_num(scenario.plan_a_share*100)}٪ / {fa_num(scenario.plan_b_share*100)}٪")
    with row2[1]:
        kpi_card("فروش آنلاین ماهانه", fa_num(scenario.monthly_online_sales), "real", "online", "Baseline فعلی")
    with row2[2]:
        kpi_card("فالوئر Instagram", fa_num(scenario.instagram_followers), "real", "followers", "Snapshot حدود ۱۷ اوت ۲۰۲۶")
    with row2[3]:
        kpi_card("خروجی محتوا / روز", fa_num(scenario.total_content_per_day), "real", "content", f"{fa_num(scenario.stories_per_day)} استوری + {fa_num(scenario.reels_per_day)} ریلز")

    st.write("")
    source_legend()

    left, right = st.columns([1.55, 1])
    with left:
        section_heading("SALES MODEL", "روند درآمد دمو", "تاریخچه این نمودار مصنوعی است و با Scenario Controls بازتولید می‌شود.")
        fig = px.area(
            monthly,
            x="month",
            y="revenue",
            markers=True,
            labels={"month": "ماه", "revenue": "درآمد"},
        )
        fig.update_traces(line_color=ACCENT, fillcolor="rgba(29,203,255,.09)")
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)
    with right:
        section_heading("CHANNEL MIX", "ترکیب فروش فعلی", "سهم کانال‌ها از تعداد فروش محاسبه‌شده.")
        channel = pd.DataFrame(
            {
                "کانال": ["فروش تلفنی", "فروش آنلاین"],
                "تعداد": [kpis["monthly_phone_units"], kpis["monthly_online_units"]],
            }
        )
        fig = px.pie(
            channel,
            names="کانال",
            values="تعداد",
            hole=0.68,
            color_discrete_sequence=[ACCENT, "#5877FF"],
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)

    f1, f2 = st.columns([1, 1.4])
    with f1:
        section_heading("DEMO FUNNEL", "قیف مصنوعی لید", "نرخ‌های مراحل هنوز داده واقعی Call Center نیستند.")
        fig = go.Figure(
            go.Funnel(
                y=funnel["stage"].map(FUNNEL_FA),
                x=funnel["count"],
                textinfo="value+percent initial",
                marker={"color": ["#1DCBFF", "#27B5F4", "#339FEA", "#4188DE", "#5470CF", "#6B5ABF"]},
            )
        )
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)
    with f2:
        section_heading("EXPERIMENTAL FORECAST", "پیش‌بینی ۳ ماه آینده", "Linear Regression روی داده تاریخی مصنوعی؛ صرفاً برای نمایش معماری.")
        forecast_show = forecast.copy()
        forecast_show["series"] = forecast_show["series"].map(FORECAST_SERIES_FA).fillna(forecast_show["series"])
        fig = px.line(
            forecast_show,
            x="month",
            y="revenue",
            color="series",
            markers=True,
            labels={"month": "ماه", "revenue": "درآمد", "series": "نوع داده"},
            color_discrete_sequence=[ACCENT, "#8B73FF"],
        )
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)

    section_heading("MANAGEMENT SIGNALS", "سیگنال‌های مدیریتی", "Insightهای قانون‌محور؛ با تغییر Scenario دوباره محاسبه می‌شوند.")
    cols = st.columns(2)
    for i, item in enumerate(insights[:6]):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-type">{item["type"]}</div>
                    <div class="insight-title">{item["title"]}</div>
                    <div class="insight-text">{item["text"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def data_center_page(data: Dict[str, pd.DataFrame]):
    page_header(
        "مرکز داده",
        "لایه ورود داده برای CSV امروز و API / Database / n8n در نسخه بعدی. هیچ داده‌ای در V0.2 به سیستم داخلی NIK متصل نیست.",
    )
    source_legend()

    section_heading("CONNECTORS", "آمادگی اتصال", "این کارت‌ها وضعیت معماری آینده را نشان می‌دهند؛ هنوز اتصال واقعی فعال نیست.")
    c1, c2, c3, c4 = st.columns(4)
    for col, title, status, detail in [
        (c1, "API / CRM", "آماده طراحی", "Read-only contract"),
        (c2, "n8n", "Placeholder", "Automation workflow"),
        (c3, "Database", "غیرفعال", "Read-only future"),
        (c4, "CSV Import", "فعال", "Prototype input"),
    ]:
        with col:
            st.markdown(
                f'<div class="glass-panel"><div class="section-kicker">{status}</div><div style="font-size:1.05rem;font-weight:750">{title}</div><div class="kpi-note">{detail}</div></div>',
                unsafe_allow_html=True,
            )

    section_heading("CSV IMPORT", "ورود فایل", "اگر ساختار فایل درست باشد، برای Preview همان Dataset استفاده می‌شود.")
    names = ["sales", "customers", "leads", "sms", "nikpos", "content_snapshot"]
    labels = {
        "sales": "فروش",
        "customers": "مشتریان",
        "leads": "سرنخ‌ها",
        "sms": "پیامک",
        "nikpos": "نیک‌پوز",
        "content_snapshot": "محتوا",
    }
    uploaded = {}
    rows = [names[:3], names[3:]]
    for group in rows:
        cols = st.columns(3)
        for col, name in zip(cols, group):
            with col:
                uploaded[name] = st.file_uploader(f"CSV {labels[name]}", type=["csv"], key=f"up_{name}")

    active = data.copy()
    for name, up in uploaded.items():
        if up is not None:
            try:
                df = load_uploaded_csv(up)
                valid, msg = validate_upload(name, df)
                if valid:
                    active[name] = df
                    st.success(f"{labels[name]}: {msg}")
                else:
                    st.warning(f"{labels[name]}: {msg}")
            except Exception as exc:
                st.error(f"خواندن فایل {labels[name]} ممکن نبود: {exc}")

    section_heading("DATA QUALITY", "کیفیت داده", "امتیازها روی Datasetهای فعلی محاسبه می‌شوند.")
    q = quality_table(active)
    q_show = q.copy()
    q_show["Dataset"] = q_show["Dataset"].map(DATASET_FA).fillna(q_show["Dataset"])
    q_show["Status"] = q_show["Status"].map(QUALITY_STATUS_FA).fillna(q_show["Status"])
    q_show["Quality Score"] = q_show["Quality Score"].map(pct)
    q_show = q_show.rename(
        columns={
            "Dataset": "مجموعه داده",
            "Record Count": "تعداد رکورد",
            "Missing Values": "مقادیر خالی",
            "Duplicate Records": "رکورد تکراری",
            "Invalid Values": "مقادیر نامعتبر",
            "Quality Score": "امتیاز کیفیت",
            "Status": "وضعیت",
        }
    )
    st.dataframe(q_show, use_container_width=True, hide_index=True)

    section_heading("PREVIEW", "مرور داده", "برای کنترل سریع ساختار Dataset.")
    selected = st.selectbox(
        "مجموعه داده",
        list(active.keys()),
        format_func=lambda x: {
            "sales": "فروش",
            "customers": "مشتریان",
            "leads": "سرنخ‌ها",
            "sms": "پیامک",
            "nikpos": "نیک‌پوز",
            "subscriptions": "اشتراک‌ها",
            "content_snapshot": "Snapshot محتوا",
        }.get(x, x),
    )
    preview = active[selected].head(200).copy()
    if "channel" in preview.columns:
        preview["channel"] = preview["channel"].map(CHANNEL_FA).fillna(preview["channel"])
    if "plan" in preview.columns:
        preview["plan"] = preview["plan"].map(PLAN_FA).fillna(preview["plan"])
    if "stage" in preview.columns:
        preview["stage"] = preview["stage"].map(FUNNEL_FA).fillna(preview["stage"])
    preview = preview.rename(columns={c: DATAFRAME_COL_FA.get(c, c) for c in preview.columns})
    st.dataframe(preview, use_container_width=True, hide_index=True)


def sales_analytics_page(scenario, data, kpis, daily, monthly):
    page_header(
        "تحلیل فروش",
        "تفکیک Baseline واقعی از محاسبات ۳۰روزه؛ هیچ Revenue به‌عنوان فروش حسابداری واقعی معرفی نمی‌شود.",
    )

    row = st.columns(4)
    with row[0]:
        kpi_card("فروش تلفنی / روز", fa_num(scenario.daily_phone_sales), "real", "phone", "Baseline")
    with row[1]:
        kpi_card("فروش تلفنی / ماه", fa_num(kpis["monthly_phone_units"]), "derived", "sales", f"{fa_num(scenario.sales_days_per_month)} روز × فروش روزانه")
    with row[2]:
        kpi_card("فروش آنلاین / ماه", fa_num(scenario.monthly_online_sales), "real", "online", "Baseline")
    with row[3]:
        kpi_card("Revenue مدل", f"{toman(kpis['monthly_revenue'])} تومان", "derived", "revenue", "Units × ASP")

    st.write("")
    st.info(
        f"فرض فعلی مدل: {fa_num(scenario.sales_days_per_month)} روز فروش در ماه. "
        "Revenue و Units ماهانه Derived هستند و باید با Sales Raw واقعی جایگزین شوند."
    )

    left, right = st.columns(2)
    with left:
        section_heading("SYNTHETIC TREND", "فروش روزانه دمو", "۹۰ روز اخیر؛ تولیدشده بر اساس Scenario.")
        fig = px.line(daily.tail(90), x="date", y="units", labels={"date": "تاریخ", "units": "تعداد"})
        fig.update_traces(line_color=ACCENT)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        section_heading("SYNTHETIC TREND", "درآمد ماهانه دمو", "برای نمایش رفتار Dynamic و Forecast.")
        fig = px.area(monthly, x="month", y="revenue", labels={"month": "ماه", "revenue": "درآمد"})
        fig.update_traces(line_color=ACCENT, fillcolor="rgba(29,203,255,.08)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    sales = data["sales"].copy()
    by_channel = sales.groupby("channel", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))
    by_channel["channel"] = by_channel["channel"].map(CHANNEL_FA).fillna(by_channel["channel"])
    by_plan = sales.groupby("plan", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))
    by_plan["plan"] = by_plan["plan"].map(PLAN_FA).fillna(by_plan["plan"])
    l, r = st.columns(2)
    with l:
        fig = px.bar(by_channel, x="channel", y="units", labels={"channel": "کانال", "units": "تعداد"}, color_discrete_sequence=[ACCENT])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with r:
        fig = px.bar(by_plan, x="plan", y="revenue", labels={"plan": "طرح", "revenue": "درآمد"}, color_discrete_sequence=["#6E8EFF"])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    section_heading("PLAN ECONOMICS", "اقتصاد پلن فعلی", "Mix و قیمت‌ها از Sidebar قابل تغییرند.")
    pp = plan_performance(scenario).copy()
    pp["plan"] = pp["plan"].map(PLAN_FA).fillna(pp["plan"])
    pp["share"] = pp["share"].map(pct)
    pp["unit_price"] = pp["unit_price"].map(lambda x: f"{toman(x)} تومان")
    pp["revenue"] = pp["revenue"].map(lambda x: f"{toman(x)} تومان")
    pp["units"] = pp["units"].map(fa_num)
    pp = pp.rename(columns={"plan": "طرح", "units": "تعداد", "share": "سهم", "unit_price": "قیمت واحد", "revenue": "درآمد"})
    st.dataframe(pp, use_container_width=True, hide_index=True)


def customer_intelligence_page(customers_model, segment_profile, risk_stats):
    page_header(
        "هوشمندی مشتریان",
        "این بخش کاملاً Prototype است: Segmentation و Churn روی دیتای مصنوعی اجرا می‌شوند تا معماری آینده را نمایش دهند.",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("مشتریان مصنوعی", fa_num(len(customers_model)))
    c2.metric("ریسک زیاد / بسیار زیاد", pct(risk_stats["high_or_very_high_share"]))
    c3.metric("AUC آزمایشی", fa_digits(f"{risk_stats['synthetic_holdout_auc']:.3f}"))

    left, right = st.columns(2)
    with left:
        seg = customers_model["segment"].value_counts().rename_axis("segment").reset_index(name="customers")
        seg["segment"] = seg["segment"].map(SEGMENT_FA).fillna(seg["segment"])
        fig = px.bar(
            seg,
            x="segment",
            y="customers",
            labels={"segment": "بخش مشتری", "customers": "تعداد"},
            color_discrete_sequence=[ACCENT],
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        risk = customers_model["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="customers")
        risk["risk_level"] = risk["risk_level"].map(RISK_FA).fillna(risk["risk_level"])
        fig = px.pie(risk, names="risk_level", values="customers", hole=0.62, color_discrete_sequence=[ACCENT, "#5E85FF", "#826BFF", "#B15FFF"])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    section_heading("RFM PROFILE", "پروفایل سگمنت‌ها")
    profile = segment_profile[["segment", "customers", "recency", "frequency", "monetary_value"]].copy()
    profile["segment"] = profile["segment"].map(SEGMENT_FA).fillna(profile["segment"])
    profile["recency"] = profile["recency"].round(1)
    profile["frequency"] = profile["frequency"].round(2)
    profile["monetary_value"] = profile["monetary_value"].map(lambda x: f"{toman(x)} تومان")
    profile = profile.rename(
        columns={
            "segment": "بخش مشتری",
            "customers": "تعداد",
            "recency": "روز از آخرین فعالیت",
            "frequency": "تکرار خرید",
            "monetary_value": "ارزش مالی",
        }
    )
    st.dataframe(profile, use_container_width=True, hide_index=True)

    section_heading("RISK QUEUE", "مشتریان نیازمند توجه")
    cols = ["customer_id", "city", "industry", "segment", "recency", "frequency", "monetary_value", "risk_score", "risk_level"]
    risk_table = customers_model.sort_values("risk_score", ascending=False)[cols].head(100).copy()
    risk_table["segment"] = risk_table["segment"].map(SEGMENT_FA).fillna(risk_table["segment"])
    risk_table["risk_level"] = risk_table["risk_level"].map(RISK_FA).fillna(risk_table["risk_level"])
    risk_table["monetary_value"] = risk_table["monetary_value"].map(lambda x: f"{toman(x)} تومان")
    risk_table = risk_table.rename(
        columns={
            "customer_id": "شناسه مشتری",
            "city": "شهر",
            "industry": "صنعت",
            "segment": "بخش مشتری",
            "recency": "روز از آخرین فعالیت",
            "frequency": "تکرار خرید",
            "monetary_value": "ارزش مالی",
            "risk_score": "امتیاز ریسک",
            "risk_level": "سطح ریسک",
        }
    )
    st.dataframe(risk_table, use_container_width=True, hide_index=True)
    st.warning("Production-grade نیست: Logistic Regression روی Target مصنوعی آموزش دیده است.")


def nikpos_page(scenario, data):
    page_header(
        "تحلیل نیک‌پوز",
        "تفکیک اقتصاد فروش واقعی/Derived از Usage مصنوعی دستگاه؛ NIKPOS = Data Acquisition و NIKSMS = Activation / Retention.",
    )
    plans = plan_performance(scenario)
    devices = data["nikpos"]

    row = st.columns(4)
    with row[0]:
        kpi_card("Plan A", f"{toman(scenario.price_plan_a)} تومان", "real", "price", "Baseline فعلی")
    with row[1]:
        kpi_card("Plan B", f"{toman(scenario.price_plan_b)} تومان", "real", "price", "Baseline فعلی")
    with row[2]:
        kpi_card("ASP", f"{toman(scenario.average_selling_price)} تومان", "derived", "revenue", "بر اساس Mix فعلی")
    with row[3]:
        kpi_card("Active Device Rate", pct(devices["active_device"].mean()), "synthetic", "sales", "دیتای Usage مصنوعی")

    st.write("")
    left, right = st.columns(2)
    with left:
        plans_show = plans.copy()
        plans_show["plan"] = plans_show["plan"].map(PLAN_FA).fillna(plans_show["plan"])
        fig = px.bar(plans_show, x="plan", y="revenue", labels={"plan": "طرح", "revenue": "درآمد"}, color_discrete_sequence=[ACCENT])
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        city = devices.groupby("city", as_index=False).agg(devices=("device_id", "count"), captures_30d=("captures_30d", "sum"))
        city = city.sort_values("captures_30d", ascending=False).head(10)
        fig = px.bar(city, x="city", y="captures_30d", labels={"city": "شهر", "captures_30d": "ثبت شماره"}, color_discrete_sequence=["#6F8EFF"])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    section_heading("PRICE CONTEXT", "تاریخچه قیمت و کمپین", "Historical context؛ برای تحلیل واقعی باید start_date / end_date به دیتاست فروش اضافه شود.")
    history = PRICE_HISTORY.copy()
    history["list_price"] = history["list_price"].map(lambda x: f"{toman(x)} تومان")
    history["campaign_price"] = history["campaign_price"].map(lambda x: f"{toman(x)} تومان")
    history = history.rename(columns={"period": "دوره / سناریو", "list_price": "قیمت مرجع", "campaign_price": "قیمت اجرا", "type": "نوع"})
    st.dataframe(history, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="glass-panel">
            <div class="section-kicker">PRODUCT DATA MODEL</div>
            <div style="font-weight:750;margin-bottom:6px">نیک‌پوز سخت‌افزار جمع‌آوری دیتاست؛ نیک اس‌ام‌اس لایه فعال‌سازی دیتاست.</div>
            <div class="insight-text">در نسخه واقعی می‌توان Eventهایی مثل customer_capture، yellow_button، vip_tag، referral_button و successful_referral را از دستگاه وارد Analytics کرد.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def content_page(scenario):
    page_header(
        "تحلیل محتوا و اینستاگرام",
        "برای اولین بار Snapshot واقعی ۱۰ ریلز از مدل Synthetic جدا شده؛ هدف این صفحه سنجش توزیع عملکرد، نه فقط Average View است.",
    )
    m = content_metrics(scenario)

    row = st.columns(5)
    with row[0]:
        kpi_card("فالوئر", fa_num(m["instagram_followers"]), "real", "followers", "Snapshot")
    with row[1]:
        kpi_card("استوری / روز", fa_num(m["stories_per_day"]), "real", "content", "Baseline")
    with row[2]:
        kpi_card("ریلز / روز", fa_num(m["reels_per_day"]), "real", "content", "Baseline")
    with row[3]:
        kpi_card("کل محتوا / روز", fa_num(m["total_content_per_day"]), "derived", "content", "Stories + Reels")
    with row[4]:
        kpi_card("تیم محتوا", fa_num(m["content_team_size"]), "real", "followers", "۵ نفر")

    st.write("")
    section_heading("10-REEL SNAPSHOT", "عملکرد ۱۰ ریلز اخیر", "این داده Snapshot واقعی است؛ Watch Time / Reach / Saves در دسترس نیست.")

    s1 = st.columns(4)
    with s1[0]:
        kpi_card("Total Views", fa_num(m["total_views"]), "real", "content", "۱۰ ریلز")
    with s1[1]:
        kpi_card("Average Views", fa_num(m["average_views"]), "derived", "content", "به‌شدت تحت تأثیر Hitها")
    with s1[2]:
        kpi_card("Median Views", fa_num(m["median_views"]), "derived", "content", "نماینده بهتر محتوای معمول")
    with s1[3]:
        kpi_card("Top 3 Share", pct(m["top3_view_share"]), "derived", "content", "سهم ۳ ریلز برتر از کل View")

    st.write("")
    s2 = st.columns(4)
    with s2[0]:
        kpi_card("کامنت", fa_num(m["total_comments"]), "real", "content", "مجموع ۱۰ ریلز")
    with s2[1]:
        kpi_card("اشتراک‌گذاری", fa_num(m["total_shares"]), "real", "content", "مجموع ۱۰ ریلز")
    with s2[2]:
        kpi_card("Comments + Shares", fa_num(m["total_interactions"]), "derived", "content", "Interaction Proxy")
    with s2[3]:
        kpi_card("Interaction / View", pct(m["interaction_rate"]), "derived", "content", "بدون Saves / Likes")

    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        reel_plot = REEL_SNAPSHOT.copy()
        fig = px.bar(
            reel_plot,
            x="reel",
            y="views",
            labels={"reel": "محتوا", "views": "View"},
            color="views",
            color_continuous_scale=["#123750", "#1DCBFF"],
        )
        fig.add_hline(
            y=m["median_views"],
            line_dash="dash",
            line_color="#A8B7C7",
            annotation_text=f"Median: {fa_num(m['median_views'])}",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    with right:
        scatter = REEL_SNAPSHOT.copy()
        fig = px.scatter(
            scatter,
            x="views",
            y="interactions",
            size="shares",
            hover_name="reel",
            labels={"views": "View", "interactions": "کامنت + اشتراک‌گذاری"},
            color="interaction_rate",
            color_continuous_scale=["#44556A", "#1DCBFF"],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-type">CONTENT INTELLIGENCE</div>
            <div class="insight-title">Average به‌تنهایی تصویر غلط می‌دهد</div>
            <div class="insight-text">
                Average View برابر {fa_num(m["average_views"])} ولی Median فقط {fa_num(m["median_views"])} است.
                سه ریلز برتر {pct(m["top3_view_share"])} کل ویوها را ساخته‌اند؛ بنابراین Performance فعلی Long-tail / Hit-driven است.
                برای Scorecard تیم، Median View، Share Rate، Comment Rate، Watch Time، Saves، Leads و Sales باید کنار هم باشند.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("ATTRIBUTION", "فروش منتسب به محتوا", "این عدد Estimated است و نباید دوباره به Total Sales اضافه شود.")
    a1, a2, a3 = st.columns(3)
    a1.metric("فروش منتسب / روز", fa_num(m["estimated_sales_per_day"], 1))
    a2.metric("فروش منتسب / ماه", fa_num(m["estimated_sales_per_month"]))
    a3.metric("خروجی محتوا / ماه", fa_num(m["total_content_per_month"]))
    st.caption("برای Attribution واقعی: source tracking، campaign ID، CTA، link click، CRM linkage و attribution window لازم است.")

    table = REEL_SNAPSHOT.copy()
    table["interaction_rate"] = table["interaction_rate"].map(pct)
    table["share_rate"] = table["share_rate"].map(pct)
    table["comment_rate"] = table["comment_rate"].map(pct)
    table = table.rename(
        columns={
            "reel": "ریلز",
            "views": "ویو",
            "comments": "کامنت",
            "shares": "اشتراک‌گذاری",
            "interactions": "کامنت + اشتراک",
            "interaction_rate": "Interaction/View",
            "share_rate": "Share/View",
            "comment_rate": "Comment/View",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def sms_page(data):
    page_header(
        "تحلیل پیامک",
        "فعلاً Dataset پیامک مصنوعی است. Claimهای بازاریابی مثل Open Rate ۹۸٪ تا زمان دریافت Source داخلی به‌عنوان KPI واقعی وارد نشده‌اند.",
    )
    sms = data["sms"].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("ارسال‌شده / دمو", fa_num(sms["sent"].sum()))
    c2.metric("Delivery Rate / دمو", pct(sms["delivered"].sum() / max(sms["sent"].sum(), 1)))
    c3.metric("Click Rate / دمو", pct(sms["clicks"].sum() / max(sms["delivered"].sum(), 1)))

    l, r = st.columns(2)
    with l:
        fig = px.line(sms.tail(90), x="date", y="delivery_rate", labels={"date": "تاریخ", "delivery_rate": "نرخ تحویل"})
        fig.update_traces(line_color=ACCENT)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with r:
        fig = px.area(sms.tail(90), x="date", y="sent", labels={"date": "تاریخ", "sent": "تعداد پیامک"})
        fig.update_traces(line_color="#6F8EFF", fillcolor="rgba(111,142,255,.08)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.info("MARKETING CLAIM ≠ INTERNAL KPI. برای KPI واقعی باید Sent / Delivered / Click / Conversion از دیتای داخلی NIKSMS خوانده شود.")


def anomaly_page(sales_anomalies, sms_anomalies):
    page_header(
        "تشخیص ناهنجاری",
        "Rolling Mean + Z-score روی داده مصنوعی؛ هدف نمایش سازوکار Alerting آینده است.",
    )
    c1, c2 = st.columns(2)
    c1.metric("ناهنجاری فروش / دمو", fa_num(int(sales_anomalies["is_anomaly"].sum())))
    c2.metric("ناهنجاری پیامک / دمو", fa_num(int(sms_anomalies["is_anomaly"].sum())))

    l, r = st.columns(2)
    with l:
        fig = px.line(sales_anomalies, x="date", y="revenue", labels={"date": "تاریخ", "revenue": "درآمد"})
        flagged = sales_anomalies[sales_anomalies["is_anomaly"]]
        fig.add_scatter(x=flagged["date"], y=flagged["revenue"], mode="markers", name="ناهنجاری", marker=dict(size=9, color="#FF7A90"))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        fig = px.line(sms_anomalies, x="date", y="delivery_rate", labels={"date": "تاریخ", "delivery_rate": "نرخ تحویل"})
        flagged = sms_anomalies[sms_anomalies["is_anomaly"]]
        fig.add_scatter(x=flagged["date"], y=flagged["delivery_rate"], mode="markers", name="ناهنجاری", marker=dict(size=9, color="#FF7A90"))
        st.plotly_chart(style_fig(fig), use_container_width=True)


def predictions_page(forecast, forecast_stats, customers_model, risk_stats):
    page_header(
        "پیش‌بینی‌ها",
        "مدل‌ها ساده و Explainable نگه داشته شده‌اند؛ هیچ Forecast یا Churn Output در V0.2 Production-grade نیست.",
    )
    c1, c2 = st.columns(2)
    c1.metric("R² روند درآمد / دمو", fa_digits(f"{forecast_stats['r2']:.3f}"))
    c2.metric("AUC ریزش / دمو", fa_digits(f"{risk_stats['synthetic_holdout_auc']:.3f}"))

    forecast_show = forecast.copy()
    forecast_show["series"] = forecast_show["series"].map(FORECAST_SERIES_FA).fillna(forecast_show["series"])
    fig = px.line(
        forecast_show,
        x="month",
        y="revenue",
        color="series",
        markers=True,
        labels={"month": "ماه", "revenue": "درآمد", "series": "نوع داده"},
        color_discrete_sequence=[ACCENT, "#8A72FF"],
    )
    st.plotly_chart(style_fig(fig, 470), use_container_width=True)

    section_heading("CHURN PROTOTYPE", "مشتریان با ریسک بالاتر")
    top = customers_model.sort_values("risk_score", ascending=False)[
        ["customer_id", "recency", "frequency", "monetary_value", "sms_usage", "nikpos_usage", "risk_score", "risk_level"]
    ].head(50).copy()
    top["risk_level"] = top["risk_level"].map(RISK_FA).fillna(top["risk_level"])
    top["monetary_value"] = top["monetary_value"].map(lambda x: f"{toman(x)} تومان")
    top = top.rename(
        columns={
            "customer_id": "شناسه مشتری",
            "recency": "روز از آخرین فعالیت",
            "frequency": "تکرار خرید",
            "monetary_value": "ارزش مالی",
            "sms_usage": "مصرف پیامک",
            "nikpos_usage": "استفاده از نیک‌پوز",
            "risk_score": "امتیاز ریسک",
            "risk_level": "سطح ریسک",
        }
    )
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.warning("Forecasts و ML outputs آزمایشی هستند و نباید مبنای تصمیم عملیاتی قطعی قرار گیرند.")


def insights_page(insights):
    page_header(
        "بینش‌های خودکار",
        "Rule-based Insight Engine؛ متن Insightها از Scenario، Snapshot محتوا و خروجی مدل‌های دمو ساخته می‌شود.",
    )
    cols = st.columns(2)
    for i, item in enumerate(insights):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-type">{item["type"]}</div>
                    <div class="insight-title">{item["title"]}</div>
                    <div class="insight-text">{item["text"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def pipeline_page():
    page_header(
        "خط لوله تحلیل",
        "معماری امروز Local/Synthetic است؛ دکمه‌های API و n8n نشان می‌دهند ورودی واقعی در آینده از کجا وارد همین Pipeline می‌شود.",
    )
    stages = [
        "ورود داده",
        "اعتبارسنجی",
        "پاک‌سازی",
        "محاسبه KPI",
        "تحلیل روند",
        "تشخیص ناهنجاری",
        "Segmentation",
        "Risk Scoring",
        "Forecast",
        "Insight Engine",
        "Dashboard Update",
    ]
    for i in range(0, len(stages), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(stages):
                with col:
                    st.markdown(f'<div class="pipeline-step">{fa_num(idx + 1)}<br>{stages[idx]}</div>', unsafe_allow_html=True)
        if i + 4 < len(stages):
            st.markdown("<div style='text-align:center;color:#1DCBFF;font-size:1.35rem;margin:4px 0'>↓</div>", unsafe_allow_html=True)

    st.write("")
    section_heading("FUTURE INPUT", "معماری اتصال بعدی")
    st.markdown(
        """
        <div class="glass-panel" style="text-align:center;line-height:2.1;font-weight:650">
            NIK Database / CRM / Panel
            <span style="color:#1DCBFF"> → </span>
            Read-only API
            <span style="color:#1DCBFF"> → </span>
            n8n / Data Pipeline
            <span style="color:#1DCBFF"> → </span>
            Validation
            <span style="color:#1DCBFF"> → </span>
            Analytics Engine
            <span style="color:#1DCBFF"> → </span>
            NIK Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )


def settings_page(scenario, kpis):
    page_header(
        "تنظیمات و سناریو",
        "این صفحه برای Demo مدیریتی است؛ با تغییر یک Input، Dataset مصنوعی و تمام خروجی‌های وابسته دوباره محاسبه می‌شوند.",
    )
    rows = [
        ("قیمت طرح A", f"{toman(scenario.price_plan_a)} تومان", "Baseline / قابل تغییر"),
        ("قیمت طرح B", f"{toman(scenario.price_plan_b)} تومان", "Baseline / قابل تغییر"),
        ("سهم طرح A", pct(scenario.plan_a_share), "Scenario"),
        ("فروش تلفنی روزانه", fa_num(scenario.daily_phone_sales), "Baseline"),
        ("فروش آنلاین ماهانه", fa_num(scenario.monthly_online_sales), "Baseline"),
        ("صف لید", fa_num(scenario.lead_backlog), "Baseline"),
        ("استوری / روز", fa_num(scenario.stories_per_day), "Baseline"),
        ("ریلز / روز", fa_num(scenario.reels_per_day), "Baseline"),
        ("فالوئر", fa_num(scenario.instagram_followers), "Snapshot"),
        ("فروش منتسب به محتوا / روز", fa_num(scenario.content_sales_per_day, 1), "Estimated"),
        ("روز فروش / ماه", fa_num(scenario.sales_days_per_month), "Assumption"),
        ("مشتری مصنوعی", fa_num(scenario.synthetic_customer_count), "Synthetic"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["متغیر", "مقدار فعلی", "نوع داده"]), use_container_width=True, hide_index=True)

    section_heading("LIVE DEPENDENCY", "وابستگی زنده")
    st.code(
        f"Daily Phone Sales = {fa_num(scenario.daily_phone_sales)}\n"
        f"Monthly Phone Units = {fa_num(kpis['monthly_phone_units'])}\n"
        f"Monthly Total Units = {fa_num(kpis['monthly_units'])}\n"
        f"Average Selling Price = {toman(scenario.average_selling_price)} Toman\n"
        f"Derived Monthly Revenue = {toman(kpis['monthly_revenue'])} Toman",
        language="text",
    )
    st.success("برای Demo: فروش تلفنی روزانه را از ۱۰ به ۱۵ تغییر بده؛ Revenue، Units، Channel Mix، Synthetic Trend، Forecast و Insightها با هم تغییر می‌کنند.")


def main():
    scenario, page = scenario_sidebar()
    data = build_synthetic_data(scenario)

    kpis = current_kpis(scenario, data["customers"])
    daily = sales_daily(data["sales"])
    monthly = sales_monthly(data["sales"])
    funnel = lead_funnel(scenario)
    _ = backlog_capacity(scenario)

    customers_model, segment_profile, risk_stats = build_models(data["customers"], scenario.seed)
    forecast, forecast_stats = revenue_forecast(monthly, 3)
    sales_anomalies = detect_anomalies(daily, "revenue", "date", 14)
    sms_anomalies = detect_anomalies(data["sms"], "delivery_rate", "date", 14)
    insights = generate_insights(scenario, kpis, monthly, risk_stats, sales_anomalies, sms_anomalies)

    if page == "Executive Overview":
        executive_overview(scenario, data, kpis, monthly, funnel, forecast, insights)
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
    st.caption("NIK INTELLIGENCE V0.2 — Prototype using aggregate baseline + synthetic/demo data. Not connected to NIK internal systems.")
    st.caption("Forecasts and ML outputs are experimental and should not be used for production decisions.")


if __name__ == "__main__":
    main()
