from __future__ import annotations

import base64
import inspect
import io
import time
from pathlib import Path
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
from media_data import (
    MEDIA_IMAGES,
    MEDIA_VIDEOS,
    get_video,
    synthetic_video_events,
    synthetic_video_metrics,
    synthetic_video_timeline,
)

# V0.7 management layer is optional at import time so a partial redeploy never
# takes the legacy Data Science pages down.
try:
    from management_engine import (
        DEPARTMENTS,
        DEFAULT_TASKS,
        MANAGEMENT_DEMO_DEFAULTS,
        ROLE_MATRIX,
        automation_checks,
        campaign_plan,
        department_kpis,
        department_summary,
        management_defaults,
        organization_score,
        production_plan,
        recommended_tasks,
    )
    MANAGEMENT_ENGINE_AVAILABLE = True
    MANAGEMENT_ENGINE_ERROR = ""
except Exception as _management_import_error:
    MANAGEMENT_ENGINE_AVAILABLE = False
    MANAGEMENT_ENGINE_ERROR = str(_management_import_error)
    DEPARTMENTS = []
    DEFAULT_TASKS = pd.DataFrame()
    ROLE_MATRIX = pd.DataFrame()
    MANAGEMENT_DEMO_DEFAULTS = {}



st.set_page_config(
    page_title="نیک اس‌ام‌اس | پنل مدیریت و اتوماسیون V0.7",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Visual system ----------
ACCENT = "#ADCBFF"
ACCENT_2 = "#4D82B8"
TEXT = "#F8FBFF"
MUTED = "#A8B9CB"
BG = "#07111D"
BORDER = "rgba(255,255,255,.10)"
BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSET_DIR / "images" / "niksms-logo.png"

# UI safety net: if Streamlit Cloud briefly loads an older Scenario module
# during a multi-file redeploy, optional fields still have safe defaults.
SCENARIO_DEFAULTS = {
    "plan_a_price": 15_000_000,
    "plan_b_price": 30_000_000,
    "plan_a_share": 0.50,
    "daily_phone_sales": 10,
    "monthly_online_sales": 20,
    "lead_backlog": 4_000,
    "stories_per_day": 9,
    "reels_per_day": 1,
    "estimated_content_sales_per_day": 2.0,
    "instagram_followers": 207_000,
    "content_team_size": 5,
    "sales_days_per_month": 30,
    "synthetic_customer_count": 5_000,
    "history_months": 12,
    "seed": 42,
}

PAGE_LABELS = {
    "Executive Overview": "مرکز فرمان مدیرعامل",
    "Revenue Intelligence": "هوشمندی درآمد",
    "Scenario Simulator": "شبیه‌ساز تصمیم",
    "Sales Analytics": "تحلیل فروش",
    "Customer Intelligence": "رفتار و ارزش مشتری",
    "Content Analytics": "تحلیل محتوا و اینستاگرام",
    "Media Intelligence": "آزمایشگاه تحلیل محتوا",
    "NIKPOS Analytics": "تحلیل نیک‌پوز",
    "SMS Analytics": "تحلیل پیامک",
    "Anomaly Detection": "تغییرات غیرعادی",
    "Predictions": "پیش‌بینی و سناریو",
    "Automated Insights": "بینش‌های خودکار",
    "Data Center": "مرکز داده",
    "Connections": "مرکز اتصال داده",
    "Analysis Pipeline": "جریان پردازش داده",
    "Settings / Scenario Controls": "تنظیمات و سناریو",
    "Organization Pulse": "نبض سازمان",
    "Task & KPI": "تسک و KPI",
    "Production & QC": "تولید و QC",
    "Campaign Planner": "برنامه‌ریز جشنواره",
    "Automation Center": "مرکز اتوماسیون",
    "Access Control": "دسترسی و نقش‌ها",
}
PAGE_ICONS = {
    "Executive Overview": "◈",
    "Revenue Intelligence": "◫",
    "Scenario Simulator": "⌘",
    "Sales Analytics": "↗",
    "Customer Intelligence": "◎",
    "Content Analytics": "▶",
    "Media Intelligence": "◉",
    "NIKPOS Analytics": "▣",
    "SMS Analytics": "✉",
    "Anomaly Detection": "⚡",
    "Predictions": "⌁",
    "Automated Insights": "✦",
    "Data Center": "▦",
    "Connections": "⇆",
    "Analysis Pipeline": "⇄",
    "Settings / Scenario Controls": "⚙",
    "Organization Pulse": "◌",
    "Task & KPI": "✓",
    "Production & QC": "▤",
    "Campaign Planner": "✦",
    "Automation Center": "⚙",
    "Access Control": "⌾",
}
NAV_GROUPS = {
    "مرکز مدیریت": ["Executive Overview", "Organization Pulse", "Task & KPI"],
    "عملیات سازمان": ["Production & QC", "Sales Analytics", "Customer Intelligence", "NIKPOS Analytics"],
    "رشد و درآمد": ["Revenue Intelligence", "Campaign Planner", "Content Analytics", "Media Intelligence", "SMS Analytics"],
    "هوشمندی": ["Scenario Simulator", "Anomaly Detection", "Predictions", "Automated Insights"],
    "سیستم و اتوماسیون": ["Automation Center", "Connections", "Access Control", "Data Center", "Analysis Pipeline", "Settings / Scenario Controls"],
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
    "real": "داده مبنای واقعی",
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
            radial-gradient(900px 520px at 88% -8%, rgba(173,203,255,.48), transparent 62%),
            radial-gradient(700px 480px at 10% 28%, rgba(91,157,220,.24), transparent 68%),
            radial-gradient(520px 360px at 52% 82%, rgba(173,203,255,.10), transparent 72%),
            linear-gradient(145deg, #06101B 0%, #081A2B 38%, #0C2B45 72%, #123A58 100%);
        background-attachment: fixed;
    }}
    .block-container {{
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }}
    [data-testid="stSidebar"] {{
        background:
            radial-gradient(circle at 50% 0%, rgba(173,203,255,.15), transparent 32%),
            linear-gradient(180deg, rgba(7,24,40,.975), rgba(4,12,22,.985));
        border-left: 1px solid rgba(173,203,255,.13);
        border-right: 0;
        box-shadow: -18px 0 70px rgba(0,0,0,.16);
        backdrop-filter: blur(28px) saturate(135%);
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
        background: radial-gradient(circle, rgba(173,203,255,.18), transparent 70%);
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
        min-height: 42px;
        border-radius: 15px;
        border: 1px solid rgba(173,203,255,.26);
        background:
            linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.015)),
            linear-gradient(145deg, rgba(173,203,255,.16), rgba(50,105,160,.10));
        color: #F4FAFF;
        font-weight: 700;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.12), 0 10px 35px rgba(3,18,32,.20);
        backdrop-filter: blur(16px);
        transition: all .18s ease;
    }}
    div.stButton > button:hover {{
        border-color: rgba(173,203,255,.58);
        background: linear-gradient(145deg, rgba(173,203,255,.22), rgba(77,130,184,.14));
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
            linear-gradient(90deg, rgba(173,203,255,.035), rgba(77,130,184,.02));
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
        background: radial-gradient(circle, rgba(173,203,255,.19), transparent 68%);
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
        background: radial-gradient(circle, rgba(77,130,184,.14), transparent 67%);
        pointer-events: none;
    }}
    .eyebrow {{
        display: inline-flex;
        gap: 8px;
        align-items: center;
        color: #E4F0FF;
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
        background: linear-gradient(90deg, #F2F7FF, #ADCBFF 55%, #8FB7DE);
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
        background:#ADCBFF;
        box-shadow:0 0 12px rgba(173,203,255,.7);
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
        background:radial-gradient(circle, rgba(173,203,255,.16), transparent 70%);
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
        background:linear-gradient(145deg,rgba(173,203,255,.15),rgba(77,130,184,.07));
        border:1px solid rgba(173,203,255,.22);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.08);
    }}
    .kpi-icon svg {{
        width:20px;height:20px;
        stroke:#E8F2FF;
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
    .src-derived {{ color:#DCEAFF;background:rgba(173,203,255,.09);border:1px solid rgba(173,203,255,.18); }}
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
        background:linear-gradient(#ADCBFF,#4D82B8);
    }}
    .insight-type {{
        color:#D9E9FF;
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
        color:#D9E9FF;
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
        padding:10px 13px;
        border-radius:15px;
        background:linear-gradient(145deg,rgba(173,203,255,.075),rgba(255,255,255,.025));
        border:1px solid rgba(173,203,255,.15);
        color:#B9D5EE;
        font-size:.76rem;
        margin-top:7px;
        box-shadow:inset 0 1px 0 rgba(255,255,255,.07);
    }}

    /* Executive composition */
    .hero-brand-row {{
        position:relative; z-index:2;
        display:flex; align-items:center; justify-content:space-between; gap:24px;
    }}
    .hero-brand-copy {{ min-width:0; flex:1; }}
    .hero-logo {{
        width:min(250px, 24vw);
        max-height:88px;
        object-fit:contain;
        filter:drop-shadow(0 10px 35px rgba(173,203,255,.22));
        opacity:.96;
    }}
    .hero-glass {{
        border-color:rgba(173,203,255,.16);
        background:
            linear-gradient(120deg, rgba(255,255,255,.09), rgba(255,255,255,.025) 50%, rgba(173,203,255,.045)),
            linear-gradient(180deg, rgba(9,32,52,.58), rgba(7,19,32,.42));
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,.13),
            inset 0 -1px 0 rgba(173,203,255,.05),
            0 30px 95px rgba(0,0,0,.24),
            0 0 70px rgba(83,148,208,.06);
    }}
    .hero-glass::before {{
        width:500px; height:230px; left:-140px; top:-125px;
        background:radial-gradient(circle, rgba(173,203,255,.27), transparent 67%);
        filter:blur(12px);
    }}
    .hero-glass::after {{
        width:330px; height:330px; right:-120px; bottom:-220px;
        background:radial-gradient(circle, rgba(99,164,223,.20), transparent 66%);
    }}

    .executive-strip {{
        display:grid;
        grid-template-columns:repeat(4,minmax(0,1fr));
        gap:10px;
        margin:14px 0 24px;
        padding:10px;
        border-radius:22px;
        border:1px solid rgba(173,203,255,.12);
        background:linear-gradient(145deg,rgba(173,203,255,.055),rgba(255,255,255,.018));
        box-shadow:inset 0 1px 0 rgba(255,255,255,.07), 0 18px 60px rgba(0,0,0,.12);
        backdrop-filter:blur(18px);
    }}
    .strip-item {{
        min-width:0;
        padding:11px 13px;
        border-radius:15px;
        border:1px solid rgba(255,255,255,.055);
        background:rgba(255,255,255,.025);
    }}
    .strip-label {{ color:#8298AD; font-size:.71rem; margin-bottom:4px; }}
    .strip-value {{ color:#F4F9FF; font-size:1.06rem; font-weight:800; }}

    .glass-kpi, .glass-panel, .insight-card, .pipeline-step {{
        transition:border-color .18s ease, transform .18s ease, box-shadow .18s ease;
    }}
    @media (hover:hover) {{
        .glass-kpi:hover, .glass-panel:hover, .insight-card:hover {{
            border-color:rgba(173,203,255,.20);
            box-shadow:inset 0 1px 0 rgba(255,255,255,.10), 0 22px 70px rgba(0,0,0,.16);
        }}
    }}
    .glass-kpi {{
        border-color:rgba(173,203,255,.12);
        background:
            linear-gradient(155deg,rgba(255,255,255,.082),rgba(255,255,255,.020)),
            linear-gradient(135deg,rgba(173,203,255,.035),transparent 70%);
    }}
    .kpi-icon {{
        box-shadow:inset 0 1px 0 rgba(255,255,255,.12), 0 8px 28px rgba(84,147,205,.10);
    }}
    .section-kicker {{ color:#ADCBFF; opacity:.92; }}
    .section-title {{ letter-spacing:-.018em; }}

    .demo-badge {{
        display:inline-flex; align-items:center; gap:7px;
        padding:6px 10px; margin-bottom:10px;
        border-radius:999px;
        color:#DDEBFA; font-size:.69rem; font-weight:800;
        border:1px solid rgba(173,203,255,.16);
        background:rgba(173,203,255,.07);
    }}
    .demo-badge::before {{
        content:""; width:7px; height:7px; border-radius:50%;
        background:#ADCBFF; box-shadow:0 0 15px rgba(173,203,255,.75);
    }}
    .media-meta {{ color:#8EA2B5; font-size:.77rem; line-height:1.8; margin-top:5px; }}

    /* Streamlit controls closer to a polished product UI */
    div[role="radiogroup"] label {{
        border-radius:12px;
        padding:5px 7px;
        margin:2px 0;
        transition:background .16s ease;
    }}
    div[role="radiogroup"] label:has(input:checked) {{
        background:linear-gradient(90deg,rgba(173,203,255,.11),rgba(173,203,255,.025));
        border:1px solid rgba(173,203,255,.11);
    }}
    [data-testid="stTabs"] button {{ font-weight:700; }}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color:#ADCBFF; }}
    [data-testid="stFileUploader"] {{
        background:rgba(255,255,255,.022);
        border:1px dashed rgba(173,203,255,.16);
        padding:5px;
    }}
    [data-testid="stDataFrame"] {{
        box-shadow:0 15px 50px rgba(0,0,0,.10);
        background:rgba(7,19,31,.25);
    }}
    ::-webkit-scrollbar {{ width:10px; height:10px; }}
    ::-webkit-scrollbar-thumb {{ background:rgba(173,203,255,.18); border-radius:999px; }}
    ::-webkit-scrollbar-track {{ background:rgba(255,255,255,.015); }}

    @media (max-width: 900px) {{
        .hero-brand-row {{ align-items:flex-start; }}
        .hero-logo {{ width:150px; }}
        .executive-strip {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    }}
    @media (max-width: 640px) {{
        .hero-brand-row {{ flex-direction:column-reverse; }}
        .hero-logo {{ width:135px; }}
        .executive-strip {{ grid-template-columns:1fr; }}
        .hero-glass {{ padding:20px; border-radius:23px; }}
    }}
    hr {{ border-color: rgba(173,203,255,.08); }}
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    /* V0.6 — Executive Command Center */
    .ceo-command {
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: 1.25fr 1fr;
        gap: 22px;
        padding: 24px;
        margin: 8px 0 18px;
        border-radius: 26px;
        border: 1px solid rgba(173,203,255,.18);
        background:
            radial-gradient(460px 220px at 100% 0%, rgba(173,203,255,.14), transparent 68%),
            linear-gradient(135deg, rgba(173,203,255,.095), rgba(255,255,255,.018));
        box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 24px 70px rgba(0,0,0,.18);
        backdrop-filter: blur(26px) saturate(140%);
    }
    .ceo-command::after {
        content:"";
        position:absolute;
        width:240px;height:240px;
        left:-120px;bottom:-165px;
        border-radius:50%;
        background:radial-gradient(circle,rgba(173,203,255,.18),transparent 70%);
        pointer-events:none;
    }
    .ceo-overline { color:#BFD7F5; font-size:.70rem; font-weight:850; letter-spacing:.09em; margin-bottom:8px; }
    .ceo-status-line { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    .ceo-status-title { color:#FFFFFF; font-size:1.65rem; font-weight:850; letter-spacing:-.025em; }
    .ceo-status-copy { color:#A9BBCD; font-size:.86rem; line-height:1.9; margin-top:8px; max-width:720px; }
    .ceo-status-badge {
        display:inline-flex;align-items:center;gap:7px;
        padding:6px 10px;border-radius:999px;
        font-size:.72rem;font-weight:850;
        border:1px solid rgba(255,255,255,.10);
    }
    .ceo-status-badge::before { content:""; width:7px;height:7px;border-radius:50%; }
    .ceo-watch { color:#FFE0A3;background:rgba(244,180,79,.09);border-color:rgba(244,180,79,.18); }
    .ceo-watch::before { background:#F4B44F; box-shadow:0 0 13px rgba(244,180,79,.60); }
    .ceo-good { color:#A9F5CF;background:rgba(48,205,135,.08);border-color:rgba(48,205,135,.17); }
    .ceo-good::before { background:#30CD87; box-shadow:0 0 13px rgba(48,205,135,.55); }
    .ceo-alert { color:#FFB2BE;background:rgba(255,91,118,.08);border-color:rgba(255,91,118,.18); }
    .ceo-alert::before { background:#FF5B76; box-shadow:0 0 13px rgba(255,91,118,.55); }
    .ceo-command-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; align-content:start; }
    .ceo-command-cell {
        min-width:0;padding:13px 14px;border-radius:17px;
        background:rgba(4,16,28,.30);border:1px solid rgba(255,255,255,.07);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
    }
    .ceo-command-label { color:#8297AB;font-size:.68rem;margin-bottom:4px; }
    .ceo-command-value { color:#F8FBFF;font-size:1.12rem;font-weight:820;letter-spacing:-.02em; }
    .ceo-command-foot { color:#6F8295;font-size:.61rem;margin-top:3px;line-height:1.5; }

    .command-metric {
        position:relative;overflow:hidden;min-height:138px;
        padding:16px 17px;border-radius:20px;
        background:linear-gradient(145deg,rgba(255,255,255,.066),rgba(255,255,255,.018));
        border:1px solid rgba(255,255,255,.095);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.065),0 16px 45px rgba(0,0,0,.12);
    }
    .command-metric::after { content:"";position:absolute;width:95px;height:95px;left:-40px;bottom:-58px;border-radius:50%;background:radial-gradient(circle,rgba(173,203,255,.13),transparent 70%); }
    .command-metric-top { display:flex;align-items:center;justify-content:space-between;gap:8px; }
    .command-metric-label { color:#91A5B9;font-size:.74rem;font-weight:650; }
    .command-metric-value { color:#FFFFFF;font-size:1.45rem;font-weight:850;letter-spacing:-.035em;margin-top:10px; }
    .command-metric-note { color:#73879A;font-size:.64rem;line-height:1.6;margin-top:4px; }
    .trend-pill { display:inline-flex;align-items:center;padding:4px 7px;border-radius:999px;font-size:.64rem;font-weight:800;white-space:nowrap; }
    .trend-up { color:#9EF1C5;background:rgba(50,205,137,.08);border:1px solid rgba(50,205,137,.15); }
    .trend-down { color:#FFA9B7;background:rgba(255,91,118,.08);border:1px solid rgba(255,91,118,.15); }
    .trend-flat { color:#D6E5F8;background:rgba(173,203,255,.08);border:1px solid rgba(173,203,255,.15); }

    .priority-card {
        min-height:184px;position:relative;overflow:hidden;
        padding:18px;border-radius:21px;
        background:linear-gradient(145deg,rgba(255,255,255,.058),rgba(255,255,255,.018));
        border:1px solid rgba(255,255,255,.09);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
    }
    .priority-number { color:#ADCBFF;font-size:.72rem;font-weight:900;letter-spacing:.08em;margin-bottom:9px; }
    .priority-title { color:#F8FBFF;font-size:1rem;font-weight:820;line-height:1.65; }
    .priority-text { color:#93A6B8;font-size:.78rem;line-height:1.9;margin-top:7px; }
    .priority-footer { position:absolute;right:18px;left:18px;bottom:14px;display:flex;justify-content:space-between;align-items:center;gap:8px; }
    .severity-pill { font-size:.62rem;font-weight:850;padding:4px 7px;border-radius:999px; }
    .sev-high { color:#FFB2BE;background:rgba(255,91,118,.08);border:1px solid rgba(255,91,118,.15); }
    .sev-med { color:#FFE0A3;background:rgba(244,180,79,.08);border:1px solid rgba(244,180,79,.15); }
    .sev-info { color:#DDEBFF;background:rgba(173,203,255,.08);border:1px solid rgba(173,203,255,.15); }
    .priority-source { color:#6F8397;font-size:.61rem; }

    .money-summary { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:4px 0 12px; }
    .money-chip { padding:11px 12px;border-radius:15px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.07); }
    .money-chip-label { color:#8296AA;font-size:.64rem; }
    .money-chip-value { color:#F8FBFF;font-size:1rem;font-weight:820;margin-top:3px; }

    .leverage-card {
        min-height:146px;padding:17px 18px;border-radius:20px;
        background:linear-gradient(145deg,rgba(173,203,255,.075),rgba(255,255,255,.018));
        border:1px solid rgba(173,203,255,.12);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
    }
    .leverage-kicker { color:#ADCBFF;font-size:.67rem;font-weight:850;letter-spacing:.07em; }
    .leverage-title { color:#F6FAFF;font-size:.92rem;font-weight:780;margin-top:6px; }
    .leverage-value { color:#FFFFFF;font-size:1.35rem;font-weight:880;letter-spacing:-.03em;margin-top:7px; }
    .leverage-note { color:#7F93A7;font-size:.65rem;line-height:1.65;margin-top:4px; }

    .gap-row {
        display:grid;grid-template-columns:42px 1fr auto;align-items:center;gap:12px;
        padding:12px 13px;margin-bottom:8px;border-radius:16px;
        background:rgba(255,255,255,.028);border:1px solid rgba(255,255,255,.065);
    }
    .gap-index { width:34px;height:34px;display:flex;align-items:center;justify-content:center;border-radius:11px;background:rgba(173,203,255,.08);border:1px solid rgba(173,203,255,.13);color:#DCEAFF;font-size:.73rem;font-weight:850; }
    .gap-title { color:#F1F7FF;font-size:.82rem;font-weight:760; }
    .gap-text { color:#758A9E;font-size:.64rem;margin-top:2px;line-height:1.55; }
    .gap-status { color:#FFD99A;font-size:.62rem;font-weight:820;padding:4px 7px;border-radius:999px;background:rgba(244,180,79,.07);border:1px solid rgba(244,180,79,.13);white-space:nowrap; }

    @media (max-width: 900px) {
        .ceo-command { grid-template-columns:1fr; }
        .money-summary { grid-template-columns:1fr; }
    }
    @media (max-width: 640px) {
        .ceo-command-grid { grid-template-columns:1fr 1fr; }
        .gap-row { grid-template-columns:38px 1fr; }
        .gap-status { grid-column:2; justify-self:start; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- V0.6 polish layer: liquid glass, motion, presentation mode ----------
V06_POLISH_CSS = """
<style>
:root {
    --nik-ice:#EAF4FF;
    --nik-sky:#ADCBFF;
    --nik-deep:#17365F;
    --nik-night:#050D17;
    --nik-glass:rgba(255,255,255,.075);
    --nik-glass-strong:rgba(255,255,255,.105);
    --nik-edge:rgba(255,255,255,.16);
}
.stApp {
    background:
        radial-gradient(800px 440px at 84% 3%, rgba(173,203,255,.34), transparent 64%),
        radial-gradient(660px 420px at 10% 32%, rgba(85,142,199,.20), transparent 66%),
        radial-gradient(520px 360px at 58% 91%, rgba(173,203,255,.075), transparent 72%),
        linear-gradient(145deg,#040B13 0%,#071421 38%,#0A2034 70%,#102D48 100%);
}
.hero-glass,.ceo-command,.glass-kpi,.glass-panel,.command-metric,.priority-card,.leverage-card,
[data-testid="stMetric"],div[data-testid="stExpander"] {
    -webkit-backdrop-filter: blur(30px) saturate(145%);
    backdrop-filter: blur(30px) saturate(145%);
}
.hero-glass,.ceo-command {
    border-color:rgba(255,255,255,.145);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.14),inset 0 -1px 0 rgba(255,255,255,.025),0 26px 74px rgba(0,0,0,.20);
}
.glass-kpi,.command-metric,.priority-card,.leverage-card,.glass-panel {
    transition:transform .22s cubic-bezier(.2,.8,.2,1),border-color .22s ease,background .22s ease,box-shadow .22s ease;
}
@media (hover:hover) and (pointer:fine) {
    .glass-kpi:hover,.command-metric:hover,.priority-card:hover,.leverage-card:hover {
        transform:translateY(-2px) scale(1.006);
        border-color:rgba(173,203,255,.25);
        background:linear-gradient(145deg,rgba(255,255,255,.095),rgba(255,255,255,.026));
        box-shadow:inset 0 1px 0 rgba(255,255,255,.15),0 24px 60px rgba(0,0,0,.18);
    }
}
.block-container > div > div > div { animation:v06-rise .42s cubic-bezier(.2,.75,.25,1) both; }
@keyframes v06-rise { from {opacity:0;transform:translateY(5px)} to {opacity:1;transform:none} }
@media (prefers-reduced-motion:reduce) {
    *,*::before,*::after { animation:none !important;transition:none !important;scroll-behavior:auto !important; }
}
.v06-brief {
    position:relative;overflow:hidden;padding:20px 22px;border-radius:22px;margin:8px 0 16px;
    background:linear-gradient(135deg,rgba(173,203,255,.105),rgba(255,255,255,.025));
    border:1px solid rgba(173,203,255,.18);box-shadow:inset 0 1px 0 rgba(255,255,255,.10);
}
.v06-brief::after {content:"";position:absolute;left:-70px;top:-90px;width:190px;height:190px;border-radius:50%;background:radial-gradient(circle,rgba(173,203,255,.18),transparent 70%)}
.v06-brief-kicker {font-size:.68rem;color:#ADCBFF;font-weight:900;letter-spacing:.08em;margin-bottom:7px}
.v06-brief-title {font-size:1.06rem;color:#F9FCFF;font-weight:850;margin-bottom:6px}
.v06-brief-copy {font-size:.82rem;color:#AFC0D1;line-height:2;max-width:1050px}
.v06-brief-footer {display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}
.confidence-pill,.trust-pill {display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:5px 9px;font-size:.64rem;font-weight:850;border:1px solid rgba(255,255,255,.10)}
.conf-high{color:#A5F2CB;background:rgba(48,205,135,.08);border-color:rgba(48,205,135,.16)}
.conf-med{color:#FFE0A3;background:rgba(244,180,79,.08);border-color:rgba(244,180,79,.16)}
.conf-low{color:#FFD0D8;background:rgba(255,91,118,.07);border-color:rgba(255,91,118,.14)}
.trust-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:8px 0 17px}
.trust-cell{padding:12px 13px;border-radius:16px;background:rgba(255,255,255,.028);border:1px solid rgba(255,255,255,.07)}
.trust-label{color:#8096AA;font-size:.64rem}.trust-value{color:#F5FAFF;font-size:.92rem;font-weight:820;margin-top:4px}.trust-note{color:#6E8397;font-size:.60rem;line-height:1.5;margin-top:3px}
.sim-hero{padding:20px 22px;border-radius:24px;background:linear-gradient(145deg,rgba(173,203,255,.085),rgba(255,255,255,.022));border:1px solid rgba(173,203,255,.15);box-shadow:inset 0 1px 0 rgba(255,255,255,.09)}
.sim-value{font-size:1.55rem;font-weight:880;color:#fff;letter-spacing:-.03em}.sim-label{font-size:.70rem;color:#88A0B7}.sim-delta-up{color:#9DF0C4}.sim-delta-down{color:#FFB4C0}.sim-delta-flat{color:#D8E6F7}
.connection-card{min-height:154px;padding:17px 18px;border-radius:20px;background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.017));border:1px solid rgba(255,255,255,.085);box-shadow:inset 0 1px 0 rgba(255,255,255,.065)}
.connection-top{display:flex;align-items:center;justify-content:space-between;gap:8px}.connection-name{font-size:.92rem;font-weight:820;color:#F6FAFF}.connection-state{font-size:.62rem;font-weight:850;padding:4px 7px;border-radius:999px}.state-on{color:#A5F2CB;background:rgba(48,205,135,.08);border:1px solid rgba(48,205,135,.15)}.state-off{color:#C8D6E7;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.08)}
.connection-copy{font-size:.68rem;color:#7D92A6;line-height:1.8;margin-top:10px}.connection-fresh{font-size:.61rem;color:#668096;margin-top:8px}
.presentation-ribbon{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:9px 12px;margin-bottom:10px;border-radius:14px;background:rgba(173,203,255,.075);border:1px solid rgba(173,203,255,.14);font-size:.69rem;color:#CFE1F4}
[data-testid="stSidebar"] { transition:transform .24s ease,opacity .24s ease; }
@media(max-width:900px){.trust-grid{grid-template-columns:1fr 1fr}}
@media(max-width:560px){.trust-grid{grid-template-columns:1fr}}
</style>
"""
st.markdown(V06_POLISH_CSS, unsafe_allow_html=True)



def asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def make_scenario(**kwargs) -> Scenario:
    """Build Scenario defensively; avoids crash if an older module is temporarily cached."""
    try:
        accepted = set(inspect.signature(Scenario).parameters)
    except Exception:
        accepted = set(kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return Scenario(**filtered)


def baseline_value(key: str, fallback=None):
    if fallback is None:
        fallback = SCENARIO_DEFAULTS.get(key)
    try:
        return BUSINESS_BASELINE.get(key, fallback)
    except Exception:
        return fallback


def scenario_value(scenario, key: str, baseline_key: str | None = None, fallback=None):
    if hasattr(scenario, key):
        return getattr(scenario, key)
    if baseline_key:
        return baseline_value(baseline_key, fallback)
    return fallback


def scenario_summary(scenario) -> dict[str, float]:
    price_a = float(scenario_value(scenario, "price_plan_a", "plan_a_price", 15_000_000))
    price_b = float(scenario_value(scenario, "price_plan_b", "plan_b_price", 30_000_000))
    share_a = float(scenario_value(scenario, "plan_a_share", "plan_a_share", 0.50))
    share_b = float(getattr(scenario, "plan_b_share", 1.0 - share_a))
    sales_days = float(scenario_value(scenario, "sales_days_per_month", "sales_days_per_month", 30))
    phone_daily = float(scenario_value(scenario, "daily_phone_sales", "daily_phone_sales", 10))
    online_monthly = float(scenario_value(scenario, "monthly_online_sales", "monthly_online_sales", 20))
    stories = float(scenario_value(scenario, "stories_per_day", "stories_per_day", 9))
    reels = float(scenario_value(scenario, "reels_per_day", "reels_per_day", 1))
    asp = float(getattr(scenario, "average_selling_price", price_a * share_a + price_b * share_b))
    monthly_phone = float(getattr(scenario, "monthly_phone_units", phone_daily * sales_days))
    monthly_units = float(getattr(scenario, "monthly_units", monthly_phone + online_monthly))
    return {
        "price_a": price_a,
        "price_b": price_b,
        "share_a": share_a,
        "share_b": share_b,
        "sales_days": sales_days,
        "phone_daily": phone_daily,
        "online_monthly": online_monthly,
        "lead_backlog": float(scenario_value(scenario, "lead_backlog", "lead_backlog", 4_000)),
        "stories": stories,
        "reels": reels,
        "total_content": stories + reels,
        "content_sales": float(scenario_value(scenario, "content_sales_per_day", "estimated_content_sales_per_day", 2.0)),
        "followers": float(scenario_value(scenario, "instagram_followers", "instagram_followers", 207_000)),
        "team_size": float(scenario_value(scenario, "content_team_size", "content_team_size", 5)),
        "customer_count": float(scenario_value(scenario, "synthetic_customer_count", None, 5_000)),
        "seed": int(scenario_value(scenario, "seed", None, 42)),
        "asp": asp,
        "monthly_phone": monthly_phone,
        "monthly_units": monthly_units,
        "monthly_revenue": monthly_units * asp,
    }


def normalize_kpis(scenario, kpis: dict) -> dict:
    """Guarantee the KPI contract even if an older analytics_engine is temporarily loaded."""
    out = dict(kpis)
    sm = scenario_summary(scenario)
    monthly_phone = float(out.get("monthly_phone_units", sm["monthly_phone"]))
    monthly_online = float(out.get("monthly_online_units", sm["online_monthly"]))
    monthly_units = float(out.get("monthly_units", monthly_phone + monthly_online))
    monthly_revenue = float(out.get("monthly_revenue", monthly_units * sm["asp"]))
    backlog = sm["lead_backlog"]
    content_monthly = sm["content_sales"] * sm["sales_days"]
    out.update({
        "monthly_phone_units": monthly_phone,
        "monthly_online_units": monthly_online,
        "monthly_units": monthly_units,
        "monthly_revenue": monthly_revenue,
        "average_selling_price": float(out.get("average_selling_price", sm["asp"])),
        "lead_pool": float(out.get("lead_pool", backlog)),
        "phone_share": float(out.get("phone_share", monthly_phone / monthly_units if monthly_units else 0.0)),
        "online_share": float(out.get("online_share", monthly_online / monthly_units if monthly_units else 0.0)),
        "content_monthly_sales_estimated": float(out.get("content_monthly_sales_estimated", out.get("content_monthly_sales", content_monthly))),
        "content_monthly_sales": float(out.get("content_monthly_sales", content_monthly)),
        "backlog_sales_volume_ratio": float(out.get("backlog_sales_volume_ratio", monthly_units / backlog if backlog else 0.0)),
        "backlog_months_of_sales": float(out.get("backlog_months_of_sales", backlog / monthly_units if monthly_units else np.inf)),
    })
    return out


def content_metrics_safe(scenario) -> dict[str, float]:
    """Supplement missing content fields when an older analytics module is loaded."""
    try:
        result = dict(content_metrics(scenario))
    except Exception:
        result = {}
    sm = scenario_summary(scenario)
    snapshot = reel_snapshot_metrics()
    defaults = {
        "stories_per_day": sm["stories"],
        "reels_per_day": sm["reels"],
        "total_content_per_day": sm["total_content"],
        "stories_per_month": sm["stories"] * sm["sales_days"],
        "reels_per_month": sm["reels"] * sm["sales_days"],
        "total_content_per_month": sm["total_content"] * sm["sales_days"],
        "estimated_sales_per_day": sm["content_sales"],
        "estimated_sales_per_month": sm["content_sales"] * sm["sales_days"],
        "instagram_followers": sm["followers"],
        "content_team_size": sm["team_size"],
        **snapshot,
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    return result

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
                <span class="source-tag src-real">داده مبنای واقعی</span>
                <span class="source-tag src-derived">محاسبه‌شده از داده مبنا</span>
                <span class="source-tag src-estimated">تخمینی / انتساب فروش</span>
                <span class="source-tag src-synthetic">داده مصنوعی / مدل آزمایشی</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(page_title: str, subtitle: str = ""):
    logo_uri = asset_data_uri(LOGO_PATH)
    logo_html = f'<img class="hero-logo" src="{logo_uri}" alt="NIKSMS">' if logo_uri else ""
    st.markdown(
        f"""
        <div class="hero-glass">
            <div class="hero-brand-row">
                <div class="hero-brand-copy">
                    <div class="eyebrow"><span>◈</span> NIK INTELLIGENCE / V0.6</div>
                    <div class="hero-title">نیک اس‌ام‌اس <span class="accent">| تحلیل داده</span></div>
                    <div class="hero-subtitle">{subtitle or "سامانه هوشمندی داده و تصمیم‌سازی مدیریتی؛ ترکیب داده مبنای واقعی، محاسبات قابل توضیح و مدل‌های آزمایشی."}</div>
                </div>
                {logo_html}
            </div>
            <div class="hero-meta">
                <span class="meta-pill"><span class="meta-dot"></span> {page_title}</span>
                <span class="meta-pill">نمای داده: ۲۹ اوت ۲۰۲۶</span>
                <span class="meta-pill">دموی مدیریتی / V0.6</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_col, n8n_col, status_col = st.columns([1, 1, 4])
    with api_col:
        if st.button("اتصال API", use_container_width=True, key=f"api_{page_title}"):
            st.toast("دکمه اتصال API فقط‌خواندنی آماده است؛ در V0.6 هیچ اطلاعات دسترسی یا اتصال واقعی وجود ندارد.")
    with n8n_col:
        if st.button("اتصال n8n", use_container_width=True, key=f"n8n_{page_title}"):
            st.toast("این دکمه جایگاه آماده اتصال n8n است؛ گردش‌کار واقعی بعداً ساخته می‌شود.")
    with status_col:
        st.markdown(
            '<div class="connector-status">وضعیت اتصال: <b>حالت دمو</b> — API، پایگاه داده و n8n هنوز متصل نیستند.</div>',
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def build_synthetic_data(scenario: Scenario) -> Dict[str, pd.DataFrame]:
    data = generate_all(scenario)
    data.setdefault("content_snapshot", REEL_SNAPSHOT.copy())
    return data


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


def confidence_label(level: str) -> tuple[str, str]:
    mapping = {
        "high": ("اطمینان بالا", "conf-high"),
        "medium": ("اطمینان متوسط", "conf-med"),
        "low": ("داده ناکافی", "conf-low"),
    }
    return mapping.get(level, mapping["medium"])


def build_ceo_brief(scenario, kpis: dict) -> tuple[str, str, str]:
    """Rule-based executive brief. It never presents synthetic ML as business fact."""
    sm = scenario_summary(scenario)
    reel = reel_snapshot_metrics()
    phone_share = float(kpis.get("phone_share", 0.0))
    total_rev = float(kpis.get("monthly_revenue", 0.0))
    plan_b_revenue_share = 0.0
    if total_rev > 0:
        plan_b_revenue_share = (float(kpis.get("monthly_units", 0.0)) * sm["share_b"] * sm["price_b"]) / total_rev
    parts = [
        f"فروش مدل فعلی حدود {fa_num(kpis.get('monthly_units', 0))} دستگاه در ماه و درآمد محاسبه‌شده {toman(kpis.get('monthly_revenue', 0))} تومان است.",
        f"حدود {pct(phone_share)} از حجم فروش مدل از کانال تلفنی می‌آید.",
        f"با ترکیب فعلی، طرح B حدود {pct(plan_b_revenue_share)} از درآمد مدل را می‌سازد.",
        f"در نمای ۱۰ ریلز، سه محتوای برتر {pct(float(reel.get('top3_view_share', 0)))} از کل بازدید را گرفته‌اند.",
        "مهم‌ترین شکاف برای تصمیم واقعی، نبود مسیر یکپارچه Lead → Call → Sale و انتساب محتوا به فروش است.",
    ]
    confidence = "high" if sm["lead_backlog"] >= 0 and total_rev > 0 else "medium"
    return " ".join(parts), confidence, "قاعده‌محور؛ مبتنی بر Baseline و محاسبات توضیح‌پذیر"


def render_data_trust_layer():
    st.markdown(
        '''<div class="trust-grid">
            <div class="trust-cell"><div class="trust-label">داده مبنای واقعی</div><div class="trust-value">قیمت، فروش، صف لید، محتوا</div><div class="trust-note">ورودی Aggregate ثبت‌شده؛ نه اتصال زنده.</div></div>
            <div class="trust-cell"><div class="trust-label">محاسبه‌شده</div><div class="trust-value">Revenue / Mix / Sensitivity</div><div class="trust-note">قابل بازتولید از ورودی‌های فعلی.</div></div>
            <div class="trust-cell"><div class="trust-label">تخمینی</div><div class="trust-value">انتساب فروش محتوا</div><div class="trust-note">تا زمان Source Tracking، اطمینان محدود.</div></div>
            <div class="trust-cell"><div class="trust-label">آزمایشی</div><div class="trust-value">Forecast / Churn / Anomaly</div><div class="trust-note">Synthetic؛ برای معماری و دمو.</div></div>
        </div>''',
        unsafe_allow_html=True,
    )


def inject_presentation_mode_css(enabled: bool):
    if not enabled:
        return
    st.markdown(
        '''<style>
        [data-testid="stSidebar"]{display:none !important}
        [data-testid="collapsedControl"]{display:none !important}
        .block-container{max-width:1540px !important;padding-left:2rem !important;padding-right:2rem !important}
        </style>''',
        unsafe_allow_html=True,
    )



def scenario_sidebar() -> Tuple[Scenario, str]:
    if "presentation_mode" not in st.session_state:
        st.session_state.presentation_mode = False
    if "nav_group" not in st.session_state:
        st.session_state.nav_group = "مرکز مدیریت"

    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=185)
    st.sidebar.markdown(
        """
        <div style="padding:2px 4px 4px">
            <div style="font-size:.70rem;color:#DCEAFF;font-weight:800;letter-spacing:.08em">NIK MANAGEMENT OS</div>
            <div style="font-size:1.22rem;color:#F7FBFF;font-weight:850;margin-top:3px">مدیریت، داده و اتوماسیون نیک</div>
            <div style="font-size:.73rem;color:#8FA7BD;margin-top:4px">Management & Automation OS · V0.7</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("حالت ارائه مدیرعامل", use_container_width=True, type="primary"):
        st.session_state.presentation_mode = True
        st.rerun()

    st.sidebar.markdown("---")
    group = st.sidebar.selectbox("حوزه", list(NAV_GROUPS.keys()), key="nav_group")
    group_pages = NAV_GROUPS[group]
    page = st.sidebar.radio(
        "صفحه",
        group_pages,
        format_func=lambda x: f"{PAGE_ICONS[x]}   {PAGE_LABELS[x]}",
        key=f"page_{group}",
    )

    with st.sidebar.expander("کنترل سناریوی مدیریتی", expanded=False):
        price_a = st.number_input("قیمت طرح A (تومان)", 1_000_000, 200_000_000, int(baseline_value("plan_a_price", 15_000_000)), 1_000_000)
        price_b = st.number_input("قیمت طرح B (تومان)", 1_000_000, 300_000_000, int(baseline_value("plan_b_price", 30_000_000)), 1_000_000)
        share_a_pct = st.slider("سهم طرح A", 0, 100, int(baseline_value("plan_a_share", 0.50) * 100), 5)
        phone = st.number_input("فروش تلفنی روزانه", 0, 500, int(baseline_value("daily_phone_sales", 10)), 1)
        online = st.number_input("فروش آنلاین ماهانه", 0, 5_000, int(baseline_value("monthly_online_sales", 20)), 5)
        backlog = st.number_input("صف فعلی لید", 0, 500_000, int(baseline_value("lead_backlog", 4_000)), 100)
        sales_days = st.slider("فرض روز فروش در ماه", 20, 31, int(baseline_value("sales_days_per_month", 30)), 1)

    with st.sidebar.expander("کنترل محتوا"):
        stories = st.number_input("استوری روزانه", 0, 100, int(baseline_value("stories_per_day", 9)), 1)
        reels = st.number_input("ریلز روزانه", 0, 20, int(baseline_value("reels_per_day", 1)), 1)
        content_sales = st.number_input("فروش منتسب به محتوا / روز", 0.0, 100.0, float(baseline_value("estimated_content_sales_per_day", 2.0)), 0.5, help="انتساب فروش تخمینی است و به فروش کل اضافه نمی‌شود.")
        followers = st.number_input("فالوئر اینستاگرام", 0, 10_000_000, int(baseline_value("instagram_followers", 207_000)), 1_000)
        team_size = st.number_input("اندازه تیم محتوا", 1, 100, int(baseline_value("content_team_size", 5)), 1)

    with st.sidebar.expander("تنظیمات مدل آزمایشی"):
        customers = st.number_input("مشتری مصنوعی", 500, 50_000, 5_000, 500)
        months = st.slider("ماه‌های داده تاریخی مصنوعی", 3, 36, 12, 1)
        seed = st.number_input("بذر تصادفی (Seed)", 1, 999_999, 42, 1)

    scenario = make_scenario(
        price_plan_a=float(price_a), price_plan_b=float(price_b), plan_a_share=float(share_a_pct) / 100,
        daily_phone_sales=int(phone), monthly_online_sales=int(online), lead_backlog=int(backlog),
        stories_per_day=int(stories), reels_per_day=int(reels), content_sales_per_day=float(content_sales),
        instagram_followers=int(followers), content_team_size=int(team_size), synthetic_customer_count=int(customers),
        history_months=int(months), sales_days_per_month=int(sales_days), seed=int(seed),
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("اجرای تحلیل کامل", use_container_width=True):
        stages = ["بارگذاری داده", "اعتبارسنجی داده", "پاک‌سازی", "محاسبه شاخص‌های کلیدی", "تحلیل روند", "تشخیص تغییرات غیرعادی", "اجرای مدل‌ها", "تولید بینش", "تحلیل کامل شد"]
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()
        for i, stage in enumerate(stages, start=1):
            status.caption(stage)
            progress.progress(int(i / len(stages) * 100))
            time.sleep(0.018)
        status.success("تحلیل کامل شد")

    st.sidebar.caption("V0.7 — Management OS؛ مدیریت + داده + اتوماسیون با Baseline و Demo.")
    if st.session_state.presentation_mode:
        page = "Executive Overview"
    return scenario, page


def _safe_mom_growth(monthly: pd.DataFrame) -> float:
    try:
        if monthly is None or monthly.empty or len(monthly) < 2:
            return 0.0
        ordered = monthly.sort_values("month").copy()
        if "mom_growth" in ordered.columns:
            value = ordered["mom_growth"].iloc[-1]
            if pd.notna(value) and np.isfinite(float(value)):
                return float(value)
        prev = float(ordered["revenue"].iloc[-2])
        cur = float(ordered["revenue"].iloc[-1])
        return (cur - prev) / prev if prev else 0.0
    except Exception:
        return 0.0


def _safe_anomaly_count(df: pd.DataFrame) -> int:
    try:
        if df is None or df.empty or "is_anomaly" not in df.columns:
            return 0
        return int(df["is_anomaly"].fillna(False).astype(bool).sum())
    except Exception:
        return 0


def _safe_high_risk_share(risk_stats: dict) -> float:
    try:
        value = float((risk_stats or {}).get("high_or_very_high_share", 0.0))
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def command_metric(label: str, value: str, note: str, trend_text: str = "", trend_kind: str = "flat", source: str = "derived"):
    trend_class = {"up": "trend-up", "down": "trend-down", "flat": "trend-flat"}.get(trend_kind, "trend-flat")
    trend_html = f'<span class="trend-pill {trend_class}">{trend_text}</span>' if trend_text else ""
    source_html = f'<span class="source-tag {SOURCE_CLASSES.get(source, "src-derived")}">{SOURCE_LABELS.get(source, "محاسبه‌شده")}</span>'
    st.markdown(
        f'''<div class="command-metric">
            <div class="command-metric-top"><div class="command-metric-label">{label}</div>{trend_html or source_html}</div>
            <div class="command-metric-value">{value}</div>
            <div class="command-metric-note">{note}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def priority_card(index: int, title: str, text: str, severity: str, source_text: str):
    sev_label = {"high": "اولویت بالا", "med": "نیازمند توجه", "info": "نیازمند داده"}.get(severity, "نیازمند توجه")
    sev_class = {"high": "sev-high", "med": "sev-med", "info": "sev-info"}.get(severity, "sev-med")
    st.markdown(
        f'''<div class="priority-card">
            <div class="priority-number">اولویت {fa_num(index)}</div>
            <div class="priority-title">{title}</div>
            <div class="priority-text">{text}</div>
            <div class="priority-footer"><span class="severity-pill {sev_class}">{sev_label}</span><span class="priority-source">{source_text}</span></div>
        </div>''',
        unsafe_allow_html=True,
    )


def _executive_priorities(scenario, kpis, sales_anomalies, sms_anomalies, risk_stats):
    sm = scenario_summary(scenario)
    reel = reel_snapshot_metrics()
    priorities = []

    phone_share = float(kpis.get("phone_share", 0.0))
    if phone_share >= 0.85:
        priorities.append({
            "weight": 100,
            "severity": "high" if phone_share >= 0.92 else "med",
            "title": "وابستگی فروش به کانال تلفنی بالاست",
            "text": f"حدود {pct(phone_share)} از تعداد فروش مدل از کانال تلفنی می‌آید. رشد فروش آنلاین یا کانال‌های قابل رهگیری می‌تواند ریسک تمرکز کانال را کاهش دهد.",
            "source": "محاسبه‌شده از داده مبنا",
        })

    top3 = float(reel.get("top3_view_share", 0.0))
    if top3 >= 0.65:
        priorities.append({
            "weight": 85,
            "severity": "med",
            "title": "عملکرد محتوا به چند محتوای پربازدید وابسته است",
            "text": f"سه ریلز برتر {pct(top3)} از کل بازدید ۱۰ ریلز ثبت‌شده را ساخته‌اند. برای ارزیابی تیم، میانه بازدید و تبدیل به لید مهم‌تر از میانگین خام است.",
            "source": "نمای واقعی محتوا",
        })

    if sm["lead_backlog"] >= 3_000:
        priorities.append({
            "weight": 80,
            "severity": "info",
            "title": "صف لید بزرگ است؛ اما سرعت پردازش را هنوز نمی‌دانیم",
            "text": f"در حال حاضر {fa_num(sm['lead_backlog'])} لید در داده مبنا ثبت شده است. بدون تعداد تماس / پاسخ / واجد شرایط نمی‌توان زمان واقعی تخلیه صف یا گلوگاه تیم فروش را محاسبه کرد.",
            "source": "داده مبنای واقعی + شکاف داده",
        })

    anomaly_count = _safe_anomaly_count(sales_anomalies) + _safe_anomaly_count(sms_anomalies)
    if anomaly_count > 0:
        priorities.append({
            "weight": 55,
            "severity": "med",
            "title": "مدل آزمایشی چند تغییر غیرعادی دیده است",
            "text": f"در داده مصنوعی فعلی {fa_num(anomaly_count)} نقطه غیرعادی در فروش/پیامک علامت خورده است. این هشدار واقعی شرکت نیست و فقط سازوکار Alert آینده را نمایش می‌دهد.",
            "source": "مصنوعی / دمو",
        })

    high_risk = _safe_high_risk_share(risk_stats)
    if high_risk >= 0.20:
        priorities.append({
            "weight": 40,
            "severity": "med",
            "title": "مدل ریزش نیاز به داده رفتار واقعی دارد",
            "text": f"مدل آزمایشی {pct(high_risk)} مشتری مصنوعی را پرریسک تشخیص داده است. قبل از استفاده عملیاتی باید فعالیت و تمدید واقعی وارد شود.",
            "source": "مدل مصنوعی",
        })

    priorities.sort(key=lambda x: x["weight"], reverse=True)
    if len(priorities) < 3:
        priorities.append({"weight": 0, "severity": "info", "title": "اتصال داده واقعی اولویت بعدی است", "text": "برای تبدیل این Command Center از نمونه اولیه به ابزار تصمیم‌گیری، اتصال Read-only به داده فروش و مرکز تماس بیشترین ارزش را دارد.", "source": "پیشنهاد معماری"})
    return priorities[:3]


def executive_overview(scenario, data, kpis, monthly, funnel, forecast, insights, customers_model=None, risk_stats=None, sales_anomalies=None, sms_anomalies=None):
    sm = scenario_summary(scenario)
    reel = reel_snapshot_metrics()
    mom = _safe_mom_growth(monthly)
    sales_anomaly_count = _safe_anomaly_count(sales_anomalies)
    sms_anomaly_count = _safe_anomaly_count(sms_anomalies)
    anomaly_count = sales_anomaly_count + sms_anomaly_count
    high_risk_share = _safe_high_risk_share(risk_stats or {})
    priorities = _executive_priorities(scenario, kpis, sales_anomalies, sms_anomalies, risk_stats or {})

    # Status is rule-based and intentionally excludes synthetic ML/anomaly output from real-business alarm logic.
    real_attention = 0
    if float(kpis.get("phone_share", 0.0)) >= 0.85:
        real_attention += 1
    if float(reel.get("top3_view_share", 0.0)) >= 0.65:
        real_attention += 1
    if sm["lead_backlog"] >= 3_000:
        real_attention += 1
    if real_attention >= 3:
        status_label, status_class = "نیازمند توجه مدیریتی", "ceo-watch"
        status_copy = "سه موضوع در داده مبنای فعلی برجسته است: تمرکز فروش روی تلفن، تمرکز عملکرد محتوا روی چند محتوای پربازدید و نبود داده عملیاتی کافی برای تفسیر صف لید."
    elif real_attention >= 1:
        status_label, status_class = "پایدار با چند نقطه قابل پیگیری", "ceo-watch"
        status_copy = "وضعیت کلی قابل کنترل است، اما چند سیگنال نیازمند بررسی مدیریتی یا تکمیل داده هستند."
    else:
        status_label, status_class = "پایدار در داده موجود", "ceo-good"
        status_copy = "در داده مبنای فعلی سیگنال پررنگی دیده نمی‌شود؛ برای نتیجه قطعی هنوز اتصال به داده واقعی لازم است."

    page_header(
        "مرکز فرمان مدیرعامل",
        "یک صفحه تصمیم‌محور برای پاسخ به چهار سؤال: الان وضعیت چیست؟ چه چیزی تغییر کرده؟ پول کجا ساخته می‌شود؟ و مدیر باید امروز به چه چیزی توجه کند؟",
    )

    mom_arrow = "↑" if mom > 0.002 else "↓" if mom < -0.002 else "—"
    anomaly_label = f"{fa_num(anomaly_count)} هشدار دمو" if anomaly_count else "بدون هشدار دمو"

    st.markdown(
        f'''<div class="ceo-command">
            <div>
                <div class="ceo-overline">مرکز فرمان مدیرعامل · Management OS V0.7</div>
                <div class="ceo-status-line"><div class="ceo-status-title">{status_label}</div><span class="ceo-status-badge {status_class}">وضعیت قاعده‌محور</span></div>
                <div class="ceo-status-copy">{status_copy} این وضعیت «امتیاز سلامت شرکت» نیست؛ جمع‌بندی توضیح‌پذیر از داده‌های فعلی نمونه اولیه است.</div>
            </div>
            <div class="ceo-command-grid">
                <div class="ceo-command-cell"><div class="ceo-command-label">درآمد ماهانه مدل</div><div class="ceo-command-value">{toman(kpis['monthly_revenue'])} تومان</div><div class="ceo-command-foot">محاسبه‌شده از فروش و میانگین قیمت</div></div>
                <div class="ceo-command-cell"><div class="ceo-command-label">تغییر ماه‌به‌ماه مدل</div><div class="ceo-command-value">{mom_arrow} {pct(abs(mom))}</div><div class="ceo-command-foot">روند مصنوعی / سناریویی</div></div>
                <div class="ceo-command-cell"><div class="ceo-command-label">هشدارهای مدل</div><div class="ceo-command-value">{anomaly_label}</div><div class="ceo-command-foot">ناهنجاری فروش + پیامک، مصنوعی</div></div>
                <div class="ceo-command-cell"><div class="ceo-command-label">وضعیت اتصال داده</div><div class="ceo-command-value">حالت دمو</div><div class="ceo-command-foot">API / DB / n8n هنوز متصل نیست</div></div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )

    brief_text, brief_conf, brief_source = build_ceo_brief(scenario, kpis)
    conf_label, conf_class = confidence_label(brief_conf)
    st.markdown(
        f'''<div class="v06-brief"><div class="v06-brief-kicker">خلاصه مدیریتی امروز</div><div class="v06-brief-title">تصویر فعلی کسب‌وکار در یک پاراگراف</div><div class="v06-brief-copy">{brief_text}</div><div class="v06-brief-footer"><span class="confidence-pill {conf_class}">{conf_label}</span><span class="trust-pill src-derived">{brief_source}</span></div></div>''',
        unsafe_allow_html=True,
    )
    render_data_trust_layer()

    section_heading("نبض لحظه‌ای", "شش عددی که باید اول دیده شوند", "اعداد واقعی، محاسبه‌شده و تخمینی عمداً از هم تفکیک شده‌اند.")
    row = st.columns(6)
    with row[0]:
        command_metric("فروش ماهانه", f"{fa_num(kpis['monthly_units'])} دستگاه", "تلفنی + آنلاین", source="derived")
    with row[1]:
        command_metric("صف لید", fa_num(sm["lead_backlog"]), "موجودی لید؛ نه نرخ تبدیل", source="real")
    with row[2]:
        command_metric("فروش تلفنی / روز", fa_num(sm["phone_daily"]), f"سهم ماهانه: {pct(kpis['phone_share'])}", source="real")
    with row[3]:
        command_metric("فروش آنلاین / ماه", fa_num(sm["online_monthly"]), f"سهم ماهانه: {pct(kpis['online_share'])}", source="real")
    with row[4]:
        command_metric("میانه بازدید ریلز", fa_num(reel["median_views"]), "نمای ثبت‌شده از ۱۰ ریلز", source="real")
    with row[5]:
        command_metric("فروش منتسب به محتوا", f"{fa_num(sm['content_sales'], 1)} / روز", "تخمینی؛ دوباره‌شماری نشود", source="estimated")

    section_heading("صف تصمیم", "سه موضوعی که امروز باید روی میز مدیرعامل باشد", "اولویت‌ها با قواعد قابل توضیح ساخته می‌شوند؛ خروجی ML مصنوعی جای داده واقعی را نمی‌گیرد.")
    pcols = st.columns(3)
    for idx, item in enumerate(priorities, start=1):
        with pcols[idx - 1]:
            priority_card(idx, item["title"], item["text"], item["severity"], item["source"])

    # Money map — all values derived from the current scenario, never hard-coded.
    try:
        plans = plan_performance(scenario).copy()
    except Exception:
        plans = pd.DataFrame({"plan": ["Plan A", "Plan B"], "units": [kpis["monthly_units"] * sm["share_a"], kpis["monthly_units"] * sm["share_b"]], "revenue": [kpis["monthly_units"] * sm["share_a"] * sm["price_a"], kpis["monthly_units"] * sm["share_b"] * sm["price_b"]]})
    plans["طرح"] = plans["plan"].map(PLAN_FA).fillna(plans["plan"])
    total_plan_revenue = float(plans["revenue"].sum()) if not plans.empty else float(kpis["monthly_revenue"])
    plan_b_revenue = float(plans.loc[plans["plan"] == "Plan B", "revenue"].sum()) if "plan" in plans else 0.0
    plan_b_rev_share = plan_b_revenue / total_plan_revenue if total_plan_revenue else 0.0
    phone_revenue = float(kpis["monthly_phone_units"]) * sm["asp"]
    online_revenue = float(kpis["monthly_online_units"]) * sm["asp"]

    section_heading("نقشه پول", "پول کجا ساخته می‌شود؟", "درآمدهای این بخش محاسبه‌شده‌اند؛ فرض می‌شود ترکیب طرح‌ها در کانال تلفنی و آنلاین یکسان است تا وقتی داده واقعی کانال × طرح وارد شود.")
    money_left, money_right = st.columns([1.05, 1])
    with money_left:
        st.markdown(
            f'''<div class="money-summary">
                <div class="money-chip"><div class="money-chip-label">سهم درآمد طرح B</div><div class="money-chip-value">{pct(plan_b_rev_share)}</div></div>
                <div class="money-chip"><div class="money-chip-label">درآمد تلفنی مدل</div><div class="money-chip-value">{toman(phone_revenue)}</div></div>
                <div class="money-chip"><div class="money-chip-label">درآمد آنلاین مدل</div><div class="money-chip-value">{toman(online_revenue)}</div></div>
            </div>''',
            unsafe_allow_html=True,
        )
        fig = px.bar(
            plans,
            x="revenue",
            y="طرح",
            orientation="h",
            text=plans["revenue"].map(lambda x: toman(float(x))),
            labels={"revenue": "درآمد", "طرح": "طرح"},
            color="revenue",
            color_continuous_scale=["#28465F", "#ADCBFF"],
        )
        fig.update_traces(textposition="inside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)
    with money_right:
        channel_df = pd.DataFrame({
            "کانال": ["فروش تلفنی", "فروش آنلاین"],
            "تعداد": [float(kpis["monthly_phone_units"]), float(kpis["monthly_online_units"])],
            "درآمد مدل": [phone_revenue, online_revenue],
        })
        fig = px.pie(channel_df, names="کانال", values="درآمد مدل", hole=.70, color_discrete_sequence=[ACCENT, "#4D82B8"])
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(annotations=[dict(text="ترکیب درآمد", x=.5, y=.5, font_size=14, showarrow=False, font_color="#EAF3FF")])
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)
        st.markdown(
            f'''<div class="glass-panel" style="margin-top:-4px"><div class="section-kicker">برداشت مدیریتی</div><div style="font-weight:780;color:#F8FBFF;margin-top:5px">طرح B با {pct(sm['share_b'])} از تعداد فروش، حدود {pct(plan_b_rev_share)} از درآمد مدل را می‌سازد.</div><div class="kpi-note">این نتیجه از اختلاف قیمت طرح‌ها به دست می‌آید و با تغییر Mix در Sidebar زنده تغییر می‌کند.</div></div>''',
            unsafe_allow_html=True,
        )

    section_heading("اهرم رشد", "اگر فقط یک متغیر را تکان بدهیم چه می‌شود؟", "تحلیل حساسیت ساده و قابل توضیح؛ پیش‌بینی قطعی نیست.")
    plus_one_revenue = sm["sales_days"] * sm["asp"]
    plus_five_revenue = 5 * plus_one_revenue
    extra_online_10 = 10 * sm["asp"]
    shift = min(0.10, max(0.0, sm["share_a"]))
    mix_delta_asp = max(0.0, sm["price_b"] - sm["price_a"]) * shift
    mix_delta_revenue = kpis["monthly_units"] * mix_delta_asp
    levers = st.columns(4)
    with levers[0]:
        st.markdown(f'''<div class="leverage-card"><div class="leverage-kicker">اهرم ۱</div><div class="leverage-title">+۱ فروش تلفنی در روز</div><div class="leverage-value">+{toman(plus_one_revenue)}</div><div class="leverage-note">اثر ماهانه مدل با فرض {fa_num(sm['sales_days'])} روز و ASP ثابت.</div></div>''', unsafe_allow_html=True)
    with levers[1]:
        st.markdown(f'''<div class="leverage-card"><div class="leverage-kicker">اهرم ۲</div><div class="leverage-title">+۵ فروش تلفنی در روز</div><div class="leverage-value">+{toman(plus_five_revenue)}</div><div class="leverage-note">تحلیل سناریو، نه پیش‌بینی عملیاتی.</div></div>''', unsafe_allow_html=True)
    with levers[2]:
        st.markdown(f'''<div class="leverage-card"><div class="leverage-kicker">اهرم ۳</div><div class="leverage-title">+۱۰ فروش آنلاین در ماه</div><div class="leverage-value">+{toman(extra_online_10)}</div><div class="leverage-note">با ترکیب طرح‌ها و متوسط قیمت فعلی.</div></div>''', unsafe_allow_html=True)
    with levers[3]:
        st.markdown(f'''<div class="leverage-card"><div class="leverage-kicker">اهرم ۴</div><div class="leverage-title">۱۰٪ جابه‌جایی سهم از A به B</div><div class="leverage-value">+{toman(mix_delta_revenue)}</div><div class="leverage-note">با فرض ثابت ماندن تعداد فروش؛ فقط اثر ترکیب طرح‌ها.</div></div>''', unsafe_allow_html=True)

    gap_left, gap_right = st.columns([1.1, .9])
    with gap_left:
        section_heading("شکاف داده", "چه چیزی هنوز برای تصمیم واقعی کم داریم؟", "این چهار اتصال بیشترین فاصله بین نمونه اولیه و Intelligence واقعی را کم می‌کنند.")
        gaps = [
            ("داده خام فروش روزانه", "تاریخ، کانال، طرح، مبلغ و وضعیت سفارش برای درآمد و تغییر ماه‌به‌ماه واقعی."),
            ("داده مرکز تماس", "تعداد تماس، پاسخ، واجد شرایط، پیگیری و فروش برای فهم واقعی صف لید و ظرفیت تیم."),
            ("انتساب محتوا به فروش", "UTM / CTA / منبع لید / CRM برای تشخیص اینکه کدام محتوا واقعاً فروش می‌سازد."),
            ("استفاده نیک‌پوز و پیامک", "دستگاه فعال، ثبت شماره، ارسال/تحویل پیامک و تمدید برای Retention و Churn واقعی."),
        ]
        for idx, (title, detail) in enumerate(gaps, start=1):
            st.markdown(f'''<div class="gap-row"><div class="gap-index">{fa_num(idx)}</div><div><div class="gap-title">{title}</div><div class="gap-text">{detail}</div></div><div class="gap-status">در انتظار اتصال</div></div>''', unsafe_allow_html=True)
    with gap_right:
        section_heading("رادار آزمایشی", "هشدارها و ریسک مدل", "این ستون کاملاً Demo/Synthetic است و نباید به‌عنوان وضعیت واقعی شرکت ارائه شود.")
        st.markdown(
            f'''<div class="glass-panel">
                <div class="money-summary">
                    <div class="money-chip"><div class="money-chip-label">ناهنجاری فروش</div><div class="money-chip-value">{fa_num(sales_anomaly_count)}</div></div>
                    <div class="money-chip"><div class="money-chip-label">ناهنجاری پیامک</div><div class="money-chip-value">{fa_num(sms_anomaly_count)}</div></div>
                    <div class="money-chip"><div class="money-chip-label">مشتری پرریسک</div><div class="money-chip-value">{pct(high_risk_share)}</div></div>
                </div>
                <div class="kpi-note">هدف این بخش فقط نشان دادن معماری هشدار و امتیازدهی ریسک آینده است. با اتصال داده واقعی، همین جایگاه از Synthetic به Real تبدیل می‌شود.</div>
            </div>''',
            unsafe_allow_html=True,
        )

    if MANAGEMENT_ENGINE_AVAILABLE:
        try:
            _ensure_management_state()
            _org_overrides = _management_overrides()
            _org_summary = department_summary(scenario, kpis, _org_overrides)
            _org_score = organization_score(scenario, kpis, _org_overrides)
            section_heading("نبض سازمان", "شش واحد در یک نگاه", "امتیاز واحدها تا اتصال داده واقعی ترکیبی از Baseline و ورودی Demo است؛ برای تصمیم قطعی استفاده نشود.")
            _render_department_cards(_org_summary, 6)
            st.caption(f"امتیاز نمونه اولیه کل سازمان: {fa_num(_org_score['score'],1)} از ۱۰۰ · {_org_score['departments_needing_attention']} واحد نیازمند پیگیری")
        except Exception as _org_render_error:
            st.info("لایه نبض سازمان در این Deploy موقتاً در دسترس نیست؛ صفحات قبلی بدون اختلال ادامه می‌دهند.")

    with st.expander("جزئیات تحلیلی — روند، قیف و پیش‌بینی آزمایشی", expanded=False):
        st.caption("این خروجی‌ها عمداً از نمای ۵ ثانیه‌ای خارج شده‌اند تا صفحه اول مدیرعامل شلوغ و فنی نشود.")
        a, b = st.columns(2)
        with a:
            fig = px.area(monthly, x="month", y="revenue", markers=True, labels={"month": "ماه", "revenue": "درآمد"})
            fig.update_traces(line_color=ACCENT, fillcolor="rgba(173,203,255,.10)")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        with b:
            fig = go.Figure(go.Funnel(y=funnel["stage"].map(FUNNEL_FA), x=funnel["count"], textinfo="value+percent initial", marker={"color": ["#ADCBFF", "#93B9DE", "#769FC8", "#5D86AF", "#496E97", "#385775"]}))
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        forecast_show = forecast.copy()
        forecast_show["series"] = forecast_show["series"].map(FORECAST_SERIES_FA).fillna(forecast_show["series"])
        fig = px.line(forecast_show, x="month", y="revenue", color="series", markers=True, labels={"month": "ماه", "revenue": "درآمد", "series": "نوع داده"}, color_discrete_sequence=[ACCENT, "#8CA7D8"])
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    source_legend()
def data_center_page(data: Dict[str, pd.DataFrame]):
    page_header(
        "مرکز داده",
        "لایه ورود داده برای فایل CSV امروز و API، پایگاه داده و n8n در نسخه بعدی. هیچ داده‌ای در V0.6 به سیستم داخلی نیک متصل نیست.",
    )
    source_legend()

    section_heading("اتصال‌ها", "آمادگی اتصال", "این کارت‌ها وضعیت معماری آینده را نشان می‌دهند؛ هنوز اتصال واقعی فعال نیست.")
    c1, c2, c3, c4 = st.columns(4)
    for col, title, status, detail in [
        (c1, "API / CRM", "آماده طراحی", "قرارداد فقط‌خواندنی"),
        (c2, "n8n", "جایگاه آماده", "گردش‌کار اتوماسیون"),
        (c3, "پایگاه داده", "غیرفعال", "اتصال فقط‌خواندنی آینده"),
        (c4, "ورود CSV", "فعال", "ورودی نمونه اولیه"),
    ]:
        with col:
            st.markdown(
                f'<div class="glass-panel"><div class="section-kicker">{status}</div><div style="font-size:1.05rem;font-weight:750">{title}</div><div class="kpi-note">{detail}</div></div>',
                unsafe_allow_html=True,
            )

    section_heading("ورود فایل CSV", "ورود فایل", "اگر ساختار فایل درست باشد، همان مجموعه داده برای پیش‌نمایش استفاده می‌شود.")
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

    section_heading("کیفیت داده", "کیفیت داده", "امتیازها روی مجموعه‌داده‌های فعلی محاسبه می‌شوند.")
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

    section_heading("پیش‌نمایش", "مرور داده", "برای کنترل سریع ساختار مجموعه داده.")
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
            "content_snapshot": "نمای داده محتوا",
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
    sm = scenario_summary(scenario)
    page_header(
        "تحلیل فروش",
        "تفکیک داده مبنای واقعی از محاسبات ۳۰روزه؛ هیچ درآمد محاسبه‌شده‌ای به‌عنوان فروش حسابداری واقعی معرفی نمی‌شود.",
    )

    row = st.columns(4)
    with row[0]:
        kpi_card("فروش تلفنی / روز", fa_num(sm["phone_daily"]), "real", "phone", "مبنای واقعی")
    with row[1]:
        kpi_card("فروش تلفنی / ماه", fa_num(kpis["monthly_phone_units"]), "derived", "sales", f"{fa_num(sm["sales_days"])} روز × فروش روزانه")
    with row[2]:
        kpi_card("فروش آنلاین / ماه", fa_num(sm["online_monthly"]), "real", "online", "مبنای واقعی")
    with row[3]:
        kpi_card("درآمد مدل", f"{toman(kpis['monthly_revenue'])} تومان", "derived", "revenue", "تعداد × میانگین قیمت فروش")

    st.write("")
    st.info(
        f"فرض فعلی مدل: {fa_num(sm["sales_days"])} روز فروش در ماه. "
        "درآمد و تعداد فروش ماهانه محاسبه‌شده‌اند و باید با داده خام فروش واقعی جایگزین شوند."
    )

    left, right = st.columns(2)
    with left:
        section_heading("روند آزمایشی", "فروش روزانه دمو", "۹۰ روز اخیر؛ تولیدشده بر اساس سناریوی فعلی.")
        fig = px.line(daily.tail(90), x="date", y="units", labels={"date": "تاریخ", "units": "تعداد"})
        fig.update_traces(line_color=ACCENT)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        section_heading("روند آزمایشی", "درآمد ماهانه دمو", "برای نمایش رفتار پویا و پیش‌بینی.")
        fig = px.area(monthly, x="month", y="revenue", labels={"month": "ماه", "revenue": "درآمد"})
        fig.update_traces(line_color=ACCENT, fillcolor="rgba(173,203,255,.08)")
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

    section_heading("اقتصاد طرح‌ها", "اقتصاد پلن فعلی", "ترکیب طرح‌ها و قیمت‌ها از نوار کناری قابل تغییرند.")
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
        "این بخش کاملاً آزمایشی است: بخش‌بندی مشتریان و ریسک ریزش روی دیتای مصنوعی اجرا می‌شوند تا معماری آینده را نمایش دهند.",
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

    section_heading("پروفایل رفتاری مشتری", "پروفایل سگمنت‌ها")
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

    section_heading("صف مشتریان پرریسک", "مشتریان نیازمند توجه")
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
    st.warning("نسخه عملیاتی نهایی نیست: رگرسیون لجستیک روی هدف مصنوعی آموزش دیده است.")


def nikpos_page(scenario, data):
    sm = scenario_summary(scenario)
    page_header(
        "تحلیل نیک‌پوز",
        "تفکیک اقتصاد فروش واقعی و محاسبه‌شده از استفاده آزمایشی دستگاه؛ نیک‌پوز لایه جمع‌آوری داده و نیک اس‌ام‌اس لایه فعال‌سازی و بازگشت مشتری است.",
    )
    plans = plan_performance(scenario)
    devices = data["nikpos"]

    row = st.columns(4)
    with row[0]:
        kpi_card("طرح A", f"{toman(sm["price_a"])} تومان", "real", "price", "مبنای فعلی")
    with row[1]:
        kpi_card("طرح B", f"{toman(sm["price_b"])} تومان", "real", "price", "مبنای فعلی")
    with row[2]:
        kpi_card("میانگین قیمت فروش", f"{toman(sm["asp"])} تومان", "derived", "revenue", "بر اساس ترکیب فعلی")
    with row[3]:
        kpi_card("نرخ دستگاه فعال", pct(devices["active_device"].mean()), "synthetic", "sales", "داده استفاده دستگاه آزمایشی")

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

    section_heading("زمینه قیمت و کمپین", "تاریخچه قیمت و کمپین", "زمینه تاریخی؛ برای تحلیل واقعی باید تاریخ شروع و پایان به دیتاست فروش اضافه شود.")
    history = PRICE_HISTORY.copy()
    history["list_price"] = history["list_price"].map(lambda x: f"{toman(x)} تومان")
    history["campaign_price"] = history["campaign_price"].map(lambda x: f"{toman(x)} تومان")
    history = history.rename(columns={"period": "دوره / سناریو", "list_price": "قیمت مرجع", "campaign_price": "قیمت اجرا", "type": "نوع"})
    st.dataframe(history, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="glass-panel">
            <div class="section-kicker">مدل داده محصول</div>
            <div style="font-weight:750;margin-bottom:6px">نیک‌پوز سخت‌افزار جمع‌آوری دیتاست؛ نیک اس‌ام‌اس لایه فعال‌سازی دیتاست.</div>
            <div class="insight-text">در نسخه واقعی می‌توان رویدادهایی مثل ثبت شماره، دسته‌بندی VIP و معرفی مشتری را مستقیماً از دستگاه وارد موتور تحلیل کرد.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def content_page(scenario):
    page_header(
        "تحلیل محتوا و اینستاگرام",
        "نمای واقعی ۱۰ ریلز از مدل آزمایشی جدا شده؛ هدف این صفحه سنجش توزیع عملکرد است، نه تکیه صرف بر میانگین بازدید.",
    )
    m = content_metrics_safe(scenario)

    row = st.columns(5)
    with row[0]:
        kpi_card("فالوئر", fa_num(m["instagram_followers"]), "real", "followers", "نمای ثبت‌شده")
    with row[1]:
        kpi_card("استوری / روز", fa_num(m["stories_per_day"]), "real", "content", "مبنای واقعی")
    with row[2]:
        kpi_card("ریلز / روز", fa_num(m["reels_per_day"]), "real", "content", "مبنای واقعی")
    with row[3]:
        kpi_card("کل محتوا / روز", fa_num(m["total_content_per_day"]), "derived", "content", "استوری + ریلز")
    with row[4]:
        kpi_card("تیم محتوا", fa_num(m["content_team_size"]), "real", "followers", "۵ نفر")

    st.write("")
    section_heading("نمای ۱۰ ریلز", "عملکرد ۱۰ ریلز اخیر", "این داده یک نمای واقعی ثبت‌شده است؛ زمان تماشا، ریچ و ذخیره هنوز در دسترس نیست.")

    s1 = st.columns(4)
    with s1[0]:
        kpi_card("مجموع بازدید", fa_num(m["total_views"]), "real", "content", "۱۰ ریلز")
    with s1[1]:
        kpi_card("میانگین بازدید", fa_num(m["average_views"]), "derived", "content", "به‌شدت تحت تأثیر ریلزهای پربازدید")
    with s1[2]:
        kpi_card("میانه بازدید", fa_num(m["median_views"]), "derived", "content", "نماینده بهتر محتوای معمول")
    with s1[3]:
        kpi_card("سهم ۳ ریلز برتر", pct(m["top3_view_share"]), "derived", "content", "سهم ۳ ریلز برتر از کل بازدید")

    st.write("")
    s2 = st.columns(4)
    with s2[0]:
        kpi_card("کامنت", fa_num(m["total_comments"]), "real", "content", "مجموع ۱۰ ریلز")
    with s2[1]:
        kpi_card("اشتراک‌گذاری", fa_num(m["total_shares"]), "real", "content", "مجموع ۱۰ ریلز")
    with s2[2]:
        kpi_card("کامنت + اشتراک‌گذاری", fa_num(m["total_interactions"]), "derived", "content", "شاخص تقریبی تعامل")
    with s2[3]:
        kpi_card("تعامل / بازدید", pct(m["interaction_rate"]), "derived", "content", "بدون ذخیره و پسند")

    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        reel_plot = REEL_SNAPSHOT.copy()
        fig = px.bar(
            reel_plot,
            x="reel",
            y="views",
            labels={"reel": "محتوا", "views": "بازدید"},
            color="views",
            color_continuous_scale=["#123750", "#ADCBFF"],
        )
        fig.add_hline(
            y=m["median_views"],
            line_dash="dash",
            line_color="#A8B7C7",
            annotation_text=f"میانه: {fa_num(m['median_views'])}",
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
            labels={"views": "بازدید", "interactions": "کامنت + اشتراک‌گذاری"},
            color="interaction_rate",
            color_continuous_scale=["#44556A", "#ADCBFF"],
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-type">هوشمندی محتوا</div>
            <div class="insight-title">میانگین به‌تنهایی تصویر دقیقی نمی‌دهد</div>
            <div class="insight-text">
                میانگین بازدید برابر {fa_num(m["average_views"])} ولی میانه فقط {fa_num(m["median_views"])} است.
                سه ریلز برتر {pct(m["top3_view_share"])} کل بازدیدها را ساخته‌اند؛ بنابراین توزیع عملکرد فعلی به چند محتوای پربازدید وابسته است.
                برای کارت امتیاز تیم، میانه بازدید، نرخ اشتراک‌گذاری، نرخ کامنت، زمان تماشا، ذخیره، لید و فروش باید کنار هم دیده شوند.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("انتساب فروش", "فروش منتسب به محتوا", "این عدد تخمینی است و نباید دوباره به فروش کل اضافه شود.")
    a1, a2, a3 = st.columns(3)
    a1.metric("فروش منتسب / روز", fa_num(m["estimated_sales_per_day"], 1))
    a2.metric("فروش منتسب / ماه", fa_num(m["estimated_sales_per_month"]))
    a3.metric("خروجی محتوا / ماه", fa_num(m["total_content_per_month"]))
    st.caption("برای انتساب واقعی فروش: رهگیری منبع، شناسه کمپین، CTA، کلیک لینک، اتصال CRM و بازه انتساب لازم است.")

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
            "interaction_rate": "تعامل/بازدید",
            "share_rate": "اشتراک/بازدید",
            "comment_rate": "کامنت/بازدید",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def media_intelligence_page():
    page_header(
        "آزمایشگاه تحلیل محتوا",
        "۵ ویدیوی واقعی و ۶ تصویر ارسالی داخل اپ قرار گرفته‌اند. خط زمانی فعلی آزمایشی است؛ بعداً ماندگاری مخاطب و رویدادهای واقعی با داده رهگیری جایگزین می‌شوند.",
    )

    video_tab, image_tab, upload_tab = st.tabs(["ویدیو و خط زمانی", "گالری تصاویر", "آپلود تست"])

    with video_tab:
        section_heading("هوشمندی ویدیو", "خط زمانی تحلیلی ویدیو", "ماندگاری مخاطب و رویدادهای کلیک/بازپخش فعلاً شبیه‌سازی شده‌اند و داده واقعی اینستاگرام نیستند.")
        options = {f"{v.media_id} — {v.title}": v.media_id for v in MEDIA_VIDEOS}
        selected_label = st.selectbox("ویدیوی مورد بررسی", list(options.keys()))
        video = get_video(options[selected_label])
        timeline = synthetic_video_timeline(video)
        events = synthetic_video_events(video)
        metrics = synthetic_video_metrics(video)

        selected_second = st.slider("ثانیه مورد بررسی روی خط زمانی", 0, int(round(video.duration)), min(3, int(round(video.duration))), 1)
        sec_row = timeline.iloc[(timeline["second"] - selected_second).abs().idxmin()]

        left, right = st.columns([0.82, 1.58])
        with left:
            st.markdown('<div class="demo-badge">خط زمانی دمو / آزمایشی</div>', unsafe_allow_html=True)
            st.video(str(video.path), start_time=int(selected_second))
            st.markdown(f'''<div class="glass-panel"><div style="font-weight:800">{video.title}</div><div class="media-meta">{video.format}<br>{video.topic}<br>مدت: {fa_digits(f"{video.duration:.1f}")} ثانیه</div></div>''', unsafe_allow_html=True)
        with right:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("حفظ مخاطب در شروع / دمو", pct(metrics["hook_hold"]))
            k2.metric("میانگین تماشا / دمو", f"{fa_num(metrics['avg_watch_time'], 1)} ثانیه")
            k3.metric("نرخ تکمیل / دمو", pct(metrics["completion_rate"]))
            k4.metric("اوج تعامل", f"ثانیه {fa_num(metrics['interaction_peak_second'])}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timeline["second"], y=timeline["retention"], mode="lines", name="ماندگاری مخاطب / دمو", line=dict(color=ACCENT, width=3), fill="tozeroy", fillcolor="rgba(173,203,255,.10)"))
            fig.add_trace(go.Scatter(x=timeline["second"], y=timeline["click_signal"], mode="lines", name="سیگنال کلیک / دمو", line=dict(color="#F3D39A", width=2, dash="dot"), yaxis="y2"))
            for _, ev in events.iterrows():
                fig.add_vline(x=float(ev["second"]), line_width=1, line_dash="dot", line_color="rgba(255,255,255,.30)")
                fig.add_annotation(x=float(ev["second"]), y=0.98, yref="paper", text=str(ev["event"]), showarrow=False, font=dict(size=9, color="#E9F2FF"), textangle=-90)
            fig.add_vline(x=selected_second, line_width=2, line_color="#FFFFFF")
            fig.update_layout(yaxis=dict(title="ماندگاری مخاطب", tickformat=".0%", range=[0,1.05]), yaxis2=dict(title="سیگنال کلیک", overlaying="y", side="right", showgrid=False, range=[0,1.15]))
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)

            r1, r2, r3 = st.columns(3)
            r1.metric("ماندگاری مخاطب در این ثانیه", pct(float(sec_row["retention"])))
            r2.metric("سیگنال تعامل", fa_num(float(sec_row["interaction_signal"]), 2))
            r3.metric("سیگنال کلیک", fa_num(float(sec_row["click_signal"]), 2))

        section_heading("نقاط رویداد", "رویدادهای آزمایشی روی ویدیو", "در آینده می‌توانند از رهگیری لینک / CRM / n8n تغذیه شوند.")
        ev_show = events.copy()
        ev_show["second"] = ev_show["second"].map(lambda x: f"00:{int(x):02d}" if int(x) < 60 else f"{int(x)//60:02d}:{int(x)%60:02d}")
        ev_show = ev_show.rename(columns={"second":"زمان", "event":"رویداد", "description":"شرح", "signal":"نوع سیگنال"})
        st.dataframe(ev_show, use_container_width=True, hide_index=True)
        st.info("جمله «مخاطب در ثانیه ۱۸ کلیک کرد» فقط زمانی داده واقعی محسوب می‌شود که رهگیری رویداد واقعی داشته باشیم. فعلاً سیگنال کلیک و خط زمانی این بخش صریحاً آزمایشی هستند.")

    with image_tab:
        section_heading("گالری تصاویر", "گالری تصویری نمونه", "۶ تصویر واقعی ارسالی برای تست نمای کارتی و اتصال آینده به عملکرد هر محتوای خلاقه.")
        cols = st.columns(3)
        for i, item in enumerate(MEDIA_IMAGES):
            with cols[i % 3]:
                st.image(str(item.path), use_container_width=True)
                st.markdown(f"**{item.title}**")
                st.caption(f"{item.category} · {item.visual_note}")
        st.caption("مرحله بعد: بازدید / ریچ / نرخ کلیک / لید / فروش هر محتوای خلاقه با شناسه محتوا به همین کارت متصل می‌شود تا مشخص شود کدام طراحی فقط زیباست و کدام واقعاً فروش می‌سازد.")

    with upload_tab:
        section_heading("تست سریع", "آپلود موقت مدیا", "برای تست جلسه؛ فایل آپلودشده ذخیره دائمی نمی‌شود و به دیتابیس متصل نیست.")
        uploaded = st.file_uploader("ویدیو یا تصویر", type=["mp4", "mov", "m4v", "jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            name = uploaded.name.lower()
            if name.endswith((".mp4", ".mov", ".m4v")):
                st.video(uploaded)
                st.success("پیش‌نمایش آماده است. برای خط زمانی واقعی باید فراداده و داده رویداد این فایل نیز وارد شود.")
            else:
                st.image(uploaded, use_container_width=True)
                st.success("پیش‌نمایش آماده است. بعداً عملکرد این محتوای خلاقه با شناسه محتوا به دیتای انتشار متصل می‌شود.")

def sms_page(data):
    page_header(
        "تحلیل پیامک",
        "فعلاً مجموعه داده پیامک فعلاً مصنوعی است. ادعاهای بازاریابی مثل نرخ بازشدن ۹۸٪ تا دریافت منبع داخلی به‌عنوان شاخص واقعی وارد نشده‌اند.",
    )
    sms = data["sms"].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("ارسال‌شده / دمو", fa_num(sms["sent"].sum()))
    c2.metric("نرخ تحویل / دمو", pct(sms["delivered"].sum() / max(sms["sent"].sum(), 1)))
    c3.metric("نرخ کلیک / دمو", pct(sms["clicks"].sum() / max(sms["delivered"].sum(), 1)))

    l, r = st.columns(2)
    with l:
        fig = px.line(sms.tail(90), x="date", y="delivery_rate", labels={"date": "تاریخ", "delivery_rate": "نرخ تحویل"})
        fig.update_traces(line_color=ACCENT)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with r:
        fig = px.area(sms.tail(90), x="date", y="sent", labels={"date": "تاریخ", "sent": "تعداد پیامک"})
        fig.update_traces(line_color="#6F8EFF", fillcolor="rgba(111,142,255,.08)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.info("ادعای بازاریابی ≠ شاخص داخلی شرکت. برای شاخص واقعی باید ارسال، تحویل، کلیک و تبدیل از داده داخلی نیک اس‌ام‌اس خوانده شود.")


def anomaly_page(sales_anomalies, sms_anomalies):
    page_header(
        "تشخیص ناهنجاری",
        "میانگین متحرک + امتیاز Z روی داده مصنوعی؛ هدف نمایش سازوکار هشداردهی آینده است.",
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
    with r:
        fig = px.line(sms_anomalies, x="date", y="delivery_rate", labels={"date": "تاریخ", "delivery_rate": "نرخ تحویل"})
        flagged = sms_anomalies[sms_anomalies["is_anomaly"]]
        fig.add_scatter(x=flagged["date"], y=flagged["delivery_rate"], mode="markers", name="ناهنجاری", marker=dict(size=9, color="#FF7A90"))
        st.plotly_chart(style_fig(fig), use_container_width=True)


def predictions_page(forecast, forecast_stats, customers_model, risk_stats):
    page_header(
        "پیش‌بینی‌ها",
        "مدل‌ها ساده و قابل توضیح نگه داشته شده‌اند؛ هیچ خروجی پیش‌بینی یا ریزش در V0.6 در سطح عملیاتی نهایی نیست.",
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

    section_heading("نمونه مدل ریزش", "مشتریان با ریسک بالاتر")
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
    st.warning("پیش‌بینی‌ها و خروجی‌های یادگیری ماشین آزمایشی هستند و نباید مبنای تصمیم عملیاتی قطعی قرار گیرند.")


def insights_page(insights):
    page_header(
        "بینش‌های خودکار",
        "موتور بینش قاعده‌محور؛ متن بینش‌ها از سناریو، نمای داده محتوا و خروجی مدل‌های آزمایشی ساخته می‌شود.",
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
        "معماری امروز محلی و آزمایشی است؛ دکمه‌های API و n8n نشان می‌دهند ورودی واقعی در آینده از کجا وارد همین خط لوله می‌شود.",
    )
    stages = [
        "ورود داده",
        "اعتبارسنجی",
        "پاک‌سازی",
        "محاسبه شاخص‌های کلیدی",
        "تحلیل روند",
        "تشخیص ناهنجاری",
        "بخش‌بندی مشتریان",
        "امتیازدهی ریسک",
        "پیش‌بینی",
        "موتور بینش",
        "به‌روزرسانی داشبورد",
    ]
    for i in range(0, len(stages), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(stages):
                with col:
                    st.markdown(f'<div class="pipeline-step">{fa_num(idx + 1)}<br>{stages[idx]}</div>', unsafe_allow_html=True)
        if i + 4 < len(stages):
            st.markdown("<div style='text-align:center;color:#ADCBFF;font-size:1.35rem;margin:4px 0'>↓</div>", unsafe_allow_html=True)

    st.write("")
    section_heading("ورودی آینده", "معماری اتصال بعدی")
    st.markdown(
        """
        <div class="glass-panel" style="text-align:center;line-height:2.1;font-weight:650">
            پایگاه داده / CRM / پنل نیک
            <span style="color:#ADCBFF"> → </span>
            API فقط‌خواندنی
            <span style="color:#ADCBFF"> → </span>
            n8n / خط لوله داده
            <span style="color:#ADCBFF"> → </span>
            اعتبارسنجی
            <span style="color:#ADCBFF"> → </span>
            موتور تحلیل
            <span style="color:#ADCBFF"> → </span>
            NIK Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )


def revenue_intelligence_page(scenario, kpis):
    sm = scenario_summary(scenario)
    page_header(
        "هوشمندی درآمد",
        "صفحه‌ای متمرکز روی اقتصاد فروش: کدام طرح و کدام کانال درآمد می‌سازد و حساسیت Revenue به تغییر متغیرهای کلیدی چقدر است.",
    )
    plan_a_units = float(kpis["monthly_units"]) * sm["share_a"]
    plan_b_units = float(kpis["monthly_units"]) * sm["share_b"]
    plan_a_revenue = plan_a_units * sm["price_a"]
    plan_b_revenue = plan_b_units * sm["price_b"]
    total = max(plan_a_revenue + plan_b_revenue, 1.0)
    phone_revenue = float(kpis["monthly_phone_units"]) * sm["asp"]
    online_revenue = float(kpis["monthly_online_units"]) * sm["asp"]

    section_heading("اقتصاد درآمد", "چه چیزی پول می‌سازد؟", "تمام اعداد این بخش محاسبه‌شده از سناریوی جاری‌اند، نه داده حسابداری زنده.")
    cols = st.columns(4)
    with cols[0]: command_metric("درآمد ماهانه مدل", f"{toman(kpis['monthly_revenue'])} تومان", "تعداد فروش × میانگین قیمت", source="derived")
    with cols[1]: command_metric("سهم درآمد طرح B", pct(plan_b_revenue / total), "اثر قیمت بالاتر طرح B", source="derived")
    with cols[2]: command_metric("درآمد تلفنی مدل", f"{toman(phone_revenue)} تومان", "با فرض Mix یکسان در کانال‌ها", source="derived")
    with cols[3]: command_metric("درآمد آنلاین مدل", f"{toman(online_revenue)} تومان", "با فرض Mix یکسان در کانال‌ها", source="derived")

    left, right = st.columns(2)
    with left:
        df = pd.DataFrame({"طرح":["طرح A","طرح B"],"درآمد":[plan_a_revenue,plan_b_revenue],"تعداد":[plan_a_units,plan_b_units]})
        fig = px.bar(df, x="طرح", y="درآمد", text=df["درآمد"].map(lambda x:toman(float(x))), labels={"درآمد":"درآمد مدل","طرح":"طرح"}, color="درآمد", color_continuous_scale=["#294964",ACCENT])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with right:
        df = pd.DataFrame({"کانال":["فروش تلفنی","فروش آنلاین"],"درآمد":[phone_revenue,online_revenue]})
        fig = px.pie(df, names="کانال", values="درآمد", hole=.72, color_discrete_sequence=[ACCENT,"#5C8EBC"])
        fig.update_traces(textinfo="percent")
        fig.update_layout(annotations=[dict(text="Revenue Mix",x=.5,y=.5,showarrow=False,font_color="#EAF4FF",font_size=13)])
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    section_heading("حساسیت درآمد", "اثر حرکت یک متغیر", "این تحلیل What-if است؛ پیش‌بینی نتیجه واقعی بازار نیست.")
    effects = [
        ("+۱ فروش تلفنی / روز", sm["sales_days"] * sm["asp"]),
        ("+۵ فروش تلفنی / روز", 5 * sm["sales_days"] * sm["asp"]),
        ("+۱۰ فروش آنلاین / ماه", 10 * sm["asp"]),
        ("+۱۰ واحد درصد سهم B", float(kpis["monthly_units"]) * max(sm["price_b"]-sm["price_a"],0) * min(.10,sm["share_a"])),
    ]
    ec = st.columns(4)
    for idx,(label,value) in enumerate(effects):
        with ec[idx]:
            st.markdown(f'''<div class="leverage-card"><div class="leverage-kicker">سناریو</div><div class="leverage-title">{label}</div><div class="leverage-value">+{toman(value)}</div><div class="leverage-note">اثر ماهانه مدل در صورت ثابت ماندن سایر متغیرها.</div></div>''', unsafe_allow_html=True)
    source_legend()


def scenario_simulator_page(scenario, kpis):
    sm = scenario_summary(scenario)
    page_header(
        "شبیه‌ساز تصمیم",
        "دو سناریو را کنار هم مقایسه کن تا اثر تصمیم روی تعداد فروش، درآمد، وابستگی تلفنی و Mix طرح‌ها بلافاصله دیده شود.",
    )
    section_heading("سناریوی مقایسه", "اگر تصمیم را تغییر بدهیم چه می‌شود؟", "سناریوی سمت راست مستقل از Sidebar است و فقط برای What-if استفاده می‌شود.")
    with st.form("v06_scenario_compare"):
        c1,c2,c3,c4 = st.columns(4)
        with c1: target_phone = st.number_input("فروش تلفنی هدف / روز",0,500,int(sm["phone_daily"] + 5),1)
        with c2: target_online = st.number_input("فروش آنلاین هدف / ماه",0,5000,int(sm["online_monthly"] + 20),5)
        with c3: target_b_share = st.slider("سهم هدف طرح B",0,100,int(round(sm["share_b"]*100)),5)
        with c4: target_days = st.slider("روز فروش / ماه",20,31,int(sm["sales_days"]),1)
        st.form_submit_button("محاسبه سناریوی هدف", use_container_width=True)
    target_share_b = target_b_share/100
    target_asp = sm["price_a"]*(1-target_share_b)+sm["price_b"]*target_share_b
    target_phone_month = target_phone*target_days
    target_units = target_phone_month+target_online
    target_revenue = target_units*target_asp
    current_revenue = float(kpis["monthly_revenue"])
    current_units = float(kpis["monthly_units"])
    revenue_delta = target_revenue-current_revenue
    units_delta = target_units-current_units
    current_phone_share = float(kpis.get("phone_share",0))
    target_phone_share = target_phone_month/target_units if target_units else 0

    a,b = st.columns(2)
    with a:
        st.markdown(f'''<div class="sim-hero"><div class="section-kicker">وضعیت فعلی</div><div class="sim-label">درآمد ماهانه مدل</div><div class="sim-value">{toman(current_revenue)} تومان</div><div class="kpi-note">{fa_num(current_units)} دستگاه · وابستگی تلفنی {pct(current_phone_share)} · ASP {toman(sm['asp'])}</div></div>''', unsafe_allow_html=True)
    with b:
        delta_class = "sim-delta-up" if revenue_delta>0 else "sim-delta-down" if revenue_delta<0 else "sim-delta-flat"
        sign = "+" if revenue_delta>0 else ""
        st.markdown(f'''<div class="sim-hero"><div class="section-kicker">سناریوی هدف</div><div class="sim-label">درآمد ماهانه مدل</div><div class="sim-value">{toman(target_revenue)} تومان</div><div class="kpi-note">{fa_num(target_units)} دستگاه · وابستگی تلفنی {pct(target_phone_share)} · ASP {toman(target_asp)}</div><div class="{delta_class}" style="font-weight:850;margin-top:8px">{sign}{toman(revenue_delta)} تومان نسبت به وضعیت فعلی</div></div>''', unsafe_allow_html=True)

    section_heading("اثر تصمیم", "چه چیزی جابه‌جا شد؟", "مقایسه مستقیم؛ بدون مدل پیچیده و قابل توضیح برای جلسه مدیریتی.")
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("تغییر تعداد فروش", f"{fa_num(units_delta)}", delta=f"{fa_num(units_delta)} دستگاه")
    d2.metric("درآمد هدف", f"{toman(target_revenue)}", delta=f"{toman(revenue_delta)} تومان")
    d3.metric("وابستگی تلفنی هدف", pct(target_phone_share), delta=fa_digits(f"{(target_phone_share-current_phone_share)*100:+.1f} واحد درصد"))
    d4.metric("سهم طرح B هدف", pct(target_share_b), delta=fa_digits(f"{(target_share_b-sm['share_b'])*100:+.1f} واحد درصد"))
    st.caption("این ابزار سناریوسازی است، نه Forecast. نتیجه واقعی به ظرفیت تیم، تقاضا، کیفیت لید، قیمت و رفتار بازار وابسته است.")


def connections_page():
    page_header(
        "مرکز اتصال داده",
        "نقشه راه اتصال NIK Intelligence به منابع واقعی. در V0.6 فقط CSV و داده Demo فعال‌اند؛ دکمه‌های اتصال نمایشی‌اند و هیچ Credential ذخیره نمی‌شود.",
    )
    section_heading("منابع داده", "از Demo تا داده زنده", "هر اتصال در آینده Read-only و با کنترل دسترسی ساخته می‌شود.")
    sources = [
        ("داده مصنوعی / Demo","فعال","برای Prototype و تست مدل‌ها.","همین اجرا",True),
        ("CSV / Upload","آماده","ورود فایل برای فروش، مشتری، لید، پیامک و نیک‌پوز.","با آپلود کاربر",True),
        ("پایگاه داده NIKSMS","متصل نیست","اتصال فقط‌خواندنی به جداول تأییدشده شرکت.","—",False),
        ("NIKPOS Devices","متصل نیست","Activation، Capture، Usage و Eventهای دستگاه.","—",False),
        ("CRM / Call Center","متصل نیست","Lead → Contact → Qualified → Sale.","—",False),
        ("Instagram / Content","متصل نیست","Metrics محتوا و Source Tracking در صورت دسترسی مجاز.","—",False),
        ("n8n","متصل نیست","Orchestration، Webhook و Automation بین منابع.","—",False),
        ("Tracked Links / UTM","متصل نیست","پیوند Content → Click → Lead → Sale.","—",False),
    ]
    cols = st.columns(4)
    for i,(name,state,copy,fresh,on) in enumerate(sources):
        with cols[i%4]:
            state_cls = "state-on" if on else "state-off"
            st.markdown(f'''<div class="connection-card"><div class="connection-top"><div class="connection-name">{name}</div><div class="connection-state {state_cls}">{state}</div></div><div class="connection-copy">{copy}</div><div class="connection-fresh">آخرین داده: {fresh}</div></div>''', unsafe_allow_html=True)
            if not on:
                if st.button(f"اتصال {name}", key=f"connect_{i}", use_container_width=True):
                    st.toast("Placeholder V0.6 — اتصال واقعی بعد از تعریف Schema، دسترسی و امنیت ساخته می‌شود.")

    section_heading("معماری آینده", "مسیر امن اتصال", "این نسخه هیچ API Key یا Credential ندارد.")
    st.code("""NIK Database / CRM / Instagram / NIKPOS
        ↓
Read-only API / Webhook
        ↓
n8n / Validation
        ↓
Data Warehouse / Curated Tables
        ↓
NIK Intelligence
        ↓
CEO Command Center / Alerts / Reports""", language="text")
    st.info("اولویت اتصال پیشنهادی: ۱) فروش روزانه ۲) Call Center ۳) Attribution محتوا ۴) Usage نیک‌پوز و SMS. قبل از این چهار مورد، افزودن ML جدید ارزش تصمیم‌گیری محدودی دارد.")



# ---------- V0.7 Management & Automation OS ----------
V07_MANAGEMENT_CSS = """
<style>
.org-grid-note{color:#7890A7;font-size:.66rem;line-height:1.8;margin-top:5px}
.org-card{position:relative;min-height:164px;padding:17px 18px;border-radius:22px;background:linear-gradient(145deg,rgba(255,255,255,.066),rgba(255,255,255,.018));border:1px solid rgba(255,255,255,.095);box-shadow:inset 0 1px 0 rgba(255,255,255,.07),0 18px 48px rgba(0,0,0,.13);overflow:hidden}
.org-card::after{content:"";position:absolute;left:-44px;bottom:-60px;width:120px;height:120px;border-radius:50%;background:radial-gradient(circle,rgba(173,203,255,.14),transparent 70%)}
.org-top{display:flex;align-items:center;justify-content:space-between;gap:8px}.org-name{color:#F7FBFF;font-size:.92rem;font-weight:830}.org-icon{width:32px;height:32px;border-radius:11px;display:flex;align-items:center;justify-content:center;background:rgba(173,203,255,.08);border:1px solid rgba(173,203,255,.13);color:#DCEAFF;font-size:.86rem}
.org-score{color:#fff;font-size:1.55rem;font-weight:900;letter-spacing:-.04em;margin-top:14px}.org-meta{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:7px}.org-pill{font-size:.61rem;font-weight:840;padding:4px 7px;border-radius:999px;border:1px solid rgba(255,255,255,.08)}.org-ok{color:#A5F2CB;background:rgba(48,205,135,.08);border-color:rgba(48,205,135,.15)}.org-watch{color:#FFE0A3;background:rgba(244,180,79,.08);border-color:rgba(244,180,79,.15)}.org-action{color:#FFB4C0;background:rgba(255,91,118,.075);border-color:rgba(255,91,118,.14)}
.management-hero{padding:20px 22px;border-radius:24px;background:linear-gradient(135deg,rgba(173,203,255,.12),rgba(255,255,255,.022));border:1px solid rgba(173,203,255,.18);box-shadow:inset 0 1px 0 rgba(255,255,255,.10),0 22px 64px rgba(0,0,0,.17);margin:5px 0 16px}.management-hero-kicker{font-size:.67rem;color:#ADCBFF;font-weight:900;letter-spacing:.09em}.management-hero-title{font-size:1.12rem;font-weight:870;color:#FAFCFF;margin-top:6px}.management-hero-copy{font-size:.76rem;line-height:1.95;color:#9BB0C3;margin-top:6px;max-width:1050px}
.auto-rule{padding:14px 15px;border-radius:18px;background:rgba(255,255,255,.032);border:1px solid rgba(255,255,255,.07);margin-bottom:8px}.auto-rule-active{border-color:rgba(244,180,79,.19);background:linear-gradient(135deg,rgba(244,180,79,.055),rgba(255,255,255,.025))}.auto-rule-name{font-size:.82rem;color:#F6FAFF;font-weight:820}.auto-rule-copy{font-size:.65rem;color:#8095A9;line-height:1.7;margin-top:4px}.auto-rule-action{font-size:.66rem;color:#C7DAED;line-height:1.7;margin-top:7px}.auto-state{display:inline-flex;padding:4px 7px;border-radius:999px;font-size:.60rem;font-weight:850;margin-top:7px}.auto-on{color:#FFE0A3;background:rgba(244,180,79,.08);border:1px solid rgba(244,180,79,.15)}.auto-off{color:#93A7BA;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07)}
.campaign-result{padding:19px 20px;border-radius:22px;background:linear-gradient(145deg,rgba(173,203,255,.09),rgba(255,255,255,.019));border:1px solid rgba(173,203,255,.15);box-shadow:inset 0 1px 0 rgba(255,255,255,.08)}.campaign-status{font-size:.68rem;color:#ADCBFF;font-weight:900;letter-spacing:.07em}.campaign-title{font-size:1.15rem;color:#fff;font-weight:880;margin-top:6px}.campaign-copy{font-size:.74rem;color:#95AABE;line-height:1.9;margin-top:6px}.campaign-names{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.campaign-name{padding:6px 10px;border-radius:999px;background:rgba(173,203,255,.07);border:1px solid rgba(173,203,255,.13);font-size:.68rem;color:#DCEAFF;font-weight:780}
.flow-strip{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:14px 15px;border-radius:18px;background:rgba(255,255,255,.027);border:1px solid rgba(255,255,255,.065);margin:8px 0 14px}.flow-node{padding:7px 10px;border-radius:12px;background:rgba(173,203,255,.07);border:1px solid rgba(173,203,255,.11);font-size:.67rem;color:#D9E8FA;font-weight:760}.flow-arrow{color:#69859D;font-size:.72rem}
.role-box{padding:16px 17px;border-radius:20px;background:rgba(255,255,255,.032);border:1px solid rgba(255,255,255,.075);min-height:132px}.role-label{font-size:.65rem;color:#7890A6}.role-value{font-size:1rem;color:#F5FAFF;font-weight:840;margin-top:5px}.role-copy{font-size:.65rem;color:#7F94A8;line-height:1.75;margin-top:6px}
@media(max-width:720px){.flow-arrow{display:none}.flow-strip{align-items:stretch}.flow-node{width:100%}}
</style>
"""
st.markdown(V07_MANAGEMENT_CSS, unsafe_allow_html=True)


def _management_ready() -> bool:
    if MANAGEMENT_ENGINE_AVAILABLE:
        return True
    page_header("لایه مدیریتی در دسترس نیست", "هسته Data Science قبلی همچنان فعال است؛ فقط فایل management_engine.py در Deploy فعلی پیدا نشده است.")
    st.error("برای فعال شدن صفحات مدیریتی، فایل management_engine.py نسخه V0.7 را کنار app.py آپلود کن. این خطا صفحات قبلی را از کار نمی‌اندازد.")
    if MANAGEMENT_ENGINE_ERROR:
        st.caption(f"جزئیات Import: {MANAGEMENT_ENGINE_ERROR}")
    return False


def _ensure_management_state():
    if not MANAGEMENT_ENGINE_AVAILABLE:
        return
    for key, value in MANAGEMENT_DEMO_DEFAULTS.items():
        st.session_state.setdefault(f"mgmt_{key}", value)
    if "mgmt_tasks" not in st.session_state:
        st.session_state.mgmt_tasks = DEFAULT_TASKS.copy()
    st.session_state.setdefault("mgmt_run_finance", "روزانه")
    st.session_state.setdefault("mgmt_run_sales", "هر ۶ ساعت")
    st.session_state.setdefault("mgmt_run_qc", "هر ۲ ساعت")
    st.session_state.setdefault("mgmt_run_marketing", "روزانه")
    st.session_state.setdefault("mgmt_run_hr", "هفتگی")
    st.session_state.setdefault("mgmt_run_development", "روزانه")


def _management_overrides() -> dict:
    if not MANAGEMENT_ENGINE_AVAILABLE:
        return {}
    _ensure_management_state()
    return {key: st.session_state.get(f"mgmt_{key}", value) for key, value in MANAGEMENT_DEMO_DEFAULTS.items()}


def _status_css(status: str) -> str:
    return "org-ok" if status == "عادی" else "org-watch" if status == "نیازمند توجه" else "org-action"


def _format_mgmt_value(value: float, unit: str) -> str:
    value = float(value)
    if unit == "تومان":
        return f"{toman(value)} تومان"
    if unit == "percent":
        return pct(value)
    if unit == "دستگاه":
        return f"{fa_num(value)} دستگاه"
    if unit == "نفر":
        return f"{fa_num(value)} نفر"
    if unit == "عدد":
        return fa_num(value)
    if unit == "شاخص":
        return fa_num(value, 1)
    return fa_num(value, 1)


def _render_department_cards(summary: pd.DataFrame, columns_count: int = 3):
    if summary is None or summary.empty:
        st.info("برای نمایش نبض سازمان داده کافی وجود ندارد.")
        return
    cols = st.columns(columns_count)
    for idx, (_, row) in enumerate(summary.iterrows()):
        with cols[idx % columns_count]:
            css = _status_css(str(row["status"]))
            st.markdown(
                f'''<div class="org-card"><div class="org-top"><div class="org-name">{row['department_name']}</div><div class="org-icon">{row['icon']}</div></div><div class="org-score">{fa_num(row['score'],1)}<span style="font-size:.72rem;color:#8196AA;font-weight:650"> / ۱۰۰</span></div><div class="org-meta"><span class="org-pill {css}">{row['status']}</span><span class="org-pill" style="color:#BACBDB">{fa_num(row['attention_count'])} KPI نیازمند پیگیری</span></div><div class="org-grid-note">امتیاز نمونه اولیه است و تا اتصال داده واقعی واحد، Health Score قطعی محسوب نمی‌شود.</div></div>''',
                unsafe_allow_html=True,
            )


def organization_pulse_page(scenario, kpis):
    if not _management_ready():
        return
    overrides = _management_overrides()
    summary = department_summary(scenario, kpis, overrides)
    org = organization_score(scenario, kpis, overrides)
    rules = automation_checks(scenario, kpis, overrides)
    active_rules = rules[rules["فعال شده"] == True] if not rules.empty else rules

    page_header("نبض سازمان", "یک نمای واحد از حسابداری، فروش، برنامه‌نویسی، منابع انسانی، QC و مارکتینگ؛ برای اینکه مدیر به‌جای شش گزارش جدا، وضعیت کل سازمان را در یک قاب ببیند.")
    st.markdown(
        f'''<div class="management-hero"><div class="management-hero-kicker">ORGANIZATION PULSE · V0.7</div><div class="management-hero-title">سلامت عملیاتی نمونه اولیه: {fa_num(org['score'],1)} از ۱۰۰ · {org['status']}</div><div class="management-hero-copy">این امتیاز ترکیبی از KPIهای واقعی/محاسبه‌شده و تعدادی ورودی Demo است. هدف فعلی، ساخت معماری مدیریت سازمان است؛ با اتصال هر واحد، KPIهای Demo همان بخش با داده واقعی جایگزین می‌شوند.</div></div>''',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("امتیاز سازمان", fa_num(org["score"], 1))
    c2.metric("واحد نیازمند توجه", fa_num(org["departments_needing_attention"]))
    c3.metric("قانون اتوماسیون فعال", fa_num(len(active_rules)))
    c4.metric("واحدهای متصل به پنل", "۶ واحد", help="اتصال فعلی مفهومی/Demo است؛ نه اتصال Database واقعی.")

    section_heading("نبض واحدها", "شش بخش اصلی شرکت")
    _render_department_cards(summary, 3)

    section_heading("جزئیات واحد", "KPI → بررسی → تسک", "یک واحد را انتخاب کن تا شاخص‌هایش را ببینی.")
    dept_names = summary["department_name"].tolist()
    selected_name = st.selectbox("انتخاب واحد", dept_names, key="v07_department_detail")
    selected_key = str(summary.loc[summary["department_name"] == selected_name, "department"].iloc[0])
    detail = department_kpis(scenario, kpis, overrides)
    detail = detail[detail["department"] == selected_key].copy()
    if not detail.empty:
        detail["مقدار"] = detail.apply(lambda r: _format_mgmt_value(r["actual"], r["unit"]), axis=1)
        detail["هدف"] = detail.apply(lambda r: _format_mgmt_value(r["target"], r["unit"]), axis=1)
        show = detail[["metric", "مقدار", "هدف", "status", "source", "note"]].rename(columns={"metric":"KPI", "status":"وضعیت", "source":"منبع", "note":"توضیح"})
        st.dataframe(show, use_container_width=True, hide_index=True)
        fig = px.bar(detail, x="score", y="metric", orientation="h", text=detail["score"].map(lambda x: f"{x:.0f}%"), labels={"score":"تحقق هدف", "metric":"KPI"}, color="score", color_continuous_scale=["#31506B", ACCENT])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)

    section_heading("اقدام خودکار", "قوانینی که همین حالا Trigger شده‌اند", "قواعد Rule-based هستند و تا اتصال n8n فقط پیشنهاد تولید می‌کنند.")
    if active_rules.empty:
        st.success("در قواعد فعلی موردی برای اقدام خودکار فعال نشده است.")
    else:
        st.dataframe(active_rules[["قانون","بخش","شدت","اقدام پیشنهادی","منبع"]], use_container_width=True, hide_index=True)


def task_kpi_page(scenario, kpis):
    if not _management_ready():
        return
    _ensure_management_state()
    overrides = _management_overrides()
    page_header("تسک و KPI", "مرکز تبدیل «بررسی» به «اقدام»: هر مسئله باید یک مسئول، یک تسک، یک KPI و یک وضعیت قابل پیگیری داشته باشد.")
    st.markdown('''<div class="flow-strip"><span class="flow-node">بررسی</span><span class="flow-arrow">←</span><span class="flow-node">مسئله</span><span class="flow-arrow">←</span><span class="flow-node">تسک</span><span class="flow-arrow">←</span><span class="flow-node">KPI</span><span class="flow-arrow">←</span><span class="flow-node">پیگیری</span><span class="flow-arrow">←</span><span class="flow-node">بستن حلقه</span></div>''', unsafe_allow_html=True)

    tasks = st.session_state.mgmt_tasks.copy()
    if not tasks.empty:
        tasks["پیشرفت"] = np.where(tasks["هدف"].astype(float) > 0, np.clip(tasks["عملکرد"].astype(float) / tasks["هدف"].astype(float), 0, 1.5), 0)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("کل تسک", fa_num(len(tasks)))
        c2.metric("اولویت بالا", fa_num((tasks["اولویت"] == "بالا").sum()))
        c3.metric("در حال انجام", fa_num((tasks["وضعیت"] == "در حال انجام").sum()))
        c4.metric("میانگین تحقق KPI", pct(float(tasks["پیشرفت"].clip(upper=1).mean())))

    section_heading("تابلوی اقدام", "تسک‌های مدیریتی", "در V0.7 تغییرات این جدول در Session نگه داشته می‌شود؛ برای استفاده واقعی باید به Database/Task Manager وصل شود.")
    edited = st.data_editor(st.session_state.mgmt_tasks, use_container_width=True, hide_index=True, num_rows="dynamic", key="v07_task_editor")
    st.session_state.mgmt_tasks = edited.copy()

    with st.expander("افزودن تسک جدید", expanded=False):
        with st.form("v07_add_task"):
            a,b,c = st.columns(3)
            with a: dept = st.selectbox("بخش", [d["name"] for d in DEPARTMENTS])
            with b: owner = st.text_input("مسئول", "سرپرست واحد")
            with c: priority = st.selectbox("اولویت", ["بالا","متوسط","پایین"])
            task = st.text_input("عنوان تسک")
            kpi_name = st.text_input("KPI مرتبط", "KPI واحد")
            d1,d2 = st.columns(2)
            with d1: target = st.number_input("هدف KPI", value=1.0, step=0.05)
            with d2: actual = st.number_input("عملکرد فعلی", value=0.0, step=0.05)
            if st.form_submit_button("افزودن به تابلوی اقدام", use_container_width=True) and task.strip():
                new_row = pd.DataFrame([[dept, task.strip(), owner.strip() or "سرپرست واحد", "Backlog", priority, kpi_name.strip() or "KPI واحد", float(target), float(actual)]], columns=DEFAULT_TASKS.columns)
                st.session_state.mgmt_tasks = pd.concat([st.session_state.mgmt_tasks, new_row], ignore_index=True)
                st.rerun()

    section_heading("تسک‌های پیشنهادی", "وقتی KPI پایین است چه کاری باز شود؟", "Rule-based و قابل توضیح؛ اجرای واقعی بعداً می‌تواند با n8n Task بسازد.")
    recs = recommended_tasks(scenario, kpis, overrides)
    if recs.empty:
        st.success("KPI نیازمند اقدام در قواعد فعلی پیدا نشد.")
    else:
        st.dataframe(recs, use_container_width=True, hide_index=True)


def production_qc_page(scenario, kpis):
    if not _management_ready():
        return
    _ensure_management_state()
    page_header("تولید و QC", "از تعداد دستگاه تولیدشده تا QC، Rework، موجودی آماده و Order تولید؛ هدف این صفحه جلوگیری از فروش بدون آمادگی عملیاتی است.")

    with st.expander("ورودی عملیاتی QC / تولید (فعلاً Demo)", expanded=False):
        q1,q2,q3,q4 = st.columns(4)
        with q1: st.number_input("آماده خروج از QC", min_value=0, step=1, key="mgmt_qc_ready")
        with q2: st.number_input("در صف QC", min_value=0, step=1, key="mgmt_qc_pending")
        with q3: st.number_input("Reject شده", min_value=0, step=1, key="mgmt_qc_rejected")
        with q4: st.number_input("Rework", min_value=0, step=1, key="mgmt_qc_rework")
        p1,p2,p3 = st.columns(3)
        with p1: st.number_input("ظرفیت تولید روزانه", min_value=0, step=1, key="mgmt_production_daily_capacity")
        with p2: st.slider("نرخ قبولی QC", 0.0, 1.0, step=0.01, key="mgmt_qc_pass_rate")
        with p3: st.number_input("موجودی قطعه/Raw (Demo)", min_value=0, step=10, key="mgmt_raw_inventory")

    o = _management_overrides()
    total_qc = o["qc_ready"] + o["qc_pending"] + o["qc_rejected"] + o["qc_rework"]
    cols = st.columns(5)
    cols[0].metric("QC شده / آماده", fa_num(o["qc_ready"]))
    cols[1].metric("در انتظار QC", fa_num(o["qc_pending"]))
    cols[2].metric("Rework", fa_num(o["qc_rework"]))
    cols[3].metric("Reject", fa_num(o["qc_rejected"]))
    cols[4].metric("نرخ قبولی", pct(o["qc_pass_rate"]))

    left,right = st.columns([1,1])
    with left:
        qcdf = pd.DataFrame({"وضعیت":["آماده","در صف","Rework","Reject"],"تعداد":[o["qc_ready"],o["qc_pending"],o["qc_rework"],o["qc_rejected"]]})
        fig = px.bar(qcdf, x="وضعیت", y="تعداد", text="تعداد", color="تعداد", color_continuous_scale=["#31506B", ACCENT])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with right:
        section_heading("Order تولید", "برای Target بعدی چند دستگاه سفارش تولید بدهیم؟")
        target_default = max(int(float(kpis.get("monthly_units", 0))), 1)
        target_units = st.number_input("Target فروش / تحویل در بازه", min_value=0, value=target_default, step=10, key="v07_prod_target")
        horizon = st.slider("بازه برنامه‌ریزی تولید (روز)", 1, 60, 30, 1, key="v07_prod_horizon")
        safety = st.slider("Safety Stock", 0, 40, 10, 5, key="v07_prod_safety") / 100
        plan = production_plan(int(target_units), int(o["qc_ready"]), int(o["qc_pending"]), float(o["qc_pass_rate"]), int(o["production_daily_capacity"]), int(horizon), float(safety))
        st.markdown(f'''<div class="campaign-result"><div class="campaign-status">PRODUCTION PLAN</div><div class="campaign-title">Order پیشنهادی: {fa_num(plan['production_order'])} دستگاه</div><div class="campaign-copy">پس از احتساب {fa_num(plan['expected_qc_release'])} دستگاه خروجی احتمالی از QC و Safety Stock، وضعیت ظرفیت: <b>{plan['status']}</b>. حداکثر تولید در این بازه {fa_num(plan['max_production_in_horizon'])} دستگاه است.</div></div>''', unsafe_allow_html=True)
        if plan["capacity_gap"] > 0:
            st.warning(f"شکاف ظرفیت: {fa_num(plan['capacity_gap'])} دستگاه. قبل از تعهد کمپین باید زمان/ظرفیت تولید اصلاح شود.")
        else:
            st.success("با فرض‌های Demo فعلی، Order پیشنهادی در بازه ظرفیت تولید قرار می‌گیرد.")


def campaign_planner_page(scenario, kpis):
    if not _management_ready():
        return
    _ensure_management_state()
    sm = scenario_summary(scenario)
    o = _management_overrides()
    current_revenue = float(kpis.get("monthly_revenue", 0))
    page_header("برنامه‌ریز جشنواره", "وقتی Revenue زیر Target است، سیستم قبل از پیشنهاد جشنواره حاشیه سود، قیمت، موجودی QC و ظرفیت تولید را همزمان کنترل می‌کند.")

    gap = max(float(o["revenue_target"]) - current_revenue, 0)
    if gap > 0:
        st.warning(f"Revenue مدل فعلی حدود {toman(gap)} تومان پایین‌تر از Target آزمایشی است. این فقط Trigger سناریوسازی است؛ Revenue حسابداری هنوز متصل نیست.")
    else:
        st.success("Revenue مدل فعلی بالاتر از Target آزمایشی است؛ جشنواره از نظر Revenue فوریت ندارد.")

    with st.form("v07_campaign_form"):
        a,b,c,d = st.columns(4)
        with a: revenue_target = st.number_input("Target درآمد ماهانه", min_value=0, value=int(o["revenue_target"]), step=100_000_000)
        with b: unit_cost = st.number_input("بهای تمام‌شده هر دستگاه (Demo)", min_value=0, value=10_000_000, step=500_000)
        with c: regular_price = st.number_input("قیمت عادی", min_value=1, value=int(sm["price_a"]), step=500_000)
        with d: campaign_price = st.number_input("قیمت جشنواره", min_value=1, value=max(int(sm["price_a"] - 2_000_000),1), step=500_000)
        e,f,g,h = st.columns(4)
        with e: days = st.slider("مدت جشنواره (روز)", 1, 14, 3, 1)
        with f: daily_target = st.number_input("Target فروش روزانه جشنواره", min_value=0, value=30, step=5)
        with g: min_margin = st.slider("حداقل حاشیه سود مجاز", 0, 70, 20, 5) / 100
        with h: safety = st.slider("Safety Stock تولید", 0, 40, 10, 5) / 100
        st.form_submit_button("تحلیل سناریوی جشنواره", use_container_width=True)

    plan = campaign_plan(
        current_revenue=current_revenue,
        revenue_target=float(revenue_target),
        regular_price=float(regular_price),
        campaign_price=float(campaign_price),
        unit_cost=float(unit_cost),
        campaign_days=int(days),
        daily_sales_target=int(daily_target),
        ready_inventory=int(o["qc_ready"]),
        qc_pending=int(o["qc_pending"]),
        qc_pass_rate=float(o["qc_pass_rate"]),
        production_daily_capacity=int(o["production_daily_capacity"]),
        min_margin_pct=float(min_margin),
        safety_stock_pct=float(safety),
    )

    st.markdown(f'''<div class="campaign-result"><div class="campaign-status">CAMPAIGN READINESS · {plan['readiness']}</div><div class="campaign-title">{plan['recommendation']}</div><div class="campaign-copy">برای پوشش سناریوی فعلی، Target پیشنهادی {fa_num(plan['target_units'])} دستگاه است. Order تولید پیشنهادی {fa_num(plan['production_order'])} دستگاه و Gross Profit مدل {toman(plan['gross_profit'])} تومان است.</div><div class="campaign-names">{''.join(f'<span class="campaign-name">{name}</span>' for name in plan['name_suggestions'])}</div></div>''', unsafe_allow_html=True)
    r1,r2,r3,r4,r5 = st.columns(5)
    r1.metric("شکاف درآمد", f"{toman(plan['revenue_gap'])}")
    r2.metric("Target دستگاه", fa_num(plan["target_units"]))
    r3.metric("حاشیه سود", pct(plan["margin"]))
    r4.metric("کف قیمت امن", f"{toman(plan['min_allowed_price'])}")
    r5.metric("Order تولید", fa_num(plan["production_order"]))

    if not plan["margin_ok"]:
        st.error("قیمت جشنواره زیر کف سود تعریف‌شده است. این سناریو باید قبل از اجرا اصلاح شود.")
    elif plan["capacity_gap"] > 0:
        st.warning(f"ظرفیت تولید در بازه برای {fa_num(plan['capacity_gap'])} دستگاه کم است.")
    else:
        st.info("سناریو از نظر Margin و ظرفیت اولیه قابل بررسی است؛ Demand واقعی، بودجه، Lead و تأیید مالی قبل از اجرا لازم‌اند.")

    section_heading("حد سود و تخفیف", "مرز تصمیم")
    cols = st.columns(4)
    cols[0].metric("سود ناخالص / دستگاه", toman(plan["gross_profit_per_unit"]))
    cols[1].metric("تخفیف پیشنهادی", pct(plan["discount_pct"]))
    cols[2].metric("حداکثر تخفیف با کف سود", pct(plan["max_discount_pct"]))
    cols[3].metric("درآمد کمپین مدل", toman(plan["campaign_revenue"]))
    st.caption("بهای تمام‌شده، Target Revenue و داده QC در این نسخه Demo هستند مگر اینکه خودتان با داده واقعی جایگزین کنید.")


def automation_center_page(scenario, kpis):
    if not _management_ready():
        return
    _ensure_management_state()
    page_header("مرکز اتوماسیون", "تعریف اینکه چه چیزی، با چه بازه‌ای بررسی شود و اگر شرطی برقرار شد چه اقدام مدیریتی یا Workflow در n8n ساخته شود.")
    st.markdown('''<div class="flow-strip"><span class="flow-node">Schedule</span><span class="flow-arrow">←</span><span class="flow-node">Check KPI</span><span class="flow-arrow">←</span><span class="flow-node">Trigger</span><span class="flow-arrow">←</span><span class="flow-node">Task / Alert</span><span class="flow-arrow">←</span><span class="flow-node">n8n / API</span><span class="flow-arrow">←</span><span class="flow-node">Verify Result</span></div>''', unsafe_allow_html=True)

    options = ["هر ۲ ساعت","هر ۶ ساعت","روزانه","هفتگی","دستی"]
    section_heading("بازه انجام خودکار", "هر واحد چند وقت یک‌بار بررسی شود؟", "در V0.7 این Schedule ذخیره دائمی نمی‌شود؛ طراحی Workflow آینده است.")
    cols = st.columns(3)
    with cols[0]: st.selectbox("حسابداری", options, key="mgmt_run_finance")
    with cols[1]: st.selectbox("فروش", options, key="mgmt_run_sales")
    with cols[2]: st.selectbox("QC دستگاه", options, key="mgmt_run_qc")
    cols2 = st.columns(3)
    with cols2[0]: st.selectbox("مارکتینگ", options, key="mgmt_run_marketing")
    with cols2[1]: st.selectbox("منابع انسانی", options, key="mgmt_run_hr")
    with cols2[2]: st.selectbox("برنامه‌نویسی", options, key="mgmt_run_development")

    trigger_ratio = st.slider("اگر درآمد از چند درصد Target پایین‌تر رفت Trigger شود؟", 50, 110, 90, 5, key="v07_revenue_trigger") / 100
    rules = automation_checks(scenario, kpis, _management_overrides(), trigger_ratio)
    active = rules[rules["فعال شده"] == True]
    c1,c2,c3 = st.columns(3)
    c1.metric("قواعد تعریف‌شده", fa_num(len(rules)))
    c2.metric("Trigger فعال", fa_num(len(active)))
    c3.metric("اتصال n8n", "آماده طراحی", help="فعلاً اتصال واقعی وجود ندارد.")

    section_heading("قواعد فعال", "اگر X شد → Y انجام شود")
    for idx, row in rules.iterrows():
        is_on = bool(row["فعال شده"])
        cls = "auto-rule auto-rule-active" if is_on else "auto-rule"
        state_cls = "auto-state auto-on" if is_on else "auto-state auto-off"
        state_text = "Trigger شده" if is_on else "غیرفعال"
        st.markdown(f'''<div class="{cls}"><div class="auto-rule-name">{row['قانون']} · {row['بخش']}</div><div class="auto-rule-copy">شرط: {row['شرط']} · منبع: {row['منبع']}</div><div class="auto-rule-action">اقدام: {row['اقدام پیشنهادی']}</div><span class="{state_cls}">{state_text}</span></div>''', unsafe_allow_html=True)

    if st.button("ساخت Workflow در n8n (Placeholder)", use_container_width=True, type="primary", key="v07_n8n_placeholder"):
        st.toast("در V0.7 فقط قرارداد Workflow طراحی شده است. اتصال واقعی بعد از تعریف Credential، Schema و سطح دسترسی ساخته می‌شود.")
    st.caption("برای فاز واقعی: هر Rule باید Trigger، Schedule، Source، Action، Owner، Retry Policy و Audit Log داشته باشد.")


def access_control_page():
    if not _management_ready():
        return
    page_header("دسترسی و نقش‌ها", "طراحی دسترسی محدود برای مدیرعامل و سرپرستان واحدها؛ فعلاً فقط ماتریس پیشنهادی است و Authentication واقعی پیاده نشده است.")
    roles = ROLE_MATRIX["نقش"].tolist() if not ROLE_MATRIX.empty else ["مدیرعامل / مدیر سیستم"]
    role = st.selectbox("پروفایل نمایشی", roles, key="v07_role_preview")
    st.markdown(f'''<div class="management-hero"><div class="management-hero-kicker">ROLE PREVIEW</div><div class="management-hero-title">{role}</div><div class="management-hero-copy">در نسخه واقعی، هر سرپرست فقط داده و تسک‌های واحد خودش را می‌بیند یا ویرایش می‌کند. داده‌های مالی، تنظیمات اتصال و سطح دسترسی فقط برای نقش‌های مجاز باز می‌ماند.</div></div>''', unsafe_allow_html=True)
    st.dataframe(ROLE_MATRIX, use_container_width=True, hide_index=True)
    section_heading("معماری دسترسی", "چه چیزی بعداً باید اضافه شود؟")
    cols = st.columns(4)
    items = [
        ("ورود سازمانی", "Email/SSO یا Provider تأییدشده"),
        ("Role-Based Access", "مجوز مشاهده و ویرایش در سطح واحد"),
        ("Audit Log", "چه کسی چه داده‌ای را تغییر داد"),
        ("Approval Flow", "تأیید جشنواره، هزینه، Production Order و تغییر KPI"),
    ]
    for col,(title,copy) in zip(cols,items):
        with col: st.markdown(f'''<div class="role-box"><div class="role-label">فاز بعد</div><div class="role-value">{title}</div><div class="role-copy">{copy}</div></div>''', unsafe_allow_html=True)
    st.warning("در V0.7 هیچ احراز هویت واقعی وجود ندارد؛ برای داده واقعی شرکت نباید تا قبل از اضافه شدن Authentication و مجوزها دسترسی عمومی داده شود.")


def settings_page(scenario, kpis):
    sm = scenario_summary(scenario)
    page_header(
        "تنظیمات و سناریو",
        "این صفحه برای دموی مدیریتی است؛ با تغییر یک ورودی، داده مصنوعی و تمام خروجی‌های وابسته دوباره محاسبه می‌شوند.",
    )
    rows = [
        ("قیمت طرح A", f"{toman(sm["price_a"])} تومان", "مبنای واقعی / قابل تغییر"),
        ("قیمت طرح B", f"{toman(sm["price_b"])} تومان", "مبنای واقعی / قابل تغییر"),
        ("سهم طرح A", pct(sm["share_a"]), "سناریو"),
        ("فروش تلفنی روزانه", fa_num(sm["phone_daily"]), "مبنای واقعی"),
        ("فروش آنلاین ماهانه", fa_num(sm["online_monthly"]), "مبنای واقعی"),
        ("صف لید", fa_num(sm["lead_backlog"]), "مبنای واقعی"),
        ("استوری / روز", fa_num(sm["stories"]), "مبنای واقعی"),
        ("ریلز / روز", fa_num(sm["reels"]), "مبنای واقعی"),
        ("فالوئر", fa_num(sm["followers"]), "نمای ثبت‌شده"),
        ("فروش منتسب به محتوا / روز", fa_num(sm["content_sales"], 1), "تخمینی"),
        ("روز فروش / ماه", fa_num(sm["sales_days"]), "فرض مدل"),
        ("مشتری مصنوعی", fa_num(sm["customer_count"]), "آزمایشی"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["متغیر", "مقدار فعلی", "نوع داده"]), use_container_width=True, hide_index=True)

    section_heading("وابستگی زنده", "وابستگی زنده")
    st.code(
        f"فروش تلفنی روزانه = {fa_num(sm['phone_daily'])}\n"
        f"فروش تلفنی ماهانه = {fa_num(kpis['monthly_phone_units'])}\n"
        f"کل فروش ماهانه = {fa_num(kpis['monthly_units'])}\n"
        f"میانگین قیمت فروش = {toman(sm['asp'])} تومان\n"
        f"درآمد ماهانه محاسبه‌شده = {toman(kpis['monthly_revenue'])} تومان",
        language="text",
    )
    st.success("برای دمو: فروش تلفنی روزانه را از ۱۰ به ۱۵ تغییر بده؛ درآمد، تعداد فروش، ترکیب کانال، روند آزمایشی، پیش‌بینی و بینش‌ها هم‌زمان تغییر می‌کنند.")


def main():
    if "presentation_mode" not in st.session_state:
        st.session_state.presentation_mode = False
    inject_presentation_mode_css(bool(st.session_state.presentation_mode))
    scenario, page = scenario_sidebar()
    if st.session_state.presentation_mode:
        left, right = st.columns([5, 1])
        with left:
            st.markdown('<div class="presentation-ribbon"><span>حالت ارائه مدیرعامل فعال است · منوها و کنترل‌های فنی پنهان شده‌اند.</span><span>V0.7</span></div>', unsafe_allow_html=True)
        with right:
            if st.button("خروج از ارائه", use_container_width=True, key="exit_presentation"):
                st.session_state.presentation_mode = False
                st.rerun()
    data = build_synthetic_data(scenario)

    kpis = normalize_kpis(scenario, current_kpis(scenario, data["customers"]))
    daily = sales_daily(data["sales"])
    monthly = sales_monthly(data["sales"])
    funnel = lead_funnel(scenario)
    _ = backlog_capacity(scenario)

    customers_model, segment_profile, risk_stats = build_models(data["customers"], int(scenario_value(scenario, "seed", None, 42)))
    forecast, forecast_stats = revenue_forecast(monthly, 3)
    sales_anomalies = detect_anomalies(daily, "revenue", "date", 14)
    sms_anomalies = detect_anomalies(data["sms"], "delivery_rate", "date", 14)
    insights = generate_insights(scenario, kpis, monthly, risk_stats, sales_anomalies, sms_anomalies)

    if page == "Executive Overview":
        executive_overview(scenario, data, kpis, monthly, funnel, forecast, insights, customers_model, risk_stats, sales_anomalies, sms_anomalies)
    elif page == "Organization Pulse":
        organization_pulse_page(scenario, kpis)
    elif page == "Task & KPI":
        task_kpi_page(scenario, kpis)
    elif page == "Production & QC":
        production_qc_page(scenario, kpis)
    elif page == "Campaign Planner":
        campaign_planner_page(scenario, kpis)
    elif page == "Automation Center":
        automation_center_page(scenario, kpis)
    elif page == "Access Control":
        access_control_page()
    elif page == "Revenue Intelligence":
        revenue_intelligence_page(scenario, kpis)
    elif page == "Scenario Simulator":
        scenario_simulator_page(scenario, kpis)
    elif page == "Connections":
        connections_page()
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
    elif page == "Media Intelligence":
        media_intelligence_page()
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
    st.caption("NIK MANAGEMENT OS V0.7 — پنل مدیریتی و اتوماسیون با داده مبنا و داده آزمایشی؛ هنوز به سیستم‌های داخلی نیک متصل نیست.")
    st.caption("پیش‌بینی‌ها و خروجی‌های یادگیری ماشین آزمایشی‌اند و نباید مبنای تصمیم قطعی عملیاتی قرار گیرند.")


if __name__ == "__main__":
    main()
