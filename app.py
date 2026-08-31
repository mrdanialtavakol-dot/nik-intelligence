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

# V0.8 CEO operations layer is also optional so the V0.7 core remains usable
# even during a partial multi-file redeploy.
try:
    from ceo_ops_engine import (
        CEO_NAME,
        CURRENT_LEAD_BACKLOG,
        DEFAULT_SALES_AGENTS,
        ORGANIZATION_ROSTER,
        TASK_COLUMNS,
        allocate_lead_backlog,
        audit_event,
        ceo_inbox,
        classify_finance_transactions,
        department_report_schedule,
        finance_automation_summary,
        generate_demo_transactions,
        generate_sales_performance_demo,
        lead_center_summary,
        new_task_row,
        reporting_summary,
        robot_task_suggestions,
        seed_ceo_tasks,
        task_followup_status,
        task_summary,
    )
    CEO_OPS_AVAILABLE = True
    CEO_OPS_ERROR = ""
except Exception as _ceo_ops_error:
    CEO_OPS_AVAILABLE = False
    CEO_OPS_ERROR = str(_ceo_ops_error)
    CEO_NAME = "کیوان میرزایی"
    CURRENT_LEAD_BACKLOG = 5_490
    DEFAULT_SALES_AGENTS = pd.DataFrame()
    ORGANIZATION_ROSTER = pd.DataFrame()
    TASK_COLUMNS = []

# V0.9 executive suite is optional; V0.8 remains usable during partial deploys.
try:
    from v09_engine import (
        V09_VERSION, CEO_PAGES, DEPARTMENT_PAGES, GROWTH_PAGES, INTELLIGENCE_PAGES, SYSTEM_PAGES,
        fx_cost_snapshot, marketing_economics_snapshot, department_kpi_catalog,
    )
    from font_loader import iransansx_css
    V09_AVAILABLE = True
    V09_ERROR = ""
except Exception as _v09_error:
    V09_AVAILABLE = False
    V09_ERROR = str(_v09_error)
    V09_VERSION = "V0.9"

# V0.10 customer growth product layer is isolated from the internal management OS.
try:
    from customer_growth_engine import (
        V10_VERSION, CUSTOMER_BUSINESS_PAGES, VERTICAL_PROFILES, PLAN_CATALOG,
        business_snapshot, segment_table, campaign_opportunities, automation_catalog,
        growth_trend, business_health_insights,
    )
    CUSTOMER_GROWTH_AVAILABLE = True
    CUSTOMER_GROWTH_ERROR = ""
except Exception as _customer_growth_import_error:
    CUSTOMER_GROWTH_AVAILABLE = False
    CUSTOMER_GROWTH_ERROR = str(_customer_growth_import_error)
    V10_VERSION = "V0.10"
    CUSTOMER_BUSINESS_PAGES = {"Customer Growth Home": "مرکز رشد کسب‌وکار"}
    VERTICAL_PROFILES = {"پوشاک": {}}
    PLAN_CATALOG = pd.DataFrame()


st.set_page_config(
    page_title="نیک اس‌ام‌اس | Management OS + Growth Intelligence V0.11",
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
try:
    _font_css = iransansx_css(BASE_DIR) if V09_AVAILABLE else ""
    if _font_css:
        st.markdown(_font_css, unsafe_allow_html=True)
except Exception:
    pass

# UI safety net: if Streamlit Cloud briefly loads an older Scenario module
# during a multi-file redeploy, optional fields still have safe defaults.
SCENARIO_DEFAULTS = {
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
    "CEO Task Center": "کارها و گزارش‌های کیوان میرزایی",
    "IT Workspace": "اتاق عملیات IT",
    "Accounting Automation": "اتوماسیون حسابداری",
    "Sales Lead Center": "مرکز پخش و پیگیری لید",
    "Support Workspace": "مدیریت پشتیبانی",
    "HR Workspace": "مدیریت منابع انسانی",
    "Marketing Workspace": "نمای کلی مارکتینگ",
    "Marketing Economics": "اقتصاد مارکتینگ",
    "Marketing Trend": "روند و عملکرد مارکتینگ",
    "Sales Funnel Ops": "پخش لید و عملکرد فروش",
    "Finance Dashboard": "نمای مالی مدیریتی",
    "IT Delivery": "آپدیت، تسک و تحویل IT",
    "FX & Supply": "ارز، تأمین و بهای ساخت",
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
    "CEO Task Center": "☰",
    "IT Workspace": "⌘",
    "Accounting Automation": "◫",
    "Sales Lead Center": "↗",
    "Support Workspace": "◍",
    "HR Workspace": "◎",
    "Marketing Workspace": "▶",
    "Marketing Economics": "◫",
    "Marketing Trend": "⌁",
    "Sales Funnel Ops": "↗",
    "Finance Dashboard": "◫",
    "IT Delivery": "⌘",
    "FX & Supply": "◇",
}
NAV_GROUPS = {
    "مرکز کیوان میرزایی": ["Executive Overview", "CEO Task Center", "Organization Pulse"],
    "واحدهای شرکت": ["IT Workspace", "Accounting Automation", "Sales Lead Center", "Support Workspace", "HR Workspace", "Production & QC", "Marketing Workspace"],
    "رشد و تصمیم": ["Revenue Intelligence", "Campaign Planner", "Scenario Simulator"],
    "هوشمندی داده": ["Content Analytics", "Media Intelligence", "Customer Intelligence", "NIKPOS Analytics", "SMS Analytics", "Anomaly Detection", "Predictions", "Automated Insights"],
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

V08_CEO_CSS = r"""
<style>
.ceo-personal-hero{position:relative;overflow:hidden;padding:22px 24px;border-radius:26px;margin:8px 0 16px;background:linear-gradient(135deg,rgba(173,203,255,.13),rgba(255,255,255,.024));border:1px solid rgba(173,203,255,.20);box-shadow:inset 0 1px 0 rgba(255,255,255,.13),0 24px 70px rgba(0,0,0,.18)}
.ceo-personal-hero::before{content:"";position:absolute;right:-70px;top:-90px;width:250px;height:250px;border-radius:50%;background:radial-gradient(circle,rgba(173,203,255,.18),transparent 68%)}
.ceo-personal-kicker{font-size:.67rem;font-weight:900;color:#ADCBFF;letter-spacing:.09em}.ceo-personal-title{font-size:1.55rem;font-weight:900;color:#fff;margin-top:6px}.ceo-personal-copy{color:#AFC0D1;font-size:.78rem;line-height:1.9;max-width:960px;margin-top:7px}
.ceo-inbox-card{padding:15px 16px;border-radius:18px;background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.018));border:1px solid rgba(255,255,255,.08);margin-bottom:8px}.ceo-inbox-top{display:flex;justify-content:space-between;gap:8px;align-items:center}.ceo-inbox-title{font-size:.82rem;font-weight:850;color:#F7FBFF}.ceo-inbox-meta{font-size:.63rem;color:#7F95AA;margin-top:5px;line-height:1.6}.ceo-inbox-action{font-size:.69rem;color:#ADCBFF;margin-top:8px;font-weight:750}
.sev-high{color:#FFC1CB;background:rgba(255,89,113,.09);border:1px solid rgba(255,89,113,.15)}.sev-med{color:#FFE0A3;background:rgba(244,180,79,.08);border:1px solid rgba(244,180,79,.14)}.sev-low{color:#A5F2CB;background:rgba(48,205,135,.08);border:1px solid rgba(48,205,135,.14)}
.team-card{min-height:142px;padding:15px 16px;border-radius:20px;background:linear-gradient(150deg,rgba(255,255,255,.065),rgba(255,255,255,.018));border:1px solid rgba(255,255,255,.08);box-shadow:inset 0 1px 0 rgba(255,255,255,.07)}.team-name{font-size:.88rem;font-weight:850;color:#F9FCFF}.team-role{font-size:.67rem;color:#8499AD;margin-top:3px}.team-chip{display:inline-flex;margin-top:10px;padding:4px 7px;border-radius:999px;color:#CFE1F4;font-size:.60rem;background:rgba(173,203,255,.055);border:1px solid rgba(173,203,255,.11)}
.ops-flow{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin:10px 0 18px}.ops-step{min-height:78px;padding:11px;border-radius:15px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);font-size:.68rem;color:#B8C7D6;line-height:1.7}.ops-step b{display:block;color:#F4F9FF;font-size:.71rem;margin-bottom:3px}
.auto-ledger{padding:16px 18px;border-radius:20px;background:linear-gradient(135deg,rgba(173,203,255,.075),rgba(255,255,255,.018));border:1px solid rgba(173,203,255,.13)}.auto-ledger-title{font-size:.93rem;color:#fff;font-weight:850}.auto-ledger-copy{font-size:.70rem;color:#8298AD;line-height:1.9;margin-top:6px}
.lead-routing-hero{padding:18px 20px;border-radius:22px;background:linear-gradient(135deg,rgba(173,203,255,.095),rgba(255,255,255,.02));border:1px solid rgba(173,203,255,.14)}.lead-routing-value{font-size:1.85rem;color:#fff;font-weight:900;letter-spacing:-.03em}.lead-routing-label{font-size:.68rem;color:#8198AE;margin-top:3px}
.report-row{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:12px 14px;border-radius:15px;margin-bottom:7px;background:rgba(255,255,255,.028);border:1px solid rgba(255,255,255,.065)}.report-name{font-size:.75rem;font-weight:800;color:#F5FAFF}.report-sub{font-size:.61rem;color:#7890A6;margin-top:2px}
.audit-pill{display:inline-flex;padding:4px 8px;border-radius:999px;font-size:.60rem;font-weight:800;background:rgba(173,203,255,.07);border:1px solid rgba(173,203,255,.12);color:#CFE1F4}
@media(max-width:980px){.ops-flow{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:560px){.ops-flow{grid-template-columns:1fr 1fr}}
</style>
"""
st.markdown(V08_CEO_CSS, unsafe_allow_html=True)



V09_LIGHT_CSS = r"""
<style>
:root{--v09-bg:#F3F8FF;--v09-sky:#ADCBFF;--v09-blue:#3D78C5;--v09-deep:#17385D;--v09-text:#102943;--v09-muted:#657D96}
html,body,[class*="css"],.stApp{font-family:"IRANSansX","Vazirmatn","Segoe UI",Tahoma,sans-serif!important}
.stApp{color:#102943!important;background:radial-gradient(820px 520px at 87% -8%,rgba(173,203,255,.92),transparent 66%),radial-gradient(680px 520px at 7% 30%,rgba(215,233,255,.92),transparent 67%),radial-gradient(560px 420px at 55% 88%,rgba(255,255,255,.95),transparent 70%),linear-gradient(145deg,#FAFDFF 0%,#F3F8FF 30%,#E7F1FF 66%,#D8E9FF 100%)!important;background-attachment:fixed!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(255,255,255,.79),rgba(238,247,255,.70))!important;border-left:1px solid rgba(255,255,255,.94)!important;border-right:0!important;box-shadow:-24px 0 80px rgba(75,119,170,.10)!important;-webkit-backdrop-filter:blur(42px) saturate(170%)!important;backdrop-filter:blur(42px) saturate(170%)!important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{color:#284663!important}[data-testid="stSidebar"] hr{border-color:rgba(64,111,158,.10)!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div{background:rgba(255,255,255,.52)!important;border-color:rgba(80,126,174,.10)!important;border-radius:16px!important}
.block-container{max-width:1540px!important;padding-top:1.1rem!important}
.hero-glass,.ceo-command,.glass-kpi,.glass-panel,.command-metric,.priority-card,.leverage-card,.v06-brief,.ceo-personal-hero,.team-card,.ceo-inbox-card,.auto-ledger,.lead-routing-hero,.connection-card,[data-testid="stMetric"],div[data-testid="stExpander"]{background:linear-gradient(145deg,rgba(255,255,255,.76),rgba(255,255,255,.44))!important;border:1px solid rgba(255,255,255,.93)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.98),0 18px 55px rgba(72,112,158,.11),0 1px 1px rgba(55,99,145,.06)!important;-webkit-backdrop-filter:blur(34px) saturate(165%)!important;backdrop-filter:blur(34px) saturate(165%)!important}
.kpi-label,.ceo-command-label,.money-chip-label,.sim-label,.trust-label,.connection-copy,.connection-fresh,.kpi-note,.section-subtitle,.ceo-status-copy,.v06-brief-copy,.ceo-personal-copy,.team-role,.ceo-inbox-meta,.gap-text,.leverage-note{color:#667F98!important}
.kpi-value,.ceo-command-value,.money-chip-value,.sim-value,.trust-value,.connection-name,.section-title,.ceo-status-title,.v06-brief-title,.ceo-personal-title,.team-name,.ceo-inbox-title,.gap-title,.leverage-title,.auto-ledger-title,.lead-routing-value{color:#122E4A!important}
.section-kicker,.ceo-overline,.v06-brief-kicker,.ceo-personal-kicker,.leverage-kicker,.ceo-inbox-action{color:#3D78C5!important}
.money-chip,.trust-cell,.gap-row,.ops-step{background:rgba(255,255,255,.50)!important;border-color:rgba(73,116,160,.10)!important}
[data-testid="stMetricLabel"] p{color:#657D96!important}[data-testid="stMetricValue"]{color:#12304E!important}
.stMarkdown,.stCaption,.stAlert,.stHeader,.stSubheader{color:#173650!important}
.stButton>button,.stDownloadButton>button{border-radius:15px!important;border:1px solid rgba(255,255,255,.92)!important;background:linear-gradient(145deg,rgba(255,255,255,.88),rgba(223,238,255,.66))!important;color:#173A60!important;box-shadow:inset 0 1px 0 #fff,0 9px 28px rgba(70,112,160,.09)!important;transition:transform .2s ease,box-shadow .2s ease!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#78A9E9,#4F84CC)!important;color:#fff!important;border-color:rgba(255,255,255,.72)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.65),0 12px 32px rgba(63,113,178,.24)!important}
@media(hover:hover) and (pointer:fine){.stButton>button:hover{transform:translateY(-1px)}.glass-kpi:hover,.command-metric:hover,.priority-card:hover,.team-card:hover{transform:translateY(-2px)!important}}
.v09-integration{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;padding:10px 12px;margin:4px 0 10px;border-radius:18px;background:rgba(255,255,255,.48);border:1px solid rgba(255,255,255,.88);box-shadow:inset 0 1px 0 #fff,0 10px 30px rgba(78,116,156,.07);backdrop-filter:blur(22px)}
.v09-integration-title{font-size:.69rem;font-weight:850;color:#355A7D}.v09-integration-state{display:flex;gap:7px;flex-wrap:wrap}.v09-connect-pill{display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;font-size:.61rem;font-weight:800;color:#3B6286;background:rgba(255,255,255,.62);border:1px solid rgba(81,130,180,.11)}.v09-dot{width:7px;height:7px;border-radius:50%;background:#7EACE6;box-shadow:0 0 0 4px rgba(126,172,230,.13)}
.v09-section-hero{position:relative;overflow:hidden;padding:20px 22px;margin:7px 0 16px;border-radius:25px;background:linear-gradient(135deg,rgba(255,255,255,.81),rgba(221,237,255,.56));border:1px solid rgba(255,255,255,.96);box-shadow:inset 0 1px 0 #fff,0 22px 62px rgba(73,115,160,.10)}
.v09-section-hero:after{content:"";position:absolute;left:-50px;top:-90px;width:210px;height:210px;border-radius:50%;background:radial-gradient(circle,rgba(173,203,255,.60),transparent 70%)}
.v09-hero-kicker{font-size:.65rem;font-weight:900;letter-spacing:.08em;color:#4A78A7}.v09-hero-title{font-size:1.25rem;font-weight:900;color:#12304F;margin-top:5px}.v09-hero-copy{font-size:.74rem;line-height:1.9;color:#657E97;max-width:960px;margin-top:5px}
.v09-icon-shell{width:42px;height:42px;border-radius:15px;display:inline-flex;align-items:center;justify-content:center;background:linear-gradient(145deg,rgba(255,255,255,.9),rgba(173,203,255,.38));border:1px solid rgba(255,255,255,.98);box-shadow:inset 0 1px 0 #fff,0 10px 24px rgba(77,124,177,.11)}.v09-icon-shell svg{width:21px;height:21px;fill:none;stroke:#3F73AA;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
.v09-subnav{padding:9px 11px;border-radius:16px;background:rgba(255,255,255,.48);border:1px solid rgba(255,255,255,.88);margin-bottom:10px;color:#5C7590;font-size:.66rem}
</style>
"""
st.markdown(V09_LIGHT_CSS, unsafe_allow_html=True)


V091_READABILITY_PATCH_CSS = r"""
<style>
/* V0.9.1 — light premium theme with stronger readability */
:root{
    --v091-bg-1:#F6FAFF;
    --v091-bg-2:#EAF3FF;
    --v091-bg-3:#DCEBFF;
    --v091-card:rgba(244,249,255,.82);
    --v091-card-strong:rgba(237,246,255,.91);
    --v091-text:#102C49;
    --v091-text-2:#284D70;
    --v091-muted:#58718A;
    --v091-soft:#71879D;
    --v091-edge:rgba(73,119,166,.16);
}
.stApp{
    color:var(--v091-text)!important;
    background:
        radial-gradient(820px 520px at 88% -9%,rgba(173,203,255,.72),transparent 66%),
        radial-gradient(720px 540px at 5% 33%,rgba(205,227,255,.78),transparent 68%),
        radial-gradient(600px 430px at 55% 90%,rgba(245,250,255,.90),transparent 72%),
        linear-gradient(145deg,var(--v091-bg-1) 0%,var(--v091-bg-2) 40%,var(--v091-bg-3) 100%)!important;
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,rgba(240,247,255,.94),rgba(224,238,255,.90))!important;
    border-left:1px solid rgba(75,119,166,.15)!important;
    box-shadow:-22px 0 64px rgba(66,104,144,.12)!important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div{color:#1F4567!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div,
[data-testid="stSidebar"] input{
    background:rgba(248,252,255,.82)!important;
    border-color:rgba(67,111,157,.16)!important;
    color:#173A5C!important;
}
.hero-glass,.ceo-command,.glass-kpi,.glass-panel,.command-metric,.priority-card,.leverage-card,
.v06-brief,.ceo-personal-hero,.team-card,.ceo-inbox-card,.auto-ledger,.lead-routing-hero,
.connection-card,.v09-section-hero,.v09-integration,[data-testid="stMetric"],div[data-testid="stExpander"]{
    background:linear-gradient(145deg,var(--v091-card-strong),rgba(226,239,255,.72))!important;
    border:1px solid rgba(255,255,255,.96)!important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,1),0 17px 45px rgba(61,101,143,.12),0 1px 1px rgba(55,99,145,.06)!important;
}
/* Primary copy */
.hero-title,.kpi-value,.section-title,.ceo-status-title,.ceo-command-value,.command-metric-value,
.priority-title,.money-chip-value,.leverage-title,.leverage-value,.gap-title,.v06-brief-title,
.trust-value,.sim-value,.connection-name,.ceo-personal-title,.ceo-inbox-title,.team-name,
.auto-ledger-title,.lead-routing-value,.report-name,.v09-hero-title,
[data-testid="stMetricValue"],h1,h2,h3,h4,h5,h6{
    color:var(--v091-text)!important;
    -webkit-text-fill-color:var(--v091-text)!important;
}
/* Secondary copy */
.hero-subtitle,.kpi-label,.kpi-note,.section-subtitle,.ceo-status-copy,.ceo-command-label,
.ceo-command-foot,.command-metric-label,.command-metric-note,.priority-text,.priority-source,
.money-chip-label,.leverage-note,.gap-text,.v06-brief-copy,.trust-label,.trust-note,.sim-label,
.connection-copy,.connection-fresh,.ceo-personal-copy,.ceo-inbox-meta,.team-role,.ops-step,
.auto-ledger-copy,.lead-routing-label,.report-sub,.v09-hero-copy,.v09-subnav,.media-meta,
.stCaption,[data-testid="stMetricLabel"] p{
    color:var(--v091-muted)!important;
    -webkit-text-fill-color:var(--v091-muted)!important;
}
.eyebrow,.section-kicker,.ceo-overline,.v06-brief-kicker,.ceo-personal-kicker,.leverage-kicker,
.ceo-inbox-action,.v09-hero-kicker,.v09-integration-title{
    color:#356FAE!important;
    -webkit-text-fill-color:#356FAE!important;
}
.hero-title .accent{
    background:linear-gradient(90deg,#173D64,#4D7FBC 55%,#2F659D)!important;
    -webkit-background-clip:text!important;
    background-clip:text!important;
    -webkit-text-fill-color:transparent!important;
}
/* Make old dark-theme components safe on light surfaces */
.ops-step b,.gap-index,.audit-pill,.team-chip,.meta-pill,.presentation-ribbon,.v09-connect-pill{
    color:#315A80!important;
    -webkit-text-fill-color:#315A80!important;
}
.money-chip,.trust-cell,.gap-row,.ops-step{
    background:rgba(242,248,255,.78)!important;
    border-color:rgba(72,117,163,.13)!important;
}
.kpi-icon,.v09-icon-shell,.gap-index{
    background:linear-gradient(145deg,rgba(255,255,255,.95),rgba(173,203,255,.46))!important;
    border-color:rgba(81,128,178,.15)!important;
}
.kpi-icon svg,.v09-icon-shell svg{stroke:#356DA8!important}
/* Controls */
.stButton>button,.stDownloadButton>button{
    color:#173E63!important;
    background:linear-gradient(145deg,rgba(252,254,255,.96),rgba(220,236,255,.88))!important;
    border-color:rgba(83,127,173,.17)!important;
}
.stButton>button[kind="primary"]{color:#fff!important;-webkit-text-fill-color:#fff!important}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,textarea{
    color:#173A5B!important;
    background:rgba(249,252,255,.90)!important;
}
/* Plotly/table surrounding text */
.stMarkdown,.stAlert,.stHeader,.stSubheader,p,li,label{
    color:var(--v091-text-2);
}
[data-testid="stDataFrame"],[data-testid="stTable"]{
    border-color:rgba(73,117,163,.14)!important;
    box-shadow:0 10px 30px rgba(61,101,143,.07)!important;
}
/* Slightly stronger source chips for accessibility */
.src-real{color:#11613F!important;background:rgba(35,170,108,.11)!important;border-color:rgba(35,170,108,.20)!important}
.src-derived{color:#2A5687!important;background:rgba(77,130,184,.11)!important;border-color:rgba(77,130,184,.20)!important}
.src-estimated{color:#78500C!important;background:rgba(232,171,57,.13)!important;border-color:rgba(196,137,28,.22)!important}
.src-synthetic{color:#5B438E!important;background:rgba(126,94,191,.11)!important;border-color:rgba(126,94,191,.19)!important}
@media(hover:hover) and (pointer:fine){
    .glass-kpi:hover,.command-metric:hover,.priority-card:hover,.team-card:hover{
        border-color:rgba(80,129,180,.24)!important;
        box-shadow:inset 0 1px 0 #fff,0 22px 52px rgba(61,101,143,.15)!important;
    }
}
</style>
"""
st.markdown(V091_READABILITY_PATCH_CSS, unsafe_allow_html=True)

V10_CUSTOMER_CSS = r"""
<style>
.customer-product-hero{position:relative;overflow:hidden;padding:24px 25px;border-radius:28px;margin:7px 0 15px;background:linear-gradient(135deg,rgba(255,255,255,.94),rgba(218,235,255,.80));border:1px solid rgba(255,255,255,.98);box-shadow:inset 0 1px 0 #fff,0 24px 64px rgba(61,101,143,.13)}
.customer-product-hero:after{content:"";position:absolute;left:-85px;top:-110px;width:310px;height:310px;border-radius:50%;background:radial-gradient(circle,rgba(125,174,236,.43),transparent 69%)}
.customer-product-kicker{position:relative;z-index:1;font-size:.66rem;font-weight:950;color:#356FAE;letter-spacing:.09em}.customer-product-title{position:relative;z-index:1;font-size:1.65rem;font-weight:950;color:#102C49;margin-top:6px}.customer-product-copy{position:relative;z-index:1;max-width:1020px;color:#58718A;font-size:.77rem;line-height:2;margin-top:6px}
.customer-health{display:grid;grid-template-columns:1.08fr repeat(3,minmax(0,1fr));gap:10px;margin:8px 0 17px}.customer-health-main,.customer-health-cell{padding:16px 17px;border-radius:21px;background:linear-gradient(145deg,rgba(251,253,255,.94),rgba(225,239,255,.80));border:1px solid rgba(255,255,255,.98);box-shadow:inset 0 1px 0 #fff,0 13px 36px rgba(66,105,148,.09)}.customer-health-main{background:linear-gradient(135deg,rgba(215,234,255,.92),rgba(248,252,255,.96))}.customer-health-label{font-size:.64rem;color:#627C95;font-weight:750}.customer-health-value{font-size:1.38rem;color:#123456;font-weight:950;margin-top:5px}.customer-health-note{font-size:.62rem;color:#72879B;line-height:1.7;margin-top:4px}
.growth-opportunity{height:100%;padding:18px;border-radius:22px;background:linear-gradient(145deg,rgba(250,253,255,.94),rgba(226,239,255,.78));border:1px solid rgba(255,255,255,.98);box-shadow:inset 0 1px 0 #fff,0 14px 40px rgba(66,105,148,.09)}.growth-opportunity-priority{display:inline-flex;padding:5px 9px;border-radius:999px;font-size:.60rem;font-weight:900;color:#2D5D8D;background:rgba(173,203,255,.35);border:1px solid rgba(88,135,185,.13)}.growth-opportunity-title{font-size:.90rem;font-weight:900;color:#12324F;margin-top:10px}.growth-opportunity-copy{font-size:.68rem;color:#607A93;line-height:1.85;margin-top:6px}.growth-opportunity-value{font-size:1.12rem;font-weight:950;color:#1B4D79;margin-top:10px}.growth-opportunity-meta{font-size:.62rem;color:#72879A;margin-top:4px}
.action-story{padding:20px 21px;border-radius:24px;background:linear-gradient(135deg,rgba(211,232,255,.86),rgba(249,252,255,.96));border:1px solid rgba(255,255,255,.98);box-shadow:inset 0 1px 0 #fff,0 16px 44px rgba(61,101,143,.10)}.action-story-line{display:flex;gap:10px;align-items:flex-start;padding:7px 0;border-bottom:1px solid rgba(62,107,153,.08)}.action-story-line:last-child{border-bottom:0}.action-story-index{width:27px;height:27px;flex:0 0 27px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.78);border:1px solid rgba(71,120,169,.12);color:#3E73AA;font-weight:950;font-size:.62rem}.action-story-text{color:#365875;font-size:.72rem;line-height:1.85}.action-story-text b{color:#102F4C}
.plan-card{height:100%;padding:20px;border-radius:25px;background:linear-gradient(145deg,rgba(251,253,255,.96),rgba(222,237,255,.84));border:1px solid rgba(255,255,255,.99);box-shadow:inset 0 1px 0 #fff,0 18px 48px rgba(64,105,149,.11)}.plan-card-featured{background:linear-gradient(145deg,rgba(215,234,255,.96),rgba(245,250,255,.98));border-color:rgba(116,164,218,.28);box-shadow:inset 0 1px 0 #fff,0 21px 56px rgba(56,105,161,.16)}.plan-eyebrow{font-size:.61rem;color:#4776A5;font-weight:900}.plan-name{font-size:1.03rem;font-weight:950;color:#123350;margin-top:5px}.plan-price{font-size:1.55rem;font-weight:950;color:#174B78;margin-top:10px}.plan-period{font-size:.62rem;color:#73899D;font-weight:700}.plan-best{font-size:.66rem;color:#617B93;line-height:1.75;margin:8px 0 10px}.plan-feature{font-size:.67rem;color:#345673;line-height:1.8;padding:4px 0}.plan-feature:before{content:"✓";color:#3976B7;font-weight:950;margin-left:6px}
.customer-automation-row{display:grid;grid-template-columns:1.1fr 1.3fr 2fr .65fr;gap:8px;align-items:center;padding:11px 12px;border-radius:16px;margin-bottom:7px;background:rgba(247,251,255,.86);border:1px solid rgba(76,121,167,.10);font-size:.66rem;color:#45647F}.customer-automation-row b{color:#173A59}.customer-status-ready{display:inline-flex;justify-content:center;padding:4px 7px;border-radius:999px;background:rgba(49,171,116,.10);color:#176A48;font-weight:850}.customer-status-suggest{display:inline-flex;justify-content:center;padding:4px 7px;border-radius:999px;background:rgba(232,171,57,.13);color:#75520B;font-weight:850}
.customer-source-note{padding:10px 12px;border-radius:15px;background:rgba(126,94,191,.07);border:1px solid rgba(126,94,191,.12);font-size:.64rem;color:#624C8C;line-height:1.8;margin:8px 0 14px}
@media(max-width:920px){.customer-health{grid-template-columns:1fr 1fr}.customer-automation-row{grid-template-columns:1fr 1fr}}
@media(max-width:560px){.customer-health{grid-template-columns:1fr}.customer-automation-row{grid-template-columns:1fr}}
</style>
"""
st.markdown(V10_CUSTOMER_CSS, unsafe_allow_html=True)


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
        "lead_backlog": float(scenario_value(scenario, "lead_backlog", "lead_backlog", 5_490)),
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
        height=height, paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="IRANSansX, Vazirmatn, Segoe UI, Tahoma", color="#324B66", size=12),
        margin=dict(l=24,r=24,t=28,b=24),
        legend=dict(bgcolor="rgba(255,255,255,.42)",bordercolor="rgba(103,148,197,.12)",borderwidth=1),
        xaxis=dict(gridcolor="rgba(71,111,154,.10)",zerolinecolor="rgba(71,111,154,.12)",linecolor="rgba(71,111,154,.14)"),
        yaxis=dict(gridcolor="rgba(71,111,154,.10)",zerolinecolor="rgba(71,111,154,.12)",linecolor="rgba(71,111,154,.14)"),
        hoverlabel=dict(bgcolor="#FFFFFF",font_color="#17324E",bordercolor="rgba(85,132,180,.20)"),
    )
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
    """Compact executive header. Context first; brand chrome second."""
    logo_uri = asset_data_uri(LOGO_PATH)
    logo_html = f'<img class="v11-head-logo" src="{logo_uri}" alt="NIKSMS">' if logo_uri else ""
    st.markdown(
        f"""
        <div class="v11-page-head">
            <div class="v11-page-copy">
                <div class="v11-page-kicker"><span class="v11-live-dot"></span>NIK EXECUTIVE OS · V0.11</div>
                <div class="v11-page-title">{page_title}</div>
                <div class="v11-page-subtitle">{subtitle or "تصمیم، اقدام و پیگیری در یک مسیر واحد؛ جزئیات فقط وقتی نمایش داده می‌شوند که برای تصمیم لازم باشند."}</div>
                <div class="v11-context-row">
                    <span class="v11-context-chip">کیوان میرزایی · مدیرعامل</span>
                    <span class="v11-context-chip v11-context-soft">Demo / Snapshot</span>
                    <span class="v11-context-chip v11-context-soft">داده زنده هنوز متصل نیست</span>
                </div>
            </div>
            <div class="v11-page-brand">{logo_html}<span>Management OS + Growth Intelligence</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Keep future connectors visible, but demote them from primary UI actions.
    status_col, api_col, n8n_col = st.columns([5.2, 1, 1])
    with status_col:
        st.markdown(
            '<div class="v11-connection-line"><span class="v11-connection-dot"></span><b>حالت دمو</b><span>API / Database / n8n آماده اتصال معماری هستند.</span></div>',
            unsafe_allow_html=True,
        )
    with api_col:
        if st.button("API", use_container_width=True, key=f"api_{page_title}"):
            st.toast("اتصال API در نسخه اجرایی فعال می‌شود؛ V0.11 هیچ Credential ذخیره نمی‌کند.")
    with n8n_col:
        if st.button("n8n", use_container_width=True, key=f"n8n_{page_title}"):
            st.toast("Workflow واقعی n8n در فاز اتصال داده فعال می‌شود.")


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
    """Navigation-first sidebar. Demo knobs stay out of the CEO's way by default."""
    if "presentation_mode" not in st.session_state:
        st.session_state.presentation_mode = False
    if "v09_main_area" not in st.session_state:
        st.session_state.v09_main_area = "میز مدیرعامل"

    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=164)
    st.sidebar.markdown(
        """
        <div class="v11-sidebar-brand">
            <div class="v11-sidebar-kicker">NIK EXECUTIVE OS</div>
            <div class="v11-sidebar-title">پنل مدیریتی نیک اس‌ام‌اس</div>
            <div class="v11-sidebar-copy">طراحی اختصاصی برای کیوان میرزایی · V0.11</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("حالت ارائه مدیرعامل", use_container_width=True, type="primary", key="v11_presentation"):
        st.session_state.presentation_mode = True
        st.rerun()

    st.sidebar.markdown('<div class="v11-side-section">ناوبری</div>', unsafe_allow_html=True)
    main_area = "میز مدیرعامل"
    if V09_AVAILABLE:
        main_area = st.sidebar.selectbox(
            "حوزه اصلی",
            ["میز مدیرعامل", "واحدهای شرکت", "رشد و تصمیم", "هوشمندی سازمان", "مشتریان کسب‌وکار", "سیستم و اتوماسیون"],
            key="v09_main_area",
        )
        if main_area == "میز مدیرعامل":
            page = st.sidebar.radio("صفحه", list(CEO_PAGES), format_func=lambda x: CEO_PAGES[x], key="v09_ceo_page")
        elif main_area == "واحدهای شرکت":
            department = st.sidebar.selectbox("واحد", list(DEPARTMENT_PAGES), key="v09_department")
            st.sidebar.markdown(f'<div class="v09-subnav">{department} · فقط ابزارهای همین واحد</div>', unsafe_allow_html=True)
            pages = DEPARTMENT_PAGES[department]
            page = st.sidebar.radio("بخش", list(pages), format_func=lambda x: pages[x], key=f"v09_department_page_{department}")
        elif main_area == "رشد و تصمیم":
            page = st.sidebar.radio("صفحه", list(GROWTH_PAGES), format_func=lambda x: GROWTH_PAGES[x], key="v09_growth_page")
        elif main_area == "هوشمندی سازمان":
            page = st.sidebar.radio("صفحه", list(INTELLIGENCE_PAGES), format_func=lambda x: INTELLIGENCE_PAGES[x], key="v09_intel_page")
        elif main_area == "مشتریان کسب‌وکار":
            st.sidebar.markdown('<div class="v09-subnav">NIK Growth Intelligence · تجربه مجزای صاحبان کسب‌وکار</div>', unsafe_allow_html=True)
            page = st.sidebar.radio("بخش", list(CUSTOMER_BUSINESS_PAGES), format_func=lambda x: CUSTOMER_BUSINESS_PAGES[x], key="v10_customer_business_page")
        else:
            page = st.sidebar.radio("صفحه", list(SYSTEM_PAGES), format_func=lambda x: SYSTEM_PAGES[x], key="v09_system_page")
    else:
        group = st.sidebar.selectbox("حوزه", list(NAV_GROUPS.keys()), key="nav_group")
        group_pages = NAV_GROUPS[group]
        page = st.sidebar.radio("صفحه", group_pages, format_func=lambda x: f"{PAGE_ICONS[x]}   {PAGE_LABELS[x]}", key=f"page_{group}")

    # Baseline defaults. They are always available even when the demo control panel is hidden.
    price_a = float(st.session_state.get("v11_price_a", baseline_value("plan_a_price", 15_000_000)))
    price_b = float(st.session_state.get("v11_price_b", baseline_value("plan_b_price", 30_000_000)))
    share_a_pct = int(st.session_state.get("v11_share_a_pct", int(baseline_value("plan_a_share", 0.50) * 100)))
    phone = int(st.session_state.get("v11_phone", baseline_value("daily_phone_sales", 10)))
    online = int(st.session_state.get("v11_online", baseline_value("monthly_online_sales", 20)))
    backlog = int(st.session_state.get("v11_backlog", baseline_value("lead_backlog", 5_490)))
    sales_days = int(st.session_state.get("v11_sales_days", baseline_value("sales_days_per_month", 30)))
    stories = int(st.session_state.get("v11_stories", baseline_value("stories_per_day", 9)))
    reels = int(st.session_state.get("v11_reels", baseline_value("reels_per_day", 1)))
    content_sales = float(st.session_state.get("v11_content_sales", baseline_value("estimated_content_sales_per_day", 2.0)))
    followers = int(st.session_state.get("v11_followers", baseline_value("instagram_followers", 207_000)))
    team_size = int(st.session_state.get("v11_team_size", baseline_value("content_team_size", 5)))
    customers = int(st.session_state.get("v11_customers", 5_000))
    months = int(st.session_state.get("v11_months", 12))
    seed = int(st.session_state.get("v11_seed", 42))

    st.sidebar.markdown('<div class="v11-side-section">ابزارهای دمو</div>', unsafe_allow_html=True)
    show_demo_controls = st.sidebar.checkbox(
        "نمایش کنترل‌های سناریو",
        value=False,
        help="برای ارائه مدیریتی خاموش بماند؛ فقط هنگام تست سناریو آن را باز کن.",
        key="v11_show_demo_controls",
    )
    if show_demo_controls:
        with st.sidebar.expander("فروش و درآمد", expanded=True):
            price_a = st.number_input("قیمت طرح A (تومان)", 1_000_000, 200_000_000, int(price_a), 1_000_000, key="v11_price_a")
            price_b = st.number_input("قیمت طرح B (تومان)", 1_000_000, 300_000_000, int(price_b), 1_000_000, key="v11_price_b")
            share_a_pct = st.slider("سهم طرح A", 0, 100, int(share_a_pct), 5, key="v11_share_a_pct")
            phone = st.number_input("فروش تلفنی روزانه", 0, 500, int(phone), 1, key="v11_phone")
            online = st.number_input("فروش آنلاین ماهانه", 0, 5_000, int(online), 5, key="v11_online")
            backlog = st.number_input("صف فعلی لید", 0, 500_000, int(backlog), 100, key="v11_backlog")
            sales_days = st.slider("روز فروش در ماه", 20, 31, int(sales_days), 1, key="v11_sales_days")
        with st.sidebar.expander("محتوا و مدل", expanded=False):
            stories = st.number_input("استوری روزانه", 0, 100, int(stories), 1, key="v11_stories")
            reels = st.number_input("ریلز روزانه", 0, 20, int(reels), 1, key="v11_reels")
            content_sales = st.number_input("فروش منتسب به محتوا / روز", 0.0, 100.0, float(content_sales), 0.5, key="v11_content_sales")
            followers = st.number_input("فالوئر اینستاگرام", 0, 10_000_000, int(followers), 1_000, key="v11_followers")
            team_size = st.number_input("اندازه تیم محتوا", 1, 100, int(team_size), 1, key="v11_team_size")
            customers = st.number_input("مشتری مصنوعی", 500, 50_000, int(customers), 500, key="v11_customers")
            months = st.slider("ماه‌های تاریخچه مصنوعی", 3, 36, int(months), 1, key="v11_months")
            seed = st.number_input("Seed", 1, 999_999, int(seed), 1, key="v11_seed")
        if st.sidebar.button("اجرای تحلیل کامل", use_container_width=True, key="v11_run_analysis"):
            stages = ["بارگذاری", "اعتبارسنجی", "محاسبه KPI", "تحلیل روند", "مدل‌ها", "آماده"]
            progress = st.sidebar.progress(0); status = st.sidebar.empty()
            for i, stage in enumerate(stages, start=1):
                status.caption(stage); progress.progress(int(i / len(stages) * 100)); time.sleep(0.01)
            status.success("تحلیل به‌روز شد")

    scenario = make_scenario(
        price_plan_a=float(price_a), price_plan_b=float(price_b), plan_a_share=float(share_a_pct) / 100,
        daily_phone_sales=int(phone), monthly_online_sales=int(online), lead_backlog=int(backlog),
        stories_per_day=int(stories), reels_per_day=int(reels), content_sales_per_day=float(content_sales),
        instagram_followers=int(followers), content_team_size=int(team_size), synthetic_customer_count=int(customers),
        history_months=int(months), sales_days_per_month=int(sales_days), seed=int(seed),
    )

    st.sidebar.markdown(
        '<div class="v11-sidebar-status"><span></span><div><b>Demo Mode</b><small>API / n8n / Database هنوز متصل نیست</small></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("V0.11 · مدیریت داخلی نیک + Growth Intelligence مشتریان")
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

    if CEO_OPS_AVAILABLE:
        _ensure_ceo_state()
        reports = department_report_schedule()
        classified_finance = classify_finance_transactions(st.session_state.ceo_finance_transactions, st.session_state.ceo_finance_threshold)
        fin_summary = finance_automation_summary(classified_finance)
        inbox = ceo_inbox(int(sm["lead_backlog"]), st.session_state.ceo_tasks, reports, fin_summary)
        task_stats = task_summary(st.session_state.ceo_tasks)
        report_stats = reporting_summary(reports)
        st.markdown(
            f'''<div class="ceo-personal-hero"><div class="ceo-personal-kicker">پنل اختصاصی مدیرعامل · {CEO_NAME}</div><div class="ceo-personal-title">صبح مدیریتی در یک قاب</div><div class="ceo-personal-copy">این نسخه برای کم‌کردن رفت‌وبرگشت گزارش، پیگیری دستی تسک و تصمیم‌گیری پراکنده طراحی شده است. موارد زیر از Task Engine و Reporting SLA نمونه اولیه جمع می‌شوند و تا اتصال منابع واقعی، فقط معماری فرایند را نشان می‌دهند.</div></div>''',
            unsafe_allow_html=True,
        )
        ctask = st.columns(4)
        ctask[0].metric("موارد Inbox", fa_num(len(inbox)))
        ctask[1].metric("تسک نیازمند پیگیری", fa_num(task_stats["needs_followup"]))
        ctask[2].metric("گزارش عقب‌افتاده Demo", fa_num(report_stats["overdue"]))
        ctask[3].metric("صف لید فعلی", fa_num(sm["lead_backlog"]), help="Baseline واقعی ثبت‌شده در 2026-08-31")

    mom_arrow = "↑" if mom > 0.002 else "↓" if mom < -0.002 else "—"
    anomaly_label = f"{fa_num(anomaly_count)} هشدار دمو" if anomaly_count else "بدون هشدار دمو"

    st.markdown(
        f'''<div class="ceo-command">
            <div>
                <div class="ceo-overline">مرکز فرمان مدیرعامل · Executive OS V0.11</div>
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
        fig.update_layout(annotations=[dict(text="ترکیب درآمد", x=.5, y=.5, font_size=14, showarrow=False, font_color="#173A5D")])
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)
        st.markdown(
            f'''<div class="glass-panel" style="margin-top:-4px"><div class="section-kicker">برداشت مدیریتی</div><div style="font-weight:800;color:#173A5D;margin-top:5px">طرح B با {pct(sm['share_b'])} از تعداد فروش، حدود {pct(plan_b_rev_share)} از درآمد مدل را می‌سازد.</div><div class="kpi-note">این نتیجه از اختلاف قیمت طرح‌ها به دست می‌آید و با تغییر Mix در Sidebar زنده تغییر می‌کند.</div></div>''',
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
            section_heading("نبض سازمان", "هفت واحد در یک نگاه", "امتیاز واحدها تا اتصال داده واقعی ترکیبی از Baseline و ورودی Demo است؛ برای تصمیم قطعی استفاده نشود.")
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
        "لایه ورود داده برای فایل CSV امروز و API، پایگاه داده و n8n در نسخه بعدی. هیچ داده‌ای در V0.10 به سیستم داخلی نیک متصل نیست.",
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
    integration_toolbar("فروش", "v09_sales_analytics")

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
    integration_toolbar("مارکتینگ", "v09_content")
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
    integration_toolbar("مارکتینگ", "v09_media")

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
            fig.add_trace(go.Scatter(x=timeline["second"], y=timeline["click_signal"], mode="lines", name="سیگنال کلیک / دمو", line=dict(color="#8A6FC1", width=2, dash="dot"), yaxis="y2"))
            for _, ev in events.iterrows():
                fig.add_vline(x=float(ev["second"]), line_width=1, line_dash="dot", line_color="rgba(59,98,134,.22)")
                fig.add_annotation(x=float(ev["second"]), y=0.98, yref="paper", text=str(ev["event"]), showarrow=False, font=dict(size=9, color="#4A6A89"), textangle=-90)
            fig.add_vline(x=selected_second, line_width=2, line_color="#3D78C5")
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
        "مدل‌ها ساده و قابل توضیح نگه داشته شده‌اند؛ هیچ خروجی پیش‌بینی یا ریزش در V0.10 در سطح عملیاتی نهایی نیست.",
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
            st.markdown("<div style='text-align:center;color:#356FAE;font-size:1.35rem;margin:4px 0'>↓</div>", unsafe_allow_html=True)

    st.write("")
    section_heading("ورودی آینده", "معماری اتصال بعدی")
    st.markdown(
        """
        <div class="glass-panel" style="text-align:center;line-height:2.1;font-weight:650">
            پایگاه داده / CRM / پنل نیک
            <span style="color:#356FAE;font-weight:900"> → </span>
            API فقط‌خواندنی
            <span style="color:#356FAE;font-weight:900"> → </span>
            n8n / خط لوله داده
            <span style="color:#356FAE;font-weight:900"> → </span>
            اعتبارسنجی
            <span style="color:#356FAE;font-weight:900"> → </span>
            موتور تحلیل
            <span style="color:#356FAE;font-weight:900"> → </span>
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
        fig.update_layout(annotations=[dict(text="Revenue Mix",x=.5,y=.5,showarrow=False,font_color="#173A5D",font_size=13)])
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
        "نقشه راه اتصال NIK Intelligence به منابع واقعی. در V0.10 فقط CSV و داده Demo فعال‌اند؛ دکمه‌های اتصال نمایشی‌اند و هیچ Credential ذخیره نمی‌شود.",
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
                    st.toast("Placeholder V0.10 — اتصال واقعی بعد از تعریف Schema، دسترسی و امنیت ساخته می‌شود.")

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


V101_VISUAL_POLISH_CSS = r"""
<style>
/* =========================================================
   V0.10.1 — FINAL VISUAL POLISH / ACCESSIBILITY LAYER
   This block MUST remain after every legacy CSS block.
   No business logic is changed here.
   ========================================================= */
:root{
    --nik101-bg:#DDEBFA;
    --nik101-bg-soft:#EAF3FC;
    --nik101-surface:rgba(248,252,255,.89);
    --nik101-surface-2:rgba(232,243,255,.84);
    --nik101-surface-3:rgba(218,234,251,.78);
    --nik101-edge:rgba(62,105,149,.18);
    --nik101-edge-soft:rgba(255,255,255,.88);
    --nik101-text:#0D2A45;
    --nik101-text-2:#244A6C;
    --nik101-muted:#45627C;
    --nik101-soft:#5A738B;
    --nik101-blue:#2F6FAE;
    --nik101-blue-2:#477FBB;
    --nik101-accent:#ADCBFF;
    --nik101-success:#146747;
    --nik101-warning:#77510A;
    --nik101-danger:#8A2E42;
}

/* Canvas: brighter than dark mode, but no washed-out white-on-white surfaces. */
.stApp{
    color:var(--nik101-text)!important;
    background:
        radial-gradient(900px 560px at 88% -8%,rgba(173,203,255,.70),transparent 67%),
        radial-gradient(760px 560px at 4% 30%,rgba(202,224,249,.74),transparent 70%),
        radial-gradient(620px 460px at 56% 94%,rgba(247,251,255,.72),transparent 72%),
        linear-gradient(145deg,#EFF6FD 0%,#E4EFFA 38%,#D8E8F8 68%,#CFE1F5 100%)!important;
    background-attachment:fixed!important;
}

/* Sidebar: one clearly separated glass rail. */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,rgba(237,246,255,.97),rgba(218,234,250,.94))!important;
    border-left:1px solid rgba(59,104,149,.18)!important;
    box-shadow:-22px 0 64px rgba(49,89,129,.13)!important;
    -webkit-backdrop-filter:blur(34px) saturate(150%)!important;
    backdrop-filter:blur(34px) saturate(150%)!important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{
    color:#244A6A!important;
    -webkit-text-fill-color:#244A6A!important;
}
[data-testid="stSidebar"] hr{border-color:rgba(57,101,145,.15)!important}

/* Streamlit navigation and controls. */
[data-testid="stSidebar"] [data-baseweb="select"]>div,
[data-testid="stSelectbox"] [data-baseweb="select"]>div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
textarea{
    color:#123957!important;
    -webkit-text-fill-color:#123957!important;
    background:rgba(248,252,255,.94)!important;
    border-color:rgba(54,101,148,.19)!important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.94)!important;
}
[data-baseweb="popover"],
[data-baseweb="popover"]>div,
[role="listbox"]{
    background:#F5FAFF!important;
    color:#123957!important;
}
[role="option"]{color:#173E5E!important;background:#F5FAFF!important}
[role="option"]:hover,[role="option"][aria-selected="true"]{background:#DCEBFA!important;color:#0C3150!important}

/* Radio navigation: selected item reads as a deliberate menu state. */
[data-testid="stSidebar"] [role="radiogroup"] label{
    border-radius:13px!important;
    padding:.30rem .42rem!important;
    transition:background .18s ease,box-shadow .18s ease!important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
    background:linear-gradient(135deg,rgba(255,255,255,.92),rgba(190,216,245,.75))!important;
    box-shadow:inset 0 1px 0 #fff,0 8px 22px rgba(61,103,148,.10)!important;
}
[data-testid="stSidebar"] [role="radiogroup"] label p{color:#284E70!important;font-weight:720!important}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#0E3659!important;font-weight:900!important}

/* Global readable typography. */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p,
.stMarkdown,.stHeader,.stSubheader{
    color:var(--nik101-text-2)!important;
}
h1,h2,h3,h4,h5,h6{color:var(--nik101-text)!important;-webkit-text-fill-color:var(--nik101-text)!important}
a{color:#1E619D!important;text-decoration-color:rgba(30,97,157,.34)!important}
small,.stCaption,[data-testid="stCaptionContainer"]{color:var(--nik101-muted)!important}

/* Unified liquid-glass surfaces across ALL generations of the app. */
.hero-glass,.ceo-command,.glass-kpi,.glass-panel,.command-metric,.priority-card,.leverage-card,
.v06-brief,.ceo-personal-hero,.team-card,.ceo-inbox-card,.auto-ledger,.lead-routing-hero,
.connection-card,.v09-section-hero,.v09-integration,.customer-product-hero,.customer-health-main,
.customer-health-cell,.growth-opportunity,.action-story,.plan-card,.customer-automation-row,
.org-card,.management-hero,.auto-rule,.campaign-result,.flow-strip,.role-box,.report-row,
.money-chip,.trust-cell,.gap-row,.ops-step,[data-testid="stMetric"],div[data-testid="stExpander"]{
    background:linear-gradient(150deg,var(--nik101-surface),var(--nik101-surface-2))!important;
    border:1px solid rgba(255,255,255,.94)!important;
    outline:1px solid rgba(66,111,157,.08)!important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.98),
        inset 0 -1px 0 rgba(92,133,177,.05),
        0 16px 44px rgba(55,96,138,.11)!important;
    -webkit-backdrop-filter:blur(28px) saturate(145%)!important;
    backdrop-filter:blur(28px) saturate(145%)!important;
}

/* Deliberately stronger cards for primary executive information. */
.hero-glass,.ceo-command,.ceo-personal-hero,.customer-product-hero,.management-hero{
    background:
        radial-gradient(260px 150px at 92% 0%,rgba(173,203,255,.52),transparent 72%),
        linear-gradient(145deg,rgba(250,253,255,.94),rgba(220,236,253,.86))!important;
    border-color:rgba(255,255,255,.98)!important;
}

/* All legacy dark-theme headline classes. */
.hero-title,.kpi-value,.section-title,.ceo-status-title,.ceo-command-value,.command-metric-value,
.priority-title,.money-chip-value,.leverage-title,.leverage-value,.gap-title,.v06-brief-title,
.trust-value,.sim-value,.connection-name,.ceo-personal-title,.ceo-inbox-title,.team-name,
.auto-ledger-title,.lead-routing-value,.report-name,.v09-hero-title,.org-name,.org-score,
.management-hero-title,.auto-rule-name,.campaign-title,.role-value,.customer-product-title,
.customer-health-value,.growth-opportunity-title,.growth-opportunity-value,.plan-name,.plan-price,
.action-story-text b,.customer-automation-row b{
    color:var(--nik101-text)!important;
    -webkit-text-fill-color:var(--nik101-text)!important;
}

/* Secondary copy: never use pale gray on an ice surface. */
.hero-subtitle,.kpi-label,.kpi-note,.section-subtitle,.ceo-status-copy,.ceo-command-label,
.ceo-command-foot,.command-metric-label,.command-metric-note,.priority-text,.priority-source,
.money-chip-label,.leverage-note,.gap-text,.v06-brief-copy,.trust-label,.trust-note,.sim-label,
.connection-copy,.connection-fresh,.ceo-personal-copy,.ceo-inbox-meta,.team-role,
.auto-ledger-copy,.lead-routing-label,.report-sub,.v09-hero-copy,.v09-subnav,.media-meta,
.org-grid-note,.management-hero-copy,.auto-rule-copy,.auto-rule-action,.campaign-copy,
.role-label,.role-copy,.customer-product-copy,.customer-health-label,.customer-health-note,
.growth-opportunity-copy,.growth-opportunity-meta,.plan-period,.plan-best,.plan-feature,
.action-story-text,.customer-automation-row{
    color:var(--nik101-muted)!important;
    -webkit-text-fill-color:var(--nik101-muted)!important;
}

/* Minimum size for explanatory microcopy — tiny copy was a readability issue. */
.kpi-note,.command-metric-note,.priority-source,.gap-text,.leverage-note,.trust-note,.connection-fresh,
.ceo-inbox-meta,.team-role,.report-sub,.org-grid-note,.auto-rule-copy,.auto-rule-action,.role-label,
.role-copy,.customer-health-note,.growth-opportunity-copy,.growth-opportunity-meta,.plan-period,
.plan-best,.plan-feature,.customer-source-note,.customer-automation-row{
    font-size:max(.68rem,11px)!important;
    line-height:1.72!important;
}

/* Kicker / accent text gets a darker brand blue on light surfaces. */
.eyebrow,.section-kicker,.ceo-overline,.v06-brief-kicker,.ceo-personal-kicker,.leverage-kicker,
.ceo-inbox-action,.v09-hero-kicker,.v09-integration-title,.management-hero-kicker,
.campaign-status,.customer-product-kicker,.plan-eyebrow,.growth-opportunity-priority{
    color:var(--nik101-blue)!important;
    -webkit-text-fill-color:var(--nik101-blue)!important;
}

/* Icon system: higher contrast + cleaner liquid glass. */
.kpi-icon,.v09-icon-shell,.org-icon,.gap-index{
    color:#1E5A91!important;
    background:
        radial-gradient(circle at 28% 20%,rgba(255,255,255,.96),transparent 38%),
        linear-gradient(145deg,rgba(247,252,255,.98),rgba(173,203,255,.68))!important;
    border:1px solid rgba(70,118,167,.20)!important;
    box-shadow:inset 0 1px 0 #fff,0 8px 22px rgba(55,104,153,.13)!important;
}
.kpi-icon svg,.v09-icon-shell svg,.org-icon svg,.gap-index svg{
    stroke:#225F98!important;
    color:#225F98!important;
    opacity:1!important;
}

/* Status pills use dark semantic text, not pastel text. */
.src-real,.org-ok,.conf-high,.state-on,.customer-status-ready{color:#126344!important;background:rgba(32,165,105,.12)!important;border-color:rgba(29,135,88,.20)!important}
.src-derived,.trend-flat,.sev-info,.team-chip,.audit-pill,.v09-connect-pill,.campaign-name,.flow-node{color:#275B8E!important;background:rgba(89,144,203,.12)!important;border-color:rgba(73,123,174,.18)!important}
.src-estimated,.org-watch,.conf-med,.auto-on,.gap-status,.customer-status-suggest,.sev-med{color:#73500C!important;background:rgba(232,171,57,.14)!important;border-color:rgba(187,133,30,.22)!important}
.src-synthetic{color:#5B438E!important;background:rgba(126,94,191,.11)!important;border-color:rgba(126,94,191,.19)!important}
.org-action,.conf-low,.sev-high,.trend-down{color:#862F43!important;background:rgba(215,79,105,.10)!important;border-color:rgba(187,64,89,.18)!important}
.auto-off,.state-off{color:#4F657B!important;background:rgba(105,132,158,.09)!important;border-color:rgba(80,111,142,.16)!important}

/* V0.7 organization components were injected after the old light theme. Fix them here. */
.org-pill:not(.org-ok):not(.org-watch):not(.org-action){color:#3C5F7F!important;background:rgba(230,241,252,.86)!important;border-color:rgba(77,121,165,.14)!important}
.flow-arrow{color:#426B91!important}
.auto-rule-active{
    background:linear-gradient(145deg,rgba(255,248,230,.90),rgba(236,245,255,.88))!important;
    border-color:rgba(192,139,37,.19)!important;
}

/* Inline legacy text inside glass cards. CSS !important intentionally beats inline non-important colors. */
.glass-panel div[style*="color:#F8FBFF"],
.org-pill[style*="color:#BACBDB"]{color:#214A6D!important;-webkit-text-fill-color:#214A6D!important}

/* Buttons: distinct text and surface. */
.stButton>button,.stDownloadButton>button{
    color:#123E64!important;
    -webkit-text-fill-color:#123E64!important;
    background:linear-gradient(145deg,rgba(252,254,255,.98),rgba(205,226,248,.92))!important;
    border:1px solid rgba(56,105,153,.21)!important;
    box-shadow:inset 0 1px 0 #fff,0 9px 24px rgba(55,99,145,.11)!important;
    font-weight:800!important;
}
.stButton>button[kind="primary"]{
    color:#fff!important;
    -webkit-text-fill-color:#fff!important;
    background:linear-gradient(135deg,#4E87C8,#2E69A8)!important;
    border-color:rgba(255,255,255,.54)!important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.54),0 12px 28px rgba(39,91,148,.25)!important;
}
.stButton>button:disabled,.stDownloadButton>button:disabled{opacity:.66!important;color:#506A82!important;-webkit-text-fill-color:#506A82!important}

/* Tabs and expanders. */
[data-testid="stTabs"] [role="tab"]{color:#45647F!important;font-weight:800!important}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:#123D62!important;font-weight:900!important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background-color:#3976B4!important;height:3px!important;border-radius:99px!important}
details summary,div[data-testid="stExpander"] summary p{color:#234A6C!important;font-weight:800!important}

/* Metrics, alerts, tables and helper UI. */
[data-testid="stMetricLabel"] p{color:#45627C!important;font-weight:760!important}
[data-testid="stMetricValue"]{color:#0D2E4D!important}
[data-testid="stAlert"]{background:rgba(240,248,255,.92)!important;border:1px solid rgba(65,111,157,.16)!important;color:#214866!important}
[data-testid="stAlert"] p,[data-testid="stAlert"] div{color:#214866!important}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid rgba(63,108,153,.14)!important;border-radius:17px!important;overflow:hidden!important;background:rgba(247,251,255,.82)!important}
code,pre{color:#173C5C!important;background:rgba(227,239,252,.86)!important;border-color:rgba(72,117,162,.14)!important}

/* Sliders / progress / checkbox accents. */
[data-baseweb="slider"] div[role="slider"]{background:#2F6FAE!important;border-color:#fff!important}
[data-testid="stProgress"]>div>div>div{background:linear-gradient(90deg,#77A8DF,#2F6FAE)!important}

/* Customer product specific surfaces: keep premium but make copy one level darker. */
.customer-product-hero,.plan-card-featured,.customer-health-main{
    background:
        radial-gradient(230px 150px at 94% 0%,rgba(173,203,255,.48),transparent 72%),
        linear-gradient(145deg,rgba(250,253,255,.96),rgba(212,232,252,.88))!important;
}
.customer-source-note{color:#54427B!important;background:rgba(126,94,191,.08)!important;border-color:rgba(102,73,163,.15)!important}

/* Hover = subtle precision, not flashy movement. */
@media(hover:hover) and (pointer:fine){
    .glass-kpi:hover,.command-metric:hover,.priority-card:hover,.team-card:hover,.org-card:hover,
    .growth-opportunity:hover,.plan-card:hover,.connection-card:hover{
        transform:translateY(-1px)!important;
        border-color:rgba(55,106,157,.22)!important;
        box-shadow:inset 0 1px 0 #fff,0 20px 48px rgba(55,96,138,.14)!important;
    }
    .stButton>button:hover,.stDownloadButton>button:hover{
        border-color:rgba(45,98,151,.30)!important;
        box-shadow:inset 0 1px 0 #fff,0 12px 28px rgba(55,99,145,.14)!important;
    }
}

/* Mobile: preserve contrast and avoid microscopic helper text. */
@media(max-width:720px){
    .block-container{padding-left:.75rem!important;padding-right:.75rem!important}
    .customer-product-title,.ceo-personal-title,.ceo-status-title{font-size:1.30rem!important}
    .customer-health-note,.growth-opportunity-copy,.plan-feature,.role-copy,.auto-rule-copy{font-size:.72rem!important}
}
</style>
"""
st.markdown(V101_VISUAL_POLISH_CSS, unsafe_allow_html=True)


V11_EXECUTIVE_UX_CSS = r"""
<style>
/* =========================================================
   V0.11 — EXECUTIVE UX / PRODUCT POLISH / PERFORMANCE PASS
   Last layer wins. Business logic stays untouched.
   ========================================================= */
:root{
    --v11-ink:#0B2238;
    --v11-ink-2:#24435F;
    --v11-muted:#557087;
    --v11-line:rgba(58,102,145,.16);
    --v11-glass:rgba(247,251,255,.78);
    --v11-glass-strong:rgba(249,252,255,.91);
    --v11-glass-soft:rgba(225,238,251,.68);
    --v11-blue:#2C6DA8;
    --v11-blue-deep:#174A76;
    --v11-accent:#ADCBFF;
}

/* Calmer canvas: premium light, not washed-out white. */
.stApp{
    background:
        radial-gradient(900px 560px at 92% -10%,rgba(173,203,255,.58),transparent 68%),
        radial-gradient(760px 520px at -6% 25%,rgba(205,225,247,.72),transparent 70%),
        linear-gradient(145deg,#F2F7FC 0%,#E9F1F9 46%,#DEEAF6 100%)!important;
    color:var(--v11-ink)!important;
}
.block-container{max-width:1440px!important;padding-top:1rem!important;padding-bottom:2.2rem!important}

/* Sidebar becomes a navigation rail, not a control panel. */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,rgba(246,250,254,.96),rgba(227,238,249,.94))!important;
    border-left:1px solid var(--v11-line)!important;
    box-shadow:-18px 0 55px rgba(39,77,113,.10)!important;
}
.v11-sidebar-brand{padding:2px 3px 8px}.v11-sidebar-kicker{font-size:.64rem;letter-spacing:.10em;color:#51779B;font-weight:900}.v11-sidebar-title{font-size:1.04rem;color:#123956;font-weight:900;margin-top:4px}.v11-sidebar-copy{font-size:.67rem;color:#637C93;margin-top:3px;line-height:1.6}.v11-side-section{font-size:.63rem;font-weight:900;color:#6B8297;letter-spacing:.05em;margin:14px 2px 6px}.v11-sidebar-status{display:flex;align-items:center;gap:9px;margin-top:14px;padding:10px 11px;border-radius:15px;background:rgba(255,255,255,.58);border:1px solid rgba(67,111,153,.12)}.v11-sidebar-status>span{width:8px;height:8px;border-radius:50%;background:#5A92CB;box-shadow:0 0 0 4px rgba(90,146,203,.12)}.v11-sidebar-status b{display:block;color:#244A69;font-size:.70rem}.v11-sidebar-status small{display:block;color:#71869A!important;font-size:.60rem;margin-top:1px}

/* Compact page header. Page context is the hero, not the brand name. */
.v11-page-head{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;padding:21px 23px;margin:3px 0 9px;border-radius:25px;background:radial-gradient(280px 160px at 95% 0%,rgba(173,203,255,.42),transparent 72%),linear-gradient(145deg,rgba(251,253,255,.90),rgba(223,237,251,.76));border:1px solid rgba(255,255,255,.95);outline:1px solid rgba(63,107,150,.07);box-shadow:inset 0 1px 0 #fff,0 16px 40px rgba(52,92,129,.09);backdrop-filter:blur(28px) saturate(145%)}
.v11-page-copy{min-width:0;max-width:930px}.v11-page-kicker{display:flex;align-items:center;gap:8px;font-size:.65rem;font-weight:900;letter-spacing:.08em;color:#47729A}.v11-live-dot{width:7px;height:7px;border-radius:50%;background:#3A7CB8;box-shadow:0 0 0 4px rgba(58,124,184,.10)}.v11-page-title{font-size:clamp(1.55rem,2.6vw,2.35rem);line-height:1.2;font-weight:950;color:var(--v11-ink);letter-spacing:-.035em;margin-top:7px}.v11-page-subtitle{max-width:840px;color:var(--v11-muted);font-size:.86rem;line-height:1.95;margin-top:7px}.v11-context-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}.v11-context-chip{display:inline-flex;padding:5px 9px;border-radius:999px;font-size:.61rem;font-weight:850;color:#254F72;background:rgba(173,203,255,.26);border:1px solid rgba(65,111,156,.10)}.v11-context-soft{background:rgba(255,255,255,.52);color:#647B90}.v11-page-brand{display:flex;flex-direction:column;align-items:flex-end;gap:6px;color:#758A9C;font-size:.58rem;white-space:nowrap}.v11-head-logo{width:138px;height:auto;object-fit:contain;filter:drop-shadow(0 8px 18px rgba(52,88,121,.10))}
.v11-connection-line{min-height:38px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:6px 10px;color:#61798F;font-size:.66rem}.v11-connection-line b{color:#285373}.v11-connection-dot{width:7px;height:7px;border-radius:50%;background:#6D9FD0;box-shadow:0 0 0 4px rgba(109,159,208,.10)}

/* Strong hierarchy: primary, metric, utility. Not every card gets the same glass/shadow. */
.hero-glass,.ceo-command,.ceo-personal-hero,.customer-product-hero,.management-hero,.v09-section-hero{
    background:radial-gradient(240px 140px at 95% 0%,rgba(173,203,255,.40),transparent 74%),linear-gradient(145deg,rgba(250,253,255,.92),rgba(221,235,249,.78))!important;
    border:1px solid rgba(255,255,255,.96)!important;
    outline:1px solid rgba(57,100,142,.07)!important;
    box-shadow:inset 0 1px 0 #fff,0 16px 40px rgba(49,88,125,.09)!important;
}
.glass-kpi,.command-metric,[data-testid="stMetric"],.priority-card,.growth-opportunity,.plan-card,.team-card,.org-card{
    background:linear-gradient(150deg,rgba(249,252,255,.84),rgba(229,240,251,.73))!important;
    border:1px solid rgba(255,255,255,.92)!important;
    outline:1px solid rgba(62,105,147,.06)!important;
    box-shadow:inset 0 1px 0 #fff,0 10px 28px rgba(52,91,127,.08)!important;
}
.trust-cell,.report-row,.customer-automation-row,.flow-strip,.role-box,.gap-row,.ops-step,.money-chip,.connection-card,.v09-integration,div[data-testid="stExpander"]{
    background:rgba(244,249,254,.62)!important;
    border:1px solid rgba(65,108,150,.11)!important;
    outline:0!important;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.82)!important;
}

/* Buttons: unmistakable hierarchy and readable labels. */
.stButton>button,.stDownloadButton>button{min-height:38px!important;border-radius:13px!important;color:#235070!important;background:rgba(248,252,255,.78)!important;border:1px solid rgba(57,103,148,.16)!important;box-shadow:inset 0 1px 0 #fff,0 6px 16px rgba(50,91,130,.06)!important;font-weight:850!important}
.stButton>button[kind="primary"]{color:#fff!important;background:linear-gradient(145deg,#3F7DB8,#285E91)!important;border-color:rgba(38,89,139,.34)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 9px 22px rgba(45,93,139,.18)!important}
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid rgba(69,121,174,.18)!important;outline-offset:2px!important}

/* Typography: stronger title rhythm, calmer helper copy. */
.section-title{font-size:1.25rem!important;line-height:1.45!important;letter-spacing:-.018em!important}.section-subtitle{max-width:900px!important;font-size:.77rem!important;line-height:1.85!important}.section-kicker{font-size:.62rem!important;letter-spacing:.06em!important}.kpi-value,.command-metric-value{letter-spacing:-.025em!important}.kpi-note,.command-metric-note{color:#60788E!important}

/* Cleaner controls and denser tables for executive use. */
[data-testid="stDataFrame"],[data-testid="stTable"]{background:rgba(249,252,255,.76)!important;box-shadow:0 8px 24px rgba(49,88,125,.06)!important}.stTabs [role="tablist"]{gap:4px!important}.stTabs [role="tab"]{border-radius:10px 10px 0 0!important;padding-left:12px!important;padding-right:12px!important}

/* Render heavy lower sections more efficiently in modern browsers. */
.stPlotlyChart,.glass-panel,.action-story,.customer-health-main{content-visibility:auto;contain-intrinsic-size:360px}

.v11-error-card{margin-top:8px;padding:14px 16px;border-radius:16px;color:#5D3440;background:rgba(255,244,247,.78);border:1px solid rgba(171,73,97,.13);font-size:.76rem;line-height:1.8}.v11-error-card span{color:#7D5A63}.v11-footer{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:22px;padding-top:12px;border-top:1px solid rgba(65,107,148,.10);color:#6A8196;font-size:.62rem}

@media(hover:hover) and (pointer:fine){
    .glass-kpi:hover,.command-metric:hover,.priority-card:hover,.growth-opportunity:hover,.plan-card:hover{transform:translateY(-1px)!important;box-shadow:inset 0 1px 0 #fff,0 14px 34px rgba(51,92,130,.10)!important}
}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important;animation:none!important}}
@media(max-width:800px){.v11-page-head{flex-direction:column}.v11-page-brand{align-items:flex-start}.v11-head-logo{width:122px}.v11-page-title{font-size:1.45rem}.block-container{padding-left:.75rem!important;padding-right:.75rem!important}}
</style>
"""
st.markdown(V11_EXECUTIVE_UX_CSS, unsafe_allow_html=True)

def _management_ready() -> bool:
    if MANAGEMENT_ENGINE_AVAILABLE:
        return True
    page_header("لایه مدیریتی در دسترس نیست", "هسته Data Science قبلی همچنان فعال است؛ فقط فایل management_engine.py در Deploy فعلی پیدا نشده است.")
    st.error("برای فعال شدن صفحات مدیریتی، فایل management_engine.py نسخه V0.8 را کنار app.py آپلود کن. این خطا صفحات قبلی را از کار نمی‌اندازد.")
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
    st.session_state.setdefault("mgmt_run_support", "روزانه")
    st.session_state.setdefault("mgmt_run_development", "روزانه")


def _management_overrides() -> dict:
    if not MANAGEMENT_ENGINE_AVAILABLE:
        return {}
    _ensure_management_state()
    return {key: st.session_state.get(f"mgmt_{key}", value) for key, value in MANAGEMENT_DEMO_DEFAULTS.items()}


def _ceo_ops_ready(show_error: bool = True) -> bool:
    if CEO_OPS_AVAILABLE:
        return True
    if show_error:
        st.warning("لایه V0.8 CEO Ops در Deploy فعلی پیدا نشده است؛ صفحات V0.7 همچنان فعال‌اند.")
        if CEO_OPS_ERROR:
            st.caption(f"جزئیات Import: {CEO_OPS_ERROR}")
    return False


def _ensure_ceo_state():
    if not CEO_OPS_AVAILABLE:
        return
    if "ceo_tasks" not in st.session_state:
        st.session_state.ceo_tasks = seed_ceo_tasks()
    if "ceo_sales_agents" not in st.session_state:
        st.session_state.ceo_sales_agents = DEFAULT_SALES_AGENTS.copy()
    if "ceo_finance_threshold" not in st.session_state:
        st.session_state.ceo_finance_threshold = 50_000_000
    if "ceo_finance_transactions" not in st.session_state:
        st.session_state.ceo_finance_transactions = generate_demo_transactions()
    if "ceo_audit_log" not in st.session_state:
        st.session_state.ceo_audit_log = pd.DataFrame(columns=["زمان", "کاربر", "اقدام", "بخش", "جزئیات"])


def _append_audit(action: str, department: str, detail: str = "", actor: str = CEO_NAME):
    if not CEO_OPS_AVAILABLE:
        return
    _ensure_ceo_state()
    event = pd.DataFrame([audit_event(actor, action, department, detail)])
    st.session_state.ceo_audit_log = pd.concat([st.session_state.ceo_audit_log, event], ignore_index=True)


def _add_ceo_task(department: str, title: str, assignee: str, priority: str = "متوسط", kpi: str = "—", source: str = "مدیرعامل", due_hours: int = 24, followup_hours: int = 24, note: str = ""):
    if not CEO_OPS_AVAILABLE or not title.strip():
        return False
    _ensure_ceo_state()
    row = new_task_row(department, title, assignee, priority, kpi, source, CEO_NAME, due_hours, followup_hours, note)
    st.session_state.ceo_tasks = pd.concat([st.session_state.ceo_tasks, row], ignore_index=True)
    _append_audit("ایجاد تسک", department, title)
    return True


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
                f'''<div class="org-card"><div class="org-top"><div class="org-name">{row['department_name']}</div><div class="org-icon">{row['icon']}</div></div><div class="org-score">{fa_num(row['score'],1)}<span style="font-size:.72rem;color:#4E6A83;font-weight:700"> / ۱۰۰</span></div><div class="org-meta"><span class="org-pill {css}">{row['status']}</span><span class="org-pill" style="color:#BACBDB">{fa_num(row['attention_count'])} KPI نیازمند پیگیری</span></div><div class="org-grid-note">امتیاز نمونه اولیه است و تا اتصال داده واقعی واحد، Health Score قطعی محسوب نمی‌شود.</div></div>''',
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
        f'''<div class="management-hero"><div class="management-hero-kicker">ORGANIZATION PULSE · V0.8</div><div class="management-hero-title">سلامت عملیاتی نمونه اولیه: {fa_num(org['score'],1)} از ۱۰۰ · {org['status']}</div><div class="management-hero-copy">این امتیاز ترکیبی از KPIهای واقعی/محاسبه‌شده و تعدادی ورودی Demo است. هدف فعلی، ساخت معماری مدیریت سازمان است؛ با اتصال هر واحد، KPIهای Demo همان بخش با داده واقعی جایگزین می‌شوند.</div></div>''',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("امتیاز سازمان", fa_num(org["score"], 1))
    c2.metric("واحد نیازمند توجه", fa_num(org["departments_needing_attention"]))
    c3.metric("قانون اتوماسیون فعال", fa_num(len(active_rules)))
    c4.metric("واحدهای متصل به پنل", fa_num(int(org.get("department_count", 7))) + " واحد", help="اتصال فعلی مفهومی/Demo است؛ نه اتصال Database واقعی.")

    section_heading("نبض واحدها", "هفت بخش اصلی شرکت")
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

    section_heading("تابلوی اقدام", "تسک‌های مدیریتی", "در V0.8 تغییرات این جدول در Session نگه داشته می‌شود؛ برای استفاده واقعی باید به Database/Task Manager وصل شود.")
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
    integration_toolbar("تولید و QC", "v09_production")

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
    section_heading("بازه انجام خودکار", "هر واحد چند وقت یک‌بار بررسی شود؟", "در V0.8 این Schedule ذخیره دائمی نمی‌شود؛ طراحی Workflow آینده است.")
    cols = st.columns(3)
    with cols[0]: st.selectbox("حسابداری", options, key="mgmt_run_finance")
    with cols[1]: st.selectbox("فروش", options, key="mgmt_run_sales")
    with cols[2]: st.selectbox("QC دستگاه", options, key="mgmt_run_qc")
    cols2 = st.columns(4)
    with cols2[0]: st.selectbox("مارکتینگ", options, key="mgmt_run_marketing")
    with cols2[1]: st.selectbox("منابع انسانی", options, key="mgmt_run_hr")
    with cols2[2]: st.selectbox("پشتیبانی", options, key="mgmt_run_support")
    with cols2[3]: st.selectbox("برنامه‌نویسی", options, key="mgmt_run_development")

    trigger_ratio = st.slider("اگر درآمد از چند درصد Target پایین‌تر رفت Trigger شود؟", 50, 110, 90, 5, key="v08_revenue_trigger") / 100
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

    if st.button("ساخت Workflow در n8n (Placeholder)", use_container_width=True, type="primary", key="v08_n8n_placeholder"):
        st.toast("در V0.8 فقط قرارداد Workflow طراحی شده است. اتصال واقعی بعد از تعریف Credential، Schema و سطح دسترسی ساخته می‌شود.")
    st.caption("برای فاز واقعی: هر Rule باید Trigger، Schedule، Source، Action، Owner، Retry Policy و Audit Log داشته باشد.")


def access_control_page():
    if not _management_ready():
        return
    page_header("دسترسی و نقش‌ها", "طراحی دسترسی محدود برای مدیرعامل و سرپرستان واحدها؛ فعلاً فقط ماتریس پیشنهادی است و Authentication واقعی پیاده نشده است.")
    roles = ROLE_MATRIX["نقش"].tolist() if not ROLE_MATRIX.empty else ["مدیرعامل / مدیر سیستم"]
    role = st.selectbox("پروفایل نمایشی", roles, key="v08_role_preview")
    st.markdown(f'''<div class="management-hero"><div class="management-hero-kicker">ROLE PREVIEW</div><div class="management-hero-title">{role}</div><div class="management-hero-copy">در نسخه واقعی، هر سرپرست فقط داده و تسک‌های واحد خودش را می‌بیند یا ویرایش می‌کند. داده‌های مالی، تنظیمات اتصال و سطح دسترسی فقط برای نقش‌های مجاز باز می‌ماند.</div></div>''', unsafe_allow_html=True)
    st.dataframe(ROLE_MATRIX, use_container_width=True, hide_index=True)
    if CEO_OPS_AVAILABLE and ORGANIZATION_ROSTER is not None and not ORGANIZATION_ROSTER.empty:
        section_heading("دفتر سازمان", "افراد ثبت‌شده در V0.8", "برای هر سرپرست در فاز Authentication فقط Scope واحد خودش باز می‌شود.")
        st.dataframe(ORGANIZATION_ROSTER[["department_name", "name", "role"]].rename(columns={"department_name":"بخش", "name":"نام", "role":"نقش"}), use_container_width=True, hide_index=True)
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
    st.warning("در V0.8 هیچ احراز هویت واقعی وجود ندارد؛ برای داده واقعی شرکت نباید تا قبل از اضافه شدن Authentication و مجوزها دسترسی عمومی داده شود.")


def _render_team_roster(department_key: str, columns_count: int = 3):
    if not CEO_OPS_AVAILABLE or ORGANIZATION_ROSTER is None or ORGANIZATION_ROSTER.empty:
        st.info("اعضای این واحد هنوز در دفتر سازمانی V0.8 ثبت نشده‌اند.")
        return
    team = ORGANIZATION_ROSTER[ORGANIZATION_ROSTER["department"] == department_key].copy()
    if team.empty:
        st.info("اسامی اعضای این واحد هنوز ثبت نشده است.")
        return
    cols = st.columns(columns_count)
    for idx, (_, person) in enumerate(team.iterrows()):
        with cols[idx % columns_count]:
            lead = " · مسئول/سرپرست" if bool(person.get("is_lead", False)) else ""
            st.markdown(
                f'''<div class="team-card"><div class="team-name">{person['name']}</div><div class="team-role">{person['role']}</div><span class="team-chip">{person['department_name']}{lead}</span></div>''',
                unsafe_allow_html=True,
            )


def _department_task_form(department_name: str, default_owner: str, key_prefix: str):
    if not _ceo_ops_ready(False):
        return
    with st.expander("ایجاد تسک برای این واحد", expanded=False):
        with st.form(f"{key_prefix}_task_form"):
            title = st.text_input("عنوان تسک", key=f"{key_prefix}_task_title")
            c1, c2, c3 = st.columns(3)
            with c1:
                owner = st.text_input("مسئول", value=default_owner, key=f"{key_prefix}_task_owner")
            with c2:
                priority = st.selectbox("اولویت", ["بالا", "متوسط", "پایین"], key=f"{key_prefix}_task_priority")
            with c3:
                due_hours = st.number_input("مهلت (ساعت)", min_value=1, max_value=720, value=24, step=6, key=f"{key_prefix}_task_due")
            kpi_name = st.text_input("KPI مرتبط", value="—", key=f"{key_prefix}_task_kpi")
            note = st.text_area("یادداشت / خروجی مورد انتظار", key=f"{key_prefix}_task_note")
            submitted = st.form_submit_button("ایجاد تسک و ورود به پیگیری", use_container_width=True, type="primary")
            if submitted and title.strip():
                if _add_ceo_task(department_name, title, owner, priority, kpi_name, "مدیرعامل", int(due_hours), 24, note):
                    st.success("تسک ایجاد شد و وارد صف پیگیری مدیرعامل شد.")
                    st.rerun()


def _department_task_table(department_name: str):
    if not CEO_OPS_AVAILABLE:
        return
    _ensure_ceo_state()
    frame = task_followup_status(st.session_state.ceo_tasks)
    frame = frame[frame["بخش"] == department_name].copy()
    if frame.empty:
        st.info("برای این واحد هنوز تسکی در Task Engine ثبت نشده است.")
        return
    cols = ["شناسه", "عنوان", "مسئول", "اولویت", "وضعیت", "KPI", "موعد", "نیازمند پیگیری"]
    st.dataframe(frame[cols], use_container_width=True, hide_index=True)


def ceo_task_center_page(scenario, kpis):
    if not _ceo_ops_ready():
        return
    _ensure_ceo_state()
    sm = scenario_summary(scenario)
    reports = department_report_schedule()
    classified = classify_finance_transactions(st.session_state.ceo_finance_transactions, st.session_state.ceo_finance_threshold)
    fin = finance_automation_summary(classified)
    inbox = ceo_inbox(int(sm["lead_backlog"]), st.session_state.ceo_tasks, reports, fin)
    stats = task_summary(st.session_state.ceo_tasks)
    report_stats = reporting_summary(reports)

    page_header(
        "کارها و گزارش‌های کیوان میرزایی",
        "Inbox مدیریتی واحد: تسک‌های نیازمند پیگیری، گزارش‌های عقب‌افتاده، پیشنهاد ربات مدیریتی و مسیر ارسال یک‌کلیکی به واحدها.",
    )
    st.markdown(
        f'''<div class="ceo-personal-hero"><div class="ceo-personal-kicker">CEO OPERATIONS · {CEO_NAME}</div><div class="ceo-personal-title">هیچ تسک مهمی نباید بدون Owner، موعد و گزارش بماند.</div><div class="ceo-personal-copy">هدف این صفحه حذف پیگیری شفاهی و گزارش‌های پراکنده است. در نسخه عملیاتی، n8n یا Task API در زمان مشخص Reminder می‌فرستد، Update را ثبت می‌کند و فقط Exception را به مدیرعامل Escalate می‌کند.</div></div>''',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("تسک باز", fa_num(stats["open"]))
    c2.metric("نیازمند پیگیری", fa_num(stats["needs_followup"]))
    c3.metric("گزارش عقب‌افتاده Demo", fa_num(report_stats["overdue"]))
    c4.metric("موارد Inbox", fa_num(len(inbox)))

    section_heading("Inbox مدیرعامل", "چه چیزی الان نیازمند توجه است؟", "Baseline واقعی از Demo جدا نگه داشته شده است.")
    if inbox.empty:
        st.success("موردی برای پیگیری در موتور فعلی پیدا نشد.")
    else:
        for idx, (_, item) in enumerate(inbox.iterrows()):
            sev = "sev-high" if item["شدت"] == "بالا" else "sev-med" if item["شدت"] == "متوسط" else "sev-low"
            st.markdown(
                f'''<div class="ceo-inbox-card"><div class="ceo-inbox-top"><div class="ceo-inbox-title">{item['عنوان']}</div><span class="source-tag {sev}">{item['شدت']}</span></div><div class="ceo-inbox-meta">{item['بخش']} · {item['نوع']} · {item['منبع']}</div><div class="ceo-inbox-action">اقدام پیشنهادی: {item['اقدام']}</div></div>''',
                unsafe_allow_html=True,
            )

    section_heading("گزارش‌دهی", "SLA گزارش واحدها", "اعداد این جدول Demo هستند؛ معماری برای حل مشکل گزارش‌دهی ناقص طراحی شده است.")
    reports_show = reports.copy()
    reports_show["آخرین گزارش"] = pd.to_datetime(reports_show["آخرین گزارش"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    reports_show["موعد بعدی"] = pd.to_datetime(reports_show["موعد بعدی"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(reports_show[["بخش", "مسئول گزارش", "تناوب", "آخرین گزارش", "موعد بعدی", "وضعیت", "منبع"]], use_container_width=True, hide_index=True)
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        report_dept = st.selectbox("ارسال درخواست گزارش برای", reports["بخش"].tolist(), key="v08_report_request_dept")
    with rc2:
        if st.button("ثبت Reminder", use_container_width=True, key="v08_report_reminder"):
            _append_audit("درخواست گزارش", report_dept, "Reminder ثبت شد؛ اتصال واقعی پیام‌رسان/n8n در فاز بعد.")
            st.toast("Reminder در Audit Log ثبت شد. ارسال واقعی بعد از اتصال n8n فعال می‌شود.")

    section_heading("Task Engine", "تابلوی کار مدیرعامل", "در این Prototype ذخیره‌سازی Session است؛ نسخه واقعی باید Database + Audit Log داشته باشد.")
    edited = st.data_editor(st.session_state.ceo_tasks, use_container_width=True, hide_index=True, num_rows="dynamic", key="v08_ceo_task_editor")
    if isinstance(edited, pd.DataFrame):
        st.session_state.ceo_tasks = edited.copy()

    _department_task_form("مدیریت شرکت", CEO_NAME, "ceo_general")

    section_heading("ربات مدیریتی", "تسک‌های پیشنهادی بر اساس Rule", "ربات فعلی Rule-based است؛ پیشنهاد می‌دهد ولی بدون کلیک مدیر چیزی ارسال نمی‌شود.")
    suggestions = robot_task_suggestions(int(sm["lead_backlog"]), st.session_state.ceo_tasks, reports, fin)
    if suggestions.empty:
        st.success("پیشنهاد جدیدی وجود ندارد.")
    else:
        for idx, (_, suggestion) in enumerate(suggestions.iterrows()):
            a, b = st.columns([4, 1])
            with a:
                st.markdown(
                    f'''<div class="ceo-inbox-card"><div class="ceo-inbox-title">{suggestion['عنوان']}</div><div class="ceo-inbox-meta">{suggestion['بخش']} · {suggestion['مسئول']} · KPI: {suggestion['KPI']}</div><div class="ceo-inbox-action">چرا؟ {suggestion['دلیل']}</div></div>''',
                    unsafe_allow_html=True,
                )
            with b:
                if st.button("تبدیل به تسک", key=f"v08_robot_task_{idx}", use_container_width=True):
                    _add_ceo_task(str(suggestion["بخش"]), str(suggestion["عنوان"]), str(suggestion["مسئول"]), str(suggestion["اولویت"]), str(suggestion["KPI"]), "ربات مدیریتی / Rule-based", 24, 12, str(suggestion["دلیل"]))
                    st.toast("پیشنهاد به Task Engine اضافه شد.")
                    st.rerun()

    section_heading("ردپای مدیریتی", "Audit Log Session", "هر اقدام مهم در نسخه واقعی باید با User، زمان و Before/After ذخیره شود.")
    if st.session_state.ceo_audit_log.empty:
        st.caption("هنوز رویدادی در این Session ثبت نشده است.")
    else:
        st.dataframe(st.session_state.ceo_audit_log.sort_values("زمان", ascending=False).head(100), use_container_width=True, hide_index=True)


def it_workspace_page(scenario, kpis):
    if not _ceo_ops_ready():
        return
    _ensure_ceo_state()
    page_header(
        "اتاق عملیات IT",
        "برای موضوعاتی که این روزها بیشترین رفت‌وبرگشت مدیریتی دارند: آپدیت‌های جدید، تسک‌های برنامه‌نویسان، Blockerها و قالب ظاهری جدید سایت.",
    )
    integration_toolbar("فناوری اطلاعات", "v09_it_workspace")
    section_heading("تیم", "فناوری اطلاعات / برنامه‌نویسی")
    _render_team_roster("it", 3)

    overrides = _management_overrides() if MANAGEMENT_ENGINE_AVAILABLE else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("تکمیل Sprint · Demo", pct(float(overrides.get("sprint_completion", 0.78))))
    c2.metric("آمادگی Release · Demo", pct(float(overrides.get("release_readiness", 0.86))))
    c3.metric("باگ باز · Demo", fa_num(float(overrides.get("open_bugs", 14))))
    c4.metric("باگ Critical · Demo", fa_num(float(overrides.get("critical_bugs", 2))))

    section_heading("Daily Engineering Brief", "گزارش روزانه‌ای که کیوان میرزایی باید ببیند", "به‌جای مکالمه‌های پراکنده، هر روز فقط چهار پاسخ کوتاه.")
    st.markdown(
        '''<div class="ops-flow"><div class="ops-step"><b>۱. چه چیزی Release شد؟</b>نسخه / Feature / Deploy</div><div class="ops-step"><b>۲. چه چیزی در حال ساخت است؟</b>Owner + درصد پیشرفت</div><div class="ops-step"><b>۳. Blocker چیست؟</b>مانع فنی یا تصمیم مدیریتی</div><div class="ops-step"><b>۴. سایت</b>وضعیت قالب ظاهری و Acceptance</div><div class="ops-step"><b>۵. باگ مهم</b>Critical / ETA رفع</div><div class="ops-step"><b>۶. تصمیم لازم</b>فقط چیزی که نیاز به کیوان میرزایی دارد</div></div>''',
        unsafe_allow_html=True,
    )

    section_heading("تمرکز فعلی", "موضوعات ثبت‌شده برای پیگیری")
    focus = pd.DataFrame([
        ["آپدیت‌های جدید", "تیم IT", "Release Notes + وضعیت Deploy + Blocker", "روزانه"],
        ["تسک‌های برنامه‌نویسان", "تیم IT", "Owner + ETA + KPI + گزارش", "روزانه"],
        ["قالب ظاهری جدید سایت", "تیم IT", "Scope + Acceptance Criteria + قبل/بعد", "تا نهایی‌شدن"],
    ], columns=["موضوع", "Owner", "خروجی مورد انتظار", "تناوب گزارش"])
    st.dataframe(focus, use_container_width=True, hide_index=True)
    _department_task_table("فناوری اطلاعات / برنامه‌نویسی")
    _department_task_form("فناوری اطلاعات / برنامه‌نویسی", "تیم IT", "it")


def accounting_automation_page():
    if not _ceo_ops_ready():
        return
    _ensure_ceo_state()
    page_header(
        "اتوماسیون حسابداری",
        "هدف: دریافت ماشینی تراکنش، طبقه‌بندی، ثبت دوطرفه و تطبیق خودکار؛ انسان فقط برای استثنا، ابهام، مبلغ پرریسک و تأییدهای قانونی وارد شود.",
    )
    integration_toolbar("حسابداری", "v09_accounting")
    section_heading("تیم", "حسابداری")
    _render_team_roster("accounting", 2)

    st.markdown(
        '''<div class="auto-ledger"><div class="auto-ledger-title">اصل طراحی: Straight-Through Processing، نه حذف کورکورانه کنترل انسانی</div><div class="auto-ledger-copy">برای تراکنش‌های کم‌ریسک با شناسه یکتا، سند معتبر، کدینگ مشخص و تطبیق بانکی، ثبت می‌تواند کاملاً ماشینی باشد. تراکنش مبهم، تکراری، فاقد سند یا بالاتر از آستانه باید وارد Exception Queue شود. این مدل هم خطای انسانی را کم می‌کند و هم کنترل مالی را حفظ می‌کند.</div></div>''',
        unsafe_allow_html=True,
    )

    section_heading("معماری", "مسیر خودکار تراکنش", "برای اتصال واقعی، کدینگ رسمی حسابداری نیک و قوانین مالیاتی/تأیید باید از تیم حسابداری وارد شوند.")
    st.markdown(
        '''<div class="ops-flow"><div class="ops-step"><b>۱. دریافت</b>Bank / Gateway / Invoice / Payroll</div><div class="ops-step"><b>۲. Idempotency</b>جلوگیری از ثبت تکراری</div><div class="ops-step"><b>۳. Match</b>فاکتور + طرف حساب + مبلغ</div><div class="ops-step"><b>۴. Coding</b>تعیین حساب بدهکار/بستانکار</div><div class="ops-step"><b>۵. Reconcile</b>تطبیق با بانک/درگاه</div><div class="ops-step"><b>۶. Post</b>ثبت خودکار یا Exception</div></div>''',
        unsafe_allow_html=True,
    )

    threshold = st.number_input("آستانه تأیید انسانی مبلغ (تومان)", min_value=0, max_value=2_000_000_000, value=int(st.session_state.ceo_finance_threshold), step=5_000_000, key="v08_finance_threshold")
    st.session_state.ceo_finance_threshold = int(threshold)
    classified = classify_finance_transactions(st.session_state.ceo_finance_transactions, threshold)
    summary = finance_automation_summary(classified)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("تراکنش Demo", fa_num(summary["transactions"]))
    c2.metric("Auto-post آماده", fa_num(summary["auto_post"]))
    c3.metric("صف استثنا", fa_num(summary["exceptions"]))
    c4.metric("نرخ پردازش مستقیم", pct(summary["auto_post_rate"]))
    c5.metric("عدم تطبیق بانک", fa_num(summary["unreconciled"]))

    left, right = st.columns(2)
    with left:
        section_heading("دفتر روزنامه پیشنهادی", "ثبت خودکار قابل توضیح")
        auto = classified[classified["قابل Auto-post"]].head(25).copy()
        if auto.empty:
            st.info("در نمونه فعلی تراکنشی آماده Auto-post نیست.")
        else:
            auto["مبلغ"] = auto["مبلغ"].map(lambda x: f"{toman(float(x))} تومان")
            st.dataframe(auto[["شناسه تراکنش", "جهت", "شرح", "مبلغ", "حساب بدهکار", "حساب بستانکار", "وضعیت ثبت"]], use_container_width=True, hide_index=True)
    with right:
        section_heading("Exception Queue", "جایی که انسان باید وارد شود")
        exc = classified[~classified["قابل Auto-post"]].head(25).copy()
        if exc.empty:
            st.success("صف استثنا خالی است.")
        else:
            exc["مبلغ"] = exc["مبلغ"].map(lambda x: f"{toman(float(x))} تومان")
            st.dataframe(exc[["شناسه تراکنش", "شرح", "مبلغ", "اطمینان طبقه‌بندی", "مسیر کنترل"]], use_container_width=True, hide_index=True)

    section_heading("پیشنهادهای V0.8", "چطور حسابداری کم‌نیرو و کم‌خطا شود؟")
    ideas = pd.DataFrame([
        ["ورودی تراکنش", "Webhook/API بانک و درگاه + شناسه یکتا", "حذف ورود دستی"],
        ["ثبت حسابداری", "Rule Engine + کدینگ رسمی + سند دوطرفه", "کاهش خطای طبقه‌بندی"],
        ["تطبیق", "Bank/Gateway Reconciliation خودکار", "پیدا کردن مغایرت در همان روز"],
        ["اسناد", "Invoice/Order Match قبل از ثبت", "جلوگیری از ثبت بدون مدرک"],
        ["کنترل", "Auto-post برای اطمینان بالا، Exception برای بقیه", "انسان فقط در موارد پرریسک"],
        ["گزارش مدیرعامل", "Cash In/Out، مغایرت، مطالبات، پرداختی‌های بزرگ", "گزارش Exception-based"],
        ["Audit", "ثبت User/Rule/Before/After", "قابل حسابرسی و برگشت‌پذیر"],
    ], columns=["لایه", "پیشنهاد", "نتیجه"])
    st.dataframe(ideas, use_container_width=True, hide_index=True)

    cconn = st.columns(4)
    for idx, label in enumerate(["اتصال بانک", "اتصال درگاه", "اتصال نرم‌افزار حسابداری", "اتصال حقوق و دستمزد"]):
        with cconn[idx]:
            if st.button(f"{label} · Placeholder", use_container_width=True, key=f"v08_fin_conn_{idx}"):
                st.toast("فعلاً Placeholder است. برای اتصال واقعی API/Schema/Credential و سطح دسترسی لازم است.")

    st.warning("کدینگ حساب‌ها و قواعد مالیاتی این صفحه Demo هستند؛ قبل از ثبت واقعی باید با کدینگ رسمی نیک و تأیید مدیر حسابداری جایگزین شوند.")
    _department_task_table("حسابداری")
    _department_task_form("حسابداری", "حسین جودکی", "accounting")


def sales_lead_center_page(scenario):
    if not _ceo_ops_ready():
        return
    _ensure_ceo_state()
    sm = scenario_summary(scenario)
    backlog = int(sm["lead_backlog"])
    page_header(
        "مرکز پخش و پیگیری لید",
        "صف فعلی ۵۴۹۰ لید به یک سیستم Routing نیاز دارد: ورودی، حذف تکراری، اولویت، تخصیص، SLA تماس، نتیجه و Follow-up باید قابل اندازه‌گیری باشد.",
    )
    integration_toolbar("فروش", "v09_sales_center")
    st.markdown(
        f'''<div class="lead-routing-hero"><div class="section-kicker">Baseline فعلی</div><div class="lead-routing-value">{fa_num(backlog)} لید</div><div class="lead-routing-label">این عدد Baseline واقعی است؛ عملکرد کارشناسان و Routing پایین این صفحه تا اتصال CRM/Call Center، Demo است.</div></div>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''<div class="ops-flow"><div class="ops-step"><b>۱. Lead In</b>Website / Instagram / Phone / Referral</div><div class="ops-step"><b>۲. Validate</b>شماره معتبر + حذف Duplicate</div><div class="ops-step"><b>۳. Score</b>Age + Source + Intent</div><div class="ops-step"><b>۴. Route</b>Capacity × Performance × Fairness</div><div class="ops-step"><b>۵. SLA</b>First Touch + Follow-up</div><div class="ops-step"><b>۶. Outcome</b>Won / Lost / Follow-up / Recycle</div></div>''',
        unsafe_allow_html=True,
    )

    st.warning("نام کارشناسان فروش در اطلاعات فعلی به من داده نشده است؛ جدول زیر Placeholder است و می‌توانی اسامی واقعی را مستقیم جایگزین کنی.")
    edited_agents = st.data_editor(st.session_state.ceo_sales_agents, use_container_width=True, hide_index=True, num_rows="dynamic", key="v08_sales_agents_editor")
    if isinstance(edited_agents, pd.DataFrame):
        st.session_state.ceo_sales_agents = edited_agents.copy()

    allocation = allocate_lead_backlog(backlog, st.session_state.ceo_sales_agents)
    summary = lead_center_summary(backlog, st.session_state.ceo_sales_agents)
    performance = generate_sales_performance_demo(st.session_state.ceo_sales_agents)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("صف لید", fa_num(summary["backlog"]))
    c2.metric("کارشناس فعال · Demo", fa_num(summary["active_agents"]))
    c3.metric("ظرفیت تماس اولیه / روز · Demo", fa_num(summary["daily_contact_capacity"]))
    first_days = summary["first_touch_days_proxy"]
    c4.metric("زمان تماس اولیه کل صف · Proxy", "—" if pd.isna(first_days) else f"{fa_num(first_days)} روز", help="این زمان فروش نیست؛ فقط Backlog ÷ ظرفیت تماس اولیه Demo است.")

    left, right = st.columns(2)
    with left:
        section_heading("Routing", "پیشنهاد تقسیم صف")
        st.dataframe(allocation, use_container_width=True, hide_index=True)
        if st.button("ثبت پخش لید در Audit Log · Demo", use_container_width=True, type="primary", key="v08_route_leads"):
            _append_audit("پیشنهاد پخش لید", "فروش تلفنی", f"Routing برای {backlog} لید محاسبه شد.")
            st.toast("Routing ثبت شد؛ ارسال واقعی لید بعد از اتصال CRM فعال می‌شود.")
    with right:
        section_heading("Performance", "خروجی ۳۰روزه شبیه‌سازی‌شده")
        show = performance.copy()
        show["نرخ تبدیل"] = show["نرخ تبدیل"].map(pct)
        st.dataframe(show, use_container_width=True, hide_index=True)

    section_heading("KPI واقعی موردنیاز", "فروشنده فقط با فروش نهایی سنجیده نشود")
    kpis_needed = pd.DataFrame([
        ["Leads Received", "چند لید به هر فروشنده رسید؟"],
        ["First Touch SLA", "چند درصد در زمان هدف تماس گرفتند؟"],
        ["Contact Rate", "چند لید پاسخ دادند؟"],
        ["Qualified Rate", "چند لید واقعاً واجد شرایط بودند؟"],
        ["Win Rate", "فروش موفق / لید دریافتی"],
        ["Loss Reason", "چرا فروش ناموفق شد؟"],
        ["Follow-up Compliance", "پیگیری‌های موعددار انجام شدند؟"],
        ["Revenue per Lead", "هر لید چه ارزشی ایجاد کرد؟"],
    ], columns=["KPI", "معنا"])
    st.dataframe(kpis_needed, use_container_width=True, hide_index=True)
    _department_task_table("فروش تلفنی")
    _department_task_form("فروش تلفنی", "سرپرست فروش", "sales")


def _simple_department_workspace(dept_key: str, title: str, subtitle: str, default_owner: str, automation_ideas: list[tuple[str, str]], key_prefix: str, scenario, kpis):
    if not _ceo_ops_ready():
        return
    _ensure_ceo_state()
    page_header(title, subtitle)
    integration_toolbar(title, f"v09_{key_prefix}_workspace")
    section_heading("تیم", title)
    _render_team_roster(dept_key, 3)

    if MANAGEMENT_ENGINE_AVAILABLE:
        detail = department_kpis(scenario, kpis, _management_overrides())
        if dept_key == "support":
            management_key = "support"
        elif dept_key == "hr":
            management_key = "hr"
        elif dept_key == "marketing":
            management_key = "marketing"
        else:
            management_key = dept_key
        detail = detail[detail["department"] == management_key].copy()
        if not detail.empty:
            detail["مقدار"] = detail.apply(lambda r: _format_mgmt_value(r["actual"], r["unit"]), axis=1)
            detail["هدف"] = detail.apply(lambda r: _format_mgmt_value(r["target"], r["unit"]), axis=1)
            st.dataframe(detail[["metric", "مقدار", "هدف", "status", "source"]].rename(columns={"metric": "KPI", "status": "وضعیت", "source": "منبع"}), use_container_width=True, hide_index=True)

    section_heading("اتوماسیون پیشنهادی", "چه چیزهایی از کار دستی حذف شود؟")
    ideas = pd.DataFrame(automation_ideas, columns=["اتوماسیون", "خروجی برای مدیریت"])
    st.dataframe(ideas, use_container_width=True, hide_index=True)
    dept_name = title.replace("مدیریت ", "") if title.startswith("مدیریت ") else title
    _department_task_table(dept_name)
    _department_task_form(dept_name, default_owner, key_prefix)


def support_workspace_page(scenario, kpis):
    _simple_department_workspace(
        "support",
        "پشتیبانی",
        "پشتیبانی باید برای مدیرعامل به Exception تبدیل شود: Backlog، SLA، موضوعات پرتکرار و Escalation؛ نه گزارش طولانی از تمام تیکت‌ها.",
        "خانم ملیکا جمع دار",
        [
            ("دسته‌بندی خودکار تیکت", "موضوعات پرتکرار و Root Cause"),
            ("SLA Timer", "هشدار فقط برای تیکت‌های نزدیک نقض SLA"),
            ("Escalation Rule", "ارسال موارد بحرانی برای سرپرست/مدیرعامل"),
            ("Daily Support Brief", "Backlog + حل‌شده + Escalation + تصمیم لازم"),
        ],
        "support",
        scenario,
        kpis,
    )


def hr_workspace_page(scenario, kpis):
    _simple_department_workspace(
        "hr",
        "منابع انسانی",
        "HR در پنل مدیرعامل باید ظرفیت تیم، حضور، استخدام باز، Onboarding و مسائل نیازمند تصمیم را خلاصه کند.",
        "خانم مقصودی",
        [
            ("ورود حضور/مرخصی", "گزارش خودکار ظرفیت و غیبت"),
            ("Onboarding Checklist", "هیچ مرحله‌ای برای نیروی جدید فراموش نشود"),
            ("Review Reminder", "یادآوری دوره‌ای ارزیابی و جلسه 1:1"),
            ("People Exception Brief", "فقط ریسک خروج، غیبت غیرعادی و نیاز استخدام"),
        ],
        "hr",
        scenario,
        kpis,
    )


def marketing_workspace_page(scenario, kpis):
    _simple_department_workspace(
        "marketing",
        "مارکتینگ",
        "مارکتینگ از تولید محتوا به سیستم قابل اندازه‌گیری Content → Lead → Sale → Revenue تبدیل شود و فقط KPIهای تصمیم‌ساز به مدیرعامل گزارش شوند.",
        "امیر عباس حبیبی",
        [
            ("Content Calendar", "Owner + Deadline + Status برای هر خروجی"),
            ("UTM / CTA Tracking", "اتصال محتوا به Lead و فروش"),
            ("Campaign Approval", "بودجه + Offer + Margin + Inventory قبل از اجرا"),
            ("Daily Marketing Brief", "محتوای برتر، Lead، فروش منتسب، تصمیم لازم"),
        ],
        "marketing",
        scenario,
        kpis,
    )


def _v09_icon_svg(kind: str) -> str:
    icons = {
        "spark": '<svg viewBox="0 0 24 24"><path d="M12 3l1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15z"/></svg>',
        "sales": '<svg viewBox="0 0 24 24"><path d="M4 18l5-5 4 3 7-9"/><path d="M15 7h5v5"/></svg>',
        "finance": '<svg viewBox="0 0 24 24"><path d="M3 8l9-5 9 5"/><path d="M5 10v8M10 10v8M14 10v8M19 10v8M3 21h18"/></svg>',
        "code": '<svg viewBox="0 0 24 24"><path d="M8 5L3 12l5 7M16 5l5 7-5 7M14 3l-4 18"/></svg>',
        "factory": '<svg viewBox="0 0 24 24"><path d="M3 21V9l6 3V8l6 4V5h5v16H3z"/><path d="M7 16h2M12 16h2M17 16h2"/></svg>',
        "support": '<svg viewBox="0 0 24 24"><path d="M5 15a7 7 0 0 1 14 0"/><path d="M4 14v5h4v-5H4zM16 14v5h4v-5h-4z"/><path d="M16 21h-4"/></svg>',
        "people": '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3"/><circle cx="17" cy="9" r="2"/><path d="M3 20c.7-4 2.8-6 6-6s5.3 2 6 6M15 15c3 0 5 1.7 6 5"/></svg>',
        "currency": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8 9h7a2 2 0 0 1 0 4H9a2 2 0 0 0 0 4h7M12 6v12"/></svg>',
    }
    return icons.get(kind, icons["spark"])


def integration_toolbar(department: str, key_prefix: str):
    st.markdown(
        f'''<div class="v09-integration"><div class="v09-integration-title">{department} · لایه اتصال</div><div class="v09-integration-state"><span class="v09-connect-pill"><span class="v09-dot"></span>API · آماده اتصال</span><span class="v09-connect-pill"><span class="v09-dot"></span>n8n · آماده اتصال</span><span class="v09-connect-pill">Audit · طراحی شده</span></div></div>''',
        unsafe_allow_html=True,
    )
    a, b, _ = st.columns([1, 1, 4])
    with a:
        if st.button("اتصال API", key=f"{key_prefix}_api", use_container_width=True):
            st.toast("API Connector در فاز اتصال واقعی فعال می‌شود؛ این نسخه Credential ذخیره نمی‌کند.")
    with b:
        if st.button("اتصال n8n", key=f"{key_prefix}_n8n", use_container_width=True):
            st.toast("Workflow / Webhook n8n در فاز اجرایی فعال می‌شود.")


def department_hero(title: str, copy: str, icon: str = "spark", kicker: str = "EXECUTIVE WORKSPACE"):
    st.markdown(
        f'''<div class="v09-section-hero"><div style="display:flex;gap:12px;align-items:flex-start"><div class="v09-icon-shell">{_v09_icon_svg(icon)}</div><div><div class="v09-hero-kicker">{kicker}</div><div class="v09-hero-title">{title}</div><div class="v09-hero-copy">{copy}</div></div></div></div>''',
        unsafe_allow_html=True,
    )


def finance_dashboard_page(scenario, kpis):
    department_hero("نمای مالی مدیریتی", "مدیرعامل در این صفحه فقط جریان پول، مغایرت، Exception و موارد نیازمند تصمیم را می‌بیند؛ ثبت روزمره باید توسط موتور حسابداری انجام شود.", "finance", "FINANCE · CEO VIEW")
    integration_toolbar("حسابداری", "v09_finance_dash")
    if CEO_OPS_AVAILABLE:
        _ensure_ceo_state()
        classified = classify_finance_transactions(st.session_state.ceo_finance_transactions, st.session_state.ceo_finance_threshold)
        fin = finance_automation_summary(classified)
    else:
        fin = {}
    cols = st.columns(4)
    cols[0].metric("درآمد ماهانه مدل", f"{toman(float(kpis.get('monthly_revenue', 0)))} تومان")
    cols[1].metric("Auto-post · Demo", pct(float(fin.get("auto_post_rate", 0.0))))
    cols[2].metric("Exception · Demo", fa_num(float(fin.get("exceptions", 0))))
    cols[3].metric("مغایرت بانکی", "در انتظار اتصال")
    section_heading("دید مدیرعامل", "چه چیزی باید از حسابداری بالا بیاید؟", "نه تمام تراکنش‌ها؛ فقط جریان پول، ریسک و Exception.")
    st.dataframe(pd.DataFrame([
        ["Cash In / Cash Out", "لحظه‌ای / روزانه", "Bank + Gateway", "در انتظار API"],
        ["Bank Reconciliation", "روزانه", "Bank + Ledger", "در انتظار API"],
        ["مطالبات/بدهی سررسید", "روزانه", "Accounting", "در انتظار API"],
        ["تراکنش بزرگ/نامعمول", "لحظه‌ای", "Rule Engine", "قابل اتوماسیون"],
        ["بستن ماه", "ماهانه", "Ledger", "نیازمند Approval"],
    ], columns=["خروجی", "بازه", "منبع", "وضعیت"]), use_container_width=True, hide_index=True)
    _department_task_table("حسابداری")
    _department_task_form("حسابداری", "حسین جودکی", "v09_finance_task")


def sales_funnel_ops_page(scenario):
    department_hero("پخش لید و عملکرد کارشناسان", "۵۴۹۰ لید باید از یک صف مبهم به یک سیستم قابل پیگیری تبدیل شود: تخصیص، تماس اول، پیگیری، موفق/ناموفق و دلیل از دست رفتن.", "sales", "SALES OPERATIONS")
    integration_toolbar("فروش", "v09_sales_ops")
    if not CEO_OPS_AVAILABLE:
        st.info("موتور عملیات CEO در این Deploy موجود نیست.")
        return
    _ensure_ceo_state()
    backlog = int(scenario_summary(scenario)["lead_backlog"])
    allocation = allocate_lead_backlog(backlog, st.session_state.ceo_sales_agents)
    performance = generate_sales_performance_demo(st.session_state.ceo_sales_agents)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("لید موجود", fa_num(backlog))
    c2.metric("کارشناس فعال · Demo", fa_num(int(st.session_state.ceo_sales_agents.get("فعال", pd.Series(dtype=bool)).fillna(False).sum())))
    c3.metric("فروش موفق · Demo", fa_num(float(performance["فروش موفق"].sum())) if "فروش موفق" in performance else "—")
    c4.metric("فروش ناموفق · Demo", fa_num(float(performance["ناموفق"].sum())) if "ناموفق" in performance else "—")
    left,right=st.columns(2)
    with left:
        section_heading("Routing", "تخصیص بر اساس ظرفیت × عملکرد × عدالت")
        st.dataframe(allocation, use_container_width=True, hide_index=True)
    with right:
        section_heading("Outcome", "عملکرد کارشناسان · شبیه‌سازی")
        st.dataframe(performance, use_container_width=True, hide_index=True)
    st.caption("نسخه واقعی باید برای هر Lead ID تاریخچه Event از ورود تا Won / Lost / Recycle داشته باشد.")


def it_delivery_page(scenario, kpis):
    department_hero("آپدیت، تسک و تحویل IT", "برای موضوعات پرتکرار مدیریت با برنامه‌نویسان: چه چیزی در حال ساخت است، Blocker چیست، نسخه بعدی چه زمانی Release می‌شود و قالب جدید سایت در چه مرحله‌ای است.", "code", "PRODUCT DELIVERY")
    integration_toolbar("فناوری اطلاعات", "v09_it_delivery")
    _render_team_roster("it", 3)
    section_heading("قالب گزارش", "هر آپدیت باید فقط ۵ پاسخ داشته باشد")
    st.dataframe(pd.DataFrame([
        ["چی تغییر کرد؟", "خروجی قابل مشاهده / لینک نسخه"],
        ["Owner کیست؟", "یک نفر مسئول نهایی"],
        ["Blocker چیست؟", "فقط مانع واقعی"],
        ["ETA چیست؟", "زمان تحویل + Confidence"],
        ["معیار پذیرش؟", "Definition of Done قبل از Release"],
    ], columns=["سؤال", "خروجی اجباری"]), use_container_width=True, hide_index=True)
    _department_task_table("فناوری اطلاعات / برنامه‌نویسی")
    _department_task_form("فناوری اطلاعات / برنامه‌نویسی", "تیم IT", "v09_it_delivery_task")


def fx_supply_page(scenario, kpis):
    department_hero("ارز، تأمین و بهای ساخت", "یوان چین برای خرید قطعات و تولید نیک‌پوز، و درهم امارات برای جریان مالی شرکت دوم در دبی. نرخ‌ها تا اتصال API دستی‌اند و به‌عنوان داده واقعی خودکار معرفی نمی‌شوند.", "currency", "SUPPLY & FX CONTROL")
    integration_toolbar("تولید / تأمین / ارز", "v09_fx")
    if not V09_AVAILABLE:
        st.info("موتور V0.9 موجود نیست.")
        return
    cny = st.number_input("نرخ یوان چین / تومان", min_value=0.0, value=0.0, step=500.0, key="v09_cny_rate_input")
    aed = st.number_input("نرخ درهم امارات / تومان", min_value=0.0, value=0.0, step=500.0, key="v09_aed_rate_input")
    with st.expander("بهای ساخت و Batch", expanded=True):
        r1,r2,r3 = st.columns(3)
        units = r1.number_input("تعداد Batch تولید", min_value=0, value=100, step=10, key="v09_fx_units")
        component = r2.number_input("قطعات هر دستگاه / CNY", min_value=0.0, value=0.0, step=10.0, key="v09_component_cny")
        local = r3.number_input("هزینه داخلی هر دستگاه / تومان", min_value=0.0, value=0.0, step=100_000.0, key="v09_local_unit")
        r4,r5,r6 = st.columns(3)
        freight = r4.number_input("حمل کل Batch / تومان", min_value=0.0, value=0.0, step=100_000.0, key="v09_freight")
        dubai_in = r5.number_input("ورودی شرکت دبی / AED", min_value=0.0, value=0.0, step=100.0, key="v09_dubai_in")
        dubai_out = r6.number_input("خروجی شرکت دبی / AED", min_value=0.0, value=0.0, step=100.0, key="v09_dubai_out")
    snap = fx_cost_snapshot(cny, aed, component, int(units), freight, local, dubai_in, dubai_out)
    cards=st.columns(4)
    cards[0].metric("بهای ساخت هر دستگاه", "—" if snap["unit_total_toman"]<=0 else f"{toman(snap['unit_total_toman'])} تومان")
    cards[1].metric("هزینه Batch", "—" if snap["batch_total_toman"]<=0 else f"{toman(snap['batch_total_toman'])} تومان")
    cards[2].metric("اثر +۵٪ یوان", "—" if snap["cny_5pct_impact"]<=0 else f"+{toman(snap['cny_5pct_impact'])} تومان")
    cards[3].metric("خالص شرکت دبی", "—" if aed<=0 else f"{toman(snap['dubai_net_toman'])} تومان")
    if cny<=0 or aed<=0:
        st.info("نرخ واقعی وارد نشده است. اتصال FX API باید Source، Timestamp و Approval داشته باشد.")
    st.dataframe(pd.DataFrame([
        ["CNY +5%", snap["batch_if_cny_plus_5"], "افزایش بهای قطعات"],
        ["CNY -5%", snap["batch_if_cny_minus_5"], "کاهش بهای قطعات"],
    ], columns=["سناریو", "هزینه Batch / تومان", "اثر"]), use_container_width=True, hide_index=True)


def marketing_economics_page(scenario, kpis):
    department_hero("اقتصاد مارکتینگ", "درآمد آنلاین کل با درآمد منتسب به مارکتینگ یکی نیست. این صفحه عمداً این دو را جدا می‌کند تا Attribution و ROI قابل دفاع بمانند.", "finance", "MARKETING ECONOMICS")
    integration_toolbar("مارکتینگ", "v09_marketing_econ")
    sm=scenario_summary(scenario)
    a,b,c=st.columns(3)
    avg_salary=a.number_input("میانگین حقوق هر نفر / ماه (تومان)",min_value=0.0,value=0.0,step=1_000_000.0,key="v09_mkt_avg_salary")
    overhead=b.number_input("هزینه غیرحقوقی مارکتینگ / ماه",min_value=0.0,value=0.0,step=1_000_000.0,key="v09_mkt_overhead")
    tracked=c.number_input("Revenue واقعی منتسب به مارکتینگ / تومان",min_value=0.0,value=0.0,step=1_000_000.0,key="v09_mkt_rev_input",help="فقط وقتی Source Tracking واقعی دارید وارد شود.")
    team_count=max(int(sm.get("content_team_size",5)),1)
    team_cost=avg_salary*team_count+overhead
    snap=marketing_economics_snapshot(sm["online_monthly"],sm["asp"],sm["content_sales"],sm["sales_days"],team_cost,tracked)
    cols=st.columns(5)
    cols[0].metric("میانگین حقوق", "—" if avg_salary<=0 else f"{toman(avg_salary)} تومان")
    cols[1].metric("هزینه تیم", "—" if team_cost<=0 else f"{toman(team_cost)} تومان")
    cols[2].metric("درآمد آنلاین مدل",f"{toman(snap['online_revenue_model'])} تومان")
    cols[3].metric("فروش منتسب تخمینی",f"{fa_num(snap['estimated_attributed_units'],1)} دستگاه")
    cols[4].metric("ROI قابل دفاع", "—" if snap["verified_roi"]<=0 else f"{snap['verified_roi']:.2f}×")
    st.warning("تا اتصال UTM / CTA / CRM، Revenue مارکتینگ قطعی نمایش داده نمی‌شود.")


def marketing_trend_page(scenario):
    department_hero("روند و عملکرد مارکتینگ", "مدیرعامل باید بداند تیم در حال بهتر شدن است یا فقط چند محتوای Hit میانگین را بالا برده‌اند. Median و ثبات روند کنار Average دیده می‌شوند.", "spark", "MARKETING TREND")
    integration_toolbar("مارکتینگ", "v09_marketing_trend")
    reel=REEL_SNAPSHOT.copy()
    if reel.empty:
        st.info("Snapshot ریلز موجود نیست.")
        return
    view_col = "views" if "views" in reel.columns else reel.columns[1]
    show=reel.copy(); show["ردیف"]=range(1,len(show)+1)
    fig=px.line(show,x="ردیف",y=view_col,markers=True,labels={"ردیف":"ریلز",view_col:"بازدید"})
    fig.update_traces(line_color="#4F84CC")
    st.plotly_chart(style_fig(fig,360),use_container_width=True)
    m=reel_snapshot_metrics(); c1,c2,c3,c4=st.columns(4)
    c1.metric("میانگین بازدید",fa_num(m.get("average_views",0)))
    c2.metric("میانه بازدید",fa_num(m.get("median_views",0)))
    c3.metric("سهم ۳ ریلز برتر",pct(m.get("top3_view_share",0)))
    c4.metric("Comments + Shares",fa_num(m.get("total_comments",0)+m.get("total_shares",0)))

def executive_overview_v09(scenario, data, kpis, monthly, funnel, forecast, insights, customers_model=None, risk_stats=None, sales_anomalies=None, sms_anomalies=None):
    sm = scenario_summary(scenario)
    page_header("میز مدیرعامل", "فقط استثنا، تصمیم و اقدام؛ جزئیات هر واحد در منوی همان واحد قرار گرفته است.")
    inbox_count = 0; needs_followup = 0; overdue_reports = 0
    if CEO_OPS_AVAILABLE:
        _ensure_ceo_state()
        reports = department_report_schedule()
        classified = classify_finance_transactions(st.session_state.ceo_finance_transactions, st.session_state.ceo_finance_threshold)
        fin = finance_automation_summary(classified)
        inbox = ceo_inbox(int(sm["lead_backlog"]), st.session_state.ceo_tasks, reports, fin)
        ts = task_summary(st.session_state.ceo_tasks); rs = reporting_summary(reports)
        inbox_count=len(inbox); needs_followup=ts.get("needs_followup",0); overdue_reports=rs.get("overdue",0)
    st.markdown(
        f'''<div class="ceo-personal-hero"><div class="ceo-personal-kicker">EXECUTIVE MORNING · {CEO_NAME}</div><div class="ceo-personal-title">یک نگاه، سه تصمیم، بدون شلوغی.</div><div class="ceo-personal-copy">این صفحه عمداً خلاصه است. مدیرعامل نباید نمودارهای هر واحد را اینجا ببیند؛ فقط آنچه تغییر کرده، عقب افتاده یا نیازمند تصمیم است بالا می‌آید.</div></div>''',
        unsafe_allow_html=True,
    )
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("درآمد ماهانه مدل", f"{toman(float(kpis.get('monthly_revenue',0)))} تومان", help="محاسبه‌شده از Baseline؛ هنوز حسابداری واقعی نیست")
    c2.metric("فروش ماهانه", f"{fa_num(float(kpis.get('monthly_units',0)))} دستگاه")
    c3.metric("صف لید", fa_num(sm["lead_backlog"]), help="Baseline فعلی")
    c4.metric("نیازمند پیگیری", fa_num(needs_followup), help="Task Engine / Demo تا اتصال واقعی")

    brief_text, brief_conf, brief_source = build_ceo_brief(scenario, kpis)
    conf_label, conf_class = confidence_label(brief_conf)
    st.markdown(f'''<div class="v06-brief"><div class="v06-brief-kicker">خلاصه مدیریتی امروز</div><div class="v06-brief-title">آنچه ارزش توجه دارد</div><div class="v06-brief-copy">{brief_text}</div><div class="v06-brief-footer"><span class="confidence-pill {conf_class}">{conf_label}</span><span class="trust-pill src-derived">{brief_source}</span></div></div>''', unsafe_allow_html=True)

    priorities = _executive_priorities(scenario, kpis, sales_anomalies, sms_anomalies, risk_stats or {})[:3]
    section_heading("صف تصمیم", "سه موضوع روی میز", "بقیه داده‌ها در صفحات واحدها هستند.")
    cols = st.columns(3)
    for i,item in enumerate(priorities,1):
        with cols[i-1]: priority_card(i,item["title"],item["text"],item["severity"],item["source"])

    section_heading("عملیات مدیرعامل", "گزارش و تسک", "اگر گزارشی نرسد یا تسکی از SLA عبور کند، اینجا باید بالا بیاید.")
    a,b,c = st.columns(3)
    a.metric("Inbox مدیریتی", fa_num(inbox_count))
    b.metric("گزارش عقب‌افتاده · Demo", fa_num(overdue_reports))
    c.metric("وضعیت اتصال", "Demo / آماده API")

    if MANAGEMENT_ENGINE_AVAILABLE:
        try:
            _ensure_management_state()
            org = department_summary(scenario,kpis,_management_overrides())
            section_heading("نبض سازمان", "فقط وضعیت واحدها", "برای جزئیات وارد همان واحد شوید.")
            _render_department_cards(org, min(4,max(1,len(org))))
        except Exception:
            st.info("نبض سازمان در این Deploy موقتاً در دسترس نیست.")

    with st.expander("نمای تحلیلی عمیق‌تر", expanded=False):
        st.caption("این بخش برای زمانی است که مدیر بخواهد از Summary وارد تحلیل شود؛ در حالت عادی بسته می‌ماند.")
        x,y=st.columns(2)
        with x:
            fig=px.area(monthly,x="month",y="revenue",markers=True,labels={"month":"ماه","revenue":"درآمد"})
            fig.update_traces(line_color="#4F84CC",fillcolor="rgba(173,203,255,.24)")
            st.plotly_chart(style_fig(fig,330),use_container_width=True)
        with y:
            fig=go.Figure(go.Funnel(y=funnel["stage"].map(FUNNEL_FA),x=funnel["count"],textinfo="value+percent initial",marker={"color":["#87B3EC","#9FC4F3","#B5D2F6","#C6DDF8","#D6E8FA","#E4F0FC"]}))
            st.plotly_chart(style_fig(fig,330),use_container_width=True)
    source_legend()

def _customer_growth_ready() -> bool:
    if CUSTOMER_GROWTH_AVAILABLE:
        return True
    department_hero("Growth Intelligence در دسترس نیست", "فایل customer_growth_engine.py در Deploy فعلی پیدا نشده؛ پنل مدیریتی داخلی همچنان سالم است.", "people", "CUSTOMER PRODUCT")
    st.error("برای فعال‌شدن بخش مشتریان کسب‌وکار، customer_growth_engine.py نسخه V0.10 را کنار app.py قرار بده.")
    if CUSTOMER_GROWTH_ERROR:
        st.caption(f"جزئیات Import: {CUSTOMER_GROWTH_ERROR}")
    return False


def customer_product_header(title: str, subtitle: str):
    logo_uri = asset_data_uri(LOGO_PATH)
    logo_html = f'<img class="hero-logo" src="{logo_uri}" alt="NIKSMS">' if logo_uri else ""
    st.markdown(
        f'''<div class="customer-product-hero"><div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start"><div><div class="customer-product-kicker">NIK GROWTH INTELLIGENCE · CUSTOMER EDITION</div><div class="customer-product-title">{title}</div><div class="customer-product-copy">{subtitle}</div></div>{logo_html}</div></div>''',
        unsafe_allow_html=True,
    )
    integration_toolbar("پنل رشد کسب‌وکار", f"v10_{abs(hash(title))}")
    st.markdown('<div class="customer-source-note"><b>نسخه دمو:</b> تمام اعداد این بخش مصنوعی‌اند و برای نمایش منطق محصول مشتری ساخته شده‌اند. این بخش از داده‌های داخلی NIK و پنل مدیرعامل جداست.</div>', unsafe_allow_html=True)


def _customer_vertical_selector(key: str) -> str:
    return st.selectbox("نوع کسب‌وکار نمونه", list(VERTICAL_PROFILES.keys()), key=key, help="داده‌های دمو با انتخاب صنعت تغییر می‌کنند.")


def _render_customer_health(snapshot: dict):
    st.markdown(
        f'''<div class="customer-health">
        <div class="customer-health-main"><div class="customer-health-label">امتیاز سلامت ارتباط با مشتری</div><div class="customer-health-value">{fa_num(snapshot['health_score'])} / ۱۰۰</div><div class="customer-health-note">شاخص دمو بر اساس رشد دیتابیس، مشتری خوابیده و سهم VIP.</div></div>
        <div class="customer-health-cell"><div class="customer-health-label">دارایی مشتری</div><div class="customer-health-value">{fa_num(snapshot['total_customers'])}</div><div class="customer-health-note">شماره مشتری ثبت‌شده در باشگاه.</div></div>
        <div class="customer-health-cell"><div class="customer-health-label">شماره جدید / ۳۰ روز</div><div class="customer-health-value">+{fa_num(snapshot['captured_30d'])}</div><div class="customer-health-note">رشد دارایی مشتری.</div></div>
        <div class="customer-health-cell"><div class="customer-health-label">مشتری خوابیده ۴۵+ روز</div><div class="customer-health-value">{fa_num(snapshot['dormant_45'])}</div><div class="customer-health-note">فرصت بالقوه برای بازگشت.</div></div>
        </div>''',
        unsafe_allow_html=True,
    )


def customer_growth_home_page():
    if not _customer_growth_ready():
        return
    customer_product_header("مرکز رشد کسب‌وکار", "هدف این صفحه نشان‌دادن داده نیست؛ هدف این است که صاحب کسب‌وکار هر روز بداند چه اقدامی می‌تواند مشتری قبلی را دوباره به درآمد تبدیل کند.")
    vertical = _customer_vertical_selector("v10_growth_home_vertical")
    snapshot = business_snapshot(vertical)
    _render_customer_health(snapshot)

    section_heading("اقدام پیشنهادی امروز", f"{fa_num(snapshot['repeat_dormant'])} مشتری با احتمال بازگشت بالاتر", "سیستم به‌جای نمایش صرف آمار، فرصت را به یک Action قابل اجرا تبدیل می‌کند.")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            f'''<div class="action-story">
            <div class="action-story-line"><div class="action-story-index">۱</div><div class="action-story-text"><b>{fa_num(snapshot['dormant_45'])} مشتری</b> بیشتر از ۴۵ روز است برنگشته‌اند.</div></div>
            <div class="action-story-line"><div class="action-story-index">۲</div><div class="action-story-text">از این تعداد <b>{fa_num(snapshot['repeat_dormant'])} نفر</b> قبلاً بیش از دو بار خرید کرده‌اند.</div></div>
            <div class="action-story-line"><div class="action-story-index">۳</div><div class="action-story-text"><b>پیشنهاد:</b> برای همین گروه یک پیام بازگشت با مزیت محدود ارسال شود.</div></div>
            <div class="action-story-line"><div class="action-story-index">۴</div><div class="action-story-text">هزینه تقریبی ارسال: <b>{toman(snapshot['suggested_campaign_cost'])} تومان</b>.</div></div>
            <div class="action-story-line"><div class="action-story-index">۵</div><div class="action-story-text">سیستم سگمنت و متن را آماده می‌کند؛ کاربر فقط تأیید می‌کند.</div></div>
            </div>''', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("ساخت کمپین بازگشت", type="primary", use_container_width=True, key="v10_prepare_winback"):
                st.session_state.v10_campaign_prepared = True
                st.toast("کمپین دمو ساخته شد؛ هنوز هیچ پیام واقعی ارسال نشده است.")
        with b2:
            if st.button("مشاهده مخاطبان", use_container_width=True, key="v10_show_audience"):
                st.session_state.v10_show_audience = not st.session_state.get("v10_show_audience", False)
        if st.session_state.get("v10_show_audience", False):
            st.info(f"دمو: {fa_num(snapshot['repeat_dormant'])} مخاطب با شرط «۴۵+ روز بدون بازگشت + بیش از ۲ خرید قبلی» انتخاب شده‌اند.")
    with right:
        st.markdown(
            f'''<div class="growth-opportunity"><span class="growth-opportunity-priority">نتیجه آزمایشی پس از ۷ روز</span><div class="growth-opportunity-title">کمپین بازگشت چه نتیجه‌ای می‌تواند بسازد؟</div><div class="growth-opportunity-value">{fa_num(snapshot['demo_returners'])} نفر برگشتند</div><div class="growth-opportunity-meta">{fa_num(snapshot['demo_purchasers'])} خرید ثبت شد</div><div class="growth-opportunity-value">{toman(snapshot['demo_returned_revenue'])} تومان</div><div class="growth-opportunity-meta">ارزش بازگشت ایجادشده · دمو / Synthetic</div></div>''',
            unsafe_allow_html=True,
        )
        if st.session_state.get("v10_campaign_prepared", False):
            if st.button("اجرای نتیجه آزمایشی", use_container_width=True, key="v10_run_demo_campaign"):
                st.session_state.v10_campaign_demo_done = True
        if st.session_state.get("v10_campaign_demo_done", False):
            st.success(f"دمو اجرا شد: {fa_num(snapshot['demo_returners'])} بازگشت، {fa_num(snapshot['demo_purchasers'])} خرید و {toman(snapshot['demo_returned_revenue'])} تومان ارزش بازگشت مدل‌سازی‌شده.")

    section_heading("فرصت‌های بعدی", "سیستم چه چیزهای دیگری می‌بیند؟", "چهار Opportunity آماده که می‌توانند به Action تبدیل شوند.")
    opportunities = campaign_opportunities(vertical)
    cols = st.columns(4)
    for i, row in opportunities.iterrows():
        with cols[i % 4]:
            st.markdown(
                f'''<div class="growth-opportunity"><span class="growth-opportunity-priority">{row['priority']}</span><div class="growth-opportunity-title">{row['title']}</div><div class="growth-opportunity-copy">{row['reason']}</div><div class="growth-opportunity-value">{fa_num(row['audience'])} مشتری</div><div class="growth-opportunity-meta">هزینه ارسال ≈ {toman(row['cost'])} تومان</div></div>''',
                unsafe_allow_html=True,
            )

    section_heading("روند", "آیا باشگاه مشتریان در حال قوی‌تر شدن است؟", "داده زیر مصنوعی است و فقط شکل محصول نهایی را نمایش می‌دهد.")
    trend = growth_trend(vertical)
    a, b = st.columns(2)
    with a:
        fig = px.area(trend, x="month", y="customer_base", labels={"month":"ماه", "customer_base":"تعداد مشتری"})
        fig.update_traces(line_color="#4F84CC", fillcolor="rgba(79,132,204,.13)")
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)
    with b:
        fig = px.line(trend, x="month", y="repeat_rate", markers=True, labels={"month":"ماه", "repeat_rate":"نرخ خرید مجدد"})
        fig.update_traces(line_color="#3D78C5")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)


def customer_segments_portal_page():
    if not _customer_growth_ready():
        return
    customer_product_header("مشتریان و سگمنت‌ها", "هر شماره فقط یک رکورد نیست؛ سیستم باید بفهمد کدام مشتری ارزشمند، جدید، خوابیده یا در آستانه خرید مجدد است.")
    vertical = _customer_vertical_selector("v10_segments_vertical")
    snapshot = business_snapshot(vertical)
    seg = segment_table(vertical)
    _render_customer_health(snapshot)
    left, right = st.columns([1.15, 1])
    with left:
        fig = px.pie(seg, names="segment", values="customers", hole=.58)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(style_fig(fig, 385), use_container_width=True)
    with right:
        show = seg.copy()
        show["share"] = show["share"].map(pct)
        show = show.rename(columns={"segment":"سگمنت", "customers":"مشتری", "definition":"تعریف", "recommended_action":"اقدام پیشنهادی", "share":"سهم"})
        st.dataframe(show, use_container_width=True, hide_index=True, height=385)
    st.info("در نسخه واقعی، تعریف سگمنت‌ها باید با رفتار واقعی همان صنف، خرید، مراجعه، مبلغ و Recency/Frequency تنظیم شود؛ یک Rule برای همه کسب‌وکارها کافی نیست.")


def smart_campaigns_portal_page():
    if not _customer_growth_ready():
        return
    customer_product_header("کمپین‌های هوشمند", "سیستم Audience، دلیل انتخاب، هزینه و خروجی مورد انتظار را کنار هم می‌گذارد تا کاربر به‌جای ساخت دستی کمپین، تصمیم بگیرد و اجرا کند.")
    vertical = _customer_vertical_selector("v10_campaigns_vertical")
    campaigns = campaign_opportunities(vertical)
    for _, row in campaigns.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1])
            with c1:
                st.markdown(f"**{row['title']}**")
                st.caption(row["reason"])
                st.write(row["message"])
            c2.metric("مخاطب", fa_num(row["audience"]))
            c3.metric("هزینه مدل", f"{toman(row['cost'])} تومان")
            c4.metric("ارزش مدل", f"{toman(row['expected_value'])} تومان")
            if st.button("آماده‌سازی کمپین", key=f"v10_campaign_prepare_{row['campaign_id']}"):
                st.toast("کمپین به حالت Draft رفت. در نسخه واقعی مرحله Approval → SMS Engine → Result Tracking اجرا می‌شود.")
    st.caption("هیچ پیام واقعی از این Prototype ارسال نمی‌شود. نرخ خرید و ارزش کمپین در این صفحه Synthetic هستند.")


def business_automations_portal_page():
    if not _customer_growth_ready():
        return
    customer_product_header("اتوماسیون رشد", "هدف این است که صاحب کسب‌وکار مجبور نباشد هر روز به یاد بیاورد چه کسی را پیگیری کند؛ Triggerها این کار را به سیستم می‌سپارند.")
    vertical = _customer_vertical_selector("v10_automation_vertical")
    auto = automation_catalog(vertical)
    for _, row in auto.iterrows():
        status_cls = "customer-status-ready" if row["status"] == "آماده" else "customer-status-suggest"
        st.markdown(f'''<div class="customer-automation-row"><b>{row['automation']}</b><span>{row['trigger']}</span><span>{row['action']}</span><span class="{status_cls}">{row['status']}</span></div>''', unsafe_allow_html=True)
    st.write("")
    section_heading("اتصال آینده", "مسیر اجرای واقعی", "هر Automation باید قابل توقف، Audit و اندازه‌گیری باشد.")
    st.code("Trigger → Segment → Rule → Approval (optional) → NIKSMS → Event Tracking → Sale/Return → ROI", language="text")


def business_roi_portal_page():
    if not _customer_growth_ready():
        return
    customer_product_header("نتیجه و بازگشت سرمایه", "کاربر نباید فقط بداند چند پیام فرستاده؛ باید بداند چه تعداد مشتری برگشتند، چند خرید ساختند و ارزش ایجادشده چه بوده است.")
    vertical = _customer_vertical_selector("v10_roi_vertical")
    snapshot = business_snapshot(vertical)
    k = st.columns(5)
    k[0].metric("مخاطب کمپین", fa_num(snapshot["suggested_campaign_target"]))
    k[1].metric("هزینه ارسال", f"{toman(snapshot['suggested_campaign_cost'])} تومان")
    k[2].metric("بازگشت / Demo", fa_num(snapshot["demo_returners"]))
    k[3].metric("خرید / Demo", fa_num(snapshot["demo_purchasers"]))
    k[4].metric("ارزش بازگشت / Demo", f"{toman(snapshot['demo_returned_revenue'])} تومان")
    trend = growth_trend(vertical)
    left, right = st.columns(2)
    with left:
        fig = px.bar(trend, x="month", y="campaign_revenue", labels={"month":"ماه", "campaign_revenue":"ارزش بازگشت / دمو"})
        fig.update_traces(marker_color="#7DA9DF")
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)
    with right:
        st.markdown(f'''<div class="action-story"><div class="action-story-line"><div class="action-story-index">۱</div><div class="action-story-text">هزینه پیامک مدل: <b>{toman(snapshot['suggested_campaign_cost'])} تومان</b></div></div><div class="action-story-line"><div class="action-story-index">۲</div><div class="action-story-text">ارزش خرید ثبت‌شده در مدل: <b>{toman(snapshot['demo_returned_revenue'])} تومان</b></div></div><div class="action-story-line"><div class="action-story-index">۳</div><div class="action-story-text">در نسخه واقعی فقط Revenue دارای Attribution معتبر در ROI حساب می‌شود.</div></div></div>''', unsafe_allow_html=True)
        st.warning("این ROI برای نمایش منطق محصول است؛ قبل از فروش تجاری باید Attribution واقعی، هزینه تخفیف، حاشیه سود و بازه Attribution وارد شوند.")


def subscription_plans_portal_page():
    if not _customer_growth_ready():
        return
    customer_product_header("پلن‌های اشتراک Growth Intelligence", "سه سطح پیشنهادی برای تبدیل Intelligence و Automation به یک SaaS درآمد تکرارشونده. قیمت‌ها فعلاً پیشنهاد محصول‌اند، نه قیمت مصوب نیک.")
    st.markdown('<div class="customer-source-note"><b>Pricing Draft:</b> قیمت‌های زیر برای تست Positioning و ارائه مدیریتی هستند و باید بعداً با Cost-to-Serve، ارزش اقتصادی، تحقیق بازار و تست willingness-to-pay نهایی شوند.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, row in PLAN_CATALOG.reset_index(drop=True).iterrows():
        featured = " plan-card-featured" if i == 1 else ""
        features = "".join(f'<div class="plan-feature">{feature}</div>' for feature in row["features"])
        with cols[i]:
            st.markdown(f'''<div class="plan-card{featured}"><div class="plan-eyebrow">{"پیشنهاد اصلی" if i == 1 else "پلن اشتراک"}</div><div class="plan-name">{row['plan']}</div><div class="plan-price">{fa_num(row['price_monthly'])} تومان</div><div class="plan-period">ماهانه / قیمت پیشنهادی دمو</div><div class="plan-best">{row['best_for']}</div>{features}</div>''', unsafe_allow_html=True)
            if st.button(row["cta"], type="primary" if i == 1 else "secondary", use_container_width=True, key=f"v10_plan_{i}"):
                st.toast("این دکمه فعلاً برای Demo Pricing است؛ خرید واقعی متصل نشده است.")
    st.write("")
    section_heading("منطق درآمد برای نیک", "چرا این Add-on می‌تواند پولی باشد؟", "ارزش فقط اشتراک نیست؛ افزایش مصرف پیامک، ماندگاری مشتری، فروش NIKPOS و کاهش Churn هم بخشی از اقتصاد محصول است.")
    st.markdown('''<div class="action-story"><div class="action-story-line"><div class="action-story-index">۱</div><div class="action-story-text"><b>Subscription:</b> درآمد تکرارشونده مستقیم از Intelligence و Automation.</div></div><div class="action-story-line"><div class="action-story-index">۲</div><div class="action-story-text"><b>SMS Usage:</b> پیشنهادهای قابل اجرا می‌توانند مصرف هدفمند پیامک را افزایش دهند.</div></div><div class="action-story-line"><div class="action-story-index">۳</div><div class="action-story-text"><b>Retention:</b> پنلی که برای کاربر پول می‌سازد، احتمال تمدید و ماندگاری او را افزایش می‌دهد.</div></div><div class="action-story-line"><div class="action-story-index">۴</div><div class="action-story-text"><b>NIKPOS:</b> ارزش دستگاه از «جمع‌کردن شماره» به «ساختن ورودی موتور رشد» ارتقا پیدا می‌کند.</div></div></div>''', unsafe_allow_html=True)

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


def _runtime_error_panel(page: str, exc: Exception):
    """Page-level error boundary so one workspace never takes the whole OS down."""
    st.error("این بخش موقتاً با خطا روبه‌رو شده، اما سایر قسمت‌های NIK Management OS همچنان قابل استفاده‌اند.")
    st.markdown(f'<div class="v11-error-card"><b>بخش:</b> {PAGE_LABELS.get(page, page)}<br><span>برای اصلاح، فقط همین خطا و نام صفحه کافی است؛ نیازی به توضیح دوباره کل پروژه نیست.</span></div>', unsafe_allow_html=True)
    with st.expander("جزئیات فنی برای تیم توسعه", expanded=False):
        st.code(f"{type(exc).__name__}: {exc}", language="text")


def main():
    if "presentation_mode" not in st.session_state:
        st.session_state.presentation_mode = False
    inject_presentation_mode_css(bool(st.session_state.presentation_mode))
    scenario, page = scenario_sidebar()

    if st.session_state.presentation_mode:
        left, right = st.columns([5, 1])
        with left:
            st.markdown('<div class="presentation-ribbon"><span>حالت ارائه مدیرعامل فعال است · فقط تصمیم‌ها و KPIهای ضروری دیده می‌شوند.</span><span>V0.11</span></div>', unsafe_allow_html=True)
        with right:
            if st.button("خروج از ارائه", use_container_width=True, key="exit_presentation"):
                st.session_state.presentation_mode = False
                st.rerun()

    # Core dataset/KPIs are cached and cheap enough to keep universal.
    data = build_synthetic_data(scenario)
    kpis = normalize_kpis(scenario, current_kpis(scenario, data["customers"]))
    monthly = sales_monthly(data["sales"])
    funnel = lead_funnel(scenario)
    daily = sales_daily(data["sales"]) if page in {"Sales Analytics", "Anomaly Detection", "Automated Insights"} else pd.DataFrame()

    # Heavy ML/forecast/anomaly work is now lazy. This is the biggest V0.11 speed improvement.
    customers_model = data["customers"]
    segment_profile = pd.DataFrame()
    risk_stats = {}
    forecast = pd.DataFrame()
    forecast_stats = {}
    sales_anomalies = pd.DataFrame()
    sms_anomalies = pd.DataFrame()
    insights = []

    model_pages = {"Customer Intelligence", "Predictions", "Automated Insights"}
    forecast_pages = {"Predictions", "Automated Insights"}
    anomaly_pages = {"Anomaly Detection", "Automated Insights"}

    if page in model_pages:
        customers_model, segment_profile, risk_stats = build_models(
            data["customers"], int(scenario_value(scenario, "seed", None, 42))
        )
    if page in forecast_pages:
        forecast, forecast_stats = revenue_forecast(monthly, 3)
    if page in anomaly_pages:
        if daily.empty:
            daily = sales_daily(data["sales"])
        sales_anomalies = detect_anomalies(daily, "revenue", "date", 14)
        sms_anomalies = detect_anomalies(data["sms"], "delivery_rate", "date", 14)
    if page == "Automated Insights":
        insights = generate_insights(scenario, kpis, monthly, risk_stats, sales_anomalies, sms_anomalies)

    try:
        if page == "Executive Overview":
            executive_overview_v09(scenario, data, kpis, monthly, funnel, pd.DataFrame(), [], data["customers"], {}, pd.DataFrame(), pd.DataFrame())
        elif page == "CEO Task Center": ceo_task_center_page(scenario, kpis)
        elif page == "IT Workspace": it_workspace_page(scenario, kpis)
        elif page == "Accounting Automation": accounting_automation_page()
        elif page == "Sales Lead Center": sales_lead_center_page(scenario)
        elif page == "Support Workspace": support_workspace_page(scenario, kpis)
        elif page == "HR Workspace": hr_workspace_page(scenario, kpis)
        elif page == "Marketing Workspace": marketing_workspace_page(scenario, kpis)
        elif page == "Marketing Economics": marketing_economics_page(scenario, kpis)
        elif page == "Marketing Trend": marketing_trend_page(scenario)
        elif page == "Sales Funnel Ops": sales_funnel_ops_page(scenario)
        elif page == "Finance Dashboard": finance_dashboard_page(scenario, kpis)
        elif page == "IT Delivery": it_delivery_page(scenario, kpis)
        elif page == "FX & Supply": fx_supply_page(scenario, kpis)
        elif page == "Organization Pulse": organization_pulse_page(scenario, kpis)
        elif page == "Task & KPI": task_kpi_page(scenario, kpis)
        elif page == "Production & QC": production_qc_page(scenario, kpis)
        elif page == "Campaign Planner": campaign_planner_page(scenario, kpis)
        elif page == "Automation Center": automation_center_page(scenario, kpis)
        elif page == "Access Control": access_control_page()
        elif page == "Revenue Intelligence": revenue_intelligence_page(scenario, kpis)
        elif page == "Scenario Simulator": scenario_simulator_page(scenario, kpis)
        elif page == "Connections": connections_page()
        elif page == "Data Center": data_center_page(data)
        elif page == "Sales Analytics":
            if daily.empty: daily = sales_daily(data["sales"])
            sales_analytics_page(scenario, data, kpis, daily, monthly)
        elif page == "Customer Intelligence": customer_intelligence_page(customers_model, segment_profile, risk_stats)
        elif page == "NIKPOS Analytics": nikpos_page(scenario, data)
        elif page == "Content Analytics": content_page(scenario)
        elif page == "Media Intelligence": media_intelligence_page()
        elif page == "SMS Analytics": sms_page(data)
        elif page == "Anomaly Detection": anomaly_page(sales_anomalies, sms_anomalies)
        elif page == "Predictions": predictions_page(forecast, forecast_stats, customers_model, risk_stats)
        elif page == "Automated Insights": insights_page(insights)
        elif page == "Customer Growth Home": customer_growth_home_page()
        elif page == "Customer Segments Portal": customer_segments_portal_page()
        elif page == "Smart Campaigns Portal": smart_campaigns_portal_page()
        elif page == "Business Automations Portal": business_automations_portal_page()
        elif page == "Business ROI Portal": business_roi_portal_page()
        elif page == "Subscription Plans Portal": subscription_plans_portal_page()
        elif page == "Analysis Pipeline": pipeline_page()
        elif page == "Settings / Scenario Controls": settings_page(scenario, kpis)
        else:
            st.info("صفحه انتخاب‌شده در این نسخه تعریف نشده است.")
    except Exception as exc:
        _runtime_error_panel(page, exc)

    st.markdown('<div class="v11-footer"><span>NIK Management OS + Growth Intelligence · V0.11</span><span>Prototype / Demo Data · آماده اتصال به منابع واقعی</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
