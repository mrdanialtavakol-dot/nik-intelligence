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

PAGE_LABELS = {
    "Executive Overview": "نمای کلی مدیریتی",
    "Data Center": "مرکز داده",
    "Sales Analytics": "تحلیل فروش",
    "Customer Intelligence": "هوشمندی مشتریان",
    "NIKPOS Analytics": "تحلیل نیک‌پوز",
    "Content Analytics": "تحلیل محتوا",
    "SMS Analytics": "تحلیل پیامک",
    "Anomaly Detection": "تشخیص ناهنجاری",
    "Predictions": "پیش‌بینی‌ها",
    "Automated Insights": "بینش‌های خودکار",
    "Analysis Pipeline": "خط لوله تحلیل",
    "Settings / Scenario Controls": "تنظیمات و سناریو",
}
SEGMENT_FA = {
    "High Value": "باارزش", "Growth": "در حال رشد", "Regular": "عادی",
    "At Risk": "در معرض ریزش", "Inactive": "غیرفعال",
}
RISK_FA = {"Low": "کم", "Medium": "متوسط", "High": "زیاد", "Very High": "بسیار زیاد"}
PLAN_FA = {"Plan A": "طرح A", "Plan B": "طرح B"}
CHANNEL_FA = {"Phone": "فروش تلفنی", "Online": "فروش آنلاین", "phone": "فروش تلفنی", "online": "فروش آنلاین"}
FUNNEL_FA = {
    "New Leads": "سرنخ‌های جدید", "Contacted": "تماس گرفته‌شده",
    "Qualified": "واجد شرایط", "Interested": "علاقه‌مند",
    "Purchased": "خرید کرده", "Active Customer": "مشتری فعال",
}
QUALITY_STATUS_FA = {"Healthy": "سالم", "Watch": "نیازمند پایش", "Needs Review": "نیازمند بررسی"}
DATASET_FA = {"Sales": "فروش", "Customers": "مشتریان", "Leads": "سرنخ‌ها", "Sms": "پیامک", "Nikpos": "نیک‌پوز"}
FORECAST_SERIES_FA = {"Historical": "تاریخی", "Forecast": "پیش‌بینی"}
DATAFRAME_COL_FA = {
    "date":"تاریخ", "month":"ماه", "channel":"کانال", "plan":"طرح", "units":"تعداد", "revenue":"درآمد",
    "customer_id":"شناسه مشتری", "signup_date":"تاریخ عضویت", "city":"شهر", "industry":"صنعت", "last_activity":"آخرین فعالیت",
    "purchase_count":"تعداد خرید", "sms_usage":"مصرف پیامک", "nikpos_usage":"استفاده از نیک‌پوز", "recency":"روز از آخرین فعالیت",
    "frequency":"تکرار خرید", "monetary_value":"ارزش مالی", "lead_source":"منبع سرنخ", "customer_status":"وضعیت مشتری",
    "lead_id":"شناسه سرنخ", "created_date":"تاریخ ایجاد", "stage":"مرحله", "sent":"ارسال‌شده", "delivered":"تحویل‌شده",
    "delivery_rate":"نرخ تحویل", "clicks":"کلیک", "device_id":"شناسه دستگاه", "activation_date":"تاریخ فعال‌سازی",
    "active_device":"دستگاه فعال", "captures_30d":"ثبت شماره ۳۰ روزه", "z_score":"امتیاز Z", "is_anomaly":"ناهنجاری",
    "rolling_mean":"میانگین متحرک", "series":"نوع داده", "risk_score":"امتیاز ریسک", "risk_level":"سطح ریسک",
}

def fa_digits(value) -> str:
    return str(value).translate(str.maketrans("0123456789,.%", "۰۱۲۳۴۵۶۷۸۹٬٫٪"))

def fa_num(value: float, decimals: int = 0) -> str:
    return fa_digits(f"{value:,.{decimals}f}")


st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; color: #F5F5F5; direction: rtl; font-family: Tahoma, "Segoe UI", Arial, sans-serif; }}
    [data-testid="stSidebar"] {{ background: #0D0F13; border-left: 1px solid {BORDER}; border-right: 0; direction: rtl; }}
    [data-testid="stSidebar"] * {{ text-align: right; }}
    .stMarkdown, .stCaption, .stAlert, .stHeader, .stSubheader {{ direction: rtl; text-align: right; }}
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
        return f"{fa_num(value / 1_000_000_000, 2)} میلیارد تومان"
    if abs(value) >= 1_000_000:
        return f"{fa_num(value / 1_000_000, 1)} میلیون تومان"
    return f"{fa_num(value)} تومان"


def pct(value: float) -> str:
    return fa_digits(f"{value:.1%}")


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
        return False, "ستون‌های مورد انتظار وجود ندارند: " + ", ".join(sorted(missing))
    return True, "ساختار فایل تأیید شد"


def scenario_sidebar() -> Tuple[Scenario, str]:
    st.sidebar.markdown("## NIK INTELLIGENCE")
    st.sidebar.caption("پلتفرم خودکار هوشمندی داده | نسخه ۰.۱")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "منوی اصلی",
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
        format_func=lambda x: PAGE_LABELS[x],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### کنترل سناریو")
    price_a = st.sidebar.number_input("قیمت طرح A (تومان)", 1_000_000, 200_000_000, 15_000_000, 1_000_000)
    price_b = st.sidebar.number_input("قیمت طرح B (تومان)", 1_000_000, 300_000_000, 30_000_000, 1_000_000)
    share_a_pct = st.sidebar.slider("سهم طرح A", 0, 100, 50, 5)
    phone = st.sidebar.number_input("فروش تلفنی روزانه", 0, 500, 10, 1)
    online = st.sidebar.number_input("فروش آنلاین ماهانه", 0, 5_000, 20, 5)
    backlog = st.sidebar.number_input("تعداد سرنخ‌های در صف", 0, 500_000, 4_000, 100)
    stories = st.sidebar.number_input("تعداد استوری روزانه", 0, 100, 9, 1)
    content_sales = st.sidebar.number_input("فروش منتسب به محتوا در روز", 0.0, 100.0, 2.0, 0.5)
    customers = st.sidebar.number_input("تعداد مشتری آزمایشی", 500, 50_000, 5_000, 500)
    months = st.sidebar.slider("ماه‌های داده تاریخی", 3, 36, 12, 1)
    sales_days = st.sidebar.slider("روزهای فروش در ماه", 20, 31, 30, 1)
    seed = st.sidebar.number_input("بذر تولید داده", 1, 999_999, 42, 1)

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
    if st.sidebar.button("اجرای تحلیل", use_container_width=True, type="primary"):
        stages = [
            "بارگذاری داده",
            "اعتبارسنجی داده",
            "پاک‌سازی داده",
            "محاسبه شاخص‌های کلیدی",
            "تحلیل روندها",
            "تشخیص ناهنجاری‌ها",
            "اجرای مدل‌ها",
            "تولید بینش‌ها",
            "تحلیل کامل شد",
        ]
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()
        for i, stage in enumerate(stages, start=1):
            status.caption(stage)
            progress.progress(int(i / len(stages) * 100))
            time.sleep(0.04)
        st.session_state["analysis_complete"] = True
        status.success("تحلیل کامل شد")

    st.sidebar.caption("نسخه آزمایشی با داده‌های مصنوعی/دمو؛ بدون اتصال به سیستم‌های داخلی NIK.")
    return scenario, page


def page_header(title: str, subtitle: str = ""):
    st.markdown('<div class="nik-title">NIK INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="nik-subtitle">پلتفرم خودکار هوشمندی داده | نسخه ۰.۱ | نمونه اولیه</div>', unsafe_allow_html=True)
    st.markdown('<div class="demo-banner">دمو / داده مصنوعی — بدون اتصال به سیستم‌های داخلی NIK</div>', unsafe_allow_html=True)
    st.header(title)
    if subtitle:
        st.caption(subtitle)


def executive_overview(scenario, data, customers_model, kpis, monthly, funnel, forecast, insights):
    page_header("نمای کلی مدیریتی", "خلاصه‌ای که مدیر در چند ثانیه وضعیت کسب‌وکار را درک کند.")

    c1, c2, c3 = st.columns(3)
    c1.metric("درآمد ماهانه", toman(kpis["monthly_revenue"]))
    c2.metric("تعداد فروش ماهانه", fa_num(kpis["monthly_units"]))
    c3.metric("مشتریان فعال", fa_num(kpis["active_customers"]))
    c4, c5, c6 = st.columns(3)
    c4.metric("مخزن سرنخ‌ها", fa_num(kpis["lead_pool"]))
    c5.metric("میانگین قیمت فروش", toman(kpis["average_selling_price"]))
    c6.metric("تبدیل سرنخ به خرید", pct(kpis["lead_purchase_conversion"]))

    left, right = st.columns([1.45, 1])
    with left:
        fig = px.line(monthly, x="month", y="revenue", markers=True, title="روند درآمد آزمایشی", labels={"month":"ماه","revenue":"درآمد"})
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        channel = pd.DataFrame(
            {
                "کانال": ["فروش تلفنی", "فروش آنلاین"],
                "تعداد": [kpis["monthly_phone_units"], kpis["monthly_online_units"]],
            }
        )
        fig = px.pie(channel, names="کانال", values="تعداد", hole=0.58, title="ترکیب فعلی کانال‌های فروش")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    f1, f2 = st.columns([1, 1.25])
    with f1:
        fig = go.Figure(go.Funnel(y=funnel["stage"].map(FUNNEL_FA), x=funnel["count"], textinfo="value+percent initial"))
        fig.update_layout(title="قیف آزمایشی سرنخ تا خرید")
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)
    with f2:
        forecast_show = forecast.copy()
        forecast_show["series"] = forecast_show["series"].map(FORECAST_SERIES_FA).fillna(forecast_show["series"])
        fig = px.line(forecast_show, x="month", y="revenue", color="series", markers=True, title="تاریخچه درآمد و پیش‌بینی ۳ ماه آینده", labels={"month":"ماه","revenue":"درآمد","series":"نوع داده"})
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.subheader("سیگنال‌های مدیریتی")
    for item in insights[:5]:
        st.markdown(
            f'<div class="insight-card"><div class="insight-type">{item["type"]}</div><b>{item["title"]}</b><br>{item["text"]}</div>',
            unsafe_allow_html=True,
        )

    st.caption("پیش‌بینی‌ها و خروجی مدل‌های یادگیری ماشین آزمایشی هستند و نباید مبنای تصمیم‌گیری عملیاتی قطعی قرار گیرند.")


def data_center_page(data: Dict[str, pd.DataFrame]):
    page_header("مرکز داده", "به‌صورت پیش‌فرض داده‌های مصنوعی فعال هستند؛ امکان بارگذاری CSV نیز برای نمونه اولیه وجود دارد.")
    st.info("فایل‌های CSV به هیچ API یا دیتابیس NIK ارسال نمی‌شوند. فقط اگر ساختار فایل با الگوی مورد انتظار سازگار باشد، داده مصنوعی همان بخش جایگزین می‌شود.")

    names = ["sales", "customers", "leads", "sms", "nikpos"]
    uploaded = {}
    cols = st.columns(5)
    for col, name in zip(cols, names):
        with col:
            uploaded[name] = st.file_uploader(f"بارگذاری فایل CSV مربوط به { {"sales":"فروش","customers":"مشتریان","leads":"سرنخ‌ها","sms":"پیامک","nikpos":"نیک‌پوز"}[name] }", type=["csv"], key=f"up_{name}")

    active = data.copy()
    for name, up in uploaded.items():
        if up is not None:
            try:
                df = load_uploaded_csv(up)
                valid, msg = validate_upload(name, df)
                if valid:
                    active[name] = df
                    st.success(f"{msg}. فایل بارگذاری‌شده فعال شد.")
                else:
                    st.warning(f"{msg}. داده مصنوعی همچنان فعال است.")
            except Exception as exc:
                st.error(f"خواندن فایل CSV ممکن نبود: {exc}")

    q = quality_table(active)
    q_show = q.copy()
    q_show["Dataset"] = q_show["Dataset"].map(DATASET_FA).fillna(q_show["Dataset"])
    q_show["Status"] = q_show["Status"].map(QUALITY_STATUS_FA).fillna(q_show["Status"])
    q_show["Quality Score"] = q_show["Quality Score"].map(pct)
    q_show = q_show.rename(columns={"Dataset":"مجموعه داده","Record Count":"تعداد رکورد","Missing Values":"مقادیر خالی","Duplicate Records":"رکورد تکراری","Invalid Values":"مقادیر نامعتبر","Quality Score":"امتیاز کیفیت","Status":"وضعیت"})
    st.subheader("کیفیت داده")
    st.dataframe(q_show, use_container_width=True, hide_index=True)

    st.subheader("مرور داده‌ها")
    selected = st.selectbox("مجموعه داده", list(active.keys()), format_func=lambda x: {"sales":"فروش","customers":"مشتریان","leads":"سرنخ‌ها","sms":"پیامک","nikpos":"نیک‌پوز","subscriptions":"اشتراک‌ها"}.get(x, x))
    preview = active[selected].head(200).copy()
    if "channel" in preview.columns:
        preview["channel"] = preview["channel"].map(CHANNEL_FA).fillna(preview["channel"])
    if "plan" in preview.columns:
        preview["plan"] = preview["plan"].map(PLAN_FA).fillna(preview["plan"])
    if "stage" in preview.columns:
        preview["stage"] = preview["stage"].map(FUNNEL_FA).fillna(preview["stage"])
    if "risk_level" in preview.columns:
        preview["risk_level"] = preview["risk_level"].map(RISK_FA).fillna(preview["risk_level"])
    preview = preview.rename(columns={c: DATAFRAME_COL_FA.get(c, c) for c in preview.columns})
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.caption(f"نمایش ۲۰۰ ردیف اول از {fa_num(len(active[selected]))} رکورد.")


def sales_analytics_page(scenario, data, kpis, daily, monthly):
    page_header("تحلیل فروش")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("فروش تلفنی روزانه", fa_num(scenario.daily_phone_sales))
    c2.metric("فروش تلفنی ماهانه", fa_num(kpis["monthly_phone_units"]))
    c3.metric("فروش آنلاین ماهانه", fa_num(kpis["monthly_online_units"]))
    c4.metric("درآمد سناریوی فعلی", toman(kpis["monthly_revenue"]))

    if len(monthly) >= 2:
        mom = monthly.iloc[-1]["revenue"] / monthly.iloc[-2]["revenue"] - 1 if monthly.iloc[-2]["revenue"] else 0
        st.metric("تغییر درآمد نسبت به ماه قبل", pct(mom))

    left, right = st.columns(2)
    with left:
        fig = px.line(daily.tail(90), x="date", y="units", title="فروش روزانه — ۹۰ روز اخیر", labels={"date":"تاریخ","units":"تعداد فروش"})
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        fig = px.line(monthly, x="month", y="revenue", markers=True, title="درآمد ماهانه", labels={"month":"ماه","revenue":"درآمد"})
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    sales = data["sales"].copy()
    if not sales.empty:
        by_channel = sales.groupby("channel", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))
        by_channel["channel"] = by_channel["channel"].map(CHANNEL_FA).fillna(by_channel["channel"])
        by_plan = sales.groupby("plan", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum"))
        by_plan["plan"] = by_plan["plan"].map(PLAN_FA).fillna(by_plan["plan"])
        l, r = st.columns(2)
        with l:
            fig = px.bar(by_channel, x="channel", y="units", title="تعداد فروش تاریخی بر اساس کانال", labels={"channel":"کانال","units":"تعداد فروش"})
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with r:
            fig = px.bar(by_plan, x="plan", y="revenue", title="درآمد تاریخی بر اساس طرح", labels={"plan":"طرح","revenue":"درآمد"})
            st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("اقتصاد طرح‌های فعلی")
    pp = plan_performance(scenario).copy()
    pp["plan"] = pp["plan"].map(PLAN_FA).fillna(pp["plan"])
    pp["share"] = pp["share"].map(pct)
    pp["unit_price"] = pp["unit_price"].map(toman)
    pp["revenue"] = pp["revenue"].map(toman)
    pp["units"] = pp["units"].map(lambda x: fa_num(x))
    pp = pp.rename(columns={"plan":"طرح","units":"تعداد","share":"سهم","unit_price":"قیمت واحد","revenue":"درآمد"})
    st.dataframe(pp, use_container_width=True, hide_index=True)


def customer_intelligence_page(customers_model, segment_profile, risk_stats):
    page_header("هوشمندی مشتریان", "بخش‌بندی مشتریان به روش RFM و مدل آزمایشی ریسک ریزش.")
    c1, c2, c3 = st.columns(3)
    c1.metric("مشتریان آزمایشی", fa_num(len(customers_model)))
    c2.metric("ریسک زیاد / بسیار زیاد", pct(risk_stats["high_or_very_high_share"]))
    c3.metric("AUC آزمایشی مدل", fa_digits(f"{risk_stats['synthetic_holdout_auc']:.3f}"))

    left, right = st.columns(2)
    with left:
        seg = customers_model["segment"].value_counts().rename_axis("segment").reset_index(name="customers")
        seg["segment"] = seg["segment"].map(SEGMENT_FA).fillna(seg["segment"])
        fig = px.bar(seg, x="segment", y="customers", title="بخش‌بندی آزمایشی مشتریان با RFM", labels={"segment": "بخش مشتری", "customers": "تعداد مشتری"})
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        risk = customers_model["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="customers")
        risk["risk_level"] = risk["risk_level"].map(RISK_FA).fillna(risk["risk_level"])
        fig = px.pie(risk, names="risk_level", values="customers", hole=0.55, title="ریسک آزمایشی ریزش مشتری")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("پروفایل بخش‌های مشتریان")
    profile = segment_profile[["segment", "customers", "recency", "frequency", "monetary_value"]].copy()
    profile["segment"] = profile["segment"].map(SEGMENT_FA).fillna(profile["segment"])
    profile["recency"] = profile["recency"].round(1)
    profile["frequency"] = profile["frequency"].round(2)
    profile["monetary_value"] = profile["monetary_value"].map(toman)
    profile = profile.rename(columns={"segment":"بخش مشتری","customers":"تعداد مشتری","recency":"روز از آخرین فعالیت","frequency":"تکرار خرید","monetary_value":"ارزش مالی"})
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("مشتریان نیازمند توجه")
    cols = ["customer_id", "city", "industry", "segment", "recency", "frequency", "monetary_value", "risk_score", "risk_level"]
    risk_table = customers_model.sort_values("risk_score", ascending=False)[cols].head(100).copy()
    risk_table["segment"] = risk_table["segment"].map(SEGMENT_FA).fillna(risk_table["segment"])
    risk_table["risk_level"] = risk_table["risk_level"].map(RISK_FA).fillna(risk_table["risk_level"])
    risk_table["monetary_value"] = risk_table["monetary_value"].map(toman)
    risk_table = risk_table.rename(columns={"customer_id":"شناسه مشتری","city":"شهر","industry":"صنعت","segment":"بخش مشتری","recency":"روز از آخرین فعالیت","frequency":"تکرار خرید","monetary_value":"ارزش مالی","risk_score":"امتیاز ریسک","risk_level":"سطح ریسک"})
    st.dataframe(risk_table, use_container_width=True, hide_index=True)
    st.warning("مدل آزمایشی ریسک ریزش: رگرسیون لجستیک روی برچسب‌های مصنوعی آموزش دیده است. AUC نمایش‌داده‌شده فقط برای دمو است و اعتبارسنجی عملیاتی محسوب نمی‌شود.")


def nikpos_page(scenario, data):
    page_header("تحلیل نیک‌پوز")
    plans = plan_performance(scenario)
    devices = data["nikpos"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("تعداد فروش طرح A", fa_num(plans.loc[plans.plan == "Plan A", "units"].iloc[0]))
    c2.metric("تعداد فروش طرح B", fa_num(plans.loc[plans.plan == "Plan B", "units"].iloc[0]))
    c3.metric("میانگین قیمت فروش", toman(scenario.average_selling_price))
    c4.metric("دستگاه‌های آزمایشی فعال", pct(devices["active_device"].mean()))

    l, r = st.columns(2)
    with l:
        plans_show = plans.copy()
        plans_show["plan"] = plans_show["plan"].map(PLAN_FA).fillna(plans_show["plan"])
        fig = px.bar(plans_show, x="plan", y="revenue", title="درآمد سناریوی فعلی بر اساس طرح", labels={"plan":"طرح","revenue":"درآمد"})
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with r:
        city = devices.groupby("city", as_index=False).agg(devices=("device_id", "count"), captures_30d=("captures_30d", "sum"))
        city = city.sort_values("captures_30d", ascending=False).head(10)
        fig = px.bar(city, x="city", y="captures_30d", title="ثبت شماره آزمایشی ۳۰ روزه بر اساس شهر", labels={"city":"شهر","captures_30d":"تعداد ثبت شماره"})
        st.plotly_chart(style_fig(fig), use_container_width=True)

    monthly = data["sales"].copy()
    if not monthly.empty:
        monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M").dt.to_timestamp()
        trend = monthly.groupby(["month", "plan"], as_index=False).agg(units=("units", "sum"))
        trend["plan"] = trend["plan"].map(PLAN_FA).fillna(trend["plan"])
        fig = px.line(trend, x="month", y="units", color="plan", markers=True, title="روند ماهانه فروش نیک‌پوز بر اساس طرح", labels={"month":"ماه","units":"تعداد","plan":"طرح"})
        st.plotly_chart(style_fig(fig), use_container_width=True)


def content_page(scenario):
    page_header("تحلیل محتوا", "انتساب فروش به محتوا در این نسخه فقط تخمینی و آزمایشی است.")
    m = content_metrics(scenario)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("استوری در روز", fa_num(m["stories_per_day"]))
    c2.metric("استوری در ماه", fa_num(m["stories_per_month"]))
    c3.metric("فروش تخمینی روزانه", fa_num(m["estimated_sales_per_day"], 1))
    c4.metric("فروش تخمینی ماهانه", fa_num(m["estimated_sales_per_month"]))

    st.markdown(
        f'<div class="insight-card"><div class="insight-type">انتساب تخمینی / آزمایشی</div>'
        f'<b>شاخص تقریبی فروش منتسب به محتوا</b><br>فروش تخمینی به ازای هر استوری: {fa_num(m["sales_per_story"], 2)}. '
        'این عدد نرخ تبدیل بازاریابی قطعی نیست؛ چون Reach، تعداد نمایش، شناسه منبع و بازه Attribution در دسترس نیست.</div>',
        unsafe_allow_html=True,
    )

    days = pd.DataFrame({"day": np.arange(1, scenario.sales_days_per_month + 1)})
    days["stories"] = scenario.stories_per_day
    days["estimated_sales"] = scenario.content_sales_per_day
    fig = px.line(days.rename(columns={"day":"روز","stories":"استوری","estimated_sales":"فروش تخمینی"}), x="روز", y=["استوری", "فروش تخمینی"], title="فعالیت محتوایی سناریو با فرض ماهانه", labels={"value":"مقدار","variable":"شاخص"})
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("برای انتساب واقعی فروش به محتوا باید منبع ورودی، شناسه کمپین، برچسب کانال و مدل Attribution مشخص وجود داشته باشد.")


def sms_page(data):
    page_header("تحلیل پیامک")
    sms = data["sms"].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("پیامک‌های آزمایشی ارسال‌شده", fa_num(sms["sent"].sum()))
    c2.metric("نرخ تحویل آزمایشی", pct(sms["delivered"].sum() / sms["sent"].sum()))
    c3.metric("نرخ کلیک آزمایشی", pct(sms["clicks"].sum() / max(sms["delivered"].sum(), 1)))

    l, r = st.columns(2)
    with l:
        fig = px.line(sms.tail(90), x="date", y="delivery_rate", title="نرخ تحویل پیامک — ۹۰ روز اخیر", labels={"date":"تاریخ","delivery_rate":"نرخ تحویل"})
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with r:
        fig = px.line(sms.tail(90), x="date", y="sent", title="حجم پیامک — ۹۰ روز اخیر", labels={"date":"تاریخ","sent":"تعداد پیامک"})
        fig.update_traces(line_color=GOLD)
        st.plotly_chart(style_fig(fig), use_container_width=True)


def anomaly_page(sales_anomalies, sms_anomalies):
    page_header("تشخیص ناهنجاری", "روش قابل توضیح میانگین متحرک و Z-score؛ آستانه تشخیص |Z| ≥ ۲.۵ است.")
    c1, c2 = st.columns(2)
    c1.metric("ناهنجاری‌های درآمد فروش", fa_num(int(sales_anomalies["is_anomaly"].sum())))
    c2.metric("ناهنجاری‌های تحویل پیامک", fa_num(int(sms_anomalies["is_anomaly"].sum())))

    l, r = st.columns(2)
    with l:
        fig = px.line(sales_anomalies, x="date", y="revenue", title="درآمد روزانه و نقاط غیرعادی", labels={"date":"تاریخ","revenue":"درآمد"})
        flagged = sales_anomalies[sales_anomalies["is_anomaly"]]
        fig.add_scatter(x=flagged["date"], y=flagged["revenue"], mode="markers", name="ناهنجاری", marker=dict(size=10, color=GOLD))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        fig = px.line(sms_anomalies, x="date", y="delivery_rate", title="نرخ تحویل پیامک و نقاط غیرعادی", labels={"date":"تاریخ","delivery_rate":"نرخ تحویل"})
        flagged = sms_anomalies[sms_anomalies["is_anomaly"]]
        fig.add_scatter(x=flagged["date"], y=flagged["delivery_rate"], mode="markers", name="ناهنجاری", marker=dict(size=10, color=GOLD))
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("موارد علامت‌گذاری‌شده")
    tab1, tab2 = st.tabs(["فروش", "پیامک"])
    with tab1:
        sales_flags = sales_anomalies[sales_anomalies["is_anomaly"]].sort_values("date", ascending=False).rename(columns={c: DATAFRAME_COL_FA.get(c, c) for c in sales_anomalies.columns})
        st.dataframe(sales_flags, use_container_width=True, hide_index=True)
    with tab2:
        sms_flags = sms_anomalies[sms_anomalies["is_anomaly"]].sort_values("date", ascending=False).rename(columns={c: DATAFRAME_COL_FA.get(c, c) for c in sms_anomalies.columns})
        st.dataframe(sms_flags, use_container_width=True, hide_index=True)


def predictions_page(forecast, forecast_stats, customers_model, risk_stats):
    page_header("پیش‌بینی‌ها", "فقط مدل‌های ساده، قابل توضیح و آزمایشی.")
    c1, c2 = st.columns(2)
    c1.metric("R² مدل روند درآمد", fa_digits(f"{forecast_stats['r2']:.3f}"))
    c2.metric("AUC آزمایشی ریزش", fa_digits(f"{risk_stats['synthetic_holdout_auc']:.3f}"))

    forecast_show = forecast.copy()
    forecast_show["series"] = forecast_show["series"].map(FORECAST_SERIES_FA).fillna(forecast_show["series"])
    fig = px.line(forecast_show, x="month", y="revenue", color="series", markers=True, title="درآمد تاریخی و ۳ ماه آینده", labels={"month":"ماه","revenue":"درآمد","series":"نوع داده"})
    st.plotly_chart(style_fig(fig, 470), use_container_width=True)

    st.subheader("مدل آزمایشی ریسک ریزش")
    top = customers_model.sort_values("risk_score", ascending=False)[
        ["customer_id", "recency", "frequency", "monetary_value", "sms_usage", "nikpos_usage", "risk_score", "risk_level"]
    ].head(50).copy()
    top["risk_level"] = top["risk_level"].map(RISK_FA).fillna(top["risk_level"])
    top["monetary_value"] = top["monetary_value"].map(toman)
    top = top.rename(columns={"customer_id":"شناسه مشتری","recency":"روز از آخرین فعالیت","frequency":"تکرار خرید","monetary_value":"ارزش مالی","sms_usage":"مصرف پیامک","nikpos_usage":"استفاده از نیک‌پوز","risk_score":"امتیاز ریسک","risk_level":"سطح ریسک"})
    st.dataframe(top, use_container_width=True, hide_index=True)
    st.warning("پیش‌بینی‌ها و خروجی مدل‌های یادگیری ماشین آزمایشی هستند و نباید مبنای تصمیم‌گیری عملیاتی قطعی قرار گیرند.")


def insights_page(insights):
    page_header("بینش‌های خودکار", "بینش‌های قانون‌محور که از سناریوی فعلی و خروجی تحلیل آزمایشی تولید می‌شوند.")
    for item in insights:
        st.markdown(
            f'<div class="insight-card"><div class="insight-type">{item["type"]}</div><b>{item["title"]}</b><br>{item["text"]}</div>',
            unsafe_allow_html=True,
        )


def pipeline_page():
    page_header("خط لوله تحلیل")
    stages = [
        "ورود داده",
        "اعتبارسنجی داده",
        "پاک‌سازی داده",
        "محاسبه KPI",
        "تحلیل روند",
        "تشخیص ناهنجاری",
        "بخش‌بندی مشتریان",
        "امتیازدهی ریسک",
        "پیش‌بینی",
        "تولید بینش",
        "به‌روزرسانی داشبورد",
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
        "ورودی ← داده ← تحلیل ← KPI ← نمودار ← مدل ← بینش ← به‌روزرسانی داشبورد",
        language="text",
    )
    st.caption("دکمه «اجرای تحلیل» در نوار کناری ترتیب مراحل تحلیل را نمایش می‌دهد. تغییر ورودی‌ها نیز بلافاصله کل برنامه را دوباره محاسبه می‌کند.")


def settings_page(scenario, kpis):
    page_header("تنظیمات و کنترل سناریو", "تمام مقادیر اصلی از ورودی‌های قابل تغییر نوار کناری محاسبه می‌شوند.")
    rows = [
        ("قیمت طرح A", toman(scenario.price_plan_a)),
        ("قیمت طرح B", toman(scenario.price_plan_b)),
        ("سهم طرح A", pct(scenario.plan_a_share)),
        ("سهم طرح B", pct(scenario.plan_b_share)),
        ("فروش تلفنی روزانه", fa_num(scenario.daily_phone_sales)),
        ("فروش آنلاین ماهانه", fa_num(scenario.monthly_online_sales)),
        ("تعداد سرنخ‌های در صف", fa_num(scenario.lead_backlog)),
        ("استوری در روز", fa_num(scenario.stories_per_day)),
        ("فروش منتسب به محتوا در روز", fa_num(scenario.content_sales_per_day, 1)),
        ("مشتریان آزمایشی", fa_num(scenario.synthetic_customer_count)),
        ("ماه‌های تاریخی", fa_num(scenario.history_months)),
        ("روزهای فروش در ماه", fa_num(scenario.sales_days_per_month)),
        ("بذر تولید داده", fa_num(scenario.seed)),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["تنظیم", "مقدار فعلی"]), use_container_width=True, hide_index=True)

    st.subheader("بررسی وابستگی زنده")
    st.code(
        f"فروش تلفنی روزانه = {fa_num(scenario.daily_phone_sales)}\n"
        f"فروش تلفنی ماهانه = {fa_num(kpis['monthly_phone_units'])}\n"
        f"کل فروش ماهانه = {fa_num(kpis['monthly_units'])}\n"
        f"میانگین قیمت فروش = {toman(scenario.average_selling_price)}\n"
        f"درآمد ماهانه = {toman(kpis['monthly_revenue'])}\n"
        f"تبدیل سرنخ به خرید = {pct(kpis['lead_purchase_conversion'])}",
        language="text",
    )
    st.success("برای تست دمو، فروش تلفنی روزانه را از ۱۰ به ۱۵ تغییر بده؛ درآمد، تعداد فروش، قیف، روند تاریخی، پیش‌بینی، ترکیب کانال‌ها و بینش‌های خودکار بلافاصله دوباره محاسبه می‌شوند.")


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
    st.caption("نسخه آزمایشی با داده‌های مصنوعی/دمو؛ بدون اتصال به سیستم‌های داخلی NIK.")
    st.caption("پیش‌بینی‌ها و خروجی مدل‌های یادگیری ماشین آزمایشی هستند و نباید مبنای تصمیم‌گیری عملیاتی قطعی قرار گیرند.")


if __name__ == "__main__":
    main()
