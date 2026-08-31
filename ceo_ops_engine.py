from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


CEO_NAME = "کیوان میرزایی"
CURRENT_LEAD_BACKLOG = 5_490


ORGANIZATION_ROSTER = pd.DataFrame(
    [
        ("it", "فناوری اطلاعات / برنامه‌نویسی", "حمید تهرانی", "عضو تیم IT", False),
        ("it", "فناوری اطلاعات / برنامه‌نویسی", "مسعود طاهری", "عضو تیم IT", False),
        ("it", "فناوری اطلاعات / برنامه‌نویسی", "سروش کلانتریان", "عضو تیم IT", False),
        ("it", "فناوری اطلاعات / برنامه‌نویسی", "پوریا سلیمانی", "عضو تیم IT", False),
        ("it", "فناوری اطلاعات / برنامه‌نویسی", "کوروش عذت پور", "عضو تیم IT", False),
        ("accounting", "حسابداری", "حسین جودکی", "مدیر حسابداری", True),
        ("accounting", "حسابداری", "مشتبا بیان", "کارمند حسابداری", False),
        ("hr", "منابع انسانی", "خانم مقصودی", "مسئول منابع انسانی", True),
        ("support", "پشتیبانی", "خانم ملیکا جمع دار", "سرپرست پشتیبانی", True),
        ("support", "پشتیبانی", "خانم آزاد", "نیروی پشتیبانی", False),
        ("marketing", "مارکتینگ", "امیر عباس حبیبی", "سرپرست مارکتینگ", True),
        ("marketing", "مارکتینگ", "دانیال توکل", "عضو تیم مارکتینگ", False),
        ("marketing", "مارکتینگ", "سحر نور محمدی", "عضو تیم مارکتینگ", False),
        ("marketing", "مارکتینگ", "داریوش مشتاقی", "عضو تیم مارکتینگ", False),
        ("marketing", "مارکتینگ", "امین عین آبادی", "عضو تیم مارکتینگ", False),
    ],
    columns=["department", "department_name", "name", "role", "is_lead"],
)

DEPARTMENT_ORDER = ["it", "accounting", "sales", "support", "hr", "qc", "marketing"]
DEPARTMENT_NAMES = {
    "it": "فناوری اطلاعات / برنامه‌نویسی",
    "accounting": "حسابداری",
    "sales": "فروش تلفنی",
    "support": "پشتیبانی",
    "hr": "منابع انسانی",
    "qc": "QC دستگاه",
    "marketing": "مارکتینگ",
}


DEFAULT_SALES_AGENTS = pd.DataFrame(
    [
        ("کارشناس فروش ۱", True, 55, 0.11, 0),
        ("کارشناس فروش ۲", True, 50, 0.10, 0),
        ("کارشناس فروش ۳", True, 45, 0.09, 0),
        ("کارشناس فروش ۴", True, 40, 0.08, 0),
    ],
    columns=["کارشناس", "فعال", "ظرفیت لید روزانه", "نرخ تبدیل هدف", "لید تخصیص‌یافته فعلی"],
)


TASK_COLUMNS = [
    "شناسه",
    "بخش",
    "عنوان",
    "مسئول",
    "ایجادکننده",
    "منبع",
    "اولویت",
    "وضعیت",
    "KPI",
    "زمان ایجاد",
    "موعد",
    "بازه پیگیری (ساعت)",
    "آخرین گزارش",
    "گزارش اجباری",
    "یادداشت",
]


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=None).floor("min")


def seed_ceo_tasks(now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = pd.Timestamp(now) if now is not None else _now()
    rows = [
        (
            "T-1001", "فناوری اطلاعات / برنامه‌نویسی", "گزارش روزانه وضعیت آپدیت‌های جدید و Blockerها",
            "تیم IT", CEO_NAME, "مدیریت", "بالا", "در حال انجام", "درصد گزارش‌های به‌موقع",
            now - pd.Timedelta(hours=20), now + pd.Timedelta(hours=4), 24, now - pd.Timedelta(hours=18), True,
            "موضوع جاری: آپدیت‌های جدید و هماهنگی تیم برنامه‌نویسی.",
        ),
        (
            "T-1002", "فناوری اطلاعات / برنامه‌نویسی", "نهایی‌کردن Scope قالب ظاهری جدید سایت و معیار پذیرش",
            "تیم IT", CEO_NAME, "مدیریت", "بالا", "در حال انجام", "آمادگی Release",
            now - pd.Timedelta(days=1), now + pd.Timedelta(days=2), 24, now - pd.Timedelta(hours=22), True,
            "خروجی باید قبل/بعد، Owner و زمان Release داشته باشد.",
        ),
        (
            "T-1003", "فروش تلفنی", "طراحی Stage و SLA برای صف ۵۴۹۰ لید و پخش منصفانه بین کارشناسان",
            "سرپرست فروش", "ربات مدیریتی", "Rule-based", "بالا", "پیشنهاد", "Lead → Contact → Sale",
            now - pd.Timedelta(hours=2), now + pd.Timedelta(days=1), 12, pd.NaT, True,
            "نام کارشناسان فروش هنوز ثبت نشده؛ در پنل قابل ویرایش است.",
        ),
        (
            "T-1004", "حسابداری", "تعریف ورودی تراکنش، کدینگ رسمی و قواعد ثبت خودکار دوطرفه",
            "حسین جودکی", "ربات مدیریتی", "Rule-based", "بالا", "پیشنهاد", "Auto-post Rate",
            now - pd.Timedelta(hours=1), now + pd.Timedelta(days=3), 24, pd.NaT, True,
            "هدف: ثبت ماشینی تراکنش‌های مطمئن و ارسال فقط استثناها برای بررسی انسانی.",
        ),
        (
            "T-1005", "پشتیبانی", "تعریف گزارش روزانه Backlog، SLA و مسائل پرتکرار برای مدیرعامل",
            "خانم ملیکا جمع دار", "ربات مدیریتی", "Rule-based", "متوسط", "پیشنهاد", "SLA پاسخگویی",
            now - pd.Timedelta(hours=1), now + pd.Timedelta(days=2), 24, pd.NaT, True,
            "گزارش باید خلاصه و Exception-based باشد.",
        ),
    ]
    return pd.DataFrame(rows, columns=TASK_COLUMNS)


def new_task_row(
    department: str,
    title: str,
    assignee: str,
    priority: str = "متوسط",
    kpi: str = "—",
    source: str = "مدیرعامل",
    creator: str = CEO_NAME,
    due_hours: int = 24,
    followup_hours: int = 24,
    note: str = "",
) -> pd.DataFrame:
    now = _now()
    task_id = f"T-{uuid4().hex[:6].upper()}"
    row = [[
        task_id,
        department,
        title.strip(),
        assignee.strip() or "سرپرست واحد",
        creator,
        source,
        priority,
        "Backlog",
        kpi.strip() or "—",
        now,
        now + pd.Timedelta(hours=max(int(due_hours), 1)),
        max(int(followup_hours), 1),
        pd.NaT,
        True,
        note,
    ]]
    return pd.DataFrame(row, columns=TASK_COLUMNS)


def task_followup_status(tasks: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if tasks is None or tasks.empty:
        return pd.DataFrame(columns=list(TASK_COLUMNS) + ["موعد گذشته", "گزارش عقب‌افتاده", "نیازمند پیگیری", "پیگیری بعدی"])
    now = pd.Timestamp(now) if now is not None else _now()
    out = tasks.copy()
    out["موعد"] = pd.to_datetime(out["موعد"], errors="coerce")
    out["آخرین گزارش"] = pd.to_datetime(out["آخرین گزارش"], errors="coerce")
    closed = out["وضعیت"].astype(str).isin(["انجام شد", "بسته", "Done"])
    out["موعد گذشته"] = (~closed) & out["موعد"].notna() & (out["موعد"] < now)
    last = out["آخرین گزارش"].fillna(pd.to_datetime(out["زمان ایجاد"], errors="coerce"))
    followup = pd.to_numeric(out["بازه پیگیری (ساعت)"], errors="coerce").fillna(24).clip(lower=1)
    out["پیگیری بعدی"] = last + pd.to_timedelta(followup, unit="h")
    report_required = out["گزارش اجباری"].fillna(False).astype(bool)
    out["گزارش عقب‌افتاده"] = (~closed) & report_required & (out["پیگیری بعدی"] < now)
    out["نیازمند پیگیری"] = out["موعد گذشته"] | out["گزارش عقب‌افتاده"]
    return out


def task_summary(tasks: pd.DataFrame, now: pd.Timestamp | None = None) -> Dict[str, int]:
    frame = task_followup_status(tasks, now)
    if frame.empty:
        return {"total": 0, "open": 0, "overdue": 0, "report_overdue": 0, "needs_followup": 0}
    closed = frame["وضعیت"].astype(str).isin(["انجام شد", "بسته", "Done"])
    return {
        "total": int(len(frame)),
        "open": int((~closed).sum()),
        "overdue": int(frame["موعد گذشته"].sum()),
        "report_overdue": int(frame["گزارش عقب‌افتاده"].sum()),
        "needs_followup": int(frame["نیازمند پیگیری"].sum()),
    }


def department_report_schedule(now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = pd.Timestamp(now) if now is not None else _now()
    rows = [
        ("فناوری اطلاعات / برنامه‌نویسی", "روزانه", 24, now - pd.Timedelta(hours=22), "حمید / تیم IT", "Demo"),
        ("حسابداری", "روزانه", 24, now - pd.Timedelta(hours=30), "حسین جودکی", "Demo"),
        ("فروش تلفنی", "هر ۶ ساعت", 6, now - pd.Timedelta(hours=9), "سرپرست فروش", "Demo"),
        ("پشتیبانی", "روزانه", 24, now - pd.Timedelta(hours=16), "خانم ملیکا جمع دار", "Demo"),
        ("منابع انسانی", "هفتگی", 168, now - pd.Timedelta(days=3), "خانم مقصودی", "Demo"),
        ("QC دستگاه", "روزانه", 24, now - pd.Timedelta(hours=27), "سرپرست QC", "Demo"),
        ("مارکتینگ", "روزانه", 24, now - pd.Timedelta(hours=19), "امیر عباس حبیبی", "Demo"),
    ]
    df = pd.DataFrame(rows, columns=["بخش", "تناوب", "SLA ساعت", "آخرین گزارش", "مسئول گزارش", "منبع"])
    df["موعد بعدی"] = df["آخرین گزارش"] + pd.to_timedelta(df["SLA ساعت"], unit="h")
    df["عقب‌افتاده"] = df["موعد بعدی"] < now
    df["وضعیت"] = np.where(df["عقب‌افتاده"], "نیازمند گزارش", "به‌موقع")
    return df


def reporting_summary(reports: pd.DataFrame) -> Dict[str, int]:
    if reports is None or reports.empty:
        return {"departments": 0, "overdue": 0, "on_time": 0}
    overdue = int(reports["عقب‌افتاده"].fillna(False).astype(bool).sum())
    return {"departments": int(len(reports)), "overdue": overdue, "on_time": int(len(reports) - overdue)}


# ---------------- Accounting automation prototype ----------------

FINANCE_RULES: List[Dict[str, Any]] = [
    {"keywords": ["فروش دستگاه", "پرداخت آنلاین", "خرید نیک پوز"], "direction": "ورودی", "debit": "بانک / درگاه", "credit": "درآمد فروش", "confidence": 0.99},
    {"keywords": ["واریز مشتری", "وصول فاکتور"], "direction": "ورودی", "debit": "بانک", "credit": "حساب‌های دریافتنی", "confidence": 0.97},
    {"keywords": ["حقوق", "دستمزد"], "direction": "خروجی", "debit": "هزینه حقوق و دستمزد", "credit": "بانک", "confidence": 0.99},
    {"keywords": ["تبلیغات", "کمپین", "مارکتینگ"], "direction": "خروجی", "debit": "هزینه بازاریابی", "credit": "بانک", "confidence": 0.97},
    {"keywords": ["قطعه", "برد", "تامین کننده"], "direction": "خروجی", "debit": "موجودی / خرید قطعات", "credit": "بانک", "confidence": 0.95},
    {"keywords": ["ارسال", "پست", "حمل"], "direction": "خروجی", "debit": "هزینه ارسال", "credit": "بانک", "confidence": 0.97},
    {"keywords": ["اجاره"], "direction": "خروجی", "debit": "هزینه اجاره", "credit": "بانک", "confidence": 0.99},
]


def generate_demo_transactions(count: int = 72, seed: int = 81) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    now = _now()
    templates = [
        ("ورودی", "فروش دستگاه نیک پوز - پرداخت آنلاین", 15_000_000, 30_000_000),
        ("ورودی", "واریز مشتری بابت فاکتور", 5_000_000, 60_000_000),
        ("خروجی", "پرداخت حقوق و دستمزد", 12_000_000, 120_000_000),
        ("خروجی", "هزینه تبلیغات و کمپین", 3_000_000, 80_000_000),
        ("خروجی", "خرید قطعه و برد از تامین کننده", 8_000_000, 160_000_000),
        ("خروجی", "هزینه ارسال و پست سفارش", 500_000, 8_000_000),
        ("خروجی", "پرداخت اجاره", 25_000_000, 90_000_000),
        ("خروجی", "پرداخت متفرقه بدون شرح کافی", 1_000_000, 25_000_000),
    ]
    rows = []
    for i in range(max(int(count), 1)):
        direction, desc, low, high = templates[int(rng.integers(0, len(templates)))]
        amount = float(rng.integers(int(low), int(high) + 1))
        occurred_at = now - pd.Timedelta(hours=float(rng.uniform(0, 24 * 10)))
        source = rng.choice(["بانک", "درگاه پرداخت", "فایل حسابداری", "صندوق"])
        counterparty = rng.choice(["مشتری", "تامین‌کننده", "پرسنل", "پست", "پلتفرم تبلیغاتی", "نامشخص"])
        invoice_match = bool(rng.random() > (0.08 if "فروش" in desc or "فاکتور" in desc else 0.18))
        bank_match = bool(rng.random() > 0.06)
        external_id = f"TX-{occurred_at.strftime('%m%d')}-{i:04d}"
        idem = sha1(f"{external_id}|{amount}|{direction}".encode("utf-8")).hexdigest()[:14]
        rows.append((external_id, occurred_at, source, direction, desc, counterparty, amount, invoice_match, bank_match, idem))
    df = pd.DataFrame(rows, columns=["شناسه تراکنش", "زمان", "منبع", "جهت", "شرح", "طرف حساب", "مبلغ", "تطبیق سند", "تطبیق بانک", "کلید یکتا"])
    # Create a tiny duplicate signal for the control layer, not a real company claim.
    if len(df) > 15:
        df.loc[df.index[-1], "کلید یکتا"] = df.loc[df.index[-2], "کلید یکتا"]
    return df.sort_values("زمان", ascending=False).reset_index(drop=True)


def classify_finance_transactions(transactions: pd.DataFrame, approval_threshold: float = 50_000_000) -> pd.DataFrame:
    if transactions is None or transactions.empty:
        return pd.DataFrame()
    out = transactions.copy()
    duplicate_keys = out["کلید یکتا"].duplicated(keep=False)
    debit_accounts: List[str] = []
    credit_accounts: List[str] = []
    confidences: List[float] = []
    mapped: List[bool] = []
    for _, row in out.iterrows():
        desc = str(row.get("شرح", ""))
        direction = str(row.get("جهت", ""))
        match = None
        for rule in FINANCE_RULES:
            if direction != rule["direction"]:
                continue
            if any(keyword in desc for keyword in rule["keywords"]):
                match = rule
                break
        if match:
            debit_accounts.append(str(match["debit"]))
            credit_accounts.append(str(match["credit"]))
            confidences.append(float(match["confidence"]))
            mapped.append(True)
        else:
            debit_accounts.append("نیازمند تعیین حساب")
            credit_accounts.append("نیازمند تعیین حساب")
            confidences.append(0.45)
            mapped.append(False)

    out["حساب بدهکار"] = debit_accounts
    out["حساب بستانکار"] = credit_accounts
    out["اطمینان طبقه‌بندی"] = confidences
    out["کدینگ مشخص"] = mapped
    out["تکراری"] = duplicate_keys.values
    out["مبلغ بالای آستانه"] = pd.to_numeric(out["مبلغ"], errors="coerce").fillna(0) > float(max(approval_threshold, 0))
    out["قابل Auto-post"] = (
        out["کدینگ مشخص"].astype(bool)
        & (out["اطمینان طبقه‌بندی"] >= 0.95)
        & out["تطبیق بانک"].fillna(False).astype(bool)
        & out["تطبیق سند"].fillna(False).astype(bool)
        & (~out["تکراری"].astype(bool))
        & (~out["مبلغ بالای آستانه"].astype(bool))
    )

    reasons = []
    for _, r in out.iterrows():
        problem = []
        if not bool(r["کدینگ مشخص"]): problem.append("کدینگ نامشخص")
        if float(r["اطمینان طبقه‌بندی"]) < 0.95: problem.append("اطمینان پایین")
        if not bool(r["تطبیق بانک"]): problem.append("عدم تطبیق بانک")
        if not bool(r["تطبیق سند"]): problem.append("سند/فاکتور نامعتبر یا ناموجود")
        if bool(r["تکراری"]): problem.append("احتمال تراکنش تکراری")
        if bool(r["مبلغ بالای آستانه"]): problem.append("نیازمند تأیید مبلغ بالا")
        reasons.append("، ".join(problem) if problem else "ثبت خودکار")
    out["مسیر کنترل"] = reasons
    out["وضعیت ثبت"] = np.where(out["قابل Auto-post"], "آماده ثبت خودکار", "صف استثنا")
    return out


def finance_automation_summary(classified: pd.DataFrame) -> Dict[str, float]:
    if classified is None or classified.empty:
        return {
            "transactions": 0, "auto_post": 0, "exceptions": 0, "auto_post_rate": 0.0,
            "inflow": 0.0, "outflow": 0.0, "net_cash": 0.0, "unreconciled": 0,
        }
    auto = classified["قابل Auto-post"].fillna(False).astype(bool)
    amounts = pd.to_numeric(classified["مبلغ"], errors="coerce").fillna(0.0)
    inflow = float(amounts[classified["جهت"].astype(str) == "ورودی"].sum())
    outflow = float(amounts[classified["جهت"].astype(str) == "خروجی"].sum())
    unreconciled = int((~classified["تطبیق بانک"].fillna(False).astype(bool)).sum())
    return {
        "transactions": int(len(classified)),
        "auto_post": int(auto.sum()),
        "exceptions": int((~auto).sum()),
        "auto_post_rate": float(auto.mean()),
        "inflow": inflow,
        "outflow": outflow,
        "net_cash": inflow - outflow,
        "unreconciled": unreconciled,
    }


# ---------------- Sales lead routing prototype ----------------


def normalize_sales_agents(agents: pd.DataFrame | None = None) -> pd.DataFrame:
    base = DEFAULT_SALES_AGENTS.copy() if agents is None or agents.empty else agents.copy()
    required = DEFAULT_SALES_AGENTS.columns.tolist()
    for col in required:
        if col not in base.columns:
            base[col] = DEFAULT_SALES_AGENTS[col].iloc[0] if col in DEFAULT_SALES_AGENTS else 0
    base = base[required].copy()
    base["فعال"] = base["فعال"].fillna(True).astype(bool)
    base["ظرفیت لید روزانه"] = pd.to_numeric(base["ظرفیت لید روزانه"], errors="coerce").fillna(0).clip(lower=0)
    base["نرخ تبدیل هدف"] = pd.to_numeric(base["نرخ تبدیل هدف"], errors="coerce").fillna(0.08).clip(lower=0.01, upper=0.80)
    base["لید تخصیص‌یافته فعلی"] = pd.to_numeric(base["لید تخصیص‌یافته فعلی"], errors="coerce").fillna(0).clip(lower=0)
    base["کارشناس"] = base["کارشناس"].fillna("کارشناس فروش").astype(str)
    return base


def allocate_lead_backlog(backlog: int, agents: pd.DataFrame | None = None) -> pd.DataFrame:
    backlog = max(int(backlog), 0)
    frame = normalize_sales_agents(agents)
    active = frame[frame["فعال"]].copy()
    if active.empty:
        return frame.assign(**{"سهم پیشنهادی صف": 0, "Batch روزانه پیشنهادی": 0, "روز تقریبی برای تماس اولیه": np.nan})
    # Capacity is the primary factor. Conversion target lightly prioritizes stronger desks without starving others.
    quality = np.clip(active["نرخ تبدیل هدف"].to_numpy(dtype=float), 0.02, 0.50)
    capacity = active["ظرفیت لید روزانه"].to_numpy(dtype=float)
    weights = capacity * (0.75 + quality)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    exact = backlog * weights / weights.sum()
    allocation = np.floor(exact).astype(int)
    remaining = backlog - int(allocation.sum())
    if remaining > 0:
        order = np.argsort(-(exact - allocation))
        for idx in order[:remaining]:
            allocation[idx] += 1
    active["سهم پیشنهادی صف"] = allocation
    active["Batch روزانه پیشنهادی"] = active["ظرفیت لید روزانه"].round().astype(int)
    active["روز تقریبی برای تماس اولیه"] = np.where(
        active["ظرفیت لید روزانه"] > 0,
        np.ceil(active["سهم پیشنهادی صف"] / active["ظرفیت لید روزانه"]),
        np.nan,
    )
    inactive = frame[~frame["فعال"]].copy()
    if not inactive.empty:
        inactive["سهم پیشنهادی صف"] = 0
        inactive["Batch روزانه پیشنهادی"] = 0
        inactive["روز تقریبی برای تماس اولیه"] = np.nan
    return pd.concat([active, inactive], ignore_index=True)


def generate_sales_performance_demo(agents: pd.DataFrame | None = None, days: int = 30, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = normalize_sales_agents(agents)
    rows = []
    for _, agent in frame.iterrows():
        if not bool(agent["فعال"]):
            rows.append((agent["کارشناس"], 0, 0, 0, 0, 0, 0.0))
            continue
        capacity = float(agent["ظرفیت لید روزانه"])
        received = int(round(capacity * max(days, 1) * rng.uniform(0.72, 0.96)))
        contacted = int(round(received * rng.uniform(0.62, 0.86)))
        target_conv = float(agent["نرخ تبدیل هدف"])
        won = int(round(contacted * np.clip(rng.normal(target_conv, 0.015), 0.02, 0.45)))
        lost = int(round(contacted * rng.uniform(0.28, 0.46)))
        followup = max(contacted - won - lost, 0)
        conversion = won / received if received else 0.0
        rows.append((agent["کارشناس"], received, contacted, won, lost, followup, conversion))
    return pd.DataFrame(rows, columns=["کارشناس", "لید دریافتی", "تماس‌شده", "فروش موفق", "ناموفق", "پیگیری", "نرخ تبدیل"])


def lead_center_summary(backlog: int, agents: pd.DataFrame | None = None) -> Dict[str, float]:
    allocation = allocate_lead_backlog(backlog, agents)
    active = allocation[allocation["فعال"]].copy() if not allocation.empty else allocation
    daily_capacity = float(active["ظرفیت لید روزانه"].sum()) if not active.empty else 0.0
    days = float(np.ceil(backlog / daily_capacity)) if daily_capacity > 0 else np.nan
    return {
        "backlog": int(max(backlog, 0)),
        "active_agents": int(len(active)),
        "daily_contact_capacity": daily_capacity,
        "first_touch_days_proxy": days,
    }


# ---------------- CEO management inbox ----------------


def robot_task_suggestions(
    lead_backlog: int,
    tasks: pd.DataFrame,
    reports: pd.DataFrame,
    finance_summary: Dict[str, float] | None = None,
) -> pd.DataFrame:
    suggestions: List[Dict[str, Any]] = []
    finance_summary = finance_summary or {}
    if int(lead_backlog) >= 5_000:
        suggestions.append({
            "بخش": "فروش تلفنی", "عنوان": f"تقسیم صف {int(lead_backlog):,} لید بر اساس ظرفیت + SLA تماس اولیه",
            "مسئول": "سرپرست فروش", "اولویت": "بالا", "KPI": "Lead First-touch SLA",
            "دلیل": "حجم صف بالاست و بدون Routing و Outcome Tracking مدیریت ظرفیت ممکن نیست.",
        })
    task_stats = task_summary(tasks)
    if task_stats["report_overdue"] > 0:
        suggestions.append({
            "بخش": "مدیریت شرکت", "عنوان": "فعال‌کردن پیگیری خودکار گزارش تسک‌های بدون Update",
            "مسئول": CEO_NAME, "اولویت": "بالا", "KPI": "گزارش‌های به‌موقع",
            "دلیل": f"{task_stats['report_overdue']} تسک در نمونه فعلی گزارش به‌موقع ندارد.",
        })
    if reports is not None and not reports.empty:
        overdue = reports[reports["عقب‌افتاده"].fillna(False).astype(bool)]
        for _, row in overdue.head(3).iterrows():
            suggestions.append({
                "بخش": str(row["بخش"]), "عنوان": "ارسال Reminder و درخواست گزارش خلاصه مدیریتی",
                "مسئول": str(row["مسئول گزارش"]), "اولویت": "متوسط", "KPI": "Report SLA",
                "دلیل": "گزارش Demo از SLA تعریف‌شده عبور کرده است.",
            })
    if int(finance_summary.get("exceptions", 0)) > 0:
        suggestions.append({
            "بخش": "حسابداری", "عنوان": "پاک‌سازی صف استثناهای ثبت خودکار و تکمیل قواعد کدینگ",
            "مسئول": "حسین جودکی", "اولویت": "بالا", "KPI": "Auto-post Rate",
            "دلیل": f"{int(finance_summary.get('exceptions', 0))} تراکنش Demo هنوز برای ثبت خودکار آماده نیست.",
        })
    suggestions.append({
        "بخش": "فناوری اطلاعات / برنامه‌نویسی", "عنوان": "تعریف Daily Engineering Brief: Release، Blocker، قالب سایت، تصمیم لازم",
        "مسئول": "تیم IT", "اولویت": "بالا", "KPI": "Engineering Report SLA",
        "دلیل": "موضوعات فنی روزانه باید به یک گزارش کوتاه و قابل پیگیری برای مدیرعامل تبدیل شوند.",
    })
    return pd.DataFrame(suggestions).drop_duplicates(subset=["بخش", "عنوان"]).reset_index(drop=True)


def ceo_inbox(
    lead_backlog: int,
    tasks: pd.DataFrame,
    reports: pd.DataFrame,
    finance_summary: Dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    task_frame = task_followup_status(tasks)
    if not task_frame.empty:
        for _, row in task_frame[task_frame["نیازمند پیگیری"]].head(5).iterrows():
            rows.append({
                "نوع": "تسک", "شدت": "بالا" if bool(row["موعد گذشته"]) else "متوسط",
                "عنوان": str(row["عنوان"]), "بخش": str(row["بخش"]),
                "اقدام": f"درخواست Update از {row['مسئول']}", "منبع": "Task Engine",
            })
    if reports is not None and not reports.empty:
        for _, row in reports[reports["عقب‌افتاده"]].head(4).iterrows():
            rows.append({
                "نوع": "گزارش", "شدت": "متوسط", "عنوان": f"گزارش {row['بخش']} به‌روزرسانی نشده",
                "بخش": str(row["بخش"]), "اقدام": f"Reminder به {row['مسئول گزارش']}", "منبع": "Reporting SLA · Demo",
            })
    if int(lead_backlog) >= 5_000:
        rows.append({
            "نوع": "فروش", "شدت": "بالا", "عنوان": f"صف فروش به {int(lead_backlog):,} لید رسیده است",
            "بخش": "فروش تلفنی", "اقدام": "بازکردن مرکز پخش لید و تعیین SLA", "منبع": "Baseline واقعی",
        })
    finance_summary = finance_summary or {}
    if int(finance_summary.get("exceptions", 0)):
        rows.append({
            "نوع": "حسابداری", "شدت": "متوسط", "عنوان": f"{int(finance_summary['exceptions'])} تراکنش Demo در صف استثنا",
            "بخش": "حسابداری", "اقدام": "بازبینی Rule / سند / تطبیق", "منبع": "Accounting Automation · Demo",
        })
    return pd.DataFrame(rows, columns=["نوع", "شدت", "عنوان", "بخش", "اقدام", "منبع"])


def audit_event(actor: str, action: str, department: str, detail: str = "") -> Dict[str, Any]:
    return {
        "زمان": _now(),
        "کاربر": actor,
        "اقدام": action,
        "بخش": department,
        "جزئیات": detail,
    }
