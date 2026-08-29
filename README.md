# NIK INTELLIGENCE V0.2

**نیک اس‌ام‌اس | تحلیل داده**  
**Executive Data Intelligence Prototype**

> Prototype using aggregate baseline + synthetic/demo data. Not connected to NIK internal systems.
>
> Forecasts and ML outputs are experimental and should not be used for production decisions.

## What changed in V0.2

V0.2 is designed for a management presentation rather than a technical demo.

### Visual / UX
- Premium dark executive UI
- Liquid-glass / glassmorphism cards
- NIK blue accent
- Persian RTL interface
- Clear data-confidence labels on KPI cards
- API and n8n connector placeholder buttons
- Cleaner sidebar with grouped scenario controls
- Management-first Executive Snapshot

### Data interpretation
Every important number is classified as one of:

- **Baseline واقعی** — aggregate business input supplied for the prototype
- **محاسبه‌شده** — derived from baseline and explicit assumptions
- **تخمینی** — attribution / estimate, not a verified causal KPI
- **مصنوعی** — generated demo data or experimental model output

This prevents synthetic/model numbers from being presented as real internal NIK data.

## Current baseline snapshot

Date context: **29 August 2026**

- Plan A: 15,000,000 Toman
- Plan B: 30,000,000 Toman
- Plan mix: 50 / 50
- Daily phone sales: 10 devices
- Monthly online sales: 20 devices
- Lead backlog: 4,000
- Instagram stories: 9/day
- Instagram reels: approximately 1/day
- Total Instagram content: approximately 10/day
- Estimated content-attributed sales: approximately 2/day
- Instagram followers: approximately 207K
- Content team: 5 people

With a 30-day sales assumption:

- Phone units/month = 300
- Total units/month = 320
- ASP = 22.5M Toman
- Derived monthly revenue = 7.2B Toman

These are calculations, not accounting revenue records.

## 10-reel real snapshot

The content page includes the supplied 10-reel performance snapshot:

- Total Views: 397,409
- Average Views: 39,741
- Median Views: 10,020
- Total Comments: 6,647
- Total Shares: 10,435
- Comments + Shares: 17,082
- Interaction/View proxy: approximately 4.3%
- Top 3 reels represent approximately 76.9% of total views

The dashboard explicitly explains why Median and distribution are important for hit-driven content performance.

## Important analytical correction

V0.1 displayed `Monthly Sales / Lead Backlog` as a Lead-to-Purchase conversion proxy.

V0.2 no longer labels that ratio as conversion.

`Lead Backlog` is a stock, while monthly sales are a flow. A real conversion rate requires matched lead-level data and a defined cohort/window.

The dashboard therefore treats backlog comparisons as **volume context only**, not as a true conversion KPI or processing-time estimate.

## Project files

- `app.py` — Streamlit interface, RTL design, Liquid Glass UI and presentation flow
- `business_data.py` — aggregate baseline, real 10-reel snapshot and historical business context
- `data_generator.py` — realistic synthetic datasets
- `analytics_engine.py` — KPI and business calculations
- `ml_engine.py` — K-Means, Logistic Regression, Linear Regression forecast and anomaly tools
- `insight_engine.py` — dynamic rule-based management insights
- `requirements.txt` — dependencies

## Placeholder connectors

V0.2 includes visible buttons for:

- `اتصال API`
- `اتصال n8n`

They are intentionally placeholders.

No API, database, CRM, n8n workflow or NIK internal credential is connected yet.

Future architecture:

```text
NIK Database / CRM / Panel
        |
    Read-only API
        |
       n8n
        |
Validation / Cleaning
        |
Analytics Engine
        |
NIK Intelligence
```

## Run

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Fast GitHub / Streamlit update

If an older version is already deployed:

1. Extract the V0.2 ZIP.
2. Open the existing GitHub repository.
3. Upload all V0.2 files into the repository root.
4. Confirm overwriting files with the same names.
5. Commit changes.
6. Streamlit Community Cloud should rebuild automatically.
7. If it does not, open the app menu and use **Reboot**.

`Main file path` remains:

```text
app.py
```

## Recommended management demo

1. Open **نمای مدیریتی**
2. Explain the four data-confidence tags
3. Show 10 phone sales/day and 20 online/month as baseline
4. Show 320 units and 7.2B revenue as derived under a 30-day assumption
5. Open **تحلیل محتوا و اینستاگرام**
6. Show Average 39.7K vs Median 10K
7. Show that Top 3 reels produce ~76.9% of total views
8. Explain why content measurement needs Watch Time / Saves / Leads / Sales
9. Open **مرکز داده**
10. Show API / n8n placeholders
11. Open **خط لوله تحلیل**
12. Explain future read-only integration
13. Return to Executive Overview
14. Change Daily Phone Sales from 10 to 15
15. Show that Units, Revenue, trends, forecast and insights update dynamically

## Next production data needed

The highest-value next datasets are:

- Daily Sales raw data
- Leads with stage history
- Call Center: calls / answered / qualified / sale
- Content per post/reel/story
- Reach / Saves / Watch Time / Completion
- Attribution source
- SMS sent / delivered / click / conversion
- NIKPOS active device and usage events
- Customer industry / city / plan
- Subscription renewal / churn
- Campaign spend / lead / sale / revenue
- Referral funnel and reward cost
