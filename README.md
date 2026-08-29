# NIK INTELLIGENCE V0.1

**Automated Data Intelligence Platform**  
**Status:** Prototype / Proof of Concept

> Prototype using synthetic/demo data. Not connected to NIK internal systems.
>
> Forecasts and ML outputs are experimental and should not be used for production decisions.

## What this prototype demonstrates

NIK INTELLIGENCE V0.1 is a local Streamlit application designed to demonstrate an end-to-end internal Data Intelligence concept without connecting to confidential NIK SMS data.

The live dependency chain is:

`INPUT -> DATA -> ANALYSIS -> KPI -> CHART -> MODEL -> INSIGHT -> DASHBOARD UPDATE`

Changing scenario controls regenerates the synthetic data and recalculates the downstream metrics, charts, forecast, models, funnel, channel mix, and rule-based insights.

## Project files

- `app.py` — Streamlit application, navigation, UI, scenario controls, CSV upload and demo flow.
- `data_generator.py` — realistic synthetic datasets for sales, customers, leads, SMS, NIKPOS usage and subscriptions.
- `analytics_engine.py` — KPI calculations, funnel, backlog capacity proxy, content metrics, aggregation and data quality.
- `ml_engine.py` — K-Means RFM-style segmentation, Logistic Regression churn-risk prototype, Linear Regression revenue forecast and Rolling Z-score anomaly detection.
- `insight_engine.py` — dynamic Automated Rule-based Insights.
- `requirements.txt` — Python dependencies.

## Dependencies

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn

No external database, API key, NIK credential, cloud deployment or internal-system connection is required.

## Default demo scenario

- Plan A: 15,000,000 Toman
- Plan B: 30,000,000 Toman
- Plan mix: 50 / 50
- Lead backlog: 4,000
- Daily phone sales: 10
- Monthly online sales: 20
- Stories per day: 9
- Estimated content-attributed sales per day: 2
- Synthetic customers: 5,000
- Sales days per month: 30

With the default mix, Average Selling Price is calculated dynamically as:

`0.50 * 15M + 0.50 * 30M = 22.5M Toman`

Monthly phone units are calculated as:

`Daily Phone Sales * Sales Days per Month`

Monthly total units are:

`Monthly Phone Units + Monthly Online Sales`

Monthly revenue is:

`Monthly Units * Dynamic Average Selling Price`

These values are not hard-coded into dashboard cards.

## Run on Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Run on Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Application sections

1. Executive Overview
2. Data Center
3. Sales Analytics
4. Customer Intelligence
5. NIKPOS Analytics
6. Content Analytics
7. SMS Analytics
8. Anomaly Detection
9. Predictions
10. Automated Insights
11. Analysis Pipeline
12. Settings / Scenario Controls

## Data Science methods used

### Customer Segmentation

A prototype K-Means model uses RFM-style features:

- Recency
- Frequency
- Monetary Value

Clusters are mapped to business labels: High Value, Growth, Regular, At Risk and Inactive.

### Prototype Churn Risk Model

A Logistic Regression model uses:

- Recency
- Frequency
- Monetary Value
- SMS Usage
- NIKPOS Usage
- Activity Score

The training target is synthetically generated for this proof of concept. Any displayed AUC is therefore illustrative and must not be interpreted as real production validation.

### Revenue Forecast

A simple Linear Regression trend model forecasts the next three months from synthetic monthly revenue history. It is deliberately simple and explainable.

### Anomaly Detection

The prototype uses a rolling mean, rolling standard deviation and Z-score. Observations with absolute Z-score of 2.5 or greater are flagged.

## Content attribution warning

Content-attributed sales are displayed as **Estimated / Synthetic Attribution**. This is not a verified causal attribution model.

Real attribution would require source tracking such as:

- campaign IDs
- channel/source tagging
- lead-source capture
- click and session tracking
- CRM linkage
- attribution windows
- an explicitly defined attribution model

## Lead backlog interpretation

The prototype shows `Lead Backlog / Daily Phone Sales` as a capacity proxy. This should not be treated as a true lead-processing SLA because sales throughput is not identical to contact-processing capacity.

A production version should use actual sales-agent activity data such as calls attempted, leads contacted, qualification outcomes, agent availability and handling time.

## CSV import

The Data Center includes local CSV uploaders for:

- Sales
- Customers
- Leads
- SMS
- NIKPOS

Synthetic data remains the default. V0.1 only accepts an uploaded dataset as active in the Data Center when expected prototype columns are present. No external API or database connection is created.

## Suggested manager demo flow

1. Open NIK INTELLIGENCE.
2. Show Executive Overview.
3. Point out the `DEMO / SYNTHETIC DATA` label.
4. Show the default scenario controls.
5. Click `RUN ANALYSIS`.
6. Show the Analysis Pipeline.
7. Return to KPI cards and Sales Funnel.
8. Open Customer Intelligence.
9. Show the Prototype Churn Risk Model and RFM segmentation.
10. Open Anomaly Detection.
11. Open Predictions and show the three-month forecast.
12. Change `Daily Phone Sales` from `10` to `15`.
13. Show that Revenue, Monthly Units, Conversion, Channel Mix, Funnel, synthetic trend, forecast and insights recalculate.
14. Open Automated Insights.
15. Explain that the current data is synthetic and no NIK internal systems are connected.
16. Explain the future read-only data architecture below.

## Future architecture — not implemented in V0.1

```text
NIK Database
    |
Read-only Connection / API
    |
Data Pipeline
    |
Data Warehouse
    |
Analytics Engine
    |
ML
    |
NIK Intelligence Dashboard
    |
Alerts / Reports / AI Assistant
```

The intended production principle is to keep operational systems separate and provide NIK Intelligence with controlled, read-only access through a documented data contract or API.

## Security boundary

V0.1 requires:

- no NIK internal data
- no API keys
- no database credentials
- no CRM credentials
- no cloud deployment
- no authentication layer

This is intentionally a standalone proof of concept.
