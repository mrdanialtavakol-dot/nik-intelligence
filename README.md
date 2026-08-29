# NIK INTELLIGENCE V0.3

Management Data Intelligence Prototype for NIK SMS / NIKPOS.

> Prototype using aggregate baseline + synthetic/demo data. Not connected to NIK internal systems.
>
> Forecasts, churn outputs and the Media Intelligence timeline are experimental/demo outputs and must not be used as production facts.

## What changed in V0.3

- New executive-first CEO home page: 8 decision KPIs + 3 management signals.
- Main visual theme rebuilt around `#ADCBFF` with dark blue gradients and glass surfaces.
- Official NIKSMS logo included in the app.
- New **Media Intelligence Lab** with:
  - 5 supplied vertical videos.
  - 6 supplied content images.
  - interactive second selector.
  - demo retention curve.
  - demo click/interaction signal.
  - synthetic event markers (Hook / Drop / Proof / Replay / CTA).
  - temporary media upload preview.
- API and n8n connector buttons remain visual placeholders only.
- Fixed the V0.2 `Scenario(...)` TypeError risk and the anomaly-page column bug.
- Scenario construction is defensive against short-lived mixed-file deployments during Streamlit rebuilds.

## Important deployment note

V0.2 could fail if `app.py` was updated while `data_generator.py` remained from V0.1. The new `app.py` passes fields such as `reels_per_day`, `instagram_followers`, and `content_team_size`; an old `Scenario` class does not know those arguments.

**For V0.3 replace ALL project files, not only `app.py`.**

The repository root should contain:

```text
app.py
analytics_engine.py
business_data.py
data_generator.py
insight_engine.py
media_data.py
ml_engine.py
requirements.txt
README.md
VERSION.txt
assets/
  images/
  videos/
```

Then reboot the Streamlit app once.

## Run locally

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## CEO first-screen KPIs

The first screen intentionally prioritizes:

1. Derived monthly revenue
2. Derived monthly units
3. Current lead backlog
4. Daily phone sales baseline
5. Average selling price
6. Monthly online sales baseline
7. Instagram follower snapshot
8. Estimated content-attributed sales/day

Technical ML metrics, anomaly details and experimental forecast are moved below the first-screen decision layer.

## Media Intelligence truth labels

The uploaded videos/images are real supplied assets. However, no per-second Instagram event stream was provided. Therefore the timeline curves and Click/Replay markers in V0.3 are explicitly labeled **DEMO / SYNTHETIC TIMELINE**.

A future real pipeline could be:

```text
Instagram / Website / Tracked Links / CRM
    -> API / Webhooks
    -> n8n
    -> event/content tables
    -> NIK Intelligence
```

This will allow real attribution such as content -> click -> lead -> call -> purchase where tracking data exists.
