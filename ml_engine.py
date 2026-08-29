from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def segment_customers(customers: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = customers.copy()
    features = df[["recency", "frequency", "monetary_value"]].astype(float).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=5, random_state=seed, n_init=10)
    df["cluster"] = model.fit_predict(scaled)

    profile = (
        df.groupby("cluster", as_index=False)
        .agg(
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary_value=("monetary_value", "mean"),
            customers=("customer_id", "count"),
        )
    )
    for col in ["recency", "frequency", "monetary_value"]:
        denom = max(profile[col].max() - profile[col].min(), 1e-9)
        profile[f"{col}_norm"] = (profile[col] - profile[col].min()) / denom
    profile["value_score"] = (
        -0.45 * profile["recency_norm"]
        + 0.25 * profile["frequency_norm"]
        + 0.30 * profile["monetary_value_norm"]
    )
    ordered = profile.sort_values("value_score")["cluster"].tolist()
    labels = ["Inactive", "At Risk", "Regular", "Growth", "High Value"]
    cluster_to_label = {cluster: labels[i] for i, cluster in enumerate(ordered)}
    df["segment"] = df["cluster"].map(cluster_to_label)
    profile["segment"] = profile["cluster"].map(cluster_to_label)
    return df, profile.sort_values("value_score", ascending=False).reset_index(drop=True)


def churn_risk_model(customers: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = customers.copy()
    feature_cols = ["recency", "frequency", "monetary_value", "sms_usage", "nikpos_usage", "activity_score"]
    X = df[feature_cols].astype(float).fillna(0)

    # Synthetic target generation for a prototype only.
    rng = np.random.default_rng(seed + 700)
    risk_latent = (
        0.030 * X["recency"]
        - 0.22 * X["frequency"]
        - 0.000000006 * X["monetary_value"]
        - 0.0012 * X["sms_usage"]
        - 0.0060 * X["nikpos_usage"]
        - 2.0 * X["activity_score"]
        + rng.normal(0, 0.6, len(X))
    )
    threshold = np.quantile(risk_latent, 0.68)
    y = (risk_latent >= threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train)

    test_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else float("nan")

    all_prob = model.predict_proba(scaler.transform(X))[:, 1]
    df["risk_score"] = np.round(all_prob, 4)
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)
    stats = {
        "synthetic_holdout_auc": float(auc),
        "high_or_very_high_share": float(df["risk_level"].isin(["High", "Very High"]).mean()),
    }
    return df, stats


def revenue_forecast(monthly_sales: pd.DataFrame, months_ahead: int = 3) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if monthly_sales.empty:
        return pd.DataFrame(columns=["month", "revenue", "series"]), {"r2": float("nan")}

    hist = monthly_sales[["month", "revenue"]].dropna().sort_values("month").copy()
    hist["t"] = np.arange(len(hist), dtype=float)
    model = LinearRegression()
    model.fit(hist[["t"]], hist["revenue"])
    r2 = model.score(hist[["t"]], hist["revenue"]) if len(hist) > 1 else float("nan")

    future_t = np.arange(len(hist), len(hist) + months_ahead, dtype=float)
    future_months = pd.date_range(
        hist["month"].max() + pd.offsets.MonthBegin(1), periods=months_ahead, freq="MS"
    )
    future_frame = pd.DataFrame({'t': future_t})
    future_revenue = np.maximum(0, model.predict(future_frame))

    historical = hist[["month", "revenue"]].copy()
    historical["series"] = "Historical"
    forecast = pd.DataFrame({"month": future_months, "revenue": future_revenue, "series": "Forecast"})
    return pd.concat([historical, forecast], ignore_index=True), {"r2": float(r2)}


def detect_anomalies(series_df: pd.DataFrame, value_col: str, date_col: str = "date", window: int = 14) -> pd.DataFrame:
    df = series_df[[date_col, value_col]].dropna().sort_values(date_col).copy()
    if df.empty:
        df["z_score"] = []
        df["is_anomaly"] = []
        return df
    rolling_mean = df[value_col].rolling(window=window, min_periods=max(5, window // 3)).mean()
    rolling_std = df[value_col].rolling(window=window, min_periods=max(5, window // 3)).std(ddof=0)
    z = (df[value_col] - rolling_mean) / rolling_std.replace(0, np.nan)
    df["rolling_mean"] = rolling_mean
    df["z_score"] = z.replace([np.inf, -np.inf], np.nan).fillna(0)
    df["is_anomaly"] = df["z_score"].abs() >= 2.5
    return df
