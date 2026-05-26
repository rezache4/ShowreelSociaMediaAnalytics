#!/usr/bin/env python3
"""
Assign Low / Medium / High RFE levels to each user in each rolling four-month window.

Rules:
- Recency: per-window thresholds based on recency percentiles
  - Low if recency <= p33
  - Medium if p33 < recency <= p67
  - High if recency > p67
- Frequency:
  - Low if frequency == 1
  - Medium if frequency == 2
  - High if frequency >= 3
- Engagement:
  - Low if engagement < 5
  - Medium if 5 <= engagement < 11
  - High if engagement >= 11

Outputs:
- long classification file, one row per user-window
- thresholds file for all rolling windows
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

METRICS_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_rolling_metrics.csv"
WINDOWS_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_rolling_windows.csv"

LONG_OUTPUT_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_level_classification_long.csv"
THRESHOLDS_OUTPUT_PATH = SCRIPT_DIR / "ig_dynamic_rfe_thresholds_all_windows.csv"

RECENCY_LOW_QUANTILE = 0.33
RECENCY_HIGH_QUANTILE = 0.67
ENGAGEMENT_LOW_THRESHOLD = 5.0
ENGAGEMENT_HIGH_THRESHOLD = 11.0

METRICS_DTYPES = {
    "window_id": "string",
    "window_label": "string",
    "window_index": "Int64",
    "window_start": "string",
    "window_end": "string",
    "posts_in_window": "Int64",
    "user_key": "string",
    "user_id": "string",
    "username": "string",
}

WINDOWS_DTYPES = {
    "window_id": "string",
    "window_label": "string",
    "window_index": "Int64",
    "window_start": "string",
    "window_end": "string",
    "window_total_days": "Int64",
    "posts_in_window": "Int64",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(METRICS_PATH, dtype=METRICS_DTYPES)
    windows = pd.read_csv(WINDOWS_PATH, dtype=WINDOWS_DTYPES)
    metrics["last_comment_timestamp"] = pd.to_datetime(metrics["last_comment_timestamp"], errors="coerce")
    return metrics, windows


def build_recency_thresholds(metrics: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    quantiles = (
        metrics.groupby("window_id", sort=False)["recency"]
        .quantile([RECENCY_LOW_QUANTILE, RECENCY_HIGH_QUANTILE])
        .unstack()
        .rename(
            columns={
                RECENCY_LOW_QUANTILE: "recency_low_threshold",
                RECENCY_HIGH_QUANTILE: "recency_high_threshold",
            }
        )
        .reset_index()
    )

    thresholds = windows.merge(quantiles, on="window_id", how="left")
    thresholds["recency_low_threshold"] = thresholds["recency_low_threshold"].round(2)
    thresholds["recency_high_threshold"] = thresholds["recency_high_threshold"].round(2)
    thresholds["recency_low_rule"] = thresholds["recency_low_threshold"].map(
        lambda value: f"Low if recency <= {value:.2f}" if pd.notna(value) else pd.NA
    )
    thresholds["recency_mid_rule"] = [
        (
            f"Medium if {low_value:.2f} < recency <= {high_value:.2f}"
            if pd.notna(low_value) and pd.notna(high_value)
            else pd.NA
        )
        for low_value, high_value in zip(
            thresholds["recency_low_threshold"],
            thresholds["recency_high_threshold"],
        )
    ]
    thresholds["recency_high_rule"] = thresholds["recency_high_threshold"].map(
        lambda value: f"High if recency > {value:.2f}" if pd.notna(value) else pd.NA
    )
    thresholds["frequency_low_rule"] = "Low if frequency = 1"
    thresholds["frequency_mid_rule"] = "Medium if frequency = 2"
    thresholds["frequency_high_rule"] = "High if frequency >= 3"
    thresholds["engagement_low_rule"] = f"Low if engagement < {ENGAGEMENT_LOW_THRESHOLD:g}"
    thresholds["engagement_mid_rule"] = (
        f"Medium if {ENGAGEMENT_LOW_THRESHOLD:g} <= engagement < {ENGAGEMENT_HIGH_THRESHOLD:g}"
    )
    thresholds["engagement_high_rule"] = f"High if engagement >= {ENGAGEMENT_HIGH_THRESHOLD:g}"

    return thresholds


def classify_recency(series: pd.Series, low_threshold: pd.Series, high_threshold: pd.Series) -> pd.Series:
    levels = np.where(
        series <= low_threshold,
        "Low",
        np.where(series <= high_threshold, "Medium", "High"),
    )
    return pd.Series(levels, index=series.index, dtype="string")


def classify_frequency(series: pd.Series) -> pd.Series:
    levels = np.where(
        series <= 1,
        "Low",
        np.where(series <= 2, "Medium", "High"),
    )
    return pd.Series(levels, index=series.index, dtype="string")


def classify_engagement(series: pd.Series) -> pd.Series:
    levels = np.where(
        series < ENGAGEMENT_LOW_THRESHOLD,
        "Low",
        np.where(series < ENGAGEMENT_HIGH_THRESHOLD, "Medium", "High"),
    )
    return pd.Series(levels, index=series.index, dtype="string")


def build_long_classification(metrics: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    classified = metrics.merge(
        thresholds[
            [
                "window_id",
                "recency_low_threshold",
                "recency_high_threshold",
            ]
        ],
        on="window_id",
        how="left",
    )

    classified["recency_level"] = classify_recency(
        classified["recency"],
        classified["recency_low_threshold"],
        classified["recency_high_threshold"],
    )
    classified["frequency_level"] = classify_frequency(classified["frequency"])
    classified["engagement_level"] = classify_engagement(classified["engagement"])
    classified["rfe_profile"] = (
        "R="
        + classified["recency_level"]
        + " | F="
        + classified["frequency_level"]
        + " | E="
        + classified["engagement_level"]
    )

    ordered_columns = [
        "window_id",
        "window_label",
        "window_index",
        "window_start",
        "window_end",
        "window_total_days",
        "posts_in_window",
        "user_key",
        "user_id",
        "username",
        "comment_count",
        "last_comment_timestamp",
        "recency",
        "recency_low_threshold",
        "recency_high_threshold",
        "recency_level",
        "frequency",
        "frequency_level",
        "engagement",
        "engagement_level",
        "rfe_profile",
    ]
    return classified[ordered_columns].sort_values(
        ["window_index", "user_key"],
        kind="mergesort",
    ).reset_index(drop=True)


def main() -> None:
    metrics, windows = load_inputs()
    thresholds = build_recency_thresholds(metrics, windows)
    classified = build_long_classification(metrics, thresholds)

    thresholds.to_csv(THRESHOLDS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    classified.to_csv(LONG_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(thresholds):,} rows to {THRESHOLDS_OUTPUT_PATH}")
    print(f"Saved {len(classified):,} rows to {LONG_OUTPUT_PATH}")
    print(f"Users classified: {classified['user_key'].nunique():,}")
    print(f"Rolling windows covered: {classified['window_id'].nunique():,}")


if __name__ == "__main__":
    main()
