#!/usr/bin/env python3
"""
Assign RFE classes to the final TikTok macro rows using the final threshold CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
METRICS = ["recency", "frequency", "engagement"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify the final TikTok RFE macro rows using global behavioral thresholds."
    )
    parser.add_argument(
        "--macro",
        default=SCRIPT_DIR / "tk_comments_RFE_macro_top_level_annual_2022_2025_min2_absolute_rf.csv",
        type=Path,
        help="Input macro CSV.",
    )
    parser.add_argument(
        "--thresholds",
        default=SCRIPT_DIR / "tk_comments_rfe_thresholds_top_level_annual_2022_2025_min2_absolute_rf.csv",
        type=Path,
        help="Threshold CSV.",
    )
    parser.add_argument(
        "--output",
        default=SCRIPT_DIR / "tk_comments_RFE_macro_top_level_annual_2022_2025_min2_absolute_rf_classified.csv",
        type=Path,
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_thresholds(path: Path) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(path, dtype={"metric": "string", "era": "string"})
    thresholds: dict[str, tuple[float, float]] = {}

    for metric in METRICS:
        rows = df.loc[df["metric"] == metric].copy()
        if rows.empty:
            raise ValueError(f"Missing thresholds for metric {metric}.")
        if len(rows) != 1:
            raise ValueError(f"Expected one global threshold row for metric {metric}.")

        row = rows.iloc[0]
        thresholds[metric] = (
            float(row["threshold_1"]),
            float(row["threshold_2"]),
        )

    return thresholds


def assign_class(values: pd.Series, threshold_1: float, threshold_2: float) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    classes = np.select(
        [
            numeric_values <= threshold_1,
            numeric_values <= threshold_2,
            numeric_values > threshold_2,
        ],
        [1, 2, 3],
        default=np.nan,
    )
    return pd.Series(classes, index=values.index).astype("Int64")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.macro)
    thresholds = load_thresholds(args.thresholds)

    for metric in METRICS:
        threshold_1, threshold_2 = thresholds[metric]
        df[f"{metric}_class"] = assign_class(df[metric], threshold_1, threshold_2)

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved classified macro to {args.output}")


if __name__ == "__main__":
    main()
