#!/usr/bin/env python3
"""
Assign low/medium/high RFE classes to each user-era row.

Class coding:
- 1 = Low
- 2 = Medium
- 3 = High

Thresholds are read from a CSV with one row per metric and
`threshold_1` / `threshold_2` columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["recency", "frequency", "engagement", "clumpiness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Instagram RFE macro rows into low/medium/high classes."
    )
    parser.add_argument(
        "--input",
        default="ig_comments_RFE_macro.csv",
        type=Path,
        help="Input macro CSV. Default: ig_comments_RFE_macro.csv",
    )
    parser.add_argument(
        "--thresholds",
        default="ig_comments_rfe_behavioral_thresholds.csv",
        type=Path,
        help="Threshold CSV. Default: ig_comments_rfe_behavioral_thresholds.csv",
    )
    parser.add_argument(
        "--output",
        default="ig_comments_RFE_macro_classified.csv",
        type=Path,
        help="Output classified CSV. Default: ig_comments_RFE_macro_classified.csv",
    )
    return parser.parse_args()


def load_thresholds(path: Path) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(path)
    thresholds: dict[str, tuple[float, float]] = {}

    for metric in METRICS:
        row = df.loc[df["metric"] == metric]
        if row.empty:
            raise ValueError(f"Missing thresholds for metric {metric}.")
        thresholds[metric] = (
            float(row["threshold_1"].iloc[0]),
            float(row["threshold_2"].iloc[0]),
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
    return pd.Series(classes, index=values.index, dtype="Int64")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    thresholds = load_thresholds(args.thresholds)

    for metric in METRICS:
        if metric not in df.columns:
            raise ValueError(f"Missing metric column in input CSV: {metric}")

        threshold_1, threshold_2 = thresholds[metric]
        df[f"{metric}_class"] = assign_class(df[metric], threshold_1, threshold_2)

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved classified macro CSV to {args.output}")
    for metric in METRICS:
        counts = (
            df[f"{metric}_class"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        print(f"{metric}: {counts}")


if __name__ == "__main__":
    main()
