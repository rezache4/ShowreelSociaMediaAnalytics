#!/usr/bin/env python3
"""
Plot Instagram RFE metric distributions and cumulative distributions by era.

Outputs:
- a 2x2 panel with per-era distributions for recency, frequency, engagement, clumpiness
- a 2x2 panel with per-era empirical cumulative distributions for the same metrics

Threshold lines are optional. When provided, they are shown as dashed vertical lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ERA_ORDER = ["2017-2019", "2020-2022", "2023-2026"]
ERA_COLORS = {
    "2017-2019": "#1f77b4",
    "2020-2022": "#ff7f0e",
    "2023-2026": "#2ca02c",
}
METRICS = ["recency", "frequency", "engagement", "clumpiness"]
METRIC_LABELS = {
    "recency": "Recency",
    "frequency": "Frequency",
    "engagement": "Engagement",
    "clumpiness": "Clumpiness",
}
METRIC_XLABELS = {
    "recency": "Recency",
    "frequency": "Frequency",
    "engagement": "Average words per comment",
    "clumpiness": "Std. dev. of posts-between-comments gaps",
}
THRESHOLD_STYLES = (
    {"color": "#222222", "linestyle": "--", "label": "Low/Mid cut"},
    {"color": "#6b6b6b", "linestyle": "--", "label": "Mid/High cut"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RFE distributions and cumulative distributions by era."
    )
    parser.add_argument(
        "--macro",
        default="ig_comments_RFE_macro.csv",
        type=Path,
        help="Input macro CSV. Default: ig_comments_RFE_macro.csv",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="Optional threshold CSV with threshold_1 and threshold_2 columns.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Optional CSV summary with threshold-to-percentile translations.",
    )
    parser.add_argument(
        "--output-dir",
        default="rfe_plots",
        type=Path,
        help="Directory where plot files will be saved.",
    )
    return parser.parse_args()


def load_macro(path: Path) -> pd.DataFrame:
    usecols = ["era"] + METRICS
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["era"].isin(ERA_ORDER)].copy()

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df.dropna(subset=usecols)


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


def cumulative_share(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values <= threshold))


def percentile_summary(
    df: pd.DataFrame,
    metric: str,
    thresholds: tuple[float, float],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {
        "overall": {
            "threshold_1_percentile": cumulative_share(df[metric].to_numpy(dtype=float), thresholds[0]),
            "threshold_2_percentile": cumulative_share(df[metric].to_numpy(dtype=float), thresholds[1]),
        }
    }

    for era in ERA_ORDER:
        values = df.loc[df["era"] == era, metric].to_numpy(dtype=float)
        summary[era] = {
            "threshold_1_percentile": cumulative_share(values, thresholds[0]),
            "threshold_2_percentile": cumulative_share(values, thresholds[1]),
        }

    return summary


def distribution_xlim(
    values: pd.Series,
    metric: str,
    thresholds: tuple[float, float] | None,
) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.empty:
        return (0.0, 1.0)

    lower = float(finite_values.min())
    upper = float(finite_values.max())
    q995 = float(finite_values.quantile(0.995))
    threshold_upper = max(thresholds) if thresholds is not None else lower

    if metric == "recency":
        return (0.0, max(1.0, upper))

    visible_upper = max(q995, threshold_upper)
    if upper > visible_upper:
        padding = 0.03 * visible_upper if visible_upper > 0 else 0.1
        return (max(0.0, lower), visible_upper + padding)

    padding = 0.03 * upper if upper > 0 else 0.1
    return (max(0.0, lower), upper + padding)


def cumulative_xlim(
    values: pd.Series,
    metric: str,
    thresholds: tuple[float, float] | None,
) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.empty:
        return (0.0, 1.0)

    lower = float(finite_values.min())
    upper = float(finite_values.max())
    threshold_upper = max(thresholds) if thresholds is not None else upper

    if metric == "recency":
        return (0.0, max(1.0, upper))

    visible_upper = max(upper, threshold_upper)
    padding = 0.03 * visible_upper if visible_upper > 0 else 0.1
    return (max(0.0, lower), visible_upper + padding)


def add_threshold_lines(ax: plt.Axes, thresholds: tuple[float, float]) -> None:
    for threshold, style in zip(thresholds, THRESHOLD_STYLES):
        ax.axvline(
            threshold,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.4,
            alpha=0.9,
        )


def gaussian_smoothed_density(
    values: np.ndarray,
    x_limits: tuple[float, float],
    bins: int = 240,
    sigma_bins: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, range=x_limits, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    radius = max(1, int(np.ceil(4 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()

    smoothed = np.convolve(hist, kernel, mode="same")
    return centers, smoothed


def figure_legend_handles(include_thresholds: bool) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=ERA_COLORS[era], linewidth=2.2, label=era)
        for era in ERA_ORDER
    ]
    if include_thresholds:
        for style in THRESHOLD_STYLES:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.6,
                    label=style["label"],
                )
            )
    return handles


def plot_distribution_panel(
    df: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]] | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        metric_thresholds = thresholds[metric] if thresholds is not None else None
        x_limits = distribution_xlim(df[metric], metric, metric_thresholds)

        for era in ERA_ORDER:
            values = df.loc[df["era"] == era, metric].to_numpy(dtype=float)
            x, y = gaussian_smoothed_density(values, x_limits)
            ax.plot(
                x,
                y,
                linewidth=2.1,
                color=ERA_COLORS[era],
            )
            ax.fill_between(x, y, 0, color=ERA_COLORS[era], alpha=0.06)

        if thresholds is not None:
            add_threshold_lines(ax, thresholds[metric])
        ax.set_xlim(*x_limits)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel(METRIC_XLABELS[metric])
        ax.set_ylabel("Smoothed density")
        ax.grid(alpha=0.2)

    fig.legend(
        handles=figure_legend_handles(include_thresholds=thresholds is not None),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=5 if thresholds is not None else 3,
        frameon=False,
    )
    fig.suptitle("RFE Distributions by Era", fontsize=16, y=0.975)
    fig.subplots_adjust(top=0.84, hspace=0.30, wspace=0.22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values.astype(float))
    cumulative = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
    return sorted_values, cumulative


def plot_cumulative_panel(
    df: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]] | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        metric_thresholds = thresholds[metric] if thresholds is not None else None
        x_limits = cumulative_xlim(df[metric], metric, metric_thresholds)

        for era in ERA_ORDER:
            values = df.loc[df["era"] == era, metric].to_numpy(dtype=float)
            x, y = empirical_cdf(values)
            mask = (x >= x_limits[0]) & (x <= x_limits[1])
            ax.plot(x[mask], y[mask], color=ERA_COLORS[era], linewidth=1.8)

        if thresholds is not None:
            add_threshold_lines(ax, thresholds[metric])
        ax.set_xlim(*x_limits)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel(METRIC_XLABELS[metric])
        ax.set_ylabel("Cumulative share")
        ax.grid(alpha=0.2)

    fig.legend(
        handles=figure_legend_handles(include_thresholds=thresholds is not None),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=5 if thresholds is not None else 3,
        frameon=False,
    )
    fig.suptitle("Cumulative RFE Distributions by Era", fontsize=16, y=0.975)
    fig.subplots_adjust(top=0.84, hspace=0.30, wspace=0.22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_percentile_summary_rows(
    percentile_summaries: dict[str, dict[str, dict[str, float]]],
    thresholds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for metric in METRICS:
        summary = percentile_summaries[metric]
        rows.append(
            {
                "metric": metric,
                "threshold_1": thresholds[metric][0],
                "threshold_2": thresholds[metric][1],
                "overall_threshold_1_percentile": summary["overall"]["threshold_1_percentile"],
                "overall_threshold_2_percentile": summary["overall"]["threshold_2_percentile"],
                "era_2017_2019_threshold_1_percentile": summary["2017-2019"]["threshold_1_percentile"],
                "era_2017_2019_threshold_2_percentile": summary["2017-2019"]["threshold_2_percentile"],
                "era_2020_2022_threshold_1_percentile": summary["2020-2022"]["threshold_1_percentile"],
                "era_2020_2022_threshold_2_percentile": summary["2020-2022"]["threshold_2_percentile"],
                "era_2023_2026_threshold_1_percentile": summary["2023-2026"]["threshold_1_percentile"],
                "era_2023_2026_threshold_2_percentile": summary["2023-2026"]["threshold_2_percentile"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    macro = load_macro(args.macro)
    thresholds = load_thresholds(args.thresholds) if args.thresholds else None
    percentile_summaries = (
        {
            metric: percentile_summary(macro, metric, thresholds[metric]) for metric in METRICS
        }
        if thresholds is not None and args.summary_csv is not None
        else None
    )

    distribution_path = args.output_dir / "ig_rfe_distributions_by_era.png"
    cumulative_path = args.output_dir / "ig_rfe_cumulative_distributions_by_era.png"

    plot_distribution_panel(macro, thresholds, distribution_path)
    plot_cumulative_panel(macro, thresholds, cumulative_path)
    if thresholds is not None and percentile_summaries is not None and args.summary_csv is not None:
        summary = build_percentile_summary_rows(percentile_summaries, thresholds)
        summary.to_csv(args.summary_csv, index=False)

    print(f"Saved {distribution_path}")
    print(f"Saved {cumulative_path}")
    if thresholds is not None and percentile_summaries is not None and args.summary_csv is not None:
        print(f"Saved {args.summary_csv}")


if __name__ == "__main__":
    main()
