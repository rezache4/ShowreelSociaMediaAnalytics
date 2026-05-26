#!/usr/bin/env python3
"""
Plot the final TikTok RFE metric distributions and cumulative distributions by era.

Outputs:
- a panel with per-era distributions for recency, frequency, engagement
- a panel with per-era empirical cumulative distributions for the same metrics

Threshold lines are optional and use the final global threshold CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
METRICS = ["recency", "frequency", "engagement"]
METRIC_LABELS = {
    "recency": "Recency",
    "frequency": "Frequency",
    "engagement": "Engagement",
}
METRIC_XLABELS = {
    "recency": "Share of annual posts since last comment",
    "frequency": "Distinct posts commented in year",
    "engagement": "Average words per comment",
}
THRESHOLD_STYLES = (
    {"color": "#222222", "linestyle": "--", "label": "Low/Mid cut"},
    {"color": "#6b6b6b", "linestyle": ":", "label": "Mid/High cut"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TikTok RFEC distributions and cumulative distributions by era."
    )
    parser.add_argument(
        "--macro",
        default=SCRIPT_DIR / "tk_comments_RFE_macro_top_level_annual_2022_2025_min2_absolute_rf.csv",
        type=Path,
        help="Input macro CSV.",
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
        default=SCRIPT_DIR / "tk_rfe_plots_top_level_annual_2022_2025_min2_absolute_rf",
        type=Path,
        help="Directory where plot files will be saved.",
    )
    return parser.parse_args()


def sanitize_era_label(era: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", era).strip("_")


def compact_era_label(era: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{4})", era)
    if match:
        return f"{match.group(1)[2:]}-{match.group(2)[2:]}"
    return era


def infer_era_order(df: pd.DataFrame) -> list[str]:
    if "era_start" in df.columns:
        era_lookup = df[["era", "era_start"]].dropna(subset=["era"]).copy()
        era_lookup["era_start"] = pd.to_datetime(era_lookup["era_start"], errors="coerce")
        era_lookup = era_lookup.dropna(subset=["era_start"]).drop_duplicates()
        if not era_lookup.empty:
            return (
                era_lookup.sort_values(["era_start", "era"], kind="mergesort")["era"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

    return df["era"].dropna().astype(str).drop_duplicates().tolist()


def load_macro(path: Path) -> tuple[pd.DataFrame, list[str]]:
    usecols = set(METRICS) | {"era", "era_start"}
    df = pd.read_csv(path, usecols=lambda col: col in usecols)
    if "era" not in df.columns:
        raise ValueError(f"Column `era` not found in {path}.")

    df["era"] = df["era"].astype("string").str.strip()
    df = df[df["era"].notna() & df["era"].ne("")].copy()

    era_order = infer_era_order(df)
    if not era_order:
        raise ValueError(f"No eras found in {path}.")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df["era"] = df["era"].astype("string")
    return df, era_order


def load_thresholds(
    path: Path,
    metrics: list[str],
) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(path, dtype={"metric": "string"})
    thresholds: dict[str, tuple[float, float]] = {}
    for metric in metrics:
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


def cumulative_share(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values <= threshold))


def threshold_summary(
    df: pd.DataFrame,
    metric: str,
    thresholds: tuple[float, float],
    era_order: list[str],
) -> dict[str, dict[str, float] | str]:
    metric_df = df.loc[df[metric].notna()].copy()
    threshold_1, threshold_2 = thresholds
    summary: dict[str, dict[str, float] | str] = {
        "scope": "global",
        "overall": {
            "threshold_1_value": threshold_1,
            "threshold_2_value": threshold_2,
            "threshold_1_share": cumulative_share(metric_df[metric].to_numpy(dtype=float), threshold_1),
            "threshold_2_share": cumulative_share(metric_df[metric].to_numpy(dtype=float), threshold_2),
        },
    }

    for era in era_order:
        values = metric_df.loc[metric_df["era"] == era, metric].to_numpy(dtype=float)
        summary[era] = {
            "threshold_1_value": threshold_1,
            "threshold_2_value": threshold_2,
            "threshold_1_share": cumulative_share(values, threshold_1),
            "threshold_2_share": cumulative_share(values, threshold_2),
        }
    return summary


def chunk_items(items: list[str], size: int = 3) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def percentile_annotation_text(
    summary: dict[str, dict[str, float] | str],
    era_order: list[str],
) -> str:
    if summary["scope"] == "global":
        lines = [
            f"T1 -> overall P{summary['overall']['threshold_1_share'] * 100:.1f}",
        ]
        t1_items = [
            f"{compact_era_label(era)} P{summary[era]['threshold_1_share'] * 100:.1f}"
            for era in era_order
        ]
        lines.extend(f"  {' | '.join(chunk)}" for chunk in chunk_items(t1_items))
        lines.append(f"T2 -> overall P{summary['overall']['threshold_2_share'] * 100:.1f}")
        t2_items = [
            f"{compact_era_label(era)} P{summary[era]['threshold_2_share'] * 100:.1f}"
            for era in era_order
        ]
        lines.extend(f"  {' | '.join(chunk)}" for chunk in chunk_items(t2_items))
        return "\n".join(lines)

    lines = [
        f"T1 weighted P{summary['overall']['threshold_1_share'] * 100:.1f}",
    ]
    t1_items = [
        (
            f"{compact_era_label(era)} "
            f"{summary[era]['threshold_1_value']:.4f} -> P{summary[era]['threshold_1_share'] * 100:.1f}"
        )
        for era in era_order
    ]
    lines.extend(f"  {' | '.join(chunk)}" for chunk in chunk_items(t1_items, size=2))
    lines.append(f"T2 weighted P{summary['overall']['threshold_2_share'] * 100:.1f}")
    t2_items = [
        (
            f"{compact_era_label(era)} "
            f"{summary[era]['threshold_2_value']:.4f} -> P{summary[era]['threshold_2_share'] * 100:.1f}"
        )
        for era in era_order
    ]
    lines.extend(f"  {' | '.join(chunk)}" for chunk in chunk_items(t2_items, size=2))
    return "\n".join(lines)


def add_percentile_annotation(
    ax: plt.Axes,
    summary: dict[str, dict[str, float] | str],
    era_order: list[str],
    y_anchor: float,
) -> None:
    ax.text(
        0.985,
        y_anchor,
        percentile_annotation_text(summary, era_order),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.88,
        },
    )


def distribution_xlim(
    values: pd.Series,
    metric: str,
    threshold_spec: dict[str, object] | None,
) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.empty:
        return (0.0, 1.0)

    lower = float(finite_values.min())
    upper = float(finite_values.max())
    q995 = float(finite_values.quantile(0.995))
    threshold_upper = threshold_upper_bound(threshold_spec) if threshold_spec is not None else lower

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
    threshold_spec: dict[str, object] | None,
) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.empty:
        return (0.0, 1.0)

    lower = float(finite_values.min())
    upper = float(finite_values.max())
    threshold_upper = threshold_upper_bound(threshold_spec) if threshold_spec is not None else upper

    if metric == "recency":
        return (0.0, max(1.0, upper))

    visible_upper = max(upper, threshold_upper)
    padding = 0.03 * visible_upper if visible_upper > 0 else 0.1
    return (max(0.0, lower), visible_upper + padding)


def threshold_upper_bound(threshold_spec: dict[str, object]) -> float:
    return max(threshold_spec)


def build_era_colors(era_order: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab10" if len(era_order) <= 10 else "tab20")
    if hasattr(cmap, "colors"):
        palette = list(cmap.colors)
        return {era: palette[index % len(palette)] for index, era in enumerate(era_order)}

    positions = np.linspace(0.0, 1.0, len(era_order))
    return {era: cmap(position) for era, position in zip(era_order, positions)}


def subplot_grid(metric_count: int) -> tuple[int, int]:
    if metric_count <= 1:
        return 1, 1
    if metric_count == 2:
        return 1, 2
    if metric_count <= 4:
        return 2, 2
    ncols = min(3, metric_count)
    nrows = int(np.ceil(metric_count / ncols))
    return nrows, ncols


def add_threshold_lines(
    ax: plt.Axes,
    thresholds: tuple[float, float],
    era_order: list[str],
    era_colors: dict[str, object],
) -> None:
    del era_order, era_colors
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
    centers = np.linspace(x_limits[0], x_limits[1], bins)
    if len(values) == 0:
        return centers, np.zeros_like(centers)

    hist, edges = np.histogram(values, bins=bins, range=x_limits, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    radius = max(1, int(np.ceil(4 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()

    smoothed = np.convolve(hist, kernel, mode="same")
    return centers, smoothed


def figure_legend_handles(
    include_thresholds: bool,
    era_order: list[str],
    era_colors: dict[str, object],
) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color=era_colors[era], linewidth=2.2, label=era)
        for era in era_order
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


def legend_layout(handle_count: int, include_thresholds: bool) -> tuple[int, float]:
    ncol = min(4, handle_count) if handle_count > 0 else 1
    legend_rows = int(np.ceil(handle_count / ncol)) if handle_count > 0 else 1
    top = 0.86 - 0.04 * max(0, legend_rows - 1) - (0.02 if include_thresholds else 0.0)
    return ncol, max(0.74, top)


def plot_distribution_panel(
    df: pd.DataFrame,
    metrics: list[str],
    era_order: list[str],
    era_colors: dict[str, object],
    thresholds: dict[str, tuple[float, float]] | None,
    percentile_summaries: dict[str, dict[str, dict[str, float] | str]] | None,
    output_path: Path,
) -> None:
    nrows, ncols = subplot_grid(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, metric in zip(axes, metrics):
        metric_threshold_spec = thresholds[metric] if thresholds is not None else None
        x_limits = distribution_xlim(df[metric], metric, metric_threshold_spec)

        for era in era_order:
            values = (
                df.loc[df["era"] == era, metric]
                .dropna()
                .to_numpy(dtype=float)
            )
            x, y = gaussian_smoothed_density(values, x_limits)
            ax.plot(x, y, linewidth=2.1, color=era_colors[era])
            ax.fill_between(x, y, 0, color=era_colors[era], alpha=0.06)

        if thresholds is not None:
            add_threshold_lines(ax, thresholds[metric], era_order, era_colors)
        if percentile_summaries is not None:
            add_percentile_annotation(ax, percentile_summaries[metric], era_order, y_anchor=0.97)
        ax.set_xlim(*x_limits)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel(METRIC_XLABELS[metric])
        ax.set_ylabel("Smoothed density")
        ax.grid(alpha=0.2)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    handles = figure_legend_handles(thresholds is not None, era_order, era_colors)
    ncol, top = legend_layout(len(handles), thresholds is not None)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=ncol,
        frameon=False,
    )
    fig.suptitle("TikTok RFE Distributions by Era", fontsize=16, y=0.98)
    fig.subplots_adjust(top=top, hspace=0.30, wspace=0.22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.array([]), np.array([])
    sorted_values = np.sort(values.astype(float))
    cumulative = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
    return sorted_values, cumulative


def plot_cumulative_panel(
    df: pd.DataFrame,
    metrics: list[str],
    era_order: list[str],
    era_colors: dict[str, object],
    thresholds: dict[str, tuple[float, float]] | None,
    percentile_summaries: dict[str, dict[str, dict[str, float] | str]] | None,
    output_path: Path,
) -> None:
    nrows, ncols = subplot_grid(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, metric in zip(axes, metrics):
        metric_threshold_spec = thresholds[metric] if thresholds is not None else None
        x_limits = cumulative_xlim(df[metric], metric, metric_threshold_spec)

        for era in era_order:
            values = (
                df.loc[df["era"] == era, metric]
                .dropna()
                .to_numpy(dtype=float)
            )
            x, y = empirical_cdf(values)
            mask = (x >= x_limits[0]) & (x <= x_limits[1])
            ax.plot(x[mask], y[mask], color=era_colors[era], linewidth=1.8)

        if thresholds is not None:
            add_threshold_lines(ax, thresholds[metric], era_order, era_colors)
        if percentile_summaries is not None:
            add_percentile_annotation(ax, percentile_summaries[metric], era_order, y_anchor=0.33)
        ax.set_xlim(*x_limits)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel(METRIC_XLABELS[metric])
        ax.set_ylabel("Cumulative share")
        ax.grid(alpha=0.2)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    handles = figure_legend_handles(thresholds is not None, era_order, era_colors)
    ncol, top = legend_layout(len(handles), thresholds is not None)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=ncol,
        frameon=False,
    )
    fig.suptitle("TikTok Cumulative RFE Distributions by Era", fontsize=16, y=0.98)
    fig.subplots_adjust(top=top, hspace=0.30, wspace=0.22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_percentile_summary_rows(
    percentile_summaries: dict[str, dict[str, dict[str, float] | str]],
    era_order: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        summary = percentile_summaries[metric]
        row: dict[str, float | str] = {
            "metric": metric,
            "threshold_scope": summary["scope"],
            "overall_threshold_1_value": summary["overall"]["threshold_1_value"],
            "overall_threshold_2_value": summary["overall"]["threshold_2_value"],
            "overall_threshold_1_share": summary["overall"]["threshold_1_share"],
            "overall_threshold_2_share": summary["overall"]["threshold_2_share"],
        }
        for era in era_order:
            label = sanitize_era_label(era)
            row[f"era_{label}_threshold_1_value"] = summary[era]["threshold_1_value"]
            row[f"era_{label}_threshold_2_value"] = summary[era]["threshold_2_value"]
            row[f"era_{label}_threshold_1_share"] = summary[era]["threshold_1_share"]
            row[f"era_{label}_threshold_2_share"] = summary[era]["threshold_2_share"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    macro, era_order = load_macro(args.macro)
    era_colors = build_era_colors(era_order)
    thresholds = load_thresholds(args.thresholds, METRICS) if args.thresholds else None
    percentile_summaries = (
        {
            metric: threshold_summary(macro, metric, thresholds[metric], era_order)
            for metric in METRICS
        }
        if thresholds is not None
        else None
    )

    distribution_path = args.output_dir / "tk_rfe_distributions_by_era.png"
    cumulative_path = args.output_dir / "tk_rfe_cumulative_distributions_by_era.png"

    plot_distribution_panel(
        macro,
        METRICS,
        era_order,
        era_colors,
        thresholds,
        percentile_summaries,
        distribution_path,
    )
    plot_cumulative_panel(
        macro,
        METRICS,
        era_order,
        era_colors,
        thresholds,
        percentile_summaries,
        cumulative_path,
    )
    if thresholds is not None and percentile_summaries is not None and args.summary_csv is not None:
        summary = build_percentile_summary_rows(percentile_summaries, era_order, METRICS)
        summary.to_csv(args.summary_csv, index=False)

    print(f"Metrics plotted: {', '.join(METRICS)}")
    print(f"Eras plotted: {', '.join(era_order)}")
    print(f"Saved {distribution_path}")
    print(f"Saved {cumulative_path}")
    if thresholds is not None and percentile_summaries is not None and args.summary_csv is not None:
        print(f"Saved {args.summary_csv}")


if __name__ == "__main__":
    main()
