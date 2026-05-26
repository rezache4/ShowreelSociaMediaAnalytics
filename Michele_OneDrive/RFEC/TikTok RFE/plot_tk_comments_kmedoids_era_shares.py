#!/usr/bin/env python3
"""
Plot year-wise TikTok RFE cluster composition:
- absolute counts bar chart
- within-year percentage pie charts
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tk_cluster_semantics import maybe_get_final_k6_semantics, ordered_cluster_ids


SCRIPT_DIR = Path(__file__).resolve().parent
YEAR_ORDER = ["2022", "2023", "2024", "2025"]
CLUSTER_COLORS = [
    "#2a9d8f",
    "#e76f51",
    "#264653",
    "#e9c46a",
    "#457b9d",
    "#8d99ae",
    "#f4a261",
    "#6a994e",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot year-wise TikTok RFE k-medoids cluster shares and absolute counts."
    )
    parser.add_argument(
        "--assigned",
        default=SCRIPT_DIR / "tk_comments_RFE_macro_top_level_annual_2022_2025_min2_absolute_rf_kmedoids_clusters.csv",
        type=Path,
        help="Row-level assignments CSV.",
    )
    parser.add_argument(
        "--cluster-profiles",
        default=SCRIPT_DIR / "tk_comments_rfe_kmedoids_cluster_profiles.csv",
        type=Path,
        help="Cluster profiles CSV.",
    )
    parser.add_argument(
        "--k",
        required=True,
        type=int,
        help="Which k solution to plot.",
    )
    parser.add_argument(
        "--output-dir",
        default=SCRIPT_DIR / "tk_kmedoids_era_plots",
        type=Path,
        help="Directory where charts and support tables will be saved.",
    )
    parser.add_argument(
        "--use-final-k6-semantics",
        action="store_true",
        help="Apply the named final TikTok k=6 semantics instead of generic cluster labels.",
    )
    return parser.parse_args()


def cluster_column(k: int) -> str:
    return f"cluster_k{k}"


def cluster_display_metadata(
    cluster_profile_df: pd.DataFrame,
    k: int,
    use_final_k6_semantics: bool,
) -> tuple[dict[int, dict[str, object]], list[int]]:
    subset = cluster_profile_df[cluster_profile_df["k"] == k].copy()
    if subset.empty:
        raise ValueError(f"No cluster profiles found for k={k}.")

    cluster_ids = subset["cluster_id"].astype(int).tolist()
    semantics = maybe_get_final_k6_semantics(cluster_ids, k) if use_final_k6_semantics else None
    metadata: dict[int, dict[str, object]] = {}
    for row in subset.itertuples(index=False):
        cluster_id = int(row.cluster_id)
        if semantics is not None:
            semantic = semantics[cluster_id]
            metadata[cluster_id] = {
                "display_order": semantic.display_order,
                "display_code": semantic.display_code,
                "cluster_label": semantic.display_label,
                "recency_band": semantic.recency_band,
                "frequency_band": semantic.frequency_band,
                "engagement_band": semantic.engagement_band,
            }
            continue

        metadata[cluster_id] = {
            "display_order": cluster_id,
            "display_code": f"C{cluster_id}",
            "cluster_label": f"C{cluster_id} {row.cluster_label}",
            "recency_band": row.mode_recency_band,
            "frequency_band": row.mode_frequency_band,
            "engagement_band": row.mode_engagement_band,
        }
    ordered_ids = ordered_cluster_ids(cluster_ids, semantics)
    return metadata, ordered_ids


def build_counts(assigned_df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    counts = (
        assigned_df.groupby(["era", cluster_col])
        .size()
        .reset_index(name="n_rows")
        .rename(columns={cluster_col: "cluster_id"})
    )
    counts["era_total_rows"] = counts.groupby("era")["n_rows"].transform("sum")
    counts["share_within_era"] = counts["n_rows"] / counts["era_total_rows"]
    counts["share_pct"] = counts["share_within_era"] * 100.0
    return counts


def save_tables(
    counts_df: pd.DataFrame,
    metadata: dict[int, dict[str, object]],
    output_dir: Path,
    k: int,
) -> tuple[Path, Path]:
    base_out = counts_df.copy()
    base_out.insert(2, "display_order", base_out["cluster_id"].map(lambda value: metadata[int(value)]["display_order"]))
    base_out.insert(3, "display_cluster_code", base_out["cluster_id"].map(lambda value: metadata[int(value)]["display_code"]))
    base_out.insert(4, "cluster_label", base_out["cluster_id"].map(lambda value: metadata[int(value)]["cluster_label"]))
    base_out.insert(5, "recency_band", base_out["cluster_id"].map(lambda value: metadata[int(value)]["recency_band"]))
    base_out.insert(6, "frequency_band", base_out["cluster_id"].map(lambda value: metadata[int(value)]["frequency_band"]))
    base_out.insert(7, "engagement_band", base_out["cluster_id"].map(lambda value: metadata[int(value)]["engagement_band"]))
    base_out = base_out.sort_values(["era", "display_order"], kind="mergesort").reset_index(drop=True)

    counts_out = base_out[
        [
            "era",
            "cluster_id",
            "display_order",
            "display_cluster_code",
            "cluster_label",
            "recency_band",
            "frequency_band",
            "engagement_band",
            "n_rows",
            "era_total_rows",
        ]
    ].copy()
    shares_out = base_out[
        [
            "era",
            "cluster_id",
            "display_order",
            "display_cluster_code",
            "cluster_label",
            "recency_band",
            "frequency_band",
            "engagement_band",
            "share_within_era",
            "share_pct",
        ]
    ].copy()

    counts_path = output_dir / f"tk_comments_kmedoids_k{k}_counts_by_year.csv"
    shares_path = output_dir / f"tk_comments_kmedoids_k{k}_shares_by_year_pct.csv"
    counts_out.to_csv(counts_path, index=False, encoding="utf-8-sig")
    shares_out.to_csv(shares_path, index=False, encoding="utf-8-sig")
    return counts_path, shares_path


def plot_absolute_counts(
    counts_df: pd.DataFrame,
    metadata: dict[int, dict[str, object]],
    ordered_ids: list[int],
    output_path: Path,
) -> None:
    pivot = (
        counts_df.pivot(index="era", columns="cluster_id", values="n_rows")
        .reindex(YEAR_ORDER)
        .fillna(0)
        .astype(int)
    )
    cluster_ids = [cluster_id for cluster_id in ordered_ids if cluster_id in pivot.columns]
    x = np.arange(len(YEAR_ORDER))
    width = 0.8 / max(1, len(cluster_ids))

    fig, ax = plt.subplots(figsize=(15, 8))
    for idx, cluster_id in enumerate(cluster_ids):
        offset = (idx - (len(cluster_ids) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            pivot[cluster_id].to_numpy(),
            width=width,
            color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
            label=metadata[cluster_id]["cluster_label"],
        )
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(4.0, height * 0.015),
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(YEAR_ORDER)
    ax.set_ylabel("User-year count")
    ax.set_title("TikTok RFE k-medoids Cluster Counts by Year")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pies(
    counts_df: pd.DataFrame,
    metadata: dict[int, dict[str, object]],
    ordered_ids: list[int],
    output_path: Path,
) -> None:
    ncols = 2
    nrows = math.ceil(len(YEAR_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 9))
    axes = np.atleast_1d(axes).ravel()
    color_map = {
        cluster_id: CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        for index, cluster_id in enumerate(ordered_ids)
    }

    for ax, year in zip(axes, YEAR_ORDER):
        subset = counts_df[counts_df["era"] == year].copy()
        subset["cluster_id"] = subset["cluster_id"].astype(int)
        subset["display_order"] = subset["cluster_id"].map({cluster_id: position for position, cluster_id in enumerate(ordered_ids)})
        subset = subset.sort_values("display_order", kind="mergesort")
        if subset.empty:
            ax.set_visible(False)
            continue

        ax.pie(
            subset["n_rows"].to_numpy(),
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
            startangle=90,
            counterclock=False,
            colors=[color_map[int(cluster_id)] for cluster_id in subset["cluster_id"]],
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
            textprops={"fontsize": 9},
        )
        ax.set_title(year, fontsize=12, fontweight="bold")

    for ax in axes[len(YEAR_ORDER):]:
        ax.set_visible(False)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=metadata[cluster_id]["cluster_label"],
            markerfacecolor=color_map[cluster_id],
            markersize=10,
        )
        for cluster_id in ordered_ids
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(2, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("TikTok RFE k-medoids Cluster Shares Within Each Year", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assigned_df = pd.read_csv(args.assigned)
    profiles_df = pd.read_csv(args.cluster_profiles)
    chosen_cluster_col = cluster_column(args.k)

    if chosen_cluster_col not in assigned_df.columns:
        raise ValueError(f"Column {chosen_cluster_col!r} not found in {args.assigned}.")
    if "era" not in assigned_df.columns:
        raise ValueError("Missing required column 'era' in assigned rows file.")

    assigned_df["era"] = assigned_df["era"].astype("string").str.strip()
    assigned_df = assigned_df[assigned_df["era"].isin(YEAR_ORDER)].copy()
    assigned_df[chosen_cluster_col] = pd.to_numeric(assigned_df[chosen_cluster_col], errors="coerce")
    assigned_df = assigned_df.dropna(subset=[chosen_cluster_col]).copy()
    assigned_df[chosen_cluster_col] = assigned_df[chosen_cluster_col].astype(int)

    metadata, ordered_ids = cluster_display_metadata(
        profiles_df,
        args.k,
        use_final_k6_semantics=args.use_final_k6_semantics,
    )
    counts_df = build_counts(assigned_df, chosen_cluster_col)
    counts_path, shares_path = save_tables(counts_df, metadata, args.output_dir, args.k)

    bar_path = args.output_dir / f"tk_comments_kmedoids_k{args.k}_absolute_counts_bar.png"
    pies_path = args.output_dir / f"tk_comments_kmedoids_k{args.k}_shares_pies.png"
    plot_absolute_counts(counts_df, metadata, ordered_ids, bar_path)
    plot_pies(counts_df, metadata, ordered_ids, pies_path)

    print(f"Saved counts table to {counts_path}")
    print(f"Saved shares table to {shares_path}")
    print(f"Saved absolute counts bar chart to {bar_path}")
    print(f"Saved shares pie chart panel to {pies_path}")


if __name__ == "__main__":
    main()
