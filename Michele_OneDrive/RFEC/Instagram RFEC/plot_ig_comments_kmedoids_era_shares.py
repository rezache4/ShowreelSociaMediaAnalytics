#!/usr/bin/env python3
"""
Plot era-wise cluster shares (pie charts) and absolute counts (bar chart)
from the Instagram k-medoids clustering outputs.
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

from ig_cluster_semantics import maybe_get_top_level_k6_semantics, ordered_cluster_ids


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
ERA_ORDER = ["2017-2019", "2020-2022", "2023-2026"]
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
        description="Plot era-wise cluster shares and absolute counts for Instagram k-medoids outputs."
    )
    parser.add_argument(
        "--assigned",
        default=SCRIPT_DIR / "ig_comments_RFE_macro_kmedoids_clusters.csv",
        type=Path,
        help="Row-level assignments CSV produced by cluster_ig_comments_kmedoids.py",
    )
    parser.add_argument(
        "--cluster-profiles",
        default=SCRIPT_DIR / "ig_comments_rfe_kmedoids_cluster_profiles.csv",
        type=Path,
        help="Cluster profiles CSV produced by cluster_ig_comments_kmedoids.py",
    )
    parser.add_argument(
        "--k",
        default=6,
        type=int,
        help="Which k solution to plot. Default: 6",
    )
    parser.add_argument(
        "--output-dir",
        default=WORKSPACE_DIR / "kmedoids_era_plots",
        type=Path,
        help="Directory where charts and tables will be saved.",
    )
    parser.add_argument(
        "--top-level-k6-semantics",
        action="store_true",
        help="Apply the named top-level-only k=6 cluster order and labels.",
    )
    return parser.parse_args()


def cluster_column(k: int) -> str:
    return f"cluster_k{k}"


def cluster_display_metadata(
    cluster_profile_df: pd.DataFrame,
    k: int,
    use_top_level_k6_semantics: bool,
) -> tuple[dict[int, dict[str, object]], list[int]]:
    subset = cluster_profile_df[cluster_profile_df["k"] == k].copy()
    if subset.empty:
        raise ValueError(f"No cluster profiles found for k={k}.")

    cluster_ids = subset["cluster_id"].astype(int).tolist()
    semantics = None
    if use_top_level_k6_semantics:
        semantics = maybe_get_top_level_k6_semantics(cluster_ids, k)
        if semantics is None:
            raise ValueError("The requested top-level k=6 semantics are only available for the full 6-cluster solution.")

    metadata: dict[int, dict[str, object]] = {}
    for row in subset.itertuples(index=False):
        cluster_id = int(row.cluster_id)
        if semantics is not None:
            semantic = semantics[cluster_id]
            metadata[cluster_id] = {
                "display_order": semantic.display_order,
                "display_code": semantic.display_code,
                "cluster_name": semantic.cluster_name,
                "cluster_label": semantic.display_label,
                "recency_band": semantic.recency_band,
                "frequency_band": semantic.frequency_band,
                "engagement_band": semantic.engagement_band,
                "clumpiness_band": semantic.clumpiness_band,
            }
            continue

        metadata[cluster_id] = {
            "display_order": cluster_id,
            "display_code": f"C{cluster_id}",
            "cluster_name": f"Cluster {cluster_id}",
            "cluster_label": (
                f"C{cluster_id} "
                f"({int(row.mode_recency_class)},{int(row.mode_frequency_class)},"
                f"{int(row.mode_engagement_class)},{int(row.mode_clumpiness_class)})"
            ),
            "recency_band": "",
            "frequency_band": "",
            "engagement_band": "",
            "clumpiness_band": "",
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
    enriched = counts_df.copy()
    enriched.insert(2, "display_order", enriched["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["display_order"]))
    enriched.insert(3, "display_cluster_code", enriched["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["display_code"]))
    enriched.insert(4, "cluster_name", enriched["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["cluster_name"]))
    enriched.insert(5, "cluster_label", enriched["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["cluster_label"]))
    enriched.insert(6, "recency_band", enriched["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["recency_band"]))
    enriched.insert(7, "frequency_band", enriched["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["frequency_band"]))
    enriched.insert(8, "engagement_band", enriched["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["engagement_band"]))
    enriched.insert(9, "clumpiness_band", enriched["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["clumpiness_band"]))
    enriched = enriched.sort_values(["era", "display_order"], kind="mergesort").reset_index(drop=True)

    counts_out = enriched[
        [
            "era",
            "cluster_id",
            "display_order",
            "display_cluster_code",
            "cluster_name",
            "cluster_label",
            "recency_band",
            "frequency_band",
            "engagement_band",
            "clumpiness_band",
            "n_rows",
            "era_total_rows",
        ]
    ].copy()
    counts_path = output_dir / f"ig_comments_kmedoids_k{k}_counts_by_era.csv"
    counts_out.to_csv(counts_path, index=False, encoding="utf-8-sig")

    shares_out = enriched[
        [
            "era",
            "cluster_id",
            "display_order",
            "display_cluster_code",
            "cluster_name",
            "cluster_label",
            "recency_band",
            "frequency_band",
            "engagement_band",
            "clumpiness_band",
            "share_within_era",
            "share_pct",
        ]
    ].copy()
    shares_path = output_dir / f"ig_comments_kmedoids_k{k}_shares_by_era_pct.csv"
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
        .reindex(ERA_ORDER)
        .fillna(0)
        .astype(int)
    )

    cluster_ids = [cluster_id for cluster_id in ordered_ids if cluster_id in pivot.columns]
    x = np.arange(len(ERA_ORDER))
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
                height + max(15.0, height * 0.01),
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ERA_ORDER)
    ax.set_ylabel("User-era count")
    ax.set_title("Instagram Top-level-comment k-medoids Cluster Counts by Era")
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
    cluster_ids = ordered_ids
    ncols = 3
    nrows = math.ceil(len(ERA_ORDER) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.8))
    axes = np.atleast_1d(axes).ravel()

    color_map = {cluster_id: CLUSTER_COLORS[index % len(CLUSTER_COLORS)] for index, cluster_id in enumerate(cluster_ids)}

    for ax, era in zip(axes, ERA_ORDER):
        subset = counts_df[counts_df["era"] == era].copy()
        subset["cluster_id"] = subset["cluster_id"].astype(int)
        subset["display_order"] = subset["cluster_id"].map({cluster_id: position for position, cluster_id in enumerate(cluster_ids)})
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
        ax.set_title(era, fontsize=12, fontweight="bold")

    for ax in axes[len(ERA_ORDER):]:
        ax.set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=metadata[cluster_id]["cluster_label"],
                   markerfacecolor=color_map[cluster_id], markersize=10)
        for cluster_id in cluster_ids
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(3, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Instagram Top-level-comment k-medoids Cluster Shares Within Each Era", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
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

    assigned_df = assigned_df[assigned_df["era"].isin(ERA_ORDER)].copy()
    assigned_df[chosen_cluster_col] = pd.to_numeric(assigned_df[chosen_cluster_col], errors="coerce")
    assigned_df = assigned_df.dropna(subset=[chosen_cluster_col]).copy()
    assigned_df[chosen_cluster_col] = assigned_df[chosen_cluster_col].astype(int)

    metadata, ordered_ids = cluster_display_metadata(
        profiles_df,
        args.k,
        use_top_level_k6_semantics=args.top_level_k6_semantics,
    )
    counts_df = build_counts(assigned_df, chosen_cluster_col)
    counts_path, shares_path = save_tables(counts_df, metadata, args.output_dir, args.k)

    bar_path = args.output_dir / f"ig_comments_kmedoids_k{args.k}_absolute_counts_bar.png"
    pies_path = args.output_dir / f"ig_comments_kmedoids_k{args.k}_shares_pies.png"
    plot_absolute_counts(counts_df, metadata, ordered_ids, bar_path)
    plot_pies(counts_df, metadata, ordered_ids, pies_path)

    print(f"Saved counts table to {counts_path}")
    print(f"Saved shares table to {shares_path}")
    print(f"Saved absolute counts bar chart to {bar_path}")
    print(f"Saved shares pie chart panel to {pies_path}")


if __name__ == "__main__":
    main()
