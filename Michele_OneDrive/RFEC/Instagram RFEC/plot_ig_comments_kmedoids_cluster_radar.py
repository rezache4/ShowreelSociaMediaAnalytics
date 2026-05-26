#!/usr/bin/env python3
"""
Plot per-cluster radar charts that compare RFEC centroid evolution across eras.

Method:
- start from the row-level global k-medoids assignment file
- group by cluster and era, taking the arithmetic mean of the real-valued metrics
- apply log1p only to frequency centroids
- scale all centroid metrics globally to [0, 1]
- invert normalized recency so larger radar area means "hotter" users
- draw one radar subplot per cluster with one polygon per era
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ig_cluster_semantics import maybe_get_top_level_k6_semantics, ordered_cluster_ids

try:
    from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
except ImportError:  # pragma: no cover - environment-specific fallback
    SklearnMinMaxScaler = None


ERA_ORDER = ["2017-2019", "2020-2022", "2023-2026"]
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
ERA_COLORS = {
    "2017-2019": "#1f77b4",
    "2020-2022": "#ff7f0e",
    "2023-2026": "#2ca02c",
}
RAW_METRICS = ["recency", "frequency", "engagement", "clumpiness"]
PLOT_COLUMNS = [
    "frequency_log_scaled",
    "engagement_scaled",
    "clumpiness_scaled",
    "recency_inverted_scaled",
]
AXIS_LABELS = ["Frequency (log)", "Engagement", "Clumpiness", "Recency (inv)"]


class FallbackMinMaxScaler:
    """Minimal drop-in replacement for sklearn's MinMaxScaler."""

    def fit(self, values: np.ndarray) -> "FallbackMinMaxScaler":
        values = np.asarray(values, dtype=float)
        self.data_min_ = values.min(axis=0)
        self.data_max_ = values.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        safe_range = np.where(self.data_range_ == 0.0, 1.0, self.data_range_)
        scaled = (values - self.data_min_) / safe_range
        scaled[:, self.data_range_ == 0.0] = 0.0
        return scaled

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot radar charts of RFEC centroid evolution by cluster and era."
    )
    parser.add_argument(
        "--input",
        default=SCRIPT_DIR / "ig_comments_RFE_macro_kmedoids_clusters.csv",
        type=Path,
        help="Input CSV with real metrics and global cluster assignments.",
    )
    parser.add_argument(
        "--k",
        default=6,
        type=int,
        help="Cluster solution to use when `cluster_k<k>` exists. Default: 6",
    )
    parser.add_argument(
        "--output-dir",
        default=WORKSPACE_DIR / "cluster_radar_plots",
        type=Path,
        help="Directory where radar charts and centroid tables will be saved.",
    )
    parser.add_argument(
        "--top-level-k6-semantics",
        action="store_true",
        help="Apply the named top-level-only k=6 cluster order and labels.",
    )
    return parser.parse_args()


def cluster_name_map(
    cluster_ids: list[int],
    k: int,
    use_top_level_k6_semantics: bool,
) -> tuple[dict[int, dict[str, object]], list[int]]:
    if use_top_level_k6_semantics:
        semantics = maybe_get_top_level_k6_semantics(cluster_ids, k)
        if semantics is None:
            raise ValueError("The requested top-level k=6 semantics are only available for the full 6-cluster solution.")
        metadata = {
            cluster_id: {
                "display_order": semantics[cluster_id].display_order,
                "display_code": semantics[cluster_id].display_code,
                "cluster_name": semantics[cluster_id].cluster_name,
                "cluster_label": semantics[cluster_id].display_label,
            }
            for cluster_id in cluster_ids
        }
        return metadata, ordered_cluster_ids(cluster_ids, semantics)

    metadata = {
        cluster_id: {
            "display_order": cluster_id,
            "display_code": f"C{cluster_id}",
            "cluster_name": f"Cluster {cluster_id}",
            "cluster_label": f"C{cluster_id} Cluster {cluster_id}",
        }
        for cluster_id in cluster_ids
    }
    return metadata, sorted(cluster_ids)


def pick_cluster_column(df: pd.DataFrame, k: int) -> str:
    preferred = f"cluster_k{k}"
    fallback = "cluster_best_kmedoids"

    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise ValueError(f"Neither {preferred!r} nor {fallback!r} found in the input file.")


def load_rows(path: Path, k: int) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    cluster_column = pick_cluster_column(df, k)
    required_columns = ["era", cluster_column] + RAW_METRICS
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = df[df["era"].isin(ERA_ORDER)].copy()
    for column in RAW_METRICS + [cluster_column]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    rows = rows.dropna(subset=required_columns).copy()
    rows[cluster_column] = rows[cluster_column].astype(int)
    return rows, cluster_column


def build_centroids(rows: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    centroids = (
        rows.groupby([cluster_column, "era"], as_index=False)[RAW_METRICS]
        .mean()
        .rename(columns={cluster_column: "cluster_id"})
    )
    centroids["frequency_log"] = np.log1p(centroids["frequency"])
    return centroids


def build_scaler():
    if SklearnMinMaxScaler is not None:
        return SklearnMinMaxScaler()
    return FallbackMinMaxScaler()


def scale_centroids(centroids: pd.DataFrame) -> pd.DataFrame:
    scaled = centroids.copy()
    scaler_input_columns = ["recency", "frequency_log", "engagement", "clumpiness"]
    scaler_output_columns = [
        "recency_scaled",
        "frequency_log_scaled",
        "engagement_scaled",
        "clumpiness_scaled",
    ]

    scaler = build_scaler()
    scaled_values = scaler.fit_transform(scaled[scaler_input_columns].to_numpy(dtype=float))
    scaled[scaler_output_columns] = scaled_values
    scaled["recency_inverted_scaled"] = 1.0 - scaled["recency_scaled"]
    return scaled


def close_polygon(values: list[float], angles: np.ndarray) -> tuple[list[float], np.ndarray]:
    closed_values = values + [values[0]]
    closed_angles = np.concatenate([angles, [angles[0]]])
    return closed_values, closed_angles


def wrapped_title(display_code: str, cluster_name: str, width: int = 22) -> str:
    wrapped = "\n".join(textwrap.wrap(cluster_name, width=width))
    return f"{display_code}\n{wrapped}"


def plot_radar_panel(
    centroids_scaled: pd.DataFrame,
    cluster_ids: list[int],
    metadata: dict[int, dict[str, object]],
    output_path: Path,
) -> None:
    n_clusters = len(cluster_ids)
    ncols = 3
    nrows = math.ceil(n_clusters / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 5.8 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axes_array = np.atleast_1d(axes).ravel()

    n_axes = len(AXIS_LABELS)
    base_angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)

    for axis, cluster_id in zip(axes_array, cluster_ids):
        cluster_rows = centroids_scaled[centroids_scaled["cluster_id"] == cluster_id].copy()

        axis.set_theta_offset(np.pi / 2)
        axis.set_theta_direction(-1)
        axis.set_xticks(base_angles)
        axis.set_xticklabels(AXIS_LABELS, fontsize=10)
        axis.set_ylim(0, 1)
        axis.set_yticks([0.25, 0.50, 0.75, 1.00])
        axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#666666")
        axis.grid(color="#cccccc", alpha=0.7)
        axis.set_title(
            wrapped_title(
                str(metadata[cluster_id]["display_code"]),
                str(metadata[cluster_id]["cluster_name"]),
            ),
            pad=20,
            fontsize=12,
            fontweight="bold",
        )

        for era in ERA_ORDER:
            era_row = cluster_rows[cluster_rows["era"] == era]
            if era_row.empty:
                continue

            values = era_row.iloc[0][PLOT_COLUMNS].to_numpy(dtype=float).tolist()
            closed_values, closed_angles = close_polygon(values, base_angles)
            axis.plot(closed_angles, closed_values, color=ERA_COLORS[era], linewidth=2, label=era)
            axis.fill(closed_angles, closed_values, color=ERA_COLORS[era], alpha=0.25)

    for axis in axes_array[n_clusters:]:
        axis.set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=ERA_COLORS[era], linewidth=2, label=era) for era in ERA_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle("RFEC Cluster Centroids by Era", fontsize=17, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tables(
    centroids_raw: pd.DataFrame,
    centroids_scaled: pd.DataFrame,
    metadata: dict[int, dict[str, object]],
    output_dir: Path,
    k: int,
) -> tuple[Path, Path]:
    raw_out = centroids_raw.copy()
    raw_out.insert(1, "display_order", raw_out["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["display_order"]))
    raw_out.insert(2, "display_cluster_code", raw_out["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["display_code"]))
    raw_out.insert(3, "cluster_name", raw_out["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["cluster_name"]))
    raw_out.insert(4, "cluster_label", raw_out["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["cluster_label"]))
    raw_out = raw_out.sort_values(["display_order", "era"], kind="mergesort").reset_index(drop=True)
    raw_path = output_dir / f"ig_comments_kmedoids_k{k}_cluster_era_centroids_raw.csv"
    raw_out.to_csv(raw_path, index=False, encoding="utf-8-sig")

    scaled_out = centroids_scaled.copy()
    scaled_out.insert(1, "display_order", scaled_out["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["display_order"]))
    scaled_out.insert(2, "display_cluster_code", scaled_out["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["display_code"]))
    scaled_out.insert(3, "cluster_name", scaled_out["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["cluster_name"]))
    scaled_out.insert(4, "cluster_label", scaled_out["cluster_id"].map(lambda cluster_id: metadata[int(cluster_id)]["cluster_label"]))
    scaled_out = scaled_out.sort_values(["display_order", "era"], kind="mergesort").reset_index(drop=True)
    scaled_path = output_dir / f"ig_comments_kmedoids_k{k}_cluster_era_centroids_scaled.csv"
    scaled_out.to_csv(scaled_path, index=False, encoding="utf-8-sig")
    return raw_path, scaled_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, cluster_column = load_rows(args.input, args.k)
    centroids_raw = build_centroids(rows, cluster_column)
    centroids_scaled = scale_centroids(centroids_raw)

    cluster_ids = sorted(centroids_scaled["cluster_id"].unique().tolist())
    metadata, ordered_ids = cluster_name_map(
        cluster_ids,
        args.k,
        args.top_level_k6_semantics,
    )

    radar_path = args.output_dir / f"ig_comments_kmedoids_k{args.k}_cluster_radar_by_era.png"
    raw_path, scaled_path = save_tables(centroids_raw, centroids_scaled, metadata, args.output_dir, args.k)
    plot_radar_panel(centroids_scaled, ordered_ids, metadata, radar_path)

    print(f"Assignment column used: {cluster_column}")
    print(f"Saved radar chart panel to {radar_path}")
    print(f"Saved raw centroids to {raw_path}")
    print(f"Saved scaled centroids to {scaled_path}")


if __name__ == "__main__":
    main()
