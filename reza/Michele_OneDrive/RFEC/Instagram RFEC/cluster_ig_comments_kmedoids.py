#!/usr/bin/env python3
"""
Run era-balanced weighted k-medoids clustering on Instagram RFEC class profiles.

Methodological choices:
- input features are the 4 ordinal class variables (1=Low, 2=Medium, 3=High)
- clustering is fit once on all user-era observations to obtain a stable set of
  commentator segments that can be compared across eras
- each era contributes the same total weight to the objective function, so the
  largest era does not dominate the clustering only because it has more rows
- repeated class profiles are aggregated and weighted, which is exact here
  because distance depends only on the 4 class variables
- Manhattan distance is used because all features share the same ordinal 1-3 scale
- k-medoids is implemented from scratch with BUILD initialization and SWAP
  optimization
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "recency_class",
    "frequency_class",
    "engagement_class",
    "clumpiness_class",
]


@dataclass
class KMedoidsResult:
    k: int
    medoid_indices: np.ndarray
    cluster_labels: np.ndarray
    total_cost: float
    weighted_silhouette: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster Instagram RFEC classes with era-balanced weighted k-medoids."
    )
    parser.add_argument(
        "--input",
        default="ig_comments_RFE_macro_classified.csv",
        type=Path,
        help="Input classified macro CSV.",
    )
    parser.add_argument(
        "--ks",
        default="4,5,6",
        type=str,
        help="Comma-separated k values to evaluate. Default: 4,5,6",
    )
    parser.add_argument(
        "--output-assigned",
        default="ig_comments_RFE_macro_kmedoids_clusters.csv",
        type=Path,
        help="Output CSV with cluster assignments for each evaluated k and the selected solution.",
    )
    parser.add_argument(
        "--output-models",
        default="ig_comments_rfe_kmedoids_model_selection.csv",
        type=Path,
        help="Output CSV with model comparison across k values.",
    )
    parser.add_argument(
        "--output-profiles",
        default="ig_comments_rfe_kmedoids_cluster_profiles.csv",
        type=Path,
        help="Output CSV with cluster profiles for every evaluated k.",
    )
    parser.add_argument(
        "--output-era-composition",
        default="ig_comments_rfe_kmedoids_cluster_composition_by_era.csv",
        type=Path,
        help="Output CSV with the percentage composition of clusters within each era.",
    )
    parser.add_argument(
        "--output-value-shares",
        default="ig_comments_rfe_kmedoids_cluster_value_shares.csv",
        type=Path,
        help="Output CSV with class-share distributions inside each cluster.",
    )
    parser.add_argument(
        "--output-json",
        default="ig_comments_rfe_kmedoids_diagnostics.json",
        type=Path,
        help="Output JSON with diagnostics and medoid definitions.",
    )
    return parser.parse_args()


def load_classified_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = FEATURE_COLUMNS + ["era"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + ["era"]).copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(int)
    return df


def build_era_balanced_profiles(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    era_sizes = (
        df.groupby("era", dropna=False)
        .size()
        .reset_index(name="era_rows")
        .sort_values("era", kind="mergesort")
        .reset_index(drop=True)
    )
    mean_era_size = float(era_sizes["era_rows"].mean())

    profile_by_era = (
        df.groupby(["era"] + FEATURE_COLUMNS, dropna=False)
        .size()
        .reset_index(name="raw_count")
        .merge(era_sizes, on="era", how="left", validate="many_to_one")
    )
    profile_by_era["era_scaling_factor"] = mean_era_size / profile_by_era["era_rows"]
    profile_by_era["balanced_weight_component"] = (
        profile_by_era["raw_count"] * profile_by_era["era_scaling_factor"]
    )

    profiles = (
        profile_by_era.groupby(FEATURE_COLUMNS, dropna=False)
        .agg(
            weight=("balanced_weight_component", "sum"),
            raw_count=("raw_count", "sum"),
            eras_present=("era", "nunique"),
        )
        .reset_index()
        .sort_values(FEATURE_COLUMNS, kind="mergesort")
        .reset_index(drop=True)
    )
    profiles["profile_id"] = np.arange(len(profiles), dtype=int)
    return profiles, era_sizes, mean_era_size


def manhattan_distance_matrix(values: np.ndarray) -> np.ndarray:
    differences = np.abs(values[:, None, :] - values[None, :, :])
    return differences.sum(axis=2).astype(float)


def assignment_cost(
    distance_matrix: np.ndarray,
    weights: np.ndarray,
    medoid_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    distances_to_medoids = distance_matrix[:, medoid_indices]
    nearest_positions = np.argmin(distances_to_medoids, axis=1)
    nearest_distances = distances_to_medoids[np.arange(len(distance_matrix)), nearest_positions]
    cluster_labels = nearest_positions.astype(int)
    total_cost = float(np.dot(weights, nearest_distances))
    return cluster_labels, nearest_distances, total_cost


def build_initial_medoids(distance_matrix: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    n_profiles = len(weights)
    medoids: list[int] = []

    first_costs = distance_matrix.T @ weights
    first_medoid = int(np.argmin(first_costs))
    medoids.append(first_medoid)

    nearest_distances = distance_matrix[:, first_medoid].copy()

    while len(medoids) < k:
        best_candidate: int | None = None
        best_cost = float("inf")

        for candidate in range(n_profiles):
            if candidate in medoids:
                continue

            candidate_distances = np.minimum(nearest_distances, distance_matrix[:, candidate])
            candidate_cost = float(np.dot(weights, candidate_distances))

            if candidate_cost < best_cost - 1e-12:
                best_cost = candidate_cost
                best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError("Failed to select an initial medoid.")

        medoids.append(best_candidate)
        nearest_distances = np.minimum(nearest_distances, distance_matrix[:, best_candidate])

    return np.array(medoids, dtype=int)


def swap_optimize_medoids(
    distance_matrix: np.ndarray,
    weights: np.ndarray,
    medoid_indices: np.ndarray,
) -> np.ndarray:
    medoids = medoid_indices.copy()
    n_profiles = len(weights)

    while True:
        _, _, current_cost = assignment_cost(distance_matrix, weights, medoids)
        best_improvement = 0.0
        best_swap: tuple[int, int] | None = None

        medoid_set = set(medoids.tolist())
        non_medoids = [index for index in range(n_profiles) if index not in medoid_set]

        for medoid_position, _medoid_index in enumerate(medoids):
            for candidate in non_medoids:
                trial = medoids.copy()
                trial[medoid_position] = candidate
                _, _, trial_cost = assignment_cost(distance_matrix, weights, trial)
                improvement = current_cost - trial_cost

                if improvement > best_improvement + 1e-12:
                    best_improvement = improvement
                    best_swap = (medoid_position, candidate)

        if best_swap is None:
            return np.sort(medoids)

        medoid_position, candidate = best_swap
        medoids[medoid_position] = candidate


def weighted_silhouette(
    distance_matrix: np.ndarray,
    weights: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    total_weight = float(weights.sum())
    unique_clusters = np.unique(cluster_labels)
    cluster_weight = {
        cluster: float(weights[cluster_labels == cluster].sum()) for cluster in unique_clusters
    }

    silhouettes = np.zeros(len(weights), dtype=float)

    for index in range(len(weights)):
        own_cluster = int(cluster_labels[index])
        own_cluster_mask = cluster_labels == own_cluster
        own_cluster_weight = cluster_weight[own_cluster]

        if own_cluster_weight <= 1.0:
            silhouettes[index] = 0.0
            continue

        same_cluster_distances = distance_matrix[index, own_cluster_mask]
        same_cluster_weights = weights[own_cluster_mask]

        a_numerator = float(np.dot(same_cluster_distances, same_cluster_weights))
        a_denominator = own_cluster_weight - 1.0
        a_value = a_numerator / a_denominator if a_denominator > 0.0 else 0.0

        b_value = float("inf")
        for other_cluster in unique_clusters:
            if other_cluster == own_cluster:
                continue
            other_mask = cluster_labels == other_cluster
            other_distances = distance_matrix[index, other_mask]
            other_weights = weights[other_mask]
            mean_distance = float(np.dot(other_distances, other_weights) / cluster_weight[other_cluster])
            if mean_distance < b_value:
                b_value = mean_distance

        if b_value == float("inf"):
            silhouettes[index] = 0.0
            continue

        denominator = max(a_value, b_value)
        silhouettes[index] = 0.0 if denominator <= 0.0 else (b_value - a_value) / denominator

    return float(np.dot(weights, silhouettes) / total_weight)


def run_weighted_kmedoids(
    distance_matrix: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> KMedoidsResult:
    medoids = build_initial_medoids(distance_matrix, weights, k)
    medoids = swap_optimize_medoids(distance_matrix, weights, medoids)
    cluster_labels, _, total_cost = assignment_cost(distance_matrix, weights, medoids)
    silhouette = weighted_silhouette(distance_matrix, weights, cluster_labels)

    return KMedoidsResult(
        k=k,
        medoid_indices=np.sort(medoids),
        cluster_labels=cluster_labels,
        total_cost=total_cost,
        weighted_silhouette=silhouette,
    )


def remap_cluster_labels(result: KMedoidsResult, profile_values: np.ndarray) -> KMedoidsResult:
    medoid_profiles = profile_values[result.medoid_indices]
    order = np.lexsort(
        (
            medoid_profiles[:, 3],
            medoid_profiles[:, 2],
            medoid_profiles[:, 1],
            medoid_profiles[:, 0],
        )
    )

    old_to_new = {old_label: new_label for new_label, old_label in enumerate(order)}
    remapped_labels = np.array([old_to_new[label] for label in result.cluster_labels], dtype=int)
    remapped_medoids = result.medoid_indices[order]

    return KMedoidsResult(
        k=result.k,
        medoid_indices=remapped_medoids,
        cluster_labels=remapped_labels,
        total_cost=result.total_cost,
        weighted_silhouette=result.weighted_silhouette,
    )


def select_best_result(results: list[KMedoidsResult], weights: np.ndarray) -> KMedoidsResult:
    scored = []
    for result in results:
        cluster_sizes = np.bincount(result.cluster_labels, weights=weights, minlength=result.k)
        min_share = float(cluster_sizes.min() / weights.sum())
        scored.append((result, result.weighted_silhouette, min_share))

    scored.sort(
        key=lambda item: (
            item[1],
            item[2],
            -item[0].k,
        ),
        reverse=True,
    )
    return scored[0][0]


def build_model_selection_rows(results: list[KMedoidsResult], weights: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for result in results:
        cluster_sizes = np.bincount(result.cluster_labels, weights=weights, minlength=result.k)
        cluster_shares = cluster_sizes / weights.sum()
        rows.append(
            {
                "k": result.k,
                "weighted_silhouette": result.weighted_silhouette,
                "total_cost": result.total_cost,
                "min_balanced_cluster_weight": float(cluster_sizes.min()),
                "max_balanced_cluster_weight": float(cluster_sizes.max()),
                "min_balanced_cluster_share": float(cluster_shares.min()),
                "max_balanced_cluster_share": float(cluster_shares.max()),
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def cluster_profile_rows(
    rows_df: pd.DataFrame,
    assignment_column: str,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []

    total_rows = len(rows_df)

    for cluster_id in sorted(rows_df[assignment_column].dropna().unique()):
        subset = rows_df[rows_df[assignment_column] == cluster_id].copy()

        top_profiles = (
            subset.groupby(FEATURE_COLUMNS)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .head(5)
        )
        top_profile_text = " | ".join(
            [
                (
                    f"({int(row.recency_class)},{int(row.frequency_class)},"
                    f"{int(row.engagement_class)},{int(row.clumpiness_class)}): {int(row.n)}"
                )
                for row in top_profiles.itertuples(index=False)
            ]
        )

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "n_rows": int(len(subset)),
                "share_rows": float(len(subset) / total_rows),
                "mean_recency_class": float(subset["recency_class"].mean()),
                "mean_frequency_class": float(subset["frequency_class"].mean()),
                "mean_engagement_class": float(subset["engagement_class"].mean()),
                "mean_clumpiness_class": float(subset["clumpiness_class"].mean()),
                "mode_recency_class": int(subset["recency_class"].mode().iloc[0]),
                "mode_frequency_class": int(subset["frequency_class"].mode().iloc[0]),
                "mode_engagement_class": int(subset["engagement_class"].mode().iloc[0]),
                "mode_clumpiness_class": int(subset["clumpiness_class"].mode().iloc[0]),
                "top_profiles": top_profile_text,
            }
        )

    return pd.DataFrame(summaries).sort_values("cluster_id").reset_index(drop=True)


def add_assignments_to_rows(
    rows_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    results: list[KMedoidsResult],
    best_result: KMedoidsResult,
) -> pd.DataFrame:
    assignments = profiles_df[FEATURE_COLUMNS + ["profile_id"]].copy()

    for result in results:
        assignments[f"cluster_k{result.k}"] = result.cluster_labels + 1

    assignments["cluster_best_kmedoids"] = assignments[f"cluster_k{best_result.k}"]
    assignments["cluster_best_k"] = best_result.k

    merged = rows_df.merge(assignments, on=FEATURE_COLUMNS, how="left", validate="many_to_one")
    return merged


def diagnostics_payload(
    results: list[KMedoidsResult],
    best_result: KMedoidsResult,
    profiles_df: pd.DataFrame,
    model_selection_df: pd.DataFrame,
) -> dict[str, object]:
    candidates = []
    for result in results:
        medoid_rows = (
            medoid_rows_for_result(profiles_df, result)
            .rename(
                columns={
                    "medoid_profile_id": "profile_id",
                    "medoid_weight": "weight",
                    "medoid_raw_count": "raw_count",
                    "medoid_eras_present": "eras_present",
                    "medoid_recency_class": "recency_class",
                    "medoid_frequency_class": "frequency_class",
                    "medoid_engagement_class": "engagement_class",
                    "medoid_clumpiness_class": "clumpiness_class",
                }
            )
            .to_dict(orient="records")
        )
        candidates.append(
            {
                "k": result.k,
                "weighted_silhouette": result.weighted_silhouette,
                "total_cost": result.total_cost,
                "medoids": medoid_rows,
            }
        )

    return {
        "method": {
            "algorithm": "Era-balanced weighted k-medoids (BUILD + SWAP)",
            "distance": "Manhattan distance on ordinal class variables",
            "features": FEATURE_COLUMNS,
            "weighting": (
                "Unique class profiles weighted after rescaling each era to the same total size."
            ),
        },
        "model_selection": model_selection_df.to_dict(orient="records"),
        "best_k": int(best_result.k),
        "candidates": candidates,
    }


def medoid_rows_for_result(profiles_df: pd.DataFrame, result: KMedoidsResult) -> pd.DataFrame:
    profile_lookup = profiles_df.set_index("profile_id")
    rows: list[dict[str, object]] = []

    for cluster_id, profile_id in enumerate(result.medoid_indices, start=1):
        row = profile_lookup.loc[profile_id]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "medoid_profile_id": int(profile_id),
                "medoid_weight": float(row["weight"]),
                "medoid_raw_count": int(row["raw_count"]),
                "medoid_eras_present": int(row["eras_present"]),
                "medoid_recency_class": int(row["recency_class"]),
                "medoid_frequency_class": int(row["frequency_class"]),
                "medoid_engagement_class": int(row["engagement_class"]),
                "medoid_clumpiness_class": int(row["clumpiness_class"]),
            }
        )

    return pd.DataFrame(rows)


def build_cluster_profiles_for_all_results(
    assigned_rows: pd.DataFrame,
    profiles_df: pd.DataFrame,
    results: list[KMedoidsResult],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for result in results:
        assignment_column = f"cluster_k{result.k}"
        summary_df = cluster_profile_rows(assigned_rows, assignment_column)
        medoid_df = medoid_rows_for_result(profiles_df, result)
        merged = summary_df.merge(medoid_df, on="cluster_id", how="left", validate="one_to_one")
        merged.insert(0, "k", int(result.k))
        frames.append(merged)

    return pd.concat(frames, ignore_index=True).sort_values(["k", "cluster_id"]).reset_index(drop=True)


def build_era_composition_rows(
    assigned_rows: pd.DataFrame,
    results: list[KMedoidsResult],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    era_sizes = assigned_rows.groupby("era").size().to_dict()

    for result in results:
        cluster_column = f"cluster_k{result.k}"
        counts = (
            assigned_rows.groupby(["era", cluster_column])
            .size()
            .reset_index(name="n_rows")
            .rename(columns={cluster_column: "cluster_id"})
        )
        counts["era_total_rows"] = counts["era"].map(era_sizes)
        counts["share_within_era"] = counts["n_rows"] / counts["era_total_rows"]
        counts["k"] = int(result.k)
        rows.append(counts[["k", "era", "cluster_id", "n_rows", "era_total_rows", "share_within_era"]])

    return pd.concat(rows, ignore_index=True).sort_values(["k", "era", "cluster_id"]).reset_index(drop=True)


def build_cluster_value_shares(
    assigned_rows: pd.DataFrame,
    results: list[KMedoidsResult],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    metric_name_map = {
        "recency_class": "recency",
        "frequency_class": "frequency",
        "engagement_class": "engagement",
        "clumpiness_class": "clumpiness",
    }

    for result in results:
        cluster_column = f"cluster_k{result.k}"
        for cluster_id in sorted(assigned_rows[cluster_column].dropna().unique()):
            subset = assigned_rows[assigned_rows[cluster_column] == cluster_id]
            n_rows = len(subset)
            for feature in FEATURE_COLUMNS:
                counts = subset[feature].value_counts().sort_index()
                for class_value in (1, 2, 3):
                    share = float(counts.get(class_value, 0) / n_rows) if n_rows else 0.0
                    rows.append(
                        {
                            "k": int(result.k),
                            "cluster_id": int(cluster_id),
                            "metric": metric_name_map[feature],
                            "class_value": int(class_value),
                            "n_rows": int(counts.get(class_value, 0)),
                            "share_within_cluster": share,
                        }
                    )

    return (
        pd.DataFrame(rows)
        .sort_values(["k", "cluster_id", "metric", "class_value"], kind="mergesort")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    ks = sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()})
    if any(k < 2 for k in ks):
        raise ValueError("All k values must be at least 2.")

    rows_df = load_classified_rows(args.input)
    profiles_df, era_sizes_df, mean_era_size = build_era_balanced_profiles(rows_df)

    profile_values = profiles_df[FEATURE_COLUMNS].to_numpy(dtype=int)
    weights = profiles_df["weight"].to_numpy(dtype=float)
    distance_matrix = manhattan_distance_matrix(profile_values)

    results: list[KMedoidsResult] = []
    for k in ks:
        raw_result = run_weighted_kmedoids(distance_matrix, weights, k)
        remapped = remap_cluster_labels(raw_result, profile_values)
        results.append(remapped)

    best_result = select_best_result(results, weights)
    model_selection_df = build_model_selection_rows(results, weights)
    assigned_rows = add_assignments_to_rows(rows_df, profiles_df, results, best_result)
    cluster_profiles_df = build_cluster_profiles_for_all_results(assigned_rows, profiles_df, results)
    era_composition_df = build_era_composition_rows(assigned_rows, results)
    value_shares_df = build_cluster_value_shares(assigned_rows, results)
    payload = diagnostics_payload(results, best_result, profiles_df, model_selection_df)
    payload["era_sizes"] = era_sizes_df.to_dict(orient="records")
    payload["mean_era_size_for_balancing"] = mean_era_size
    payload["n_profiles"] = int(len(profiles_df))

    assigned_rows.to_csv(args.output_assigned, index=False, encoding="utf-8-sig")
    model_selection_df.to_csv(args.output_models, index=False)
    cluster_profiles_df.to_csv(args.output_profiles, index=False)
    era_composition_df.to_csv(args.output_era_composition, index=False)
    value_shares_df.to_csv(args.output_value_shares, index=False)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved row-level assignments to {args.output_assigned}")
    print(f"Saved model selection table to {args.output_models}")
    print(f"Saved cluster profiles to {args.output_profiles}")
    print(f"Saved era composition table to {args.output_era_composition}")
    print(f"Saved cluster value shares to {args.output_value_shares}")
    print(f"Saved diagnostics JSON to {args.output_json}")
    print("\nModel comparison:")
    print(model_selection_df.to_string(index=False))
    print(f"\nSelected best k: {best_result.k}")


if __name__ == "__main__":
    main()
