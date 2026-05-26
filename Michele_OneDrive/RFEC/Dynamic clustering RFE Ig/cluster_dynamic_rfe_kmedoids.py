#!/usr/bin/env python3
"""
Build global behavioral clusters from rolling RFE classes using weighted k-medoids.

Workflow:
- Use all active user-window observations
- Collapse them into the 27 possible RFE class profiles
- Run weighted k-medoids on profile space for k from 3 to 8
- Select the best k using weighted silhouette, with interpretability checks
- Assign cluster labels to active user-window observations
- Assign "Inactive" to user-window combinations with no comments in that window

Outputs:
- k-search metrics
- profile-level cluster summaries
- final long and wide cluster assignments
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

CLASSIFICATION_LONG_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_level_classification_long.csv"
WINDOWS_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_rolling_windows.csv"

K_SEARCH_OUTPUT_PATH = SCRIPT_DIR / "ig_dynamic_rfe_kmedoids_k_search.csv"
FINAL_CLUSTER_SUMMARY_OUTPUT_PATH = SCRIPT_DIR / "ig_dynamic_rfe_cluster_summary.csv"
FINAL_PROFILE_MAP_OUTPUT_PATH = SCRIPT_DIR / "ig_dynamic_rfe_profile_to_cluster.csv"
FINAL_LONG_OUTPUT_PATH = SCRIPT_DIR / "ig_comments_dynamic_rfe_cluster_assignment_long.csv"
FINAL_WIDE_OUTPUT_PATH = SCRIPT_DIR / "ig_users_dynamic_rfe_clusters_wide.csv"
FINAL_LONG_ALL_WINDOWS_OUTPUT_PATH = SCRIPT_DIR / "ig_users_dynamic_rfe_clusters_long_all_windows.csv"

K_MIN = 3
K_MAX = 8
INACTIVE_LABEL = "Inactive"

LEVEL_TO_SCORE = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

SCORE_TO_LEVEL = {value: key for key, value in LEVEL_TO_SCORE.items()}

PREFERRED_CLUSTER_NAME_ORDER = [
    "Active Regulars",
    "Expressive Drop-Ins",
    "Recent Light Visitors",
    "Cooling Conversationalists",
    "Cooling Passersby",
    "Lapsed Enthusiasts",
    "Lost Conversationalists",
    "Lost Passersby",
]


@dataclass(frozen=True)
class KMedoidsResult:
    k: int
    medoid_indices: tuple[int, ...]
    labels: np.ndarray
    total_cost: float
    average_cost: float
    weighted_silhouette: float
    cluster_sizes: tuple[int, ...]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    classified = pd.read_csv(
        CLASSIFICATION_LONG_PATH,
        dtype={
            "window_id": "string",
            "window_label": "string",
            "window_index": "Int64",
            "window_start": "string",
            "window_end": "string",
            "user_key": "string",
            "user_id": "string",
            "username": "string",
            "recency_level": "string",
            "frequency_level": "string",
            "engagement_level": "string",
            "rfe_profile": "string",
        },
    )
    windows = pd.read_csv(
        WINDOWS_PATH,
        dtype={
            "window_id": "string",
            "window_label": "string",
            "window_index": "Int64",
            "window_start": "string",
            "window_end": "string",
        },
    )
    return classified, windows


def prepare_profile_space(classified: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    profiles = (
        classified.groupby(
            ["recency_level", "frequency_level", "engagement_level", "rfe_profile"],
            dropna=False,
            sort=False,
        )
        .size()
        .reset_index(name="weight")
        .copy()
    )

    profiles["recency_score"] = profiles["recency_level"].map(LEVEL_TO_SCORE).astype(int)
    profiles["frequency_score"] = profiles["frequency_level"].map(LEVEL_TO_SCORE).astype(int)
    profiles["engagement_score"] = profiles["engagement_level"].map(LEVEL_TO_SCORE).astype(int)
    profiles = profiles.sort_values(
        ["recency_score", "frequency_score", "engagement_score"],
        kind="mergesort",
    ).reset_index(drop=True)

    points = profiles[["recency_score", "frequency_score", "engagement_score"]].to_numpy(dtype=float)
    weights = profiles["weight"].to_numpy(dtype=float)
    return profiles, points, weights


def compute_manhattan_distances(points: np.ndarray) -> np.ndarray:
    return np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)


def assign_to_medoids(distance_matrix: np.ndarray, medoid_indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    medoid_indices = np.array(sorted(medoid_indices), dtype=int)
    distances_to_medoids = distance_matrix[:, medoid_indices]
    nearest_medoid_position = np.argmin(distances_to_medoids, axis=1)
    labels = nearest_medoid_position.astype(int)
    min_distances = distances_to_medoids[np.arange(distance_matrix.shape[0]), nearest_medoid_position]
    return labels, min_distances


def total_weighted_cost(distance_matrix: np.ndarray, weights: np.ndarray, medoid_indices: Iterable[int]) -> float:
    _, min_distances = assign_to_medoids(distance_matrix, medoid_indices)
    return float(np.dot(weights, min_distances))


def build_initial_medoids(distance_matrix: np.ndarray, weights: np.ndarray, k: int) -> list[int]:
    n_points = distance_matrix.shape[0]
    remaining = list(range(n_points))
    medoids: list[int] = []

    first_costs = np.array([np.dot(weights, distance_matrix[:, idx]) for idx in remaining], dtype=float)
    first_medoid = int(np.argmin(first_costs))
    medoids.append(first_medoid)
    remaining.remove(first_medoid)

    current_min = distance_matrix[:, first_medoid].copy()
    while len(medoids) < k:
        best_candidate = -1
        best_cost = float("inf")
        best_min = None
        for candidate in remaining:
            candidate_min = np.minimum(current_min, distance_matrix[:, candidate])
            candidate_cost = float(np.dot(weights, candidate_min))
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_candidate = candidate
                best_min = candidate_min
        medoids.append(best_candidate)
        remaining.remove(best_candidate)
        current_min = best_min  # type: ignore[assignment]

    return medoids


def pam_swap(distance_matrix: np.ndarray, weights: np.ndarray, initial_medoids: list[int]) -> list[int]:
    medoids = sorted(initial_medoids)
    current_cost = total_weighted_cost(distance_matrix, weights, medoids)
    improved = True

    while improved:
        improved = False
        best_swap = None
        best_cost = current_cost
        non_medoids = [idx for idx in range(distance_matrix.shape[0]) if idx not in medoids]

        for medoid in medoids:
            for candidate in non_medoids:
                swapped = [candidate if idx == medoid else idx for idx in medoids]
                swapped = sorted(swapped)
                candidate_cost = total_weighted_cost(distance_matrix, weights, swapped)
                if candidate_cost + 1e-9 < best_cost:
                    best_cost = candidate_cost
                    best_swap = swapped

        if best_swap is not None:
            medoids = best_swap
            current_cost = best_cost
            improved = True

    return medoids


def weighted_silhouette_score(distance_matrix: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> float:
    total_weight = float(weights.sum())
    unique_labels = np.unique(labels)
    cluster_weight_map = {label: float(weights[labels == label].sum()) for label in unique_labels}
    silhouettes = np.zeros(distance_matrix.shape[0], dtype=float)

    for idx in range(distance_matrix.shape[0]):
        own_label = labels[idx]
        own_mask = labels == own_label
        own_cluster_weight = cluster_weight_map[own_label]

        if own_cluster_weight <= 1:
            silhouettes[idx] = 0.0
            continue

        same_cluster_weights = weights[own_mask].copy()
        same_cluster_distances = distance_matrix[idx, own_mask]
        within_sum = float(np.dot(same_cluster_weights, same_cluster_distances))
        a_value = within_sum / max(own_cluster_weight - 1.0, 1.0)

        b_value = float("inf")
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_mask = labels == other_label
            other_cluster_weight = cluster_weight_map[other_label]
            if other_cluster_weight <= 0:
                continue
            between_sum = float(np.dot(weights[other_mask], distance_matrix[idx, other_mask]))
            candidate_b = between_sum / other_cluster_weight
            if candidate_b < b_value:
                b_value = candidate_b

        if not np.isfinite(b_value):
            silhouettes[idx] = 0.0
            continue

        denominator = max(a_value, b_value)
        silhouettes[idx] = 0.0 if denominator == 0 else (b_value - a_value) / denominator

    return float(np.dot(weights, silhouettes) / total_weight)


def run_weighted_kmedoids(distance_matrix: np.ndarray, weights: np.ndarray, k: int) -> KMedoidsResult:
    initial_medoids = build_initial_medoids(distance_matrix, weights, k)
    medoid_indices = tuple(pam_swap(distance_matrix, weights, initial_medoids))
    labels, min_distances = assign_to_medoids(distance_matrix, medoid_indices)

    cluster_sizes = []
    for label in range(k):
        cluster_sizes.append(int(weights[labels == label].sum()))

    total_cost = float(np.dot(weights, min_distances))
    average_cost = total_cost / float(weights.sum())
    silhouette = weighted_silhouette_score(distance_matrix, weights, labels)

    return KMedoidsResult(
        k=k,
        medoid_indices=medoid_indices,
        labels=labels,
        total_cost=total_cost,
        average_cost=average_cost,
        weighted_silhouette=silhouette,
        cluster_sizes=tuple(cluster_sizes),
    )


def choose_best_k(results: list[KMedoidsResult]) -> KMedoidsResult:
    ranked = sorted(
        results,
        key=lambda result: (
            round(result.weighted_silhouette, 10),
            -result.average_cost,
            -result.k,
        ),
        reverse=True,
    )

    best = ranked[0]
    best_cluster_share = min(best.cluster_sizes) / sum(best.cluster_sizes)

    if best_cluster_share < 0.03:
        viable = [
            result
            for result in ranked
            if min(result.cluster_sizes) / sum(result.cluster_sizes) >= 0.03
        ]
        if viable:
            best = viable[0]

    return best


def make_cluster_codes(k: int) -> list[str]:
    return [f"C{index}" for index in range(1, k + 1)]


def suggest_cluster_name(recency_level: str, frequency_level: str, engagement_level: str) -> str:
    if recency_level == "Low" and frequency_level == "Low" and engagement_level == "Low":
        return "Recent Light Visitors"
    if recency_level == "Low" and frequency_level == "Low" and engagement_level == "High":
        return "Expressive Drop-Ins"
    if recency_level == "Low" and frequency_level in {"Medium", "High"}:
        return "Active Regulars"
    if recency_level == "Medium" and frequency_level == "Low" and engagement_level == "Low":
        return "Cooling Passersby"
    if recency_level == "Medium" and frequency_level == "Low" and engagement_level == "Medium":
        return "Cooling Conversationalists"
    if recency_level in {"Medium", "High"} and frequency_level == "Low" and engagement_level == "High":
        return "Lapsed Enthusiasts"
    if recency_level == "High" and frequency_level == "Low" and engagement_level == "Low":
        return "Lost Passersby"
    if recency_level == "High" and frequency_level == "Low" and engagement_level == "Medium":
        return "Lost Conversationalists"
    return f"{recency_level} R / {frequency_level} F / {engagement_level} E"


def build_k_search_table(results: list[KMedoidsResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        cluster_sizes = list(result.cluster_sizes)
        rows.append(
            {
                "k": result.k,
                "weighted_silhouette": result.weighted_silhouette,
                "total_cost": result.total_cost,
                "average_cost": result.average_cost,
                "smallest_cluster_size": min(cluster_sizes),
                "largest_cluster_size": max(cluster_sizes),
                "smallest_cluster_share": min(cluster_sizes) / sum(cluster_sizes),
                "cluster_sizes": " | ".join(str(size) for size in cluster_sizes),
                "medoid_indices": " | ".join(str(index) for index in result.medoid_indices),
            }
        )
    return pd.DataFrame(rows).sort_values("k", kind="mergesort").reset_index(drop=True)


def build_profile_cluster_map(
    profiles: pd.DataFrame,
    result: KMedoidsResult,
) -> tuple[pd.DataFrame, dict[str, str]]:
    cluster_codes = make_cluster_codes(result.k)
    cluster_code_map = {cluster_index: cluster_codes[cluster_index] for cluster_index in range(result.k)}

    profile_map = profiles.copy()
    profile_map["cluster_index"] = result.labels
    profile_map["cluster_id"] = profile_map["cluster_index"].map(cluster_code_map)

    medoid_lookup: dict[int, dict[str, object]] = {}
    for cluster_index, medoid_index in enumerate(result.medoid_indices):
        medoid_row = profiles.iloc[medoid_index]
        medoid_lookup[cluster_index] = {
            "medoid_profile": medoid_row["rfe_profile"],
            "medoid_recency_level": medoid_row["recency_level"],
            "medoid_frequency_level": medoid_row["frequency_level"],
            "medoid_engagement_level": medoid_row["engagement_level"],
        }

    profile_map["medoid_profile"] = profile_map["cluster_index"].map(
        lambda cluster_index: medoid_lookup[int(cluster_index)]["medoid_profile"]
    )
    profile_map["medoid_recency_level"] = profile_map["cluster_index"].map(
        lambda cluster_index: medoid_lookup[int(cluster_index)]["medoid_recency_level"]
    )
    profile_map["medoid_frequency_level"] = profile_map["cluster_index"].map(
        lambda cluster_index: medoid_lookup[int(cluster_index)]["medoid_frequency_level"]
    )
    profile_map["medoid_engagement_level"] = profile_map["cluster_index"].map(
        lambda cluster_index: medoid_lookup[int(cluster_index)]["medoid_engagement_level"]
    )
    profile_map["distance_to_medoid"] = [
        int(
            abs(row["recency_score"] - LEVEL_TO_SCORE[row["medoid_recency_level"]])
            + abs(row["frequency_score"] - LEVEL_TO_SCORE[row["medoid_frequency_level"]])
            + abs(row["engagement_score"] - LEVEL_TO_SCORE[row["medoid_engagement_level"]])
        )
        for _, row in profile_map.iterrows()
    ]

    return profile_map.sort_values(["cluster_id", "weight"], ascending=[True, False], kind="mergesort"), cluster_code_map


def compute_group_medoid(cluster_rows: pd.DataFrame) -> pd.Series:
    points = cluster_rows[["recency_score", "frequency_score", "engagement_score"]].to_numpy(dtype=float)
    weights = cluster_rows["weight"].to_numpy(dtype=float)
    distance_matrix = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)
    candidate_costs = distance_matrix @ weights
    medoid_position = int(np.argmin(candidate_costs))
    return cluster_rows.iloc[medoid_position]


def build_cluster_summary(profile_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id, cluster_rows in profile_map.groupby("cluster_id", sort=True):
        total_weight = float(cluster_rows["weight"].sum())
        recency_mean = float(np.average(cluster_rows["recency_score"], weights=cluster_rows["weight"]))
        frequency_mean = float(np.average(cluster_rows["frequency_score"], weights=cluster_rows["weight"]))
        engagement_mean = float(np.average(cluster_rows["engagement_score"], weights=cluster_rows["weight"]))
        dominant_row = cluster_rows.sort_values("weight", ascending=False, kind="mergesort").iloc[0]
        medoid_row = compute_group_medoid(cluster_rows)

        dominant_recency = str(dominant_row["recency_level"])
        dominant_frequency = str(dominant_row["frequency_level"])
        dominant_engagement = str(dominant_row["engagement_level"])

        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(total_weight),
                "cluster_share": total_weight / float(profile_map["weight"].sum()),
                "medoid_profile": medoid_row["rfe_profile"],
                "medoid_recency_level": medoid_row["recency_level"],
                "medoid_frequency_level": medoid_row["frequency_level"],
                "medoid_engagement_level": medoid_row["engagement_level"],
                "mean_recency_score": recency_mean,
                "mean_frequency_score": frequency_mean,
                "mean_engagement_score": engagement_mean,
                "dominant_profile": dominant_row["rfe_profile"],
                "dominant_profile_share_within_cluster": float(dominant_row["weight"] / total_weight),
                "suggested_name": suggest_cluster_name(
                    str(medoid_row["recency_level"]),
                    str(medoid_row["frequency_level"]),
                    str(medoid_row["engagement_level"]),
                ),
                "behavioral_description": (
                    f"Medoid {medoid_row['rfe_profile']} | "
                    f"mean scores R={recency_mean:.2f}, F={frequency_mean:.2f}, E={engagement_mean:.2f} | "
                    f"dominant profile {dominant_row['rfe_profile']}"
                ),
                "dominant_recency_level": dominant_recency,
                "dominant_frequency_level": dominant_frequency,
                "dominant_engagement_level": dominant_engagement,
            }
        )

    summary = pd.DataFrame(rows).sort_values("cluster_id", kind="mergesort").reset_index(drop=True)
    return summary


def relabel_clusters_by_value(
    profile_map: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preferred_rank = {
        name: index for index, name in enumerate(PREFERRED_CLUSTER_NAME_ORDER, start=1)
    }

    relabel_frame = cluster_summary.copy()
    fallback_offset = len(PREFERRED_CLUSTER_NAME_ORDER) + 100
    relabel_frame["_rank"] = relabel_frame["suggested_name"].map(
        lambda value: preferred_rank.get(str(value), fallback_offset)
    )
    relabel_frame = relabel_frame.sort_values(
        ["_rank", "cluster_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    relabel_frame["new_cluster_id"] = [f"C{index}" for index in range(1, len(relabel_frame) + 1)]

    old_to_new = dict(zip(relabel_frame["cluster_id"], relabel_frame["new_cluster_id"]))

    profile_map = profile_map.copy()
    profile_map["cluster_id"] = profile_map["cluster_id"].map(old_to_new).astype("string")
    profile_map = profile_map.sort_values(
        ["cluster_id", "weight"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    cluster_summary = cluster_summary.copy()
    cluster_summary["cluster_id"] = cluster_summary["cluster_id"].map(old_to_new).astype("string")
    relabel_frame["new_cluster_id"] = relabel_frame["new_cluster_id"].astype("string")
    cluster_summary = cluster_summary.merge(
        relabel_frame[["new_cluster_id", "_rank"]],
        left_on="cluster_id",
        right_on="new_cluster_id",
        how="left",
    ).drop(columns="new_cluster_id")
    cluster_summary = cluster_summary.sort_values(
        ["_rank", "cluster_id"],
        kind="mergesort",
    ).drop(columns="_rank").reset_index(drop=True)
    return profile_map, cluster_summary


def build_active_cluster_assignments(
    classified: pd.DataFrame,
    profile_map: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    active = classified.merge(
        profile_map[
            [
                "rfe_profile",
                "cluster_id",
            ]
        ],
        on="rfe_profile",
        how="left",
    )
    active = active.merge(
        cluster_summary[
            [
                "cluster_id",
                "suggested_name",
                "medoid_profile",
                "medoid_recency_level",
                "medoid_frequency_level",
                "medoid_engagement_level",
            ]
        ].rename(columns={"suggested_name": "cluster_name"}),
        on="cluster_id",
        how="left",
    )
    return active.sort_values(["window_index", "user_key"], kind="mergesort").reset_index(drop=True)


def last_nonempty(series: pd.Series) -> str | pd._libs.missing.NAType:
    values = [value for value in series.dropna().astype(str) if value.strip()]
    if not values:
        return pd.NA
    return values[-1]


def build_full_user_window_matrix(active: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = (
        active.sort_values(["user_key", "window_index"], kind="mergesort")
        .groupby("user_key", sort=False)
        .agg(
            user_id=("user_id", last_nonempty),
            username=("username", last_nonempty),
        )
        .reset_index()
    )
    users = users.sort_values(["username", "user_key"], kind="mergesort").reset_index(drop=True)

    window_frame = windows[["window_id", "window_label", "window_index", "window_start", "window_end"]].copy()
    users["__join_key"] = 1
    window_frame["__join_key"] = 1
    full_grid = users.merge(window_frame, on="__join_key", how="outer").drop(columns="__join_key")

    full_long = full_grid.merge(
        active[
            [
                "user_key",
                "window_id",
                "cluster_id",
            ]
        ],
        on=["user_key", "window_id"],
        how="left",
    )
    full_long["cluster_id"] = full_long["cluster_id"].fillna(INACTIVE_LABEL)
    full_long = full_long.sort_values(["user_key", "window_index"], kind="mergesort").reset_index(drop=True)

    ordered_window_ids = windows.sort_values("window_index", kind="mergesort")["window_id"].tolist()
    wide = users.drop(columns="__join_key", errors="ignore").copy()
    pivot = full_long.pivot(index="user_key", columns="window_id", values="cluster_id")
    pivot = pivot.reindex(columns=ordered_window_ids)
    pivot = pivot.reset_index()
    wide = wide.merge(pivot, on="user_key", how="left")
    return full_long, wide


def main() -> None:
    classified, windows = load_inputs()
    profiles, points, weights = prepare_profile_space(classified)
    distance_matrix = compute_manhattan_distances(points)

    results = [run_weighted_kmedoids(distance_matrix, weights, k) for k in range(K_MIN, K_MAX + 1)]
    k_search = build_k_search_table(results)
    best_result = choose_best_k(results)

    profile_map, _ = build_profile_cluster_map(profiles, best_result)
    cluster_summary = build_cluster_summary(profile_map)
    profile_map, cluster_summary = relabel_clusters_by_value(profile_map, cluster_summary)
    active_assignments = build_active_cluster_assignments(classified, profile_map, cluster_summary)
    full_long, wide = build_full_user_window_matrix(active_assignments, windows)

    k_search["selected_k"] = k_search["k"] == best_result.k

    k_search.to_csv(K_SEARCH_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(FINAL_CLUSTER_SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    profile_map.to_csv(FINAL_PROFILE_MAP_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    active_assignments.to_csv(FINAL_LONG_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    full_long.to_csv(FINAL_LONG_ALL_WINDOWS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    wide.to_csv(FINAL_WIDE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(k_search):,} rows to {K_SEARCH_OUTPUT_PATH}")
    print(f"Saved {len(cluster_summary):,} rows to {FINAL_CLUSTER_SUMMARY_OUTPUT_PATH}")
    print(f"Saved {len(profile_map):,} rows to {FINAL_PROFILE_MAP_OUTPUT_PATH}")
    print(f"Saved {len(active_assignments):,} rows to {FINAL_LONG_OUTPUT_PATH}")
    print(f"Saved {len(full_long):,} rows to {FINAL_LONG_ALL_WINDOWS_OUTPUT_PATH}")
    print(f"Saved {len(wide):,} rows to {FINAL_WIDE_OUTPUT_PATH}")
    print(f"Selected k = {best_result.k}")
    print(f"Weighted silhouette = {best_result.weighted_silhouette:.4f}")
    print(f"Average cost = {best_result.average_cost:.4f}")


if __name__ == "__main__":
    main()
