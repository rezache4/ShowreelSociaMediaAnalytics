#!/usr/bin/env python3
"""
Estimate Markov-style transition matrices for rolling user behavior states.

Input:
- full user-by-window state assignments including Inactive

Outputs:
- long transition table for consecutive windows T -> T+1
- transition count matrix
- transition probability matrix
- heatmap of transition probabilities
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


SCRIPT_DIR = Path(__file__).resolve().parent

CLUSTER_SUMMARY_PATH = SCRIPT_DIR / "ig_dynamic_rfe_cluster_summary.csv"
FULL_LONG_PATH = SCRIPT_DIR / "ig_users_dynamic_rfe_clusters_long_all_windows.csv"

TRANSITIONS_LONG_OUTPUT_PATH = SCRIPT_DIR / "ig_dynamic_rfe_transitions_long.csv"
COUNTS_MATRIX_OUTPUT_PATH = SCRIPT_DIR / "ig_dynamic_rfe_transition_counts_matrix.csv"
PROB_MATRIX_OUTPUT_PATH = SCRIPT_DIR / "ig_dynamic_rfe_transition_probabilities_matrix.csv"
HEATMAP_OUTPUT_PATH = SCRIPT_DIR / "ig_dynamic_rfe_transition_probability_heatmap.png"

INACTIVE_LABEL = "Inactive"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "blue_white_red_readable",
    ["#1F5AA6", "#F6F7FB", "#C63D2F"],
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_summary = pd.read_csv(
        CLUSTER_SUMMARY_PATH,
        dtype={
            "cluster_id": "string",
            "suggested_name": "string",
        },
    )
    full_long = pd.read_csv(
        FULL_LONG_PATH,
        dtype={
            "user_key": "string",
            "user_id": "string",
            "username": "string",
            "window_id": "string",
            "window_label": "string",
            "window_index": "Int64",
            "window_start": "string",
            "window_end": "string",
            "cluster_id": "string",
        },
    )
    return cluster_summary, full_long


def build_state_order(cluster_summary: pd.DataFrame) -> list[str]:
    active_states = cluster_summary.sort_values("cluster_id", kind="mergesort")["cluster_id"].astype(str).tolist()
    return active_states + [INACTIVE_LABEL]


def build_state_label_map(cluster_summary: pd.DataFrame) -> dict[str, str]:
    mapping = {
        str(row["cluster_id"]): f"{row['cluster_id']} - {row['suggested_name']}"
        for _, row in cluster_summary.iterrows()
    }
    mapping[INACTIVE_LABEL] = INACTIVE_LABEL
    return mapping


def build_transitions(full_long: pd.DataFrame) -> pd.DataFrame:
    ordered = full_long.sort_values(["user_key", "window_index"], kind="mergesort").reset_index(drop=True)
    ordered["next_window_id"] = ordered.groupby("user_key", sort=False)["window_id"].shift(-1)
    ordered["next_window_index"] = ordered.groupby("user_key", sort=False)["window_index"].shift(-1)
    ordered["next_window_start"] = ordered.groupby("user_key", sort=False)["window_start"].shift(-1)
    ordered["next_window_end"] = ordered.groupby("user_key", sort=False)["window_end"].shift(-1)
    ordered["next_cluster_id"] = ordered.groupby("user_key", sort=False)["cluster_id"].shift(-1)

    transitions = ordered.dropna(
        subset=["next_window_id", "next_window_index", "next_cluster_id"]
    ).copy()
    transitions["transition_step"] = transitions["window_id"] + " -> " + transitions["next_window_id"]
    transitions = transitions.rename(
        columns={
            "window_id": "current_window_id",
            "window_index": "current_window_index",
            "window_start": "current_window_start",
            "window_end": "current_window_end",
            "cluster_id": "current_state",
            "next_window_id": "future_window_id",
            "next_window_index": "future_window_index",
            "next_window_start": "future_window_start",
            "next_window_end": "future_window_end",
            "next_cluster_id": "future_state",
        }
    )
    transitions["current_window_index"] = transitions["current_window_index"].astype(int)
    transitions["future_window_index"] = transitions["future_window_index"].astype(int)
    return transitions[
        [
            "user_key",
            "user_id",
            "username",
            "current_window_id",
            "current_window_index",
            "current_window_start",
            "current_window_end",
            "current_state",
            "future_window_id",
            "future_window_index",
            "future_window_start",
            "future_window_end",
            "future_state",
            "transition_step",
        ]
    ].reset_index(drop=True)


def build_transition_matrices(transitions: pd.DataFrame, state_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(transitions["current_state"], transitions["future_state"])
    counts = counts.reindex(index=state_order, columns=state_order, fill_value=0)
    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return counts.astype(int), probabilities


def plot_transition_heatmap(probabilities: pd.DataFrame, state_label_map: dict[str, str]) -> None:
    row_labels = [state_label_map[state] for state in probabilities.index]
    col_labels = [state_label_map[state] for state in probabilities.columns]
    values = probabilities.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    image = ax.imshow(
        values,
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=float(values.max()) if values.size else 1.0,
    )

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Future State (T+1)")
    ax.set_ylabel("Current State (T)")
    ax.set_title("Markov Transition Matrix - Rolling Quadrimesters")

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="#1F1F1F",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Transition probability")
    fig.tight_layout()
    fig.savefig(HEATMAP_OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cluster_summary, full_long = load_inputs()
    state_order = build_state_order(cluster_summary)
    state_label_map = build_state_label_map(cluster_summary)

    transitions = build_transitions(full_long)
    counts, probabilities = build_transition_matrices(transitions, state_order)

    transitions.to_csv(TRANSITIONS_LONG_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    counts.to_csv(COUNTS_MATRIX_OUTPUT_PATH, encoding="utf-8-sig")
    probabilities.to_csv(PROB_MATRIX_OUTPUT_PATH, encoding="utf-8-sig")
    plot_transition_heatmap(probabilities, state_label_map)

    print(f"Saved {len(transitions):,} rows to {TRANSITIONS_LONG_OUTPUT_PATH}")
    print(f"Saved count matrix to {COUNTS_MATRIX_OUTPUT_PATH}")
    print(f"Saved probability matrix to {PROB_MATRIX_OUTPUT_PATH}")
    print(f"Saved heatmap to {HEATMAP_OUTPUT_PATH}")
    print(f"States included: {', '.join(state_order)}")


if __name__ == "__main__":
    main()
