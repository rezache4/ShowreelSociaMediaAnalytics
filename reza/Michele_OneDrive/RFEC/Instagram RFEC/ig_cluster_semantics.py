#!/usr/bin/env python3
"""
Semantic labels for the top-level-only Instagram k=6 RFEC clustering solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ClusterSemantic:
    source_cluster_id: int
    display_order: int
    display_code: str
    cluster_name: str
    recency_band: str
    frequency_band: str
    engagement_band: str
    clumpiness_band: str

    @property
    def display_label(self) -> str:
        return f"{self.display_code} {self.cluster_name}"


TOP_LEVEL_K6_CLUSTER_SEMANTICS = [
    ClusterSemantic(
        source_cluster_id=4,
        display_order=1,
        display_code="C1",
        cluster_name="Recent Broad Active",
        recency_band="Low/Mid",
        frequency_band="Mid/High",
        engagement_band="Any",
        clumpiness_band="Low/Mid",
    ),
    ClusterSemantic(
        source_cluster_id=2,
        display_order=2,
        display_code="C2",
        cluster_name="Recent Steady Conversational",
        recency_band="Low/Mid",
        frequency_band="Low/Mid",
        engagement_band="Any",
        clumpiness_band="Low/Mid",
    ),
    ClusterSemantic(
        source_cluster_id=3,
        display_order=3,
        display_code="C3",
        cluster_name="Recent Bursty Conversational",
        recency_band="Low/Mid",
        frequency_band="Low/Mid",
        engagement_band="Mid/High",
        clumpiness_band="High",
    ),
    ClusterSemantic(
        source_cluster_id=1,
        display_order=4,
        display_code="C4",
        cluster_name="Recent Bursty Light",
        recency_band="Low/Mid",
        frequency_band="Low/Mid",
        engagement_band="Low",
        clumpiness_band="Mid/High",
    ),
    ClusterSemantic(
        source_cluster_id=6,
        display_order=5,
        display_code="C5",
        cluster_name="Dormant Conversational",
        recency_band="Mid/High",
        frequency_band="Low/Mid",
        engagement_band="Mid/High",
        clumpiness_band="Any",
    ),
    ClusterSemantic(
        source_cluster_id=5,
        display_order=6,
        display_code="C6",
        cluster_name="Dormant Light",
        recency_band="Mid/High",
        frequency_band="Low/Mid",
        engagement_band="Low",
        clumpiness_band="Any",
    ),
]

TOP_LEVEL_K6_BY_SOURCE = {
    semantic.source_cluster_id: semantic for semantic in TOP_LEVEL_K6_CLUSTER_SEMANTICS
}


def maybe_get_top_level_k6_semantics(cluster_ids: Iterable[int], k: int) -> dict[int, ClusterSemantic] | None:
    if k != 6:
        return None
    ids = {int(cluster_id) for cluster_id in cluster_ids}
    if ids == set(TOP_LEVEL_K6_BY_SOURCE):
        return {cluster_id: TOP_LEVEL_K6_BY_SOURCE[cluster_id] for cluster_id in ids}
    return None


def ordered_cluster_ids(cluster_ids: Iterable[int], semantics: dict[int, ClusterSemantic] | None) -> list[int]:
    ids = [int(cluster_id) for cluster_id in cluster_ids]
    if semantics is None:
        return sorted(ids)
    return sorted(ids, key=lambda cluster_id: semantics[cluster_id].display_order)
