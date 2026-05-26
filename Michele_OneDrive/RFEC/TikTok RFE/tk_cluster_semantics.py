#!/usr/bin/env python3
"""
Semantic labels for the final TikTok RFE k=6 clustering solution.
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

    @property
    def display_label(self) -> str:
        return f"{self.display_code} {self.cluster_name}"


FINAL_K6_CLUSTER_SEMANTICS = [
    ClusterSemantic(
        source_cluster_id=2,
        display_order=1,
        display_code="C1",
        cluster_name="Recent Conversational",
        recency_band="Low",
        frequency_band="Mid",
        engagement_band="Mid/High",
    ),
    ClusterSemantic(
        source_cluster_id=1,
        display_order=2,
        display_code="C2",
        cluster_name="Recent Light",
        recency_band="Low",
        frequency_band="Mid",
        engagement_band="Low",
    ),
    ClusterSemantic(
        source_cluster_id=5,
        display_order=3,
        display_code="C3",
        cluster_name="Fading Discursive",
        recency_band="Mid/High",
        frequency_band="Mid",
        engagement_band="High",
    ),
    ClusterSemantic(
        source_cluster_id=4,
        display_order=4,
        display_code="C4",
        cluster_name="Fading Conversational",
        recency_band="Mid/High",
        frequency_band="Mid",
        engagement_band="Mid",
    ),
    ClusterSemantic(
        source_cluster_id=3,
        display_order=5,
        display_code="C5",
        cluster_name="Cooling Light",
        recency_band="Mid",
        frequency_band="Mid",
        engagement_band="Low",
    ),
    ClusterSemantic(
        source_cluster_id=6,
        display_order=6,
        display_code="C6",
        cluster_name="Dormant Light",
        recency_band="High",
        frequency_band="Low/Mid",
        engagement_band="Low",
    ),
]

FINAL_K6_BY_SOURCE = {
    semantic.source_cluster_id: semantic for semantic in FINAL_K6_CLUSTER_SEMANTICS
}


def maybe_get_final_k6_semantics(cluster_ids: Iterable[int], k: int) -> dict[int, ClusterSemantic] | None:
    if k != 6:
        return None
    ids = {int(cluster_id) for cluster_id in cluster_ids}
    if ids == set(FINAL_K6_BY_SOURCE):
        return {cluster_id: FINAL_K6_BY_SOURCE[cluster_id] for cluster_id in ids}
    return None


def ordered_cluster_ids(cluster_ids: Iterable[int], semantics: dict[int, ClusterSemantic] | None) -> list[int]:
    ids = [int(cluster_id) for cluster_id in cluster_ids]
    if semantics is None:
        return sorted(ids)
    return sorted(ids, key=lambda cluster_id: semantics[cluster_id].display_order)
