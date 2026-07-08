"""
scheduler_reporting.py
=======================
Tiny formatting helper shared by `scheduler.py`. Mirrors
`recommendation_engine.recommender._cluster_summary` so both modules
report cluster context in the same shape - kept as its own module rather
than imported from `recommendation_engine` directly since that function
is a private helper there, not part of its public API.
"""

from __future__ import annotations

import pandas as pd

from recommendation_engine.clustering import ClusteringResult


def cluster_summary(clustering: ClusteringResult, feature_df: pd.DataFrame) -> list[dict]:
    member_counts = clustering.labels.value_counts().to_dict()
    summary = []
    for cluster_id in sorted(clustering.centroids.index):
        summary.append({
            "cluster_id": int(cluster_id),
            "label": clustering.archetypes[cluster_id],
            "risk_rank": clustering.risk_rank[cluster_id],
            "member_count": int(member_counts.get(cluster_id, 0)),
            "centroid": {
                k: round(float(v), 2) for k, v in clustering.centroids.loc[cluster_id].to_dict().items()
            },
        })
    summary.sort(key=lambda c: c["risk_rank"])
    return summary
