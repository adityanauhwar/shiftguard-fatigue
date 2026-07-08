"""
recommender.py
==============
Orchestrates the Personalized Adjustment Recommendation pipeline:

    1. feature_extraction.build_feature_table(...)  -> per-crew fatigue-
       pattern features (re-runs fatigue_engine over recent duties plus
       raw workload/physiological/subjective signals).
    2. clustering.cluster_crew(...)                  -> K-Means groups of
       crew members with similar fatigue patterns, each with a human-
       readable archetype label.
    3. rules.generate_recommendations(...)           -> per-crew, per-
       individual recommendations (cluster used only for context /
       priority escalation - see rules.py docstring).

Public entry points
--------------------
    generate_recommendations(data)             -> full-roster result dict
    generate_for_crew(data, crew_id)            -> single crew member's
                                                    recommendation dict,
                                                    still clustered
                                                    against the full roster
                                                    so cluster context is
                                                    meaningful.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .clustering import ClusteringResult, cluster_crew
from .feature_extraction import build_feature_table
from .rules import generate_recommendations as _rule_engine
from .rules import overall_priority


def _crew_result(
    crew_id: int,
    feature_df: pd.DataFrame,
    clustering: ClusteringResult,
) -> dict:
    features = feature_df.loc[crew_id]
    cluster_id = int(clustering.labels.loc[crew_id])
    centroid = clustering.centroids.loc[cluster_id]

    recs = _rule_engine(features, centroid)

    return {
        "crew_id": int(crew_id),
        "cluster_id": cluster_id,
        "cluster_label": clustering.archetypes[cluster_id],
        "cluster_risk_rank": clustering.risk_rank[cluster_id],
        "overall_priority": overall_priority(recs),
        "features": {k: round(float(v), 2) for k, v in features.to_dict().items()},
        "recommendations": [r.to_dict() for r in recs],
    }


def _cluster_summary(clustering: ClusteringResult, feature_df: pd.DataFrame) -> list[dict]:
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


def generate_recommendations(data, crew_ids: Optional[list[int]] = None) -> dict:
    """
    Run the full Personalized Adjustment Recommendation pipeline for a
    roster (or a subset of `crew_ids`, though clustering is still fit
    against that subset - pass the full roster for stable clusters and
    filter the result afterwards if you only need a few members).

    Returns
    -------
    {
      "k": 4,
      "silhouette_score": 0.31,
      "clusters": [ {cluster_id, label, risk_rank, member_count, centroid}, ... ],
      "crew": [ {crew_id, cluster_id, cluster_label, cluster_risk_rank,
                 overall_priority, features, recommendations}, ... ]
    }
    """
    feature_df = build_feature_table(data, crew_ids=crew_ids)
    clustering = cluster_crew(feature_df)

    crew_results = [
        _crew_result(crew_id, feature_df, clustering) for crew_id in feature_df.index
    ]
    # Most urgent crew members first.
    from .rules import PRIORITY_ORDER
    crew_results.sort(key=lambda c: PRIORITY_ORDER.index(c["overall_priority"]), reverse=True)

    return {
        "k": clustering.k,
        "silhouette_score": round(clustering.silhouette, 3),
        "clusters": _cluster_summary(clustering, feature_df),
        "crew": crew_results,
    }


def generate_for_crew(data, crew_id: int) -> dict:
    """
    Convenience wrapper: run the pipeline against the full roster (so
    clustering - and therefore this person's cluster context - reflects
    the whole crew population), then return just this crew member's
    entry plus their cluster's summary.
    """
    full = generate_recommendations(data)
    crew_entry = next((c for c in full["crew"] if c["crew_id"] == crew_id), None)
    if crew_entry is None:
        raise KeyError(f"No crew member with crew_id={crew_id}")

    cluster_entry = next(
        c for c in full["clusters"] if c["cluster_id"] == crew_entry["cluster_id"]
    )
    return {
        **crew_entry,
        "cluster_summary": cluster_entry,
    }
