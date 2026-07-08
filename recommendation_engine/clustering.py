"""
clustering.py
=============
Groups crew members with similar fatigue *patterns* using K-Means on the
standardized feature table from `feature_extraction.py`.

Clustering only answers "who looks like whom" - it has no notion of
what's good or bad. We add two things on top of scikit-learn's raw
output so the rest of the system can use it meaningfully:

  1. **Automatic k selection** via silhouette score over a small
     candidate range, rather than a hard-coded cluster count - crew
     rosters vary in size and fatigue-pattern diversity, so a fixed k
     would either over-merge distinct risk groups or fragment a small
     roster into meaningless singleton-ish clusters.
  2. **Archetype labeling** derived from each cluster's centroid: for
     every feature we compute how many standard deviations the
     centroid sits from the overall population mean, then build a
     human-readable label from the most extreme, operationally
     meaningful features (e.g. "Chronic Sleep-Debt / Short-Rest
     Pattern"). This keeps the label an honest, inspectable summary of
     the actual centroid rather than an arbitrary cluster index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .feature_extraction import FEATURE_NAMES

# Candidate cluster counts to try; silhouette score picks the best.
# Bounded below by 2 (a single cluster is not a grouping) and above by
# a modest ceiling so clusters stay large enough to be operationally
# meaningful (rostering decisions, not per-person micro-clusters).
K_CANDIDATES = range(2, 7)
RANDOM_STATE = 42

# A feature is considered "elevated" or "depressed" in a cluster's
# centroid, and therefore eligible to drive its label, once it's at
# least this many standard deviations from the population mean.
LABEL_Z_THRESHOLD = 0.35

# Feature -> (label if centroid is HIGH, label if centroid is LOW).
# Only features with an operationally meaningful direction are listed;
# features not listed here (e.g. mean_sectors_per_duty alone) can still
# shift risk_tier but won't by themselves name a cluster.
FEATURE_LABEL_PHRASES: dict[str, tuple[str, str]] = {
    "mean_sleep_debt_hours": ("Chronic Sleep-Debt", "Well-Recovered"),
    "circadian_variability": ("Circadian-Disrupted", "Circadian-Stable"),
    "night_duty_fraction": ("Frequent Night Ops", "Mostly Day Ops"),
    "short_rest_frequency": ("Short-Rest Pattern", "Ample Rest"),
    "mean_process_w": ("High Post-Wake Inertia", "Low Sleep Inertia"),
    "sleep_quality_mean": ("Poor Sleep Quality", "High Sleep Quality"),
    "hrv_deviation": ("Physiological Strain", "Stable Physiology"),
    "mean_samn_perelli": ("High Self-Reported Fatigue", "Low Self-Reported Fatigue"),
    "mean_base_fatigue_score": ("Elevated Fatigue Risk", "Low Fatigue Risk"),
}

MAX_LABEL_TERMS = 2


@dataclass
class ClusteringResult:
    k: int
    labels: pd.Series  # crew_id -> cluster id
    centroids: pd.DataFrame  # cluster id -> feature values, ORIGINAL (unscaled) units
    archetypes: dict[int, str]  # cluster id -> human-readable label
    risk_rank: dict[int, int]  # cluster id -> 1 (highest risk) .. k (lowest risk)
    silhouette: float


def _pick_k(scaled: np.ndarray, k_candidates: range) -> tuple[int, np.ndarray, float]:
    best_k, best_labels, best_score = None, None, -1.0
    n_samples = scaled.shape[0]
    for k in k_candidates:
        if k >= n_samples:
            continue
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score
    if best_k is None:
        # Degenerate case (too few crew members to cluster meaningfully):
        # fall back to a single group.
        best_k, best_labels, best_score = 1, np.zeros(n_samples, dtype=int), 0.0
    return best_k, best_labels, best_score


def _label_cluster(centroid_z: pd.Series) -> str:
    candidates = []
    for feature, (high_phrase, low_phrase) in FEATURE_LABEL_PHRASES.items():
        z = centroid_z.get(feature, 0.0)
        if abs(z) < LABEL_Z_THRESHOLD:
            continue
        candidates.append((abs(z), high_phrase if z > 0 else low_phrase))

    if not candidates:
        return "Balanced / Average Fatigue Profile"

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    terms = []
    for _, phrase in candidates:
        if phrase not in terms:
            terms.append(phrase)
        if len(terms) == MAX_LABEL_TERMS:
            break
    return " + ".join(terms)


def cluster_crew(feature_df: pd.DataFrame, k_candidates: range = K_CANDIDATES) -> ClusteringResult:
    """
    Run K-Means over `feature_df` (crew_id-indexed, FEATURE_NAMES columns).

    Standardizes features (K-Means is distance-based, so unscaled
    features like "sleep hours" (~0-10) would be swamped by e.g. HRV
    deviation (~tens)), auto-selects k by silhouette score, and derives
    a human-readable archetype + risk rank per cluster.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df.values)

    k, labels, sil_score = _pick_k(scaled, k_candidates)

    labels_series = pd.Series(labels, index=feature_df.index, name="cluster")

    # Centroids in ORIGINAL units (means of the raw feature values per
    # cluster) - much easier for a human reviewer to sanity-check than
    # standardized centroids.
    centroids = feature_df.groupby(labels_series).mean()
    centroids.index.name = "cluster"

    # Z-score each cluster centroid against the overall population, in
    # standardized space, purely to drive labeling.
    pop_mean = feature_df.mean()
    pop_std = feature_df.std().replace(0, 1.0)
    centroid_z = (centroids - pop_mean) / pop_std

    archetypes = {int(cid): _label_cluster(centroid_z.loc[cid]) for cid in centroids.index}

    # Risk ranking: clusters with a higher mean_base_fatigue_score are
    # riskier. Rank 1 = highest risk.
    ordered = centroids["mean_base_fatigue_score"].sort_values(ascending=False)
    risk_rank = {int(cid): rank + 1 for rank, cid in enumerate(ordered.index)}

    return ClusteringResult(
        k=k,
        labels=labels_series,
        centroids=centroids,
        archetypes=archetypes,
        risk_rank=risk_rank,
        silhouette=float(sil_score),
    )
