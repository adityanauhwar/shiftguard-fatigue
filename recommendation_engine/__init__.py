"""
recommendation_engine
======================
Personalized Adjustment Recommendation module for ShiftGuard.

Groups crew members into fatigue-pattern clusters (K-Means, unsupervised)
and layers a transparent, human-readable rule engine on top of each
individual's own metrics to produce specific, explainable rest / schedule
adjustment recommendations.

Two-stage design
-----------------
1. **Clustering (unsupervised, "what kind of fatigue profile is this?")**
   `feature_extraction.py` builds a per-crew feature vector from the
   biomathematical fatigue engine's historical outputs plus raw duty /
   sleep / self-report signals. `clustering.py` standardizes those
   features and runs K-Means (auto-selecting k via silhouette score),
   then derives a human-readable archetype label per cluster from its
   centroid (e.g. "Chronic Sleep-Debt", "Circadian-Disrupted / Night
   Ops", "Resilient / Low-Risk").

2. **Rule engine (supervised-by-domain-knowledge, "what should we do
   about it?")** `rules.py` never trusts the cluster label alone -
   clustering only tells you *which population a crew member resembles*,
   not what to do for *them* specifically. Each recommendation is
   triggered by the individual's own metrics against named, tunable
   thresholds, with the cluster archetype used only to add context and
   to weight priority (e.g. an above-average sleep debt inside a
   "Chronic Sleep-Debt" cluster is escalated relative to the same debt
   value in a "Resilient" cluster).

Public entry point: `recommender.generate_recommendations`
"""

from .recommender import generate_recommendations, generate_for_crew  # noqa: F401
