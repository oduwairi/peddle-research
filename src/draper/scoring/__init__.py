"""Composite performance scoring and tier assignment.

Three scorer versions are available:
  - v1: ``CompositeScorer`` — hand-tuned 4-signal weighted sum.
  - v2: ``SnorkelScorer`` — Snorkel weak-supervision with 8 labeling functions.
  - v3: ``HybridScorer`` — replaces v1's longevity/early-death pair with
        per-platform Kaplan-Meier survival curves. Active input for construction.

Weights and tier thresholds are configured in ``configs/scoring.yaml``.
"""

from draper.scoring.composite_scorer import CompositeScorer
from draper.scoring.hybrid_scorer import HybridScorer
from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.snorkel_scorer import SnorkelScorer
from draper.scoring.tier_assigner import TierAssigner

__all__ = [
    "CompositeScorer",
    "HybridScorer",
    "ScoredAd",
    "ScoringConfig",
    "SnorkelScorer",
    "TierAssigner",
]
