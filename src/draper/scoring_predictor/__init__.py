"""Learned text-only regressor for v3 ad-performance scores.

The v3 ``HybridScorer`` builds a composite score from post-publication signals
(Kaplan-Meier survival, weighted engagement volume + velocity). Those signals
are unavailable for synthetic ad copy that has never been published, so we
train a text-only regressor on the v3 corpus to predict the same four targets
from copy alone:

* ``composite`` — the v3 final score (the headline number).
* ``survivability``, ``engagement_volume``, ``engagement_velocity`` — the
  three normalized v3 sub-signals, kept as separate heads for explainability.
  Reddit and ``"other"`` ads have engagement signals dropped at scoring time
  (they are unreliable on those platforms), so the engagement heads are masked
  out of the loss for those rows during training.

This is Phase 1 of the plan in ``~/.claude/plans/ok-heres-important-point-
indexed-minsky.md``: train + offline-validate. Phase 2 wires the regressor
into ``src/draper/evaluation/`` as an absolute scorer arm.

Public surface (intentionally narrow — Phase 2 should only need ``score_text``
and ``load_predictor``):

* :func:`load_predictor` — load a trained checkpoint + calibrators from disk.
* :class:`ScoringPredictor.score_text` — score a single ad or batch.
"""

from __future__ import annotations

from draper.scoring_predictor.inference import ScoringPredictor, load_predictor

__all__ = ["ScoringPredictor", "load_predictor"]
