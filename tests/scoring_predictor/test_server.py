"""Smoke tests for the scoring-predictor HTTP server layer.

These tests cover the FastAPI app builder (``build_app``) and request/response
Pydantic models in ``draper.scoring_predictor.server``, without requiring a
trained checkpoint on disk.  Behavioral tests (actual model inference) belong
in test_inference.py once the Phase-2 checkpoint is committed to CI fixtures.
"""

from __future__ import annotations

import importlib


def test_server_module_importable() -> None:
    """server.py must import cleanly without a checkpoint present."""
    mod = importlib.import_module("draper.scoring_predictor.server")
    # Public API the two call sites (Modal + local dev) depend on.
    assert callable(getattr(mod, "build_app", None)), "build_app must be exported"
    for model_name in ("ScoreItem", "ScoreRequest", "HeadScores", "ScoreResponse"):
        assert hasattr(mod, model_name), f"{model_name} missing from server module"


def test_head_scores_fields() -> None:
    """HeadScores must expose the four heads the TypeScript client mirrors.

    Wire-contract parity: keep in sync with
    ``frontend/lib/agent/scoring/predictor-client.ts :: HeadScores``.
    Any field rename here requires a matching rename there.
    """
    from draper.scoring_predictor.server import HeadScores

    hs = HeadScores(
        composite=0.8,
        survivability=0.7,
        engagement_volume=0.6,
        engagement_velocity=0.5,
    )
    assert hs.composite == 0.8
    assert hs.survivability == 0.7
    assert hs.engagement_volume == 0.6
    assert hs.engagement_velocity == 0.5


def test_build_app_rejects_blank_api_key() -> None:
    """build_app must raise ValueError for an empty api_key."""
    import pytest

    from draper.scoring_predictor.server import build_app

    # We need a predictor stub — build_app only calls it inside the route
    # handler, so a simple object with score_many is enough.
    class _StubPredictor:
        def score_many(self, items: list, batch_size: int = 32) -> list:
            return []

    with pytest.raises(ValueError, match="api_key"):
        build_app(predictor=_StubPredictor(), api_key="")  # type: ignore[arg-type]
