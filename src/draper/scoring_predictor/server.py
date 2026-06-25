"""FastAPI app builder for the scoring-predictor HTTP service.

Shared between the Modal deployment (``deploy/modal_scoring_predictor.py``)
and the local dev server (``scripts/serve_scoring_predictor.py``). Both
target the same wire shape so the frontend talks to either with no
config differences beyond the base URL.

The app exposes:

* ``GET /healthz`` — liveness check; always 200 once the model is loaded.
* ``POST /score`` — batched scoring; requires ``X-API-Key`` header equal
  to the configured shared secret. Body is ``{items: [{platform, vertical,
  headline?, body?, description?}, ...]}`` (max 64 items per call).
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from draper.scoring_predictor.inference import ScoringPredictor


class ScoreItem(BaseModel):
    """One ad's worth of scorable copy + control tokens."""

    platform: str = Field(..., description="meta|tiktok|x|google|pinterest|reddit|other")
    vertical: str = Field(default="unknown")
    headline: str | None = None
    body: str | None = None
    description: str | None = None


class ScoreRequest(BaseModel):
    items: list[ScoreItem] = Field(default_factory=list, max_length=64)


class HeadScores(BaseModel):
    """Four-head score vector returned per scored item.

    Wire-contract parity: mirrors ``HeadScores`` in
    ``frontend/lib/agent/scoring/predictor-client.ts``. Any field rename here
    requires a matching rename there (and vice versa), and a matching update to
    ``tests/scoring_predictor/test_server.py::test_head_scores_fields``.
    """

    composite: float
    survivability: float
    engagement_volume: float
    engagement_velocity: float


class ScoreResponse(BaseModel):
    scores: list[HeadScores]
    latency_ms: float


def build_app(
    *,
    predictor: ScoringPredictor,
    api_key: str,
    checkpoint_label: str = "",
) -> FastAPI:
    """Build the scoring FastAPI app bound to a loaded predictor.

    Args:
        predictor: A loaded :class:`ScoringPredictor` (already on its target
            device with calibrators if available).
        api_key: Shared secret expected on the ``X-API-Key`` header. Must be
            non-empty — a blank key would let any caller through.
        checkpoint_label: Optional human-readable label echoed by ``/healthz``
            so deploy logs can confirm which weights are live.
    """
    if not api_key:
        raise ValueError("api_key must be a non-empty string")

    app = FastAPI()

    def _check_auth(x_api_key: str | None) -> None:
        import secrets

        if not x_api_key or not secrets.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "checkpoint": checkpoint_label}

    @app.post("/score", response_model=ScoreResponse)
    def score(
        req: ScoreRequest,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> ScoreResponse:
        _check_auth(x_api_key)
        if not req.items:
            return ScoreResponse(scores=[], latency_ms=0.0)

        t0 = time.perf_counter()
        items: list[dict[str, Any]] = [item.model_dump() for item in req.items]
        raw = predictor.score_many(items, batch_size=32)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        scores = [
            HeadScores(
                composite=row["composite"],
                survivability=row["survivability"],
                engagement_volume=row["engagement_volume"],
                engagement_velocity=row["engagement_velocity"],
            )
            for row in raw
        ]
        return ScoreResponse(scores=scores, latency_ms=latency_ms)

    return app
