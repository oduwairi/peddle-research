"""Shared fixtures for construction_v2 unit tests."""

from __future__ import annotations

import pytest

from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import Brief, BriefBridge, BriefProduct


@pytest.fixture
def sample_brief() -> Brief:
    return Brief(
        task="Write ad copy for the platform below.",
        product=BriefProduct(
            name="Compliantly",
            description=("Background-checks-as-a-service for fast-growth ops teams."),
            category="HR-tech",
            key_features=["SOC 2 audit trail", "API-first integrations"],
            unique_selling_points=["72-hour turnaround vs industry 7 days"],
            tone_signals=["professional"],
        ),
        bridge=BriefBridge(
            positioning="speed-first alternative to legacy HR vendors",
            target_audience="Series-A operations leads",
            angle="aspirational founder identity",
            buyer_pain="compliance review blocks weekly hires",
        ),
        platform="meta",
    )


@pytest.fixture
def sample_source_ad() -> SourceAd:
    return SourceAd(
        ad_id="ad-abc-123",
        platform="facebook",
        composite_score=0.84,
        headline="Hire fast. Stay compliant.",
        body=(
            "Compliantly turns weekly compliance reviews into a 72-hour "
            "background check pipeline. Built for ops teams who can't "
            "afford to slow down."
        ),
        description="Try free for 14 days.",
        cta="Start free trial",
        raw={"advertiser_name": "Compliantly Inc.", "content_safety_label": "safe"},
    )
