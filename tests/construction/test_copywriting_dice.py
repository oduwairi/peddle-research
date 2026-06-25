"""Tests for the copywriting derived context (no RNG).

Covers:
  - SourceAdShape derivation from populated ad fields
  - CopywritingContext builder + sidecar serialization
  - Context directive renderer (currently emits nothing — platform
    framing was dropped after the 2026-04 audit)
"""

from __future__ import annotations

import pytest

from draper.construction.formats.copywriting.dice import (  # noqa: E402
    CopywritingContext,
    SourceAdShape,
    derive_copywriting_context,
    infer_source_ad_shape,
    render_context_directive,
)
from draper.construction.personas import PersonaLibrary  # noqa: E402
from draper.scoring.schemas import ScoredAd  # noqa: E402
from draper.scraping.schemas import AdCopy, AdSource, Platform, RawAd  # noqa: E402


def _scored_ad(
    *,
    has_body: bool = True,
    platform: Platform = Platform.FACEBOOK,
    vertical: str = "ecommerce",
) -> ScoredAd:
    ad = RawAd(
        ad_id="test-ad",
        source=AdSource.META_LIBRARY,
        platform=platform,
        ad_copy=AdCopy(
            headline="A clean testable headline",
            body="Body copy long enough to trip HAS_BODY. " * 3 if has_body else "",
            cta="Shop now",
        ),
        active_days=14,
        vertical=vertical,
        advertiser_name="Test Brand",
    )
    return ScoredAd(
        ad=ad,
        composite_score=0.8,
        signal_scores={"longevity": 0.8},
        tier="high",
    )


class TestSourceAdShape:
    def test_has_body(self) -> None:
        assert infer_source_ad_shape(_scored_ad(has_body=True)) == SourceAdShape.HAS_BODY

    def test_headline_only(self) -> None:
        assert (
            infer_source_ad_shape(_scored_ad(has_body=False))
            == SourceAdShape.HEADLINE_ONLY
        )


class TestDeriveCopywritingContext:
    def test_source_ad_shape_mirrors_ad(self) -> None:
        assert (
            derive_copywriting_context(_scored_ad(has_body=True)).source_ad_shape
            == SourceAdShape.HAS_BODY
        )
        assert (
            derive_copywriting_context(_scored_ad(has_body=False)).source_ad_shape
            == SourceAdShape.HEADLINE_ONLY
        )

    def test_invariant_to_platform(self) -> None:
        """Platform is a scraping-source artifact — it must not change the
        derived context. Same ad shape across platforms → same context.
        """
        shape = derive_copywriting_context(
            _scored_ad(platform=Platform.TIKTOK)
        ).source_ad_shape
        for plat in (Platform.REDDIT, Platform.FACEBOOK, Platform.PINTEREST):
            assert (
                derive_copywriting_context(_scored_ad(platform=plat)).source_ad_shape
                == shape
            )

    def test_deterministic(self) -> None:
        """Same ad → same context, every time. No RNG involved."""
        ad = _scored_ad(platform=Platform.TIKTOK, has_body=False)
        first = derive_copywriting_context(ad)
        second = derive_copywriting_context(ad)
        assert first == second


class TestContextDirective:
    def test_renders_empty_string(self) -> None:
        """Platform framing was dropped; source_ad_shape is provenance
        only. Directive text is empty until a new axis is added.
        """
        ctx = CopywritingContext(source_ad_shape=SourceAdShape.HEADLINE_ONLY)
        rendered = render_context_directive(ctx)
        assert rendered == ""
        # Old scaffolding must not reappear.
        assert "Platform framing" not in rendered
        assert "### Scenario" not in rendered
        assert "Rationale depth" not in rendered
        assert "Persona" not in rendered


class TestSidecar:
    def test_as_sidecar_keys(self) -> None:
        ctx = CopywritingContext(source_ad_shape=SourceAdShape.HEADLINE_ONLY)
        sidecar = ctx.as_sidecar()
        assert set(sidecar.keys()) == {"source_ad_shape"}
        assert sidecar["source_ad_shape"] == "headline_only"


class TestShippedConfigs:
    def test_personas_yaml_loads(self) -> None:
        """Personas remain in use for the four non-copywriting formats."""
        lib = PersonaLibrary.from_yaml("configs/personas.yaml")
        assert len(lib) >= 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
