"""v2 quality filter: dedup + length + content-safety."""

from __future__ import annotations

from draper.construction_v2.config import FilterConfig
from draper.construction_v2.dataset.quality_filter import QualityFilter
from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import Brief
from draper.construction_v2.schemas.records import ExampleRecord


def _make_example(
    brief: Brief,
    *,
    ad_id: str = "ad-1",
    ad: str | None = None,
    think: str | None = None,
) -> ExampleRecord:
    return ExampleRecord(
        example_id=f"ex-{ad_id}",
        ad_id=ad_id,
        platform=brief.platform,
        brief=brief.model_dump(mode="json"),
        think=think or ("I want a crisp hook that lands the brief in one beat."),
        deliverable=ad or "Hire fast. Stay compliant. Free 14-day trial.",
        fidelity_coverage=0.9,
        fidelity_signature_passed=True,
        teacher_model="claude-haiku-4-5",
        batch_id="",
    )


def test_length_filter_rejects_short_ad(sample_brief: Brief) -> None:
    ex = _make_example(sample_brief, ad="x")
    result = QualityFilter(FilterConfig(min_deliverable_chars=10)).filter_all([ex])
    assert result.stats.passed == 0
    assert result.stats.rejected_length == 1
    assert "deliverable_too_short" in result.rejected[0].reason


def test_length_filter_rejects_long_assistant(sample_brief: Brief) -> None:
    huge = "x" * 50_000
    ex = _make_example(sample_brief, ad=huge)
    result = QualityFilter(FilterConfig(max_tokens=2_000)).filter_all([ex])
    assert result.stats.rejected_length == 1


def test_dedup_filter_drops_near_duplicates(sample_brief: Brief) -> None:
    a = _make_example(sample_brief, ad_id="a")
    b = _make_example(sample_brief, ad_id="b")  # identical brief + ad
    c = _make_example(
        sample_brief,
        ad_id="c",
        ad="A totally different ad about cupcakes for the weekend market.",
        think="Cupcakes are playful so I keep the rationale light.",
    )
    result = QualityFilter(FilterConfig()).filter_all([a, b, c])
    # a should be kept, b dropped, c kept.
    kept_ids = {ex.ad_id for ex in result.passed}
    assert "a" in kept_ids
    assert "c" in kept_ids
    assert "b" not in kept_ids
    assert result.stats.rejected_duplicate == 1


def test_content_safety_drops_unsafe(sample_brief: Brief) -> None:
    # Content-safety filtering moved to source_selector; QualityFilter
    # no longer checks it. This test is kept for documentation but
    # content_safety_drops_unsafe is now a selection gate.
    ex = _make_example(sample_brief)
    ads = {
        ex.ad_id: SourceAd(
            ad_id=ex.ad_id,
            platform="meta",
            composite_score=0.7,
            headline="x",
            body="y",
            description="",
            cta="",
            raw={"content_safety_label": "adult_sexual"},
        )
    }
    # The ads_by_id parameter is retained for compatibility but not used.
    result = QualityFilter(FilterConfig(), ads_by_id=ads).filter_all([ex])
    # Content-safety checks no longer occur in QualityFilter, so the example
    # passes the filter (it has no length/dedup issues).
    assert result.stats.passed == 1


def test_content_safety_keeps_safe(sample_brief: Brief) -> None:
    # Content-safety filtering moved to source_selector; QualityFilter
    # no longer checks it. This test verifies that safe-labelled content
    # is kept by the filter (which it is, since there are no other issues).
    ex = _make_example(sample_brief)
    ads = {
        ex.ad_id: SourceAd(
            ad_id=ex.ad_id,
            platform="meta",
            composite_score=0.7,
            headline="x",
            body="y",
            description="",
            cta="",
            raw={"content_safety_label": "safe"},
        )
    }
    result = QualityFilter(FilterConfig(), ads_by_id=ads).filter_all([ex])
    # With no length or dedup issues, example passes.
    assert result.stats.passed == 1


def test_filter_empty_input() -> None:
    result = QualityFilter(FilterConfig()).filter_all([])
    assert result.stats.passed == 0
    assert result.stats.total_input == 0
