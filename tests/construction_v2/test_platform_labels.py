"""Unit tests for platform-native ad-copy label projection."""

from __future__ import annotations

import pytest

from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.platform_labels import (
    LabelResult,
    PlatformLabelGroup,
    check_platform_labels_present,
    platform_group_for,
    render_labeled_ad,
)


def _adflex_ad(
    *,
    platform: str,
    headline: str = "",
    body: str = "",
    description: str = "",
    cta: str = "",
) -> SourceAd:
    return SourceAd(
        ad_id="ad-test",
        platform=platform,
        composite_score=0.5,
        headline=headline,
        body=body,
        description=description,
        cta=cta,
        raw={"source": "adflex"},
    )


# ---------------------------------------------------------------------------
# render_labeled_ad — per-platform label vocabulary
# ---------------------------------------------------------------------------


def test_render_meta_uses_primary_text_for_adflex_title() -> None:
    ad = _adflex_ad(
        platform="facebook",
        headline="Anyone who wants to be healthy needs to pay attention to gut health.",
        body="Download Your FREE Copy Today!",
        description="Download a FREE copy of my Amazon Bestseller.",
        cta="Download",
    )
    rendered = render_labeled_ad(ad)
    assert "**Primary text:** Anyone who wants to be healthy" in rendered
    assert "**Headline:** Download Your FREE Copy Today!" in rendered
    assert "**Description:** Download a FREE copy of my Amazon Bestseller." in rendered
    assert "**CTA:** Download" in rendered


def test_render_meta_instagram_shares_meta_labels() -> None:
    ad = _adflex_ad(
        platform="instagram",
        headline="Primary copy here.",
        body="Headline copy.",
        cta="Shop Now",
    )
    rendered = render_labeled_ad(ad)
    assert rendered.startswith("**Primary text:** Primary copy here.")
    assert "**Headline:** Headline copy." in rendered
    assert "**CTA:** Shop Now" in rendered


def test_render_x_subtitle_is_cta_button_not_headline() -> None:
    # On AdFlex's X scrape, attachments[0].subtitle holds the CTA button
    # label, not a card headline. So flat ``body`` -> ``CTA``.
    ad = _adflex_ad(
        platform="twitter",
        headline="Shop must-haves for your home, kitchen, and outdoors.",
        body="Shop Now",
    )
    rendered = render_labeled_ad(ad)
    assert "**Tweet:** Shop must-haves" in rendered
    assert "**CTA:** Shop Now" in rendered
    # The X spec has no Card title when description is empty.
    assert "**Card title:**" not in rendered


def test_render_pinterest_uses_description_as_main_copy() -> None:
    ad = _adflex_ad(
        platform="pinterest",
        description="This Bohemian oasis will keep you calm and your home stylish.",
    )
    rendered = render_labeled_ad(ad)
    assert "**Description:** This Bohemian oasis" in rendered
    assert "**Title:**" not in rendered  # not populated -> not rendered


def test_render_reddit_only_emits_headline_and_cta() -> None:
    ad = _adflex_ad(
        platform="reddit",
        headline="The Majestic Goose made his epic cartoon available as a T-Shirt!",
        cta="Sign Up",
    )
    rendered = render_labeled_ad(ad)
    assert "**Headline:** The Majestic Goose" in rendered
    assert "**CTA:** Sign Up" in rendered
    # Reddit spec has no Description / Tweet / etc.
    assert "**Description:**" not in rendered
    assert "**Primary text:**" not in rendered


def test_render_tiktok_caption_and_cta() -> None:
    ad = _adflex_ad(
        platform="tiktok",
        headline="Couldn't do what we do without @Shopify 🌟📦💌 #ad",
        cta="Learn More",
    )
    rendered = render_labeled_ad(ad)
    assert "**Caption:** Couldn't do what we do without @Shopify" in rendered
    assert "**CTA:** Learn More" in rendered


def test_render_skips_empty_slots() -> None:
    ad = _adflex_ad(
        platform="facebook",
        headline="Only primary text here.",
        body="",
        description="",
        cta="",
    )
    rendered = render_labeled_ad(ad)
    assert rendered == "**Primary text:** Only primary text here."


def test_render_other_falls_back_to_unlabeled_blob() -> None:
    ad = _adflex_ad(
        platform="other",
        headline="Headline copy",
        body="Body copy",
    )
    rendered = render_labeled_ad(ad)
    # Falls back to ad_copy_text — joined with newlines, no labels.
    assert "**" not in rendered
    assert "Headline copy" in rendered
    assert "Body copy" in rendered


def test_render_strips_value_whitespace_but_preserves_inner_punct() -> None:
    ad = _adflex_ad(
        platform="reddit",
        headline="  Goose tee — get it now!  ",
        cta="  Shop  ",
    )
    rendered = render_labeled_ad(ad)
    assert "**Headline:** Goose tee — get it now!" in rendered
    assert "**CTA:** Shop" in rendered


# ---------------------------------------------------------------------------
# check_platform_labels_present
# ---------------------------------------------------------------------------


def test_label_check_passes_when_all_expected_labels_appear() -> None:
    ad = _adflex_ad(
        platform="facebook",
        headline="Primary copy.",
        body="Hook line.",
        cta="Shop",
    )
    deliverable = (
        "Here's a tight Meta ad — clipped, founder-voice.\n\n"
        "**Primary text:** Primary copy.\n\n"
        "**Headline:** Hook line.\n\n"
        "**CTA:** Shop"
    )
    result = check_platform_labels_present(deliverable, ad)
    assert result.passed
    assert set(result.expected) == {"Primary text", "Headline", "CTA"}
    assert result.missing == ()


def test_label_check_fails_when_label_omitted() -> None:
    ad = _adflex_ad(
        platform="facebook",
        headline="Primary copy.",
        body="Hook line.",
        cta="Shop",
    )
    deliverable = (
        "**Primary text:** Primary copy.\n\n"
        "**Headline:** Hook line.\n\n"
        # CTA label missing
        "Shop"
    )
    result = check_platform_labels_present(deliverable, ad)
    assert not result.passed
    assert result.missing == ("CTA",)
    assert "missing_labels:CTA" in result.reason


def test_label_check_skips_other_platform() -> None:
    ad = _adflex_ad(platform="other", headline="Whatever")
    result = check_platform_labels_present("whatever", ad)
    assert result.passed
    assert result.reason == "other_skip"


def test_label_check_only_requires_populated_slot_labels() -> None:
    # Reddit spec has Headline + CTA. With CTA empty, only Headline is expected.
    ad = _adflex_ad(platform="reddit", headline="Goose tee.")
    deliverable = "**Headline:** Goose tee."
    result = check_platform_labels_present(deliverable, ad)
    assert result.passed
    assert result.expected == ("Headline",)


def test_label_check_pattern_is_case_insensitive() -> None:
    ad = _adflex_ad(platform="reddit", headline="Goose tee.", cta="Shop")
    deliverable = "**HEADLINE:** Goose tee.\n\n**cta:** Shop"
    result = check_platform_labels_present(deliverable, ad)
    assert result.passed


# ---------------------------------------------------------------------------
# platform_group_for — public helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("facebook", PlatformLabelGroup.META),
        ("instagram", PlatformLabelGroup.META),
        ("meta", PlatformLabelGroup.META),
        ("tiktok", PlatformLabelGroup.TIKTOK),
        ("twitter", PlatformLabelGroup.X),
        ("x", PlatformLabelGroup.X),
        ("pinterest", PlatformLabelGroup.PINTEREST),
        ("reddit", PlatformLabelGroup.REDDIT),
        ("google", PlatformLabelGroup.GOOGLE),
        ("other", PlatformLabelGroup.OTHER),
        ("linkedin", PlatformLabelGroup.OTHER),
        ("youtube", PlatformLabelGroup.OTHER),
        ("FACEBOOK", PlatformLabelGroup.META),  # case-insensitive
    ],
)
def test_platform_group_for(platform: str, expected: PlatformLabelGroup) -> None:
    assert platform_group_for(platform) is expected


# ---------------------------------------------------------------------------
# LabelResult dataclass surface
# ---------------------------------------------------------------------------


def test_label_result_is_immutable() -> None:
    result = LabelResult(passed=True, expected=("A",), missing=(), reason="")
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass raises FrozenInstanceError
        result.passed = False  # type: ignore[misc]
