"""Tests for the religious-scripture ad detector.

High-precision goal: faith-themed marketing (nonprofit appeals, church
events, religious retail) must NOT trip the filter. Only ads whose copy
*is* scripture text are caught.
"""

from __future__ import annotations

from draper.construction.religious_scripture import is_religious_scripture_ad
from draper.scoring.schemas import ScoredAd
from draper.scraping.schemas import AdCopy, AdSource, Platform, RawAd


def _ad(
    headline: str = "",
    body: str = "",
    description: str = "",
    cta: str = "",
) -> ScoredAd:
    return ScoredAd(
        ad=RawAd(
            ad_id="t",
            source=AdSource.ADFLEX,
            platform=Platform.FACEBOOK,
            advertiser_name="Advertiser",
            ad_copy=AdCopy(
                headline=headline, body=body, description=description, cta=cta
            ),
        ),
        composite_score=0.5,
        tier="medium",
    )


# ---------------------------------------------------------------------------
# Positive cases — these SHOULD be flagged
# ---------------------------------------------------------------------------


def test_flags_quranic_verse_with_braces_and_diacritics() -> None:
    """The exact pilot-1 offender: Quranic verse in braces, with diacritics."""
    ad = _ad(
        headline=(
            "{ الَّذِينَ أُخْرِجُوا مِن دِيَارِهِم بِغَيْرِ حَقٍّ "
            "إِلَّا أَن يَقُولُوا رَبُّنَا اللَّهُ ... } 🤲🏼\n\n"
            '{ [They are] those who have been evicted from their homes '
            'without right - only because they say, "Our Lord is Allah." '
            '... } 🤲🏼'
        ),
        body="أمل  4️⃣❤️‍🩹 4️⃣ Hope",
    )
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason == "diacritized_arabic"


def test_flags_quranic_verse_diacritics_alone() -> None:
    """Diacritized Arabic alone is enough — no braces required."""
    ad = _ad(headline="رَبُّنَا اللَّهُ الَّذِينَ أُخْرِجُوا")
    flagged, _ = is_religious_scripture_ad(ad)
    assert flagged


def test_flags_bible_verse_citation() -> None:
    ad = _ad(headline="For God so loved the world — John 3:16")
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason in {"bible_verse_citation", "canonical_scripture_phrase"}


def test_flags_bible_verse_citation_with_range() -> None:
    ad = _ad(body="Read Romans 8:28-30 for comfort in hard times.")
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason == "bible_verse_citation"


def test_flags_numbered_book_citation() -> None:
    ad = _ad(headline="1 Corinthians 13:4 — Love is patient, love is kind.")
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason == "bible_verse_citation"


def test_flags_explicit_quran_term() -> None:
    ad = _ad(headline="Learn the Quran online with native Arabic teachers")
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason == "explicit_scripture_term"


def test_flags_explicit_holy_bible() -> None:
    ad = _ad(body="The Holy Bible — new illustrated edition, now 20% off.")
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason == "explicit_scripture_term"


def test_flags_canonical_phrase_lords_prayer() -> None:
    ad = _ad(headline="Our Father who art in Heaven, hallowed be Thy name")
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason == "canonical_scripture_phrase"


def test_flags_beatitudes_phrase() -> None:
    ad = _ad(headline="Blessed are the meek, for they shall inherit the earth")
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason == "canonical_scripture_phrase"


def test_flags_arabic_invocation_transliterated() -> None:
    ad = _ad(headline="Bismillah — your daily reminder", body="Alhamdulillah")
    flagged, reason = is_religious_scripture_ad(ad)
    assert flagged
    assert reason == "canonical_scripture_phrase"


# ---------------------------------------------------------------------------
# Negative cases — these must NOT be flagged (high-precision contract)
# ---------------------------------------------------------------------------


def test_does_not_flag_generic_faith_nonprofit() -> None:
    """Nonprofit mentioning faith as audience affinity, not quoting scripture."""
    ad = _ad(
        headline="Help families in need this Ramadan",
        body=(
            "Our faith-based charity delivers meals to displaced families. "
            "Donate now to support our year-end drive."
        ),
    )
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged


def test_does_not_flag_church_event() -> None:
    ad = _ad(
        headline="Sunday service at 10am — everyone welcome",
        body="Join us at First Community Church for worship and fellowship.",
    )
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged


def test_does_not_flag_religious_retail() -> None:
    ad = _ad(
        headline="Prayer beads & meditation accessories — 30% off",
        body="Handcrafted wooden rosaries, yoga mats, and incense.",
    )
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged


def test_does_not_flag_god_bless_colloquial() -> None:
    ad = _ad(
        headline="God bless our troops — support our veterans' fund",
        body="Your donation helps veterans returning home.",
    )
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged


def test_does_not_flag_casual_arabic_without_diacritics() -> None:
    """Modern-orthography Arabic marketing copy has no diacritics."""
    ad = _ad(
        headline="أفضل عروض رمضان — وفر 30% على جميع المنتجات",
        body="توصيل مجاني اليوم",
    )
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged


def test_does_not_flag_incidental_numbers_not_verses() -> None:
    """Revelation-sounding 'John' + numbers is NOT a verse unless chapter:verse."""
    ad = _ad(
        headline="John's Diner — 3 courses, $16",
        body="Daily specials 5:30-9:00pm",
    )
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged


def test_does_not_flag_empty_ad() -> None:
    ad = _ad()
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged


def test_does_not_flag_normal_marketing() -> None:
    ad = _ad(
        headline="Save 50% on premium streaming data tools",
        body="Decrease costs 6x, increase speeds 10x. Try free for 7 days.",
    )
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged


def test_does_not_flag_three_diacritics_below_threshold() -> None:
    """Stray diacritics (brand names, loan words) shouldn't trip the filter."""
    # Three diacritics is below the threshold of 4.
    ad = _ad(headline="Latté bôld — café culture, reimagined")
    # Those accents aren't Arabic diacritics, so this should pass trivially.
    flagged, _ = is_religious_scripture_ad(ad)
    assert not flagged
