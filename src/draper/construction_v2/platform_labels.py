"""Platform-native ad-copy field labels for v2 construction.

The flat ``AdCopy(headline, body, description, cta)`` scraping schema
collapses every platform's surfaces into the same four slots. The same
slot name carries different semantics across platforms — AdFlex's
``subtitle`` is Meta's headline-below-image but X's CTA-button label;
its ``title`` is Meta's primary text but Reddit's post title.

This module projects a :class:`SourceAd` into a platform-native labeled
rendering using the same vocabulary the frontend's ``emit_campaign``
expects, in Title Case so it reads as natural prose in Draper's voice.

The teacher sees the labeled render in the user message; the student
learns to emit the same labels verbatim in its deliverable. At inference
time the orchestrator maps Title Case → snake_case for the Zod schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from draper.scraping.schemas import AdSource

if TYPE_CHECKING:
    from draper.construction_v2.dataset.source_selector import SourceAd


class PlatformLabelGroup(StrEnum):
    """Coarse platform grouping that drives label selection.

    Facebook and Instagram share the Meta surface set, so they fold into
    a single group. ``OTHER`` is the fallback for platforms we don't yet
    have a native vocabulary for (LinkedIn, YouTube) and for the literal
    ``Platform.OTHER`` ads in the corpus.
    """

    META = "meta"
    TIKTOK = "tiktok"
    X = "x"
    PINTEREST = "pinterest"
    REDDIT = "reddit"
    GOOGLE = "google"
    OTHER = "other"


_PLATFORM_TO_GROUP: Final[dict[str, PlatformLabelGroup]] = {
    "facebook": PlatformLabelGroup.META,
    "instagram": PlatformLabelGroup.META,
    "meta": PlatformLabelGroup.META,
    "tiktok": PlatformLabelGroup.TIKTOK,
    "twitter": PlatformLabelGroup.X,
    "x": PlatformLabelGroup.X,
    "pinterest": PlatformLabelGroup.PINTEREST,
    "reddit": PlatformLabelGroup.REDDIT,
    "google": PlatformLabelGroup.GOOGLE,
}


@dataclass(frozen=True)
class LabelSlot:
    """One platform-native field label and its source-side mapping.

    ``label`` is the Title Case bold-rendered label the model learns to
    emit (e.g. ``"Primary text"``). ``field`` names the
    :class:`SourceAd` attribute whose value fills the slot. When
    ``multiline`` is true the field's value is split on newlines and
    rendered as a markdown bulleted list (used for Google's
    ``Headlines[]`` / ``Descriptions[]`` arrays).
    """

    label: str
    field: str
    multiline: bool = False


# AdFlex covers the entire v3 scored corpus today. Other sources are kept
# as one-line extensions for the day they land in the corpus — adding a
# new (source, group) row plus its slots is the only change needed.
PLATFORM_LABEL_MAP: Final[dict[tuple[AdSource, PlatformLabelGroup], tuple[LabelSlot, ...]]] = {
    # AdFlex on Meta: data["title"] holds the FB primary text and
    # attachments[0].subtitle holds the FB headline-below-image — see
    # src/draper/scraping/adflex.py:454. The flat fields end up swapped
    # relative to Meta's native naming.
    (AdSource.ADFLEX, PlatformLabelGroup.META): (
        LabelSlot("Primary text", "headline"),
        LabelSlot("Headline", "body"),
        LabelSlot("Description", "description"),
        LabelSlot("CTA", "cta"),
    ),
    (AdSource.ADFLEX, PlatformLabelGroup.TIKTOK): (
        LabelSlot("Caption", "headline"),
        LabelSlot("CTA", "cta"),
    ),
    # AdFlex on X: attachments[0].subtitle is the CTA button label, not
    # a card headline — so the flat ``body`` field carries the CTA text
    # and ``headline`` carries the tweet body.
    (AdSource.ADFLEX, PlatformLabelGroup.X): (
        LabelSlot("Tweet", "headline"),
        LabelSlot("Card title", "description"),
        LabelSlot("CTA", "body"),
    ),
    # Pinterest pins have no above-image caption — the main copy lives
    # in attachments[0].description, which AdFlex normalizes to
    # ``description``.
    (AdSource.ADFLEX, PlatformLabelGroup.PINTEREST): (
        LabelSlot("Title", "headline"),
        LabelSlot("Description", "description"),
        LabelSlot("CTA", "cta"),
    ),
    (AdSource.ADFLEX, PlatformLabelGroup.REDDIT): (
        LabelSlot("Headline", "headline"),
        LabelSlot("CTA", "cta"),
    ),
}


def platform_group_for(platform: str) -> PlatformLabelGroup:
    """Map a raw platform string to its label group (OTHER for unknowns)."""
    return _PLATFORM_TO_GROUP.get(platform.lower(), PlatformLabelGroup.OTHER)


def _ad_source(ad: SourceAd) -> AdSource:
    """Recover the :class:`AdSource` from the raw ad dict.

    The v3 scored corpus is 100% AdFlex, so a missing or unparseable
    ``source`` field falls back to AdFlex rather than failing — the
    fallback matches the corpus and lets fixtures stay terse.
    """
    raw_source = ad.raw.get("source") if isinstance(ad.raw, dict) else None
    if isinstance(raw_source, str):
        try:
            return AdSource(raw_source)
        except ValueError:
            pass
    return AdSource.ADFLEX


def _format_slot(slot: LabelSlot, value: str) -> str:
    stripped = value.strip()
    if slot.multiline:
        lines = [ln.strip() for ln in stripped.split("\n") if ln.strip()]
        bullet_block = "\n".join(f"- {ln}" for ln in lines)
        return f"**{slot.label}:**\n{bullet_block}"
    return f"**{slot.label}:** {stripped}"


def render_labeled_ad(ad: SourceAd) -> str:
    """Render the source ad with platform-native field labels.

    Returns the unlabeled :attr:`SourceAd.ad_copy_text` blob when the
    platform is in :class:`PlatformLabelGroup.OTHER` or when no mapping
    is registered for the ``(source, group)`` pair. Null/empty slots are
    omitted from the labeled output rather than rendered as empty values.
    """
    group = platform_group_for(ad.platform)
    if group is PlatformLabelGroup.OTHER:
        return ad.ad_copy_text
    spec = PLATFORM_LABEL_MAP.get((_ad_source(ad), group))
    if spec is None:
        return ad.ad_copy_text

    blocks: list[str] = []
    for slot in spec:
        value = getattr(ad, slot.field, "")
        if not isinstance(value, str) or not value.strip():
            continue
        if slot.multiline:
            lines = [ln.strip() for ln in value.split("\n") if ln.strip()]
            if not lines:
                continue
        blocks.append(_format_slot(slot, value))
    return "\n\n".join(blocks)


@dataclass(frozen=True)
class LabelResult:
    """Outcome of :func:`check_platform_labels_present`.

    ``expected`` lists the labels the source ad supports for its
    ``(source, group)`` pair; ``missing`` lists the subset that did not
    appear in the deliverable. ``passed`` is true when ``missing`` is
    empty, OR when there is no mapping to check against (``OTHER``).
    """

    passed: bool
    expected: tuple[str, ...]
    missing: tuple[str, ...]
    reason: str


_LABEL_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _label_pattern(label: str) -> re.Pattern[str]:
    cached = _LABEL_RE_CACHE.get(label)
    if cached is not None:
        return cached
    pattern = re.compile(rf"\*\*\s*{re.escape(label)}\s*:\s*\*\*", re.IGNORECASE)
    _LABEL_RE_CACHE[label] = pattern
    return pattern


def check_platform_labels_present(deliverable: str, ad: SourceAd) -> LabelResult:
    """Confirm every supported label for ``ad`` appears in the deliverable.

    Returns ``passed=True`` for OTHER-group ads and for ad/source pairs
    without a registered mapping (no label vocabulary to enforce). For
    mapped pairs, only slots whose source field is populated contribute
    to ``expected`` — a slot with an empty source value is not expected
    to appear in the response.
    """
    group = platform_group_for(ad.platform)
    if group is PlatformLabelGroup.OTHER:
        return LabelResult(passed=True, expected=(), missing=(), reason="other_skip")
    spec = PLATFORM_LABEL_MAP.get((_ad_source(ad), group))
    if spec is None:
        return LabelResult(passed=True, expected=(), missing=(), reason="no_mapping_skip")

    expected: list[str] = []
    for slot in spec:
        value = getattr(ad, slot.field, "")
        if isinstance(value, str) and value.strip():
            expected.append(slot.label)

    missing = tuple(label for label in expected if not _label_pattern(label).search(deliverable))
    if not missing:
        return LabelResult(passed=True, expected=tuple(expected), missing=(), reason="")
    return LabelResult(
        passed=False,
        expected=tuple(expected),
        missing=missing,
        reason=f"missing_labels:{','.join(missing)}",
    )


__all__ = [
    "PLATFORM_LABEL_MAP",
    "LabelResult",
    "LabelSlot",
    "PlatformLabelGroup",
    "check_platform_labels_present",
    "platform_group_for",
    "render_labeled_ad",
]
