"""Source-ad selection for v2 construction.

Reads ``data/scored/v3/scored_ads.jsonl`` (the nested ScoredAd schema),
filters by ``composite_score >= min_composite``, stratifies by
``platform``, and writes the chosen ad_id list as an audit Parquet.

Why JSONL not Parquet: downstream stages need the full nested
``ad.ad_copy`` object (headline, body, description, cta) for fidelity
checks. The Parquet schema in v3 flattens these into separate columns
and loses the nested shape this pipeline relies on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from draper.construction.religious_scripture import is_religious_scripture_text
from draper.construction_v2.config import ConstructionV2Config, SelectionConfig
from draper.construction_v2.platform_labels import (
    PlatformLabelGroup,
    platform_group_for,
    render_labeled_ad,
)


class PlatformConcentration(RuntimeError):
    """Raised when a single platform exceeds ``max_platform_share`` post-select."""


logger = logging.getLogger("draper")

# Structural cleanliness signals (port of v1 selector.py:30-63).
#
# The original v1 filter also caught ``url_in_headline`` and
# ``hashtag_dump`` patterns. Audited on the v3 corpus (2026-05) those
# two were ~95% false positive: t.co URLs are Twitter's auto-appended
# permalinks, FB ads legitimately put shop links in the headline, and
# TikTok ad copy is hashtag-native by platform convention. They were
# removed because they were killing ~2,000 real high-composite ads.
# See scripts/explore/structural_drops_audit.py for the evidence.


@dataclass(frozen=True)
class SourceAd:
    """Minimal source-ad view used by v2 construction.

    Carries the verbatim ad copy fields plus enough metadata to score
    selection and audit later. We keep the raw ``ad`` dict so stage 1 /
    stage 2 prompts can pull whatever extra context they need without a
    second JSONL pass.
    """

    ad_id: str
    platform: str
    composite_score: float
    headline: str
    body: str
    description: str
    cta: str
    raw: dict[str, object]

    @property
    def ad_copy_text(self) -> str:
        """All ad copy fields joined as a single string (for fidelity)."""
        parts = [self.headline, self.body, self.description, self.cta]
        return "\n".join(p for p in parts if p)


def _iter_scored_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def _row_to_source_ad(row: dict[str, object]) -> SourceAd | None:
    """Coerce a scored JSONL row to a :class:`SourceAd`, or ``None`` on miss."""
    ad = row.get("ad")
    if not isinstance(ad, dict):
        return None
    ad_id = ad.get("ad_id")
    if not isinstance(ad_id, str) or not ad_id:
        return None
    ad_copy = ad.get("ad_copy")
    if not isinstance(ad_copy, dict):
        ad_copy = {}
    composite = row.get("composite_score", 0.0)
    if not isinstance(composite, (int, float)):
        composite = 0.0
    platform = ad.get("platform", "other")
    if not isinstance(platform, str):
        platform = "other"
    return SourceAd(
        ad_id=ad_id,
        platform=platform,
        composite_score=float(composite),
        headline=str(ad_copy.get("headline") or ""),
        body=str(ad_copy.get("body") or ""),
        description=str(ad_copy.get("description") or ""),
        cta=str(ad_copy.get("cta") or ""),
        raw=ad,
    )


def _has_structural_artifact(headline: str, body: str, description: str) -> str | None:
    """Return a short tag when the ad has obvious scraper artifacts."""
    h = headline.strip()
    b = body.strip()
    d = description.strip()
    if len(h.split()) > 40 and not b:
        return "wall_of_text_headline"
    if h and b and h.casefold() == b.casefold():
        return "headline_equals_body"
    if h and d and h.casefold() == d.casefold():
        return "headline_equals_description"
    return None


def _stratified_sample(
    ads: list[SourceAd],
    target_count: int,
    seed: int,
    *,
    allow_unbalanced: bool,
    max_platform_share: float,
    force_unbalanced: bool,
) -> list[SourceAd]:
    """Stratify by platform; take up to ``target_count`` total.

    Balanced strategy: per-platform cap = ``target_count / n_platforms``,
    fill round-robin from highest composite. Falls back to top-composite
    sampling when ``allow_unbalanced`` is true and platforms are skewed.

    When ``allow_unbalanced`` is true and the resulting selection has any
    one platform exceeding ``max_platform_share``, raises
    :class:`PlatformConcentration` unless ``force_unbalanced`` is true.
    """
    by_platform: dict[str, list[SourceAd]] = defaultdict(list)
    for ad in ads:
        by_platform[ad.platform].append(ad)

    # Sort each platform's ads by composite descending (deterministic).
    for plat in by_platform:
        by_platform[plat].sort(key=lambda a: (-a.composite_score, a.ad_id))

    if not by_platform:
        return []

    if allow_unbalanced:
        # Single sorted pool, then take top-N. Cheap and deterministic.
        flat = [a for plat in sorted(by_platform) for a in by_platform[plat]]
        flat.sort(key=lambda a: (-a.composite_score, a.ad_id))
        chosen = flat[:target_count]
        _assert_platform_share(
            chosen,
            max_platform_share=max_platform_share,
            force=force_unbalanced,
        )
        return chosen

    # Balanced strategy: cap per platform, fill round-robin.
    n_platforms = len(by_platform)
    cap = max(1, target_count // n_platforms)
    rng = random.Random(seed)
    chosen_balanced: list[SourceAd] = []
    for plat in sorted(by_platform):
        pool = by_platform[plat]
        top = pool[:cap]
        rng.shuffle(top)
        chosen_balanced.extend(top)
    chosen_balanced.sort(key=lambda a: (-a.composite_score, a.ad_id))
    return chosen_balanced[:target_count]


def _assert_platform_share(
    chosen: list[SourceAd],
    *,
    max_platform_share: float,
    force: bool,
) -> None:
    """Raise ``PlatformConcentration`` if one platform dominates the selection."""
    if not chosen:
        return
    counts = Counter(ad.platform for ad in chosen)
    total = len(chosen)
    top_platform, top_count = counts.most_common(1)[0]
    share = top_count / total
    if share <= max_platform_share:
        return
    summary = ", ".join(f"{p}={c}" for p, c in counts.most_common())
    if force:
        logger.warning(
            "Selection concentration: %s = %.1f%% (cap %.1f%%); proceeding "
            "due to force_unbalanced=True. Distribution: %s",
            top_platform,
            share * 100,
            max_platform_share * 100,
            summary,
        )
        return
    msg = (
        f"Selection is dominated by platform {top_platform!r} "
        f"({share:.1%} of {total} ads; cap {max_platform_share:.0%}). "
        f"Distribution: {summary}. "
        f"Either rebalance the source corpus, set "
        f"`selection.allow_unbalanced: false` to stratify, raise "
        f"`selection.max_platform_share`, or pass --force-unbalanced."
    )
    raise PlatformConcentration(msg)


def selection_lineage_hash(sel: SelectionConfig, scored_ads_path: Path) -> str:
    """Compute a stable hash of the selection inputs.

    Used by ``pipeline.verify_selection_lineage`` to detect that the
    operator re-ran ``select`` between ``submit`` and ``collect``,
    which would otherwise orphan in-flight briefs from their source ad
    universe.
    """
    try:
        mtime = scored_ads_path.stat().st_mtime_ns if scored_ads_path.exists() else 0
    except OSError:
        mtime = 0
    payload = json.dumps(
        {
            "scored_ads_path": str(scored_ads_path),
            "scored_ads_mtime_ns": mtime,
            "min_composite": sel.min_composite,
            # target_count is intentionally NOT in the hash — it governs
            # how many ads are taken from the eligible pool, not WHICH
            # ads. `select --target 300` writes a selection.parquet whose
            # lineage should still match a later `submit` that loads the
            # YAML default (e.g. 3000). The slice is determined by what
            # got written to the parquet, not by the config target.
            "seed": sel.seed,
            "stratify": sel.stratify,
            "allow_unbalanced": sel.allow_unbalanced,
            "max_platform_share": sel.max_platform_share,
            "english_only": sel.english_only,
            "drop_unsafe": sel.drop_unsafe,
            "min_training_quality": sel.min_training_quality,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def select_source_ads(
    config: ConstructionV2Config,
    *,
    force_unbalanced: bool = False,
    exclude_ad_ids: set[str] | None = None,
) -> list[SourceAd]:
    """Pick the source-ad batch per ``config.selection`` and audit it.

    ``force_unbalanced`` waives the :attr:`SelectionConfig.max_platform_share`
    assertion. The CLI plumbs it through from ``--force-unbalanced``.

    ``exclude_ad_ids`` removes the given ad_ids from the eligible pool
    before stratification — used to chain runs (e.g. select 300 fresh
    ads that don't overlap run100's already-trained set).
    """
    sel: SelectionConfig = config.selection
    path = Path(sel.scored_ads_path)
    if not path.exists():
        msg = f"Scored ads file not found: {path}"
        raise FileNotFoundError(msg)

    eligible: list[SourceAd] = []
    dropped: Counter[str] = Counter()
    total = 0
    unsafe_set = set(sel.unsafe_labels_to_drop)
    drop_verticals = set(sel.drop_verticals)
    for row in _iter_scored_jsonl(path):
        total += 1
        ad = _row_to_source_ad(row)
        if ad is None:
            dropped["malformed_row"] += 1
            continue
        raw_ad = row.get("ad")
        ad_dict: dict[str, object] = raw_ad if isinstance(raw_ad, dict) else {}

        # Composite floor.
        if ad.composite_score < sel.min_composite:
            dropped["composite_floor"] += 1
            continue

        # Minimum copy length (sum of all copy fields).
        copy_chars = (
            len(ad.headline.strip())
            + len(ad.body.strip())
            + len(ad.description.strip())
            + len(ad.cta.strip())
        )
        if copy_chars < sel.min_copy_chars:
            dropped["min_copy_chars"] += 1
            continue

        # English-only gate. Empty (undetected) passes through.
        if sel.english_only:
            lang = ad_dict.get("language", "")
            if not isinstance(lang, str):
                lang = ""
            if lang and lang != "en":
                dropped["non_english"] += 1
                continue

        # Content-safety gate. Drop only when labelled unsafe AND confidence
        # ≥ floor; unlabelled (empty) and "safe" always pass.
        if sel.drop_unsafe:
            cs_label = ad_dict.get("content_safety_label", "")
            cs_conf = ad_dict.get("content_safety_confidence", 0.0)
            if not isinstance(cs_label, str):
                cs_label = ""
            if not isinstance(cs_conf, (int, float)):
                cs_conf = 0.0
            if cs_label in unsafe_set and float(cs_conf) >= sel.content_safety_min_confidence:
                dropped[f"content_safety:{cs_label}"] += 1
                continue

        # Business-vertical confidence floor.
        if sel.business_vertical_min_confidence > 0:
            bv_conf = ad_dict.get("business_vertical_confidence", 0.0)
            if not isinstance(bv_conf, (int, float)):
                bv_conf = 0.0
            if float(bv_conf) < sel.business_vertical_min_confidence:
                dropped["business_vertical_low_confidence"] += 1
                continue

        # Drop-verticals (compliance exclusions).
        if drop_verticals:
            bv = ad_dict.get("business_vertical", "")
            if isinstance(bv, str) and bv in drop_verticals:
                dropped[f"vertical:{bv}"] += 1
                continue

        # Religious-scripture detector.
        if sel.drop_religious_scripture:
            flagged, reason = is_religious_scripture_text(ad.ad_copy_text)
            if flagged:
                dropped[f"scripture:{reason}"] += 1
                continue

        # Structural cleanliness — catch obvious scraper artifacts.
        if sel.drop_structural_artifacts:
            artifact = _has_structural_artifact(ad.headline, ad.body, ad.description)
            if artifact is not None:
                dropped[f"structural:{artifact}"] += 1
                continue

        # Training-quality floor. Unlabelled (0) passes through.
        if sel.min_training_quality > 0:
            tq = ad_dict.get("training_quality", 0)
            if not isinstance(tq, int):
                tq = 0
            if tq != 0 and tq < sel.min_training_quality:
                dropped[f"training_quality:{tq}"] += 1
                continue

        # Platform-native label render: drop ads whose flat copy fields
        # don't populate any slot for the platform's label spec. The
        # min_copy_chars gate sums all four flat fields, but e.g. a
        # Reddit ad only renders ``headline`` + ``cta`` — body-only ads
        # would pass min_copy_chars yet leave the teacher with an empty
        # labeled block.
        if platform_group_for(ad.platform) is not PlatformLabelGroup.OTHER:
            labeled = render_labeled_ad(ad)
            if not labeled.strip():
                dropped["empty_labeled_render"] += 1
                continue

        # Image-capable filter (image-brief skill only). Require a
        # captionable creative — a non-empty creative_url and a format
        # whose creative is a still image (image, or carousel where we
        # use the first frame). Video / OTHER are deferred to a later
        # phase that adds frame-grab; without one of these the
        # downstream caption-creatives step cannot run on the ad.
        #
        # Note: caption *availability* is NOT gated here. Captioning is
        # a construction step that runs AFTER select on the
        # selection.parquet output — gating on captions here would
        # require captions to exist before construction has decided
        # which ads to caption (cart before horse).
        if config.skill == "image_brief":
            creative_format = ad_dict.get("creative_format", "")
            creative_url = ad_dict.get("creative_url", "")
            if not isinstance(creative_format, str):
                creative_format = ""
            if not isinstance(creative_url, str):
                creative_url = ""
            if creative_format not in {"image", "carousel"}:
                dropped[f"image_capable:format={creative_format or 'missing'}"] += 1
                continue
            if not creative_url.strip():
                dropped["image_capable:missing_url"] += 1
                continue

        eligible.append(ad)

    if exclude_ad_ids:
        before = len(eligible)
        eligible = [a for a in eligible if a.ad_id not in exclude_ad_ids]
        dropped["excluded_by_prior_run"] = before - len(eligible)

    chosen = _stratified_sample(
        eligible,
        target_count=sel.target_count,
        seed=sel.seed,
        allow_unbalanced=sel.allow_unbalanced,
        max_platform_share=sel.max_platform_share,
        force_unbalanced=force_unbalanced,
    )
    _write_selection_audit(chosen, config)
    dropped_summary = (
        ", ".join(f"{reason}={count}" for reason, count in dropped.most_common()) or "none"
    )
    logger.info(
        "Source selection: %d scanned, %d eligible, %d chosen (composite ≥ %.2f). Dropped: %s",
        total,
        len(eligible),
        len(chosen),
        sel.min_composite,
        dropped_summary,
    )
    return chosen


def _write_selection_audit(chosen: list[SourceAd], config: ConstructionV2Config) -> None:
    """Write a stratification audit Parquet for downstream review.

    Each row carries a ``selection_lineage_hash`` column so
    :func:`pipeline.verify_selection_lineage` can detect a re-selected
    universe between ``submit`` and ``collect``.
    """
    audit_dir = Path(config.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    lineage = selection_lineage_hash(config.selection, Path(config.selection.scored_ads_path))
    df = pl.DataFrame(
        {
            "ad_id": [a.ad_id for a in chosen],
            "platform": [a.platform for a in chosen],
            "composite_score": [a.composite_score for a in chosen],
            "headline_len": [len(a.headline) for a in chosen],
            "body_len": [len(a.body) for a in chosen],
            "selection_lineage_hash": [lineage] * len(chosen),
        }
    )
    out_path = audit_dir / "selection.parquet"
    df.write_parquet(out_path)
    logger.info("Selection audit written: %s (lineage=%s)", out_path, lineage)


def load_source_ads_by_id(
    config: ConstructionV2Config, ad_ids: Iterable[str]
) -> dict[str, SourceAd]:
    """Materialize the chosen source ads keyed by ``ad_id``.

    Used by stage 2 (rationale) + ingest to look up the verbatim ad copy
    after stage 1 caches a Brief by ad_id.
    """
    ids = set(ad_ids)
    path = Path(config.selection.scored_ads_path)
    out: dict[str, SourceAd] = {}
    for row in _iter_scored_jsonl(path):
        ad = _row_to_source_ad(row)
        if ad is None or ad.ad_id not in ids:
            continue
        out[ad.ad_id] = ad
        if len(out) == len(ids):
            break
    return out


__all__ = [
    "PlatformConcentration",
    "SourceAd",
    "selection_lineage_hash",
    "load_source_ads_by_id",
    "select_source_ads",
]
