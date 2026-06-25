"""Pre-compute ad groupings for copywriting training-data construction.

All filters use continuous ``composite_score`` bands (Decision 1); the tier
label is retained in the data for teacher-prompt readability but is not a
filter input. Clustering constants come from ``ClusteringConfig`` in
``configs/construction.yaml``.

Writes ``copywriting_ads.jsonl`` to ``data/constructed/_clusters/``. The
shared advertiser/vertical grouping primitives are retained for tests and
possible future reuse; their manifests are written alongside.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

from draper.construction.religious_scripture import is_religious_scripture_ad
from draper.construction.schemas import (
    ClusterInfo,
    ClusteringConfig,
    ConstructionConfig,
    FormatConfig,
    TaskFormat,
)
from draper.scoring.schemas import ScoredAd
from draper.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger("draper")


def _vertical_key(ad: ScoredAd) -> str:
    """Grouping key for vertical-based clustering.

    Prefers ``business_vertical`` (the LLM-labeled business category, e.g.
    ``saas_software``) since it's the signal marketers reason about. Falls
    back to the sweep-bucket label (e.g. ``facebook:broad``) when the
    business vertical is unset — legacy data or failed labelling.
    """
    return ad.ad.business_vertical or ad.ad.vertical


def _copy_len(ad: ScoredAd) -> int:
    """Total length across headline + body + description + cta (after strip).

    Used for the copywriting filter. Many high-performing ads are headline-
    or CTA-dominant (Google Search, image-heavy FB ads), so summing across
    fields keeps them eligible instead of gating on body alone.
    """
    copy = ad.ad.ad_copy
    return (
        len((copy.headline or "").strip())
        + len((copy.body or "").strip())
        + len((copy.description or "").strip())
        + len((copy.cta or "").strip())
    )


def _score_stats(ads: list[ScoredAd]) -> dict[str, float]:
    """Return {min, max, mean} for a cluster's composite scores."""
    if not ads:
        return {}
    scores = [a.composite_score for a in ads]
    return {
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(sum(scores) / len(scores), 4),
    }


class AdClusterer:
    """Group scored ads into manifests for copywriting construction."""

    def __init__(
        self,
        scored_ads: list[ScoredAd],
        clustering: ClusteringConfig | None = None,
        formats: dict[str, FormatConfig] | None = None,
    ) -> None:
        self._cluster_cfg = clustering or ClusteringConfig()
        self._formats = formats or {}

        if self._cluster_cfg.english_only:
            filtered = [
                a for a in scored_ads if a.ad.language in {"en", ""}
            ]
            logger.info(
                "English-only filter: %d → %d ads (dropped %d non-English)",
                len(scored_ads),
                len(filtered),
                len(scored_ads) - len(filtered),
            )
            scored_ads = filtered

        if self._cluster_cfg.drop_unsafe:
            unsafe_set = set(self._cluster_cfg.unsafe_labels_to_drop)
            conf_floor = self._cluster_cfg.content_safety_min_confidence
            before = len(scored_ads)
            filtered = [
                a
                for a in scored_ads
                if not (
                    a.ad.content_safety_label in unsafe_set
                    and a.ad.content_safety_confidence >= conf_floor
                )
            ]
            logger.info(
                "Content-safety filter (labels=%s, conf>=%.2f): %d → %d ads "
                "(dropped %d)",
                sorted(unsafe_set),
                conf_floor,
                before,
                len(filtered),
                before - len(filtered),
            )
            scored_ads = filtered

        vert_floor = self._cluster_cfg.business_vertical_min_confidence
        if vert_floor > 0:
            before = len(scored_ads)
            filtered = [
                a
                for a in scored_ads
                if a.ad.business_vertical_confidence >= vert_floor
            ]
            logger.info(
                "Business-vertical filter (conf>=%.2f): %d → %d ads (dropped %d)",
                vert_floor,
                before,
                len(filtered),
                before - len(filtered),
            )
            scored_ads = filtered

        q_floor = self._cluster_cfg.training_quality_min
        if q_floor > 0:
            before = len(scored_ads)
            # Keep unlabelled (quality == 0) so the filter is a no-op on
            # legacy rows; drop only labelled ads below the floor.
            filtered = [
                a
                for a in scored_ads
                if a.ad.training_quality == 0 or a.ad.training_quality >= q_floor
            ]
            logger.info(
                "Training-quality filter (quality>=%d): %d → %d ads (dropped %d)",
                q_floor,
                before,
                len(filtered),
                before - len(filtered),
            )
            scored_ads = filtered

        drop_verticals = set(self._cluster_cfg.drop_verticals)
        if drop_verticals:
            before = len(scored_ads)
            filtered = [
                a for a in scored_ads if a.ad.business_vertical not in drop_verticals
            ]
            logger.info(
                "Vertical-drop filter (%s): %d → %d ads (dropped %d)",
                sorted(drop_verticals),
                before,
                len(filtered),
                before - len(filtered),
            )
            scored_ads = filtered

        if self._cluster_cfg.drop_religious_scripture:
            before = len(scored_ads)
            reason_counts: Counter[str] = Counter()
            kept: list[ScoredAd] = []
            for a in scored_ads:
                flagged, reason = is_religious_scripture_ad(a)
                if flagged:
                    reason_counts[reason] += 1
                else:
                    kept.append(a)
            dropped = before - len(kept)
            if dropped:
                reason_summary = ", ".join(
                    f"{r}={c}" for r, c in reason_counts.most_common()
                )
                logger.info(
                    "Religious-scripture filter: %d → %d ads (dropped %d: %s)",
                    before,
                    len(kept),
                    dropped,
                    reason_summary,
                )
            scored_ads = kept

        self._ads = scored_ads
        self._by_id: dict[str, ScoredAd] = {a.ad.ad_id: a for a in scored_ads}

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        clustering: ClusteringConfig | None = None,
        formats: dict[str, FormatConfig] | None = None,
    ) -> AdClusterer:
        """Load scored ads from JSONL and construct a clusterer."""
        records = read_jsonl(path)
        ads = [ScoredAd(**r) for r in records]
        logger.info("Loaded %d scored ads from %s", len(ads), path)
        return cls(ads, clustering=clustering, formats=formats)

    @classmethod
    def from_config(cls, cfg: ConstructionConfig) -> AdClusterer:
        """Convenience: load scored ads using paths from the full config."""
        return cls.from_jsonl(
            cfg.scored_ads_path,
            clustering=cfg.clustering,
            formats=cfg.formats,
        )

    # ------------------------------------------------------------------
    # Grouping primitives (used by multiple selectors)
    # ------------------------------------------------------------------

    def cluster_by_advertiser(self, min_size: int | None = None) -> list[ClusterInfo]:
        """Group ads by advertiser, enforcing a minimum cluster size."""
        size_floor = (
            min_size if min_size is not None else self._cluster_cfg.min_advertiser_cluster
        )
        groups: dict[str, list[ScoredAd]] = defaultdict(list)
        for ad in self._ads:
            name = ad.ad.advertiser_name.strip()
            if name:
                groups[name].append(ad)

        clusters: list[ClusterInfo] = []
        for name, ads in groups.items():
            if len(ads) < size_floor:
                continue
            tier_counts: dict[str, int] = defaultdict(int)
            for a in ads:
                tier_counts[a.tier] += 1
            platform = max(
                {a.ad.platform.value for a in ads},
                key=lambda p: sum(1 for a in ads if a.ad.platform.value == p),
            )
            vertical = max(
                {_vertical_key(a) for a in ads},
                key=lambda v: sum(1 for a in ads if _vertical_key(a) == v),
            )
            clusters.append(
                ClusterInfo(
                    cluster_id=f"adv_{name[:60]}",
                    cluster_type="advertiser",
                    advertiser_name=name,
                    platform=platform,
                    vertical=vertical,
                    ad_ids=[a.ad.ad_id for a in ads],
                    tier_counts=dict(tier_counts),
                    score_stats=_score_stats(ads),
                )
            )
        clusters.sort(key=lambda c: len(c.ad_ids), reverse=True)
        logger.info(
            "Advertiser clusters (min_size=%d): %d clusters, %d ads",
            size_floor,
            len(clusters),
            sum(len(c.ad_ids) for c in clusters),
        )
        return clusters

    def cluster_by_vertical(self, min_size: int | None = None) -> list[ClusterInfo]:
        """Group ads by vertical (e.g. ``facebook:broad``) with a size floor."""
        size_floor = min_size if min_size is not None else self._cluster_cfg.min_vertical_cluster
        groups: dict[str, list[ScoredAd]] = defaultdict(list)
        for ad in self._ads:
            vert = _vertical_key(ad).strip()
            if vert:
                groups[vert].append(ad)

        clusters: list[ClusterInfo] = []
        for vert, ads in groups.items():
            if len(ads) < size_floor:
                continue
            tier_counts: dict[str, int] = defaultdict(int)
            for a in ads:
                tier_counts[a.tier] += 1
            # Business-vertical clusters span platforms; sweep-bucket fallback
            # encodes platform as the "{platform}:{sweep}" prefix.
            if ":" in vert:
                platform = vert.split(":")[0]
            else:
                platform = max(
                    {a.ad.platform.value for a in ads},
                    key=lambda p: sum(1 for a in ads if a.ad.platform.value == p),
                )
            clusters.append(
                ClusterInfo(
                    cluster_id=f"vert_{vert}",
                    cluster_type="vertical",
                    platform=platform,
                    vertical=vert,
                    ad_ids=[a.ad.ad_id for a in ads],
                    tier_counts=dict(tier_counts),
                    score_stats=_score_stats(ads),
                )
            )
        clusters.sort(key=lambda c: len(c.ad_ids), reverse=True)
        logger.info(
            "Vertical clusters (min_size=%d): %d clusters, %d ads",
            size_floor,
            len(clusters),
            sum(len(c.ad_ids) for c in clusters),
        )
        return clusters

    # ------------------------------------------------------------------
    # Format-specific manifest builders
    # ------------------------------------------------------------------

    def get_copywriting_ads(self) -> list[ScoredAd]:
        """Single ads: score >= threshold AND total copy >= min chars (Copywriting).

        Total copy = headline + body + description + cta. Summing across fields
        keeps headline-dominant ads (common on Google Search and FB image) in
        the eligible pool. ``ClusteringConfig.max_per_vertical`` caps each
        vertical at its top-N ads by composite score so head-heavy buckets
        (saas, home_goods) don't dominate training.
        """
        fmt_cfg = self._cluster_cfg.format
        score_min = self._score_min_for(TaskFormat.COPYWRITING)
        min_chars = fmt_cfg.copywriting_min_copy_chars
        eligible = [
            ad
            for ad in self._ads
            if ad.composite_score >= score_min and _copy_len(ad) >= min_chars
        ]

        cap = self._cluster_cfg.max_per_vertical
        if cap > 0:
            by_vertical: dict[str, list[ScoredAd]] = defaultdict(list)
            for ad in eligible:
                by_vertical[_vertical_key(ad)].append(ad)
            capped: list[ScoredAd] = []
            capped_verticals: list[tuple[str, int, int]] = []
            for vert, ads in by_vertical.items():
                ads.sort(key=lambda a: a.composite_score, reverse=True)
                if len(ads) > cap:
                    capped_verticals.append((vert, len(ads), cap))
                capped.extend(ads[:cap])
            if capped_verticals:
                logger.info(
                    "Per-vertical cap (max=%d): %d verticals capped (%s)",
                    cap,
                    len(capped_verticals),
                    ", ".join(f"{v}:{n}→{c}" for v, n, c in capped_verticals),
                )
            result = capped
        else:
            result = eligible

        logger.info(
            "Copywriting ads (score >= %.2f, total copy >= %d, cap=%d): %d",
            score_min,
            min_chars,
            cap,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def lookup(self, ad_id: str) -> ScoredAd | None:
        return self._by_id.get(ad_id)

    def _score_min_for(self, task_format: TaskFormat) -> float:
        """Per-format score floor from the loaded FormatConfig (or 0.0)."""
        fmt = self._formats.get(task_format.value)
        return fmt.score_min if fmt is not None else 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def compute_and_save(self, output_dir: str | Path) -> dict[str, int]:
        """Run copywriting clustering and write manifests to ``output_dir``."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        adv = self.cluster_by_advertiser()
        vert = self.cluster_by_vertical()
        copywriting = self.get_copywriting_ads()

        write_jsonl(adv, out / "advertiser_clusters.jsonl")
        write_jsonl(vert, out / "vertical_clusters.jsonl")
        write_jsonl(
            [{"ad_id": a.ad.ad_id, "score": a.composite_score} for a in copywriting],
            out / "copywriting_ads.jsonl",
        )

        summary = {
            "advertiser_clusters": len(adv),
            "vertical_clusters": len(vert),
            "copywriting_ads": len(copywriting),
        }
        logger.info("Cluster artifacts saved to %s: %s", out, summary)
        return summary
