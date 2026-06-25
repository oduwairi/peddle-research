"""Audit the ``structural`` drop bucket in source-selection.

The structural filters (`_has_structural_artifact`) were designed to
catch v1 scraper junk: URLs in headlines, hashtag dumps, walls of text,
duplicate fields. With ~2,277 ads dropped at this gate on the current
55k corpus (28.95% of post-language pool), the question is whether
they're catching scraper artifacts (correct) or legitimate
platform-native patterns (false positives that bleed real data).

For each structural reason, this script prints:

1. Platform distribution (Google + TikTok + Reddit are the prime
   suspects for legitimate URL-in-headline and hashtag-heavy patterns).
2. Composite-score distribution (high-composite false positives are
   especially costly — those are the ads we most want to train on).
3. 5 random samples with full copy + platform + score so the operator
   can eyeball whether each looks like a scrape error or a real ad.

Run: ``uv run python scripts/explore/structural_drops_audit.py``.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from draper.construction_v2.config import ConstructionV2Config
from draper.construction_v2.dataset.source_selector import (
    _has_structural_artifact,
    _row_to_source_ad,
)

CONFIG_PATH = Path("configs/construction_v2_image.yaml")
SAMPLES_PER_REASON = 5
RNG = random.Random(0)


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    config = ConstructionV2Config.from_yaml(CONFIG_PATH)
    sel = config.selection
    scored_path = Path(sel.scored_ads_path)

    # Only audit ads that PASS the earlier gates (composite, copy length,
    # english, safety, vertical, scripture) so structural drops are
    # measured against the same pool the selector sees.
    unsafe_set = set(sel.unsafe_labels_to_drop)
    drop_verticals = set(sel.drop_verticals)

    # Bucket each ad that hits a structural artifact.
    flagged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    platform_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    composite_buckets_by_reason: dict[str, Counter[str]] = defaultdict(Counter)

    for row in _iter_rows(scored_path):
        ad = _row_to_source_ad(row)
        if ad is None:
            continue
        raw_ad = row.get("ad")
        ad_dict: dict[str, Any] = raw_ad if isinstance(raw_ad, dict) else {}

        # Earlier gates — mirror source_selector.
        if ad.composite_score < sel.min_composite:
            continue
        copy_chars = (
            len(ad.headline.strip())
            + len(ad.body.strip())
            + len(ad.description.strip())
            + len(ad.cta.strip())
        )
        if copy_chars < sel.min_copy_chars:
            continue
        if sel.english_only:
            lang = ad_dict.get("language", "")
            if isinstance(lang, str) and lang and lang != "en":
                continue
        if sel.drop_unsafe:
            cs_label = ad_dict.get("content_safety_label", "")
            cs_conf = ad_dict.get("content_safety_confidence", 0.0)
            if (
                isinstance(cs_label, str)
                and cs_label in unsafe_set
                and isinstance(cs_conf, (int, float))
                and float(cs_conf) >= sel.content_safety_min_confidence
            ):
                continue
        if sel.business_vertical_min_confidence > 0:
            bv_conf = ad_dict.get("business_vertical_confidence", 0.0)
            if (
                isinstance(bv_conf, (int, float))
                and float(bv_conf) < sel.business_vertical_min_confidence
            ):
                continue
        if drop_verticals:
            bv = ad_dict.get("business_vertical", "")
            if isinstance(bv, str) and bv in drop_verticals:
                continue

        # Now check structural.
        if not sel.drop_structural_artifacts:
            continue
        reason = _has_structural_artifact(ad.headline, ad.body, ad.description)
        if reason is None:
            continue

        flagged[reason].append(
            {
                "ad_id": ad.ad_id,
                "platform": ad.platform,
                "composite": round(ad.composite_score, 3),
                "headline": ad.headline,
                "body": ad.body,
                "description": ad.description,
                "cta": ad.cta,
                "creative_format": ad_dict.get("creative_format", ""),
            }
        )
        platform_by_reason[reason][ad.platform] += 1
        c = ad.composite_score
        bucket = (
            "≥0.90"
            if c >= 0.90
            else "0.80–0.89"
            if c >= 0.80
            else "0.70–0.79"
        )
        composite_buckets_by_reason[reason][bucket] += 1

    # ----- Report -----
    print(f"Structural-drop audit on {scored_path}")
    print(f"Config: {CONFIG_PATH} (min_composite={sel.min_composite})")
    print()

    for reason in sorted(flagged.keys(), key=lambda r: -len(flagged[r])):
        ads = flagged[reason]
        print("═" * 84)
        print(f"REASON: {reason}   ({len(ads):,} drops)")
        print("═" * 84)

        print("  By platform:")
        for plat, n in platform_by_reason[reason].most_common():
            pct = 100.0 * n / len(ads)
            print(f"    {plat:20s} {n:>5,}  ({pct:5.1f}%)")
        print()

        print("  By composite-score band:")
        for band in ["≥0.90", "0.80–0.89", "0.70–0.79"]:
            n = composite_buckets_by_reason[reason].get(band, 0)
            pct = 100.0 * n / len(ads) if ads else 0.0
            print(f"    {band:20s} {n:>5,}  ({pct:5.1f}%)")
        print()

        samples = RNG.sample(ads, min(SAMPLES_PER_REASON, len(ads)))
        print(f"  Samples ({len(samples)} of {len(ads)}):")
        for i, s in enumerate(samples, 1):
            print(
                f"\n  [{i}] ad_id={s['ad_id']}  platform={s['platform']}  "
                f"composite={s['composite']}  format={s['creative_format']}"
            )
            for field in ("headline", "body", "description", "cta"):
                value = s[field]
                if value:
                    display = value if len(value) < 200 else value[:197] + "..."
                    print(f"      {field:13s}: {display!r}")
        print()


if __name__ == "__main__":
    main()
