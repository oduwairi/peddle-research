"""Copywriting source selection.

Single structurally-clean high-score ad per bundle. Scraped ads sometimes
cram the entire creative into one field (usually ``headline``) with
embedded URLs or hashtag strings; those produce unparseable training
targets — a persona brief asking for a headline ends up paired with a
40-word wall that has no distinguishable body or CTA. We filter those at
selection time and pick a cleaner one instead.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from draper.scoring.schemas import ScoredAd
from draper.utils.io import read_jsonl

if TYPE_CHECKING:
    from draper.construction.source_selector import SourceSelector

logger = logging.getLogger("draper")

_URL_RE = re.compile(r"https?://|\bwww\.|\.com/|\.org/|\.io/", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+")
_RUNAWAY_HEADLINE_WORDS = 40


def is_structurally_clean(ad: ScoredAd) -> bool:
    """Return False for ads with scrape artifacts that break copywriting briefs.

    Rejects:
      - headlines containing embedded URLs (raw paste of landing-page link)
      - headlines containing 2+ hashtags (the hashtag strip got stuck in
        the headline field)
      - wall-of-text headlines (>40 words) with no distinct body copy,
        where the scraper lumped the entire creative into ``headline``
      - ads where two copy fields contain identical strings (``headline
        == body`` or ``headline == description``). These teach the
        student to rationalize weak source data (review batch Ex 13 —
        headline and body both "Here's What 1-Day Walk-in Shower Should
        Cost You" produced a post-hoc "repetition for clarity" rationale
        that misrepresents a scraping artifact as craft).

    A clean source ad doesn't have to match any specific shape; we just
    filter out the ones whose fields were mangled by the scraper.
    """
    copy = ad.ad.ad_copy
    headline = (copy.headline or "").strip()
    body = (copy.body or "").strip()
    description = (copy.description or "").strip()
    if not headline:
        return True
    if _URL_RE.search(headline):
        return False
    if len(_HASHTAG_RE.findall(headline)) >= 2:
        return False
    if len(headline.split()) > _RUNAWAY_HEADLINE_WORDS and not body:
        return False
    if body and headline.lower() == body.lower():
        return False
    return not (description and headline.lower() == description.lower())


def select_batches(
    selector: SourceSelector,
    consumed_ids: set[str],
    count: int,
    consumed_fingerprints: frozenset[frozenset[str]],
) -> list[list[ScoredAd]]:
    """Single high-score, structurally-clean ad per bundle."""
    records = read_jsonl(selector._clusters_dir / "copywriting_ads.jsonl")
    ad_ids = [r["ad_id"] for r in records]
    selector._rng.shuffle(ad_ids)

    batches: list[list[ScoredAd]] = []
    emitted: set[str] = set()
    rejected_structural = 0
    for ad_id in ad_ids:
        if len(batches) >= count:
            break
        if ad_id in emitted or ad_id in consumed_ids:
            continue
        if frozenset({ad_id}) in consumed_fingerprints:
            continue
        ad = selector._ads_by_id.get(ad_id)
        if ad is None:
            continue
        if not is_structurally_clean(ad):
            rejected_structural += 1
            continue
        batches.append([ad])
        emitted.add(ad_id)

    if rejected_structural:
        logger.info(
            "[copywriting] Skipped %d scrape-artifact ads (URL/hashtag/"
            "runaway headline in headline field) during selection.",
            rejected_structural,
        )
    if len(batches) < count:
        logger.warning(
            "[copywriting] Source pool exhausted: returning %d/%d batches "
            "(consumed=%d IDs, emitted=%d in this call). Expand the "
            "cluster pool or reduce the request size.",
            len(batches),
            count,
            len(consumed_ids),
            len(emitted),
        )
    return batches[:count]
