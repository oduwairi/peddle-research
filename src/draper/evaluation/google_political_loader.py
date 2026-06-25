"""Load Google Political Ads via BigQuery for proxy validation.

The Google Political Ads transparency report (`bigquery-public-data.
google_political_ads.creative_stats`) provides ad-level data with real
impression buckets and spend buckets, plus first/last served dates and
creative text — perfect for validating the longevity sub-score against
ground-truth performance metrics.

Two-phase usage:
  1. collect(): query BigQuery, write enriched RawAds to a JSONL file
     (separate from training data). Bills against the user's BQ project.
  2. load_and_score(): read the JSONL, score through CompositeScorer,
     return aligned scored ads + ground truth.

Auth: requires gcloud Application Default Credentials. Run once:
  gcloud auth application-default login
"""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from draper.scoring.composite_scorer import CompositeScorer
from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.snorkel_scorer import SnorkelScorer
from draper.scoring.tier_assigner import TierAssigner
from draper.scraping.schemas import AdCopy, AdSource, Platform, RawAd
from draper.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger("draper")

# Public BigQuery dataset — no charge for reading public datasets in some
# regions, but always bills against the *querying* project's free tier
# (1 TB/month free). Use LIMIT and select only required columns.
DATASET = "bigquery-public-data.google_political_ads"
CREATIVE_STATS_TABLE = f"`{DATASET}.creative_stats`"


class GooglePoliticalLoader:
    """Load Google Political Ads from BigQuery for validation."""

    def __init__(
        self,
        config: ScoringConfig | None = None,
        project: str | None = None,
        scorer_version: str = "v1",
    ) -> None:
        self.config = config or ScoringConfig.from_yaml()
        self._project = project
        self._scorer_version = scorer_version

    def collect(
        self,
        output_path: str | Path,
        limit: int = 1000,
        min_first_served: str = "2022-01-01",
    ) -> int:
        """Query BigQuery and write Google Political Ads to a JSONL file.

        Returns the number of records written.
        """
        try:
            from google.cloud import bigquery
        except ImportError as e:
            raise ImportError(
                "google-cloud-bigquery not installed. "
                "Run: uv pip install 'google-cloud-bigquery>=3.20'"
            ) from e

        client_kwargs: dict[str, Any] = {}
        if self._project:
            client_kwargs["project"] = self._project
        client = bigquery.Client(**client_kwargs)

        # Select fields we need. Schema may vary slightly across the public
        # dataset versions; we keep the projection conservative.
        # Schema notes (verified against bigquery-public-data.google_political_ads.creative_stats):
        # - `impressions` is a STRING bucket like "10000-15000" — parsed downstream
        # - `spend_range_min_usd`/`spend_range_max_usd` are INTEGERS (USD only)
        # - `regions` is a STRING (no language column in creative_stats)
        # We use TABLESAMPLE to get a longevity-diverse sample. Without sampling,
        # ORDER BY date_range_start DESC returns only fresh 1-day ads.
        query = f"""
        SELECT
            ad_id,
            advertiser_id,
            advertiser_name,
            ad_type,
            regions,
            num_of_days,
            spend_range_min_usd,
            spend_range_max_usd,
            impressions,
            date_range_start,
            date_range_end,
            ad_url
        FROM {CREATIVE_STATS_TABLE}
        WHERE date_range_start >= DATE('{min_first_served}')
            AND impressions IS NOT NULL
            AND impressions != '0-1000'
            AND spend_range_max_usd IS NOT NULL
            AND num_of_days IS NOT NULL
            AND num_of_days > 0
            AND date_range_end < CURRENT_DATE()
        ORDER BY FARM_FINGERPRINT(ad_id)
        LIMIT {limit}
        """

        logger.info("Running BigQuery query (limit=%d)...", limit)
        rows = list(client.query(query).result())
        logger.info("Fetched %d rows from BigQuery", len(rows))

        ads: list[RawAd] = []
        for row in rows:
            try:
                ad = self._row_to_raw_ad(dict(row))
                if ad is not None:
                    ads.append(ad)
            except Exception as e:
                logger.debug("Skipping row: %s", e)
                continue

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        written = write_jsonl(ads, out)
        logger.info("Wrote %d Google Political ads to %s", written, out)
        return written

    def load_and_score(
        self,
        path: str | Path,
    ) -> tuple[list[ScoredAd], pl.DataFrame]:
        """Load Google Political ads from JSONL, score, and align.

        Returns:
            Tuple of (scored_ads, ground_truth_df) where ground_truth_df
            has columns: ad_id, spend_mid, impression_mid, log_spend_mid,
            log_impression_mid, num_days.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Google Political validation file not found: %s", path)
            return [], pl.DataFrame()

        records = read_jsonl(path)
        if not records:
            return [], pl.DataFrame()

        valid_ads: list[RawAd] = []
        for rec in records:
            try:
                # Feed impression midpoint into views so engagement LFs
                # have a signal (Google Political ads have zero social engagement
                # but do have impression range buckets).
                imp_mid = _midpoint(rec.get("impression_lower"), rec.get("impression_upper"))
                if imp_mid is not None:
                    rec["views"] = int(imp_mid)
                ad = RawAd(**rec)
            except Exception as e:
                logger.debug("Skipping invalid record: %s", e)
                continue
            if ad.longevity_days is None or ad.longevity_days <= 0:
                continue
            valid_ads.append(ad)

        if not valid_ads:
            return [], pl.DataFrame()

        logger.info(
            "Loaded %d Google Political ads with valid longevity (of %d records)",
            len(valid_ads),
            len(records),
        )

        scorer: CompositeScorer | SnorkelScorer
        if self._scorer_version == "v3":
            from draper.scoring.hybrid_scorer import HybridScorer

            scorer = HybridScorer(self.config)  # type: ignore[assignment]
        elif self._scorer_version == "v2":
            scorer = SnorkelScorer(self.config)
        else:
            scorer = CompositeScorer(self.config)
        scored = scorer.score_batch(valid_ads)
        assigner = TierAssigner(self.config)
        scored = assigner.assign_tiers(scored)

        gt_rows = []
        for ad in valid_ads:
            spend_mid = _midpoint(ad.spend_lower, ad.spend_upper)
            imp_mid = _midpoint(ad.impression_lower, ad.impression_upper)
            gt_rows.append(
                {
                    "ad_id": ad.ad_id,
                    "spend_mid": spend_mid,
                    "impression_mid": imp_mid,
                    "log_spend_mid": math.log1p(spend_mid) if spend_mid else None,
                    "log_impression_mid": math.log1p(imp_mid) if imp_mid else None,
                    "num_days": ad.longevity_days,
                }
            )

        return scored, pl.DataFrame(gt_rows)

    @staticmethod
    def _row_to_raw_ad(row: dict[str, Any]) -> RawAd | None:
        """Convert a BigQuery row to a RawAd."""
        ad_id = str(row.get("ad_id", "") or "")
        if not ad_id:
            return None

        advertiser = str(row.get("advertiser_name", "") or "")

        spend_min = row.get("spend_range_min_usd")
        spend_max = row.get("spend_range_max_usd")
        imp_min, imp_max = _parse_impression_bucket(row.get("impressions"))

        first = _parse_bq_date(row.get("date_range_start"))
        last = _parse_bq_date(row.get("date_range_end"))
        num_days = row.get("num_of_days")

        return RawAd(
            ad_id=ad_id,
            source=AdSource.GOOGLE_TRANSPARENCY,
            platform=Platform.GOOGLE,
            ad_copy=AdCopy(headline=advertiser, body=str(row.get("ad_type", "") or "")),
            advertiser_id=str(row.get("advertiser_id", "") or ""),
            advertiser_name=advertiser,
            first_seen=first,
            last_seen=last,
            active_days=int(num_days) if num_days is not None else None,
            spend_lower=int(spend_min) if spend_min is not None else None,
            spend_upper=int(spend_max) if spend_max is not None else None,
            impression_lower=int(imp_min) if imp_min is not None else None,
            impression_upper=int(imp_max) if imp_max is not None else None,
            landing_page_url=str(row.get("ad_url", "") or ""),
        )


def _parse_impression_bucket(value: Any) -> tuple[int | None, int | None]:
    """Parse Google's impression bucket string (e.g. '10000-15000') to (min, max).

    Special cases:
    - '<1k' or similar → (0, 1000)
    - '> N' or 'N+' → (N, None)
    """
    if value is None:
        return None, None
    s = str(value).strip().lower().replace(",", "")
    if not s:
        return None, None
    # Standard "min-max" form
    if "-" in s:
        parts = s.split("-", 1)
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    # Handle "≤" or "<" prefix
    if s.startswith(("<", "≤")):
        try:
            return 0, int(s.lstrip("<≤ "))
        except ValueError:
            return None, None
    # Handle "+" or ">" suffix/prefix
    if s.endswith("+"):
        try:
            return int(s.rstrip("+ ")), None
        except ValueError:
            return None, None
    if s.startswith((">", "≥")):
        try:
            return int(s.lstrip(">≥ ")), None
        except ValueError:
            return None, None
    # Single integer
    try:
        n = int(s)
        return n, n
    except ValueError:
        return None, None


def _parse_bq_date(value: Any) -> date | None:
    """Parse a BigQuery date/datetime to a Python date."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _midpoint(lower: int | float | None, upper: int | float | None) -> float | None:
    """Compute range midpoint with fallbacks."""
    if lower is not None and upper is not None:
        return (float(lower) + float(upper)) / 2.0
    if lower is not None:
        return float(lower)
    if upper is not None:
        return float(upper)
    return None
