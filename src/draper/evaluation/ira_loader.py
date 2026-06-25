"""Load and prepare the IRA dataset for proxy validation.

The IRA (Internet Research Agency) dataset contains 3,425 Russian-sponsored
Facebook ads from 2016 with real spend, impressions, and click data.
Used as a secondary (out-of-domain) robustness check for proxy score validation.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from draper.scoring.composite_scorer import CompositeScorer
from draper.scoring.schemas import ScoredAd, ScoringConfig
from draper.scoring.snorkel_scorer import SnorkelScorer
from draper.scoring.tier_assigner import TierAssigner
from draper.scraping.schemas import AdCopy, AdSource, Platform, RawAd

logger = logging.getLogger("draper")

# IRA CSV columns:
# id,image,title,description,facebook_url,impressions,clicks,created,ended,
# cost,currency,location,residence,match,interest,behavior,politics,
# multicultural_affinity,employer,industry,field_of_study,exclude,
# language,age,placement

# IRA costs are in RUB; approximate 2016 exchange rate
_RUB_TO_USD_2016 = 1 / 65.0


class IRALoader:
    """Load IRA dataset and prepare for validation."""

    def __init__(
        self,
        config: ScoringConfig | None = None,
        scorer_version: str = "v1",
    ) -> None:
        self.config = config or ScoringConfig.from_yaml()
        self._scorer_version = scorer_version

    def load(self, path: str | Path) -> pl.DataFrame:
        """Load IRA CSV into a Polars DataFrame with computed columns."""
        df = pl.read_csv(path, infer_schema_length=5000, null_values=["", "NA"])
        logger.info("Loaded IRA dataset: %d rows", len(df))
        return df

    def load_and_score(self, path: str | Path) -> tuple[list[ScoredAd], pl.DataFrame]:
        """Load IRA ads, convert to RawAd, score, and return aligned data.

        Returns:
            Tuple of (scored_ads, ground_truth_df) where ground_truth_df
            has columns: ad_id, impressions, clicks, cost_usd, log_impressions,
            log_clicks, log_cost.
        """
        df = self.load(path)

        raw_ads: list[RawAd] = []
        gt_rows: list[dict[str, Any]] = []

        for row in df.iter_rows(named=True):
            row_dict: dict[str, Any] = dict(row)
            ad = self._row_to_raw_ad(row_dict)
            if ad is None:
                continue

            impressions = _safe_int(row_dict.get("impressions"))
            clicks = _safe_int(row_dict.get("clicks"))
            cost_raw = _safe_float(row_dict.get("cost"))

            # Skip ads with no performance data
            if impressions is None or impressions <= 0:
                continue

            cost_usd = cost_raw * _RUB_TO_USD_2016 if cost_raw is not None else None

            raw_ads.append(ad)
            gt_rows.append(
                {
                    "ad_id": ad.ad_id,
                    "impressions": impressions,
                    "clicks": clicks or 0,
                    "cost_usd": cost_usd,
                    "log_impressions": math.log1p(impressions),
                    "log_clicks": math.log1p(clicks or 0),
                    "log_cost": math.log1p(cost_usd)
                    if cost_usd is not None and cost_usd > 0
                    else None,
                }
            )

        if not raw_ads:
            logger.warning("No valid IRA ads with performance data")
            return [], pl.DataFrame()

        logger.info(
            "Prepared %d IRA ads with performance data (of %d total)",
            len(raw_ads),
            len(df),
        )

        # Score
        scorer: CompositeScorer | SnorkelScorer
        if self._scorer_version == "v3":
            from draper.scoring.hybrid_scorer import HybridScorer

            scorer = HybridScorer(self.config)  # type: ignore[assignment]
        elif self._scorer_version == "v2":
            scorer = SnorkelScorer(self.config)
        else:
            scorer = CompositeScorer(self.config)
        scored = scorer.score_batch(raw_ads)
        assigner = TierAssigner(self.config)
        scored = assigner.assign_tiers(scored)

        gt_df = pl.DataFrame(gt_rows)
        return scored, gt_df

    @staticmethod
    def _row_to_raw_ad(row: dict[str, Any]) -> RawAd | None:
        """Convert an IRA CSV row to a RawAd."""
        ad_id = str(row.get("id", ""))
        if not ad_id:
            return None

        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")

        # Parse dates
        created: date | None = _parse_ira_date(row.get("created"))
        ended: date | None = _parse_ira_date(row.get("ended"))

        # IRA ads have no social engagement (likes/comments/shares) but
        # do have impression counts — feed impressions into views so the
        # engagement LFs have a signal to work with.
        impressions = _safe_int(row.get("impressions")) or 0

        return RawAd(
            ad_id=ad_id,
            source=AdSource.META_LIBRARY,  # closest match
            platform=Platform.FACEBOOK,
            ad_copy=AdCopy(headline=title, body=description),
            first_seen=created,
            last_seen=ended,
            likes=0,
            comments=0,
            shares=0,
            reactions=0,
            views=impressions,
        )


def _parse_ira_date(value: Any) -> date | None:
    """Parse IRA date formats (ISO 8601 with timezone)."""
    if value is None or value == "":
        return None
    s = str(value)
    # Try ISO 8601 (e.g. "2016-04-06T18:03:22+00:00")
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        pass
    # Try date only
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Safely convert to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> float | None:
    """Safely convert to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
