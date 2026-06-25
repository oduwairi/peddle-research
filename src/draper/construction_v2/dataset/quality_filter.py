"""Post-ingestion quality filter for v2 examples.

Two checks, simpler than v1's because (a) the selector already vetted
the source corpus for safety / vertical / language / quality, and (b)
the ingest stage already ran parse + fidelity + grounding:

1. **Length sanity** — assistant content under ``max_tokens`` (rough
   word count proxy at 4 chars/token) and the deliverable clears a
   minimum char floor (``min_deliverable_chars``).
2. **Dedup** — TF-IDF cosine over the joint
   ``(canonical_brief_json, deliverable)`` signature. Drops near-
   duplicate examples — neighbours in TF-IDF space teach the model
   nothing new.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from draper.construction_v2.config import FilterConfig
from draper.construction_v2.dataset.source_selector import SourceAd
from draper.construction_v2.schemas.brief import canonical_dict_json
from draper.construction_v2.schemas.records import ExampleRecord, RejectionRecord

logger = logging.getLogger("draper")


class FilterStats(BaseModel):
    """Aggregate stats from a :class:`QualityFilter` run."""

    total_input: int = 0
    passed: int = 0
    rejected_length: int = 0
    rejected_duplicate: int = 0


class FilterResult(BaseModel):
    """Outcome of :meth:`QualityFilter.filter_all`."""

    passed: list[ExampleRecord] = Field(default_factory=list)
    rejected: list[RejectionRecord] = Field(default_factory=list)
    stats: FilterStats = Field(default_factory=FilterStats)


def _approx_token_count(text: str) -> int:
    """Rough token estimate (4 chars/token). Good enough for filtering."""
    return len(text) // 4


def _signature_text(example: ExampleRecord) -> str:
    """Joint dedup signature: canonical brief JSON + deliverable text."""
    return f"{canonical_dict_json(example.brief)}\n{example.deliverable}"


class QualityFilter:
    """Apply v2 quality + dedup filters in sequence."""

    def __init__(
        self,
        config: FilterConfig | None = None,
        *,
        ads_by_id: dict[str, SourceAd] | None = None,
    ) -> None:
        self._config = config or FilterConfig()
        # ads_by_id retained as a constructor arg for callers that wired
        # it; no longer used since content-safety moved to source_selector.
        self._ads_by_id = ads_by_id or {}

    def filter_all(self, examples: list[ExampleRecord]) -> FilterResult:
        stats = FilterStats(total_input=len(examples))
        rejected: list[RejectionRecord] = []
        current = list(examples)

        # 1. Length sanity
        passed, fails = self._filter_length(current)
        stats.rejected_length = len(fails)
        rejected.extend(fails)
        current = passed

        # 2. TF-IDF dedup on (canonical_brief_json, ad)
        passed, fails = self._filter_duplicates(current)
        stats.rejected_duplicate = len(fails)
        rejected.extend(fails)
        current = passed

        stats.passed = len(current)
        logger.info(
            "v2 quality filter: %d input → %d passed (length: -%d, dedup: -%d)",
            stats.total_input,
            stats.passed,
            stats.rejected_length,
            stats.rejected_duplicate,
        )
        return FilterResult(passed=current, rejected=rejected, stats=stats)

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def _filter_length(
        self, examples: list[ExampleRecord]
    ) -> tuple[list[ExampleRecord], list[RejectionRecord]]:
        max_tokens = self._config.max_tokens
        min_deliverable_chars = self._config.min_deliverable_chars
        passed: list[ExampleRecord] = []
        rejected: list[RejectionRecord] = []
        for ex in examples:
            if len(ex.deliverable) < min_deliverable_chars:
                rejected.append(
                    RejectionRecord(
                        ad_id=ex.ad_id,
                        stage="filter",
                        reason=(
                            f"deliverable_too_short:{len(ex.deliverable)}<{min_deliverable_chars}"
                        ),
                    )
                )
                continue
            approx = _approx_token_count(ex.think) + _approx_token_count(ex.deliverable)
            if approx > max_tokens:
                rejected.append(
                    RejectionRecord(
                        ad_id=ex.ad_id,
                        stage="filter",
                        reason=f"assistant_too_long:~{approx}>{max_tokens}",
                    )
                )
                continue
            passed.append(ex)
        return passed, rejected

    def _filter_duplicates(
        self, examples: list[ExampleRecord]
    ) -> tuple[list[ExampleRecord], list[RejectionRecord]]:
        if len(examples) < 2:
            return examples, []
        texts = [_signature_text(ex) for ex in examples]
        try:
            matrix = TfidfVectorizer(max_features=20_000, stop_words="english").fit_transform(texts)
        except ValueError:
            # All documents stop-word only — nothing to dedup against.
            return examples, []
        sim: Any = cosine_similarity(matrix)
        threshold = self._config.dedup_similarity_threshold
        keep: list[bool] = [True] * len(examples)
        for a in range(len(examples)):
            if not keep[a]:
                continue
            for b in range(a + 1, len(examples)):
                if keep[b] and sim[a, b] > threshold:
                    keep[b] = False
        passed: list[ExampleRecord] = []
        rejected: list[RejectionRecord] = []
        for ex, kept in zip(examples, keep, strict=True):
            if kept:
                passed.append(ex)
            else:
                rejected.append(
                    RejectionRecord(
                        ad_id=ex.ad_id,
                        stage="filter",
                        reason=(f"duplicate:tfidf>{self._config.dedup_similarity_threshold:.2f}"),
                    )
                )
        return passed, rejected


__all__ = ["FilterResult", "FilterStats", "QualityFilter"]
