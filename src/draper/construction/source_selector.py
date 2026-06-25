"""Shared ad-lookup and selection primitives.

Each :class:`FormatPipeline` owns its own ``select_batches`` function in
``formats/<name>/selector.py``. This module holds the shared state the
format selectors lean on — the scored-ad lookup, the shuffle RNG, the
clusters/pairs loaders, and the two-pass dedup helpers — and provides
the public ``select_batches(task_format, ...)`` entry point that
dispatches through the format registry. A two-pass dedup contract
(strict first, relaxed fallback) with fingerprint-level deduplication is
enforced in both passes.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path

from draper.construction.formats.registry import get_pipeline
from draper.construction.schemas import (
    AdPair,
    ClusterInfo,
    ConstructionConfig,
    TaskFormat,
)
from draper.scoring.schemas import ScoredAd
from draper.utils.io import read_jsonl

logger = logging.getLogger("draper")


class SourceSelector:
    """Load pre-computed cluster manifests and yield per-format ad batches."""

    def __init__(
        self,
        config: ConstructionConfig,
        seed: int = 42,
    ) -> None:
        self._config = config
        self._rng = random.Random(seed)
        self._clusters_dir = Path(config.clusters_dir)

        logger.info("Loading scored ads from %s...", config.scored_ads_path)
        records = read_jsonl(config.scored_ads_path)
        self._ads_by_id: dict[str, ScoredAd] = {}
        for rec in records:
            ad = ScoredAd(**rec)
            self._ads_by_id[ad.ad.ad_id] = ad
        logger.info("Loaded %d scored ads into lookup.", len(self._ads_by_id))

    def _resolve_ids(self, ad_ids: list[str]) -> list[ScoredAd]:
        """Resolve ad IDs to ScoredAd objects, skipping missing ones."""
        result: list[ScoredAd] = []
        for ad_id in ad_ids:
            ad = self._ads_by_id.get(ad_id)
            if ad is not None:
                result.append(ad)
        return result

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def select_batches(
        self,
        task_format: TaskFormat,
        consumed_ids: set[str],
        count: int,
        consumed_fingerprints: frozenset[frozenset[str]] | None = None,
    ) -> list[list[ScoredAd]]:
        """Select ``count`` batches of source ads for the given format.

        Dedup contract:
        - No duplicate ad-set within a call (even under relaxed pass).
        - No duplicate ad-set across calls (``consumed_fingerprints``).
        - Strict pass prefers fresh ads; relaxed pass unlocks reuse.
        """
        fingerprints: frozenset[frozenset[str]] = consumed_fingerprints or frozenset()
        pipeline = get_pipeline(task_format)
        batches = pipeline.select_batches(self, consumed_ids, count, fingerprints)
        logger.info(
            "[%s] Selected %d batches (requested %d)",
            task_format.value,
            len(batches),
            count,
        )
        return batches

    # ------------------------------------------------------------------
    # Dedup primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _combo_candidates(
        ids: list[str], *, min_k: int, max_k: int
    ) -> Iterable[tuple[str, ...]]:
        """k-combinations of ``ids`` for k in ``[min_k, max_k]`` (larger first)."""
        for k in range(max_k, min_k - 1, -1):
            if len(ids) >= k:
                yield from combinations(ids, k)

    def _try_emit_combo(
        self,
        candidate_pool: list[str],
        *,
        min_k: int,
        max_k: int,
        emitted_fingerprints: set[frozenset[str]],
        consumed_fingerprints: frozenset[frozenset[str]],
    ) -> tuple[list[str], frozenset[str]] | None:
        """Return the first k-combination whose fingerprint is fresh."""
        for combo in self._combo_candidates(candidate_pool, min_k=min_k, max_k=max_k):
            fp = frozenset(combo)
            if fp in emitted_fingerprints or fp in consumed_fingerprints:
                continue
            return list(combo), fp
        return None

    # ------------------------------------------------------------------
    # Shared primitives (called by per-format selector modules)
    # ------------------------------------------------------------------

    def _load_clusters(self, filename: str) -> list[ClusterInfo]:
        """Load cluster records from a JSONL file."""
        path = self._clusters_dir / filename
        records = read_jsonl(path)
        return [ClusterInfo(**r) for r in records]

    def _load_pairs(self, filename: str) -> list[AdPair]:
        """Load ad pair records from a JSONL file."""
        path = self._clusters_dir / filename
        records = read_jsonl(path)
        return [AdPair(**r) for r in records]

    def _pairs_to_batches(
        self,
        pairs: list[AdPair],
        consumed_ids: set[str],
        count: int,
        consumed_fingerprints: frozenset[frozenset[str]],
        *,
        low_first: bool,
    ) -> list[list[ScoredAd]]:
        """Convert ad pairs to source-ad batches with two-pass dedup."""
        batches: list[list[ScoredAd]] = []
        emitted_ids: set[str] = set()
        emitted_fingerprints: set[frozenset[str]] = set()

        def _try_emit(pair: AdPair, *, strict: bool) -> bool:
            if strict and (
                pair.high_ad_id in consumed_ids or pair.low_ad_id in consumed_ids
            ):
                return False
            if pair.high_ad_id in emitted_ids or pair.low_ad_id in emitted_ids:
                return False
            fp = frozenset({pair.high_ad_id, pair.low_ad_id})
            if fp in emitted_fingerprints or fp in consumed_fingerprints:
                return False
            high = self._ads_by_id.get(pair.high_ad_id)
            low = self._ads_by_id.get(pair.low_ad_id)
            if high is None or low is None:
                return False
            batches.append([low, high] if low_first else [high, low])
            emitted_ids.add(pair.high_ad_id)
            emitted_ids.add(pair.low_ad_id)
            emitted_fingerprints.add(fp)
            return True

        for pair in pairs:
            if len(batches) >= count:
                break
            _try_emit(pair, strict=True)

        if len(batches) < count:
            for pair in pairs:
                if len(batches) >= count:
                    break
                _try_emit(pair, strict=False)

        self._warn_if_short("pairs", batches, count, consumed_ids, emitted_ids)
        return batches[:count]

    def _warn_if_short(
        self,
        format_name: str,
        batches: list[list[ScoredAd]],
        requested: int,
        consumed_ids: set[str],
        emitted: set[str],
    ) -> None:
        """Log a WARNING when the selector exhausted the pool short of request."""
        if len(batches) < requested:
            logger.warning(
                "[%s] Source pool exhausted: returning %d/%d batches "
                "(consumed=%d IDs, emitted=%d in this call). Expand the "
                "cluster pool or reduce the request size.",
                format_name,
                len(batches),
                requested,
                len(consumed_ids),
                len(emitted),
            )
