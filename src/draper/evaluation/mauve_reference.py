"""MAUVE reference-corpus builder + held-out filter.

Builds the "what good looks like" pile that the MAUVE eval arm compares
generations against: top-tier (`high`) real ads from the v3 scored corpus,
partitioned by platform plus an overall ``"ALL"`` slice.

Before partitioning, every reference blob is hashed and cross-checked
against the held-out test split's ad copy. Any held-out ad that also
appears in the reference pool is *removed* from the reference — we want
to compare generations against ``reference - test_split``, never against
a pool that includes the held-out answers themselves. The held-out
overlap is typically large (the test split is sampled from v3 high-tier)
so the loader logs the removal count rather than aborting.

A hard abort is reserved for the pathological case where every reference
row would be removed — that means the held-out set is broader than the
reference, and there's nothing left to compare against. Pass
``contamination_strict=True`` to revert to abort-on-any-overlap behavior
for tests / external corpora.

Hashing is over a normalized form (lowercase, whitespace collapsed) so
trivial reformatting can't slip overlap past the filter.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

ALL_KEY = "ALL"
_WS_RE = re.compile(r"\s+")


def _normalize_for_hash(text: str) -> str:
    """Lowercase + whitespace-collapse to make hashing robust to reformatting."""
    return _WS_RE.sub(" ", text.strip().lower())


def _hash_text(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def _ad_blob(row: dict[str, object]) -> str:
    """Concatenate headline + body + description into one blob.

    Matches the shape of cleaned generations (monolithic strings, not split
    into copy fields) so MAUVE compares like-for-like.
    """
    parts = [
        str(row.get("ad_copy_headline") or ""),
        str(row.get("ad_copy_body") or ""),
        str(row.get("ad_copy_description") or ""),
    ]
    return "\n".join(p for p in parts if p.strip())


class ContaminationError(RuntimeError):
    """Raised when held-out test text overlaps the reference unrecoverably.

    Either ``contamination_strict=True`` was set and any overlap exists, or
    the overlap is total (every reference row would be filtered out).
    """


def load_reference_corpus(
    *,
    parquet_path: Path,
    tier: str = "high",
    platforms: Sequence[str] | None = None,
    held_out_texts: Iterable[str] | None = None,
    cache_dir: Path | None = None,
    force_rebuild: bool = False,
    contamination_strict: bool = False,
) -> dict[str, list[str]]:
    """Build (or load from cache) the MAUVE reference corpus.

    Returns ``{"ALL": [...], "<platform>": [...], ...}``. ``"ALL"`` always
    present; per-platform keys present only if ``platforms`` is non-None.

    When ``held_out_texts`` is provided, every reference row whose hash
    matches a held-out hash is removed. The loader logs the count and only
    raises :class:`ContaminationError` in two cases:
    - ``contamination_strict=True`` and any overlap exists.
    - The overlap is total (all reference rows would be filtered).
    """
    if cache_dir is not None and not force_rebuild:
        cached = _load_from_cache(cache_dir, platforms=platforms)
        if cached is not None:
            logger.info("Loaded reference corpus from cache: %s", cache_dir)
            return cached

    if not parquet_path.exists():
        raise FileNotFoundError(f"v3 scored parquet not found: {parquet_path}")

    df = pl.read_parquet(parquet_path)
    if "tier" not in df.columns:
        raise ValueError(f"{parquet_path} missing 'tier' column")
    df_high = df.filter(pl.col("tier") == tier)
    if df_high.is_empty():
        raise ValueError(f"No rows with tier={tier!r} in {parquet_path}")
    logger.info("Loaded %d rows at tier=%s from %s", df_high.height, tier, parquet_path)

    # Build (platform, blob) pairs once.
    blobs: list[tuple[str, str]] = []
    seen_hashes: set[str] = set()
    for row in df_high.iter_rows(named=True):
        blob = _ad_blob(row)
        if not blob:
            continue
        h = _hash_text(blob)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        platform = str(row.get("platform") or "other")
        blobs.append((platform, blob))

    # Held-out filter — remove any reference row whose hash matches a
    # held-out ad copy. Logs the count; aborts only on total overlap or when
    # ``contamination_strict`` is set.
    if held_out_texts is not None:
        held_out_hashes = {_hash_text(t) for t in held_out_texts if t and t.strip()}
        if not held_out_hashes:
            logger.warning(
                "held_out_texts provided but produced no hashes after filtering "
                "empty/whitespace strings. Contamination check skipped."
            )
        else:
            overlap = held_out_hashes & seen_hashes
            if overlap:
                n_overlap = len(overlap)
                if contamination_strict:
                    raise ContaminationError(
                        f"Reference corpus contamination (strict): {n_overlap} "
                        f"held-out text(s) appear in the v3 high-tier pool. "
                        f"First 3 hashes: {sorted(overlap)[:3]}"
                    )
                if n_overlap == len(seen_hashes):
                    raise ContaminationError(
                        f"Reference corpus contamination: all {n_overlap} "
                        f"reference rows match held-out texts — nothing left "
                        f"to compare against."
                    )
                blobs = [(p, b) for p, b in blobs if _hash_text(b) not in overlap]
                seen_hashes -= overlap
                logger.warning(
                    "Removed %d / %d held-out ads from the reference pool "
                    "(%.1f%% overlap). Reference now %d rows.",
                    n_overlap,
                    len(held_out_hashes),
                    100.0 * n_overlap / max(1, len(held_out_hashes)),
                    len(blobs),
                )
            else:
                logger.info(
                    "Held-out filter: 0 overlap among %d held-out hashes.",
                    len(held_out_hashes),
                )

    # Partition.
    out: dict[str, list[str]] = {ALL_KEY: [b for _, b in blobs]}
    if platforms is not None:
        for plat in platforms:
            out[plat] = [b for p, b in blobs if p == plat]
        for plat, items in out.items():
            if plat == ALL_KEY:
                continue
            if not items:
                logger.warning("Reference partition %r is empty.", plat)

    if cache_dir is not None:
        _write_cache(cache_dir, out)
        logger.info("Wrote reference cache to %s", cache_dir)

    return out


def _cache_path(cache_dir: Path, key: str) -> Path:
    safe = key.replace("/", "_")
    return cache_dir / f"{safe}.parquet"


def _load_from_cache(
    cache_dir: Path,
    *,
    platforms: Sequence[str] | None,
) -> dict[str, list[str]] | None:
    """Return cached corpus if every requested key has a file, else None."""
    required = [ALL_KEY] + list(platforms or [])
    paths = {k: _cache_path(cache_dir, k) for k in required}
    if not all(p.exists() for p in paths.values()):
        return None
    out: dict[str, list[str]] = {}
    for key, path in paths.items():
        df = pl.read_parquet(path)
        if "text" not in df.columns:
            return None
        out[key] = df["text"].to_list()
    return out


def _write_cache(cache_dir: Path, partitions: dict[str, list[str]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key, texts in partitions.items():
        df = pl.DataFrame({"text": texts})
        df.write_parquet(_cache_path(cache_dir, key))
