"""Reference-overlap eval arm — BLEU / chrF / ROUGE-L / METEOR / BERTScore.

The other arms answer pairwise ("is A better than B?"), per-ad ("how good is
this one ad?"), or corpus-level ("does the *distribution* look like real ads?")
questions. This arm answers a fourth: **how close is each generation to a real
winning ad?** For an ad the wording *is* the conversion mechanism, so similarity
to a proven winner is legitimate positive evidence — and it triangulates with
the learned scorer and MAUVE without an LLM judge in the loop.

Each generation is scored against two references:

* ``*_gold`` — the single real winning ad for that brief
  (``Brief.reference_assistant``, rationale-stripped via ``judge/normalize``).
* ``*_multi`` — a small pool of the *k* nearest high-tier real ads on the same
  platform (nearest by MiniLM cosine to the gold). Multi-reference blunts the
  one-to-many critique: a brief has many valid winners, and BLEU/chrF/METEOR
  were designed for multiple references.

Multi-reference aggregation:

* BLEU / chrF / METEOR consume the full reference list natively.
* ROUGE-L and BERTScore take the **max** over references (standard practice).

Score ranges are normalized to ``[0, 1]`` for cross-arm comparability: BLEU and
chrF are divided by 100; ROUGE-L and METEOR are already ``[0, 1]``; BERTScore F1
is reported raw (no baseline rescaling, so English values cluster high).

A diagnostic column ``gold_overlap_excess = rouge_l_gold - rouge_l_multi``
flags GOLD-specific echo: a fine-tune that reproduces *this brief's* winning ad
verbatim scores high here even when its broad-style overlap is ordinary. Read it
alongside the construction-stage n-gram leak guard — it is a reported signal,
never a gate.

All heavy libraries (sacrebleu, nltk, bert-score) are lazy-imported and
soft-fail to ``None`` (mirrors ``judge/similarity.cosine_similarity``):
``None`` means "not measured" (optional dep absent), distinct from ``0.0``
("measured, no overlap"). Install with ``uv pip install -e ".[refmetrics]"``.

Public surface:

* :func:`score_configs` — score one or more configs, one Parquet per config.
* :func:`load_scores` / :func:`summarize` — read + per-config/-platform aggregates.
* :func:`validate_on_upworthy` — grounding arm: does "closer to known winners"
  predict the real Upworthy A/B CTR winner?
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ._inference_io import config_example_ids as _config_example_ids
from ._inference_io import resolve_text as _resolve_text
from .judge.similarity import _embedder, rouge_l_f1
from .mauve_reference import ALL_KEY, _hash_text
from .mauve_scorer import _alias_platform
from .schemas import Brief

logger = logging.getLogger(__name__)

_EMBEDDER_LABEL = "all-MiniLM-L6-v2"

# Metric columns, in the order they appear on a row. Used for parquet schema
# overrides (so an all-``None`` column still types as Float64) and for
# null-column warnings in :func:`summarize`.
_GOLD_METRICS = ["bleu_gold", "chrf_gold", "rouge_l_gold", "meteor_gold", "bertscore_gold"]
_MULTI_METRICS = ["bleu_multi", "chrf_multi", "rouge_l_multi", "meteor_multi", "bertscore_multi"]
_METRIC_COLUMNS = [*_GOLD_METRICS, *_MULTI_METRICS, "gold_overlap_excess"]


@dataclass(frozen=True)
class ReferenceMetricResult:
    """One scored generation: all five metrics vs gold and vs the multi-ref pool."""

    config: str
    example_id: str
    platform: str  # gen-side platform, aliased (meta->facebook, x->twitter)
    used_clean: bool  # False => normalize hasn't run; raw assistant_text scored

    bleu_gold: float | None
    chrf_gold: float | None
    rouge_l_gold: float | None
    meteor_gold: float | None
    bertscore_gold: float | None

    bleu_multi: float | None
    chrf_multi: float | None
    rouge_l_multi: float | None
    meteor_multi: float | None
    bertscore_multi: float | None

    n_multi_refs: int
    gold_overlap_excess: float | None  # rouge_l_gold - rouge_l_multi

    embedding_model: str  # "" when the embedder was unavailable (first-k fallback)
    created_at: str


# --------------------------------------------------------------------------
# Lazy metric backends — each soft-fails to None when its optional dep is
# absent, so the arm degrades gracefully instead of crashing the pipeline.
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _sacrebleu() -> tuple[Any, Any] | None:
    """Return ``(BLEU(effective_order=True), CHRF())`` or None if unavailable."""
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError:
        logger.warning("sacrebleu not installed — bleu/chrf metrics will return None")
        return None
    # effective_order avoids zero sentence-BLEU when higher-order n-grams are absent.
    return BLEU(effective_order=True), CHRF()


@lru_cache(maxsize=1)
def _meteor_fn() -> Any | None:
    """Return nltk's ``meteor_score`` (ensuring the wordnet corpus) or None."""
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        logger.warning("nltk not installed — meteor metric will return None")
        return None
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception as exc:  # pragma: no cover — offline/no-network CI
            logger.warning("Failed to fetch nltk wordnet corpus: %s — meteor disabled", exc)
            return None
    return meteor_score


@lru_cache(maxsize=2)
def _bertscorer(model_type: str) -> Any | None:
    """Return a cached ``BERTScorer`` for ``model_type`` or None if unavailable."""
    try:
        from bert_score import BERTScorer
    except ImportError:
        logger.warning("bert-score not installed — bertscore metric will return None")
        return None
    try:
        return BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=False)
    except Exception as exc:  # pragma: no cover — model download / disk failure
        logger.warning("Failed to load BERTScorer(%s): %s", model_type, exc)
        return None


def _bleu_score(hyp: str, refs: Sequence[str]) -> float | None:
    sb = _sacrebleu()
    if sb is None:
        return None
    valid = [r for r in refs if r and r.strip()]
    if not hyp.strip() or not valid:
        return 0.0
    bleu, _chrf = sb
    try:
        return float(bleu.sentence_score(hyp, list(valid)).score) / 100.0
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("BLEU compute failed: %s", exc)
        return None


def _chrf_score(hyp: str, refs: Sequence[str]) -> float | None:
    sb = _sacrebleu()
    if sb is None:
        return None
    valid = [r for r in refs if r and r.strip()]
    if not hyp.strip() or not valid:
        return 0.0
    _bleu, chrf = sb
    try:
        return float(chrf.sentence_score(hyp, list(valid)).score) / 100.0
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("chrF compute failed: %s", exc)
        return None


def _rouge_l_score(hyp: str, refs: Sequence[str]) -> float:
    """Max ROUGE-L F1 over references (dep-free — never None)."""
    valid = [r for r in refs if r and r.strip()]
    if not hyp.strip() or not valid:
        return 0.0
    return max(rouge_l_f1(r, hyp) for r in valid)


def _meteor_score(hyp: str, refs: Sequence[str]) -> float | None:
    fn = _meteor_fn()
    if fn is None:
        return None
    hyp_toks = hyp.lower().split()
    ref_toks = [r.lower().split() for r in refs if r and r.strip()]
    if not hyp_toks or not ref_toks:
        return 0.0
    try:
        return float(fn(ref_toks, hyp_toks))
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("METEOR compute failed: %s", exc)
        return None


def _bertscore_score(hyp: str, refs: Sequence[str], model_type: str) -> float | None:
    scorer = _bertscorer(model_type)
    if scorer is None:
        return None
    valid = [r for r in refs if r and r.strip()]
    if not hyp.strip() or not valid:
        return 0.0
    try:
        # cands = [hyp]; refs = [[r1, r2, ...]] => max-over-refs F1 by design.
        _p, _r, f1 = scorer.score([hyp], [list(valid)])
        return float(f1[0])
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("BERTScore compute failed: %s", exc)
        return None


# metric-name -> callable(hyp, refs, bertscore_model) used by validate_on_upworthy.
_METRIC_FUNCS = {
    "bleu": lambda h, r, _bm: _bleu_score(h, r),
    "chrf": lambda h, r, _bm: _chrf_score(h, r),
    "rouge_l": lambda h, r, _bm: _rouge_l_score(h, r),
    "meteor": lambda h, r, _bm: _meteor_score(h, r),
    "bertscore": lambda h, r, bm: _bertscore_score(h, r, bm),
}


# --------------------------------------------------------------------------
# Multi-reference selection.
# --------------------------------------------------------------------------


def _precompute_ref_embeddings(
    reference_corpus_by_platform: dict[str, list[str]],
    platforms: Sequence[str],
) -> dict[str, np.ndarray] | None:
    """Encode each platform pool (+ ALL) once for nearest-ref selection.

    Returns None when the embedder is unavailable — callers then fall back to
    first-k selection (order-preserving, still deterministic).
    """
    model = _embedder()
    if model is None:
        logger.warning(
            "Embedder unavailable — multi-ref selection falls back to first-k "
            "(topical-comparability guarantee relaxed)."
        )
        return None
    out: dict[str, np.ndarray] = {}
    for key in [ALL_KEY, *platforms]:
        texts = reference_corpus_by_platform.get(key, [])
        if not texts:
            continue
        try:
            embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Reference embedding failed for %r: %s — first-k fallback.", key, exc)
            return None
        out[key] = np.asarray(embs, dtype=np.float32)
    return out


def select_multi_refs(
    *,
    gold_text: str,
    platform: str,
    reference_corpus_by_platform: dict[str, list[str]],
    k: int = 5,
    ref_embeddings: dict[str, np.ndarray] | None = None,
) -> list[str]:
    """Pick up to ``k`` high-tier same-platform ads nearest the gold ad.

    Pool = the platform partition; falls back to the ``ALL`` partition when the
    platform pool has fewer than ``k`` candidates. The gold ad's own text is
    excluded by hash (avoids a trivial self-match). When ``ref_embeddings`` is
    provided the pool is ranked by MiniLM cosine to the gold (topically
    comparable refs); otherwise the first ``k`` candidates are returned in
    corpus order.
    """
    plat_pool = reference_corpus_by_platform.get(platform, [])
    pool_key = platform if len(plat_pool) >= k else ALL_KEY
    pool = reference_corpus_by_platform.get(pool_key, [])
    if not pool:
        return []

    gold_hash = _hash_text(gold_text) if gold_text and gold_text.strip() else None
    idxs = [i for i, t in enumerate(pool) if not (gold_hash and _hash_text(t) == gold_hash)]
    if not idxs:
        return []

    if ref_embeddings is not None and pool_key in ref_embeddings and gold_text.strip():
        model = _embedder()
        if model is not None:
            try:
                gvec = np.asarray(
                    model.encode([gold_text], normalize_embeddings=True)[0], dtype=np.float32
                )
                sims = ref_embeddings[pool_key][idxs] @ gvec
                order = np.argsort(-sims)
                return [pool[idxs[i]] for i in order[:k]]
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("Nearest-ref ranking failed (%s) — first-k fallback.", exc)

    return [pool[i] for i in idxs[:k]]


# --------------------------------------------------------------------------
# Per-row + per-config scoring.
# --------------------------------------------------------------------------


def compute_row(
    *,
    config: str,
    example_id: str,
    platform: str,
    generation: str,
    gold_text: str,
    multi_refs: Sequence[str],
    used_clean: bool,
    enable_bertscore: bool,
    bertscore_model: str,
    embedding_model_label: str,
) -> ReferenceMetricResult:
    """Score one generation against its gold ad and its multi-reference pool."""
    gold_refs = [gold_text] if gold_text and gold_text.strip() else []

    def _bs(refs: Sequence[str]) -> float | None:
        return _bertscore_score(generation, refs, bertscore_model) if enable_bertscore else None

    rouge_gold = _rouge_l_score(generation, gold_refs)
    rouge_multi = _rouge_l_score(generation, multi_refs)
    excess = rouge_gold - rouge_multi  # both dep-free floats

    return ReferenceMetricResult(
        config=config,
        example_id=example_id,
        platform=platform,
        used_clean=used_clean,
        bleu_gold=_bleu_score(generation, gold_refs),
        chrf_gold=_chrf_score(generation, gold_refs),
        rouge_l_gold=rouge_gold,
        meteor_gold=_meteor_score(generation, gold_refs),
        bertscore_gold=_bs(gold_refs),
        bleu_multi=_bleu_score(generation, multi_refs),
        chrf_multi=_chrf_score(generation, multi_refs),
        rouge_l_multi=rouge_multi,
        meteor_multi=_meteor_score(generation, multi_refs),
        bertscore_multi=_bs(multi_refs),
        n_multi_refs=len(multi_refs),
        gold_overlap_excess=excess,
        embedding_model=embedding_model_label,
        created_at=datetime.now(UTC).isoformat(),
    )


def score_configs(
    *,
    briefs_by_id: dict[str, Brief],
    reference_corpus_by_platform: dict[str, list[str]],
    gold_texts_by_id: dict[str, str],
    configs: Sequence[str],
    inferences_clean_dir: Path,
    inferences_raw_dir: Path,
    out_dir: Path,
    platforms: Sequence[str],
    k_multi: int = 5,
    enable_bertscore: bool = True,
    bertscore_model: str = "roberta-large",
    seed: int = 42,
) -> dict[str, Path]:
    """Score one or more configs, writing one Parquet per config.

    Returns ``{config_name: parquet_path}``. Each Parquet has one row per
    (config, example_id) — per-ad, not corpus-level. The multi-reference pool
    for each brief is built once (it depends only on the gold ad + platform,
    not the config) and reused across all configs.

    ``gold_texts_by_id`` maps example_id -> the cleaned GOLD ad copy; the caller
    builds it (the GOLD config name is split-specific, so it is resolved
    upstream rather than hardcoded here).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if ALL_KEY not in reference_corpus_by_platform:
        raise ValueError(f"reference_corpus_by_platform missing required key {ALL_KEY!r}")

    ref_embeddings = _precompute_ref_embeddings(reference_corpus_by_platform, platforms)
    embedding_model_label = _EMBEDDER_LABEL if ref_embeddings is not None else ""

    # Build the multi-ref pool per brief once (config-independent).
    multi_refs_by_id: dict[str, list[str]] = {}
    for ex_id, brf in briefs_by_id.items():
        multi_refs_by_id[ex_id] = select_multi_refs(
            gold_text=gold_texts_by_id.get(ex_id, ""),
            platform=_alias_platform(brf.platform),
            reference_corpus_by_platform=reference_corpus_by_platform,
            k=k_multi,
            ref_embeddings=ref_embeddings,
        )

    overrides = {c: pl.Float64 for c in _METRIC_COLUMNS}
    written: dict[str, Path] = {}
    for config in configs:
        ids = _config_example_ids(
            config=config,
            inferences_clean_dir=inferences_clean_dir,
            inferences_raw_dir=inferences_raw_dir,
        )
        rows: list[dict[str, object]] = []
        n_dropped = 0
        for ex_id in ids:
            brief = briefs_by_id.get(ex_id)
            if brief is None:
                n_dropped += 1
                continue
            text, used_clean = _resolve_text(
                config=config,
                example_id=ex_id,
                inferences_clean_dir=inferences_clean_dir,
                inferences_raw_dir=inferences_raw_dir,
            )
            if not text.strip():
                n_dropped += 1
                continue
            row = compute_row(
                config=config,
                example_id=ex_id,
                platform=_alias_platform(brief.platform),
                generation=text,
                gold_text=gold_texts_by_id.get(ex_id, ""),
                multi_refs=multi_refs_by_id.get(ex_id, []),
                used_clean=used_clean,
                enable_bertscore=enable_bertscore,
                bertscore_model=bertscore_model,
                embedding_model_label=embedding_model_label,
            )
            rows.append(asdict(row))

        if not rows:
            logger.warning("config=%s: no rows produced (dropped=%d). Skipping.", config, n_dropped)
            continue

        df = pl.DataFrame(rows, schema_overrides=overrides)
        out_path = out_dir / f"{config}.parquet"
        df.write_parquet(out_path)
        written[config] = out_path
        logger.info(
            "config=%s: wrote %d rows to %s (dropped=%d)", config, df.height, out_path, n_dropped
        )

    return written


def load_scores(out_dir: Path, config: str) -> pl.DataFrame:
    """Read the per-config Parquet written by :func:`score_configs`."""
    path = out_dir / f"{config}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No reference-metrics parquet at {path}")
    return pl.read_parquet(path)


def summarize(
    *,
    out_dir: Path,
    configs: Sequence[str],
    by: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Per-config (and optionally per-platform) mean/median of each metric.

    Reads ``{out_dir}/{config}.parquet`` for each named config, concatenates,
    then groups by ``("config", *by)``. A metric column that is entirely null
    (e.g. BERTScore when ``bert-score`` is not installed) is logged so its
    absence is visible rather than silently missing.
    """
    frames: list[pl.DataFrame] = []
    for config in configs:
        path = out_dir / f"{config}.parquet"
        if not path.exists():
            logger.warning("No reference-metrics parquet for config=%s at %s", config, path)
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"No reference-metrics parquets found for configs {list(configs)} in {out_dir}"
        )
    df = pl.concat(frames, how="vertical_relaxed")

    for col in _METRIC_COLUMNS:
        if col in df.columns and df[col].null_count() == df.height:
            logger.warning(
                "Metric column %r is entirely null — likely a missing optional "
                "dependency (bert-score / nltk). Install the [refmetrics] extra.",
                col,
            )

    group_cols = ["config", *list(by or [])]
    aggs: list[pl.Expr] = [pl.len().alias("n")]
    for col in _METRIC_COLUMNS:
        if col in df.columns:
            aggs.append(pl.col(col).mean().alias(f"{col}_mean"))
            aggs.append(pl.col(col).median().alias(f"{col}_median"))
    return df.group_by(group_cols).agg(aggs).sort(group_cols)


# --------------------------------------------------------------------------
# Grounding arm: do the metrics predict real A/B winners?
# --------------------------------------------------------------------------


def _variant_text(variant: object) -> str:
    """Headline + excerpt of an Upworthy variant (matches judge-validation usage)."""
    headline = str(getattr(variant, "headline", "") or "")
    excerpt = str(getattr(variant, "excerpt", "") or "")
    return f"{headline}\n{excerpt}".strip()


def validate_on_upworthy(
    *,
    pairs: list[tuple[Any, Any]],
    metrics: Sequence[str] = ("bleu", "chrf", "rouge_l", "meteor"),
    enable_bertscore: bool = False,
    bertscore_model: str = "roberta-large",
) -> dict[str, Any]:
    """Test whether "more similar to known winners" predicts the A/B CTR winner.

    For each metric, every variant is scored by its similarity to a held-out
    pool of *other tests'* winners (leave-one-pair-out by ``test_id``, so a
    variant is never compared to its own test's winner). The winner should
    outscore the loser more often than chance — the same evidentiary bar the
    LLM-judge arm is held to.

    Returns ``{metric: PairwiseValidationResult}``. BERTScore is opt-in
    (``enable_bertscore``) because leave-one-pair-out against the full winner
    pool is expensive for a neural metric — pair it with a pair ``limit``.
    """
    from .proxy_validation import ProxyValidator

    metric_list = [m for m in metrics if m != "bertscore"]
    if enable_bertscore:
        metric_list.append("bertscore")

    # One text per winning variant, keyed by test_id (a winner may recur across
    # several (winner, loser) pairs when a test has multiple losers).
    winner_by_test: dict[str, str] = {}
    for winner, _loser in pairs:
        tid = str(getattr(winner, "test_id", ""))
        winner_by_test.setdefault(tid, _variant_text(winner))

    results: dict[str, Any] = {}
    for metric in metric_list:
        fn = _METRIC_FUNCS.get(metric)
        if fn is None:
            logger.warning("Unknown reference metric %r — skipping.", metric)
            continue

        def score_fn(variant: object, _fn: Any = fn) -> float:
            own = str(getattr(variant, "test_id", ""))
            pool = [t for tid, t in winner_by_test.items() if tid != own]
            val = _fn(_variant_text(variant), pool, bertscore_model)
            return float(val) if val is not None else 0.0

        results[metric] = ProxyValidator.validate_pairwise_winners(
            pairs=pairs,
            score_fn=score_fn,
            source=f"upworthy:{metric}",
            limitations=[
                "Reference pool = other tests' winners (leave-one-pair-out).",
                "Headlines, not ad copy — a proxy for the creative-similarity hypothesis.",
            ],
        )
    return results
