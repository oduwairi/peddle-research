"""MAUVE distribution-matching eval arm.

The other eval arms answer pairwise ("is A better than B?") or per-ad
("how good is this one ad?") questions. This arm answers a corpus-level
question — "does the *distribution* of generations look like the
distribution of real high-performing ads?" — using MAUVE (Pillutla et al.,
NeurIPS 2021; JMLR 2023).

Why we want it:

* The pretrained GPT-2-large encoder isn't ours, so there's no "circular
  eval" objection like the learned-scorer arm has.
* No LLM judges → no position/length/self-preference bias.
* It maps directly onto the Humpback / backtranslation training objective:
  a model trained to imitate real ads should produce a distribution
  closer to real ads than a generic instruction-tuned baseline does.

Pipeline per config:

1. Read all that config's generations off ``inferences_clean/<config>/`` (or
   raw ``inferences/`` fallback). For ``GOLD`` the texts come from
   ``Brief.reference_assistant``.
2. Featurize *once* with GPT-2-large (the expensive step).
3. For each platform slice + an overall ``"ALL"`` slice, compute MAUVE
   between the slice's features and the reference pool's features. Bootstrap
   ``bootstrap_n`` times by resampling rows of the cached feature matrix —
   cheap because we're not re-encoding text.
4. Write one Parquet per config: rows are ``(config, platform, mauve,
   ci_low, ci_high, n_gen, n_ref, ...)``.

Public surface:

* :func:`score_configs` — runs the scorer across one or more configs and
  writes one Parquet per config.
* :func:`load_scores` — reads a previously-written per-config Parquet.
* :func:`summarize` — per-config (and optionally per-platform) aggregates.

Spec: ``docs/project/MAUVE_INTEGRATION_PLAN.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ._inference_io import config_example_ids as _config_example_ids
from ._inference_io import resolve_text as _resolve_text
from .mauve_reference import ALL_KEY
from .schemas import Brief

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MauveResult:
    config: str
    platform: str  # "ALL" for the overall slice
    mauve: float
    ci_low: float
    ci_high: float
    n_gen: int
    n_ref: int
    n_dropped: int  # EXTRACTION_FAILED + empty-text generations
    embedding_model: str
    bootstrap_n: int
    seed: int
    created_at: str


# v2 briefs carry platform labels {meta, x, google, ...} (Brief-schema enum,
# matching the production frontend). The v3 reference corpus and
# cfg.mauve.platforms use the scraping-era labels {facebook, twitter, ...}.
# Without aliasing, gen ads with platform="meta" hash into an empty
# reference bucket and the per-platform slice is silently dropped from
# output (visible in v2 MAUVE runs as missing facebook/twitter rows).
# Normalize *only on the gen side* — the reference corpus and the
# cfg.mauve.platforms list stay in their canonical (scraping-era) labels.
_PLATFORM_ALIAS_GEN_TO_REF: dict[str, str] = {
    "meta": "facebook",
    "x": "twitter",
}


def _alias_platform(platform: str) -> str:
    return _PLATFORM_ALIAS_GEN_TO_REF.get(platform, platform)


def _gather_generations(
    *,
    config: str,
    briefs_by_id: dict[str, Brief],
    inferences_clean_dir: Path,
    inferences_raw_dir: Path,
) -> tuple[list[str], list[str], int]:
    """Return (texts, platforms, n_dropped) aligned 1:1 for one config.

    GOLD goes through the same clean-inference pipeline as other configs.
    ``Brief.reference_assistant`` is the raw teacher output (ad copy plus
    rationale prose); the rationale must be stripped first or MAUVE
    compares copy+prose against copy-only and collapses. The rationale-
    stripped version is produced by ``judge/normalize.py`` and cached under
    ``inferences_clean/<config>/``. The cache path is split-specific:
      * v1 split: ``inferences_clean/GOLD/``
      * v2 split (active default): ``inferences_clean/GOLD_v2/``
    Pass the appropriate config name (``"GOLD"`` or ``"GOLD_v2"``) to
    target the correct cache directory.
    """
    texts: list[str] = []
    platforms: list[str] = []
    n_dropped = 0

    ids = _config_example_ids(
        config=config,
        inferences_clean_dir=inferences_clean_dir,
        inferences_raw_dir=inferences_raw_dir,
    )
    for ex_id in ids:
        brief = briefs_by_id.get(ex_id)
        if brief is None:
            n_dropped += 1
            continue
        text, _used_clean = _resolve_text(
            config=config,
            example_id=ex_id,
            inferences_clean_dir=inferences_clean_dir,
            inferences_raw_dir=inferences_raw_dir,
        )
        if not text.strip():
            n_dropped += 1
            continue
        texts.append(text)
        platforms.append(_alias_platform(brief.platform))

    return texts, platforms, n_dropped


def _featurize(
    texts: Sequence[str],
    *,
    model_name: str,
    device_id: int,
    max_text_length: int,
    batch_size: int,
    name: str,
    verbose: bool,
) -> np.ndarray:
    """Encode texts once with the MAUVE library's featurizer.

    Returns a numpy array of shape (len(texts), hidden_dim). Reused across
    every (platform, bootstrap) slice for this config — re-encoding text is
    the bottleneck.
    """
    # Lazy import — pulls torch/transformers, only wanted at score time.
    from mauve.compute_mauve import get_features_from_input

    features: np.ndarray = get_features_from_input(
        features=None,
        tokenized_texts=None,
        texts=list(texts),
        featurize_model_name=model_name,
        max_len=max_text_length,
        device_id=device_id,
        name=name,
        batch_size=batch_size,
        verbose=verbose,
    )
    return features


def _bootstrap_ci(
    *,
    p_features: np.ndarray,
    q_features: np.ndarray,
    bootstrap_n: int,
    rng: np.random.Generator,
    mauve_kwargs: dict[str, Any],
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) over ``bootstrap_n`` resamples.

    Resamples row indices of ``p_features`` with replacement. ``q_features``
    stays fixed (reference is the population we're comparing against).
    """
    from mauve import compute_mauve

    point = compute_mauve(p_features=p_features, q_features=q_features, **mauve_kwargs)
    point_score = float(point.mauve)

    if bootstrap_n <= 0:
        return point_score, float("nan"), float("nan")

    n_p = p_features.shape[0]
    if n_p < 2:
        logger.warning(
            "Bootstrap CI requested but n_p=%d < 2; cannot resample meaningfully. "
            "Returning point estimate with NaN CI.",
            n_p,
        )
        return point_score, float("nan"), float("nan")

    samples: list[float] = []
    for i in range(bootstrap_n):
        idx = rng.integers(0, n_p, size=n_p)
        resampled = p_features[idx]
        res = compute_mauve(
            p_features=resampled,
            q_features=q_features,
            **{**mauve_kwargs, "seed": mauve_kwargs.get("seed", 25) + i},
        )
        samples.append(float(res.mauve))
    arr = np.asarray(samples)
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return point_score, ci_low, ci_high


def score_configs(
    *,
    briefs_by_id: dict[str, Brief],
    reference_corpus_by_platform: dict[str, list[str]],
    configs: Sequence[str],
    inferences_clean_dir: Path,
    inferences_raw_dir: Path,
    out_dir: Path,
    platforms: Sequence[str],
    embedding_model: str = "gpt2-large",
    bootstrap_n: int = 100,
    seed: int = 42,
    device_id: int = -1,
    max_text_length: int = 1024,
    batch_size: int = 8,
    kmeans_num_redo: int = 1,
    kmeans_max_iter: int = 200,
    verbose: bool = False,
) -> dict[str, Path]:
    """Score one or more configs, writing one Parquet per config.

    Returns ``{config_name: parquet_path}``. Each Parquet has one row per
    (config, platform) plus a final ``"ALL"`` row per config — corpus-level
    rather than per-ad.

    ``reference_corpus_by_platform`` must include the ``"ALL"`` key. Per-
    platform keys are required for every platform in ``platforms``.

    The reference corpus is featurized *once* across all configs (it's the
    same in every iteration), so this is much cheaper than running the
    scorer one config at a time on the CLI.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if ALL_KEY not in reference_corpus_by_platform:
        raise ValueError(f"reference_corpus_by_platform missing required key {ALL_KEY!r}")

    # Tuned for thesis-sized corpora — defaults are heavy.
    mauve_kwargs: dict[str, Any] = {
        "num_buckets": "auto",
        "kmeans_explained_var": 0.9,
        "kmeans_num_redo": kmeans_num_redo,
        "kmeans_max_iter": kmeans_max_iter,
        "divergence_curve_discretization_size": 25,
        "mauve_scaling_factor": 5,
        "seed": seed,
        "verbose": verbose,
    }

    # Featurize the reference pool once per platform slice. ``device_id`` and
    # ``max_text_length`` are forwarded only as featurize args (compute_mauve
    # never sees them again because we hand it precomputed features).
    logger.info(
        "Featurizing reference corpus (model=%s, device_id=%d)…",
        embedding_model,
        device_id,
    )
    ref_features: dict[str, np.ndarray] = {}
    for plat in [ALL_KEY, *platforms]:
        texts = reference_corpus_by_platform.get(plat, [])
        if not texts:
            logger.warning("Reference corpus for platform=%r is empty; skipping.", plat)
            continue
        ref_features[plat] = _featurize(
            texts,
            model_name=embedding_model,
            device_id=device_id,
            max_text_length=max_text_length,
            batch_size=batch_size,
            name=f"ref:{plat}",
            verbose=verbose,
        )
        logger.info("Reference features platform=%s shape=%s", plat, ref_features[plat].shape)

    written: dict[str, Path] = {}
    for config in configs:
        gen_texts, gen_platforms, n_dropped = _gather_generations(
            config=config,
            briefs_by_id=briefs_by_id,
            inferences_clean_dir=inferences_clean_dir,
            inferences_raw_dir=inferences_raw_dir,
        )
        if not gen_texts:
            logger.warning(
                "config=%s: no generations to score (dropped=%d). Skipping.",
                config,
                n_dropped,
            )
            continue

        logger.info(
            "config=%s: featurizing %d generations (dropped=%d)…",
            config,
            len(gen_texts),
            n_dropped,
        )
        gen_features = _featurize(
            gen_texts,
            model_name=embedding_model,
            device_id=device_id,
            max_text_length=max_text_length,
            batch_size=batch_size,
            name=f"gen:{config}",
            verbose=verbose,
        )

        rows: list[dict[str, object]] = []
        rng = np.random.default_rng(seed)
        for plat in [ALL_KEY, *platforms]:
            q_features = ref_features.get(plat)
            if q_features is None:
                continue
            if plat == ALL_KEY:
                idx = np.arange(len(gen_texts))
            else:
                idx = np.asarray([i for i, p in enumerate(gen_platforms) if p == plat], dtype=int)
            if idx.size == 0:
                logger.info(
                    "config=%s platform=%s: 0 generations match, skipping slice.",
                    config,
                    plat,
                )
                continue
            p_slice = gen_features[idx]

            try:
                point, lo, hi = _bootstrap_ci(
                    p_features=p_slice,
                    q_features=q_features,
                    bootstrap_n=bootstrap_n,
                    rng=rng,
                    mauve_kwargs=mauve_kwargs,
                )
            except Exception as exc:  # surface library failure per slice
                logger.warning(
                    "config=%s platform=%s: MAUVE compute failed (%s); writing NaN.",
                    config,
                    plat,
                    exc,
                )
                point, lo, hi = float("nan"), float("nan"), float("nan")

            result = MauveResult(
                config=config,
                platform=plat,
                mauve=point,
                ci_low=lo,
                ci_high=hi,
                n_gen=int(p_slice.shape[0]),
                n_ref=int(q_features.shape[0]),
                n_dropped=n_dropped if plat == ALL_KEY else 0,
                embedding_model=embedding_model,
                bootstrap_n=bootstrap_n,
                seed=seed,
                created_at=datetime.now(UTC).isoformat(),
            )
            rows.append(asdict(result))
            logger.info(
                "config=%s platform=%s mauve=%.4f ci=(%.4f, %.4f) n_gen=%d n_ref=%d",
                config,
                plat,
                point,
                lo,
                hi,
                result.n_gen,
                result.n_ref,
            )

        if not rows:
            logger.warning("config=%s: no MAUVE rows produced. Skipping write.", config)
            continue

        df = pl.DataFrame(rows)
        out_path = out_dir / f"{config}.parquet"
        df.write_parquet(out_path)
        written[config] = out_path
        logger.info("config=%s: wrote %d rows to %s", config, df.height, out_path)

    return written


def load_scores(out_dir: Path, config: str) -> pl.DataFrame:
    """Read the per-config Parquet written by :func:`score_configs`."""
    path = out_dir / f"{config}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No MAUVE parquet at {path}")
    return pl.read_parquet(path)


def summarize(
    *,
    out_dir: Path,
    configs: Sequence[str],
    by: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Per-config summary across the MAUVE rows.

    Reads ``{out_dir}/{config}.parquet`` for each named config, concatenates,
    then groups by ``("config", *by)`` (``by`` is typically ``["platform"]``
    or ``None`` for the headline table). Returns mean / median / min / max
    of the ``mauve`` column along with row count.
    """
    frames: list[pl.DataFrame] = []
    for config in configs:
        path = out_dir / f"{config}.parquet"
        if not path.exists():
            logger.warning("No MAUVE parquet for config=%s at %s", config, path)
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No MAUVE parquets found for configs {list(configs)} in {out_dir}")
    df = pl.concat(frames, how="vertical_relaxed")

    # Check for NaN values (from MAUVE compute failures) and warn.
    nan_rows = df.filter(pl.col("mauve").is_nan())
    if not nan_rows.is_empty():
        logger.warning(
            "Found %d rows with NaN MAUVE scores (compute failures). "
            "These will aggregate to NaN in summary statistics. "
            "Affected configs: %s",
            nan_rows.height,
            nan_rows["config"].unique().to_list(),
        )

    group_cols = ["config", *list(by or [])]
    aggs = [
        pl.len().alias("n"),
        pl.col("mauve").mean().alias("mauve_mean"),
        pl.col("mauve").median().alias("mauve_median"),
        pl.col("mauve").min().alias("mauve_min"),
        pl.col("mauve").max().alias("mauve_max"),
        pl.col("ci_low").mean().alias("ci_low_mean"),
        pl.col("ci_high").mean().alias("ci_high_mean"),
        pl.col("n_gen").sum().alias("n_gen_total"),
    ]
    return df.group_by(group_cols).agg(aggs).sort(group_cols)
