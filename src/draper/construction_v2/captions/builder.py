"""Build and persist VLM captions of source-ad creatives.

Pipeline:

1. :func:`iter_captionable_ads` reads the v3 scored corpus and yields
   ads whose ``creative_format`` is ``image`` or ``carousel`` and whose
   ``creative_url`` is non-empty. The captioning step is format-aware so
   the v2 selection step downstream can stay format-agnostic — selection
   filters on whether a caption exists, not on the creative format.

2. :func:`build_caption_request` constructs one vision-enabled
   ``BatchRequest`` per ad. The user message is a list-of-blocks
   ``[image_url, text-prompt]`` that all three batch clients translate
   into their native vision wire format (Phase 1 refactor).

3. The CLI submits the batch via the existing
   :mod:`draper.construction.batch` providers. ``task_format`` is
   ``"vlm_caption_v1"`` (this module) rather than ``"<skill>_v2"`` so
   caption batches live alongside teacher batches in the registry
   without colliding.

4. :func:`parse_caption_response` extracts the caption text + token
   counts from a finished ``BatchResponse``.

5. :func:`write_caption_rows` appends or rewrites
   ``data/captions/v1/captions.parquet``. Re-running a slice rewrites
   the rows for that slice's ad_ids (last-write-wins on ad_id).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from draper.construction.batch.types import BatchRequest, BatchResponse
from draper.construction_v2.teacher.image_brief_single_pass import CAPTION_RAW_KEY
from draper.utils.io import read_jsonl

# Identifier used by the BatchRegistry to scope caption batches. Distinct
# from ``<skill>_v2`` so caption batches and teacher batches coexist.
CAPTION_TASK_FORMAT: str = "vlm_caption_v1"

# Default output location for the caption corpus. One file across all
# providers and runs; rows keyed by ad_id.
CAPTIONS_OUTPUT_PATH: Path = Path("data/captions/v1/captions.parquet")

# Captionable formats. Carousel ads have multiple attachments but
# ``creative_url`` is the first frame (per ``scraping/adflex.py``); we
# caption that first frame as a stand-in for the carousel. Video / OTHER
# is deferred to a later phase that adds frame-grab plumbing.
CAPTIONABLE_FORMATS: frozenset[str] = frozenset({"image", "carousel"})

# Image-only URL suffixes. AdFlex's ``creative_format`` field doesn't
# perfectly match what's behind ``creative_url`` — some "carousel" rows
# point at .mp4 first-frames, which the image captioner cannot use.
# Filter on the URL extension as a belt-and-suspenders check.
IMAGE_URL_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})


LITERAL_CAPTION_PROMPT: str = (
    "You are describing an advertising creative image so it can be "
    "used as supervision for a model that generates image briefs.\n"
    "\n"
    "Describe what is in this image, factually and concretely. Cover:\n"
    "- the subject and any people, products, or objects\n"
    "- the setting and background\n"
    "- composition and framing (close-up, wide, overhead, etc.)\n"
    "- lighting and color\n"
    "- any visible text or logos\n"
    "- the photographic or design style (photography, illustration, "
    "3D render, flat graphic, etc.)\n"
    "\n"
    "Be specific. Do not interpret intent or strategy — only describe "
    "what is literally visible. Keep the description to 4-7 sentences."
)


STRATEGIC_CAPTION_PROMPT: str = (
    "You are describing an advertising creative image so it can be "
    "used as supervision for a model that generates image briefs.\n"
    "\n"
    "Describe both what is in this image AND why it works for the ad's "
    "job. Cover:\n"
    "- the hero subject and how it is staged\n"
    "- composition and framing decisions (and what they emphasize)\n"
    "- mood, lighting, and color palette (and the emotional register "
    "they create)\n"
    "- style choices (photography vs illustration vs 3D vs flat "
    "graphic, with references where useful)\n"
    "- how the visual supports a likely angle or buyer "
    "(e.g. 'frames the product as an effortless daily ritual' rather "
    "than just 'person holding bottle')\n"
    "- anything notable about restraint — what was deliberately left OUT\n"
    "\n"
    "Be specific and concrete. Avoid generic marketing language "
    "('eye-catching', 'engaging'). Keep the description to 4-7 sentences."
)


CAPTION_PROMPT_VARIANTS: dict[str, str] = {
    "literal": LITERAL_CAPTION_PROMPT,
    "strategic": STRATEGIC_CAPTION_PROMPT,
}


@dataclass(frozen=True)
class CaptionableAd:
    """Minimal ad shape needed for captioning — no SourceAd dependency."""

    ad_id: str
    creative_url: str
    creative_format: str


@dataclass(frozen=True)
class CaptionRow:
    """One row in the output captions Parquet."""

    ad_id: str
    creative_url: str
    creative_format: str
    caption: str
    captioner_model: str
    caption_prompt_version: str
    captioned_at: str
    provider_error: str


# ---------------------------------------------------------------------------
# Source iteration
# ---------------------------------------------------------------------------


def iter_captionable_ads(
    scored_ads_path: Path | str,
    *,
    exclude_ad_ids: set[str] | None = None,
    include_ad_ids: set[str] | None = None,
) -> Iterator[CaptionableAd]:
    """Yield ads from the v3 scored corpus that have a captionable creative.

    Reads the JSONL form so the nested ``ad`` dict is available without
    re-flattening. Each yielded :class:`CaptionableAd` carries just what
    the captioner needs (ad_id + creative_url + format).

    ``include_ad_ids``: if provided, only emit ads whose ad_id is in
    this set. Used by the production caption-submit path to intersect
    the captionable corpus with the current ``selection.parquet`` —
    captioning is a construction step, not a sourcing step, so only
    selected ads get captioned.
    """
    path = Path(scored_ads_path)
    if not path.exists():
        msg = f"scored ads file not found: {path}"
        raise FileNotFoundError(msg)
    excluded = exclude_ad_ids or set()
    included = include_ad_ids
    for row in read_jsonl(path):
        ad = row.get("ad") if isinstance(row, dict) else None
        if not isinstance(ad, dict):
            continue
        ad_id = ad.get("ad_id")
        if not isinstance(ad_id, str) or ad_id in excluded:
            continue
        if included is not None and ad_id not in included:
            continue
        fmt = ad.get("creative_format", "")
        if not isinstance(fmt, str) or fmt not in CAPTIONABLE_FORMATS:
            continue
        url = ad.get("creative_url", "")
        if not isinstance(url, str) or not url.strip():
            continue
        # AdFlex carousel rows occasionally point at .mp4 first-frames;
        # the image captioner cannot use those.
        # Strip both query params (?) and fragments (#) before checking suffix.
        url_lower = url.lower().split("?", 1)[0].split("#", 1)[0]
        if not any(url_lower.endswith(suf) for suf in IMAGE_URL_SUFFIXES):
            continue
        yield CaptionableAd(ad_id=ad_id, creative_url=url, creative_format=fmt)


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------


def build_caption_request(
    ad: CaptionableAd,
    *,
    model: str,
    prompt_variant: str = "literal",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    mime_type: str = "image/jpeg",
) -> BatchRequest:
    """Construct one vision-enabled batch request for an ad's creative.

    The user-message ``content`` is a list-of-blocks: an ``image_url``
    block (translated by each provider's wire format helper into the
    provider-native vision blob) followed by a ``text`` block carrying
    the chosen captioning prompt. The system field is left ``None`` —
    the entire instruction lives in the user-turn text block to keep
    Gemini's wire format compact.
    """
    prompt = CAPTION_PROMPT_VARIANTS.get(prompt_variant)
    if prompt is None:
        msg = (
            f"unknown prompt_variant {prompt_variant!r}; "
            f"choose from {sorted(CAPTION_PROMPT_VARIANTS.keys())}"
        )
        raise ValueError(msg)
    return BatchRequest(
        custom_id=f"caption-{ad.ad_id}",
        system=None,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "url": ad.creative_url,
                        "mime_type": mime_type,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_caption_response(
    resp: BatchResponse,
    *,
    ad: CaptionableAd,
    captioner_model: str,
    prompt_variant: str,
    captioned_at: str | None = None,
) -> CaptionRow:
    """Turn one provider :class:`BatchResponse` into a :class:`CaptionRow`."""
    ts = captioned_at or datetime.now(UTC).isoformat()
    return CaptionRow(
        ad_id=ad.ad_id,
        creative_url=ad.creative_url,
        creative_format=ad.creative_format,
        caption=resp.content.strip(),
        captioner_model=captioner_model,
        caption_prompt_version=prompt_variant,
        captioned_at=ts,
        provider_error=resp.error,
    )


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def captions_path(output_dir: Path | str | None = None) -> Path:
    """Resolve the captions Parquet path; default is :data:`CAPTIONS_OUTPUT_PATH`."""
    if output_dir is None:
        return CAPTIONS_OUTPUT_PATH
    return Path(output_dir) / "captions.parquet"


def _rows_to_df(rows: list[CaptionRow]) -> pl.DataFrame:
    if not rows:
        # Empty DF with the canonical schema; lets later writes round-trip.
        return pl.DataFrame(
            schema={
                "ad_id": pl.String,
                "creative_url": pl.String,
                "creative_format": pl.String,
                "caption": pl.String,
                "captioner_model": pl.String,
                "caption_prompt_version": pl.String,
                "captioned_at": pl.String,
                "provider_error": pl.String,
            }
        )
    payload: list[dict[str, Any]] = [
        {
            "ad_id": r.ad_id,
            "creative_url": r.creative_url,
            "creative_format": r.creative_format,
            "caption": r.caption,
            "captioner_model": r.captioner_model,
            "caption_prompt_version": r.caption_prompt_version,
            "captioned_at": r.captioned_at,
            "provider_error": r.provider_error,
        }
        for r in rows
    ]
    return pl.DataFrame(payload)


def write_caption_rows(
    rows: list[CaptionRow],
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Append :class:`CaptionRow` rows to the captions Parquet.

    Last-write-wins on ``ad_id``: re-running a slice overwrites any
    existing rows for the same ads, keyed by the latest ``captioned_at``.
    Created on first call. The Parquet directory is created if absent.

    Writes to a temporary file first, then renames atomically to prevent
    data loss on crash-mid-write.
    """
    out = Path(output_path) if output_path else CAPTIONS_OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    new_df = _rows_to_df(rows)
    if out.exists():
        existing = pl.read_parquet(out)
        # Drop existing rows whose ad_id we are about to rewrite.
        if not new_df.is_empty():
            rewriting = set(new_df["ad_id"].to_list())
            existing = existing.filter(~pl.col("ad_id").is_in(rewriting))
        combined = pl.concat([existing, new_df], how="vertical_relaxed")
    else:
        combined = new_df

    # Write to a temp file in the same directory, then atomic rename to
    # prevent losing data if the process dies mid-write.
    temp_out = out.parent / f".{out.name}.tmp-{Path.cwd().stat().st_ino}"
    try:
        combined.write_parquet(temp_out)
        temp_out.replace(out)
    except Exception as exc:
        # Clean up the temp file if write failed (best-effort).
        with suppress(Exception):
            temp_out.unlink(missing_ok=True)
        raise exc from exc
    return out


# ---------------------------------------------------------------------------
# Cost estimation (surfaced by the CLI before any submission)
# ---------------------------------------------------------------------------


# Conservative defaults derived from the Phase-0 smoke (gemini-3.5-flash,
# 50 ads × 2 prompts): ~833 input tokens per caption (image tiles +
# prompt text) and ~117 output tokens per caption. The CLI uses these
# only for the pre-submit estimate; the actual cost comes from the
# provider's usage_metadata in the collected batch.
DEFAULT_INPUT_TOKENS_PER_AD: int = 850
DEFAULT_OUTPUT_TOKENS_PER_AD: int = 120


def estimate_caption_cost_usd(
    *,
    n_ads: int,
    input_price_per_m: float,
    output_price_per_m: float,
    batch_discount: float = 0.5,
    input_tokens_per_ad: int = DEFAULT_INPUT_TOKENS_PER_AD,
    output_tokens_per_ad: int = DEFAULT_OUTPUT_TOKENS_PER_AD,
) -> float:
    """Estimate USD cost of captioning ``n_ads`` at the given prices.

    ``input_price_per_m`` and ``output_price_per_m`` are per-million-token
    sync prices in USD. ``batch_discount`` is applied multiplicatively
    (0.5 = the standard 50% batch tier).
    """
    if n_ads <= 0:
        return 0.0
    sync_per_ad = (
        input_tokens_per_ad * input_price_per_m + output_tokens_per_ad * output_price_per_m
    ) / 1_000_000
    return sync_per_ad * n_ads * batch_discount


# ---------------------------------------------------------------------------
# Caption ⇄ SourceAd join (consumed by image-brief teacher submission)
# ---------------------------------------------------------------------------


def load_captions_lookup(
    captions_parquet: Path | str | None = None,
) -> dict[str, str]:
    """Load ``ad_id → caption`` from the captions Parquet.

    Returns an empty dict if the Parquet doesn't exist yet — callers can
    decide whether that's a hard error or a graceful degradation. Rows
    with non-empty ``provider_error`` are skipped so a failed caption
    can never sneak into the supervision stream.
    """
    path = Path(captions_parquet) if captions_parquet else CAPTIONS_OUTPUT_PATH
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    if df.is_empty():
        return {}
    out: dict[str, str] = {}
    for row in df.iter_rows(named=True):
        if row.get("provider_error"):
            continue
        caption = row.get("caption", "")
        ad_id = row.get("ad_id", "")
        if isinstance(caption, str) and caption.strip() and isinstance(ad_id, str):
            out[ad_id] = caption
    return out


def enrich_source_ads_with_captions(
    ads: list[Any],
    *,
    captions_parquet: Path | str | None = None,
    require_caption: bool = True,
) -> tuple[list[Any], list[str]]:
    """Attach captions to ``ad.raw[CAPTION_RAW_KEY]`` for each ad in ``ads``.

    ``ads`` is typed as ``list[Any]`` because the captions module avoids
    importing :class:`SourceAd` to prevent an import cycle. The shape
    needed is just ``ad.raw`` being a mutable dict and ``ad.ad_id``
    being a string.

    Returns ``(enriched_ads, missing_ad_ids)``. When ``require_caption``
    is True, ads without a caption are dropped from the returned list;
    when False they pass through unchanged. ``missing_ad_ids`` always
    lists every ad_id without a caption regardless of the filter.

    WARNING: this function mutates ``ad.raw[CAPTION_RAW_KEY]`` in place.
    Callers MUST ensure each ad in ``ads`` is uniquely held and not
    referenced elsewhere. The prepare_source_ads hook in skills.py
    ensures this by filtering and returning a new list.
    """
    lookup = load_captions_lookup(captions_parquet)
    out: list[Any] = []
    missing: list[str] = []
    for ad in ads:
        ad_id = getattr(ad, "ad_id", None)
        if not isinstance(ad_id, str):
            continue
        caption = lookup.get(ad_id)
        if caption is None:
            missing.append(ad_id)
            if require_caption:
                continue
            out.append(ad)
            continue
        raw = getattr(ad, "raw", None)
        if isinstance(raw, dict):
            raw[CAPTION_RAW_KEY] = caption
        out.append(ad)
    return out, missing
