"""VLM captioning of source-ad creatives for the image-brief skill.

The image-brief teacher needs a description of the real winning creative
that ran with each source ad. This subpackage builds that caption
corpus: it submits VLM batch jobs (one row per creative URL) and writes
results to ``data/captions/v1/captions.parquet`` keyed by ``ad_id``.

The captioning pipeline reuses the construction-v2 ``BatchRegistry``
lifecycle and ``BatchClient`` providers, but tags its batches with a
dedicated ``task_format`` so caption batches live alongside teacher
batches in the same registry without colliding.
"""

from __future__ import annotations

from draper.construction_v2.captions.builder import (
    CAPTION_PROMPT_VARIANTS,
    CAPTION_TASK_FORMAT,
    CAPTIONS_OUTPUT_PATH,
    LITERAL_CAPTION_PROMPT,
    STRATEGIC_CAPTION_PROMPT,
    build_caption_request,
    captions_path,
    enrich_source_ads_with_captions,
    estimate_caption_cost_usd,
    iter_captionable_ads,
    load_captions_lookup,
    parse_caption_response,
    write_caption_rows,
)

__all__ = [
    "CAPTION_PROMPT_VARIANTS",
    "CAPTION_TASK_FORMAT",
    "CAPTIONS_OUTPUT_PATH",
    "LITERAL_CAPTION_PROMPT",
    "STRATEGIC_CAPTION_PROMPT",
    "build_caption_request",
    "captions_path",
    "enrich_source_ads_with_captions",
    "estimate_caption_cost_usd",
    "iter_captionable_ads",
    "load_captions_lookup",
    "parse_caption_response",
    "write_caption_rows",
]
