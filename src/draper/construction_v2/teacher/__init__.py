"""Teacher-facing prompts and request builders.

The production teacher is ``single_pass``: one batch call per ad emits
``<brief>``, ``<think>``, and the verbatim-ad deliverable in a single
response. ``pipeline.submit_single_pass`` and ``pipeline.collect_batch``
drive this path.

``image_brief_single_pass`` mirrors this for the image-brief skill —
same three-region structure, deliverable is ``<image_brief>`` JSON
instead of ad copy.

``brief_extractor`` (Stage 1) and ``rationale_prompt`` (Stage 2) are
legacy two-stage modules retained until Phase 4 removes them.
"""

from draper.construction_v2.teacher.brief_extractor import (
    AD_COPY_TASK,
    BRIEF_EXTRACTION_SYSTEM_PROMPT,
    build_brief_batch_requests,
    extract_brief,
    parse_brief_response_content,
)
from draper.construction_v2.teacher.image_brief_single_pass import (
    CAPTION_RAW_KEY,
    IMAGE_BRIEF_TEACHER_SYSTEM,
    ImageBriefParseResult,
    build_image_brief_request,
    build_image_brief_user_message,
    parse_image_brief_response,
)
from draper.construction_v2.teacher.rationale_prompt import (
    RATIONALE_TEACHER_SYSTEM,
    build_rationale_messages,
    build_rationale_request,
)
from draper.construction_v2.teacher.single_pass import (
    SINGLE_PASS_TEACHER_SYSTEM,
    SinglePassParseResult,
    build_single_pass_request,
    parse_single_pass_response,
)

__all__ = [
    "AD_COPY_TASK",
    "BRIEF_EXTRACTION_SYSTEM_PROMPT",
    "CAPTION_RAW_KEY",
    "IMAGE_BRIEF_TEACHER_SYSTEM",
    "ImageBriefParseResult",
    "RATIONALE_TEACHER_SYSTEM",
    "SINGLE_PASS_TEACHER_SYSTEM",
    "SinglePassParseResult",
    "build_brief_batch_requests",
    "build_image_brief_request",
    "build_image_brief_user_message",
    "build_rationale_messages",
    "build_rationale_request",
    "build_single_pass_request",
    "extract_brief",
    "parse_brief_response_content",
    "parse_image_brief_response",
    "parse_single_pass_response",
]
