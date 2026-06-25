"""Construction v2 — backtranslation pipeline (structured brief +
``<think>`` + verbatim ad).

Public surface is re-exported from the responsibility subpackages
(``schemas``, ``teacher``, ``ingest``, ``dataset``). The CLI in
``scripts/construct_v2.py`` is the entry point; ``pipeline`` carries
the shared orchestration helpers it dispatches to.

See ``docs/project/CONSTRUCTION_V2_ARCHITECTURE.md``.
"""

from draper.construction_v2.config import (
    BriefExtractionConfig,
    ConstructionV2Config,
    DatasetConfig,
    FilterConfig,
    RationaleConfig,
    SelectionConfig,
)
from draper.construction_v2.dataset import (
    QualityFilter,
    SourceAd,
    build_dataset,
    load_source_ads_by_id,
    select_source_ads,
)
from draper.construction_v2.ingest import (
    ParsedResponse,
    ParseRejection,
    check_bridge_leak,
    check_deliverable_fidelity,
    check_think_grounding,
    parse_response,
)
from draper.construction_v2.schemas import (
    PLATFORM_ASPECT_RATIO,
    STATIC_SYSTEM_PROMPT,
    SUPPORTED_PLATFORMS,
    AdObjective,
    AspectRatio,
    Brief,
    BriefBridge,
    BriefProduct,
    CreativeDirection,
    ExampleRecord,
    ImageBrief,
    ImageBriefInput,
    RejectionRecord,
    aspect_ratio_for_platform,
    canonical_dict_json,
    canonical_image_brief_input_json,
    canonical_json,
)
from draper.construction_v2.teacher import (
    AD_COPY_TASK,
    BRIEF_EXTRACTION_SYSTEM_PROMPT,
    RATIONALE_TEACHER_SYSTEM,
    build_brief_batch_requests,
    build_rationale_messages,
    build_rationale_request,
    extract_brief,
    parse_brief_response_content,
)

__all__ = [
    "AD_COPY_TASK",
    "BRIEF_EXTRACTION_SYSTEM_PROMPT",
    "AdObjective",
    "AspectRatio",
    "Brief",
    "BriefBridge",
    "BriefExtractionConfig",
    "BriefProduct",
    "CreativeDirection",
    "ConstructionV2Config",
    "DatasetConfig",
    "ExampleRecord",
    "FilterConfig",
    "ImageBrief",
    "ImageBriefInput",
    "PLATFORM_ASPECT_RATIO",
    "ParseRejection",
    "ParsedResponse",
    "QualityFilter",
    "RATIONALE_TEACHER_SYSTEM",
    "RationaleConfig",
    "RejectionRecord",
    "STATIC_SYSTEM_PROMPT",
    "SUPPORTED_PLATFORMS",
    "SelectionConfig",
    "SourceAd",
    "aspect_ratio_for_platform",
    "build_brief_batch_requests",
    "build_dataset",
    "build_rationale_messages",
    "build_rationale_request",
    "canonical_dict_json",
    "canonical_image_brief_input_json",
    "canonical_json",
    "check_bridge_leak",
    "check_deliverable_fidelity",
    "check_think_grounding",
    "extract_brief",
    "load_source_ads_by_id",
    "parse_brief_response_content",
    "parse_response",
    "select_source_ads",
]
