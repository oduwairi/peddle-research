"""Pydantic schemas for the v2 construction pipeline.

``brief`` holds the structured Brief that both the teacher and the
trained student see (canonical JSON serialization is locked here).
``records`` holds the post-ingest example + rejection shapes.
"""

from draper.construction_v2.schemas.brief import (
    STATIC_SYSTEM_PROMPT,
    SUPPORTED_PLATFORMS,
    Brief,
    BriefBridge,
    BriefProduct,
    canonical_dict_json,
    canonical_json,
)
from draper.construction_v2.schemas.image_brief import (
    PLATFORM_ASPECT_RATIO,
    AdObjective,
    AspectRatio,
    CreativeDirection,
    ImageBrief,
    ImageBriefInput,
    aspect_ratio_for_platform,
    canonical_image_brief_input_json,
)
from draper.construction_v2.schemas.records import ExampleRecord, RejectionRecord

__all__ = [
    "STATIC_SYSTEM_PROMPT",
    "SUPPORTED_PLATFORMS",
    "AdObjective",
    "AspectRatio",
    "Brief",
    "BriefBridge",
    "BriefProduct",
    "CreativeDirection",
    "ExampleRecord",
    "ImageBrief",
    "ImageBriefInput",
    "PLATFORM_ASPECT_RATIO",
    "RejectionRecord",
    "aspect_ratio_for_platform",
    "canonical_dict_json",
    "canonical_image_brief_input_json",
    "canonical_json",
]
