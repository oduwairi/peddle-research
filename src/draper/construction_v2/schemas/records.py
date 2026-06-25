"""Shared v2 record types.

The ingest stage writes :class:`ExampleRecord` rows. The quality filter
reads + filters them. The dataset builder reshapes them into HF chat
format. Defined here so all three modules share a single source of
truth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExampleRecord(BaseModel):
    """One v2 training example, post-parse / fidelity / grounding.

    ``deliverable`` is the freeform text the teacher emitted after
    ``</think>``. For the ad-copy slice that's the source ad reproduced
    verbatim; future skill slices will produce other artifact shapes.

    ``brief`` is the skill's brief as a canonical dict
    (``model.model_dump(mode="json")``) — copywriting stores a
    :class:`~draper.construction_v2.schemas.brief.Brief`, image_brief an
    :class:`~draper.construction_v2.schemas.image_brief.ImageBriefInput`.
    Held as a dict (not a typed model) so the dataset builder + quality
    filter render any skill's brief via ``canonical_dict_json`` without
    importing that skill's model.
    """

    model_config = ConfigDict(extra="forbid")

    example_id: str
    ad_id: str
    platform: Literal["meta", "tiktok", "x", "google", "pinterest", "reddit"]
    brief: dict[str, Any]
    think: str
    deliverable: str

    # Diagnostic metadata. Persisted alongside the example so the audit
    # log can join filter rejections back to construction-time signals
    # without re-running the checks.
    fidelity_coverage: float = 0.0
    fidelity_signature_passed: bool = False
    teacher_model: str = ""
    batch_id: str = Field(default="", description="Empty for chat-mode rows.")


class RejectionRecord(BaseModel):
    """One ingestion or filter rejection — written to the audit log."""

    model_config = ConfigDict(extra="forbid")

    ad_id: str
    stage: Literal["parse", "fidelity", "grounding", "leak", "labels", "content_bridge", "filter"]
    reason: str


__all__ = ["ExampleRecord", "RejectionRecord"]
