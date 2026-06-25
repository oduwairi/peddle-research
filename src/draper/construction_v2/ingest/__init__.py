"""Validation pipeline for teacher responses.

``response_parser`` splits raw text into ``<think>`` + freeform
deliverable. ``fidelity`` confirms the deliverable matches the source
ad verbatim and that the think trace is grounded in the brief's bridge
fields. ``leak_guard`` enforces the n-gram cap on bridge-field copy so
the brief doesn't echo the ad's exact phrasing. Platform-label
checking (``check_platform_labels_present`` / ``LabelResult``) is
re-exported here from ``platform_labels`` for import convenience.
"""

from draper.construction_v2.ingest.fidelity import (
    FidelityResult,
    GroundingResult,
    check_deliverable_fidelity,
    check_think_grounding,
)
from draper.construction_v2.ingest.leak_guard import (
    DEFAULT_NGRAM_N,
    LeakResult,
    check_bridge_leak,
)
from draper.construction_v2.ingest.response_parser import (
    ParsedResponse,
    ParseRejection,
    parse_response,
)
from draper.construction_v2.platform_labels import (
    LabelResult,
    check_platform_labels_present,
)

__all__ = [
    "DEFAULT_NGRAM_N",
    "FidelityResult",
    "GroundingResult",
    "LabelResult",
    "LeakResult",
    "ParseRejection",
    "ParsedResponse",
    "check_bridge_leak",
    "check_deliverable_fidelity",
    "check_platform_labels_present",
    "check_think_grounding",
    "parse_response",
]
