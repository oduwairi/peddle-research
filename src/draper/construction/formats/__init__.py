"""Per-format construction pipelines.

The active format is ``copywriting`` (``formats/copywriting/``), which owns
the constructor, source selector, rubric, persona-sampling rule, ingestion
check, and format-specific quality filters.

Prior formats (positioning, diagnostic, optimization, channel_format_fit)
were retired in 2026-04 and preserved under ``archive/construction/formats/``.

Shared construction modules (``ingestion``, ``quality_filter``, ``dice``,
``source_selector``) dispatch to the registered pipeline via
:func:`draper.construction.formats.registry.get_pipeline` so a new format
could drop in without touching the shared orchestrators.
"""

from draper.construction.formats.base import FormatPipeline
from draper.construction.formats.registry import get_pipeline

__all__ = ["FormatPipeline", "get_pipeline"]
