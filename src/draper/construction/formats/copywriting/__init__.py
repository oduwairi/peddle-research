"""Copywriting format package (backtranslation pipeline).

Owns:

- ``constructor`` — :class:`CopywritingConstructor`
- ``dice`` — derived per-bundle context (source_ad_shape only, no RNG —
  see ``dice.py`` docstring for why)
- ``selector`` — structurally-clean single-ad selection
- ``ingestion`` — word-coverage + verbatim signature checks
- ``quality_filter`` — schema-leak + ad-centrality guards, min-length floor
- ``rubric`` — (intentionally empty — see ``rubric.py``)
- ``pipeline`` — :class:`CopywritingPipeline` glue + registry hook

Importing this package registers :class:`CopywritingPipeline` with
:mod:`draper.construction.formats.registry`.
"""

from draper.construction.formats.copywriting.constructor import (
    CopywritingConstructor,
)
from draper.construction.formats.copywriting.pipeline import CopywritingPipeline
from draper.construction.formats.registry import register

register(CopywritingPipeline())

__all__ = ["CopywritingConstructor", "CopywritingPipeline"]
