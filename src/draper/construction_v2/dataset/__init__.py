"""Source selection, quality filtering, and final dataset assembly.

``source_selector`` loads the v3-scored corpus and stratifies a
selection by platform. ``quality_filter`` applies post-ingest dedup,
length, and content-safety gates. ``builder`` assembles the final
HuggingFace ``DatasetDict`` (train / val / test, stratified by platform)
that the trainer consumes.
"""

from draper.construction_v2.dataset.builder import build_dataset
from draper.construction_v2.dataset.quality_filter import (
    FilterResult,
    FilterStats,
    QualityFilter,
)
from draper.construction_v2.dataset.source_selector import (
    SourceAd,
    load_source_ads_by_id,
    select_source_ads,
)

__all__ = [
    "FilterResult",
    "FilterStats",
    "QualityFilter",
    "SourceAd",
    "build_dataset",
    "load_source_ads_by_id",
    "select_source_ads",
]
