"""Unit tests for the v2 config loader + per-field validators."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from draper.construction_v2.config import (
    BriefExtractionConfig,
    ConstructionV2Config,
    DatasetConfig,
    FilterConfig,
    RationaleConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Only the production config is canonical; the per-provider smoke YAMLs
# are scheduled for deletion in Phase 2 (replaced by `--provider` +
# `--run-id` flags against the unified config).
CONFIG_PATHS = [
    REPO_ROOT / "configs/construction_v2.yaml",
]


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: p.name)
def test_from_yaml_loads_shipped_configs(config_path: Path) -> None:
    """Every shipped v2 yaml must load without unknown-key errors.

    ``extra="forbid"`` on every section means a stale key in a yaml
    (e.g. the dropped ``require_tone_signals``) would fail loud here.
    """
    cfg = ConstructionV2Config.from_yaml(config_path)
    assert isinstance(cfg, ConstructionV2Config)
    assert cfg.selection.scored_ads_path
    # The unified config binds providers; per-provider models drive
    # single-pass submissions.
    assert cfg.providers, "configs/construction_v2.yaml must populate `providers:`"
    for provider, p in cfg.providers.items():
        assert p.model, f"providers[{provider!r}] missing model"
    # Legacy two-stage fields are optional and may be absent.
    if cfg.brief_extraction is not None:
        assert cfg.brief_extraction.model
    if cfg.rationale is not None:
        assert cfg.rationale.model
    assert 0.0 <= cfg.filter.dedup_similarity_threshold <= 1.0


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: p.name)
def test_yamls_have_no_stale_keys(config_path: Path) -> None:
    """Defence-in-depth: per-section keys must subset the config schema."""
    raw = yaml.safe_load(config_path.read_text())
    data = raw.get("construction_v2", raw)

    sections: dict[str, type] = {
        "selection": ConstructionV2Config.model_fields["selection"].annotation,
        "single_pass": ConstructionV2Config.model_fields["single_pass"].annotation,
        "batch": ConstructionV2Config.model_fields["batch"].annotation,
        "filter": ConstructionV2Config.model_fields["filter"].annotation,
        "dataset": ConstructionV2Config.model_fields["dataset"].annotation,
    }
    for section_name, section_cls in sections.items():
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        allowed = set(section_cls.model_fields)
        stray = set(section) - allowed
        assert not stray, f"{config_path.name}:{section_name} has stray keys {stray}"


def test_brief_extraction_max_tokens_validator() -> None:
    with pytest.raises(ValidationError):
        BriefExtractionConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        BriefExtractionConfig(max_tokens=20_001)
    # Inside the range is fine.
    BriefExtractionConfig(max_tokens=4_000)


def test_brief_extraction_temperature_validator() -> None:
    with pytest.raises(ValidationError):
        BriefExtractionConfig(temperature=-0.1)
    with pytest.raises(ValidationError):
        BriefExtractionConfig(temperature=2.1)
    BriefExtractionConfig(temperature=0.0)
    BriefExtractionConfig(temperature=2.0)


def test_brief_extraction_forbid_ngram_overlap_validator() -> None:
    with pytest.raises(ValidationError):
        BriefExtractionConfig(forbid_ngram_overlap=1)
    with pytest.raises(ValidationError):
        BriefExtractionConfig(forbid_ngram_overlap=11)
    BriefExtractionConfig(forbid_ngram_overlap=5)


def test_rationale_validators() -> None:
    with pytest.raises(ValidationError):
        RationaleConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        RationaleConfig(temperature=3.0)


def test_filter_validators() -> None:
    with pytest.raises(ValidationError):
        FilterConfig(dedup_similarity_threshold=-0.1)
    with pytest.raises(ValidationError):
        FilterConfig(dedup_similarity_threshold=1.1)
    with pytest.raises(ValidationError):
        FilterConfig(min_deliverable_chars=5)
    with pytest.raises(ValidationError):
        FilterConfig(min_deliverable_chars=2_000)
    FilterConfig(dedup_similarity_threshold=0.5, min_deliverable_chars=40)


def test_construction_config_rejects_unknown_top_level_keys() -> None:
    """``extra=forbid`` should catch typos at the top level."""
    with pytest.raises(ValidationError):
        ConstructionV2Config(output_dir="x", unknown_section={})  # type: ignore[call-arg]


def test_dataset_config_defaults_sum_to_one() -> None:
    """Sanity: default split ratios must add up exactly to 1.0."""
    cfg = DatasetConfig()
    total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    assert abs(total - 1.0) < 1e-9


def test_require_tone_signals_flag_is_gone() -> None:
    """The flag was unused; ``extra=forbid`` should now reject it."""
    with pytest.raises(ValidationError):
        BriefExtractionConfig(require_tone_signals=True)  # type: ignore[call-arg]
