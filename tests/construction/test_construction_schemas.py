"""Tests for construction schemas and configuration."""

from __future__ import annotations

import pytest

from draper.construction.schemas import (
    ChatMessage,
    ClusterInfo,
    ConstructionConfig,
    DatasetSplitConfig,
    ExampleMetadata,
    FormatConfig,
    PromptStyle,
    QualityFilterConfig,
    TaskFormat,
    TrainingExample,
)


class TestChatMessage:
    def test_basic(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_roles(self) -> None:
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role=role, content="test")  # type: ignore[arg-type]
            assert msg.role == role


class TestTrainingExample:
    def test_defaults(self) -> None:
        ex = TrainingExample(
            task_format=TaskFormat.COPYWRITING,
            messages=[
                ChatMessage(role="user", content="Write me ad copy."),
                ChatMessage(role="assistant", content="Here's the copy..."),
            ],
        )
        assert ex.example_id
        assert len(ex.example_id) == 12
        assert ex.task_format == TaskFormat.COPYWRITING
        assert len(ex.messages) == 2
        assert ex.metadata.construction_model == ""

    def test_roundtrip_json(self) -> None:
        ex = TrainingExample(
            task_format=TaskFormat.COPYWRITING,
            messages=[
                ChatMessage(role="system", content="You are a copywriter."),
                ChatMessage(role="user", content="Write me ad copy."),
                ChatMessage(role="assistant", content="Headline: buy now."),
            ],
            metadata=ExampleMetadata(
                source_ad_ids=["ad1"],
                source_tiers=["high"],
                platform="facebook",
                vertical="facebook:broad",
                construction_model="chat",
            ),
        )
        json_str = ex.model_dump_json()
        restored = TrainingExample.model_validate_json(json_str)
        assert restored.example_id == ex.example_id
        assert restored.task_format == TaskFormat.COPYWRITING
        assert len(restored.messages) == 3
        assert restored.metadata.source_ad_ids == ["ad1"]
        assert restored.metadata.platform == "facebook"


class TestClusterInfo:
    def test_basic(self) -> None:
        ci = ClusterInfo(
            cluster_id="adv_TestBrand",
            cluster_type="advertiser",
            advertiser_name="TestBrand",
            platform="facebook",
            vertical="facebook:ecommerce",
            ad_ids=["a1", "a2", "a3"],
            tier_counts={"high": 2, "low": 1},
        )
        assert ci.cluster_id == "adv_TestBrand"
        assert len(ci.ad_ids) == 3
        assert ci.tier_counts["high"] == 2


class TestTaskFormat:
    def test_copywriting_only(self) -> None:
        assert list(TaskFormat) == [TaskFormat.COPYWRITING]
        assert TaskFormat.COPYWRITING == "copywriting"


class TestFormatConfig:
    def test_defaults_full_style_split(self) -> None:
        fmt = FormatConfig(target=100)
        assert PromptStyle.DATA_GROUNDED in fmt.valid_styles
        assert PromptStyle.CONTEXT_DISTILLED in fmt.valid_styles
        assert PromptStyle.NATURAL in fmt.valid_styles
        assert sum(fmt.style_ratios.values()) == pytest.approx(1.0)

    def test_backtranslation_only(self) -> None:
        fmt = FormatConfig(
            target=50,
            valid_styles=[PromptStyle.BACKTRANSLATION],
            style_ratios={PromptStyle.BACKTRANSLATION.value: 1.0},
        )
        assert fmt.valid_styles == [PromptStyle.BACKTRANSLATION]

    def test_rejects_mismatched_style_ratio(self) -> None:
        with pytest.raises(ValueError, match="not in valid_styles"):
            FormatConfig(
                target=100,
                valid_styles=[PromptStyle.DATA_GROUNDED],
                style_ratios={
                    PromptStyle.DATA_GROUNDED.value: 0.5,
                    PromptStyle.NATURAL.value: 0.5,
                },
            )

    def test_rejects_ratios_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            FormatConfig(
                target=100,
                style_ratios={
                    PromptStyle.DATA_GROUNDED.value: 0.40,
                    PromptStyle.CONTEXT_DISTILLED.value: 0.40,
                    PromptStyle.NATURAL.value: 0.10,
                },
            )


class TestConstructionConfig:
    def test_from_yaml(self) -> None:
        cfg = ConstructionConfig.from_yaml("configs/construction.yaml")
        assert len(cfg.formats) == 1
        assert cfg.target_for(TaskFormat.COPYWRITING) > 0
        assert cfg.quality_filter.dedup_threshold == 0.80
        assert cfg.dataset.train_ratio == 0.85

    def test_prompt_style_fallback_ratios(self) -> None:
        cfg = ConstructionConfig.from_yaml("configs/construction.yaml")
        assert cfg.prompt_style.data_grounded_ratio == 0.30
        assert cfg.prompt_style.context_distilled_ratio == 0.50
        assert cfg.prompt_style.natural_ratio == 0.20

    def test_copywriting_is_backtranslation_only(self) -> None:
        cfg = ConstructionConfig.from_yaml("configs/construction.yaml")
        assert cfg.valid_styles_for(TaskFormat.COPYWRITING) == [PromptStyle.BACKTRANSLATION]

    def test_config_rejects_unknown_format(self) -> None:
        bad_data = {
            "formats": {
                "positioning": {
                    "target": 100,
                    "valid_styles": ["data_grounded"],
                    "style_ratios": {"data_grounded": 1.0},
                }
            }
        }
        with pytest.raises(ValueError, match="Unknown format"):
            ConstructionConfig(**bad_data)

    def test_overgeneration_buffer(self) -> None:
        cfg = ConstructionConfig.from_yaml("configs/construction.yaml")
        assert cfg.overgeneration_buffer == 1.25
        copywriting_target = cfg.target_for(TaskFormat.COPYWRITING)
        assert cfg.raw_target_for(TaskFormat.COPYWRITING) == int(
            round(copywriting_target * 1.25)
        )

    def test_defaults(self) -> None:
        cfg = ConstructionConfig()
        assert cfg.quality_filter == QualityFilterConfig()
        assert cfg.dataset == DatasetSplitConfig()
        assert cfg.overgeneration_buffer == 1.25
        assert cfg.target_for(TaskFormat.COPYWRITING) == 0
