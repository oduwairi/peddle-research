"""Tests for the dataset builder."""

from __future__ import annotations

from draper.construction.dataset_builder import DatasetBuilder
from draper.construction.schemas import (
    ChatMessage,
    ExampleMetadata,
    TaskFormat,
    TrainingExample,
)


def _make_examples(
    n: int,
    task_format: TaskFormat = TaskFormat.COPYWRITING,
    platform: str = "facebook",
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for i in range(n):
        examples.append(
            TrainingExample(
                task_format=task_format,
                messages=[
                    ChatMessage(role="system", content="You are a strategist."),
                    ChatMessage(role="user", content=f"Task {i}"),
                    ChatMessage(role="assistant", content=f"Response {i} " * 50),
                ],
                metadata=ExampleMetadata(
                    source_ad_ids=[f"ad_{i}"],
                    source_tiers=["high"],
                    platform=platform,
                    vertical=f"{platform}:broad",
                    construction_model="test",
                ),
            )
        )
    return examples


class TestStratifiedSplit:
    def test_basic_split(self) -> None:
        examples = _make_examples(100)
        builder = DatasetBuilder(
            constructed_dir="data/constructed",
            output_dir="data/final",
        )
        train, val, test = builder.stratified_split(examples)
        assert len(train) + len(val) + len(test) == 100
        assert len(train) >= 80  # ~85%
        assert len(val) >= 5  # ~7.5%
        assert len(test) >= 5  # ~7.5%

    def test_small_group_goes_to_train(self) -> None:
        examples = _make_examples(2)
        builder = DatasetBuilder(
            constructed_dir="data/constructed",
            output_dir="data/final",
        )
        train, val, test = builder.stratified_split(examples)
        # With only 2 examples, all should go to train
        assert len(train) == 2


class TestBuild:
    def test_builds_dataset_dict(self) -> None:
        examples = _make_examples(20)
        builder = DatasetBuilder(
            constructed_dir="data/constructed",
            output_dir="/tmp/draper_test_build",
        )
        ds = builder.build(examples)
        assert "train" in ds
        assert "validation" in ds
        assert "test" in ds
        total = sum(len(split) for split in ds.values())
        assert total == 20

    def test_hf_record_columns(self) -> None:
        examples = _make_examples(5)
        records = DatasetBuilder._to_hf_records(examples)
        assert len(records) == 5
        r = records[0]
        assert "example_id" in r
        assert "task_format" in r
        assert "messages" in r
        assert "platform" in r
        assert "vertical" in r
        assert "source_tiers" in r
        assert "construction_model" in r
        # Messages should be list of dicts
        assert isinstance(r["messages"], list)
        assert r["messages"][0]["role"] == "system"
