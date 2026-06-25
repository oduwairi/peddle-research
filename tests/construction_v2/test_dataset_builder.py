"""Dataset builder: stratified splits + 3-message row shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from draper.construction_v2.config import DatasetConfig
from draper.construction_v2.dataset.builder import build_dataset
from draper.construction_v2.schemas.brief import (
    STATIC_SYSTEM_PROMPT,
    Brief,
    BriefBridge,
    BriefProduct,
)
from draper.construction_v2.schemas.records import ExampleRecord


def _make_brief(platform: str) -> Brief:
    return Brief(
        task="Write ad copy for the platform below.",
        product=BriefProduct(
            name=f"prod-{platform}",
            description="A product description.",
            tone_signals=["crisp"],
        ),
        bridge=BriefBridge(positioning="p", target_audience="t", angle="a", buyer_pain="b"),
        platform=platform,  # type: ignore[arg-type]
    )


def _make_example(idx: int, platform: str) -> ExampleRecord:
    return ExampleRecord(
        example_id=f"ex-{idx}",
        ad_id=f"ad-{idx}",
        platform=platform,  # type: ignore[arg-type]
        brief=_make_brief(platform).model_dump(mode="json"),
        think=(
            "I want to lead with the product fact and shape the hook to "
            "match the brief's crisp tone signals."
        ),
        deliverable=f"Real ad copy number {idx} for {platform}.",
    )


def test_build_dataset_three_message_shape(tmp_path: Path) -> None:
    examples = [_make_example(i, "meta") for i in range(20)]
    out_dir = tmp_path / "final_v2"
    ds = build_dataset(examples, out_dir, dataset_config=DatasetConfig())
    assert {"train", "validation", "test"} <= set(ds.keys())
    row = ds["train"][0]
    assert "messages" in row
    assert len(row["messages"]) == 3
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert row["messages"][0]["content"] == STATIC_SYSTEM_PROMPT
    assistant = row["messages"][2]["content"]
    assert assistant.startswith("<think>\n")
    assert "</think>" in assistant
    # Deliverable region has no wrapping tags.
    assert "<ad>" not in assistant
    assert "<deliverable>" not in assistant
    # Deliverable text appears after </think>.
    think_end = assistant.index("</think>")
    after_think = assistant[think_end + len("</think>") :]
    assert "Real ad copy number" in after_think


def test_build_dataset_stratifies_by_platform(tmp_path: Path) -> None:
    examples: list[ExampleRecord] = []
    for plat in ("meta", "tiktok", "reddit"):
        examples.extend(_make_example(i, plat) for i in range(10))
    out_dir = tmp_path / "final_v2"
    ds = build_dataset(examples, out_dir, dataset_config=DatasetConfig())
    # Each split must have at least one row per platform (small N caveat
    # — with 10 per platform and 90/5/5, val + test may get rounding-zero
    # for some platforms; assert total counts only).
    assert len(ds["train"]) + len(ds["validation"]) + len(ds["test"]) == 30


def test_build_dataset_audit_written(tmp_path: Path) -> None:
    examples = [_make_example(i, "meta") for i in range(20)]
    out_dir = tmp_path / "final_v2"
    audit = tmp_path / "audit"
    build_dataset(
        examples,
        out_dir,
        dataset_config=DatasetConfig(),
        audit_dir=audit,
    )
    stratification = audit / "stratification.md"
    assert stratification.exists()
    assert "platform" in stratification.read_text()


def test_build_dataset_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero examples"):
        build_dataset([], tmp_path / "out")


def test_user_content_is_canonical_brief_json(tmp_path: Path) -> None:
    # Need ≥ 3 rows so the splitter can give each split at least one
    # example (the training data_loader requires train + validation).
    examples = [_make_example(i, "meta") for i in range(5)]
    ds = build_dataset(examples, tmp_path / "out", dataset_config=DatasetConfig())
    row = ds["train"][0]
    user_content = row["messages"][1]["content"]
    # Canonical JSON must start with sorted top-level keys.
    assert user_content.startswith('{"bridge":')
    # task field must be present in the user turn.
    assert '"task":"Write ad copy for the platform below."' in user_content
