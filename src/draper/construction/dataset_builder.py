"""Assemble filtered training examples into a HuggingFace Dataset.

Loads JSONL from each format directory, applies stratified splitting, and
saves as Arrow files in ``data/final/``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from draper.construction.schemas import (
    DatasetSplitConfig,
    TaskFormat,
    TrainingExample,
)
from draper.utils.io import read_jsonl

logger = logging.getLogger("draper")


class DatasetBuilder:
    """Combines per-format JSONL into a single HuggingFace DatasetDict."""

    def __init__(
        self,
        constructed_dir: str | Path,
        output_dir: str | Path,
        split_config: DatasetSplitConfig | None = None,
    ) -> None:
        self._constructed_dir = Path(constructed_dir)
        self._output_dir = Path(output_dir)
        self._split = split_config or DatasetSplitConfig()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all_examples(self) -> list[TrainingExample]:
        """Load examples from all format subdirectories."""
        all_examples: list[TrainingExample] = []
        for fmt in TaskFormat:
            path = self._constructed_dir / fmt.value / "examples.jsonl"
            if not path.exists():
                logger.warning("No examples for %s at %s", fmt.value, path)
                continue
            records = read_jsonl(path)
            examples = [TrainingExample(**r) for r in records]
            all_examples.extend(examples)
            logger.info("Loaded %d examples for %s", len(examples), fmt.value)
        logger.info("Total examples loaded: %d", len(all_examples))
        return all_examples

    # ------------------------------------------------------------------
    # Stratified split
    # ------------------------------------------------------------------

    def stratified_split(
        self,
        examples: list[TrainingExample],
    ) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
        """Split examples into train/val/test, stratified by task_format + platform."""
        # Group by stratification key
        groups: dict[str, list[TrainingExample]] = defaultdict(list)
        for ex in examples:
            key = f"{ex.task_format}:{ex.metadata.platform}"
            groups[key].append(ex)

        train: list[TrainingExample] = []
        val: list[TrainingExample] = []
        test: list[TrainingExample] = []

        for _key, group in groups.items():
            n = len(group)
            n_val = max(1, round(n * self._split.val_ratio))
            n_test = max(1, round(n * self._split.test_ratio))
            n_train = n - n_val - n_test
            if n_train < 1:
                # Too few examples in this stratum — put all in train
                train.extend(group)
                continue
            train.extend(group[:n_train])
            val.extend(group[n_train : n_train + n_val])
            test.extend(group[n_train + n_val :])

        logger.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
        return train, val, test

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_hf_records(examples: list[TrainingExample]) -> list[dict[str, Any]]:
        """Convert TrainingExamples to flat dicts for HuggingFace Dataset."""
        records: list[dict[str, Any]] = []
        for ex in examples:
            records.append(
                {
                    "example_id": ex.example_id,
                    "task_format": ex.task_format.value,
                    "messages": [{"role": m.role, "content": m.content} for m in ex.messages],
                    "platform": ex.metadata.platform,
                    "vertical": ex.metadata.vertical,
                    "source_tiers": ex.metadata.source_tiers,
                    "construction_model": ex.metadata.construction_model,
                }
            )
        return records

    # ------------------------------------------------------------------
    # Build & save
    # ------------------------------------------------------------------

    def build(self, examples: list[TrainingExample] | None = None) -> DatasetDict:
        """Build the complete DatasetDict from examples.

        If ``examples`` is None, loads from the constructed directory.
        """
        if examples is None:
            examples = self.load_all_examples()

        if not examples:
            logger.warning("No examples to build dataset from")
            return DatasetDict()

        train_ex, val_ex, test_ex = self.stratified_split(examples)

        ds = DatasetDict(
            {
                "train": Dataset.from_list(self._to_hf_records(train_ex)),
                "validation": Dataset.from_list(self._to_hf_records(val_ex)),
                "test": Dataset.from_list(self._to_hf_records(test_ex)),
            }
        )
        logger.info("DatasetDict built: %s", {k: len(v) for k, v in ds.items()})
        return ds

    def save(self, ds: DatasetDict) -> None:
        """Save the DatasetDict to disk and write statistics."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(self._output_dir))
        logger.info("Dataset saved to %s", self._output_dir)

        # Write statistics
        stats = self._compute_statistics(ds)
        stats_path = self._output_dir / "statistics.json"
        with stats_path.open("w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Statistics written to %s", stats_path)

    def build_and_save(self, examples: list[TrainingExample] | None = None) -> DatasetDict:
        """Convenience: build + save in one call."""
        ds = self.build(examples)
        if ds:
            self.save(ds)
        return ds

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_statistics(ds: DatasetDict) -> dict[str, Any]:
        """Compute per-split, per-format, per-platform statistics."""
        stats: dict[str, Any] = {"splits": {}}
        for split_name, split_ds in ds.items():
            split_stats: dict[str, Any] = {"total": len(split_ds)}

            # Per format
            by_format: dict[str, int] = defaultdict(int)
            by_platform: dict[str, int] = defaultdict(int)
            for row in split_ds:
                by_format[str(row["task_format"])] += 1
                by_platform[str(row["platform"])] += 1

            split_stats["by_format"] = dict(by_format)
            split_stats["by_platform"] = dict(by_platform)
            stats["splits"][split_name] = split_stats

        stats["total"] = sum(s["total"] for s in stats["splits"].values())
        return stats
