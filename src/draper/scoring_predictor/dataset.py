"""HuggingFace-compatible dataset + collator for the scoring predictor.

Wraps a Polars split DataFrame (columns laid out by
:func:`draper.scoring_predictor.data.examples_to_polars`) as a
``torch.utils.data.Dataset`` and provides a collate function that pads to the
max length in the batch and stacks the target / mask / weight tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence


_TARGET_COLS = (
    "target_composite",
    "target_survivability",
    "target_engagement_volume",
    "target_engagement_velocity",
)
_MASK_COLS = (
    "mask_composite",
    "mask_survivability",
    "mask_engagement_volume",
    "mask_engagement_velocity",
)


class ScoringDataset(Dataset[dict[str, Any]]):
    """Map-style dataset over a split parquet."""

    def __init__(
        self,
        df: pl.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        include_targets: bool = True,
    ) -> None:
        self._records: list[dict[str, Any]] = df.to_dicts()
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._include_targets = include_targets

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self._records[idx]
        tokens = self._tokenizer(
            rec["text"],
            truncation=True,
            max_length=self._max_length,
            padding=False,
            return_token_type_ids=False,
        )
        item: dict[str, Any] = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        if self._include_targets:
            item["targets"] = [float(rec[c]) for c in _TARGET_COLS]
            item["target_mask"] = [bool(rec[c]) for c in _MASK_COLS]
            item["sample_weight"] = float(rec["sample_weight"])
        return item


@dataclass
class ScoringCollator:
    """Pads ``input_ids`` / ``attention_mask`` and stacks target tensors."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # ``tokenizer.pad`` handles input_ids / attention_mask with proper
        # left/right side, multiple-of, and dtype.
        encodings = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        padded = self.tokenizer.pad(
            encodings,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch: dict[str, torch.Tensor] = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }
        if "targets" in features[0]:
            batch["targets"] = torch.tensor(
                [f["targets"] for f in features], dtype=torch.float32
            )
            batch["target_mask"] = torch.tensor(
                [f["target_mask"] for f in features], dtype=torch.bool
            )
            batch["sample_weight"] = torch.tensor(
                [f["sample_weight"] for f in features], dtype=torch.float32
            )
        return batch
