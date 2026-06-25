"""Assemble v2 examples into an HF DatasetDict for training.

Row shape (arch doc §5):

::

    {
      "messages": [
        {"role": "system",    "content": STATIC_SYSTEM_PROMPT},
        {"role": "user",      "content": canonical_dict_json(brief_dict)},
        {"role": "assistant", "content":
            "<think>{R}</think>\\n\\n{deliverable}"
        },
      ],
      "metadata": {"example_id": ..., "ad_id": ..., "platform": ...},
    }

Stratifies by ``platform``, splits 90/5/5 with a deterministic seed.
Writes a stratification audit Markdown alongside.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Value

from draper.construction_v2.config import DatasetConfig
from draper.construction_v2.schemas.brief import (
    STATIC_SYSTEM_PROMPT,
    canonical_dict_json,
)
from draper.construction_v2.schemas.records import ExampleRecord

logger = logging.getLogger("draper")


# Explicit features for HF Dataset — required so zero-row splits still
# materialize with a valid schema. Without this, `Dataset.from_list([])`
# raises ``SchemaInferenceError``. The list-of-dict bracket form yields
# the chat-format the data_loader expects (array-of-struct, not
# struct-of-array).
_FEATURES = Features(
    {
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "metadata": {
            "example_id": Value("string"),
            "ad_id": Value("string"),
            "platform": Value("string"),
        },
    }
)


def _row_for_example(example: ExampleRecord) -> dict[str, Any]:
    user_content = canonical_dict_json(example.brief)
    assistant_content = f"<think>\n{example.think}\n</think>\n\n{example.deliverable}"
    return {
        "messages": [
            {"role": "system", "content": STATIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "example_id": example.example_id,
            "ad_id": example.ad_id,
            "platform": example.platform,
        },
    }


def _split_stratified(
    examples: list[ExampleRecord], config: DatasetConfig
) -> tuple[list[ExampleRecord], list[ExampleRecord], list[ExampleRecord]]:
    """Per-platform 90/5/5 split, deterministic by ``config.seed``."""
    by_platform: dict[str, list[ExampleRecord]] = defaultdict(list)
    for ex in examples:
        by_platform[ex.platform].append(ex)

    rng = random.Random(config.seed)
    train: list[ExampleRecord] = []
    val: list[ExampleRecord] = []
    test: list[ExampleRecord] = []
    for plat in sorted(by_platform):
        bucket = by_platform[plat]
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * config.train_ratio))
        n_val = int(round(n * config.val_ratio))
        # Guarantee non-empty val/test when the bucket can spare a row.
        # The training data_loader requires both ``train`` and
        # ``validation`` to be present; an empty val split also trips
        # HF Datasets' arrow-schema check on save_to_disk.
        if n >= 3:
            if n_val == 0:
                n_val = 1
                n_train = min(n_train, n - 2)
            if (n - n_train - n_val) == 0:
                n_train = max(1, n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 1
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])
    return train, val, test


def _write_audit(
    train: list[ExampleRecord],
    val: list[ExampleRecord],
    test: list[ExampleRecord],
    audit_dir: Path,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)

    def _counts(rows: list[ExampleRecord]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r.platform] += 1
        return dict(counts)

    train_c = _counts(train)
    val_c = _counts(val)
    test_c = _counts(test)
    all_platforms = sorted(set(train_c) | set(val_c) | set(test_c))

    lines: list[str] = [
        "# Construction v2 — split stratification audit\n",
        "| platform | train | val | test | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for plat in all_platforms:
        t, v, te = train_c.get(plat, 0), val_c.get(plat, 0), test_c.get(plat, 0)
        lines.append(f"| {plat} | {t} | {v} | {te} | {t + v + te} |")
    totals = (len(train), len(val), len(test))
    lines.append(f"| **total** | {totals[0]} | {totals[1]} | {totals[2]} | {sum(totals)} |")

    out_path = audit_dir / "stratification.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Stratification audit written: %s", out_path)


def build_dataset(
    examples: list[ExampleRecord],
    output_dir: str | Path,
    *,
    dataset_config: DatasetConfig | None = None,
    audit_dir: str | Path | None = None,
) -> DatasetDict:
    """Build + save an HF DatasetDict at ``output_dir``."""
    cfg = dataset_config or DatasetConfig()
    if not examples:
        msg = "build_dataset called with zero examples"
        raise ValueError(msg)

    train, val, test = _split_stratified(examples, cfg)

    def _to_dataset(rows: list[ExampleRecord]) -> Dataset:
        return Dataset.from_list([_row_for_example(e) for e in rows], features=_FEATURES)

    out = DatasetDict(
        {
            "train": _to_dataset(train),
            "validation": _to_dataset(val),
            "test": _to_dataset(test),
        }
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(output_path))
    logger.info(
        "Saved v2 DatasetDict to %s — train=%d, val=%d, test=%d",
        output_path,
        len(train),
        len(val),
        len(test),
    )
    if audit_dir is not None:
        _write_audit(train, val, test, Path(audit_dir))
    return out


__all__ = ["build_dataset"]
