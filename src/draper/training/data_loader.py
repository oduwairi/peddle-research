"""Dataset loading helpers for QLoRA training.

The HF Dataset at ``data/final/`` already has a ``messages`` column with the
chat-format examples. TRL's SFTTrainer consumes this column natively and
applies the tokenizer's chat_template internally, so this module is mostly
a thin wrapper around ``datasets.load_from_disk``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from draper.utils.logging import get_logger

logger = get_logger("draper.training")

REQUIRED_SPLITS = ("train", "validation")
REQUIRED_COLUMN = "messages"


def load_dataset_dict(dataset_dir: str | Path) -> DatasetDict:
    """Load the constructed dataset and validate its shape.

    Raises
    ------
    FileNotFoundError
        If ``dataset_dir`` doesn't exist.
    ValueError
        If a required split or the ``messages`` column is missing.
    """
    path = Path(dataset_dir)
    if not path.exists():
        msg = f"Dataset directory not found: {path}"
        raise FileNotFoundError(msg)

    ds = load_from_disk(str(path))
    if not isinstance(ds, DatasetDict):
        msg = f"Expected a DatasetDict at {path}, got {type(ds).__name__}"
        raise ValueError(msg)

    for split in REQUIRED_SPLITS:
        if split not in ds:
            msg = f"Required split {split!r} missing from {path} (have: {list(ds.keys())})"
            raise ValueError(msg)
        if REQUIRED_COLUMN not in ds[split].column_names:
            msg = (
                f"Split {split!r} is missing the {REQUIRED_COLUMN!r} column "
                f"(have: {ds[split].column_names})"
            )
            raise ValueError(msg)

    logger.info(
        "Loaded dataset from %s: %s",
        path,
        {k: len(v) for k, v in ds.items()},
    )
    return ds


def subset_for_smoke(ds: DatasetDict, n: int) -> DatasetDict:
    """Take the first ``n`` rows of every split.

    Used by the smoke path so the loop runs in seconds without burning
    cloud GPU time.
    """
    out = DatasetDict()
    for split, data in ds.items():
        n_take = min(n, len(data))
        out[split] = data.select(range(n_take))
    logger.info("Smoke subset: %s", {k: len(v) for k, v in out.items()})
    return out


def render_first_example(ds: DatasetDict, tokenizer: Any, split: str = "train") -> str:
    """Apply the tokenizer's chat_template to the first row of ``split``.

    Used by ``train.py inspect`` to verify visually that the chat template
    wraps assistant turns the way TRL's ``assistant_only_loss`` expects.
    """
    row = ds[split][0]
    rendered = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return str(rendered)
