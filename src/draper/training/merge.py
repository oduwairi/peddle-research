"""Merge a trained LoRA adapter into the base model for vLLM serving."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from draper.utils.logging import get_logger

logger = get_logger("draper.training")

SaveMethod = Literal["merged_16bit", "merged_4bit", "lora"]


def merge_adapter(
    adapter_dir: str | Path,
    base_model: str,
    out_dir: str | Path,
    *,
    max_seq_length: int = 4096,
    save_method: SaveMethod = "merged_16bit",
) -> Path:
    """Load the base model + adapter, merge, and save weights ready for vLLM.

    Uses Unsloth's ``save_pretrained_merged`` which handles dequant + merge
    + save in a single call. Returns the output directory.

    A ``merged_meta.json`` is written next to the weights with provenance —
    base model id, adapter source, save method, timestamp — so the serving
    step can audit which checkpoint produced which artifact.
    """
    from unsloth import FastLanguageModel

    adapter_path = Path(adapter_dir)
    out_path = Path(out_dir)
    if not adapter_path.exists():
        msg = f"Adapter directory not found: {adapter_path}"
        raise FileNotFoundError(msg)

    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model %s for merge", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # need fp16/bf16 base to merge cleanly
        dtype=None,
    )

    logger.info("Loading adapter from %s", adapter_path)
    model.load_adapter(str(adapter_path))

    logger.info("Merging and saving to %s (method=%s)", out_path, save_method)
    model.save_pretrained_merged(
        str(out_path),
        tokenizer,
        save_method=save_method,
    )

    meta = {
        "base_model": base_model,
        "adapter_dir": str(adapter_path.resolve()),
        "save_method": save_method,
        "max_seq_length": max_seq_length,
        "merged_at": datetime.now(UTC).isoformat(),
    }
    (out_path / "merged_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Merge complete; provenance at %s", out_path / "merged_meta.json")
    return out_path
