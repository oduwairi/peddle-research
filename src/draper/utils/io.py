"""JSONL/Parquet I/O helpers and checkpoint management for resume support."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel

# --- JSONL ---


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Sequence[dict[str, Any] | BaseModel], path: str | Path) -> int:
    """Write records to a JSONL file. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, BaseModel):
                f.write(record.model_dump_json() + "\n")
            else:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def append_jsonl(records: Sequence[dict[str, Any] | BaseModel], path: str | Path) -> int:
    """Append records to a JSONL file. Creates file if it doesn't exist. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, BaseModel):
                f.write(record.model_dump_json() + "\n")
            else:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def update_jsonl_records(
    path: Path,
    updates: dict[str, dict[str, Any]],
) -> int:
    """Update specific records in a JSONL file by ad_id.

    Reads all records, merges updates for matching ad_ids (shallow merge
    for top-level keys, deep merge for nested dicts), and rewrites the file.

    Args:
        path: Path to the JSONL file.
        updates: Mapping of ad_id → fields to merge into that record.

    Returns:
        Number of records updated.
    """
    records = read_jsonl(path)
    updated = 0
    for record in records:
        ad_id = record.get("ad_id", "")
        if ad_id in updates:
            patch = updates[ad_id]
            for key, value in patch.items():
                if isinstance(value, dict) and isinstance(record.get(key), dict):
                    record[key].update(value)
                else:
                    record[key] = value
            updated += 1

    # Rewrite the file
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return updated


# --- Parquet ---


def _flatten_for_parquet(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a scored ad record into Parquet-safe scalar columns.

    Expands nested models (ad, ad_copy) into prefixed flat keys and
    drops deeply nested blobs (raw_data, demographics) that can't be
    represented in a columnar format.
    """
    flat: dict[str, Any] = {}

    ad = record.get("ad", {})
    if isinstance(ad, dict):
        for k, v in ad.items():
            if k == "raw_data":
                continue  # arbitrary nested blob — skip
            if k == "demographics":
                continue  # arbitrary nested blob — skip
            if k == "ad_copy" and isinstance(v, dict):
                for ck, cv in v.items():
                    flat[f"ad_copy_{ck}"] = cv
            elif isinstance(v, list):
                flat[k] = json.dumps(v) if v else ""
            else:
                flat[k] = v
    else:
        flat["ad"] = str(ad)

    # Top-level scored fields
    for k in ("composite_score", "tier", "scoring_version"):
        if k in record:
            flat[k] = record[k]

    # Signal scores as individual columns
    signals = record.get("signal_scores", {})
    if isinstance(signals, dict):
        for sk, sv in signals.items():
            flat[f"signal_{sk}"] = sv

    # Tier probabilities as individual columns (v2 only)
    tier_probs = record.get("tier_probs", {})
    if isinstance(tier_probs, dict):
        for tk, tv in tier_probs.items():
            flat[f"tier_prob_{tk}"] = tv

    return flat


def jsonl_to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> int:
    """Convert a scored-ads JSONL file to Parquet. Returns row count.

    Flattens nested structures into scalar columns and drops blobs
    (raw_data, demographics) that don't fit a columnar format.
    """
    records = read_jsonl(jsonl_path)
    if not records:
        return 0
    flat_records = [_flatten_for_parquet(r) for r in records]
    df = pl.DataFrame(flat_records)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    return len(df)


def read_parquet(path: str | Path) -> pl.DataFrame:
    """Read a Parquet file into a Polars DataFrame."""
    return pl.read_parquet(path)


# --- Checkpoints ---


class Checkpoint:
    """Simple file-based checkpoint for resumable operations.

    Stores a JSON dict at {path}.checkpoint.json with arbitrary state
    (e.g. last cursor, page number, count processed).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(f"{path}.checkpoint.json")
        self._state: dict[str, Any] = {}
        if self._path.exists():
            with self._path.open("r") as f:
                self._state = json.load(f)

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def update(self, **kwargs: Any) -> None:
        """Update checkpoint state and persist to disk."""
        self._state.update(kwargs)
        self._save()

    def clear(self) -> None:
        """Remove checkpoint file."""
        self._state = {}
        if self._path.exists():
            self._path.unlink()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            json.dump(self._state, f, indent=2, default=str)
