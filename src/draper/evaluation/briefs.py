"""Loader for the held-out copywriting test split into Brief objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import load_from_disk

from .schemas import Brief, UrlScenario


def load_test_briefs(split_dir: str | Path) -> list[Brief]:
    """Load the HF Arrow dataset at ``split_dir`` and convert to Briefs.

    Handles two on-disk shapes:
      * v1 (``data/final/``): example_id/task_format/platform/vertical/
        source_tiers/construction_model live as top-level columns.
      * v2 (``data/constructed_v2/final_v2/``): only ``messages`` + a
        ``metadata`` dict with ``example_id``/``ad_id``/``platform``. The
        multi-format / vertical / source_tier columns were retired in
        2026-04 when non-copywriting formats moved to ``archive/``.

    The training data uses the chat format
    ``[{role:system}, {role:user}, {role:assistant}]``; we split that into
    a brief (system+user) and the reference assistant response (held back
    from inference, kept for diagnostic comparison only).
    """
    ds = load_from_disk(str(split_dir))
    briefs: list[Brief] = []
    for row in ds:
        messages: list[dict[str, str]] = row["messages"]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        meta: dict[str, Any] = row.get("metadata") or {}
        example_id = row.get("example_id") or meta.get("example_id") or ""
        platform = row.get("platform") or meta.get("platform") or "unknown"
        briefs.append(
            Brief(
                example_id=example_id,
                task_format=row.get("task_format") or "copywriting",
                platform=platform,
                vertical=row.get("vertical") or meta.get("vertical") or "unknown",
                source_tiers=list(row.get("source_tiers") or []),
                construction_model=row.get("construction_model"),
                system=system,
                user=user,
                reference_assistant=assistant,
            )
        )
    return briefs


def load_url_scenarios(path: str | Path) -> list[UrlScenario]:
    """Load Arm 2 URL-anchored scenarios from a JSONL file.

    Each line is one scenario. Returns an empty list when the file does
    not exist (Arm 2 is curated separately and may not be ready yet).
    """
    p = Path(path)
    if not p.exists():
        return []
    import json

    scenarios: list[UrlScenario] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data: dict[str, Any] = json.loads(line)
            scenarios.append(UrlScenario.model_validate(data))
    return scenarios
