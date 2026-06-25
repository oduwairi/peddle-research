"""Internal helpers for resolving per-(config, example_id) inference text.

Shared by :mod:`draper.evaluation.learned_scorer` and
:mod:`draper.evaluation.mauve_scorer`.  Both arms need to enumerate which
example_ids have any source text for a given config, and to resolve that text
preferring the LLM-cleaned copy over the raw assistant output.

These are package-internal; nothing outside ``draper.evaluation`` should
import from here directly.
"""

from __future__ import annotations

from pathlib import Path

from .driver import load_inferences_for_config
from .judge.normalize import EXTRACTION_FAILED, load_clean


def resolve_text(
    *,
    config: str,
    example_id: str,
    inferences_clean_dir: Path,
    inferences_raw_dir: Path,
) -> tuple[str, bool]:
    """Resolve the text to score for one (config, example_id).

    Returns ``(text, used_clean)``. Falls back to the raw inference's
    ``assistant_text`` only when no clean version exists. Returns ``("", ?)``
    when both sources are missing or extraction failed.
    """
    rec = load_clean(inferences_clean_dir, config, example_id)
    if rec is not None:
        if rec.assistant_text_clean == EXTRACTION_FAILED:
            return "", True  # explicit failure — don't fall back to raw
        return rec.assistant_text_clean, True

    # No clean cache — fall back to the raw assistant_text. Useful when
    # ``score`` is run before ``normalize`` (e.g., dev iteration); the
    # ``used_clean=False`` column lets the user spot the gap.
    raw_inferences = load_inferences_for_config(inferences_raw_dir, config)
    inf = raw_inferences.get(example_id)
    if inf is None:
        return "", False
    return inf.assistant_text, False


def config_example_ids(
    *,
    config: str,
    inferences_clean_dir: Path,
    inferences_raw_dir: Path,
) -> list[str]:
    """List all example_ids we have any source text for, for this config.

    Union of clean and raw to be defensive; in practice a clean dir mirrors
    the raw dir 1:1 once ``normalize`` has been run.
    """
    seen: set[str] = set()
    clean_dir = inferences_clean_dir / config
    if clean_dir.exists():
        for p in clean_dir.glob("*.json"):
            seen.add(p.stem)
    raw_dir = inferences_raw_dir / config
    if raw_dir.exists():
        for p in raw_dir.glob("*.json"):
            seen.add(p.stem)
    return sorted(seen)
