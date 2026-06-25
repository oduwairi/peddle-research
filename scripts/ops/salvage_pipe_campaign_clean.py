"""Salvage arm-2 (pipe) clean ad-copy records from emitted campaign objects.

Many ``B_pipe`` / ``C_pipe`` agent-pipeline inferences emitted a structured
campaign but recorded only a one-line *summary* as ``assistant_text``
("Drafted a PINTEREST campaign: ..."). The LLM normalizer was fed that summary
and returned ``<EXTRACTION_FAILED>`` even though the canonical published copy
sits in ``inference.campaign``. That silently dropped those briefs from every
arm that consumes ``inferences_clean/`` (learned scorer, judges, reference
metrics), shrinking the paired-ablation intersection.

This script overlays a valid cleaned record for exactly those failed/missing
cases by flattening the campaign object directly (no LLM call), via
``campaign_published_copy`` — the same helper ``eval.py normalize`` now uses to
pick its raw source. The on-disk ``raw_text_sha256`` is keyed on the flattened
campaign copy, so a later ``eval.py normalize`` run (which also prefers campaign
copy) finds a matching cache and leaves these records untouched.

Idempotent: only writes records whose current clean text is missing or
``<EXTRACTION_FAILED>``. Re-running is a no-op. ``--dry-run`` previews counts.

Usage::

    uv run python scripts/ops/salvage_pipe_campaign_clean.py --configs B_pipe,C_pipe
    uv run python scripts/ops/salvage_pipe_campaign_clean.py --configs B_pipe,C_pipe --dry-run
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from draper.evaluation.judge.normalize import (
    EXTRACTION_FAILED,
    CleanedRecord,
    campaign_published_copy,
    clean_path,
    load_clean,
    save_clean,
)

app = typer.Typer(add_completion=False)

FLATTEN_TAG = "campaign-flatten:v1"


def _is_valid(text: str | None) -> bool:
    """Match the downstream notion of a usable clean record."""
    if not text:
        return False
    t = text.strip()
    return bool(t) and EXTRACTION_FAILED not in t


def _valid_ids(clean_root: Path, config: str) -> set[str]:
    ids: set[str] = set()
    for p in (clean_root / config).glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if _is_valid(data.get("assistant_text_clean")):
            ids.add(data.get("example_id", p.stem))
    return ids


@app.command()
def main(
    configs: str = typer.Option("B_pipe,C_pipe", help="Comma-separated arm-2 config names."),
    inferences_root: Path = typer.Option(  # noqa: B008
        Path("data/eval/inferences"), help="Raw inference root."
    ),
    clean_root: Path = typer.Option(  # noqa: B008
        Path("data/eval/inferences_clean"), help="Cleaned-record root."
    ),
    report_against: str = typer.Option(
        "A,C", help="Configs to intersect with for the paired-ablation count."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing."),
) -> None:
    config_names = [c.strip() for c in configs.split(",") if c.strip()]
    now = datetime.now(UTC).isoformat()

    for config in config_names:
        inf_dir = inferences_root / config
        if not inf_dir.exists():
            typer.echo(f"[skip] no inference dir: {inf_dir}")
            continue
        salvaged = no_campaign = no_copy = already_valid = 0
        for p in sorted(inf_dir.glob("*.json")):
            inf = json.loads(p.read_text(encoding="utf-8"))
            eid = inf["example_id"]
            campaign = inf.get("campaign")
            if not campaign:
                no_campaign += 1
                continue
            existing = load_clean(clean_root, config, eid)
            if existing is not None and _is_valid(existing.assistant_text_clean):
                already_valid += 1
                continue
            copy = campaign_published_copy(campaign)
            if not copy.strip():
                no_copy += 1
                continue
            record = CleanedRecord(
                example_id=eid,
                config=config,
                assistant_text_clean=copy,
                extractor_model=FLATTEN_TAG,
                extracted_at=now,
                raw_text_sha256=hashlib.sha256(copy.encode("utf-8")).hexdigest(),
            )
            if not dry_run:
                save_clean(record, clean_root)
            salvaged += 1
            if salvaged <= 3:
                preview = copy.replace("\n", " ")[:90]
                typer.echo(f"    {config}/{eid}: {preview!r}")
        verb = "would salvage" if dry_run else "salvaged"
        typer.echo(
            f"[{config}] {verb}={salvaged}  already_valid={already_valid}  "
            f"no_campaign={no_campaign}  no_copy_in_campaign={no_copy}  "
            f"-> {clean_path(clean_root, config, '<id>').parent}"
        )

    # Paired-ablation intersection report.
    ref = [c.strip() for c in report_against.split(",") if c.strip()]
    sets = {c: _valid_ids(clean_root, c) for c in [*ref, *config_names]}
    inter = set.intersection(*sets.values()) if sets else set()
    counts = "  ".join(f"{c}={len(ids)}" for c, ids in sets.items())
    typer.echo("\nvalid clean counts:  " + counts)
    typer.echo(
        f"intersection ({'∩'.join(sets)}) = {len(inter)}"
        + ("   [DRY-RUN: reflects pre-salvage state]" if dry_run else "")
    )


if __name__ == "__main__":
    app()
