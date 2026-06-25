"""Backfill platform-native field labels into v2 raw teacher responses.

The v2 teacher prompt was changed (2026-05-22) to embed platform-native
field labels (``**Primary text:**``, ``**Title:**``, ``**CTA:**``, …) in
the deliverable so the student learns to emit the same shape the
frontend's ``emit_campaign`` parser expects. Runs submitted before that
change have unlabeled flat-text deliverables.

This script rewrites ``responses_raw.jsonl`` in place (after taking a
``.pre-label-backfill.bak`` snapshot) so each deliverable carries the
labels for its source ad's populated slots. Strategy: find every
populated field's value as a verbatim substring in the deliverable and
prepend the slot's bold label. Failures (field value not findable
verbatim) are logged but the row is left untouched — the new ingest
label gate will reject those.

Usage::

    uv run python scripts/ops/backfill_v2_labels.py --run-id run100
    uv run python scripts/ops/backfill_v2_labels.py --run-id run300 \
        --dry-run
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from draper.construction_v2.config import ConstructionV2Config
from draper.construction_v2.dataset.source_selector import (
    SourceAd,
    load_source_ads_by_id,
)
from draper.construction_v2.platform_labels import (
    PLATFORM_LABEL_MAP,
    PlatformLabelGroup,
    platform_group_for,
)
from draper.scraping.schemas import AdSource

console = Console()
app = typer.Typer(add_completion=False)


# Common Unicode pairs the teacher LLMs flatten — fidelity tolerates
# these, so the backfill needs to as well or it spuriously fails on rows
# the ingest pipeline accepts.
_QUOTE_EQUIVALENTS: dict[str, str] = {
    "’": "'",  # right single quote
    "‘": "'",  # left single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "–": "-",  # en dash
    "—": "-",  # em dash
    " ": " ",  # nbsp
}


def _normalize_for_locate(s: str) -> str:
    for src_ch, dst_ch in _QUOTE_EQUIVALENTS.items():
        s = s.replace(src_ch, dst_ch)
    return s


def _locate_value(haystack: str, value: str) -> tuple[int, int] | None:
    """Find ``value`` in ``haystack``. Return (start, end) on the
    haystack's original indices, or None if not findable even after
    Unicode normalization."""
    idx = haystack.find(value)
    if idx != -1:
        return idx, idx + len(value)
    # Fall back to normalized comparison. Both sides flatten curly→straight
    # quotes / nbsp/dashes; the match span on the *normalized* haystack is
    # also valid on the original since each substitution preserves length
    # (single char → single char).
    norm_h = _normalize_for_locate(haystack)
    norm_v = _normalize_for_locate(value)
    idx = norm_h.find(norm_v)
    if idx == -1:
        return None
    return idx, idx + len(value)


def _split_think(content: str) -> tuple[str, str]:
    """Split content into (think_block, deliverable). think_block keeps
    the ``<think>...</think>`` tags so the reassembled content is
    byte-identical when the deliverable is unchanged."""
    end_tag = "</think>"
    idx = content.find(end_tag)
    if idx == -1:
        return "", content
    head = content[: idx + len(end_tag)]
    tail = content[idx + len(end_tag) :]
    return head, tail


def _inject_labels(deliverable: str, ad: SourceAd) -> tuple[str | None, str, list[str]]:
    """Inject platform labels into a deliverable.

    Returns ``(new_deliverable, status, missing_fields)``.
    ``new_deliverable`` is None when at least one populated field could
    not be located verbatim (the row is unchanged in that case).
    """
    group = platform_group_for(ad.platform)
    if group is PlatformLabelGroup.OTHER:
        return deliverable, "other_skip", []
    raw_source = ad.raw.get("source") if isinstance(ad.raw, dict) else None
    try:
        ad_source = AdSource(raw_source) if isinstance(raw_source, str) else AdSource.ADFLEX
    except ValueError:
        ad_source = AdSource.ADFLEX
    spec = PLATFORM_LABEL_MAP.get((ad_source, group))
    if spec is None:
        return deliverable, "no_mapping_skip", []

    result = deliverable
    injected: list[str] = []
    missing: list[str] = []
    for slot in spec:
        value = getattr(ad, slot.field, "")
        if not isinstance(value, str) or not value.strip():
            continue
        value_stripped = value.strip()
        span = _locate_value(result, value_stripped)
        if span is None:
            missing.append(slot.label)
            continue
        start, end = span
        before = result[:start]
        original_slice = result[start:end]
        after = result[end:]
        # Preserve the deliverable's actual byte-form of the value
        # (which may have flattened curly→straight quotes) rather than
        # the source ad's form — keeps the rest of the deliverable's
        # punctuation consistent with itself.
        if slot.multiline:
            lines = [ln.strip() for ln in original_slice.split("\n") if ln.strip()]
            bullet_block = "\n".join(f"- {ln}" for ln in lines)
            labeled = f"**{slot.label}:**\n{bullet_block}"
        else:
            labeled = f"**{slot.label}:** {original_slice}"
        result = before + labeled + after
        injected.append(slot.label)

    if missing:
        return None, f"missing_fields:{','.join(missing)}", missing
    if not injected:
        return result, "no_populated_slots", []
    return result, f"injected:{','.join(injected)}", []


@app.command()
def backfill(
    run_id: Annotated[str, typer.Option("--run-id", help="Run ID under data/constructed_v2/runs/")],
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Construction v2 config YAML"),
    ] = Path("configs/construction_v2.yaml"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="Report counts without writing"),
    ] = False,
) -> None:
    """Inject platform labels into the run's responses_raw.jsonl."""
    config = ConstructionV2Config.from_yaml(config_path)
    run_dir = Path(config.output_dir) / "runs" / run_id / "copywriting"
    src = run_dir / "responses_raw.jsonl"
    if not src.exists():
        console.print(f"[red]responses_raw.jsonl not found:[/red] {src}")
        raise typer.Exit(1)

    # Load and validate responses_raw.jsonl
    rows: list[dict[str, object]] = []
    for line_num, line in enumerate(src.open(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                tp = type(row).__name__
                console.print(f"[yellow]Line {line_num}: expected dict, got {tp}; skip[/yellow]")
                continue
            rows.append(row)
        except json.JSONDecodeError as exc:
            console.print(f"[yellow]Line {line_num}: JSON parse error: {exc}; skipping[/yellow]")
            continue

    # Extract ad_ids, filtering out rows that lack the key
    ad_ids: list[str] = []
    for row in rows:
        aid = row.get("ad_id")
        if isinstance(aid, str):
            ad_ids.append(aid)

    if not ad_ids:
        console.print("[red]No rows with valid ad_id found[/red]")
        raise typer.Exit(1)

    ads = load_source_ads_by_id(config, ad_ids)

    per_status: Counter[str] = Counter()
    per_platform_ok: Counter[str] = Counter()
    per_platform_fail: Counter[str] = Counter()
    rewritten: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for row in rows:
        ad_id_obj = row.get("ad_id")
        if not isinstance(ad_id_obj, str):
            per_status["missing_ad_id"] += 1
            rewritten.append(row)
            failures.append(
                {"ad_id": f"<invalid type: {type(ad_id_obj).__name__}>", "reason": "missing_ad_id"}
            )
            continue

        ad_id: str = ad_id_obj
        ad = ads.get(ad_id)
        if ad is None:
            per_status["missing_ad"] += 1
            rewritten.append(row)
            failures.append({"ad_id": ad_id, "reason": "missing_ad"})
            continue
        content = row.get("content", "")
        if not isinstance(content, str):
            per_status["non_string_content"] += 1
            rewritten.append(row)
            continue
        head, tail = _split_think(content)
        new_tail, status, _missing = _inject_labels(tail, ad)
        per_status[status.split(":")[0]] += 1
        if new_tail is None:
            per_platform_fail[ad.platform] += 1
            rewritten.append(row)
            failures.append({"ad_id": ad_id, "platform": ad.platform, "reason": status})
            continue
        per_platform_ok[ad.platform] += 1
        new_row = dict(row)
        new_row["content"] = head + new_tail
        rewritten.append(new_row)

    # Report
    table = Table(title=f"backfill run_id={run_id}", show_header=True)
    table.add_column("status")
    table.add_column("count", justify="right")
    for status, count in per_status.most_common():
        table.add_row(status, str(count))
    console.print(table)

    plat_table = Table(title="per-platform", show_header=True)
    plat_table.add_column("platform")
    plat_table.add_column("ok", justify="right")
    plat_table.add_column("missing", justify="right")
    for plat in sorted(set(per_platform_ok) | set(per_platform_fail)):
        plat_table.add_row(
            plat,
            str(per_platform_ok.get(plat, 0)),
            str(per_platform_fail.get(plat, 0)),
        )
    console.print(plat_table)

    if dry_run:
        console.print("[yellow]--dry-run set; no files written[/yellow]")
        if failures:
            console.print(f"[yellow]first 5 failures:[/yellow] {failures[:5]}")
        return

    # Write backup then rewrite responses_raw.jsonl
    backup = src.with_suffix(".jsonl.pre-label-backfill.bak")
    if not backup.exists():
        shutil.copy(src, backup)
        console.print(f"[green]Backup written:[/green] {backup}")

    # Write to temp file first, then atomically move to target
    # This prevents corruption if the script crashes mid-write.
    temp_path = src.with_suffix(".jsonl.tmp")
    try:
        with temp_path.open("w") as f:
            for row in rewritten:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp_path.replace(src)
        console.print(f"[green]Wrote:[/green] {src} ({len(rewritten)} rows)")
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        console.print(f"[red]Write failed: {exc}[/red]")
        raise typer.Exit(code=1) from None

    if failures:
        fail_path = run_dir.parent / "_audit" / "label_backfill_failures.jsonl"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("w") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        console.print(f"[yellow]Failures logged:[/yellow] {fail_path} ({len(failures)} rows)")


if __name__ == "__main__":
    app()
