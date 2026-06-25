"""Language enrichment CLI — detects and backfills language codes on the raw ad corpus.

Runs langdetect on ``headline + body`` and writes ``RawAd.language`` in place.
Default target is ``data/raw/adflex_ads.jsonl`` (bare RawAd records); pass
``--path`` to enrich a specific file (e.g. a scored_ads.jsonl). Auto-detects
bare vs ScoredAd-wrapped records so the same script works on either format.

Safe to re-run: ads that already have a non-empty ``language`` are skipped
unless ``--force`` is set.

Usage:
    python scripts/ops/enrich_language.py                                # raw corpus
    python scripts/ops/enrich_language.py --path data/scored/v3/scored_ads.jsonl
    python scripts/ops/enrich_language.py --force                        # re-detect everything
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from draper.utils.language import detect_language
from draper.utils.logging import setup_logging

app = typer.Typer(help="Backfill language codes on ad JSONL files (raw by default).")
console = Console()

RAW_ADS_PATH = Path("data/raw/adflex_ads.jsonl")


@app.command()
def run(
    path: str = typer.Option(
        "",
        "--path",
        "-p",
        help="Path to a JSONL file. Defaults to data/raw/adflex_ads.jsonl.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-detect language even if already set.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Detect but do not write back.",
    ),
) -> None:
    setup_logging()

    target = Path(path) if path else RAW_ADS_PATH
    if not target.exists():
        console.print(f"[red]Not found: {target}[/red]")
        raise typer.Exit(1)
    targets = [target]
    console.print(f"[bold]Processing:[/bold] {target}")
    console.print()

    grand_total = grand_detected = grand_skipped = grand_failed = 0
    grand_langs: Counter[str] = Counter()

    for jsonl_path in targets:
        console.rule(f"[bold]{jsonl_path}[/bold]")

        raw_lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
        records: list[dict[str, object]] = []
        parse_errors = 0
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
        if parse_errors:
            console.print(f"[yellow]  ⚠ Skipped {parse_errors} malformed lines[/yellow]")

        total = len(records)
        detected = skipped = failed = 0
        langs: Counter[str] = Counter()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Detecting language…", total=total)

            for rec in records:
                ad = rec.get("ad", rec)  # ScoredAd wraps ad; also handle bare RawAd
                current_lang = ad.get("language", "")

                if current_lang and not force:
                    langs[current_lang] += 1
                    skipped += 1
                    progress.advance(task)
                    continue

                copy = ad.get("ad_copy", {}) or {}
                headline = copy.get("headline", "") or ""
                body = copy.get("body", "") or ""
                description = copy.get("description", "") or ""
                lang = detect_language(headline, body, description)

                if lang:
                    ad["language"] = lang
                    langs[lang] += 1
                    detected += 1
                else:
                    ad["language"] = ""
                    langs["unknown"] += 1
                    failed += 1

                progress.advance(task)

        # Write back
        if not dry_run:
            jsonl_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                encoding="utf-8",
            )
            console.print(f"[green]✓ Written {total} records back to {jsonl_path.name}[/green]")
        else:
            console.print("[yellow]Dry-run — no changes written.[/yellow]")

        # Per-file summary
        console.print(
            f"  total={total}  detected={detected}  skipped={skipped}  failed(too-short)={failed}"
        )

        # Language distribution table
        table = Table(title="Language distribution", show_header=True, header_style="bold cyan")
        table.add_column("Lang", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Share", justify="right")
        for lang_code, count in langs.most_common(20):
            share = f"{count / total:.1%}"
            is_en = lang_code == "en"
            style = "green" if is_en else ("dim" if lang_code == "unknown" else "")
            table.add_row(lang_code, str(count), share, style=style)
        if len(langs) > 20:
            rest = sum(v for k, v in langs.items() if k not in dict(langs.most_common(20)))
            table.add_row("… other", str(rest), f"{rest/total:.1%}", style="dim")
        console.print(table)

        grand_total += total
        grand_detected += detected
        grand_skipped += skipped
        grand_failed += failed
        grand_langs.update(langs)

    if len(targets) > 1:
        console.rule("[bold]Grand total[/bold]")
        en = grand_langs.get("en", 0)
        non_en = grand_total - en - grand_langs.get("unknown", 0)
        console.print(
            f"  total={grand_total}  en={en} ({en/grand_total:.1%})  "
            f"non-en={non_en} ({non_en/grand_total:.1%})  "
            f"undetected={grand_langs.get('unknown',0)}"
        )


if __name__ == "__main__":
    app()
