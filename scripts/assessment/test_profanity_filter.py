"""Diagnostic run of the content-safety filter on the raw ad corpus.

For each ad, records which level fired:
  L1_wordlist   — explicit better-profanity match
  L2_ml         — alt-profanity-check toxicity >= threshold
  L0_clean      — both signals negative
  L0_skip_lang  — non-English, skipped (libraries are English-only)
  L0_empty      — no ad copy at all

Reports:
  - Overall level distribution
  - Top profane words at L1
  - ML score histogram and worst-offender samples at L2
  - Spot-check samples per level
  - Breakdown by advertiser for the flagged set (repeat offenders?)

Run:
    python scripts/assessment/test_profanity_filter.py
    python scripts/assessment/test_profanity_filter.py --sample 20 --threshold 0.6
    python scripts/assessment/test_profanity_filter.py --path data/raw/adflex_ads.jsonl
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from scripts.assessment.profanity_filter import (
    DEFAULT_ML_THRESHOLD,
    SafetyResult,
    classify_content_safety,
)

console = Console()
app = typer.Typer()

RAW_ADS_PATH = Path("data/raw/adflex_ads.jsonl")


def _load_ads(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    ads: list[dict] = []
    bad = 0
    for ln in lines:
        if not ln.strip():
            continue
        try:
            ads.append(json.loads(ln))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        console.print(f"[yellow]⚠ Skipped {bad} malformed lines[/yellow]")
    return ads


def _snippet(text: str, n: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")


@app.command()
def run(
    path: str = typer.Option("", "--path", "-p", help="JSONL to scan."),
    sample: int = typer.Option(8, "--sample", "-s", help="Samples per bucket."),
    threshold: float = typer.Option(
        DEFAULT_ML_THRESHOLD, "--threshold", "-t", help="L2 ML probability cutoff."
    ),
    seed: int = typer.Option(42, "--seed", help="Sampling seed."),
    limit: int = typer.Option(0, "--limit", "-n", help="Cap ads processed (0 = all)."),
) -> None:
    random.seed(seed)

    target = Path(path) if path else RAW_ADS_PATH
    if not target.exists():
        console.print(f"[red]Not found: {target}[/red]")
        raise typer.Exit(1)

    console.rule(f"[bold]{target}[/bold]")
    ads = _load_ads(target)
    if limit:
        ads = ads[:limit]
    total = len(ads)
    console.print(f"[bold]Scanning[/bold] {total:,} ads  (ML threshold={threshold})")
    console.print()

    level_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    word_counts: Counter[str] = Counter()
    ml_scores: list[float] = []
    buckets: dict[str, list[tuple[dict, SafetyResult]]] = {
        "L1_wordlist": [],
        "L2_ml": [],
        "L0_clean": [],
        "L0_skip_lang": [],
        "L0_empty": [],
    }
    advertiser_flagged: Counter[str] = Counter()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Classifying…", total=total)
        for ad in ads:
            copy = ad.get("ad_copy") or {}
            lang = ad.get("language", "") or ""
            lang_counts[lang or "unknown"] += 1

            result = classify_content_safety(
                headline=copy.get("headline", "") or "",
                body=copy.get("body", "") or "",
                description=copy.get("description", "") or "",
                cta=copy.get("cta", "") or "",
                advertiser_name=ad.get("advertiser_name", "") or "",
                language=lang,
                ml_threshold=threshold,
            )
            level_counts[result.level] += 1

            if result.flag:
                advertiser_flagged[ad.get("advertiser_name", "?")] += 1
            if result.level == "L1_wordlist":
                word_counts.update(result.profane_words)
            if result.level == "L2_ml":
                ml_scores.append(result.ml_score)

            buckets[result.level].append((ad, result))
            progress.advance(task)

    # --- Summary table ------------------------------------------------------
    flagged = level_counts["L1_wordlist"] + level_counts["L2_ml"]
    console.rule("[bold]Level distribution[/bold]")
    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Level")
    tbl.add_column("Count", justify="right")
    tbl.add_column("Share", justify="right")
    for lvl in ("L1_wordlist", "L2_ml", "L0_clean", "L0_skip_lang", "L0_empty"):
        c = level_counts[lvl]
        style = "red" if lvl.startswith("L1") or lvl.startswith("L2") else ""
        tbl.add_row(lvl, f"{c:,}", f"{c/total:.1%}" if total else "—", style=style)
    console.print(tbl)
    console.print(
        f"[bold]Flagged total:[/bold] {flagged:,} / {total:,} "
        f"({flagged / total:.2%} if you dropped them)"
    )
    console.print()

    # --- Language coverage --------------------------------------------------
    console.rule("[bold]Ad language distribution[/bold]")
    lang_tbl = Table(show_header=True, header_style="bold cyan")
    lang_tbl.add_column("Lang")
    lang_tbl.add_column("Count", justify="right")
    lang_tbl.add_column("Share", justify="right")
    for lang, c in lang_counts.most_common(10):
        lang_tbl.add_row(lang, f"{c:,}", f"{c/total:.1%}")
    console.print(lang_tbl)
    en_share = lang_counts.get("en", 0) / total if total else 0
    console.print(f"[dim]English coverage of filter: {en_share:.1%}[/dim]\n")

    # --- Top profane words (L1) --------------------------------------------
    if word_counts:
        console.rule("[bold]Top profane words (L1 wordlist)[/bold]")
        w_tbl = Table(show_header=True, header_style="bold cyan")
        w_tbl.add_column("Word")
        w_tbl.add_column("Hits", justify="right")
        for w, c in word_counts.most_common(25):
            w_tbl.add_row(w, f"{c:,}")
        console.print(w_tbl)
        console.print()

    # --- ML score distribution (L2) ----------------------------------------
    if ml_scores:
        console.rule("[bold]L2 ML score distribution[/bold]")
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        ranges = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        counts = [sum(1 for s in ml_scores if lo <= s < hi) for lo, hi in ranges]
        h_tbl = Table(show_header=True, header_style="bold cyan")
        h_tbl.add_column("Range")
        h_tbl.add_column("Count", justify="right")
        for (lo, hi), c in zip(ranges, counts, strict=False):
            h_tbl.add_row(f"{lo:.2f}–{hi:.2f}", f"{c:,}")
        console.print(h_tbl)
        console.print()

    # --- Top advertisers among flagged -------------------------------------
    if advertiser_flagged:
        console.rule("[bold]Top flagged advertisers[/bold]")
        a_tbl = Table(show_header=True, header_style="bold cyan")
        a_tbl.add_column("Advertiser")
        a_tbl.add_column("Flagged ads", justify="right")
        for name, c in advertiser_flagged.most_common(15):
            a_tbl.add_row(name or "?", f"{c:,}")
        console.print(a_tbl)
        console.print()

    # --- Sample each bucket -------------------------------------------------
    for level in ("L1_wordlist", "L2_ml", "L0_clean", "L0_skip_lang"):
        items = buckets[level]
        if not items:
            continue
        console.rule(f"[bold]Samples — {level}[/bold]")
        if level == "L2_ml":
            items = sorted(items, key=lambda x: -x[1].ml_score)[: sample * 2]
        chosen = random.sample(items, min(sample, len(items)))
        for ad, result in chosen:
            copy = ad.get("ad_copy") or {}
            headline = _snippet(copy.get("headline", ""))
            body = _snippet(copy.get("body", ""), 200)
            tag = (
                f"words={list(result.profane_words)}"
                if result.profane_words
                else f"ml={result.ml_score:.2f}"
                if result.ml_score
                else ""
            )
            console.print(
                f"[bold]{ad.get('advertiser_name', '?')}[/bold] "
                f"[dim]({ad.get('language', '')})[/dim]  {tag}"
            )
            console.print(f"  H: {headline}")
            console.print(f"  B: {body}")
            console.print()


if __name__ == "__main__":
    app()
